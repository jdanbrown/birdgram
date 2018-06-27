from collections import OrderedDict
from functools import partial
import os.path
from pathlib import Path
import re
from typing import Iterable, Optional, Tuple

from attrdict import AttrDict
import audiosegment
from dataclasses import dataclass
import pandas as pd
from potoo.pandas import requires_cols
from potoo.util import or_else, round_sig, strip_startswith
import tqdm

from cache import cache
from constants import cache_dir, data_dir, standard_sample_rate_hz
from datasets import audio_path_files, DATASETS, metadata_from_dataset
from datatypes import Audio, Recording, RecordingDF
import metadata
from util import *


@dataclass
class Load(DataclassConfig):

    channels: int = 1
    sample_rate: int = standard_sample_rate_hz
    sample_width_bit: int = 16
    cache_audio: bool = True

    @property
    def deps(self) -> AttrDict:
        return None

    @property
    def audio_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'channels',
            'sample_rate',
            'sample_width_bit',
            'cache_audio',
        ]})

    @cache(version=1, key=lambda self, datasets, *args, **kwargs: (
        {k: DATASETS[k] for k in datasets},
        args,
        kwargs,
    ))
    def recs(
        self,
        datasets: Iterable[str] = None,
        paths: Iterable[Tuple[str, str]] = None,
        limit: int = None,
        drop_invalid: bool = True,
    ) -> RecordingDF:
        """Load recs.{**metadata} from fs"""
        return (
            self.recs_paths(datasets, paths)
            # .sample(n=limit, random_state=0)  # TODO Why does this return empty? More useful than [:limit]
            [:limit]
            .pipe(lambda df: pd.concat(axis=1, objs=[df, self.metadata(df)]))
            .sort_values('species')
            [lambda df: ~np.array(drop_invalid) | (df.samples_n != 0)]  # Filter out invalid/empty audios
            .reset_index(drop=True)
            .pipe(RecordingDF)
        )

    def recs_paths(
        self,
        datasets: Iterable[str] = None,
        paths: Iterable[Tuple[str, str]] = None,
    ) -> RecordingDF:
        """Load recs.{id,dataset,path} <- fs"""
        return self._recs_paths((paths or []) + [
            (dataset, path)
            for dataset, config in DATASETS.items()
            if dataset in (datasets or [])
            for path in tqdm(disable=True, desc='recs_paths', unit=' paths', iterable=(
                audio_path_files.read(dataset)
            ))
        ])

    def _recs_paths(self, paths: Iterable[Tuple[str, str]]) -> RecordingDF:
        """Load recs.{id,dataset,path} <- paths"""
        # Helpful error msg for common mistake (because 'paths' is a helpfully short but unhelpfully unclear name)
        if paths and not isinstance(paths[0], tuple):
            raise ValueError(f'Expected paths=[(dataset, path), ...], got paths=[{paths[0]!r}, ...]')
        return RecordingDF([
            # Recording(...).asdict()  # XXX Bottleneck (xc)
            dict(
                # id=os.path.splitext(path)[0],
                id=path.rsplit('.', 1)[0],
                dataset=dataset,
                path=path,
                # filesize_b=os.path.getsize(path),  # XXX Bottleneck (xc) -- O(n) stat calls
            )
            for dataset, path in tqdm(disable=True, desc='_recs_paths', unit=' paths', iterable=(
                paths
            ))
            # for path in [os.path.relpath(path, data_dir)]  # XXX Bottleneck (xc)
            for path in [strip_startswith(str(path), str(data_dir), check=True).lstrip('/')]
        ])

    METADATA = [
        # TODO De-dupe these with datasets.metadata_from_dataset
        'species',
        'species_longhand',
        'species_com_name',
        'species_query',
        'basename',
        # TODO De-dupe these with self._metadata
        'duration_s',
        'samples_mb',
        'samples_n',
    ]

    @short_circuit(lambda self, recs: recs.get(self.METADATA))
    def metadata(self, recs: RecordingDF) -> RecordingDF:
        """.metadata <- .audio"""
        log('Load.metadata:in', **{
            'len(recs)': len(recs),
            'len(recs) per dataset': recs.dataset.value_counts().to_dict(),
        })
        # Performance (600 peterson recs):
        #   - Scheduler: [TODO Measure -- 'threads' is like the outcome, like n-1 of the rest]
        #   - Bottlenecks (no_dask): [TODO Measure]
        #   - TODO TODO Revisiting with ~87k xc recs...
        metadata = map_progress(self._metadata, df_rows(recs),
            # use='dask', scheduler='threads',    # Optimal for 600 peterson recs on laptop
            # Perf comparison:                             machine     cpu   disk_r     disk_w  disk_io_r  disk_io_w
            # use='dask', scheduler='threads',    # n1-standard-16   5-20%   1-5m/s  10-120m/s      10-50     50-500
            # use='dask', scheduler='processes',  # n1-standard-16  10-50%  5-20m/s    ~100m/s     50-200    300-600
            use='dask', scheduler='processes', get_kwargs=dict(num_workers=os.cpu_count() * 2),
        )
        # Filter out dropped rows (e.g. junky audio file)
        metadata = [x for x in metadata if x is not None]
        # Convert to df
        metadata = RecordingDF(metadata)
        log('Load.metadata:out', **{
            'sum(duration_h)': round_sig(metadata.duration_s.sum() / 3600, 3),
            'sum(samples_mb)': round_sig(metadata.samples_mb.sum(), 3),
            'sum(samples_n)': int(metadata.samples_n.sum()),
            'n_species': len(set(metadata.species)),
        })
        return metadata

    # Cache hit avoids loading audio (~1000x bigger: ~1MB audio vs. ~1KB metadata)
    # Avoid Series.get(cols): it returns nan for unknown cols instead of None overall (df.get(cols) gives None overall)
    @short_circuit(lambda self, rec: AttrDict(rec[self.METADATA]) if set(self.METADATA).issubset(rec.index) else None)
    @cache(version=0, key=lambda self, rec: rec.id)
    def _metadata(self, rec: Row) -> AttrDict:
        """metadata <- .audio"""
        audio = self._audio(rec)
        audio = audio.unbox
        samples = audio.to_numpy_array()
        return AttrDict(
            **metadata_from_dataset(rec.id, rec.dataset),
            # TODO De-dupe these with self.METADATA
            duration_s=audio.duration_seconds,
            samples_mb=len(samples) * audio.sample_width / 1024**2,
            samples_n=len(samples),
        )

    def audio_to_wav(self, recs: RecordingDF):
        """Compute and save .wav files for recs (return nothing)"""
        log('Load.audio_to_wav:in', **{
            'len(recs)': len(recs),
            'len(recs) per dataset': recs.dataset.value_counts().to_dict(),
        })
        # Precompute and filter out exists=True so we get a more accurate estimate of progress on the exists=False set
        recs = (recs
            .pipe(df_apply_progress, use='dask', scheduler='processes', f=lambda row: (
                row.set_value('cache_path', self._cache_path(row.path))
            ))
            .pipe(df_apply_progress, use='dask', scheduler='threads', f=lambda row: (
                row.set_value('exists', (Path(data_dir) / row.cache_path).exists())
            ))
            [lambda df: ~df.exists]
        )
        # Convert recs to .wav (and return nothing)
        self.audio(recs, load=False)

    @short_circuit(lambda self, recs, **kwargs: recs.get('audio'))
    def audio(self, recs: RecordingDF, load=True, **kwargs) -> Column['Box[Optional[Audio]]']:
        """.audio <- .path"""
        log('Load.audio:in', **{
            'len(recs)': len(recs),
            'len(recs) per dataset': recs.dataset.value_counts().to_dict(),
        })
        # Performance (600 peterson recs):
        #   - Scheduler: no_dask[.85s], synchronous[.93s], threads[.74s], processes[25s]
        #   - Bottlenecks (no_dask):
        #          ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #             600    0.303    0.001    0.312    0.001 audio_segment.py:108(read_wav_audio)
        #             600    0.170    0.000    0.170    0.000 {method 'read' of '_io.BufferedReader' objects}
        #               1    0.060    0.060    0.845    0.845 <string>:1(<module>)
        #           61176    0.018    0.000    0.039    0.000 {built-in method builtins.isinstance}
        #             600    0.015    0.000    0.015    0.000 {built-in method io.open}
        audio = map_progress(partial(self._audio, load=load), df_rows(recs), **{
            **dict(
                # use='dask', scheduler='threads',  # Optimal for cache hits (disk read), but not cache misses (ffmpeg)
                # use='dask', scheduler='processes', get_kwargs=dict(num_workers=os.cpu_count() * 2),  # FIXME Quiet...
                use='dask', scheduler='processes', get_kwargs=dict(num_workers=os.cpu_count() * 2), partition_size=10,
            ),
            **kwargs,
        })
        log('Load.audio:out', **{
            'len(audio)': len(audio),
        })
        return audio

    @short_circuit(lambda self, rec, **kwargs: rec.get('audio'))
    # Caching doesn't help here, since our bottleneck is file read (.wav), which is also cache hit's bottleneck
    def _audio(
        self,
        rec: Row,
        load=True,  # load=False if you just want to trigger lots of wav encodings
    ) -> 'Box[Optional[Audio]]':
        """audio <- .path, and (optionally) cache a standardized .wav for faster subsequent loads"""

        path = rec.path
        c = self.audio_config

        # Interpret relative paths as relative to data_dir (leave absolute paths as is)
        if not os.path.isabs(path):
            path = os.path.join(data_dir, path)

        # "Drop" audio files that fail during either transcription or normal load
        #   - TODO Find a way to minimize the surface area of exceptions that we swallow here. Currently huge and bad.
        try:

            # Cache transcribed audio, if requested
            if not c.cache_audio:
                cache_path = None
            else:
                cache_path = self._cache_path(path)
                if os.path.exists(cache_path):
                    log.char('info', '•')
                else:
                    # log.char('info', '!')
                    log(f'Caching: {cache_path}')
                    in_audio = audiosegment.from_file(path)
                    std_audio = in_audio.resample(
                        channels=c.channels,
                        sample_rate_Hz=c.sample_rate,
                        sample_width=c.sample_width_bit // 8,
                    )
                    std_audio.export(ensure_parent_dir(cache_path), 'wav')
                path = cache_path

            # Caching aside, always load from disk for consistency
            if load:
                audio = audiosegment.from_file(path)
            else:
                audio = None

        except Exception as e:
            # "Drop" invalid audio files by replacing them with a 0s audio, so we can detect and filter out downstream
            log('Load._audio: WARNING: Dropping invalid audio file', **dict(
                error=str(e),
                dataset=rec.dataset,
                id=rec.id,
                path=rec.path,
                # filesize_b=rec.filesize_b,
                cache_in_path=path,
                cache_in_path_exists=os.path.exists(path),
                cache_in_filesize_b=or_else(None, lambda: os.path.getsize(path)),
                cache_out_path=cache_path,
                cache_out_path_exists=or_else(None, lambda: os.path.exists(cache_path)),
                cache_out_filesize_b=or_else(None, lambda: os.path.getsize(cache_path)),
            ))
            audio = audiosegment.empty()
            audio.name = path
            audio.seg.frame_rate = c.sample_rate

        if load:

            # Make audiosegment.AudioSegment attrs more ergonomic
            audio = self._ergonomic_audio(audio)

            # Box to avoid downstream pd.Series errors, since AudioSegment is iterable and np.array tries to flatten it
            audio = box(audio)

            return audio

    def _cache_path(self, path: str) -> str:
        c = self.audio_config
        rel_path_noext, _ext = os.path.splitext(os.path.relpath(path, data_dir))
        params_id = f'{c.sample_rate}hz-{c.channels}ch-{c.sample_width_bit}bit'
        return f'{cache_dir}/{params_id}/{rel_path_noext}.wav'

    # HACK Make our own Audio instead of monkeypatching audiosegment.AudioSegment
    def _ergonomic_audio(self, audio: audiosegment.AudioSegment) -> audiosegment.AudioSegment:
        """Make audiosegment.AudioSegment attrs more ergonomic"""
        audio = audiosegment.AudioSegment(audio.seg, audio.name)
        # Save the full path
        audio.path = audio.name
        # More ergonomic .name (which is never used as a path)
        if audio.path.startswith(cache_dir):
            # Relative cache path, excluding the leading 'hz=...,ch=...,bit=.../' dir
            name = os.path.relpath(audio.path, cache_dir).split('/', 1)[1]
        else:
            # Else relative data path
            name = os.path.relpath(audio.path, data_dir)
        # Extensions are boring
        name, _ext = os.path.splitext(name)
        audio.name = name
        return audio
