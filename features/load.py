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
import parse
from potoo.pandas import requires_cols
from potoo.pretty import pp
from potoo.util import or_else, path_is_contained_by, round_sig, strip_startswith
import structlog
import tqdm

from cache import cache
from config import config
from constants import cache_audio_dir, cache_dir, data_dir, standard_sample_rate_hz
from datasets import audio_path_files, DATASETS, metadata_from_dataset
from datatypes import Audio, Recording, RecordingDF
import metadata
from util import *

log = structlog.get_logger(__name__)

# TODO Can we load a slice, instead of the whole input file?
#   - Maybe way faster for large inputs when we know we only want 10/5/3s in the end


@dataclass
class Load(DataclassConfig):

    should_transcode: bool = True
    # Relevant iff should_transcode
    sample_rate: int = standard_sample_rate_hz
    channels: int = 1
    sample_width_bit: int = 16
    # TODO Change default to something more space efficient?
    #   - Goal: ~10x space savings with e.g. mp4 32k vs. wav (see notebooks/audio_codecs_data_volume)
    #   - [ ] QA: Run a model comp for e.g. mp4 vs. wav, to ensure e.g. high-freq species aren't getting junked
    #   - [ ] QA: Run a perf comp to ensure that loading metadata from e.g. mp4 vs. wav doesn't create a cpu bottleneck
    format: str = 'wav'
    bitrate: str = None
    codec: str = None

    # Disable if you want to be able to resample (hz,ch,bit) for files that we've already cached
    #   - See usage below for details
    fail_if_resample_cached_audio: bool = True

    @property
    def deps(self) -> AttrDict:
        return None

    @property
    def audio_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'should_transcode',
            'channels',
            'sample_rate',
            'sample_width_bit',
            'format',
            'bitrate',
            'codec',
        ]})

    @cache(version=2, tags='recs', key=lambda self, datasets=None, paths=None, *args, **kwargs: (
        {k: DATASETS[k] for k in (datasets or [])},
        paths or self.recs_paths(datasets=datasets),
        args,
        kwargs,
    ))
    def recs(
        self,
        datasets: Iterable[str] = None,
        paths: Iterable[Tuple[str, str]] = None,
        limit: int = None,
        drop_invalid: bool = True,
        log_dropped: bool = True,
    ) -> RecordingDF:
        """Load recs.{**metadata} from fs"""
        recs = (
            self.recs_paths(datasets, paths)
            # .sample(n=limit, random_state=0)  # TODO Why does this return empty? More useful than [:limit]
            [:limit]
            .pipe(lambda df: pd.concat(axis=1, objs=[df, self.metadata(df)]))
            .reset_index(drop=True)
        )
        if drop_invalid:
            # Filter out invalid audios
            drop_ix = recs.samples_n == 0  # samples_n is 0 if input is either empty or couldn't be read
            dropped = recs[drop_ix]
            if dropped.size:
                recs = recs[~drop_ix]
                if log_dropped:
                    log.warn(f'Dropping {len(dropped)} invalid audios:')
                    for _id in dropped.id:
                        print(f'  {_id}')
        return (recs
            .sort_values('species')
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
                id=path,
                dataset=dataset,
                path=path,  # XXX rec.path, slowly being replaced by rec.id
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
        # TODO De-dupe these with self._metadata
        'duration_s',
        'samples_mb',
        'samples_n',
        'sample_rate',
        'channels',
        'sample_width_bit',
    ]

    @requires_nonempty_rows
    @short_circuit(lambda self, recs: recs.get(self.METADATA))
    def metadata(self, recs: RecordingDF) -> RecordingDF:
        """.metadata <- .audio"""
        log.info(**{
            'len(recs)': len(recs),
        })
        metadata = map_progress(self._metadata, df_rows(recs), n=len(recs), desc='audio_metadata',
            **config.load.metadata_progress_kwargs,
        )
        # Filter out dropped rows (e.g. junky audio file)
        metadata = [x for x in metadata if x is not None]
        # Convert to df
        metadata = RecordingDF(metadata)
        log.debug('done', **{
            'sum(duration_h)': round_sig(metadata.duration_s.sum() / 3600, 3),
            'sum(samples_mb)': round_sig(metadata.samples_mb.sum(), 3),
            'sum(samples_n)': int(metadata.samples_n.sum()),
            'n_species': len(set(metadata.species)),
        })
        return metadata

    # Cache hit avoids loading audio (~1000x bigger: ~1MB audio vs. ~1KB metadata)
    # Avoid Series.get(cols): it returns nan for unknown cols instead of None overall (df.get(cols) gives None overall)
    @short_circuit(lambda self, rec: AttrDict(rec[self.METADATA]) if set(self.METADATA).issubset(rec.index) else None)
    @cache(version=5, tags='rec', key=lambda self, rec: rec.id)
    def _metadata(self, rec: Row) -> AttrDict:
        """metadata <- .audio"""
        # _audio_no_transcode because we want the metadata from the raw input file, not the standardized version
        #   - TODO Can speed up ~6x via ffprobe/mediainfo_json instead of full ffmpeg read (see notebooks/pydub_ffmpeg_read_perf)
        audio = self._audio_no_transcode(rec)  # Pull
        samples = audio.to_numpy_array()
        return AttrDict(
            **metadata_from_dataset(rec.id, rec.dataset),
            # TODO De-dupe these with self.METADATA
            duration_s=audio.duration_seconds,
            samples_mb=len(samples) * audio.sample_width / 1024**2,
            samples_n=len(samples),
            sample_rate=audio.frame_rate,
            channels=audio.channels,
            sample_width_bit=audio.sample_width * 8,
        )

    @requires_nonempty_rows
    @short_circuit(lambda self, recs, **kwargs: recs.get('audio'))
    def audio(self, recs: RecordingDF, load=True, **progress_kwargs) -> RecordingDF:
        """
        .audio <- .path + metadata(hz,ch,bit)
        - Returns rec where rec.id = rec.audio.unbox.name
        """
        log.info(**{
            'len(recs)': len(recs),
        })
        audio = map_progress(
            partial(self._audio, load=load), df_rows(recs), n=len(recs), desc='audio',
            **(progress_kwargs or config.load.audio_progress_kwargs),
        )
        if load:
            # Don't return recs.audio to the caller if load=False since it would be all None and recs.id would be stale
            recs = recs.assign(
                audio=box.many(audio),  # Box: AudioSegment is iterable so pd.Series/np.array try to flatten it
            )
            # Propagate audio.name (= path) to rec.id
            recs = recs.assign(
                id=lambda df: df.audio.map(lambda audio: audio.unbox.name),
            )
        log.debug('done', **{
            'len(audio)': len(audio),
        })
        return recs

    @requires_cols('audio')  # No @short_circuit (our purpose is to re-encode the existing .audio)
    @requires_nonempty_rows
    def transcode_audio(self, recs: RecordingDF, load=True, **progress_kwargs) -> RecordingDF:
        """
        .audio <- .audio
        - Returns recs where rec.id = rec.audio.unbox.name
        - Returns recs.audio that have .name and ._data reflecting the transcoding
        - Returns recs.audio that are persisted to file
        """
        log.info(**{
            'len(recs)': len(recs),
            'audio_config': self.audio_config,
        })
        audio = map_progress(
            partial(self._transcode_audio, load=load), recs.audio.map(unbox), n=len(recs), desc='transcode_audio',
            **(progress_kwargs or config.load.audio_progress_kwargs),
        )
        if load:
            # Don't return recs.audio to the caller if load=False since it would be all None and recs.id would be stale
            recs = recs.assign(
                audio=box.many(audio),  # Box: AudioSegment is iterable so pd.Series/np.array try to flatten it
            )
            # Propagate audio.name (= path) to rec.id
            recs = recs.assign(
                id=lambda df: df.audio.map(lambda audio: audio.unbox.name),
            )
        log.debug('done', **{
            'len(audio)': len(audio),
            'audio_config': self.audio_config,
        })
        return recs

    def _audio_no_transcode(self, rec: Row) -> Audio:
        """
        audio <- .path
        - Load the raw, unstandardized input file at rec.id, instead of the usual standardized transcoded file
        """
        return self.replace(should_transcode=False)._audio(rec, load=True)

    @short_circuit(lambda self, rec, **kwargs: rec.get('audio') and rec.audio.unbox)
    # @cache-ing doesn't help here, since our bottleneck is file read (.wav), which is also cache hit's bottleneck [+ pkl!]
    # @requires_cols('path', 'sample_rate', 'channels', 'sample_width_bit')  # TODO [Nope, see below]
    def _audio(
        self,
        rec: Row,
        load=True,  # load=False if you just want to trigger lots of wav encodings and skip O(n) read ops
    ) -> Optional[Audio]:
        """
        audio <- .path
        - If self.should_transcode, also requires rec.{sample_rate,channels,sample_width_bit}
        - If self.should_transcode, cache a standardized audio file for faster subsequent loads
        - If load, returns audio where .name is both a stable id and path (relative to data_dir)
        """
        c = self.audio_config

        # Require rec.id (= path) to be under data_dir
        if Path(rec.id).is_absolute():
            raise ValueError(f"rec.id[{rec.id}] must be relative to data_dir[{data_dir}]")
        rec_abs_path = Path(data_dir) / rec.id

        # TODO Helpful? For the catch I added somewhere else (I forgot where...)
        if not rec_abs_path.exists():
            raise FileNotFoundError(rec_abs_path)

        # Audio: transcode + cache + load (if requested)
        audio = None
        try:
            if c.should_transcode:
                audio = self._transcode_path(rec, load=load)  # (Returns None if load=False)
                # NOTE If load=False then audio is now None, which also means rec.id won't update in the caller
            elif load:
                audio = self.read_audio(rec.id)

        # "Drop" audio files that fail during either transcription or normal load
        #   - One common source of these is "restricted" xc recs where audio.mp3 downloads as html instead of mp3, e.g.
        #       - https://www.xeno-canto.org/308503 -> https://www.xeno-canto.org/308503/download
        #   - TODO Find a way to minimize the surface area of exceptions that we swallow here. Currently huge and bad.
        except Exception as e:

            # Re-raise these errors instead of dropping the audio
            if isinstance(e, FileNotFoundError):
                raise

            # Unpack ffmpeg error msgs from CouldntDecodeError (as already cleaned up by util.audio_from_file)
            if isinstance(e, pydub.exceptions.CouldntDecodeError):
                ffmpeg_msg = str(e)
                e_msg = f'{type(e).__name__}(...)'  # Abbreviate error msg, since it's huge and noisy
            else:
                ffmpeg_msg = None
                e_msg = str(e)

            # Try to detect "Download disabled for this species" pages (which we naively download as audio.mp3), e.g.
            #   - https://www.xeno-canto.org/308503 -> https://www.xeno-canto.org/308503/download
            filesize_b = or_else(None, lambda: rec_abs_path.stat().st_size)
            download_disabled_msg = 'Download disabled for this species'
            download_restricted_msg = 'restricted due to conservation concerns'
            download_disabled = False
            download_restricted = False
            try:
                if filesize_b < 1024**2:  # Don't bother on large files
                    with open(rec_abs_path, 'rt') as f:
                        audio_data = f.read()
                    download_disabled = download_disabled_msg in audio_data
                    download_restricted = download_restricted_msg in audio_data
            except Exception as e:
                pass

            # Give an informative warning
            log_msg = (
                f'Dropping restricted xc recording ({download_restricted_msg!r})' if download_restricted else
                f'Dropping disabled xc recording ({download_disabled_msg!r})' if download_disabled else
                f'Dropping invalid audio file:\n{ffmpeg_msg}\n' if ffmpeg_msg else
                f'Dropping invalid audio file'
            )
            log.warn(log_msg, **dict(
                error=e_msg,
                dataset=rec.get('dataset'),
                id=rec.get('id'),
                exists=rec_abs_path.exists(),
                filesize_b=filesize_b,
            ))

            # "Drop" invalid audio files by replacing them with a 0s audio, so we can detect and filter out downstream
            if load:
                audio = audiosegment.empty()
                audio.name = rec.id
                audio.seg.frame_rate = c.sample_rate

        # Integrity checks
        if audio is not None:
            assert audio_abs_path(audio).exists(), f"{audio.name}"

        return audio

    def _transcode_path(self, rec: Row, load=True) -> Optional[Audio]:
        """
        Transcode an audio path to a new file, as per our audio_config
        - Minimize r/w ops, e.g. for efficient bulk usage
        """
        c = self.audio_config
        assert not Path(rec.id).is_absolute()

        # HACK Mock an audio to detect cache hit vs. miss inside _transcode_audio, so we can skip the audio read
        if 'sample_rate' in rec:
            mock_audio = AttrDict(
                name=rec.id,
                frame_rate=rec.sample_rate,
                channels=rec.channels,
                sample_width=rec.sample_width_bit // 8,
            )
            (would_noop, audio_id) = self._transcode_audio(
                audio=mock_audio,
                dry_run=True,  # Return True if would do work, and don't actually do any work
                # load=load,  # (No effect when dry_run)
            )
            if would_noop:
                if not load:
                    # Skip the audio read because load=False means return None (and the caller isn't expecting a new id)
                    return None
                else:
                    return self.read_audio(audio_id)

        # Else we incur an audio read + _transcode_audio
        return self._transcode_audio(
            self.read_audio(rec.id),
            load=load,
        )

    def _transcode_audio(
        self,
        audio: Audio,
        load=True,
        dry_run=False,
        unsafe_fs=False,  # For tests only (until we have a mock fs)
    ) -> """Union[  # (Quote return type to avoid weird cloudpickle error...)
        Optional[Audio],        # If not dry_run
        Tuple[bool, 'AudioId'], # If dry_run
    ]""":
        """
        Transcode an audio to a file, as per our audio_config
        - Returns audio where .name is both a stable id and path (relative to data_dir)
        - Returns audio with .name and ._data reflecting the transcoding
        - Returns audio that is persisted to file
        """
        c = self.audio_config

        # WARNING This was a huge time sink to debug last time I refactored it (the rec.id overhaul)
        #   - TODO Add tests [requires mocking fs for audio.export + Path.exists]

        # FIXME char_log.char '•'/'!' on hit/miss might have caused, a long time ago, some 'IOStream.flush timed out' errors
        # in remote kernels (see cache.py)

        # Input expectations
        #   - If dry_run, ok to mock just audio.name
        #   - If not dry_run, audio must be a real Audio
        id = audio.name
        input_id = id

        # Plan all ops by adding them to the audio id
        #   - Execute the pipeline of ops only if the final audio file doesn't already exist in cache
        #   - Always return the new audio id, to reflect the transcoded audio

        # Do we need to resample?
        #   - Not if the input audio's (hz,ch,bit) match our audio_config
        #   - [Smeared concerns between here and _audio_id_simplify_ops, but we can't detect (hz,ch,bit) there...]
        (c_hz, c_ch, c_bit) = (c.sample_rate, c.channels, c.sample_width_bit)
        (a_hz, a_ch, a_bit) = (audio.frame_rate, audio.channels, audio.sample_width * 8)
        do_resample = (c_hz, c_ch, c_bit) != (a_hz, a_ch, a_bit)
        if do_resample:
            id = audio_id_add_ops(id, 'resample(%s,%s,%s)' % (c_hz, c_ch, c_bit))

            # Regression check: fail if we try to resample cache/audio/ files
            #   - Our typical usage resamples raw audio files once only upon copying into cache/audio/
            #   - This check guards against a subtle bug in pydub.AudioSegment.from_file(format=None), where you'll
            #     always resample mp3/mp4 files read from cache/audio/ because they have nonstandard file exts. The
            #     solution is to always pass a valid format, which util.audio_from_file now does. (More details there.)
            #   - TODO Turn this into a regression test [requires a mock fs]
            if self.fail_if_resample_cached_audio and path_is_contained_by(Path(data_dir) / input_id, cache_audio_dir):
                raise AssertionError('Refusing to resample a cache/audio/ file (%s): %s -> %s for %s' % (
                    'fail_if_resample_cached_audio=True', (a_hz, a_ch, a_bit), (c_hz, c_ch, c_bit), input_id,
                ))

        # Do we need to change the encoding?
        #   - audio_id_add_ops will determine this for us, by returning the id we pass it if not
        id = audio_id_add_ops(id, 'enc(%s)' % ','.join({
            'wav': [self.format],
            'mp3': [self.format, self.bitrate],
            'mp4': [self.format, self.codec, self.bitrate],
        }[self.format]))

        # Do we need to write the output file?
        #   - Not if it already exists, which can happen for two different reasons:
        #       1. The output id is the same as the input id (i.e. no resample and no encoding change)
        #       2. The output id is different (resample or encoding change), but the cache/audio/ file already exists
        do_write = (
            not unsafe_fs and  # HACK Don't Path.exists() in tests, where we don't yet have a mock fs
            not (Path(data_dir) / id).exists()
        )
        if do_write:
            assert path_is_contained_by(Path(data_dir) / id, cache_audio_dir), \
                f"Refusing to (plan to) write outside of our cache_audio_dir: {id}"

        if not dry_run:
            # Execute the pipeline only if the output file doesn't already exist
            if do_write:
                # Resample
                #   - Log as input_id since id already has all the ops in it
                if do_resample:
                    log.debug(f'Resample ({a_hz},{a_ch},{a_bit})->({c_hz},{c_ch},{c_bit}): {input_id}')
                    audio = audio.resample(sample_rate_Hz=c_hz, channels=c_ch, sample_width=c_bit // 8)
                # Write (which does the encoding)
                #   - Export with final id, since that determines the output file path
                log.info(f'Write: {id}')
                f = audio.export(
                    ensure_parent_dir(audio_abs_path(audio_replace(audio, name=id))),
                    format=self.format,
                    bitrate=self.bitrate,
                    codec=self.codec,
                )
                f.close()  # Don't leak fd's

        if dry_run:
            # Return whether the caller can skip calling us for real because there's no work to do
            would_noop = (
                id == input_id if unsafe_fs else  # HACK For tests
                not do_write
            )
            return (would_noop, id)
        elif not load:
            # Skip read operation
            #   - e.g. for bulk-transcode cache warming, where we don't need the return
            return None
        elif id == input_id:
            # Skip read operation, since we'd just be re-reading the input audio from disk again
            #   - This assumes we didn't change audio anywhere along the way
            assert not do_write  # The only branch above where we change audio
            return audio
        else:
            # Else re-read audio from file so that audio._data reflects the transcoding
            #   - Property: audio._data always reflects audio.name
            #   - e.g. if audio.name is 'foo.enc(mp3,64k)', then audio._data should be the bytes given by a 64k mp3 encoding
            return self.read_audio(id)

    @classmethod
    def read_audio(cls, id: str, **kwargs) -> Audio:
        log.info(f'Read: {id}')
        audio = audio_from_file_in_data_dir(id, **kwargs)
        return audio
