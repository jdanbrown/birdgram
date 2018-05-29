from collections import OrderedDict
from functools import partial
import glob
import os.path
import re
from typing import List

from attrdict import AttrDict
import audiosegment
from dataclasses import dataclass
import pandas as pd
from potoo.pandas import requires_cols, df_cats_to_str
from potoo.util import round_sig

from cache import cache
from constants import cache_dir, data_dir, standard_sample_rate_hz
from datasets import DATASETS, metadata_from_dataset
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

    def recs(self, datasets: List[str] = None, limit: int = None) -> RecordingDF:
        """Load recs with metadata from fs"""
        return RecordingDF(
            self.recs_paths(datasets)
            [:limit]
            .pipe(lambda df: pd.concat(axis=1, objs=[df, self.metadata(df)]))
            .sort_values('species')
        )

    def recs_paths(self, datasets: List[str] = None) -> RecordingDF:
        """Load recs.{id,dataset,path} <- fs"""
        return RecordingDF([
            Recording(
                id=os.path.splitext(os.path.relpath(path, data_dir))[0],
                dataset=dataset,
                path=os.path.relpath(path, data_dir),
            ).asdict()
            for dataset, pattern in DATASETS.items()
            if not datasets or dataset in datasets
            for path in glob.glob(f'{data_dir}/{pattern}')
            if not os.path.isdir(path)
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
        metadata = RecordingDF(map_with_progress(self._metadata, df_rows(recs), scheduler='threads'))
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
        samples = audio.to_numpy_array()
        return AttrDict(
            **metadata_from_dataset(rec.id, rec.dataset),
            # TODO De-dupe these with self.METADATA
            duration_s=audio.duration_seconds,
            samples_mb=len(samples) * audio.sample_width / 1024**2,
            samples_n=len(samples),
        )

    @short_circuit(lambda self, recs: recs.get('audio'))
    def audio(self, recs: RecordingDF) -> Column[Audio]:
        """.audio <- .path"""
        log('Load.audio:in', **{
            'len(recs)': len(recs),
            'len(recs) per dataset': recs.dataset.value_counts().to_dict(),
        })
        audio = map_with_progress(self._audio, df_rows(recs), scheduler='processes')
        log('Load.audio:out', **{
            'len(audio)': len(audio),
        })
        return audio

    @short_circuit(lambda self, rec: rec.get('audio'))
    # Caching doesn't help here, since our bottleneck is file read (.wav), which is also cache hit's bottleneck
    def _audio(self, rec: Row) -> Audio:
        """audio <- .path, and (optionally) cache a standardized .wav for faster subsequent loads"""

        path = rec.path
        c = self.audio_config

        # Interpret relative paths as relative to data_dir (leave absolute paths as is)
        if not os.path.isabs(path):
            path = os.path.join(data_dir, path)

        # Cache transcribed audio, if requested
        if c.cache_audio:
            rel_path_noext, _ext = os.path.splitext(os.path.relpath(path, data_dir))
            params_id = f'{c.sample_rate}hz-{c.channels}ch-{c.sample_width_bit}bit'
            cache_path = f'{cache_dir}/{params_id}/{rel_path_noext}.wav'
            if not os.path.exists(cache_path):
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
        audio = audiosegment.from_file(path)

        # Make audiosegment.AudioSegment attrs more ergonomic
        audio = self._ergonomic_audio(audio)

        return audio

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
