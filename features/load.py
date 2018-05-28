from collections import OrderedDict
import glob
import os.path
import re
from typing import List

import audiosegment
import pandas as pd
from potoo.pandas import consumes_cols, df_cats_to_str

from cache import cache
from constants import cache_dir, data_dir, standard_sample_rate_hz
from datasets import DATASETS, metadata_from_audio
from datatypes import Recording, RecordingDF
import metadata
from util import df_apply_with_progress, ensure_parent_dir


def load_recs(datasets: List[str] = None) -> RecordingDF:
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


@consumes_cols('id', 'dataset', 'path')
def recs_load_metadata(recs: RecordingDF, **kwargs) -> RecordingDF:
    if 'species' not in recs:
        recs = recs_load_data(recs, rec_load=rec_load_cached, include_metadata=True, include_audio=False, **kwargs)
    return recs


@consumes_cols('id', 'dataset', 'path')
def recs_load_audio(recs: RecordingDF, **kwargs) -> RecordingDF:
    if 'audio' not in recs:
        recs = recs_load_data(recs, rec_load=rec_load, include_metadata=False, include_audio=True, **kwargs)
    return recs


@consumes_cols('id', 'dataset', 'path')
def recs_load_metadata_and_audio(recs: RecordingDF, **kwargs) -> RecordingDF:
    if 'species' not in recs or 'audio' not in recs:
        recs = recs_load_data(recs, rec_load=rec_load, include_metadata=True, include_audio=True, **kwargs)
    return recs


@consumes_cols('id', 'dataset', 'path')
def recs_load_data(recs: RecordingDF, rec_load, dask_opts={}, **kwargs) -> RecordingDF:
    return (recs
        .pipe(df_apply_with_progress, **dask_opts, f=lambda row: pd.Series({
            **row,
            **rec_load(Recording(**row), **kwargs).asdict(),
        }))
        .pipe(RecordingDF)
        .pipe(lambda df:
            df.sort_values('species') if 'species' in df else df
        )
        .pipe(RecordingDF)
    )


@cache(version=1, verbose=0)
def rec_load_cached(rec: Recording, *args, **kwargs) -> Recording:
    """Like rec_load, except drop audio and cache (to skip .wav file read on cache hit)"""
    rec_out = rec_load(rec, *args, **kwargs)
    rec_out.audio = rec.audio or None  # Drop new .audio / preserve existing .audio
    return rec_out


# Caching doesn't help here, since our bottleneck is file read (.wav), which is also cache hit's bottleneck
def rec_load(rec: Recording, include_audio=True, include_metadata=True, **kwargs) -> Recording:
    """Load metadata and audio onto an existing rec"""
    audio = load_audio(rec.path, **kwargs)
    samples = audio.to_numpy_array()
    return Recording(**{
        **rec.asdict(),
        **(dict() if not include_metadata else dict(
            **metadata_from_audio(rec.id, rec.dataset),
            duration_s=audio.duration_seconds,
            samples_mb=len(samples) * audio.sample_width / 1024**2,
            samples_n=len(samples),
        )),
        **(dict() if not include_audio else dict(
            audio=audio,
        )),
    })


def load_audio(
    path: str,
    cache: bool = True,
    channels: int = 1,
    sample_rate: int = standard_sample_rate_hz,
    sample_width_bit: int = 16,
    verbose: bool = False,
) -> audiosegment.AudioSegment:
    """
    Load an audio file, and (optionally) cache a standardized .wav for faster subsequent loads
    """

    # Interpret relative paths as relative to data_dir (leave absolute paths as is)
    if not os.path.isabs(path):
        path = os.path.join(data_dir, path)

    _print = print if verbose else lambda *args, **kwargs: None
    if not cache:
        audio = audiosegment.from_file(path)
    else:

        rel_path_noext, _ext = os.path.splitext(os.path.relpath(path, data_dir))
        params_id = f'{sample_rate}hz-{channels}ch-{sample_width_bit}bit'
        cache_path = f'{cache_dir}/{params_id}/{rel_path_noext}.wav'
        if not os.path.exists(cache_path):
            _print(f'Caching: {cache_path}')
            in_audio = audiosegment.from_file(path)
            std_audio = in_audio.resample(
                channels=channels,
                sample_rate_Hz=sample_rate,
                sample_width=sample_width_bit // 8,
            )
            std_audio.export(ensure_parent_dir(cache_path), 'wav')
        # Always load from disk, for consistency
        audio = audiosegment.from_file(cache_path)

    # HACK Make audiosegment.AudioSegment attrs more ergonomic
    audio = audiosegment_std_name(audio)

    return audio


def audiosegment_std_name(
    audio: audiosegment.AudioSegment,
    data_dir=data_dir,
    cache_dir=cache_dir,
) -> audiosegment.AudioSegment:
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
