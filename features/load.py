from collections import OrderedDict
import glob
import os.path
import re
from typing import List

import audiosegment
import dataclasses
import pandas as pd

from constants import cache_dir, data_dir, standard_sample_rate_hz
from datasets import datasets, metadata_from_audio
from datatypes import Recording
import metadata
from util import df_apply_with_progress, ensure_parent_dir


def load_recs_paths(dataset_ids: List[str] = None) -> pd.DataFrame:
    return pd.DataFrame([
        dict(
            dataset=dataset,
            path=os.path.relpath(path, data_dir),
        )
        for dataset, pattern in datasets.items()
        if not dataset_ids or dataset in dataset_ids
        for path in glob.glob(f'{data_dir}/{pattern}')
        if not os.path.isdir(path)
    ])


def load_recs_data(recs_paths: pd.DataFrame, dask_opts={}, **kwargs) -> pd.DataFrame:
    return (recs_paths
        .pipe(df_apply_with_progress, **dask_opts, f=lambda row: pd.Series(OrderedDict(
            dataclasses.asdict(load_rec(
                row.dataset,
                row.path,
                **kwargs,
            ))
        )))
        .astype({
            # Map str -> category for cols that have category dtypes available
            'species': metadata.species.df.shorthand.dtype,
            'species_longhand': metadata.species.df.longhand.dtype,
            'species_com_name': metadata.species.df.com_name.dtype,
        })
        # Default sort is taxo
        .sort_values('species')
    )


def load_rec(
    dataset: str,
    path: str,
    metadata_only=False,
    audio=False,
    **kwargs,
) -> Recording:
    audio = load_audio(path, **kwargs)
    samples = audio.to_numpy_array()
    return Recording(
        **metadata_from_audio(dataset, audio),
        duration_s=audio.duration_seconds,
        samples_mb=len(samples) * audio.sample_width / 1024**2,
        samples_n=len(samples),
        audio=audio if audio and not metadata_only else None,
    )


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
