from collections import OrderedDict
import glob
import os.path
import re
from typing import List

import attr
import audiosegment
import pandas as pd

from constants import cache_dir, data_dir, standard_sample_rate_hz
from datatypes import Recording
from datasets import mlsp2013
import metadata
from util import df_apply_with_progress


def load_recs_paths(datasets: List[str] = None) -> pd.DataFrame:
    return pd.DataFrame([
        dict(
            dataset=dataset,
            path=os.path.relpath(path, data_dir),
        )
        for dataset, pattern in {
            'recordings': 'recordings/*',
            'recordings-new': 'recordings-new/*',
            'peterson-field-guide': 'peterson-field-guide/*/audio/*',
            'birdclef-2015': 'birdclef-2015/organized/wav/*',
            'warblrb10k': 'dcase-2018/warblrb10k_public_wav/*',
            'ff1010bird': 'dcase-2018/ff1010bird_wav/*',
            'nips4b': 'nips4b/all_wav/*',
            'mlsp-2013': 'mlsp-2013/mlsp_contest_dataset/essential_data/src_wavs/*',
        }.items()
        if not datasets or dataset in datasets
        for path in glob.glob(f'{data_dir}/{pattern}')
        if not os.path.isdir(path)
    ])


def load_recs_data(recs_paths: pd.DataFrame, dask_opts={}, **kwargs) -> pd.DataFrame:
    return (recs_paths
        .pipe(df_apply_with_progress, **dask_opts, f=lambda row: pd.Series(OrderedDict(
            attr.asdict(load_rec(
                row.dataset,
                row.path,
                **kwargs,
            ))
        )))
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
        samples=samples if not metadata_only else None,
        audio=audio if audio and not metadata_only else None,
    )


def metadata_from_audio(dataset, audio) -> dict:
    name = audio.name
    name_parts = name.split('/')
    basename = name_parts[-1]
    species = None
    species_query = None
    if dataset == 'peterson-field-guide':
        species_query = name.split('/')[1]
    elif dataset == 'recordings-new':
        m = re.match(r'^([A-Z]{4}) ', basename)
        if m: species_query = m.groups()[0]
    elif dataset == 'mlsp-2013':
        # TODO Generalize species[species_query] to work on multi-label species (e.g. 'SOSP,WIWA')
        #   - Works fine for now because it passes through queries it doesn't understand, and these are already codes
        train_labels = mlsp2013.train_labels_for_filename.get(basename, ['XX'])  # If missing it's an unlabeled test rec
        species = 'none' if train_labels == [] else ','.join(sorted(train_labels))
    return OrderedDict(
        dataset=dataset,
        name=audio.name,
        species=species or metadata.species[species_query, 'shorthand'] or 'XX',
        species_query=species_query,
        basename=basename,
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
