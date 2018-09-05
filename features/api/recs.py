from typing import Sequence

import pandas as pd
from potoo.pandas import *
import structlog

from api.server_globals import sg
from datasets import xc_meta_to_xc_raw_recs, xc_raw_recs_to_xc_recs
from util import *
from viz import *


def xc_meta(
    species: str,
    limit: int = 10,
) -> pd.DataFrame:
    limit = np.clip(limit, 0, 50)
    return (sg.xc_meta
        [lambda df: df.species == species]
        [:limit]
    )


def xc_species_html(
    species: str,
    quality: Sequence[str] = ['A', 'B'],  # TODO Test
    limit: int = 10,
    thumb_s: float = 1.5,  # Omit thumb if falsy
    audio_s: float = 10,  # Omit audio if falsy
) -> pd.DataFrame:
    limit = np.clip(limit, 0, 50)
    thumb_s = np.clip(thumb_s, 0, 10)
    audio_s = np.clip(audio_s, 0, 30)
    plot_many_kwargs = dict(
        scale=dict(h=80),  # Best if h is multiple of 40 (because of low-level f=40 in Melspectro)
        progress=dict(use='dask', scheduler='threads'),  # threads > sync, threads >> processes
        _nocache=True,  # Dev: disable plot_many cache since it's blind to most of our sub-many code changes [TODO Revisit]
    )
    return (sg.xc_meta
        [lambda df: df.species == species]
        [lambda df: df.quality.isin(quality)]
        .sort_index(ascending=True)  # This is actually descending... [why?]
        [:limit]
        # Heavy: .audio, .spectro, .feat [How to make these more lightweight?]
        .pipe(lambda df: (df
            .pipe(xc_meta_to_xc_raw_recs, load=sg.load)
            .pipe(xc_raw_recs_to_xc_recs, projection=sg.projection)
        ))
        # Clip .audio/.spectro to thumb_s/audio_s
        .pipe(df_assign_first, **{
            **({} if not thumb_s else {
                'thumb': df_cell_spectros(plot_thumb.many, sg.features, thumb_s=thumb_s, **plot_many_kwargs),
            }),
            **({} if not audio_s else {
                'micro': df_cell_spectros(plot_spectro_micro.many, sg.features, wrap_s=audio_s, **plot_many_kwargs),
            }),
        })
        [lambda df: ['quality', *[c for c in ['thumb', 'micro'] if c in df]]]
    )


# TODO
def xc_similar_html(id: int) -> pd.DataFrame:
    return pd.DataFrame([
        dict(id=id, note='similar'),
    ])
