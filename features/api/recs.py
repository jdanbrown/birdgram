from typing import Sequence

from more_itertools import one
import pandas as pd
from potoo.pandas import *
import structlog

from api.server_globals import sg
from api.util import *
from datasets import xc_meta_to_xc_raw_recs, xc_raw_recs_to_xc_recs
from sp14.model import rec_neighbors_by, rec_probs, Search
from util import *
from viz import *


def xc_meta(
    species: str,
    n_recs: int = 10,
) -> pd.DataFrame:
    n_recs = np.clip(n_recs, 0, 50)
    return (sg.xc_meta
        [lambda df: df.species == species_for_query(species)]
        [:n_recs]
    )


def xc_species_html(
    species: str = None,
    quality: str = None,
    n_recs: int = 10,
    thumb_s: float = 0,
    audio_s: float = 5,
    scale: float = 2,
) -> pd.DataFrame:

    # Params
    if not species: return pd.DataFrame([])
    quality = quality or 'ab'
    quality = [q.upper() for q in quality]
    quality = [{'N': 'no score'}.get(q, q) for q in quality]
    n_recs = np.clip(n_recs, 0, 50)
    thumb_s = np.clip(thumb_s, 0, 10)
    audio_s = np.clip(audio_s, 0, 30)
    scale = np.clip(scale, .5, 10)

    return (sg.xc_meta
        [lambda df: df.species == species_for_query(species)]
        [lambda df: df.quality.isin(quality)]
        .sort_index(ascending=True)  # This is actually descending... [why?]
        [:n_recs]
        .pipe(recs_featurize, thumb_s=thumb_s, audio_s=audio_s, scale=scale)
        .reset_index()
        .pipe(recs_view)
        [lambda df: [
            'xc_id', 'similar', 'species', 'quality',
            *[c for c in ['thumb', 'micro'] if c in df]
        ]]
    )


def xc_similar_html(
    xc_id: int,
    quality: str = None,
    n_sp: int = 3,
    n_recs_r: int = 3,
    n_total: int = 9,
    thumb_s: float = 0,
    audio_s: float = 5,
    scale: float = 2,
) -> pd.DataFrame:

    # Params
    quality  = quality or 'ab'
    quality  = [q.upper() for q in quality]
    quality  = [{'N': 'no score'}.get(q, q) for q in quality]
    n_sp     = n_sp     and np.clip(n_sp,     0,  50)
    n_recs_r = n_recs_r and np.clip(n_recs_r, 0,  1000)
    n_total  = n_total  and np.clip(n_total,  0,  100)
    thumb_s  = thumb_s  and np.clip(thumb_s,  0,  10)
    audio_s  = audio_s  and np.clip(audio_s,  0,  30)
    scale    = scale    and np.clip(scale,    .5, 10)

    # Lookup query_rec from xc_meta
    query_rec = (sg.xc_meta
        [lambda df: df.id == xc_id]
        .pipe(recs_featurize, spectro=False, thumb_s=None, audio_s=None, scale=1)
        .pipe(lambda df: one(df_rows(df)))
    )

    # Compute query_probs from search
    query_probs = (
        rec_probs(query_rec, sg.search)
        [:n_sp]
    )

    # Compute search_recs from xc_meta, and featurize (.audio, .feat)
    #   - TODO Featurize is slow and heavy, because we can't filter yet until we have .dist, which relies on .feat...
    # memory.log.level = 'debug'  # TODO Nontrivial number of cache misses slowing us down -- why are there misses?
    search_recs = (sg.xc_meta
        # Filter
        [lambda df: df.species.isin(query_probs.species)]
        [lambda df: df.quality.isin(quality)]
        .pipe(df_remove_unused_categories)
        # Sample n_recs_r per species
        .groupby('species').apply(lambda g: (g
            .sample(n=n_recs_r)  # TODO HACK Sample to make go faster, until we build up a full cache
        ))
        .reset_index(level=0, drop=True)  # species, from groupby
        # Featurize
        .pipe(recs_featurize, spectro=False, thumb_s=None, audio_s=None, scale=1)
        # Include query_rec in results (already featurized)
        .pipe(lambda df: df if query_rec.name in df.index else pd.concat([
            DF([query_rec]).pipe(df_set_index_name, 'xc_id'),  # Restore index name, lost by df->series->df
            df,
        ]))
    )

    # Compute result_recs from (query_rec + search_recs).feat
    result_recs = (
        rec_neighbors_by(
            query_rec=query_rec,
            search_recs=search_recs if query_rec.name in search_recs.index else pd.concat([
                DF([query_rec]).pipe(df_set_index_name, 'xc_id'),  # HACK Force compat with xc_recs, e.g. if from user_recs
                search_recs,
            ]),
            by=Search.X,  # TODO Add user control to toggle dist function
        )
        # TODO [later] Restore n_recs (maybe after we've built enough cache to no longer need n_recs_r?)
        # .groupby('species').apply(lambda g: (g
        #     .sort_values('dist', ascending=True)
        #     [:n_recs]
        # ))
        # .reset_index(level=0, drop=True)  # species, from groupby
        .sort_values('dist', ascending=True)
        [:n_total]
    )

    # Featurize result_recs (.spectro) + view
    return (result_recs
        .reset_index()
        .pipe(recs_featurize, thumb_s=thumb_s, audio_s=audio_s, scale=scale)
        .pipe(recs_view)
        [lambda df: [
            'dist', 'xc_id', 'similar', 'species', 'quality',
            *[c for c in ['thumb', 'micro'] if c in df],
            'month_day', 'background_species', 'place', 'remarks',
        ]]
        .pipe(df_col_map,
            background_species=lambda xs: ', '.join(xs),
        )
    )


def recs_featurize(
    recs: pd.DataFrame,
    thumb_s: float,
    audio_s: float,
    scale: float,
    audio=True,
    feat=True,
    spectro=True,
) -> pd.DataFrame:
    plot_many_kwargs = dict(
        scale=dict(h=int(40 * scale)),  # Best if h is multiple of 40 (because of low-level f=40 in Melspectro)
        progress=dict(use='dask', scheduler='threads'),  # threads > sync, threads >> processes
        _nocache=True,  # Dev: disable plot_many cache since it's blind to most of our sub-many code changes [TODO Revisit]
    )
    return (recs
        # Heavy: .audio, .spectro, .feat [How to make these more lightweight?]
        .pipe(lambda df: df if df.get('audio', pd.Series()).notnull().any() else (df
            .pipe(xc_meta_to_xc_raw_recs, load=sg.load)
        ))
        # TODO Complexify: feat,spectro = False,False / True,False / True,True
        .pipe(lambda df: df if df.get('spectro', pd.Series()).notnull().any() else (df
            # TODO HACK Workaround bug
            #   - In server, .spectro column is present but all nan, which breaks downstream
            #   - In notebook, works fine
            #   - Workaround: force-drop a .spectro column if present
            #   - Tech debt: Not general, very error prone -- e.g. does this affect .feat? .audio?
            .drop(columns=['spectro'], errors='ignore')
            .pipe(xc_raw_recs_to_xc_recs, projection=sg.projection, audio=audio, feat=feat, spectro=spectro)
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
    )


def recs_view(
    recs: pd.DataFrame,
) -> pd.DataFrame:
    return (recs
        .pipe(lambda df: df if 'xc_id' not in df else (df
            .assign(
                # [TODO Clean up: Have to do .similar before .xc_id, since we mutate .xc_id]
                similar=lambda df: df_map_rows(df, lambda row: f'''
                    <a href="/recs/xc/similar?xc_id={row.xc_id}">similar</a>
                '''),
                xc_id=lambda df: df_map_rows(df, lambda row: f'''
                    <a href="https://www.xeno-canto.org/{row.xc_id}">XC</a>
                '''),
            )
        ))
        .pipe(lambda df: df if 'species' not in df else (df
            .rename(columns={
                'species_com_name': 'com_name',
            })
            .assign(
                species=lambda df: df_map_rows(df, lambda row: f'''
                    <a href="/recs/xc/species?species={row.species}">{row.com_name}</a> <br/>
                    <a href="/recs/xc/species?species={row.species}">{row.species}</a>
                '''),
            )
        ))
    )


def species_for_query(species_query: str) -> str:
    species_query = species_query.strip()
    species = metadata.species[species_query]
    if not species:
        raise ApiError(400, 'No species found', species_query=species_query)
    return species.shorthand
