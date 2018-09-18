import inspect
import linecache
from typing import Sequence

from more_itertools import one
import pandas as pd
from potoo.pandas import *
import structlog

from api.server_globals import sg
from api.util import *
from datasets import xc_meta_to_paths, xc_meta_to_raw_recs, xc_raw_recs_to_recs
from sp14.model import rec_neighbors_by, rec_probs, Search
from util import *
from viz import *


def xc_meta(
    species: str,
    n_recs: int = 10,
) -> pd.DataFrame:
    species = species_for_query(species)
    require(n_recs > 0)
    n_recs = np.clip(n_recs, 0, 50)
    return (sg.xc_meta
        [lambda df: df.species == species]
        [:n_recs]
    )


def xc_species_html(
    species: str = None,
    quality: str = None,
    n_recs: int = 10,
    audio_s: float = 10,
    thumb_s: float = 0,
    scale: float = 2,
) -> pd.DataFrame:

    # Params
    require(n_recs > 0)
    require(audio_s > 0)
    species = species_for_query(species)
    if not species: return pd.DataFrame([])
    quality = quality or 'ab'
    quality = [q.upper() for q in quality]
    quality = [{'N': 'no score'}.get(q, q) for q in quality]
    n_recs = np.clip(n_recs, 0, 50)
    audio_s = np.clip(audio_s, 0, 30)
    thumb_s = np.clip(thumb_s, 0, 10)
    scale = np.clip(scale, .5, 10)

    return (sg.xc_meta
        [lambda df: df.species == species]
        .pipe(require_nonempty_df, 'No recs found', species=species)
        [lambda df: df.quality.isin(quality)]
        .pipe(require_nonempty_df, 'No recs found', species=species, quality=quality)
        .sort_index(ascending=True)  # This is actually descending... [why?]
        [:n_recs]
        .reset_index()
        .pipe(recs_featurize, audio_s=audio_s, thumb_s=thumb_s, scale=scale)
        .pipe(recs_view)
        [lambda df: [c for c in [
            'xc_id', 'similar', 'com_name', 'species', 'quality',
            'thumb', 'micro',
            'duration_s', 'month_day', 'background_species', 'place', 'remarks',
        ] if c in df]]
    )


# TODO TODO Update for .audio/slice/.spectro/.feat (like /species)
def xc_similar_html(
    xc_id: int,
    quality: str = None,
    n_sp: int = 3,
    n_recs_r: int = 3,
    n_total: int = 9,
    audio_s: float = 10,
    thumb_s: float = 0,
    scale: float = 2,
    **plot_many_kwargs,
) -> pd.DataFrame:

    # Params
    require(audio_s > 0)
    quality  = quality or 'ab'
    quality  = [q.upper() for q in quality]
    quality  = [{'N': 'no score'}.get(q, q) for q in quality]
    n_sp     = n_sp     and np.clip(n_sp,     0,  None)  # TODO Try unlimited (was: 50)
    n_recs_r = n_recs_r and np.clip(n_recs_r, 0,  None)  # TODO Try unlimited (was: 1000)
    n_total  = n_total  and np.clip(n_total,  0,  None)  # TODO Try unlimited (was: 100)
    audio_s  = audio_s  and np.clip(audio_s,  0,  30)
    thumb_s  = thumb_s  and np.clip(thumb_s,  0,  10)
    scale    = scale    and np.clip(scale,    .5, 10)

    # Lookup query_rec from xc_meta
    query_rec = (sg.xc_meta
        [lambda df: df.id == xc_id]
        .pipe(require_nonempty_df, 'No recs found', xc_id=xc_id)
        .pipe(recs_featurize_metadata)
        .pipe(recs_featurize_audio, load=sg.load)
        .pipe(recs_featurize_feat)
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
        .pipe(require_nonempty_df, 'No recs found', species=species)
        [lambda df: df.quality.isin(quality)]
        .pipe(require_nonempty_df, 'No recs found', species=species, quality=quality)
        .pipe(df_remove_unused_categories)
        # Sample n_recs_r per species
        .pipe(lambda df: df if n_recs_r is None else (df
            .groupby('species').apply(lambda g: (g
                .sample(n=min(n_recs_r, len(g)), random_state=0)  # TODO HACK Sample to make go faster, until we build up a full cache
            ))
        ))
        .reset_index(level=0, drop=True)  # species, from groupby
        # Featurize
        .pipe(recs_featurize_metadata)
        .pipe(recs_featurize_audio, load=sg.load)
        .pipe(recs_featurize_feat)
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
        .pipe(recs_featurize_spectro)
        .pipe(recs_featurize_micro_thumb, audio_s=audio_s, thumb_s=thumb_s, scale=scale, **plot_many_kwargs)
        .pipe(recs_view)
        [lambda df: [c for c in [
            'dist',
            'xc_id', 'similar', 'com_name', 'species', 'quality',
            'thumb', 'micro',
            'duration_s', 'month_day', 'background_species', 'place', 'remarks',
        ] if c in df]]
    )


def recs_featurize(
    recs: pd.DataFrame,
    audio_s: float,
    thumb_s: float,
    scale: float,
    **plot_many_kwargs,
) -> pd.DataFrame:
    return (recs
        .pipe(recs_featurize_metdata_audio_slice, audio_s=audio_s)
        .pipe(recs_featurize_spectro)
        .pipe(recs_featurize_feat)
        .pipe(recs_featurize_micro_thumb, audio_s=audio_s, thumb_s=thumb_s, scale=scale, **plot_many_kwargs)
    )


progress_kwargs = dict(
    # use='dask', scheduler='threads',  # TODO TODO Restore par
    # use='sync',
    use=None,  # Dev: silence progress bars
)


def recs_featurize_metdata_audio_slice(recs: pd.DataFrame, audio_s: float) -> pd.DataFrame:
    """Featurize: Add .audio with slice"""
    assert audio_s is not None and audio_s > 0, f"{audio_s}"
    try:
        # Try loading sliced .audio directly
        return (recs
            .pipe(recs_featurize_metadata, to_paths=lambda recs: [
                (dataset, abs_sliced_path)
                for (dataset, abs_path) in xc_meta_to_paths(recs)
                for id in [str(Path(abs_path).relative_to(data_dir))]
                for sliced_id in [audio_id_add_ops(id,
                    # HACK TODO Find a principled way to synthesize id for sliced audio (multiple concerns to untangle...)
                    'resample(%(sample_rate)s,%(channels)s,%(sample_width_bit)s)' % sg.load.audio_config,
                    'enc(wav)',
                    'slice(%s,%s)' % (0, int(1000 * audio_s)),
                    'spectro_denoise()',
                    'enc(%(format)s,%(codec)s,%(bitrate)s)' % config.audio_persist.audio_kwargs,
                )]
                for abs_sliced_path in [str(Path(data_dir) / sliced_id)]
            ])
            .pipe(recs_featurize_audio, load=load_for_audio_persist())
            # TODO TODO WIP Not working yet... (still 2x "Read:" in the api logs, on cache miss or hit)
        )
    except FileNotFoundError:
        # Fallback to loading full .audio and computing the slice ourselves (which will cache for next time)
        log.warn('Falling back to uncached audio slices', audio_s=audio_s, len_recs=len(recs))
        return (recs
            .pipe(recs_featurize_metadata)
            .pipe(recs_featurize_audio, load=sg.load)
            .pipe(recs_featurize_slice, audio_s=audio_s)
            .pipe(recs_audio_persist, progress_kwargs=progress_kwargs)
        )


# TODO Sort out "audio metadata" vs. "xc_meta"
#   - How about unify by folding audio metadata into xc_meta like page metadata?
#       def xc_meta_to_raw_recs(load):
#           xc_paths = xc_meta.map(...)
#           xc_raw_recs = (
#               load.recs(xc_paths)
#               .join(xc_meta.pipe(clean))
#           )
def recs_featurize_metadata(recs: pd.DataFrame, to_paths=None) -> pd.DataFrame:
    """Featurize: Add audio metadata (not .audio) <- xc_meta"""
    return (recs
        .pipe(xc_meta_to_raw_recs, to_paths=to_paths, load=sg.load)
    )


def recs_featurize_audio(
    recs: pd.DataFrame,
    load,  # Explicit load to help us stay aware of which one we're using at all times (lots of wav vs. mp4 confusion)
) -> pd.DataFrame:
    """Featurize: Add .audio"""
    return (recs
        .pipe(load.audio, **progress_kwargs)  # procs barf on serdes error
    )


def recs_featurize_slice(recs: pd.DataFrame, audio_s: float) -> pd.DataFrame:
    """Featurize: Slice .audio (before .spectro/.feat/.thumb)"""
    return (recs
        .pipe(df_map_rows_progress, desc='slice_audio', **progress_kwargs, f=lambda row: (
            sg.features.slice_audio(row, 0, audio_s)
        ))
    )


def recs_featurize_spectro(recs: pd.DataFrame) -> pd.DataFrame:
    """Featurize: Add .spectro"""
    return (recs
        # TODO HACK Workaround bug
        #   - In server, .spectro column is present but all nan, which breaks downstream
        #   - In notebook, works fine
        #   - Workaround: force-drop .spectro column if present
        #   - Tech debt: Not general, very error prone -- e.g. does this affect .feat? .audio?
        .drop(columns=['spectro'], errors='ignore')
        .assign(spectro=lambda df: sg.features.spectro(df, **progress_kwargs, cache=True))  # threads >> sync, procs
    )


def recs_featurize_feat(recs: pd.DataFrame) -> pd.DataFrame:
    """Featurize: Add .feat"""
    return (recs
        .pipe(sg.projection.transform)
    )


def recs_featurize_micro_thumb(
    recs: pd.DataFrame,
    audio_s: float,
    thumb_s: float,
    scale: float,
    **plot_many_kwargs,
) -> pd.DataFrame:
    """Featurize: Add .thumb, .micro <- .spectro, .audio"""
    plot_many_kwargs = {
        **plot_many_kwargs,
        'scale': dict(h=int(40 * scale)),  # Best if h is multiple of 40 (because of low-level f=40 in Melspectro)
        'progress': dict(**progress_kwargs),  # threads > sync, threads >> processes
        '_nocache': True,  # Dev: disable plot_many cache since it's blind to most of our sub-many code changes [TODO Revisit]
    }
    return (recs
        # Clip .audio/.spectro to audio_s/thumb_s
        .pipe(df_assign_first, **{
            **({} if not audio_s else dict(
                micro=df_cell_spectros(plot_micro.many, sg.features, **plot_many_kwargs,
                    pad_s=audio_s,
                ),
            )),
            **({} if not thumb_s else dict(
                thumb=df_cell_spectros(plot_thumb.many, sg.features, **plot_many_kwargs,
                    thumb_s=thumb_s,
                ),
            )),
        })
    )


def recs_view(
    recs: pd.DataFrame,
) -> pd.DataFrame:
    return (recs
        .pipe(lambda df: df if 'xc_id' not in df else (df
            .assign(
                # TODO Simplify: Have to do .similar before .xc_id, since we mutate .xc_id
                similar=lambda df: df_map_rows(df, lambda row: f'''
                    <a href="/recs/xc/similar?xc_id={row.xc_id}">similar</a>
                '''),
                xc_id=lambda df: df_map_rows(df, lambda row: f'''
                    <a href="https://www.xeno-canto.org/{row.xc_id}">{row.xc_id}</a>
                '''),
            )
        ))
        .pipe(lambda df: df if 'species' not in df else (df
            .rename(columns={
                'species_com_name': 'com_name',
            })
            .assign(
                # species=lambda df: df_map_rows(df, lambda row: f'''
                #     <a href="/recs/xc/species?species={row.species}">{row.species}</a> <br/>
                #     <a href="/recs/xc/species?species={row.species}">{row.com_name}</a>
                # '''),
                # TODO Simplify: Have to do .com_name before .species, since we mutate .species
                com_name=lambda df: df_map_rows(df, lambda row: f'''
                    <a href="/recs/xc/species?species={row.species}">{row.com_name}</a>
                '''),
                species=lambda df: df_map_rows(df, lambda row: f'''
                    <a href="/recs/xc/species?species={row.species}">{row.species}</a>
                '''),
            )
        ))
        .pipe(lambda df: df if 'background_species' not in df else (df
            .pipe(df_col_map,
                background_species=lambda xs: ', '.join(xs),
            )
        ))
    )


def species_for_query(species_query: str) -> str:
    species_query = species_query.strip()
    species = metadata.species[species_query]
    if not species:
        raise ApiError(400, 'No species found', species_query=species_query)
    return species.shorthand


def require(x: bool, **data):
    if not x:
        caller = inspect.stack(context=0)[1]
        require_stmt = linecache.getline(caller.filename, caller.lineno).strip()
        raise ApiError(400, require_stmt, **data)


def require_nonempty_df(df: pd.DataFrame, msg: str, **data) -> pd.DataFrame:
    if df.empty:
        raise ApiError(400, msg, **data)
    return df
