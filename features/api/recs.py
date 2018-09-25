import inspect
import linecache
import numbers
from typing import Optional, Sequence

from more_itertools import one
import pandas as pd
from potoo.pandas import *
from potoo.util import ensure_startswith
import sklearn
import structlog

from api.server_globals import sg
from api.util import *
from cache import *
from config import config
from datasets import xc_meta_to_paths, xc_meta_to_raw_recs, xc_raw_recs_to_recs
from sp14.model import rec_neighbors_by, rec_probs, Search
from util import *
from viz import *


def xc_meta(
    species: str,
    quality: str = None,
    n_recs: int = 10,
) -> pd.DataFrame:

    # Params
    species = species_for_query(species)
    quality = quality or 'ab'
    quality = [q.upper() for q in quality]
    quality = [{'N': 'no score'}.get(q, q) for q in quality]
    n_recs = np.clip(n_recs, 0, 50)

    return (sg.xc_meta
        [lambda df: df.species == species]
        [lambda df: df.quality.isin(quality)]
        [:n_recs]
    )


def xc_species_html(
    species: str = None,
    quality: str = None,
    n_recs: int = 10,
    audio_s: float = 10,
    thumb_s: float = 0,
    scale: float = 2,
    view: bool = None,
    sp_cols: str = None,
    drop_uncached_slice: bool = None,
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
        .pipe(df_require_nonempty_api, 'No recs found', species=species)
        [lambda df: df.quality.isin(quality)]
        .pipe(df_require_nonempty_api, 'No recs found', species=species, quality=quality)
        .sort_index(ascending=True)  # This is actually descending... [why?]
        [:n_recs]
        .reset_index(drop=True)  # Drop RangeIndex (irregular after slice)
        .pipe(recs_featurize, audio_s=audio_s, thumb_s=thumb_s, scale=scale, drop_uncached_slice=drop_uncached_slice)
        .pipe(recs_view, view=view, sp_cols=sp_cols)
        [lambda df: [c for c in [
            'xc', 'xc_id',
            'com_name', 'species', 'quality',
            'thumb', 'slice',
            'duration_s', 'month_day', 'background_species', 'place', 'remarks',
        ] if c in df]]
    )


def xc_similar_html(
    xc_id: int,
    quality: str = None,
    n_sp: int = 3,
    sample_r: int = 3,
    n_total: int = 9,
    audio_s: float = 10,
    thumb_s: float = 0,
    scale: float = 2,
    # dist: str = 'f',      # dist (d_*) for picking closest k recs [TODO Not yet used, currently we return all search_recs]
    d_feats: str = 'fp',    # f (d_f, "feat dist"), p (d_p, "preds dist")
    d_metrics: str = '2c',  # 2 (l2), 1 (l1), c (cosine)
    sort: str = None,       # XXX Won't need after making `dist` work
    random_state: int = 0,
    view: bool = None,
    sp_cols: str = None,
    drop_uncached_slice: bool = None,
    skip_load_audio: bool = None,
    **plot_many_kwargs,
) -> pd.DataFrame:

    # Params
    require(audio_s > 0)
    quality   = quality or 'ab'
    quality   = [q.upper() for q in quality]
    quality   = [{'N': 'no score'}.get(q, q) for q in quality]
    n_sp      = n_sp     and np.clip(n_sp,     0,  None)  # TODO Try unlimited (was: 50)
    sample_r  = sample_r and np.clip(sample_r, 0,  None)  # TODO Try unlimited (was: 1000)
    n_total   = n_total  and np.clip(n_total,  0,  None)  # TODO Try unlimited (was: 100)
    audio_s   = audio_s  and np.clip(audio_s,  0,  30)
    thumb_s   = thumb_s  and np.clip(thumb_s,  0,  10)
    scale     = scale    and np.clip(scale,    .5, 10)
    # d_        = lambda x: ensure_startswith(x, 'd_')  # XXX Unused
    # dist      = d_(dist)
    d_feats   = list(d_feats)
    d_metrics = list(d_metrics)
    sort      = sort or 'd_slp'  # TODO Default to first 'd_{m}{f}' [these strs get computed below...]

    # Utils
    d_ = lambda f, m: f'd_{f}{m}'

    # Lookup query_rec from xc_meta
    query_rec = (sg.xc_meta
        [lambda df: df.id == xc_id]
        .pipe(df_require_nonempty_api, 'No recs found', xc_id=xc_id)
        .pipe(recs_featurize_metdata_audio_slice, audio_s=audio_s,
            drop_uncached_slice=drop_uncached_slice, skip_load_audio=skip_load_audio,
        )
        .pipe(recs_featurize_feat)
        .pipe(lambda df: one(df_rows(df)))
    )

    # Compute query_sp_p from search
    query_sp_p = (
        rec_probs(query_rec, sg.search)
        [:n_sp]
        .rename(columns={'p': 'sp_p'})
    )

    # Compute search_recs from xc_meta, and featurize (.audio, .feat)
    #   - TODO Featurize is slow and heavy, because we can't filter yet until we have .dist, which relies on .feat...
    search_recs = (sg.xc_meta
        # Filter
        [lambda df: df.species.isin(query_sp_p.species)]
        .pipe(df_require_nonempty_api, 'No recs found', species=query_sp_p.species)
        [lambda df: df.quality.isin(quality)]
        .pipe(df_require_nonempty_api, 'No recs found', species=query_sp_p.species, quality=quality)
        .pipe(df_remove_unused_categories)
        # HACK Sample sample_r recs per species
        .pipe(lambda df: df if sample_r is None else (df
            .groupby('species').apply(lambda g: (g
                .sample(n=min(sample_r, len(g)), random_state=random_state)
            ))
        ))
        .reset_index(level=0, drop=True)  # species, from groupby
        # Featurize
        # TODO TODO FIXME Slow load.read_audio -- can we just cached .feat?
        .pipe(recs_featurize_metdata_audio_slice, audio_s=audio_s,
            drop_uncached_slice=drop_uncached_slice, skip_load_audio=skip_load_audio,
        )
        .pipe(recs_featurize_feat)
        # Include query_rec in results (already featurized)
        .pipe(lambda df: df if query_rec.xc_id in df.xc_id else pd.concat(
            sort=True,  # [Silence "non-concatenation axis" warning -- not sure what we want, or if it matters...]
            objs=[
                DF([query_rec]),
                df,
            ],
        ))
    )

    # Compute closest k recs
    closest_recs = (
        search_recs  # HACK Keep all search_recs for now, while we iterate on scoring functions
        # rec_neighbors_by(query_rec=query_rec, search_recs=search_recs, by=..., n=...)  # Eventually
    )

    # Augment closest_recs with all dist metrics, so user can evaluate
    d_feats = {f: {
        'f': Search.X,
        'p': sg.search.species_proba,  # (Last investigated: notebooks/app_ideas_6_with_pca)
    }[f] for f in d_feats}
    d_metrics = {m: sklearn.metrics.pairwise.distance_metrics()[{
        '2': 'l2',
        '1': 'l1',
        'c': 'cosine',
    }[m]] for m in d_metrics}
    dist_recs = (closest_recs
        .pipe(lambda df: (df
            .assign(**{
                # Show progress per metric [TODO Clean up messy code]
                k: v
                for f, F in d_feats.items()
                for F_df in [one_progress(desc=f, n=len(df), x=lambda: F(df))]  # Slow: do once, not d_metrics times
                for m, M in d_metrics.items()
                for k in [d_(f, m)]
                for v in [one_progress(desc=k, n=len(df), x=lambda: (
                    M(F_df, F(DF([query_rec])))
                ))]
            })
        ))
    )

    # [orphaned] [later] Restore n_recs (maybe after we've built enough cache to no longer need sample_r?)
    # .groupby('species').apply(lambda g: (g
    #     .sort_values('dist', ascending=True)
    #     [:n_recs]
    # ))
    # .reset_index(level=0, drop=True)  # species, from groupby

    # Rank results
    #   - [later] Add ebird_priors prob
    result_recs = (dist_recs
        # Join in .sp_p for scoring functions
        #   - [Using sort=True to silence "non-concatenation axis" warning -- not sure what we want, or if it matters]
        .merge(how='left', on='species', right=query_sp_p[['species', 'sp_p']],
            sort=True,  # [Silence "non-concatenation axis" warning -- not sure what we want, or if it matters...]
        )
        # Scores (d_*)
        #   - A distance measure in [0,inf), lower is better
        #   - Examples: -log(p), feat dist (d_f), preds dist (d_p)
        #   - Can be meaningfully combined by addition, e.g.
        #       - -log(p) + -log(q) = -log(pq)
        #       - -log(p) + d_f     = ... [Meaningful? Helpful to rescale?]
        #       - -log(p) + d_p     = ... [Meaningful? Helpful to rescale?]
        #       - d_f     + d_p     = ... [Meaningful? Helpful to rescale?]
        .pipe(lambda df: (df
            # Mock scores for query_rec so that it always shows at the top
            .pipe(df_map_rows, lambda row: row if row.xc_id != query_rec.xc_id else series_assign(row,
                sp_p=1,
                # **{d: 0 for d in d_feats},  # XXX Outdated by d_metrics/d_feats [and apparently don't need anymore?]
            ))
            # Derived scores
            .assign(
                d_slp=lambda df: np.abs(-np.log(df.sp_p)),  # d_slp: "species log prob" (abs for 1->0 i/o -0)
                # [ ] Add score that combines (d,slp)
            )
        ))
        # Sort by score (chosen by user)
        .sort_values(sort, ascending=True)
        .reset_index(drop=True)  # Drop RangeIndex (shuffled after sort)
        # Limit
        [:n_total]
    )

    # Featurize result_recs: .spectro + recs_view
    view_recs = (result_recs
        .pipe(recs_featurize_audio, load=load_for_audio_persist())  # TODO TODO FIXME Slow load.read_audio
        .pipe(recs_featurize_slice_thumb, audio_s=audio_s, thumb_s=thumb_s, scale=scale, **plot_many_kwargs)
        .pipe(recs_view, view=view, sp_cols=sp_cols)
        .pipe(lambda df: (df
            .pipe(df_reorder_cols, first=[  # Manually order d_* cols [Couldn't get to work above]
                'd_slp',
                *[c for c in [d_(f, m) for m in '2c' for f in 'fp'] if c in df],
            ])
        ))
        [lambda df: [c for c in [
            'xc', 'xc_id',
            *unique_everseen([
                # sort,  # Show sort col first, for feedback to user  # XXX Nope, confusing to change col order
                *[c for c in df if c.startswith('d_')]  # Scores (d_*)
            ]),
            'com_name', 'species', 'quality',
            'thumb', 'slice',
            'duration_s', 'month_day', 'background_species', 'place', 'remarks',
        ] if c in df]]
    )

    return view_recs


def recs_featurize(
    recs: pd.DataFrame,
    audio_s: float,
    thumb_s: float,
    scale: float,
    drop_uncached_slice: bool = None,
    **plot_many_kwargs,
) -> pd.DataFrame:
    return (recs
        .pipe(recs_featurize_metdata_audio_slice, audio_s=audio_s,
            drop_uncached_slice=drop_uncached_slice, skip_load_audio=skip_load_audio,
        )
        .pipe(recs_featurize_feat)
        .pipe(recs_featurize_slice_thumb, audio_s=audio_s, thumb_s=thumb_s, scale=scale, **plot_many_kwargs)
    )


def recs_featurize_metdata_audio_slice(
    recs: pd.DataFrame,
    audio_s: float,
    # HACK Drop audios with no cache/audio/ slice file instead of recomputing ("Falling back") (which warms cache)
    #   - Invalid input audios don't produce a cache/audio/ file, so if you get one then you're stuck always falling back
    drop_uncached_slice: bool = None,
    # TODO TODO FIXME SLow load.read_audio
    skip_load_audio: bool = None,
) -> pd.DataFrame:
    """Featurize: Add .audio with slice"""

    # Params
    assert audio_s is not None and audio_s > 0, f"{audio_s}"
    drop_uncached_slice = False if drop_uncached_slice is None else drop_uncached_slice
    skip_load_audio = False if skip_load_audio is None else skip_load_audio
    assert not (not drop_uncached_slice and skip_load_audio), "Can't skip load audio and compute uncache slice"

    # FIXME "10.09s bug": If you write a 10s-sliced audio to mp4 you get 10.09s in the mp4 file
    #   - To inspect, use `ffprobe` or `ffprobe -show_packets`
    #   - This messes up e.g. any spectro/slice/thumb that expects its input to be precisely ≤10s, else it wraps
    #   - All downstreams currently have to deal with this themselves, e.g. via plot_slice(slice_s=10)
    #   - Takeaways after further investigation:
    #       - It's just a fact of life that non-pcm mp4/mp3 encodings don't precisely preserve audio duration
    #       - We can always precisely slice the pcm samples once they're decoded from mp4/mp3, but as long as we're
    #         dealing in non-pcm encodings (for compression) we're stuck dealing with imprecise audio durations

    def to_sliced_id(id: str) -> Optional[str]:
        # Use first sliced id whose cache file exists (O(n) stat() calls)
        #   - HACK Find a principled way to synthesize id for sliced audio (multiple concerns here to untangle...)

        # Return first id whose cache/audio/ file exists
        resample = [
            'resample(%(sample_rate)s,%(channels)s,%(sample_width_bit)s)' % sg.load.audio_config,
        ]
        slice_enc = [
            'enc(wav)',
            'slice(%s,%s)' % (0, int(1000 * audio_s)),
            'spectro_denoise()',
            'enc(%(format)s,%(codec)s,%(bitrate)s)' % config.audio_persist.audio_kwargs,
        ]
        sliced_ids = [
            audio_id_add_ops(id, *resample, *slice_enc),
            audio_id_add_ops(id, *slice_enc),
        ]
        for sliced_id in sliced_ids:
            if (Path(data_dir) / sliced_id).exists():
                return sliced_id

        if drop_uncached_slice:
            # Drop and warn
            log.warn('Dropping id with no cached slice (maybe b/c invalid input audio)', id=id, sliced_ids=sliced_ids)
            return None
        else:
            # Give the caller a representative id (that we know doesn't exist) and let them deal with it
            return sliced_ids[0]

    # HACK Do O(n) stat() calls else "Falling back" incurs O(n) .audio read+slice if any audio.mp3 didn't need to .resample(...)
    #   - e.g. cache/audio/xc/data/RIRA/185212/audio.mp3.enc(wav)
    #   - Repro: xc_similar_html(sort='d_fc', sp_cols='species', xc_id=381417, n_total=5, n_sp=17, sample_r=1)
    to_paths_sliced = lambda recs: iter_progress(desc='to_paths_sliced', n=len(recs), **config.api.recs.progress_kwargs, xs=(
        (dataset, abs_sliced_path)
        for (dataset, abs_path) in xc_meta_to_paths(recs)
        for id in [str(Path(abs_path).relative_to(data_dir))]
        for sliced_id in [to_sliced_id(id)]
        if sliced_id is not None
        for abs_sliced_path in [str(Path(data_dir) / sliced_id)]
    ))

    try:
        # Try loading sliced .audio directly, bailing if any audio file doesn't exist
        return (recs
            .pipe(recs_featurize_metadata, to_paths=to_paths_sliced)
            .pipe(lambda df: df if skip_load_audio else (df
                .pipe(recs_featurize_audio, load=load_for_audio_persist())  # TODO TODO FIXME Slow load.read_audio
            ))
        )
    except FileNotFoundError as e:
        # Fallback to loading full .audio and computing the slice ourselves (which will cache for next time)
        #   - This is significantly slower (O(n)) than loading sliced .audio directly
        log.warn('Falling back to uncached audio slices', audio_s=audio_s, len_recs=len(recs), path_not_found=str(e))
        return (recs
            .pipe(recs_featurize_metadata)
            .pipe(recs_featurize_audio, load=sg.load)
            .pipe(recs_featurize_slice, audio_s=audio_s)
            .pipe(recs_audio_persist, progress_kwargs=config.api.recs.progress_kwargs)
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
        .reset_index()  # xc_id
    )


def recs_featurize_audio(
    recs: pd.DataFrame,
    load,  # Explicit load to help us stay aware of which one we're using at all times (lots of wav vs. mp4 confusion)
) -> pd.DataFrame:
    """Featurize: Add .audio"""
    return (recs
        .pipe(load.audio, **config.api.recs.progress_kwargs)  # procs barf on serdes error
    )


def recs_featurize_slice(recs: pd.DataFrame, audio_s: float) -> pd.DataFrame:
    """Featurize: Slice .audio (before .spectro/.feat/.thumb)"""
    return (recs
        .pipe(df_map_rows_progress, desc='slice_audio', **config.api.recs.progress_kwargs, f=lambda row: (
            sg.features.slice_audio(row, 0, audio_s)
        ))
    )


def recs_featurize_feat(recs: pd.DataFrame) -> pd.DataFrame:
    """Featurize: Add .feat"""
    return (recs
        .pipe(sg.projection.transform)
    )


def recs_featurize_slice_thumb(
    recs: pd.DataFrame,
    audio_s: float,
    thumb_s: float,
    scale: float,
    **plot_many_kwargs,
) -> pd.DataFrame:
    """Featurize: Add .thumb, .slice <- .spectro, .audio"""
    plot_many_kwargs = {
        **plot_many_kwargs,
        'scale': dict(h=int(40 * scale)),  # Best if h is multiple of 40 (because of low-level f=40 in Melspectro)
        'progress': dict(**config.api.recs.progress_kwargs),  # threads > sync, threads >> processes
        '_nocache': True,  # Dev: disable plot_many cache since it's blind to most of our sub-many code changes [TODO Revisit]
    }
    return (recs
        .pipe(recs_featurize_spectro)
        # Clip .audio/.spectro to audio_s/thumb_s
        .pipe(df_assign_first, **{
            **({} if not audio_s else dict(
                slice=df_cell_spectros(plot_slice.many, sg.features, **plot_many_kwargs,
                    pad_s=audio_s,  # Use pad_s instead of slice_s, else excessive writes (slice->mp4->slice->mp4)
                ),
            )),
            **({} if not thumb_s else dict(
                thumb=df_cell_spectros(plot_thumb.many, sg.features, **plot_many_kwargs,
                    thumb_s=thumb_s,
                ),
            )),
        })
    )


def recs_featurize_spectro(recs: pd.DataFrame) -> pd.DataFrame:
    """Featurize: Add .spectro"""
    return (recs
        # HACK Workaround some bug I haven't debugged yet
        #   - In server, .spectro column is present but all nan, which breaks downstream
        #   - In notebook, works fine
        #   - Workaround: force-drop .spectro column if present
        #   - Tech debt: Not general, very error prone -- e.g. does this affect .feat? .audio?
        .drop(columns=['spectro'], errors='ignore')
        .assign(spectro=lambda df: sg.features.spectro(df, **config.api.recs.progress_kwargs, cache=True))  # threads >> sync, procs
    )


def recs_view(
    recs: pd.DataFrame,
    view: bool = None,  # Disable the fancy stuff, e.g. in case you want to compute on the output data
    sp_cols: str = None,
) -> pd.DataFrame:

    # Params
    view = view if view is not None else True
    sp_cols = (sp_cols or 'com_name').split(',')

    # Utils
    round_sig_frac = lambda x, n: (  # Like round_sig but only on fractional digits
        round(x) if x >= 10**(n - 1) else
        round_sig(x, n)
    )
    df_if_col = lambda df, col, f: df if col not in df else f(df)
    df_col_map_if_col = lambda df, **cols: df_col_map(df, **{k: v for k, v in cols.items() if k in df})

    if not view:
        return recs

    return (recs
        .pipe(lambda df: df_col_map(df, **{
            # Scores (d_*)
            c: lambda x, c=c: '''<a href="{{ req_query_with(sort=%r) }}" >%s</a>''' % (c, round_sig_frac(x, 2))
            for c in df if c.startswith('d_')
        }))
        .pipe(df_if_col, 'xc_id', lambda df: (df
            .assign(
                # TODO Simplify: Have to do .xc before .xc_id, since we mutate .xc_id
                xc=lambda df: df_map_rows(df, lambda row: f'''
                    <a href="https://www.xeno-canto.org/%(xc_id)s">XC</a>
                ''' % row),
                xc_id=lambda df: df_map_rows(df, lambda row: '''
                    <a href="{{ req_href('/recs/xc/similar')(xc_id=%(xc_id)r) }}">%(xc_id)s</a>
                ''' % row),
            )
        ))
        .pipe(df_if_col, 'species', lambda df: (df
            .rename(columns={
                'species_com_name': 'com_name',
            })
            .assign(
                # TODO Simplify: Have to save .species/.com_name, since we mutate both
                _species=lambda df: df.species,
                _com_name=lambda df: df.com_name,
                species=lambda df: df_map_rows(df, lambda row: '''
                    <a href="{{ req_href('/recs/xc/species')(species=%(_species)r) }}" title="%(_com_name)s" >%(_species)s</a>
                ''' % row),
                com_name=lambda df: df_map_rows(df, lambda row: '''
                    <a href="{{ req_href('/recs/xc/species')(species=%(_species)r) }}" title="%(_species)s"  >%(_com_name)s</a>
                ''' % row),
            )
            # Keep sp_cols only
            .drop(columns=[c for c in {'species', 'com_name'} - set(sp_cols)])
        ))
        .pipe(df_col_map_if_col,
            background_species=lambda xs: ', '.join(xs),
        )
    )


def species_for_query(species_query: str) -> str:
    species_query = species_query and species_query.strip()
    species = metadata.species[species_query]
    if not species:
        raise ApiError(400, 'No species found', species_query=species_query)
    return species.shorthand


def require(x: bool, **data):
    """Shorthand to keep error handling concise"""
    if not x:
        caller = inspect.stack(context=0)[1]
        require_stmt = linecache.getline(caller.filename, caller.lineno).strip()
        raise ApiError(400, require_stmt, **data)


def df_require_nonempty_api(df: pd.DataFrame, msg: str, **data) -> pd.DataFrame:
    """Shorthand to keep error handling concise"""
    return df_require_nonempty(df, ApiError(400, msg, **data))
