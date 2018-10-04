from functools import lru_cache
import inspect
import linecache
import numbers
from typing import *

from attrdict import AttrDict
import bleach
from more_itertools import chunked, one
import pandas as pd
from potoo.pandas import *
from potoo.util import ensure_startswith
import sklearn
import structlog
from toolz import compose

from api.server_globals import sg, sg_load
from api.util import *
from cache import *
from config import config
from datasets import xc_meta_to_path, xc_meta_to_raw_recs, xc_raw_recs_to_recs
from load import Load
from logging_ import *
from payloads import *
from sp14.model import rec_neighbors_by, rec_preds, Search
from util import *
from viz import *

log = structlog.get_logger(__name__)

dist_info = {
    '2': sk.metrics.pairwise.distance_metrics()['l2'],
    '1': sk.metrics.pairwise.distance_metrics()['l1'],
    'c': sk.metrics.pairwise.distance_metrics()['cosine'],
}

defaults = AttrDict(

    # Shared
    n_recs       = 36,      # Fits within mobile screen (iphone 8, portrait, safari)
    audio_s      = 10,      # Hardcoded for precompute
    quality      = 'ab',
    scale        = 1,
    dists        = '2c',    # 2 (l2), 1 (l1), c (cosine)
    view         = None,
    sp_cols      = None,
    random_state = 0,

    # xc_species_html
    species      = None,
    cluster      = 'aw',    # agglom + ward
    cluster_k    = 6,       # TODO Record user's cluster_k per sp so we can learn cluster_k ~ sp
    sort         = 'c_pc',  # (Separate from 'rank' so that e.g. /similar? doesn't transfer to /species?)

    # xc_similar_html
    xc_id        = None,
    n_sp         = None,
    group_sp     = 'y',
    n_sp_recs    = 3,
    rank         = 'd_pc',  # (Separate from 'sort' so that e.g. /species? doesn't transfer to /similar?)

)


def xc_meta(
    species : str,
    quality : str = defaults.quality,
    n_recs  : int = defaults.n_recs,
) -> pd.DataFrame:
    with log_time_context():

        # Params
        species = species_for_query(species)
        quality = quality or 'ab'
        quality = [q.upper() for q in quality]
        quality = [{'N': 'no score'}.get(q, q) for q in quality]
        n_recs = np.clip(n_recs, 0, None)

        return (sg.xc_meta
            [lambda df: df.species == species]
            [lambda df: df.quality.isin(quality)]
            [:n_recs]
        )


def xc_species_html(
    species      : str   = defaults.species,
    quality      : str   = defaults.quality,
    cluster      : str   = defaults.cluster,
    cluster_k    : int   = defaults.cluster_k,
    n_recs       : int   = defaults.n_recs,
    audio_s      : float = defaults.audio_s,
    scale        : float = defaults.scale,
    dists        : str   = defaults.dists,
    sort         : str   = defaults.sort,
    view         : bool  = defaults.view,
    sp_cols      : str   = defaults.sp_cols,
    random_state : int   = defaults.random_state,
) -> pd.DataFrame:
    with log_time_context():

        # Return no results so user gets blank controls to fill in
        if not species:
            return pd.DataFrame()

        # Params
        species   = species_for_query(species)
        if not species: return pd.DataFrame([])
        quality   = quality and [q for q in quality for q in [q.upper()] for q in [{'N': 'no score'}.get(q, q)]]
        cluster   = cluster or None  # Validation in recs_cluster
        cluster_k = cluster_k or defaults.cluster_k
        cluster_k = np.clip(cluster_k, 1, 50)
        n_recs    = n_recs and np.clip(n_recs, 0, None)
        audio_s   = np.clip(audio_s, 0, 30)
        scale     = np.clip(scale, 0, 5)
        dists     = list(dists)

        # TODO How to support multiple precomputed search_recs so user can choose e.g. 10s vs. 5s?
        require(audio_s == config.api.recs.search_recs.params.audio_s)  # Can't change audio_s for precomputed search_recs
        del audio_s

        return (sg.search_recs

            # Filter
            .pipe(log_time_df, desc='filter', f=lambda df: (df
                [lambda df: df.species == species]
                .pipe(df_require_nonempty_for_api, 'No recs found', species=species)
                [lambda df: df.columns if not quality else df.quality.isin(quality)]
                .pipe(df_require_nonempty_for_api, 'No recs found', species=species, quality=quality)
                .reset_index(drop=True)  # Reset to RangeIndex after filter
                .pipe(df_remove_unused_categories)  # Drop unused cats after filter
            ))

            # Cluster
            #   - Before n_recs, so that clustering (data concern) is independent of sampling (view concern)
            .pipe(log_time_df, desc='cluster', f=lambda df: (df
                .pipe(recs_cluster, dists=dists, cluster=cluster, cluster_k=cluster_k, random_state=random_state)
            ))

            # TODO Sample or top n?
            #   - Sample works better for clusters
            #   - Top n maybe works better for 'date'? Or maybe not? Anything else it works particularly well for?
            # Top recs by `sort` (e.g. .date)
            #   - TODO Can't sort by xc_id since it's added by recs_featurize (-> recs_featurize_metdata_audio_slice), below
            .pipe(log_time_df, desc='sort', f=lambda df: (df
                # Sample
                .pipe(lambda df: df if n_recs is None else (df
                    .sample(min(n_recs, len(df)), random_state=random_state)
                ))
                # Sort
                .pipe(lambda df: df.sort_values(**one([
                    dict(by=sort, ascending=any([
                        sort in ['quality', 'month_day', 'time'],
                        sort.startswith('c_'),  # Cluster (c_*)
                    ]))
                    for sort in [sort if sort in df else 'date']
                ])))
                .reset_index(drop=True)  # Reset to RangeIndex after sort
            ))

            # View
            .pipe(log_time_df, desc='view:spectro_disp (slow ok)', f=lambda df: (df
                .pipe(recs_featurize_spectro_disp, scale=scale)  # .spectro_disp <- .spectro_bytes, .audio_bytes
            ))
            .pipe(log_time_df, desc='view', f=lambda df: (df
                .pipe(recs_view, view=view, sp_cols=sp_cols, links=['cluster', 'sort'])
                .pipe(recs_view_cols, sort_bys=[sort])
            ))

        )


def xc_similar_html(
    xc_id        : int   = defaults.xc_id,
    quality      : str   = defaults.quality,
    n_sp         : int   = defaults.n_sp,
    group_sp     : str   = defaults.group_sp,
    n_sp_recs    : int   = defaults.n_sp_recs,
    n_recs       : int   = defaults.n_recs,
    audio_s      : float = defaults.audio_s,
    scale        : float = defaults.scale,
    dists        : str   = defaults.dists,
    rank         : str   = defaults.rank,
    view         : bool  = defaults.view,
    sp_cols      : str   = defaults.sp_cols,
    random_state : int   = defaults.random_state,
) -> pd.DataFrame:
    with log_time_context():

        # Return no results so user gets blank controls to fill in
        if not xc_id:
            return pd.DataFrame()

        # Params
        quality   = quality  and [q for q in quality for q in [q.upper()] for q in [{'N': 'no score'}.get(q, q)]]
        n_sp      = n_sp     and np.clip(n_sp, 0, None)
        require(group_sp in ['y', 'n', '', None])
        group_sp  = group_sp == 'y'
        n_sp_recs = (n_sp_recs or None) and np.clip(n_sp_recs, 0, None)
        n_recs    = n_recs   and np.clip(n_recs, 0, 1000)
        n_recs    = n_recs - 1  # HACK Make room for query_rec in the results ("recs in view", not "search result recs")
        audio_s   = audio_s  and np.clip(audio_s, 0, 30)
        scale     = np.clip(scale, 0, 5)
        dists     = list(dists)

        # TODO How to support multiple precomputed search_recs so user can choose e.g. 10s vs. 5s?
        require(audio_s == config.api.recs.search_recs.params.audio_s)  # Can't change audio_s for precomputed search_recs
        del audio_s

        # Get precomputed search_recs
        search_recs = sg.search_recs

        # Lookup query_rec from search_recs
        query_rec = log_time_df(search_recs, desc='query_rec', f=lambda df: (df
            [lambda df: df.xc_id == xc_id]
            .pipe(df_require_nonempty_for_api, 'query_rec not found', xc_id=xc_id)
            .reset_index(drop=True)  # Reset to RangeIndex after filter
            .pipe(lambda df: one(df_rows(df)))
        ))

        # Predict query_sp_p from search model
        query_sp_p = log_time(desc='rec_preds', f=lambda: (
            # We could reuse query_rec.f_preds, but:
            #   - The code complexity to map f_preds (np.array(probs)) to rec_preds (df[species, prob]) is nontrivial (>1h refactor)
            #   - Always using rec_preds ensures that behavior is consistent between user rec and xc rec
            #   - Skipping rec_preds for xc recs would only save ~150ms
            rec_preds(query_rec, sg.search)
            [:n_sp]
            .rename(columns={'p': 'sp_p'})
        ))

        # Filter search_recs for query_sp_p
        search_recs = log_time_df(search_recs, desc='search_recs:filter', f=lambda df: (df
            # Filter
            [lambda df: df.species.isin(query_sp_p.species)]
            .pipe(df_require_nonempty_for_api, 'No recs found', species=query_sp_p.species)
            [lambda df: df.columns if not quality else df.quality.isin(quality)]
            .pipe(df_require_nonempty_for_api, 'No recs found', species=query_sp_p.species, quality=quality)
            .reset_index(drop=True)  # Reset to RangeIndex after filter
            .pipe(df_remove_unused_categories)  # Drop unused cats after filter
        ))

        # Exclude query_rec from results (in case it's an xc rec)
        search_recs = log_time_df(search_recs, desc='search_recs:exclude_query_rec', f=lambda df: (df
            [lambda df: df.xc_id != query_rec.xc_id]
            .reset_index(drop=True)  # Reset to RangeIndex after filter
        ))

        # Compute dists for query_rec, so user can interactively compare and evaluate them
        #   - O(n)
        #   - (Preds dist last investigated in notebooks/app_ideas_6_with_pca)
        dist_recs = log_time_df(search_recs, desc='dist_recs', f=lambda df: (df
            .pipe(lambda df: df.assign(**{  # (.pipe to avoid error-prone lambda scoping inside dict comp)
                d_(f, d): (
                    d_compute(
                        list(df[f_col]),  # series->list, else errors -- but don't v.tolist(), else List[list] i/o List[array]
                        [query_rec[f_col]],
                    )
                    .round(6)  # Else near-zero but not-zero stuff is noisy (e.g. 6.5e-08)
                )
                for f_col, f, _f_compute in sg.feat_info
                for d, d_compute in dist_info.items()
                if d in dists
            }))
        ))

        # Rank results
        #   - O(n log k)
        #   - [later] Add ebird_priors prob
        sort_bys = ['slp', rank] if group_sp else rank
        slp = lambda sp_p: np.abs(-np.log(sp_p))  # slp: "species log prob" (abs for 1->0 i/o -0)
        ranked_recs = (dist_recs
            .pipe(log_time_df, desc='ranked_recs:merge', f=lambda df: (df
                # Join in .sp_p for scoring functions
                #   - [Using sort=True to silence "non-concatenation axis" warning -- not sure what we want, or if it matters]
                .merge(how='left', on='species', right=query_sp_p[['species', 'sp_p']],
                    sort=True,  # [Silence "non-concatenation axis" warning -- not sure what we want, or if it matters...]
                )
            ))
            .pipe(log_time_df, desc='ranked_recs:scores', f=lambda df: (df
                # Scores (d_*)
                #   - A distance measure in [0,inf), lower is better
                #   - Examples: -log(p), feat dist (d_f), preds dist (d_p)
                #   - Can be meaningfully combined by addition, e.g.
                #       - -log(p) + -log(q) = -log(pq)
                #       - -log(p) + d_f     = ... [Meaningful? Helpful to rescale?]
                #       - -log(p) + d_p     = ... [Meaningful? Helpful to rescale?]
                #       - d_f     + d_p     = ... [Meaningful? Helpful to rescale?]
                .assign(
                    slp=lambda df: slp(df.sp_p),
                )
            ))
            .pipe(log_time_df, desc='ranked_recs:top_recs_per_sp', f=lambda df: (df
                # Top recs per sp
                .pipe(lambda df: df if n_sp_recs is None else (df
                    .set_index('xc_id')
                    .ix[lambda df: (df
                        .groupby('species')[rank].nsmallest(n_sp_recs)
                        .index.get_level_values('xc_id')
                    )]
                    .reset_index()  # .xc_id
                ))
            ))
            .pipe(log_time_df, desc='ranked_recs:top_recs_overall', f=lambda df: (df
                # Top recs overall
                .sort_values(sort_bys)[:n_recs]
                .reset_index(drop=True)  # Reset to RangeIndex after sort
            ))
        )

        # Featurize + view ranked_recs
        return (ranked_recs

            # Add .label per species
            #   - Before including query_rec, so that it doesn't have one
            .pipe(log_time_df, desc='label_species', f=lambda df: (df
                .merge(how='left', on='species', right=(df
                    [['species']].drop_duplicates().reset_index(drop=True).pipe(df_set_index_name, 'label').reset_index()
                ))
            ))

            # Include query_rec in view
            #   - HACK Include it as the first result with some mocked-out columns
            .pipe(log_time_df, desc='add_query_rec', f=lambda df: (df
                .pipe(lambda df: pd.concat(
                    sort=True,  # [Silence "non-concatenation axis" warning -- not sure what we want, or if it matters...]
                    objs=[
                        (
                            # Expand query_rec cols to match df cols
                            pd.concat([DF([query_rec]), df[:0]])
                            # Add query_rec.slp for user feedback (mostly for model debugging)
                            .assign(
                                sp_p=lambda df: one(query_sp_p[query_sp_p.species == query_rec.species].sp_p.values),
                                slp=lambda df: slp(df.sp_p),
                            )
                            # Mock d_* cols as 0
                            .pipe(lambda df: df.fillna({k: 0 for k in df if k.startswith('d_')}))
                        ),
                        df,
                    ],
                ))
                .reset_index(drop=True)  # Reset to RangeIndex after concat
            ))

            # View
            .pipe(log_time_df, desc='view:spectro_disp (slow ok)', f=lambda df: (df
                .pipe(recs_featurize_spectro_disp, scale=scale)  # .spectro_disp <- .spectro_bytes, .audio_bytes
            ))
            .pipe(log_time_df, desc='view', f=lambda df: (df
                .pipe(recs_view, view=view, sp_cols=sp_cols, links=['rank'])
                .pipe(lambda df: (df
                    # Manually order d_* cols [Couldn't get to work above]
                    .pipe(df_reorder_cols, first=[
                        'slp',
                        *[c for c in [d_(f, m) for m in '2c' for f in 'fp'] if c in df],
                    ])
                ))
                .pipe(recs_view_cols, sort_bys=sort_bys)
            ))

        )


# WARNING Unsafe to @cache since it contains methods and stuff
#   - And if you do @cache, joblib.memory will silently cache miss every time, leaving behind partial .pkl writes :/
@lru_cache()
def get_feat_info() -> List[Tuple]:
    """For sg.feat_info"""
    return [
        ('feat',    'f', None),
        ('f_preds', 'p', partial(sg.search.species_proba,
            _cache=True,  # Must explicitly request caching (enable caching upstream after we ensure no perf regression in model eval)
        )),
        # ('f_likely', 'l', ...),  # TODO ebird_priors
    ]


def d_(f: str, d: str) -> str:
    return f'd_{f}{d}'


# TODO Should we migrate feat to f_feat for consistency, or is it too much trouble?
#   - Nontrivial: would need to clearly and simply delineate .feat code vs. .f_feat code
#   - Roughly boilds down to model vs. api, but we'd need a reliable rename layer when api calls into model
def feat_cols(cols: Union[Iterable[str], pd.DataFrame]) -> Iterable[str]:
    if isinstance(cols, pd.DataFrame):
        cols = list(cols.columns)
    return [c for c in cols if (
        c == 'feat' or
        c.startswith('f_')  # e.g. f_preds, f_likely
    )]


@lru_cache()
def get_search_recs(
    refresh=False,
    cache_type='hybrid',  # None | 'hybrid' | 'parquet' | 'sqlite'
) -> pd.DataFrame:
    """For sg.search_recs"""
    log.info()

    # Compute key
    key_show = ','.join(
        '%s[%s]' % (k, v)
        for expr in config.api.recs.search_recs.cache.key.show
        for (k, v) in [(expr.split('.')[-1], eval(expr))]
        for (k, v) in ([(k, v)] if not isinstance(v, dict) else v.items())
    )
    key = sha1hex(json_dumps_canonical({expr: eval(expr) for expr in [
        *config.api.recs.search_recs.cache.key.opaque,  # Stuff that's too big/complex to stuff into the human-visible filename
        *config.api.recs.search_recs.cache.key.show,  # Include in case we want just the sha key to be usable elsewhere
    ]}))[:7]

    # Args for df_cache_*
    compute = _compute_search_recs
    path = f"payloads/search_recs-{key_show}-{key}"
    name = 'search_recs'

    # Delegate to parquet/sqlite
    if not cache_type:
        return compute()
    elif cache_type == 'hybrid':
        return df_cache_hybrid(compute=compute, path=path, refresh=refresh,
            desc=name,
        )
    elif cache_type == 'parquet':
        return df_cache_parquet(compute=compute, path=path, refresh=refresh,
            desc=name,
        )
    elif cache_type == 'sqlite':
        return df_cache_sqlite(compute=compute, path=path, refresh=refresh,
            table=name,
            # If any of the big array cols are list i/o np.array then you'll see ~10x slowdown in serdes
            #   - [And this shouldn't be specific to sqlite serdes...]
            #   - Rely on logging (size+times) to catch this
            col_conversions=lambda df: {} if df.empty else {
                k: fg
                for k, v in df.iloc[0].items()
                for fg in [(
                    (np_save_to_bytes, np_load_from_bytes) if isinstance(v, np.ndarray) else    # np.array <-> npy (bytes)
                    (json_dumps_canonical, json.loads)     if isinstance(v, (list, dict)) else  # list/dict <-> json (str)
                    None
                )]
                if fg
            },
            # Don't bother specifying a schema
            #   - sqlite "type affinity" is pretty fluid with types: https://sqlite.org/datatype3.html#type_affinity
            #   - feat cols end up as TEXT instead of BLOB, but TEXT accepts and returns BLOB data (python bytes) as is
            #   - No other cols meaningfully impact size, so the remaining concerns are data fidelity and client compat
            # dtype={'feat': sqla.BLOB, ...},
        )
    else:
        raise ValueError(f"Unknown cache_type[{cache_type}]")


def _compute_search_recs() -> pd.DataFrame:
    log.info(**{'len(sg.xc_meta)': len(sg.xc_meta), **sg_load.config.xc_meta})
    return (sg.xc_meta
        # Limit (for faster dev)
        [:config.api.recs.search_recs.params.get('limit')]
        # Featurize (audio meta + .feat)
        .pipe(recs_featurize_pre_rank)
    )


def recs_featurize_pre_rank(
    recs: pd.DataFrame,
    load_sliced: 'LoadLike' = load_for_audio_persist,
) -> pd.DataFrame:
    if callable(load_sliced): load_sliced = load_sliced()
    # TODO How to support multiple precomputed search_recs so user can choose e.g. 10s vs. 5s?
    audio_s = config.api.recs.search_recs.params.audio_s
    # Batch for mem safety
    #   - TODO TODO Fix progress logging here so that it's readable (write a _map_progress_log_each?)
    return pd.concat(objs=map_progress(
        desc='recs_featurize_pre_rank:batches', **config.sync_progress_kwargs,
        xs=chunked(recs.index,
            # 1000,  # Killed on remote n1-highcpu-16 during 27/36
            # 250,  # Mem safe on remote n1-highcpu-16
            len(recs),  # TODO TODO Still debugging chunking (seeing an assertion failure...)
        ),
        f=lambda ix: (recs
            .loc[ix]
            # Audio metadata, without .audio
            .pipe(recs_featurize_metdata_audio_slice,
                audio_s=config.api.recs.search_recs.params.audio_s,
                load_sliced=load_sliced,
                load_full=None,
                # HACK Drop uncached audios to avoid big slow O(n) "Falling back"
                #   - Good: this correctly drops audios whose input file is invalid, and thus doesn't produce a sliced cache/audio/ file
                #   - Bad: this incorrectly drops any valid audios that haven't been _manually_ cached warmed
                #   - TODO How to better propagate invalid audios (e.g. empty cache file) so we can handle this more robustly
                drop_uncached_slice=True,
                # Don't load .audio for pre-rank recs (only for final n_recs recs, below)
                no_audio=True,
            )
            .pipe(recs_featurize_recs_for_sp)
            .pipe(recs_featurize_feat)
            .pipe(recs_featurize_f_)
            .pipe(recs_featurize_spectro_bytes, load_audio=load_sliced, pad_s=audio_s,
                scale=1,  # Fix scale=1 for precompute, deferring scale=N to view logic [currently in .html.j2 as inline style]
            )
            .pipe(recs_featurize_audio_bytes)
        ),
    ))


# TODO Simplify: replace with sg.search_recs lookup (callers: +xc_species_html -xc_similar_html)
def recs_featurize(
    recs: pd.DataFrame,
    audio_s: float,
    scale: float,
    load_sliced: 'LoadLike' = load_for_audio_persist,
    load_full:   'LoadLike' = lambda: sg.load,
    **plot_many_kwargs,
) -> pd.DataFrame:
    if callable(load_sliced): load_sliced = load_sliced()
    if callable(load_full):   load_full   = load_full()
    return (recs
        .pipe(recs_featurize_metdata_audio_slice,
            audio_s=audio_s,
            load_sliced=load_sliced,
            load_full=load_full,
        )
        .pipe(recs_featurize_recs_for_sp)
        .pipe(recs_featurize_feat)
        .pipe(recs_featurize_f_)
        .pipe(recs_featurize_spectro_bytes, pad_s=audio_s, scale=scale, **plot_many_kwargs,
            load_audio=None,  # Use the .audio we just loaded (above)
        )
        .pipe(recs_featurize_audio_bytes)
        .pipe(recs_featurize_spectro_disp)  # .spectro_disp <- .spectro_bytes, .audio_bytes
    )


def recs_featurize_metdata_audio_slice(
    recs: pd.DataFrame,
    audio_s: float,
    load_sliced: Load,
    load_full: Optional[Load],
    # HACK Drop audios with no cache/audio/ slice file instead of recomputing ("Falling back") (which warms cache)
    #   - Invalid input audios don't produce a cache/audio/ file, so if you get one then you're stuck always falling back
    drop_uncached_slice: bool = None,
    # Skip loading .audio (e.g. for intermediate stages of xc_similar_html)
    no_audio: bool = None,
) -> pd.DataFrame:
    """Featurize: Add .audio with slice"""

    # Params
    assert audio_s is not None and audio_s > 0, f"{audio_s}"
    drop_uncached_slice = False if drop_uncached_slice is None else drop_uncached_slice
    no_audio = False if no_audio is None else no_audio
    assert not (not drop_uncached_slice and no_audio), "Can't skip audio and compute uncached slices"

    # FIXME "10.09s bug": If you write a 10s-sliced audio to mp4 you get 10.09s in the mp4 file
    #   - To inspect, use `ffprobe` or `ffprobe -show_packets`
    #   - This messes up e.g. any spectro plot that expects its input to be precisely ≤10s, else it wraps
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
            'resample(%(sample_rate)s,%(channels)s,%(sample_width_bit)s)' % load_sliced.audio_config,
        ]
        slice_enc = [
            'enc(wav)',
            'slice(%s,%s)' % (0, int(1000 * audio_s)),
            'spectro_denoise()',
            'enc(%(format)s,%(codec)s,%(bitrate)s)' % config.audio.audio_persist.audio_kwargs,
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
    #   - Repro: xc_similar_html(sort='d_fc', sp_cols='species', xc_id=381417, n_recs=5, n_sp=17)
    @cache(version=0, key=lambda recs: recs.id)  # Slow: ~13s for 35k NA-CA recs
    def to_paths_sliced(recs) -> Iterable[Tuple[str, str]]:
        return [
            dataset_path
            for dataset_path in map_progress_df_rows(recs, desc='to_paths_sliced', **config.api.recs.progress_kwargs,
                f=lambda rec: one(
                    sliced_id and (dataset, str(Path(data_dir) / sliced_id))
                    for (dataset, abs_path) in [xc_meta_to_path(rec)]
                    for id in [str(Path(abs_path).relative_to(data_dir))]
                    for sliced_id in [to_sliced_id(id)]
                ),
            )
            if dataset_path  # Filter out None's from dropped sliced_id's
        ]

    try:
        # Try loading sliced .audio directly, bailing if any audio file doesn't exist
        return (recs
            .pipe(recs_featurize_metadata, to_paths=to_paths_sliced)
            .pipe(lambda df: df if no_audio else (df
                .pipe(recs_featurize_audio, load=load_sliced)
            ))
        )
    except FileNotFoundError as e:
        # Fallback to loading full .audio and computing the slice ourselves (which will cache for next time)
        #   - This is significantly slower (O(n)) than loading sliced .audio directly
        log.warn('Falling back to uncached audio slices', audio_s=audio_s, len_recs=len(recs), path_not_found=str(e))
        return (recs
            .pipe(recs_featurize_metadata)
            .pipe(recs_featurize_audio, load=load_full)
            .pipe(recs_featurize_slice_audio, audio_s=audio_s)
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
        # HACK Drop xc_meta_to_raw_recs *_stack cols: for notebook not api, and the df_cell wrappers clog up serdes (e.g. sqlite)
        [lambda df: [c for c in df if not c.endswith('_stack')]]
    )


def recs_featurize_audio(
    recs: pd.DataFrame,
    load: Load,  # Explicit load to help us stay aware of which one we're using at all times (lots of wav vs. mp4 confusion)
) -> pd.DataFrame:
    """Featurize: Add .audio"""
    return (recs
        .pipe(load.audio, **config.api.recs.progress_kwargs)  # procs barf on serdes error
    )


def recs_featurize_slice_audio(recs: pd.DataFrame, audio_s: float) -> pd.DataFrame:
    """Featurize: .audio <- sliced .audio (before .spectro/.feat)"""
    return (recs
        .pipe(df_map_rows_progress, desc='slice_audio', **config.api.recs.progress_kwargs, f=lambda row: (
            sg.features.slice_audio(row, 0, audio_s)
        ))
    )


def recs_featurize_feat(recs: pd.DataFrame) -> pd.DataFrame:
    """Featurize: Add .feat"""
    return (recs
        .pipe(sg.projection.transform)
        .pipe(df_col_map, feat=lambda x: x.astype(np.float32))  # TODO Push float32 up into Projection._feat (slow cache bust)
    )


def recs_featurize_f_(recs: pd.DataFrame) -> pd.DataFrame:
    """Featurize: Add .f_*"""
    return (recs
        .pipe(lambda df: df.assign(**{  # (.pipe to avoid error-prone lambda scoping inside dict comp)
            f_col: list(v)  # series->list, else errors -- but don't v.tolist(), else List[list] i/o List[array]
            for f_col, _f, f_compute in sg.feat_info
            if f_compute  # Skip e.g. .feat, which we already computed above
            for v in [one_progress(desc=f_col, n=len(df), f=lambda: f_compute(df))]
        }))
    )


def recs_featurize_spectro_bytes(
    recs: pd.DataFrame,
    pad_s: float,
    scale: float,
    load_audio: Optional[Load] = None,  # Use .audio if load_audio=None, else load .audio ourselves (using load_audio)
    format='png',
    **plot_kwargs,
) -> pd.DataFrame:
    """Featurize: Add .spectro_bytes, spectro_bytes_mimetype <- .audio"""
    with log_time_context():
        plot_kwargs = {
            **plot_kwargs,
            'scale': dict(h=int(40 * scale)),  # Best if h is multiple of 40 (because of low-level f=40 in Melspectro)
        }
        return (recs
            .pipe(df_assign_first,
                spectro_bytes_mimetype=format_to_mimetype(format),
                spectro_bytes=lambda df: map_progress_df_rows(df, desc='rec_spectro_bytes',
                    f=partial(rec_spectro_bytes, pad_s=pad_s, load_audio=load_audio, format=format, **plot_kwargs),
                    # FIXME Saw hangs with dask threads
                    #   - Repro: use='dask', scheduler='threads', 1k recs, cache hit and/or miss
                    #   - No repro: use='dask', scheduler='synchronous'
                    #   - HACK Going with use='sync' to work around...
                    # **config.api.recs.progress_kwargs,  # threads >> sync, procs
                    use='sync',
                ),
            )
        )


@cache(version=0, tags='rec', key=lambda rec, **kwargs: (rec.id, kwargs, sg.features))
def rec_spectro_bytes(
    rec: Row,
    pad_s: float,
    load_audio: Optional[Load] = None,  # Use .audio if load_audio=None, else load .audio ourselves (using load_audio)
    format='png',
    **plot_kwargs,
) -> bytes:
    rec = rec.copy()  # Copy so we can mutate
    rec.spectro = (sg.features
        # FIXME Slow (load_audio) vs. incorrect (load)
        #   - features.load=load_audio is correct, but very slow because it doesn't share .spectro cache with features.load=load
        #   - features.spectro cache key includes load.format='wav'|'mp4', which makes load_audio vs. load bust cache
        #   - TODO Should load.format be part of features.spectro cache key? Seems like no, but think harder...
        # .replace(**dict(load=load_audio) if load_audio else {})  # Set features.load for features._cache() -> load._audio()
        ._spectro(rec, cache=True)
    )
    spectro_img = plot_spectro(rec, sg.features, pad_s=pad_s, **plot_kwargs,
        show=False,   # Return img instead of plotting
        audio=False,  # Return PIL.Image (no audio) instead of Displayable (with embedded html audio)
    )
    # TODO Optimize png/img size -- currently bigger than mp4 audio! (a rough start: notebooks/png_compress)
    spectro_bytes = pil_img_save_to_bytes(
        spectro_img.convert('RGB'),  # Drop alpha channel
        format=format,
    )
    return spectro_bytes



def recs_featurize_audio_bytes(recs: pd.DataFrame) -> pd.DataFrame:
    """
    Featurize: Add .audio_bytes, .audio_bytes_mimetype <- .id
    - Directly read file for rec.id (audio id), failing if it doesn't exist (no transcoding/caching)
    - Doesn't use (or need) .audio
    """
    with log_time_context():
        return (recs
            .assign(
                audio_bytes_mimetype = lambda df: df.id.map(audio_id_to_mimetype),
                audio_bytes          = lambda df: map_progress(audio_id_to_bytes, df.id, desc='audio_bytes'),
            )
        )


def recs_featurize_spectro_disp(
    recs: pd.DataFrame,
    scale: float = None,
) -> pd.DataFrame:
    """Featurize: Add .spectro_disp <- .spectro_bytes, .audio_bytes"""
    return (recs
        # .spectro_disp
        .pipe(df_assign_first,
            spectro_disp=lambda df: df_map_rows(df, lambda rec: (
                df_cell_display(display_with_audio_bytes(
                    display_with_style(
                        pil_img_open_from_bytes(rec.spectro_bytes),  # (Infers image format from bytes)
                        # TODO Make scale a recs_view concern
                        #   - TODO Restore multiples of 40px, to avoid pixel blurring [defer until mobile app]
                        style_css=scale and '.bubo-audio-container img { height: %s; }' % (
                            # '%spx' % int(40 * scale),  # XXX Switched px->rem for responsive style, but this was multiples of 40px
                            '%.3frem' % (1.5 * scale),  # Good on mobile, good enough on desktop (but not multiples of 40px)
                        ),
                    ),
                    audio_bytes=rec.audio_bytes,
                    mimetype=rec.audio_bytes_mimetype,
                ))
            )),
        )
    )


def recs_featurize_recs_for_sp(recs: pd.DataFrame) -> pd.DataFrame:
    """Featurize: Add .recs_for_sp, the total num recs (any quality) for each rec's .species"""
    return (recs
        .merge(how='left', on='species', right=_recs_for_sp())
    )


@lru_cache()  # (Not actually a bottleneck yet: ~0.1s for the 35k CA recs)
def _recs_for_sp() -> pd.DataFrame:
    return (sg.xc_meta
        .assign(recs_for_sp=1).groupby('species').recs_for_sp.sum()
        .reset_index()  # groupby key
        [['species', 'recs_for_sp']]
    )


def recs_cluster(
    recs: pd.DataFrame,
    dists: List[str],
    cluster: Optional[str],  # None -> skip clustering
    cluster_k: int,  # TODO Generalize user controls for more general kwargs
    random_state: int,
) -> pd.DataFrame:

    if not cluster:
        return recs

    def cluster_info_for(cluster: str, d_compute: callable) -> sk.base.ClusterMixin:
        (cluster, flags) = (cluster[0], list(cluster[1:]))

        agglom_linkages = {
            'w': 'ward',
            'c': 'complete',
            'a': 'average',
            # 's': 'single',  # TODO Requires sklearn 0.20 [blocked on QA'ing model eval -- see env.yml]
        }
        k = flags[0] if flags else 'w'
        require(k in agglom_linkages)
        agglom_linkage = agglom_linkages[k]

        cluster_info = {

            # kmeans
            #   - http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
            #   - http://scikit-learn.org/stable/modules/clustering.html#k-means
            #   - TODO Is it problematic that we don't d_compute here?
            'k': sk.cluster.KMeans(
                n_clusters=min(cluster_k, len(recs)),
                # verbose=1,  # Noisy
                # n_jobs=-1,  # Slow for small inputs (proc overhead)
                random_state=random_state,
            ),

            # agglom (hierarchical)
            #   - http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
            #   - http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
            'a': sk.cluster.AgglomerativeClustering(
                n_clusters=min(cluster_k, len(recs)),
                # memory=None,
                # compute_full_tree='auto',
                linkage=agglom_linkage,
                affinity=d_compute if agglom_linkage != 'ward' else 'euclidean',  # ward requires euclidean
            )

            # dbscan
            #   - http://scikit-learn.org/stable/modules/clustering.html#dbscan
            #   - http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
            #   - TODO Compute cluster centroid as mean of its core samples
            #   - TODO Figure out usable eps (finicky)
            # 'd': sk.cluster.DBSCAN(
            #     metric=d_compute,
            #     eps=100,  # (Default: .5)
            #     # min_samples=5,
            #     # n_jobs=-1,  # Slow for small inputs (proc overhead)
            # ),

        }
        require(cluster in cluster_info.keys())
        return cluster_info[cluster]

    return (recs
        .pipe(lambda df: df_assign_first(df, **{  # (.pipe to avoid error-prone lambda scoping inside dict comp)
            k: cluster_label_and_dist(df, cluster_info_for(cluster, d_compute), f_col, d_compute)
            for f_col, f, _f_compute in sg.feat_info
            for d, d_compute in dist_info.items()
            if d in dists
            for k in [f'c_{f}{d}']
        }))
    )


def cluster_label_and_dist(
    recs: pd.DataFrame,
    cluster,
    f_col,
    d_compute,
    show=False,
) -> "Col[Tuple['label', 'dist_to_center']]":

    if len(recs) <= 1:
        # Avoid e.g. agglom failing when len(X) == 1
        cluster.labels_ = [0] * len(recs)
    else:
        X = np.array(list(recs[f_col]))  # Series[np.ndarray[p]] -> np.ndarray[n,p]
        cluster.fit(X)

    # Extract inferred attrs (somewhat) generically across various clustering estimators
    #   - TODO Compute centroids manually for estimators that don't return it (kmeans returns it, agglom/dbscan don't)
    labels = cluster.labels_
    centroids = None if not hasattr(cluster, 'cluster_centers_') else cluster.cluster_centers_

    if show:
        ipy_print({
            k: eval(f'cluster.{k}')
            for k in [
                'n_iter_', 'inertia_', 'cluster_centers_.shape', 'labels_',  # kmeans
                'core_sample_indices_', 'components_.shape', 'labels_',      # dbscan
                'labels_', 'n_leaves_', 'n_components_', 'children_',        # agglom
            ]
            if hasattr(cluster, k.split('.')[0])
        })

    # TODO Is it coherent to d_compute here, e.g. for kmeans which doesn't use d_compute in fit?
    return [
        (label, dist)
        for f_x, label in zip(recs[f_col], labels)
        for dist in [
            None if not centroids else
            np.asscalar(d_compute([f_x], [centroids[label]]))
        ]
    ]


def recs_view(
    recs: pd.DataFrame,
    view: bool = None,  # Disable the fancy stuff, e.g. in case you want to compute on the output data
    sp_cols: str = None,
    links: List[str] = [],
) -> pd.DataFrame:

    # Params
    view = view if view is not None else True
    sp_cols = (sp_cols or 'com_name').split(',')

    # Utils
    round_sig_frac = lambda x, n: (  # Like round_sig but only on fractional digits
        round(x) if x >= 10**(n - 1) else
        round_sig(x, n)
    )
    simple_num = lambda x: (
        '0' if np.isclose(x, 0, atol=1e-3) else
        str(x).lstrip('0')
    )
    show_num = lambda x, n=2: simple_num(round_sig_frac(x, n))
    df_if_cols = lambda df, cols, f: f(df) if set([cols] if isinstance(cols, str) else cols) <= set(df.columns) else df
    df_col_map_if_col = lambda df, **cols: df_col_map(df, **{k: v for k, v in cols.items() if k in df})

    if not view:
        return recs

    return (recs

        # Strip html in freeform xc inputs (e.g. remarks)
        #   - Remarks is the only field where markdown is encouraged, but don't assume it's the only place it will show up
        #   - https://www.xeno-canto.org/upload
        .pipe(df_col_map_if_col,
            time               = lambda x: bleach.clean(x, strip=True),
            type               = lambda x: bleach.clean(x, strip=True),
            subspecies         = lambda x: bleach.clean(x, strip=True),
            background_species = lambda xs: [bleach.clean(x, strip=True) for x in xs],
            recordist          = lambda x: bleach.clean(x, strip=True),
            elevation          = lambda x: bleach.clean(x, strip=True),
            place              = lambda x: bleach.clean(x, strip=True),
            remarks            = lambda x: bleach.clean(x, strip=True),
        )

        # xc, xc_id
        .pipe(df_if_cols, 'xc_id', lambda df: (df
            .assign(
                # Save raw cols before we junk them up with html
                _xc_id=lambda df: df.xc_id,
                xc=lambda df: df_map_rows(df, lambda row: f'''
                    <a href="https://www.xeno-canto.org/%(_xc_id)s">XC</a>
                ''' % row),
                xc_id=lambda df: df_map_rows(df, lambda row: '''
                    <a href="{{ req_href('/recs/xc/similar')(xc_id=%(_xc_id)r) }}">%(_xc_id)s</a>
                ''' % row),
            )
        ))

        # Scores (d_*)
        .pipe(lambda df: df_col_map(df, **{
            c: lambda x, c=c: '''<a href="{{ req_query_with(rank=%r) }}">%s</a>''' % (c, show_num(x, 2))
            for c in df
            if c.startswith('d_') and 'rank' in links
        }))
        # Clusters (c_*)
        .pipe(lambda df: df_col_map(df, **{
            c: lambda xs, c=c: '''
                <a href="{{ req_query_with(sort=%(c)r) }}"><span class="label-i i-%(cluster)s">%(cluster)s</span>%(maybe_dist)s</a>
            ''' % dict(
                c=c,
                cluster=xs[0],
                maybe_dist='' if xs[1] is None else ' %s' % show_num(xs[1], 2),
            )
            for c in df
            if c.startswith('c_') and 'cluster' in links
        }))

        # species, com_name
        .pipe(df_if_cols, 'species', lambda df: (df
            .assign(
                com_name=lambda df: df.species_com_name,  # [Push this rename upstream, somewhere?]
            )
            .assign(
                # Save raw cols before we junk them up with html
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
            # Label sp_cols
            .pipe(df_if_cols, 'label', lambda df: (df
                .assign(**{
                    c: df_map_rows(df, lambda row: '''
                        <div class="labeled-species">
                            %(c)s
                            <span class="label-i i-%(label).0f">%(label_text)s</span>
                        </div>
                    ''' % dict(
                        label=row.label,
                        label_text='&nbsp;' if pd.notnull(row.label) else '?',
                        c=row[c],
                    ))
                    for c in sp_cols
                })
            ))
        ))

        # sp_recs
        .assign(
            sp_recs=lambda df: df.recs_for_sp,  # TODO Push this upstream into recs_featurize_pre_rank (+ bump cache version)
        )

        # Sort-by cols: quality, date, month_day, time
        #   - WARNING Don't rename these (e.g. season=month_day) else names don't connect all the way through recs->view->sort->recs
        .pipe(df_if_cols, 'date', lambda df: df.assign(
            date=lambda df: df_map_rows(df, lambda row: '%(year).0f-%(month_day)s' % row),  # .year is float (b/c None)
        ))
        .pipe(df_col_map_if_col, **{
            c: lambda x, c=c: '''<a href="{{ req_query_with(sort=%r) }}">%s</a>''' % (c, x)
            for c in ['quality', 'date', 'month_day', 'time']
            if 'sort' in links
        })

        # background_species
        .pipe(df_col_map_if_col,
            background_species=lambda xs: ' '.join(xs),
        )

        # license
        .assign(
            license=lambda df: df.license_type,
        )

        # lat_lng
        .pipe(df_if_cols, ['lat', 'lng'], lambda df: df.assign(
            lat_lng=lambda df: df_map_rows(df, lambda row: '''
                <a href="https://www.google.com/maps/place/%(lat)s,%(lng)s/@%(lat)s,%(lng)s,6z">%(lat)s, %(lng)s</a>
            ''' % row),
        ))

        # place
        #   - Reverse place parts so that high-order info is first (and visually aligned)
        .pipe(df_col_map_if_col,
            place=lambda x: ', '.join(reversed(x.split(', '))),
        )

        # Fill any remaining nulls with ''
        #   - Strip cats else they'll reject '' (unless it's a valid cat)
        .pipe(df_cat_to_str)
        .fillna('')

        # df_cell_str all the strs to prevent df.to_html from truncating long strs
        #   - HACK df_cell_str everything to fix errant "..."s
        .applymap(lambda x: (
            df_cell_str(show_num(x)) if isinstance(x, (numbers.Number, np.generic)) else
            df_cell_str(x)           if isinstance(x, str) else
            x
        ))

    )


def recs_view_cols(
    df: pd.DataFrame,
    sort_bys: str = None,
    prepend: Iterable[str] = [],
    append: Iterable[str] = [],
) -> pd.DataFrame:
    return (df
        [lambda df: [c for c in [
            *prepend,
            *[c for c in [
                'xc', 'xc_id',
                'slp',
                *unique_everseen(c for c in df if c.startswith('d_')),  # Scores (d_*)
                *unique_everseen(c for c in df if c.startswith('c_')),  # Clusters (c_*)
                'com_name', 'species',
            ] if c not in sort_bys],
            *(sort_bys if sort_bys else []),
            *[c for c in [
                'spectro_disp', 'sp_recs',
                # 'duration_s',  # TODO Surface the original duration (.duration_s is the sliced duration)
                'quality', 'date', 'month_day', 'time',
                'type', 'subspecies', 'background_species',
                'license', 'recordist', 'bird_seen', 'playback_used',
                'elevation', 'lat_lng', 'place', 'remarks',
            ] if c not in sort_bys],
            *append,
        ] if c in df]]
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


def df_require_nonempty_for_api(df: pd.DataFrame, msg: str, **data) -> pd.DataFrame:
    """Shorthand to keep error handling concise"""
    return df_require_nonempty(df, ApiError(400, msg, **data))
