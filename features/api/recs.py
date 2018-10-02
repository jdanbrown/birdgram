from functools import lru_cache
import inspect
import linecache
import numbers
from typing import *

from more_itertools import one
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


def xc_meta(
    species: str,
    quality: str = None,
    n_recs: int = 10,
) -> pd.DataFrame:
    with log_time_context():

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
    quality: str = 'ab',
    n_recs: int = 50,
    audio_s: float = 10,
    scale: float = 2,
    view: bool = None,
    sp_cols: str = None,
    sort: str = 'date',  # Name apart from 'rank' so that e.g. /similar?rank=d_pc doesn't transfer to /species?sort=...
) -> pd.DataFrame:
    with log_time_context():

        # Return no results so user gets blank controls to fill in
        if not species:
            return pd.DataFrame()

        # Params
        species = species_for_query(species)
        if not species: return pd.DataFrame([])
        quality = quality and [q for q in quality for q in [q.upper()] for q in [{'N': 'no score'}.get(q, q)]]
        n_recs  = n_recs and np.clip(n_recs, 0, None)
        audio_s = np.clip(audio_s, 0, 30)
        scale   = np.clip(scale, 0, 5)

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

            # Top recs by `sort` (e.g. .date)
            #   - TODO Can't sort by xc_id since it's added by recs_featurize (-> recs_featurize_metdata_audio_slice), below
            .pipe(log_time_df, desc='sort', f=lambda df: (df
                .pipe(lambda df: df.sort_values(**one([
                    dict(by=sort, ascending=sort in ['quality', 'time'])
                    for sort in [sort if sort in df else 'date']
                ])))
                [:n_recs]
                .reset_index(drop=True)  # Reset to RangeIndex after sort
            ))

            # Featurize
            .pipe(log_time_df, desc='spectro_disp', f=lambda df: (df
                .pipe(recs_featurize_spectro_disp, scale=scale)  # .spectro_disp <- .spectro_bytes, .audio_bytes
            ))

            # View
            .pipe(log_time_df, desc='view', f=lambda df: (df
                .pipe(recs_view, view=view, sp_cols=sp_cols, links=['sort'])
                [lambda df: [c for c in [
                    'xc', 'xc_id',
                    'com_name', 'species', 'spectro_disp',
                    'quality', 'date', 'time',
                    'type', 'subspecies', 'background_species',
                    'recordist', 'elevation', 'place', 'remarks', 'bird_seen', 'playback_used',
                    'recs_for_sp',
                    # 'duration_s',  # TODO Surface the original duration (.duration_s is the sliced duration)
                ] if c in df]]
            ))

        )


def xc_similar_html(
    xc_id: int = None,
    quality: str = 'ab',
    n_sp: int = None,
    group_sp: str = 'y',
    n_sp_recs: int = 3,
    n_total: int = 30,
    audio_s: float = 10,
    scale: float = 2,
    dists: str = '2c',  # 2 (l2), 1 (l1), c (cosine)
    rank: str = 'd_pc',  # Name apart from 'sort' so that e.g. /species?sort=date doesn't transfer to /similar?rank=...
    random_state: int = 0,
    view: bool = None,
    sp_cols: str = None,
) -> pd.DataFrame:
    with log_time_context():

        # Return no results so user gets blank controls to fill in
        if not xc_id:
            return pd.DataFrame()

        # Params
        quality   = quality  and [q for q in quality for q in [q.upper()] for q in [{'N': 'no score'}.get(q, q)]]
        n_sp      = n_sp     and np.clip(n_sp,     0,  None)
        require(group_sp in ['y', 'n', '', None])
        group_sp  = group_sp == 'y'
        n_sp_recs = (n_sp_recs or None) and np.clip(n_sp_recs, 0, None)
        n_total   = n_total  and np.clip(n_total,  0,  1000)
        audio_s   = audio_s  and np.clip(audio_s,  0,  30)
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
        dist_info = {
            '2': sk.metrics.pairwise.distance_metrics()['l2'],
            '1': sk.metrics.pairwise.distance_metrics()['l1'],
            'c': sk.metrics.pairwise.distance_metrics()['cosine'],
        }
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
        d_slp = lambda sp_p: np.abs(-np.log(sp_p))  # d_slp: "species log prob" (abs for 1->0 i/o -0)
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
                    d_slp=lambda df: d_slp(df.sp_p),
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
                .sort_values(['d_slp', rank] if group_sp else rank)[:n_total]
                .reset_index(drop=True)  # Reset to RangeIndex after sort
            ))
        )

        # Featurize + view ranked_recs
        return (ranked_recs

            # Include query_rec in view
            #   - HACK Include it as the first result with some mocked-out columns
            .pipe(log_time_df, desc='add_query_rec', f=lambda df: (df
                .pipe(lambda df: pd.concat(
                    sort=True,  # [Silence "non-concatenation axis" warning -- not sure what we want, or if it matters...]
                    objs=[
                        (
                            # Expand query_rec cols to match df cols
                            pd.concat([DF([query_rec]), df[:0]])
                            # Add query_rec.d_slp for user feedback (mostly for model debugging)
                            .assign(
                                sp_p=lambda df: one(query_sp_p[query_sp_p.species == query_rec.species].sp_p.values),
                                d_slp=lambda df: d_slp(df.sp_p),
                            )
                            # Mock d_* cols as 0
                            .pipe(lambda df: df.fillna({k: 0 for k in df if k.startswith('d_')}))
                        ),
                        df,
                    ],
                ))
                .reset_index(drop=True)  # Reset to RangeIndex after concat
            ))

            # Featurize
            .pipe(log_time_df, desc='spectro_disp (slow ok)', f=lambda df: (df
                .pipe(recs_featurize_spectro_disp, scale=scale)  # .spectro_disp <- .spectro_bytes, .audio_bytes
            ))

            # View
            .pipe(log_time_df, desc='view', f=lambda df: (df
                .pipe(recs_view, view=view, sp_cols=sp_cols, links=['rank'])
                .pipe(lambda df: (df
                    .pipe(df_reorder_cols, first=[  # Manually order d_* cols [Couldn't get to work above]
                        'd_slp',
                        *[c for c in [d_(f, m) for m in '2c' for f in 'fp'] if c in df],
                    ])
                ))
                [lambda df: [c for c in [
                    'xc', 'xc_id',
                    *unique_everseen(c for c in df if c.startswith('d_')),  # Scores (d_*)
                    'com_name', 'species', 'spectro_disp',
                    'quality', 'date', 'time',
                    'type', 'subspecies', 'background_species',
                    'recordist', 'elevation', 'place', 'remarks', 'bird_seen', 'playback_used',
                    'recs_for_sp',
                    # 'duration_s',  # TODO Surface the original duration (.duration_s is the sliced duration)
                ] if c in df]]
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
                #   - TODO Figure out a better way to propagate invalid audios (e.g. empty cache file) so we can handle this more robustly
                drop_uncached_slice=True,
                # Don't load .audio for pre-rank recs (only for final n_total recs, below)
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
    #   - Repro: xc_similar_html(sort='d_fc', sp_cols='species', xc_id=381417, n_total=5, n_sp=17)
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

        # TODO TODO Testing
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

        # TODO TODO XXX
        # plot_many_kwargs = {
        #     **plot_kwargs,
        #     'progress': dict(**config.api.recs.progress_kwargs),  # threads > sync, threads >> processes
        #     '_nocache': True,  # Dev: disable plot_many cache since it's blind to most of our sub-many code changes [TODO Revisit]
        # }
        # return (recs
        #     # HACK Workaround some bug I haven't debugged yet
        #     #   - In server, .spectro column is present but all nan, which breaks downstream
        #     #   - In notebook, works fine
        #     #   - Workaround: force-drop .spectro column if present
        #     .drop(columns=['spectro'], errors='ignore')
        #     # .spectro
        #     .assign(spectro=lambda df: (sg.features
        #         # FIXME Slow (load_audio) vs. incorrect (load)
        #         #   - features.load=load_audio is correct, but very slow because it doesn't share .spectro cache with features.load=load
        #         #   - features.spectro cache key includes load.format='wav'|'mp4', which makes load_audio vs. load bust cache
        #         #   - TODO Should load.format be part of features.spectro cache key? Seems like no, but think harder...
        #         # .replace(**dict(load=load_audio) if load_audio else {})  # Set features.load for features._cache() -> load._audio()
        #         .spectro(df,
        #             cache=True,
        #             # FIXME Saw hangs with dask threads
        #             #   - Repro: use='dask', scheduler='threads', 1k recs, cache hit and/or miss
        #             #   - No repro: use='dask', scheduler='synchronous'
        #             #   - HACK Going with use='sync' to work around...
        #             # **config.api.recs.progress_kwargs,  # threads >> sync, procs
        #             use='sync',
        #         )
        #     ))
        #     # .spectro_bytes
        #     .pipe(df_assign_first,
        #         spectro_bytes_mimetype=format_to_mimetype(format),
        #         spectro_bytes=lambda df: [
        #             b
        #             for imgs in [plot_slice.many(df, sg.features, **plot_many_kwargs,
        #                 pad_s=pad_s,  # Careful: use pad_s instead of slice_s, else excessive writes (slice->mp4->slice->mp4)
        #                 show=False,   # Return img instead of plotting
        #                 audio=False,  # Return PIL.Image (no audio) instead of Displayable (with embedded html audio)
        #             )]
        #             for b in map_progress(desc='pil_img_save_to_bytes', xs=imgs, f=lambda img: (
        #                 # TODO Optimize png/img size -- currently bigger than mp4 audio! (a rough start: notebooks/png_compress)
        #                 pil_img_save_to_bytes(
        #                     img.convert('RGB'),  # Drop alpha channel
        #                     format=format,
        #                 )
        #             ))
        #         ],
        #     )
        #     # Drop intermediate .spectro col, since downstreams should only depend on .spectro_bytes
        #     .drop(columns=[
        #         'spectro',
        #     ])
        # )


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
    # TODO TODO Replace plot_slice -> plot_spectro: nothing changes except we can remove the warning comment about slice_s
    spectro_img = plot_slice(rec, sg.features, **plot_kwargs,
        pad_s=pad_s,  # Careful: use pad_s instead of slice_s, else excessive writes (slice->mp4->slice->mp4)
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
                        style_css=scale and '.bubo-audio-container img { height: %spx; }' % int(40 * scale),
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
    df_if_cols = lambda df, cols, f: f(df) if set([cols] if isinstance(cols, str) else cols) <= set(df.columns) else df
    df_col_map_if_col = lambda df, **cols: df_col_map(df, **{k: v for k, v in cols.items() if k in df})

    if not view:
        return recs

    return (recs
        .pipe(lambda df: df_col_map(df, **{
            # Rank by scores (d_*)
            c: lambda x, c=c: '''<a href="{{ req_query_with(rank=%r) }}">%s</a>''' % (c, round_sig_frac(x, 2))
            for c in df
            if c.startswith('d_') and 'rank' in links
        }))
        .pipe(df_if_cols, 'xc_id', lambda df: (df
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
        .pipe(df_if_cols, 'species', lambda df: (df
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
        .pipe(df_if_cols, 'date', lambda df: df.assign(
            date=lambda df: df_map_rows(df, lambda row: '%(year).0f-%(month_day)s' % row),  # .year is float (b/c None)
        ))
        .pipe(df_col_map_if_col, **{
            # Sort by cols
            c: lambda x, c=c: '''<a href="{{ req_query_with(sort=%r) }}">%s</a>''' % (c, x)
            for c in ['quality', 'date', 'time']
            if 'sort' in links
        })
        .pipe(df_col_map_if_col,
            # df_cell_str to prevent df.to_html from truncating long strs
            type=lambda x: df_cell_str('<br>'.join(textwrap.wrap(x,
                width=max(20, int(len(x) / 1.8) or np.inf),
            ))),
        )
        .pipe(df_col_map_if_col,
            background_species=lambda xs: ' '.join(xs),
        )
        .pipe(df_col_map_if_col,
            # df_cell_str to prevent df.to_html from truncating long strs
            background_species=lambda x: df_cell_str('<br>'.join(line.strip(',') for line in textwrap.wrap(x,
                width=max((4 + 1) * 3, int(len(x) / 1.5) or np.inf),
            ))),
        )
        .pipe(df_if_cols, ['recordist', 'license_type'], lambda df: df.assign(
            recordist=lambda df: df_map_rows(df, lambda row: '''
                %(recordist)s<br/>%(license_type)s
            ''' % row),
        ))
        .pipe(df_if_cols, ['place', 'lat', 'lng'], lambda df: df.assign(
            # df_cell_str to prevent df.to_html from truncating long strs
            place=lambda df: df_map_rows(df, lambda row: df_cell_str('''
                %(place)s<br/>
                <a href="https://www.google.com/maps/place/%(lat)s,%(lng)s/@%(lat)s,%(lng)s,6z">(%(lat)s, %(lng)s)</a>
            ''' % row)),
        ))
        .pipe(df_col_map_if_col,
            # df_cell_str to prevent df.to_html from truncating long strs
            remarks=lambda x: df_cell_str('<br>'.join(textwrap.wrap(x,
                width=max(80, int(len(x) / 1.8) or np.inf),
            ))),
        )
        # Fill any remaining nulls with ''
        #   - Strip cats else they'll reject '' (unless it's a valid cat)
        .pipe(df_cat_to_str)
        .fillna('')
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
