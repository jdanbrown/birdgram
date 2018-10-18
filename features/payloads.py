import os
from pathlib import Path
import pickle
import time
from typing import *

from frozendict import frozendict
from more_itertools import one
import numpy as np
import pandas as pd
from potoo.pandas import df_require_index_is_trivial
from potoo.plot import *
from potoo.util import strip_endswith
import sh
import structlog

from datasets import *
from util import *

log = structlog.get_logger(__name__)


def df_cache_hybrid(
    compute: Callable[[], pd.DataFrame],
    path: str,
    desc: str,  # Only for logging [HACK Used as the sqlite table name if write_mobile_payload]
    refresh=False,  # Force cache miss (compute + write)
    feat_cols: Iterable[str] = None,  # Infer if None
    write_mobile_payload=False,  # Also write the mobile payload (ignored on read from cache hit)
    display_sizes=True,  # Logging
    plot_sizes=False,  # Logging
    # to_sql (for write_mobile_payload)
    to_sql_kwargs=frozendict(),
    # to_parquet/read_parquet
    engine='fastparquet',  # 'fastparquet' | 'pyarrow'
    compression='uncompressed',  # 'uncompressed' (fast!) | 'gzip' (slow) | 'snappy' (not installed)
    read_parquet_kwargs=frozendict(),
    **to_parquet_kwargs,
) -> pd.DataFrame:
    """
    Save/load a df to file(s), quickly
    - .npy for heavy np.array cols
    - .parquet for everything else
    - Sadly, all the simpler approaches I tried were _significantly_ slower
        - notebooks/api_dev_search_recs_hybrid
        - notebooks/api_dev_search_recs_parquet
        - notebooks/api_dev_search_recs_sqlite
        - notebooks/api_search_recs_pkl_parquet_sqlite
    """

    # Params
    assert not Path(path).is_absolute(), f"{path}"
    path = ensure_dir(Path(cache_dir) / path)
    manifest_path = path / 'manifest.pkl'
    feat_path = lambda k: path / f"feat-{k}.npy"
    bytes_path = lambda k: path / f"bytes-{k}.parquet"
    lite_path = path / 'lite.parquet'

    # State
    file_sizes = {}
    mobile_file_sizes = {}

    # Utils
    rel_to_cache_dir = lambda p: str(Path(p).relative_to(cache_dir))
    rel = lambda p: str(Path(p).relative_to(path))
    size_path = lambda p: Path(p).stat().st_size
    naturalsize_path = lambda p: naturalsize(size_path(p))

    # Use manifest.json to track completion of writes
    #   - Mark completion of all file writes (else we can incorrectly conclude cache hit vs. miss)
    #   - Indicate which files were written (else we can read too many files if multiple different write attempts)
    if refresh or not manifest_path.exists():
        # Cache miss
        #   - Blindly overwrite any files from previous attempts
        log.info(f'Miss: {rel_to_cache_dir(path)}')

        # Compute
        with log_time_context(f'Compute: {desc}'):
            df = (
                compute()
                .pipe(df_require_index_is_trivial)  # Nontrivial indexes aren't supported (complex and don't need it)
            )

        # Measure write time, excluding compute
        with log_time_context(f'Miss: {desc}'):

            # Params (written to manifest here, read from manifest below, to mitigate code/data incompats)
            bytes_cols = [
                'audio_bytes',
                'spectro_bytes',
            ]

            # Infer feat_cols if not provided
            if feat_cols is None:
                # Inference: all cols with value type np.ndarray
                feat_cols = [] if df.empty else [
                    k for k in df if isinstance(df.iloc[0][k], np.ndarray)
                ]
                log.info('Miss: Inferred feat_cols%s' % feat_cols)

            # Write lite.parquet (all cols in one file)
            log.debug(f'Miss: Writing {rel(lite_path)}')
            (df
                .drop(columns=bytes_cols + feat_cols)
                .to_parquet(lite_path, engine=engine, compression=compression, **to_parquet_kwargs)
            )
            log.info(f'Miss: Wrote {rel(lite_path)} ({naturalsize_path(lite_path)})')
            file_sizes[rel(lite_path)] = size_path(lite_path)

            # Write bytes-*.parquet (single col per file)
            for k in bytes_cols:
                log.debug(f'Miss: Writing {rel(bytes_path(k))}')
                (df
                    [[k]]
                    .to_parquet(bytes_path(k), engine=engine, compression=compression, **to_parquet_kwargs)
                )
                log.info(f'Miss: Wrote {rel(bytes_path(k))} ({naturalsize_path(bytes_path(k))})')
                file_sizes[rel(bytes_path(k))] = size_path(bytes_path(k))

            # Write feat-*.npy (one col per file)
            for k in feat_cols:
                x = np.array(list(df[k]))
                log.debug(f'Miss: Writing {rel(feat_path(k))}: {x.dtype}')
                np.save(feat_path(k), x)
                log.info(f'Miss: Wrote {rel(feat_path(k))}: {x.dtype} ({naturalsize_path(feat_path(k))})')
                file_sizes[f'{rel(feat_path(k))}: {x.dtype}'] = size_path(feat_path(k))

            # Write manifest to mark completion of writes
            #   - tmp + atomic rename (else empty manifest.pkl -> stuck with cache hit that fails)
            #   - Make tmp_path in target dir instead of /tmp else os.rename will fail if dirs are on separate mounts
            tmp_path = Path(manifest_path).with_suffix('.tmp')
            with open(tmp_path, mode='wb') as f:
                pickle.dump(file=f, obj=dict(
                    # Record bytes_cols + feat_cols so we know which files to read on cache hit
                    #   - Unsafe to assume it's the same as all feat-*.npy files, since who knows what bugs created those...
                    bytes_cols=bytes_cols,
                    feat_cols=feat_cols,
                    # Record df.columns so we can restore col order
                    columns=list(df.columns),
                    # Record dtypes so we can restore categories (and maybe other stuff in there too)
                    dtypes=df.dtypes,
                ))
            os.rename(tmp_path, manifest_path)

        # Output df should be the same as the df we will subsequently read on cache hit
        #   - TODO Tests (see notebooks/api_dev_search_recs_hybrid)

    else:
        # Cache hit
        log.info(f'Hit: {rel_to_cache_dir(path)}')
        with log_time_context('Hit'):

            # Read manifest
            with open(manifest_path, 'rb') as f:
                manifest = pickle.load(f)
            bytes_cols = manifest['bytes_cols']
            feat_cols = manifest['feat_cols']

            # Read lite.parquet
            log.debug(f'Hit: Reading {rel(lite_path)} ({naturalsize_path(lite_path)})')
            lite = (
                pd.read_parquet(lite_path, engine=engine, **read_parquet_kwargs,
                    index=False,  # Else you get a trivial index with name 'index'
                )
                .pipe(df_require_index_is_trivial)  # (Guaranteed by cache-miss logic)
            )
            log.info(f'Hit: Read {rel(lite_path)} ({naturalsize_path(lite_path)})')
            file_sizes[rel(lite_path)] = size_path(lite_path)

            # Read bytes-*.parquet
            bytess: Mapping[str, pd.DataFrame] = {}
            for k in bytes_cols:
                log.debug(f'Hit: Reading {rel(bytes_path(k))} ({naturalsize_path(bytes_path(k))})')
                bytess[k] = (
                    pd.read_parquet(bytes_path(k), engine=engine, **read_parquet_kwargs,
                        index=False,  # Else you get a trivial index with name 'index'
                    )
                    .pipe(df_require_index_is_trivial)  # (Guaranteed by cache-miss logic)
                )
                log.info(f'Hit: Read {rel(bytes_path(k))} ({naturalsize_path(bytes_path(k))})')
                file_sizes[rel(bytes_path(k))] = size_path(bytes_path(k))

            # Read feat-*.npy
            feats: Mapping[str, np.ndarray] = {}
            for k in feat_cols:
                log.debug(f'Hit: Reading {rel(feat_path(k))} ({naturalsize_path(feat_path(k))})')
                feats[k] = np.load(feat_path(k))
                log.info(f'Hit: Read {rel(feat_path(k))}: {feats[k].dtype} ({naturalsize_path(feat_path(k))})')
                file_sizes[f'{rel(feat_path(k))}: {feats[k].dtype}'] = size_path(feat_path(k))

            # Build df
            log.info(f'Hit: Join lite + bytes + feats')
            df = (lite
                .assign(**{
                    k: x[k]
                    for k, x in bytess.items()
                }, **{
                    k: list(x)  # np.ndarray[m,n] -> List[np.ndarray[n]], else df.assign barfs
                    for k, x in feats.items()
                })
                .pipe(df_require_index_is_trivial)  # (Guaranteed by cache-miss logic)
                [manifest['columns']]               # Restore col order
                .astype(manifest['dtypes'])         # Restore categories (and maybe other dtype stuff too)
            )

        # Output df should be the same as the df we computed and returned on cache miss
        #   - TODO Tests (see notebooks/api_dev_search_recs_hybrid)

    # HACK Write mobile payload, if requested [TODO Factor this out into a separate function]
    if write_mobile_payload:
        with log_time_context('Mobile: Writing payload'):

            # Utils
            def write_file(path, mode, data):
                with open(path, mode) as f:
                    f.write(data)

            # Params
            mobile_feat_cols = [
                # 'feat',  # Omit, since it's big and empirically not very useful
                'f_preds',
            ]
            mobile_path = path / 'mobile'
            mobile_db_path = mobile_path / f'{desc}.sqlite3'
            mobile_file_path = lambda species, xc_id, name, format: (
                mobile_path / f'files/{species}/{xc_id}-{name}.{format}'
            )

            # Create and connect to sqlite file
            with sqla_oneshot_eng_conn_tx(f'sqlite:///{ensure_parent_dir(mobile_db_path)}') as conn:
                mobile_df = df

                # Materialize mobile_feat_cols
                #   - e.g. .f_preds -> .f_preds_0, .f_preds_1, ...
                for k in mobile_feat_cols:
                    n = one(df[k].map(len).drop_duplicates())
                    with log_time_context(f'Mobile: Materialize feat col: {k}'):
                        mobile_df = (mobile_df
                            .pipe(lambda df: df.assign(**{  # (.pipe to avoid error-prone lambda scoping inside dict comp)
                                f'{k}_{i}': df[k].map(lambda x: x[i])
                                for i in range(n)
                            }))
                        )

                # Write bytes_cols to files (one file per cell)
                for bytes_col in bytes_cols:
                    bytes_name = strip_endswith(bytes_col, '_bytes', check=True)
                    mobile_files_df = (mobile_df
                        .assign(mobile_file_path=lambda df: [
                            mobile_file_path(
                                species=row.species,
                                xc_id=row.xc_id,
                                name=bytes_name,
                                format=mimetype_to_format(row[f'{bytes_col}_mimetype']),
                            )
                            for row in df_rows(df)
                        ])
                    )
                    # Ensure dirs (O(species))
                    map_progress(desc=f'Mobile: Ensure dirs: {bytes_col}', use='dask', scheduler='threads',
                        xs=mobile_files_df.mobile_file_path.map(lambda p: p.parent).drop_duplicates(),
                        f=lambda dir: (
                            # ensure_dir(dir)  # Known good, but forks
                            dir.mkdir(parents=True, exist_ok=True)  # Doesn't fork, but might barf in some edge cases...?
                        ),
                    )
                    # Write files (O(xc_id * bytes_cols))
                    map_progress_df_rows(desc=f'Mobile: Write files: {bytes_col}', use='dask', scheduler='threads',
                        df=mobile_files_df,
                        f=lambda row: (
                            write_file(row.mobile_file_path, 'wb', row[bytes_col])
                        ),
                    )
                    # HACK Approx du for all the files we just wrote (without O(n) stat of each file)
                    mobile_file_sizes[rel(mobile_file_path(species='*', xc_id='*', name=bytes_name, format='*'))] = (
                        df[bytes_col].map(len).sum()
                    )

                # Drop non-sqlite cols
                mobile_df = mobile_df.drop(columns=[
                    # bytes_col are now in files
                    *bytes_cols,
                    *[f'{k}_mimetype' for k in bytes_cols],
                    # All feat_cols (not just the mobile_feat_cols that we materialized as sqlite cols)
                    *feat_cols,
                ])

                # Write sqlite .db file
                #   - HACK Copy/pasted from the df_cache_sqlite write path [TODO Factor out common parts for reuse]
                table = desc
                # Convert cols to sql representation
                col_conversions = {} if mobile_df.empty else {
                    k: fg
                    for k, v in mobile_df.iloc[0].items()
                    for fg in [(
                        (json_dumps_canonical, json.loads) if isinstance(v, (list, dict)) else  # list/dict <-> json (str)
                        None
                    )]
                    if fg
                }
                mobile_df = mobile_df.assign(**{
                    k: map_progress(use='log_time_all', desc=f'Mobile: Covert col for sql: {to_sql.__name__}({k})',
                        xs=mobile_df[k],
                        f=to_sql,
                    )
                    for k, (to_sql, _) in col_conversions.items()
                })
                # Write to sql
                #   - Fail if nontrivial indexes, since they're error-prone to manage and we don't really benefit from them here
                df_require_index_is_trivial(mobile_df)
                log.debug(f'Mobile: Writing {rel(mobile_db_path)}')
                mobile_df.to_sql(table, conn,
                    index=False,  # Silently drop indexes (we shouldn't have any)
                    if_exists='replace',
                    **{**to_sql_kwargs,
                        'chunksize': 1000,  # Safe default for big writes (pd default writes all rows at once -- mem unsafe)
                    },
                )
                log.info(f'Mobile: Wrote {rel(mobile_db_path)} ({naturalsize_path(mobile_db_path)})')
                mobile_file_sizes[rel(mobile_db_path)] = size_path(mobile_db_path)

    # Debug: display/plot file sizes, if requested
    for file_sizes in [
        file_sizes,
        *[mobile_file_sizes if write_mobile_payload else []],
    ]:
        file_sizes_df = (
            pd.DataFrame(dict(file=file, size=size) for file, size in file_sizes.items())
            .pipe(lambda df: df.append(pd.DataFrame([dict(file='total', size=df['size'].sum())])))
            .sort_values('size', ascending=False)
        )
        if display_sizes:
            display(file_sizes_df)
        if plot_sizes:
            display(file_sizes_df
                .pipe(df_ordered_cat, 'file', transform=reversed)
                .pipe(ggplot)
                + aes(x='file', y='size')
                + geom_col()
                + coord_flip()
                + scale_y_continuous(labels=labels_bytes(), breaks=breaks_bytes(pow=3))
                + theme_figsize(aspect=1/8)
            )

    return df


# Slow when you have to np_save_to_bytes, use df_cache_hybrid instead
#   - See notebooks in df_cache_hybrid docstring
def df_cache_parquet(
    compute: Callable[[], pd.DataFrame],
    path: str,
    desc: str,  # Only for logging
    refresh=False,  # Force cache miss (compute + write)
    # to_parquet/read_parquet
    #   - Perf notes in notebooks/api_search_recs_pkl_parquet_sqlite
    engine='fastparquet',  # 'fastparquet' | 'pyarrow'
    compression='uncompressed',  # 'uncompressed' (fast!) | 'gzip' (slow) | 'snappy' (not installed)
    read_parquet_kwargs=frozendict(),
    **to_parquet_kwargs,
) -> pd.DataFrame:

    # Params
    assert not Path(path).is_absolute(), f"{path}"
    path = ensure_parent_dir(Path(cache_dir) / f"{path}.parquet")

    if refresh or not path.exists():
        # Cache miss:

        # Compute
        #   - sync_progress_kwargs: don't interfere with user's progress bars in compute(), but still show start/end/elapsed
        df = one_progress(desc=f'df_cache_parquet:compute[{desc}]', **config.sync_progress_kwargs, f=lambda: (
            compute()
        ))

        # Write to parquet
        one_progress(desc=f'df_cache_parquet:df.to_parquet[{desc}]', n=len(df), f=lambda: (
            df.to_parquet(path, engine=engine, compression=compression, **to_parquet_kwargs)
        ))

        # Return df
        #   - Should be the same as the df we will subsequently read on cache hit
        return df

    else:
        # Cache hit:

        # Read df from parquet
        size = path.stat().st_size
        df = one_progress(desc=f'df_cache_parquet:pd.read_parquet[{desc}]', size=size, f=lambda: (
            pd.read_parquet(path, engine=engine, compression=compression, **read_parquet_kwargs)
        ))

        # Return df
        #   - Should be the same as the df we computed and returned on cache miss
        #   - TODO Tests (see notebooks/api_dev_search_recs_sqlite)
        return df


# Slow when you have to np_save_to_bytes, use df_cache_hybrid instead
#   - See notebooks in df_cache_hybrid docstring
#   - TODO Revist .sqlite vs. hybrid once we have an idea how mobile will use search_recs
def df_cache_sqlite(
    compute: Callable[[], pd.DataFrame],
    path: Optional[str],  # None for in-mem db
    table: str,
    col_conversions: Callable[[pd.DataFrame], Mapping[str, Tuple[
        Callable[[any], any],  # to_sql: elem -> elem
        Callable[[any], any],  # from_sql: elem -> elem
    ]]] = lambda df: {},
    refresh=False,  # Force cache miss (compute + write)
    read_sql_table_kwargs=frozendict(),
    **to_sql_kwargs,
) -> pd.DataFrame:

    # Params
    metadata_table = f'_{table}_bubo_metadata'
    if not path:
        db_url = f'sqlite:///'  # In-mem db
    else:
        assert not Path(path).is_absolute(), f"{path}"
        path = ensure_parent_dir(Path(cache_dir) / f"{path}.sqlite3")
        db_url = f'sqlite:///{path}'

    # Connect so we can check if table exists
    #   - (.sqlite3 file is created on eng/conn creation, so checking if file exists isn't as reliable as table exists)
    log.debug('Connect', db_url=db_url)
    with sqla_oneshot_eng_conn_tx(db_url) as conn:
        if refresh or not conn.engine.has_table(table):
            # Cache miss:

            # Compute
            #   - sync_progress_kwargs: don't interfere with user's progress bars in compute(), but still show start/end/elapsed
            real_df = one_progress(desc=f'df_cache_sqlite:compute[{table}]', **config.sync_progress_kwargs, f=lambda: (
                compute()
            ))
            sql_df = real_df

            # Convert cols to sql representation
            col_conversions = col_conversions(sql_df)
            sql_df = sql_df.assign(**{
                k: map_progress(desc=f'df_cache_sqlite:col_to_sql[{table}: {to_sql.__name__}({k})]', xs=sql_df[k], f=to_sql)
                for k, (to_sql, _) in col_conversions.items()
            })

            # Write to sql
            #   - sql_df, not real_df
            #   - Fail if nontrivial indexes, since they're error-prone to manage and we don't really benefit from them here
            df_require_index_is_trivial(sql_df)
            one_progress(desc=f'df_cache_sqlite:df.to_sql[{table}]', n=len(sql_df), f=lambda: (
                sql_df.to_sql(table, conn,
                    index=False,  # Silently drop indexes (we shouldn't have any)
                    if_exists='replace' if refresh else 'fail',  # Fail if table exists and not refresh (logic bug)
                    **{**to_sql_kwargs,
                        'chunksize': 1000,  # Safe default for big writes (pd default writes all rows at once -- mem unsafe)
                    },
                )
            ))

            # Write metadata to sql (so we can reconstruct df on read)
            #   - real_df, not sql_df
            #   - sync_progress_kwargs: we should be fast, dask progress bars are slow, still show start/end/elapsed
            metadata = dict(
                dtypes=real_df.dtypes,  # Includes categories
                col_conversions=col_conversions,
            )
            one_progress(desc=f'df_cache_sqlite:df.to_sql[{metadata_table}]', **config.sync_progress_kwargs, f=lambda: (
                pd.DataFrame({k: [v] for k, v in metadata.items()})
                .applymap(pickle.dumps)  # Pickle since some values unsafe for json (e.g. dtypes, categories)
                .to_sql(metadata_table, conn,
                    index=False,  # Don't write `index` col
                    if_exists='replace' if refresh else 'fail',  # Fail if table exists and not refresh (logic bug)
                )
            ))

            # Return df
            #   - real_df, not sql_df
            #   - Should be the same as the df we will subsequently read on cache hit
            return real_df

        else:
            # Cache hit:

            # Read metadata from sql (so we can reconstruct the written df)
            #   - Lightweight: do early to fail fast on bugs, before the heavy stuff below
            #   - sync_progress_kwargs: we should be fast, dask progress bars are slow, still show start/end/elapsed
            metadata = one_progress(desc=f'df_cache_sqlite:pd.read_sql_table[{metadata_table}]', **config.sync_progress_kwargs,
                f=lambda: (
                    pd.read_sql_table(metadata_table, conn)
                    .applymap(pickle.loads)
                    .pipe(lambda df: dict(one(df_rows(df))))
                ),
            )
            dtypes = metadata['dtypes']
            col_conversions = metadata['col_conversions']

            # Count table's rows, to give user a measure of how heavy the read_sql_table operation will be
            count_sql = 'select count(*) from %s' % conn.engine.dialect.identifier_preparer.quote(table)
            log.debug(count_sql)
            [(n_rows,)] = conn.execute(count_sql).fetchall()

            # Read df from sql
            #   - (Fast: ~2s for 35k recs)
            df = one_progress(desc=f'df_cache_sqlite:pd.read_sql_table[{table}]', n=n_rows, f=lambda: (
                pd.read_sql_table(table, conn,
                    index_col=None,  # Restore no indexes (we shouldn't have accepted any on write)
                    **{**read_sql_table_kwargs,
                        # (Default opts here)
                        # chunksize=None (default) is faster than pd.concat(... chunksize=1000 ...)
                    },
                )
            ))

            # Convert cols from sql representation
            #   - (Slow: each col.map(np_load_from_bytes) takes ~7.5s for 35k recs)
            #       - TODO No good for server startup, going with parquet for now (see notebooks/api_dev_search_recs_sqlite)
            df = df.assign(**{
                k: map_progress(desc=f'df_cache_sqlite:col_from_sql[{table}: {from_sql.__name__}({k})]', xs=df[k], f=from_sql)
                for k, (_, from_sql) in col_conversions.items()
            })

            # Restore dtypes (after col conversion)
            #   - Includes categories
            #   - (Fast: ~0s for 35k recs)
            df = one_progress(desc='dtypes', f=lambda: (
                df.astype(dtypes)
            ))

            # Return df
            #   - Should be the same as the df we computed and returned on cache miss
            #   - TODO Tests (see notebooks/api_dev_search_recs_sqlite)
            return df
