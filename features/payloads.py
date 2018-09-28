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
import structlog

from datasets import *
from util import *

log = structlog.get_logger(__name__)


def df_cache_hybrid(
    compute: Callable[[], pd.DataFrame],
    path: str,
    desc: str,  # Only for logging
    refresh=False,  # Force cache miss (compute + write)
    feat_cols: Iterable[str] = None,  # Infer if None
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
    non_feats_path = path / 'non_feats.parquet'
    feat_path = lambda k: path / f"feat-{k}.npy"
    rel = lambda p: str(Path(p).relative_to(cache_dir))

    # Utils
    basename = os.path.basename
    naturalsize_path = lambda p: naturalsize(Path(p).stat().st_size)

    # Use manifest.json to track completion of writes
    #   - Mark completion of all file writes (else we can incorrectly conclude cache hit vs. miss)
    #   - Indicate which files were written (else we can read too many files if multiple different write attempts)
    if refresh or not manifest_path.exists():
        # Cache miss
        #   - Blindly overwrite any files from previous attempts
        log.info(f'Miss: {rel(path)}')

        # Compute
        #   - use='sync': don't interfere with user's progress bars in compute(), but still show start/end/elapsed
        df = one_progress(use='sync', desc=f'df_cache_hybrid:compute[{desc}]', f=lambda: (
            compute()
            .pipe(df_require_index_is_trivial)  # Nontrivial indexes aren't supported (complex and don't need it)
        ))

        # Measure write time, excluding compute
        start_s = time.time()

        # Infer feat_cols if not provided
        if feat_cols is None:
            # Inference: all cols with value type np.ndarray
            feat_cols = [] if df.empty else [
                k for k in df if isinstance(df.iloc[0][k], np.ndarray)
            ]
            log.info('Miss: Inferred feat_cols%s' % feat_cols)

        # Write non_feats.parquet (all cols in one file)
        log.debug(f'Miss: Writing {basename(rel(non_feats_path))}')
        (df
            .drop(columns=feat_cols)
            .to_parquet(non_feats_path, engine=engine, compression=compression, **to_parquet_kwargs)
        )
        log.info(f'Miss: Wrote {basename(rel(non_feats_path))} ({naturalsize_path(non_feats_path)})')

        # Write feat-*.npy (one col per file)
        for k in feat_cols:
            x = np.array(list(df[k]))
            log.debug(f'Miss: Writing {basename(rel(feat_path(k)))}: {x.dtype}')
            np.save(feat_path(k), x)
            log.info(f'Miss: Wrote {basename(rel(feat_path(k)))}: {x.dtype} ({naturalsize_path(feat_path(k))})')

        # Write manifest to mark completion of writes
        #   - tmp + atomic rename (else empty manifest.pkl -> stuck with cache hit that fails)
        #   - Make tmp_path in target dir instead of /tmp else os.rename will fail if dirs are on separate mounts
        tmp_path = Path(manifest_path).with_suffix('.tmp')
        with open(tmp_path, mode='wb') as f:
            pickle.dump(file=f, obj=dict(
                # Record feat_cols so we know which files to read on cache hit
                #   - Unsafe to assume it's the same as all feat-*.npy files, since who knows what bugs created those...
                feat_cols=feat_cols,
                # Record df.columns so we can restore col order
                columns=list(df.columns),
                # Record dtypes so we can restore categories (and maybe other stuff in there too)
                dtypes=df.dtypes,
            ))
        os.rename(tmp_path, manifest_path)

        # Return df
        #   - Should be the same as the df we will subsequently read on cache hit
        log.info(f'Miss: Done (took: %.3fs)' % (time.time() - start_s))
        return df

    else:
        # Cache hit
        log.info(f'Hit: {rel(path)}')
        start_s = time.time()

        # Read manifest
        with open(manifest_path, 'rb') as f:
            manifest = pickle.load(f)
        feat_cols = manifest['feat_cols']
        non_feats_size = naturalsize_path(non_feats_path)
        feat_size = {k: naturalsize_path(feat_path(k)) for k in feat_cols}

        # Read non_feats.parquet
        log.debug(f'Hit: Reading {basename(rel(non_feats_path))} ({non_feats_size})')
        non_feats = (
            pd.read_parquet(non_feats_path, engine=engine, **read_parquet_kwargs)
            .pipe(df_require_index_is_trivial)  # (Guaranteed by cache-miss logic)
        )
        log.info(f'Hit: Read {basename(rel(non_feats_path))} ({non_feats_size})')

        # Read feat-*.npy
        feats: Mapping[str, np.ndarray] = {}
        for k in feat_cols:
            log.debug(f'Hit: Reading {basename(rel(feat_path(k)))} ({feat_size[k]})')
            x = np.load(feat_path(k))
            log.info(f'Hit: Read {basename(rel(feat_path(k)))}: {x.dtype} ({feat_size[k]})')
            feats[k] = x

        # Build df
        log.info(f'Hit: Join non_feats + feats')
        df = (
            non_feats.assign(**{
                k: list(x)  # np.ndarray[m,n] -> List[np.ndarray[n]], else df.assign barfs
                for k, x in feats.items()
            })
            .pipe(df_require_index_is_trivial)  # (Guaranteed by cache-miss logic)
            [manifest['columns']]               # Restore col order
            .astype(manifest['dtypes'])         # Restore categories (and maybe other dtype stuff too)
        )

        # Return df
        #   - Should be the same as the df we computed and returned on cache miss
        #   - TODO Tests (see notebooks/api_dev_search_recs_hybrid)
        log.info(f'Hit: Done (took: %.3fs)' % (time.time() - start_s))
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
        #   - use='sync': don't interfere with user's progress bars in compute(), but still show start/end/elapsed
        df = one_progress(desc=f'df_cache_parquet:compute[{desc}]', use='sync', f=lambda: (
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
            #   - use='sync': don't interfere with user's progress bars in compute(), but still show start/end/elapsed
            real_df = one_progress(desc=f'df_cache_sqlite:compute[{table}]', use='sync', f=lambda: (
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
            #   - use='sync': we should be fast, dask progress bars are slow, still show start/end/elapsed
            metadata = dict(
                dtypes=real_df.dtypes,  # Includes categories
                col_conversions=col_conversions,
            )
            one_progress(desc=f'df_cache_sqlite:df.to_sql[{metadata_table}]', use='sync', f=lambda: (
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
            #   - use='sync': we should be fast, dask progress bars are slow, still show start/end/elapsed
            metadata = one_progress(desc=f'df_cache_sqlite:pd.read_sql_table[{metadata_table}]', use='sync', f=lambda: (
                pd.read_sql_table(metadata_table, conn)
                .applymap(pickle.loads)
                .pipe(lambda df: dict(one(df_rows(df))))
            ))
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
