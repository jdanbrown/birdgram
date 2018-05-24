## For export

from addict import Dict
import dask
import dask.array as da
import dask.dataframe as dd
from more_itertools import *


## matplotlib

import matplotlib.pyplot as plt
import numpy as np


def plt_signal(y: np.array, x_scale: float = 1, show_ydtype=False, show_yticks=False):
    # Performance on ~1.1M samples:
    #   - ggplot+geom_line / qplot is _really_ slow (~30s)
    #   - df.plot is decent (~800ms)
    #   - plt.plot is fastest (~550ms)
    plt.plot(
        np.arange(len(y)) / x_scale,
        y,
    )
    if not show_yticks:
        # Think in terms of densities and ignore the scale of the y-axis
        plt.yticks([])
    if show_ydtype:
        # But show the representation dtype so the user can stay aware of overflow and space efficiency
        plt.ylabel(y.dtype.type.__name__)
        if np.issubdtype(y.dtype, np.integer):
            plt.ylim(np.iinfo(y.dtype).min, np.iinfo(y.dtype).max)


## pandas

from collections import OrderedDict
import multiprocessing
import tempfile
import time
from typing import List
import uuid

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pandas as pd
from potoo.util import get_cols


def df_rows(df):
    return [row for i, row in df.iterrows()]


def df_flatmap_list_col(df, col_name, col_f=lambda s: s):
    return (df
        .assign(**{col_name: col_f(df[col_name])})
        .pipe(df_flatmap, lambda row: [
            OrderedDict({**row, col_name: x})
            for x in row[col_name]
        ])
    )


# Based on https://github.com/pandas-dev/pandas/issues/8517#issuecomment-247785821
def df_flatmap(df, f):
    return pd.DataFrame(
        row_out
        for _, row_in in df.iterrows()
        for row_out in f(row_in)
    )


def df_reorder_cols(df: pd.DataFrame, first: List[str] = [], last: List[str] = []) -> pd.DataFrame:
    first_last = set(first) | set(last)
    return df.reindex(columns=first + [c for c in df.columns if c not in first_last] + last)


def df_apply_with_progress(
    df,
    f,
    dask=True,
    scheduler='processes',  # 'processes' | 'threads' | 'synchronous'
    npartitions=None,
    chunksize=None,
    target_time_s_per_partition=2,
    min_npartitions_per_core=5,
    **kwargs,
):
    """
    Example usage:
        df.pipe(df_apply_with_progress, lambda row:
            ...  # Transform row
        ))
    """
    if not dask:
        return df.apply(axis=1, func=f, **kwargs)
    else:
        with ProgressBar(width=get_cols() - 30):
            (time_1_row_s, meta) = timed(lambda: df[:1].apply(axis=1, func=f))
            if not npartitions and not chunksize:
                n_cores = multiprocessing.cpu_count()
                npartitions = int(len(df) * time_1_row_s / target_time_s_per_partition)  # Estimate from time_1_row_s
                npartitions = round(npartitions / n_cores) * n_cores  # Round to multiple of n_cores
                npartitions = max(npartitions, n_cores * min_npartitions_per_core)  # Min at n_cores * k (for small k)
            return (dd
                .from_pandas(df, npartitions=npartitions, chunksize=chunksize)
                .apply(axis=1, func=f, meta=meta, **kwargs)
                .compute(get=dask_get_for_scheduler_name(scheduler))
            )


# Mimic http://dask.pydata.org/en/latest/scheduling.html
def dask_get_for_scheduler_name(scheduler):
    if isinstance(scheduler, str):
        get = {
            'synchronous': dask.get,
            'threads': dask.threaded.get,
            'processes': dask.multiprocessing.get,
        }[scheduler]
    else:
        get = scheduler
    return get


## unix

import os


def ls(dir):
    return [
        os.path.join(dir, filename)
        for filename in os.listdir(dir)
    ]


## util

from functools import partial, wraps
import os
import pickle
import random
import shlex


def shuffled(xs: iter, random=random) -> list:
    xs = list(xs)  # Avoid mutation + unroll iters
    random.shuffle(xs)
    return xs


# XXX Use more_itertools.flatten
# def flatten(xss: iter) -> iter:
#     return (x for xs in xss for x in xs)


# XXX Use more_itertools.one
# def only(xs):
#     [x] = list(xs)
#     return x


def generator_to(agg):
    def decorator(f):
        @wraps(f)
        def g(*args, **kwargs):
            return agg(f(*args, **kwargs))
        return g
    return decorator


def ensure_parent_dir(path):
    mkdir_p(os.path.dirname(path))
    return path


def touch_file(path):
    ensure_parent_dir(path)
    open(path, 'w').close()
    return path


def mkdir_p(path):
    os.system('mkdir -p %s' % shlex.quote(path))


def puts(x):
    print(x)
    return x


def singleton(cls):
    """Class decorator"""
    return cls()


def timed(f):
    start_s = time.time()
    x = f()
    elapsed_s = time.time() - start_s
    return (elapsed_s, x)


# TODO Add some sort of automatic invalidation. To manually invalidate, just go delete the file you specified.
def cache_to_file_forever(path):
    def decorator(f):
        def g(*args, **kwargs):
            try:
                with open(ensure_parent_dir(path), 'rb') as fd:
                    return pickle.load(fd)
            except:
                x = f(*args, **kwargs)
                with open(ensure_parent_dir(path), 'wb') as fd:
                    pickle.dump(x, fd)
                return x
        return g
    return decorator
