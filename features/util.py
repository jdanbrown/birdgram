## Side effects

import warnings

# Suppress "FutureWarning: 'pandas.core' is private. Use 'pandas.Categorical'"
#   - https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)


## For export

from typing import Iterable, Iterator, List, Mapping, Tuple, TypeVar, Union

from attrdict import AttrDict
import dask
import dask.array as da
import dask.dataframe as dd
import dateparser
from itertools import *
from more_itertools import *
import PIL
from potoo.pandas import df_ensure, df_summary
from potoo.util import singleton
import requests


## util

from contextlib import contextmanager
from functools import partial, wraps
import os
import pickle
import random
import shlex


def coalesce(*xs):
    for x in xs:
        if x is not None:
            return x


def shuffled(xs: iter, random=random) -> list:
    xs = list(xs)  # Avoid mutation + unroll iters
    random.shuffle(xs)
    return xs


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


def timed(f):
    start_s = time.time()
    x = f()
    elapsed_s = time.time() - start_s
    return (elapsed_s, x)


def short_circuit(short_f):
    def decorator(f):
        @wraps(f)
        def g(*args, **kwargs):
            y = short_f(*args, **kwargs)
            return y if y is not None else f(*args, **kwargs)
        return g
    return decorator


class AttrContextManager:

    @contextmanager
    def context(self, **kwargs):
        saved = {k: v for k, v in self.__dict__.items() if k in kwargs}
        self.__dict__.update(kwargs)
        try:
            yield
        finally:
            self.__dict__.update(saved)


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


## unix

import os


def ls(dir):
    return [
        os.path.join(dir, filename)
        for filename in os.listdir(dir)
    ]


## dataclasses

from collections import OrderedDict

from attrdict import AttrDict
import dataclasses


class DataclassConfig:

    @property
    def _fields(self) -> AttrDict:
        return AttrDict(dataclasses.asdict(self))

    @property
    def config(self) -> AttrDict:
        return {
            k: v
            for k, v in self._fields.items()
            if k not in (self.deps or {})
        }


class DataclassConversions:

    def asdict(self) -> dict:
        """Convert to dict preserving field order, e.g. for df rows"""
        return OrderedDict(dataclasses.asdict(self))


## pandas

from collections import OrderedDict
import tempfile
import time
from typing import Iterable
import uuid

from dataclasses import dataclass
import pandas as pd


Column = Iterable
Row = pd.Series


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


@dataclass
class box:
    """Useful e.g. for putting iterables inside pd.Series/np.array"""
    unbox: any

    @classmethod
    def many(cls, xs):
        return [box(x) for x in xs]


## dask

import multiprocessing
from typing import Callable, Iterable, TypeVar

from attrdict import AttrDict
import dask as _dask
import dask.bag
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import dask.multiprocessing
from dataclasses import dataclass
import pandas as pd
from potoo.util import get_cols

X = TypeVar('X')


@singleton
@dataclass
class dask_opts(AttrContextManager):
    override_dask: bool = None
    override_scheduler: bool = None


def df_apply_with_progress(
    df: pd.DataFrame,
    f: Callable[['Row'], 'Row'],
    dask=True,
    scheduler='threads',  # 'processes' | 'threads' | 'synchronous'
    npartitions=None,
    chunksize=None,
    **kwargs,
):
    """
    Example usage:
        df.pipe(df_apply_with_progress, lambda row:
            ...  # Transform row
        ))
    """
    dask = coalesce(dask_opts.override_dask, dask)
    scheduler = coalesce(dask_opts.override_scheduler, scheduler)
    if not dask:
        return df.apply(axis=1, func=f)
    else:
        with ProgressBar(width=get_cols() - 30):
            if not npartitions and not chunksize:
                (unit_sec, meta) = timed(lambda: df[:1].apply(axis=1, func=f))
                npartitions = _npartitions_for_unit_sec(len(df), unit_sec, **kwargs)
            return (dd
                .from_pandas(df, npartitions=npartitions, chunksize=chunksize)
                .apply(axis=1, func=f, meta=meta)
                .compute(get=dask_get_for_scheduler_name(scheduler))
            )


def map_with_progress(
    f: Callable[[X], X],
    xs: Iterable[X],
    dask=True,
    scheduler='threads',  # 'processes' | 'threads' | 'synchronous'
    partition_size=None,
    npartitions=None,
    **kwargs,
) -> Iterable[X]:
    dask = coalesce(dask_opts.override_dask, dask)
    scheduler = coalesce(dask_opts.override_scheduler, scheduler)
    if not dask:
        return list(map(f, xs))
    else:
        # HACK dask.bag.from_sequence([pd.Series(...), ...]) barfs -- workaround by boxing it
        # HACK dask.bag.from_sequence([np.array(...), ...]) flattens the arrays -- workaround by boxing it
        # HACK Avoid other cases we haven't tripped over yet by boxing everything unconditionally
        wrap, unwrap = (lambda x: box(x)), (lambda x: x.unbox)
        with ProgressBar(width=get_cols() - 30):
            if not partition_size and not npartitions:
                (unit_sec, _) = timed(lambda: list(map(f, xs[:1])))
                npartitions = _npartitions_for_unit_sec(len(xs), unit_sec, **kwargs)
            return (_dask.bag
                .from_sequence(map(wrap, xs), partition_size=partition_size, npartitions=npartitions)
                .map(unwrap)
                .map(f)
                .compute(get=dask_get_for_scheduler_name(scheduler))
            )


def _npartitions_for_unit_sec(n: int, unit_sec: float, target_sec_per_partition=2, min_npartitions_per_core=5) -> int:
    n_cores = multiprocessing.cpu_count()
    npartitions = int(n * unit_sec / target_sec_per_partition)  # Estimate from unit_sec
    npartitions = round(npartitions / n_cores) * n_cores  # Round to multiple of n_cores
    npartitions = max(npartitions, n_cores * min_npartitions_per_core)  # Min at n_cores * k (for small k)
    return npartitions


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


## bubo-features

from datetime import datetime
import json

from dataclasses import dataclass
from potoo.util import singleton
import yaml


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#
#   @singleton
#   class foo: pass
#   cloudpickle.dump(foo)  # Fails with "can't pickle _thread._local objects"
#
#   class foo: pass
#   foo = foo()
#   cloudpickle.dump(foo)  # Fails with "can't pickle _thread._local objects"
#
#   class Foo: pass
#   foo = Foo()
#   cloudpickle.dump(foo)  # Ok! Use this as a workaround.
#
# @singleton
@dataclass
class Log:

    verbose: bool = True

    def __call__(self, event, **kwargs):
        """Simple, ad-hoc logging specialized for interactive usage"""
        if self.verbose:
            t = datetime.utcnow().isoformat()
            t = t[:23]  # Trim micros, keep millis
            t = t.split('T')[-1]  # Trim date for now, since we're primarily interactive usage
            # Display timestamp + event on first line
            print('[%s] %s' % (t, event))
            # Display each (k,v) pair on its own line, indented
            for k, v in kwargs.items():
                v_yaml = yaml.safe_dump(json.loads(json.dumps(v)), default_flow_style=True, width=1e9)
                v_yaml = v_yaml.split('\n')[0]  # Handle documents ([1] -> '[1]\n') and scalars (1 -> '1\n...\n')
                print('  %s: %s' % (k, v_yaml))


# Workaround for @singleton (above)
log = Log()
