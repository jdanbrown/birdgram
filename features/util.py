#
# Side effects
#

import re
import warnings

# Suppress unhelpful warning
warnings.filterwarnings('ignore', category=FutureWarning, message=(
    re.escape("'pandas.core' is private. Use 'pandas.Categorical'")
))
import pandas.core.categorical  # If this emits a warning then our filter has bitrotted

# Supress unhelpful warning
#   - Have to force-disable warnings from sklearn, which obnoxiously force-enables them
#   - https://stackoverflow.com/a/33616192/397334
#   - https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/__init__.py#L97-L99
import sklearn  # Force-enables warnings for sklearn.*
warnings.filters = [  # Force-disable the force-enabling filter
    x for x in warnings.filters for (action, message, category, module, lineno) in [x]
    if (action, module) not in [
        ('always', re.compile(r'^sklearn\.'))
    ]
]
import sklearn.ensemble.weight_boosting  # If this emits a warning then our filter has bitrotted

#
# For export
#

from contextlib import contextmanager, ExitStack
from typing import *

from attrdict import AttrDict
import PIL
from potoo import debug_print
from potoo.pandas import df_ensure, df_summary
from potoo.util import generator_to, puts, singleton, strip_startswith, tap
import tqdm

# Order for precedence: last import wins (e.g. more_itertools.take shadows toolz.take)
from toolz import *
from more_itertools import *
from itertools import *

#
# util
#

from contextlib import contextmanager
from functools import partial, wraps
import glob
import hashlib
import inspect
import io
import re
import os
import pickle
import random
import shlex
import textwrap
from typing import *
from typing.io import *

import humanize
import structlog

log = structlog.get_logger(__name__)

X = TypeVar('X')


def can_iter(x: any) -> bool:
    try:
        iter(x)
        return True
    except:
        return False


def none_or(x, f, is_none=lambda x: x is None):
    return x if is_none(x) else f(x)


def coalesce(*xs):
    for x in xs:
        if x is not None:
            return x


def shuffled(xs: iter, random=random) -> list:
    xs = list(xs)  # Avoid mutation + unroll iters
    random.shuffle(xs)
    return xs


def glob_filenames_ensure_parent_dir(pattern: str) -> Iterable[str]:
    """
    Useful for globbing in gcsfuse, which requires dirs to exist as empty objects for its files to be listable
    """
    assert not re.match(r'.*[*?[].*/', pattern), \
        f'pattern[{pattern!r}] can only contain globs (*?[) in filenames, not dirnames'
    return glob.glob(ensure_parent_dir(pattern))


def ensure_dir(path):
    return mkdir_p(path)


def ensure_parent_dir(path):
    mkdir_p(os.path.dirname(path))
    return path


def touch_file(path):
    ensure_parent_dir(path)
    open(path, 'w').close()
    return path


def mkdir_p(path):
    os.system('mkdir -p %s' % shlex.quote(str(path)))
    return path


def timed(f):
    start_s = time.time()
    x = f()
    elapsed_s = time.time() - start_s
    return (elapsed_s, x)


class timer_start:

    def __init__(self):
        self.start_s = time.time()

    def stop(self):
        if self.start_s is not None:
            self.elapsed_s = time.time() - self.start_s
            self.start_s = None
            return self.elapsed_s


def short_circuit(short_f):
    def decorator(f):
        @wraps(f)
        def g(*args, **kwargs):
            y = short_f(*args, **kwargs)
            return y if y is not None else f(*args, **kwargs)
        return g
    return decorator


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


def sha1hex(*xs: Union[bytes, str], **encode_kwargs) -> str:
    """sha1hex(x, y) == sha1hex(x + y)"""
    h = hashlib.sha1()
    for x in xs:
        if isinstance(x, str):
            x = x.encode(**encode_kwargs)
        h.update(x)
    return h.hexdigest()


def enumerate_with_n(xs: Iterable[X]) -> (int, int, Iterator[X]):
    """Like enumerate(xs) but assume xs is materialized and include n = len(xs)"""
    n = len(xs)
    return ((i, n, x) for i, x in enumerate(xs))


def assert_(cond, msg=None):
    """Raise in an expression instead of a statement"""
    if msg is not None:
        assert cond, msg
    else:
        assert cond


@contextmanager
def print_time_delta(desc='time_delta', print=print):
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print('[%s] %.03fs' % (desc, end - start))


# Sometimes better than %memit...
@contextmanager
def print_mem_delta(desc='mem_delta', collect_before=False, collect_after=False, print=print, format=None):
    import gc
    import psutil
    from potoo.pretty import pformat
    format = (
        lambda x, format=format: format % x if isinstance(format, str) else
        pformat if format is None else
        format
    )
    proc = psutil.Process(os.getpid())
    if collect_before:
        gc.collect()
    start = proc.memory_full_info()._asdict()
    try:
        yield
    finally:
        if collect_after:
            gc.collect()
        end = proc.memory_full_info()._asdict()
        diff = {k: naturalsize(end[k] - start[k]) for k in start.keys()}
        desc = '[%s] ' % desc if desc else ''
        print('%s%s' % (desc, format(diff)))


# TODO Does this reject or accept kwargs that are repeated later? (A: Whatever partial does.) Which behavior do we want?
def sub_kwargs(f: Callable, **kwargs) -> Callable:
    """Bind to f the subset of kwargs that are accepted by f"""
    f_kwargs = inspect.signature(f).parameters.keys()
    return partial(f, **{
        k: v
        for k, v in kwargs.items()
        if k in f_kwargs
    })


def dedent_and_strip(s: str) -> str:
    """Localize the two operations in dedent(s).strip(), which get separated when s is very large (e.g. many lines)"""
    return textwrap.dedent(s).strip()


def bytes_from_file_write(write: Callable[[IO[bytes]], None]) -> bytes:
    f = io.BytesIO()
    write(f)
    f.seek(0)
    return f.read()


#
# json
#

import json


def json_dumps_safe(*args, **kwargs) -> str:
    return json.dumps(*args, **kwargs, default=lambda x: (
        # Unwrap np scalar dtypes (e.g. np.int64 -> int) [https://stackoverflow.com/a/16189952/397334]
        np.asscalar(x) if isinstance(x, np.generic) else
        # Else return as is
        x
    ))


def json_dumps_canonical(obj: any, **kwargs) -> str:
    """
    Dump a canonical json representation of obj (e.g. suitable for use as a cache key)
    - json_dumps_canonical(dict(a=1, b=2)) == json_dumps_canonical(dict(b=2, a=1))
    """
    return json_dumps_safe(obj, sort_keys=True, separators=(',', ':'), **kwargs)


#
# unix
#

import os


def ls(dir):
    return [
        os.path.join(dir, filename)
        for filename in os.listdir(dir)
    ]


#
# dataclasses
#

from potoo.dataclasses import *


class DataclassConfig(DataclassUtil):
    """Expose dataclass fields via .config (for sk.base.BaseEstimator)"""

    @property
    def config(self) -> AttrDict:
        return {
            k: v
            for k, v in self.asdict().items()
            if k not in (getattr(self, 'deps', None) or {})
        }


#
# humanize
#

import humanize


def naturalsize(size: float) -> str:
    # Workaround bug: negative sizes are always formatted as bytes
    s = (
        '-%s' % humanize.naturalsize(-size) if size < 0 else
        humanize.naturalsize(size)
    )
    # Abbrev 'Bytes' -> 'B'
    s = s.replace(' Bytes', ' B')
    return s


#
# numpy
#

import io

import numpy as np


def np_vectorize_asscalar(*args, **kwargs):
    """
    Like np.vectorize but return scalars instead of 0-dim np arrays
    - Spec:
        - np_vectorize_asscalar(f)(x) == np.vectorize(f)(x)[()]
    - Examples:
        - np_vectorize_asscalar(f)(3) == f(3)
        - np_vectorize_asscalar(f)(np.array([3,4])) == np.vectorize(f)(np.array([3,4]))
    - https://stackoverflow.com/questions/32766210/making-a-vectorized-numpy-function-behave-like-a-ufunc
    - https://stackoverflow.com/questions/39272465/make-np-vectorize-return-scalar-value-on-scalar-input
    """
    f = np.vectorize(*args, **kwargs)
    return lambda *args, **kwargs: f(*args, **kwargs)[()]


def np_save_to_bytes(x: np.ndarray, **kwargs) -> bytes:
    """np.save an array to npy bytes"""
    # From https://stackoverflow.com/a/18622264/397334
    #   - https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html
    #   - .npy shouldn't be slow or big: https://stackoverflow.com/a/41425878/397334
    return bytes_from_file_write(lambda f: (
        np.save(f, x, **kwargs)
    ))


def np_load_from_bytes(b: bytes) -> np.ndarray:
    """np.load an array from npy bytes"""
    return np.load(io.BytesIO(b))


def require_np_array(x: any) -> np.ndarray:
    assert isinstance(x, np.ndarray), f"Expected np.ndarray, got {type(x).__name__}"
    return x


def require_dtype(dtype, x: np.ndarray) -> np.ndarray:
    """Useful for protecting against e.g. float64 (default) when you intend to have float32"""
    dtype = np.dtype(dtype)  # Normalize dtype (for printing), and also fail fast if caller swapped the args
    if x.dtype != dtype:
        raise ValueError(f'Expected {dtype}, got {x.dtype}')
    return x


def warn_unless_dtype(dtype, x: np.ndarray) -> np.ndarray:
    """Useful for protecting against e.g. float64 (default) when you intend to have float32"""
    dtype = np.dtype(dtype)  # Normalize dtype (for printing), and also fail fast if caller swapped the args
    if x.dtype != dtype:
        log.warn(f'Expected {dtype}, got {x.dtype}')
    return x


#
# scipy
#

import numpy as np
import scipy.fftpack as fft


def bandpass_filter(
    x: np.ndarray,
    sample_rate: int,
    lo_hz: int = None,
    hi_hz: int = None,
) -> np.ndarray:
    f = fft.rfft(x)  # f.shape == x.shape
    f_ix = fft.rfftfreq(x.size, d=1 / sample_rate)  # freq indexes, i.e. f[i] is the power of freq f_ix[i]
    f[
        (f_ix < lo_hz if lo_hz is not None else False) |
        (f_ix > hi_hz if hi_hz is not None else False)
    ] = 0
    return fft.irfft(f).astype(x.dtype)


#
# sqlalchemy
#

from contextlib import contextmanager

import sqlalchemy as sqla


@contextmanager
def sqla_oneshot_eng_conn_tx(db_url: str, **engine_kwargs):
    """Create a db engine, a connection to it, and run a single transaction, tearing down everything when done"""
    eng = sqla.create_engine(db_url, **engine_kwargs)
    try:
        # eng.begin() is a tx bracket
        #   - https://docs.sqlalchemy.org/en/latest/core/connections.html#using-transactions
        #   - https://docs.sqlalchemy.org/en/latest/core/connections.html#sqlalchemy.engine.Engine.begin
        #   - (i.e. no autocommit: https://docs.sqlalchemy.org/en/latest/core/connections.html#using-transactions)
        with eng.begin() as conn:
            yield conn
    finally:
        # Release eng conn pools
        #   - https://docs.sqlalchemy.org/en/latest/core/connections.html#engine-disposal
        #   - https://docs.sqlalchemy.org/en/latest/core/connections.html#sqlalchemy.engine.Engine.dispose
        eng.dispose()


#
# pandas
#

from collections import OrderedDict
import pickle
import tempfile
import time
from typing import *
import uuid

from dataclasses import dataclass
from frozendict import frozendict
from more_itertools import last, one
import pandas as pd
from potoo.pandas import df_flatmap


Column = Iterable
Row = pd.Series


def df_inspect(df, *xs: any):
    for x in xs:
        if hasattr(x, '__call__'):
            x = x(df)
        if isinstance(x, str):
            # print(x)
            display({'text/plain': x}, raw=True)  # display() instead of print() to match flush behavior
        else:
            if not isinstance(x, tuple):
                x = (x,)
            display(*x)  # Less reliable flush, e.g. for single-line strs (which don't make it here), and maybe others...
            # ipy_print(*x)  # Forces text/plain instead of text/html (e.g. df colors and spacing)
    return df


def df_with_totals(df, **kwargs):
    return (df
        .pipe(df_with_totals_col, **kwargs)
        .pipe(df_with_totals_row, **kwargs)
    )


def df_with_totals_col(df, **kwargs):
    return df.assign(total=lambda df: df.sum(axis=1, **kwargs))


def df_with_totals_row(df, **kwargs):
    return (df
        .append(df.sum(axis=0, **kwargs).to_frame().T)
        [df.columns]  # Preserve col order
    )


def df_flatmap_list_col(df, col_name, col_f=lambda s: s):
    return (df
        .assign(**{col_name: col_f(df[col_name])})
        .pipe(df_flatmap, lambda row: [
            OrderedDict({**row, col_name: x})
            for x in row[col_name]
        ])
    )


def df_checkpoint(
    df: pd.DataFrame,
    path: str,
    # Manually surface common kwargs
    engine='fastparquet',  # .to_parquet + .read_parquet
    compression='uncompressed',  # .to_parquet
    # And provide a way to pass arbitrary kwargs we didn't handle above
    to_parquet=dict(),
    read_parquet=dict(),
) -> pd.DataFrame:
    """
    Checkpoint a df to (parquet) file
    - Like ddf_checkpoint. Intended not for performance control like with ddfs, but for api simplicity with dfs.
    """
    df.to_parquet(path, engine=engine, compression=compression, **to_parquet)
    return pd.read_parquet(path, engine=engine, **read_parquet)


@dataclass
class box:
    """Useful e.g. for putting iterables inside pd.Series/np.array"""
    unbox: any

    @classmethod
    def many(cls, xs: Iterable['X']) -> Iterable['box[X]']:
        return [cls(x) for x in xs]


def unbox(x: 'box[X]') -> 'X':
    return x.unbox


def unbox_many(xs: Iterable['box[X]']) -> Iterable['X']:
    return [x.unbox for x in xs]


#
# matplotlib
#

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


#
# plotnine
#

from plotnine import *

# FIXME Default theme_gray() plots non-transparent bg, but theme_minimal() reverts back to transparent bg
theme_minimal_white = lambda *args, **kwargs: theme_minimal(*args, **kwargs) + theme(plot_background=element_rect('white'))


#
# sklearn
#

from functools import partial, reduce, singledispatch
import re
from typing import *

from more_itertools import flatten, one
from potoo.numpy import np_sample
from potoo.pandas import df_ordered_cat, df_reorder_cols
import sklearn as sk
import sklearn.ensemble
import sklearn.linear_model
import sklearn.multiclass
import sklearn.pipeline

X = TypeVar('X')


class DataclassEstimator(sk.base.BaseEstimator, DataclassConfig):
    """
    Manage estimator params as dataclass fields
    - http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
    """

    def get_params(self, deep=True):
        return self.config

    def set_params(self, **params):
        self.__dict__.update(params)
        return self

    @property
    def config_id(self) -> str:
        """
        Model config str id, e.g. for persistence
        - Sensitive to field ordering
        """
        return ','.join(f'{k}={v}' for k, v in self.config.items())


@singledispatch
def model_stats(model: Optional[sk.base.BaseEstimator], **kwargs) -> pd.DataFrame:
    return pd.DataFrame([
        dict(type=f'unknown:{type(model).__name__}'),
    ])


@model_stats.register(sk.pipeline.Pipeline)
def _(pipeline) -> pd.DataFrame:
    return model_stats(pipeline.steps[-1][-1])


@model_stats.register(sk.multiclass.OneVsRestClassifier)
def _(ovr, **kwargs) -> pd.DataFrame:
    return (
        pd.concat(ignore_index=True, objs=[
            model_stats(estimator, **kwargs)
            .assign(
                type=lambda df: 'ovr/' + df.type,
                n_classes=len(ovr.classes_),
                class_=class_,
            )
            for class_, estimator in zip(ovr.classes_, ovr.estimators_)
        ])
        .pipe(df_reorder_cols, first=['type', 'n_classes', 'class_'])
    )


@model_stats.register(sk.linear_model.LogisticRegression)
def _(logreg) -> pd.DataFrame:
    return (
        pd.DataFrame(
            dict(
                type='logreg',
                class_=class_,
                n_iter=n_iter,
            )
            for class_, n_iter in (
                zip(logreg.classes_, logreg.n_iter_) if logreg.n_iter_.shape == logreg.classes_.shape else
                [('*', one(logreg.n_iter_))]
            )
        )
        .pipe(df_reorder_cols, first=['type', 'class_', 'n_iter'])
    )


# TODO
# @model_stats.register(sk.linear_model.LogisticRegressionCV)
# def _(logreg_cv) -> pd.DataFrame:
#     pass


@model_stats.register(sk.linear_model.SGDClassifier)
def _(sgd) -> pd.DataFrame:
    return pd.DataFrame([OrderedDict(
        type='sgd',
        n_iter=sgd.n_iter_,
    )])


@model_stats.register(sk.ensemble.forest.BaseForest)
def _(forest) -> pd.DataFrame:
    return (
        pd.DataFrame(_tree_stats(tree.tree_) for tree in forest.estimators_)
        .reset_index().rename(columns={'index': 'tree_i'})
        .assign(
            type=lambda df: 'forest/' + df.type,
            n_trees=len(forest.estimators_),
        )
        .pipe(df_reorder_cols, first=['type', 'n_trees', 'tree_i'])
    )


@model_stats.register(sk.tree.tree.BaseDecisionTree)
def _(tree, **kwargs) -> pd.DataFrame:
    return model_stats(tree.tree_, **kwargs)


@model_stats.register(sk.tree._tree.Tree)
def _(tree) -> pd.DataFrame:
    return pd.DataFrame([_tree_stats(tree)])


# Expose a single dict of stats so that the BaseForest impl isn't bottlenecked by constructing O(n) 1-row dfs
def _tree_stats(tree: sk.tree._tree.Tree) -> OrderedDict:
    return OrderedDict(
        type='tree',
        leaf_count=(tree.children_left == -1).sum(),
        fork_count=(tree.children_left != -1).sum(),
        depth=tree_depth(tree),
        # Useful
        max_depth=tree.max_depth,  # Seems to always agree with tree_depth()
        node_count=tree.node_count,  # = num_leaves + num_internal
        # Include these in case we find a use
        capacity=tree.capacity,  # Seems to always agree with node_count
        max_n_classes=tree.max_n_classes,
        # 'n_classes',  # Excluding because it's an np.array (seems to be np.array([max_n_classes])?)
        n_features=tree.n_features,
        n_outputs=tree.n_outputs,  # 1 unless multi-label
    )


def tree_depth(tree: sk.tree._tree.Tree) -> int:
    return tree_node_depths(tree).max()


def tree_node_depths(tree: sk.tree._tree.Tree) -> 'np.ndarray[int]':
    depths = np.full(tree.node_count, -1)
    def f(node, depth):
        depths[node] = depth
        if tree.children_left[node] != -1:
            f(tree.children_left[node], depth + 1)
            f(tree.children_right[node], depth + 1)
    f(0, 0)
    assert -1 not in depths  # Else we failed to visit some node(s) in our traverse
    return depths


def combine_ensembles(ensembles: Iterable[sk.ensemble.BaseEnsemble]) -> sk.ensemble.BaseEnsemble:
    for x, y in zip(ensembles, ensembles[1:]):
        assert x.get_params() == y.get_params(), f'Not all ensemble params match: {x.get_params()} != {y.get_params()}'
    return rebuild_ensemble(
        ensembles[0],
        lambda _: list(flatten(e.estimators_ for e in ensembles)),
    )


def sample_ensemble(ensemble: sk.ensemble.BaseEnsemble, **np_sample_kwargs) -> sk.ensemble.BaseEnsemble:
    return rebuild_ensemble(ensemble, partial(np_sample, **np_sample_kwargs))


def rebuild_ensemble(
    ensemble: sk.ensemble.BaseEnsemble,
    estimators_f: Callable[[List[sk.base.BaseEstimator]], Iterable[sk.base.BaseEstimator]],
) -> sk.ensemble.BaseEnsemble:
    # Based on:
    #   - https://stackoverflow.com/a/25925913/397334
    #   - https://github.com/pydata/pyrallel/blob/master/pyrallel/ensemble.py
    estimators_ = list(estimators_f(ensemble.estimators_))
    ensemble = sk.clone(ensemble)
    ensemble.estimators_ = estimators_
    ensemble.n_estimators = len(ensemble.estimators_)
    return ensemble


def cv_results_splits_df(cv_results: Mapping[str, list]) -> pd.DataFrame:
    """Tidy the per-split facts from a cv_results_ (e.g. GridSearchCV.cv_results_)"""
    return (
        # box/unbox to allow np.array/list values inside the columns [how to avoid?]
        pd.DataFrame({k: list(map(box, v)) for k, v in cv_results.items()}).applymap(lambda x: x.unbox)
        [lambda df: [c for c in df if re.match(r'^split|^param_|^estimator$', c)]]
        .pipe(df_flatmap, lambda row: ([
            (row
                .filter(regex=r'^param_|^split%s_|^estimator$' % fold)
                .rename(lambda c: c.split('split%s_' % fold, 1)[-1])
                .set_value('fold', fold)
            )
            for fold in {
                int(m.group('fold'))
                for c in row.index
                for m in [re.match('^split(?P<fold>\d+)_', c)]
                if m
            }
        ]))
        .reset_index(drop=True)
        .rename(columns={'estimator': 'model'})
        .assign(params=lambda df: df.apply(axis=1, func=lambda row: ', '.join([
            '%s[%s]' % (strip_startswith(k, 'param_'), row[k])
            for k in reversed(row.index)
            if k.startswith('param_')
        ])))
        .assign(
            model_id=lambda df: df.apply(axis=1, func=lambda row: (
                '%s, fold[%s]' % (row.params, row.fold)
            )),
            # model_stats=lambda df: df.model.map(model_stats),  # XXX Now done by sk_hack._fit_and_score_cached
        )
        .pipe(lambda df: df_reorder_cols(df,
            first=list(flatten([
                ['model_id'],
                [c for c in df if re.match(r'^param_', c)],
                ['fold', 'train_score', 'test_score'],
            ])),
            last=[
                'model',
                # 'model_stats',  # XXX Now done by sk_hack._fit_and_score_cached
            ],
        ))
        .pipe(df_ordered_cat,
            model_id=lambda df: df.model_id.unique(),
            params=lambda df: df.params.unique(),
        )
    )


# TODO Sync with updated cv_results_splits_df. Here's the plot I used to use with this:
#
#   with warnings.catch_warnings():
#       # TODO How to avoid the 'color' legend warning since geom_col has no color mapping?
#       warnings.simplefilter(action='ignore', category=UserWarning)
#       repr(cv_results_summary_df(cv.cv_results_)
#           .filter(regex='^(param_.*|.*_time$)')
#           .pipe(df_flatmap, lambda row: [
#               row.set_value('stage', 'fit').rename({'mean_fit_time': 'mean_time', 'std_fit_time': 'std_time'}),
#               row.set_value('stage', 'score').rename({'mean_score_time': 'mean_time', 'std_score_time': 'std_time'}),
#           ])
#           .filter(regex='^(param.*|stage|mean_time|std_time)$')
#           .pipe(df_ordered_cat, 'param_classifier', transform=reversed)
#           .pipe(ggplot, aes(x='param_classifier'))
#           + geom_col(aes(y='mean_time', fill='stage'), position=position_dodge())
#           + geom_linerange(
#               aes(group='stage', y='mean_time', ymin='mean_time - 1.96 * std_time', ymax='mean_time + 1.96 * std_time'),
#               position=position_dodge(width=.9),
#           )
#           + coord_flip()
#           + scale_color_cmap_d(mpl.cm.Set1)
#           # + theme(legend_position='bottom')
#           + theme_figsize(width=4, aspect_ratio=2/1)
#           + ggtitle('Train/test runtimes (WARNING: mix of cached and uncached)')
#       )
#
def cv_results_summary_df(cv_results: Mapping[str, list]) -> pd.DataFrame:
    """Tidy the per-params facts from a cv_results_ (e.g. GridSearchCV.cv_results_)"""
    return (
        # box/unbox to allow np.array/list values inside the columns
        pd.DataFrame({k: list(map(box, v)) for k, v in cv_results.items()}).applymap(lambda x: x.unbox)
        [lambda df: [c for c in df.columns if not re.match(r'^split|^params$', c)]]
        [lambda df: list(flatten([
            [c for c in df if c.startswith('param_')],
            [c for c in df if not c.startswith('param_') and '_train_' in c],
            [c for c in df if not c.startswith('param_') and '_test_' in c],
            [c for c in df if not c.startswith('param_') and c.endswith('_fit_time')],
            [c for c in df if not c.startswith('param_') and c.endswith('_score_time')],
        ]))]
    )


# TODO Useful? I usually just want the splits_df, rarely the summary_df...
def cv_results_dfs(cv_results: Mapping[str, list]) -> (pd.DataFrame, pd.DataFrame):
    """Tidy dfs from a cv_results_ (e.g. GridSearchCV.cv_results_)"""
    return (
        cv_results_summary_df(cv_results),
        cv_results_splits_df(cv_results),
    )


def _map_progress_joblib(
    f: Callable[[X], X],
    xs: Iterable[X],
    desc: str = None,  # TODO Display somewhere (like dask + tqdm)
    n=None,  # Unused (required to implement)
    use_joblib=True,
    backend='threading',  # 'threading' | 'multiprocessing' [FIXME 'multiprocessing' is slow/dead with timeout errors]
    **kwargs,
) -> Iterable[X]:
    if not use_joblib:
        return list(map(f, xs))
    else:
        return joblib.Parallel(
            backend=backend,
            **kwargs,
        )(joblib.delayed(f)(x) for x in xs)


#
# dask
#

from pathlib import Path
import multiprocessing
from typing import *

from attrdict import AttrDict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from potoo.util import AttrContext, get_cols

X = TypeVar('X')


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
@dataclass
class _dask_opts(AttrContext):
    # FIXME When nested, winner is inner instead of outer, which is surprising
    override_use_dask: bool = None
    override_scheduler: bool = None


# Workaround for @singleton (above)
dask_opts = _dask_opts()


def dask_progress(**kwargs):
    """Context manager to show progress bar for dask .compute()"""

    from dask.diagnostics.progress import ProgressBar, format_time
    from dask.utils import ignoring
    class NamedProgressBar(ProgressBar):
        """
        Extend dask.diagnostics.ProgressBar:
        - Automatically calculate width from terminal $COLUMNS
        - Optional desc, to print as a prefix so you can distinguish your progress bars
        """

        def __init__(self, desc=None, n=None, **kwargs):
            super().__init__(**kwargs)
            self._terminal_cols = get_cols()
            self._n = n
            self._desc_pad = '%s: ' % desc if desc else ''
            # self._desc_pad = '%s:' % desc if desc else ''

        def _draw_bar(self, frac, elapsed):
            # (Mostly copy/pasted from ProgressBar)
            percent = int(100 * frac)
            elapsed = format_time(elapsed)
            msg_prefix = '\r%s[' % self._desc_pad
            # msg_prefix = '\r[%s' % self._desc_pad
            n = '(%s)' % self._n if self._n else ''
            msg_suffix = '] | %3s%% %s | %s' % (percent, n, elapsed)
            width = self._terminal_cols - len(msg_prefix) - len(msg_suffix)
            bar = '#' * int(width * frac)
            msg_bar = '%%-%ds' % width % bar
            with ignoring(ValueError):
                self._file.write(msg_prefix)
                self._file.write(msg_bar)
                self._file.write(msg_suffix)
                self._file.flush()

    return NamedProgressBar(**{
        # 'width': get_cols() - 30,
        **kwargs,
    })


def _map_progress_dask(
    f: Callable[[X], X],
    xs: Iterable[X],
    desc: str = None,
    n: int = None,
    use_dask=True,
    scheduler='threads',  # 'processes' | 'threads' | 'synchronous' [FIXME 'processes' hangs before forking]
    partition_size=None,
    npartitions=None,
    get_kwargs=None,  # TODO Sane default for num_workers (i.e. num procs) when scheduler='processes'
    **kwargs,
) -> Iterable[X]:
    use_dask = coalesce(dask_opts.override_use_dask, use_dask)
    scheduler = coalesce(dask_opts.override_scheduler, scheduler)
    if not use_dask:
        return list(map(f, xs))
    else:
        from dask import delayed
        import dask.bag
        # HACK dask.bag.from_sequence([pd.Series(...), ...]) barfs -- workaround by boxing it
        # HACK dask.bag.from_sequence([np.array(...), ...]) flattens the arrays -- workaround by boxing it
        # HACK Avoid other cases we haven't tripped over yet by boxing everything unconditionally
        wrap, unwrap = (lambda x: box(x)), (lambda x: x.unbox)
        with dask_progress(desc=desc, n=n):
            if not partition_size and not npartitions:
                xs = list(xs)
                if len(xs) == 1:
                    # Avoid bag.from_sequence for singleton xs (e.g. from one_progress), since if that one element is
                    # big (e.g. a big df) then bag will pickle it to compute the dask task id, which is often slow
                    #   - TODO Is delayed still doing tokenize/pickle? Investigate name/dask_key_name (see delayed docstring)
                    return [delayed(f)(xs[0]).compute(get=dask_get_for_scheduler('synchronous'))]
                elif len(xs) <= 100:
                    # Don't probe f(xs[0]) for small inputs, since they might be slow and npartitions=len(xs) is
                    # unlikely to bottleneck us on dask task-coordination overhead
                    npartitions = len(xs)
                else:
                    (unit_sec, _) = timed(lambda: list(map(f, xs[:1])))
                    npartitions = _npartitions_for_unit_sec(len(xs), unit_sec, **kwargs)
            return (dask.bag
                .from_sequence(map(wrap, xs), partition_size=partition_size, npartitions=npartitions)
                .map(unwrap)
                .map(f)
                .compute(get=dask_get_for_scheduler(scheduler, **(get_kwargs or {})))
            )


def _npartitions_for_unit_sec(n: int, unit_sec: float, target_sec_per_partition=2, min_npartitions_per_core=5) -> int:
    n_cores = multiprocessing.cpu_count()
    npartitions = int(n * unit_sec / target_sec_per_partition)  # Estimate from unit_sec
    npartitions = round(npartitions / n_cores) * n_cores  # Round to multiple of n_cores
    npartitions = max(npartitions, n_cores * min_npartitions_per_core)  # Min at n_cores * k (for small k)
    return npartitions


# Mimic http://dask.pydata.org/en/latest/scheduling.html
def dask_get_for_scheduler(scheduler: str, **kwargs):
    if isinstance(scheduler, str):
        import dask
        import dask.multiprocessing
        get = {
            'synchronous': dask.get,
            'threads': dask.threaded.get,
            'processes': dask.multiprocessing.get,
        }[scheduler]
    else:
        get = scheduler
    return partial(get, **kwargs)


def ddf_divisions_for_dtype(dtype: 'dtype-like', npartitions: int) -> np.ndarray:
    dtype = np.dtype(dtype)
    return list(np.linspace(
        start=np.iinfo(dtype).min,
        stop=np.iinfo(dtype).max,
        num=npartitions + 1,
        endpoint=True,
        dtype=dtype,
    ))


def ddf_checkpoint(
    ddf: 'dd.DataFrame',
    path: str,
    # TODO Think harder: bad interactions with upstream .compute()'s (e.g. checkpoints without resume, set_index without divisions)
    #   - Addressing this would require a heavier refactor in the user's pipeline, since they'd have to wrap entire
    #     prefixes of the pipeline -- and composing these would increase indentation in an awkward way...
    resume_from_checkpoint=False,
    # Manually surface common kwargs
    engine='auto',  # .to_parquet + .read_parquet
    compression='default',  # .to_parquet
    get=None,  # .compute
    # And provide a way to pass arbitrary kwargs we didn't handle above
    to_parquet=dict(),
    compute=dict(),
    read_parquet=dict(),
) -> 'dd.DataFrame':
    """
    Checkpoint a ddf to (parquet) file
    - Useful to persist the prefix of a computation before a multi-pass operation like .set_index(divisions=None), which
      needs to force the input ddf once to calculate divisions and then again to do the shuffle
    """
    import dask.dataframe as dd
    if resume_from_checkpoint and dask_parquet_file_exists(path):
        log.warn(f'Resuming from checkpoint: {path}')
    else:
        (ddf
            .to_parquet(path, compute=False, engine=engine, compression=compression, **to_parquet)
            .compute(get=get, **compute)
        )
    return dd.read_parquet(path, engine=engine, **read_parquet)


def dask_parquet_file_exists(url: str) -> bool:
    """Test if a parquet file exists, allowing any dask-friendly url"""
    import dask.dataframe as dd
    try:
        # This will read the full <url>/_metadata file, which could be big for a parquet file with many parts
        #   - TODO Get our hands directly on dask's fs module so we can do a more lightweight .exists()
        #   - But match dask's dd.read_parquet(url) behavior, e.g. treat inputs 'foo.parquet' + 'foo.parquet/_metadata' the same
        dd.read_parquet(url)
    except Exception:
        return False
    else:
        return True


def dd_read_parquet_sample(
    path: str,
    sample: Union[float, List[float]] = None,
    sample_divisions=None,
    sample_npartitions=None,
    sample_repartition_force=False,
    sample_to_parquet=dict(compression='gzip'),
    sample_get=None,  # Default: dask_get_for_scheduler('threads')
    random_state=0,
    **kwargs,
) -> 'dd.DataFrame':
    """
    dd.read_parquet with sampling, to make it easy to downsample large files for faster dev iteration
    - Same as dd.read_parquet(path, **kwargs) if sample isn't given
    """
    import dask.dataframe as dd
    if sample:
        if isinstance(sample, float):
            sample = [sample]
        in_path = path_to_sampled_path(path, *sample[:-1])
        out_path = path_to_sampled_path(path, *sample)
        if not (Path(out_path) / '_metadata').exists():
            log.info('Caching sample: %s <- %s' % (out_path, in_path))
            # Read and sample
            ddf = (
                dd.read_parquet(in_path, **kwargs)
                .sample(frac=sample[-1], replace=False, random_state=random_state)
            )
            # Repartition, if requested
            if sample_divisions or sample_npartitions:
                ddf = ddf.repartition(
                    divisions=sample_divisions,
                    npartitions=sample_npartitions,
                    force=sample_repartition_force,
                )
            # Write cached sample
            #   - Use 'threads' if repartitioning, else 'processes', to avoid ipc bottlenecks from the shuffle
            sample_get = sample_get or dask_get_for_scheduler('threads' if sample_npartitions else 'processes')
            (ddf
                .to_parquet(out_path, **sample_to_parquet, compute=False)
                .compute(get=sample_get)
            )
        # log.debug('Reading cached sample: %s' % out_path)
        path = out_path
    return dd.read_parquet(path, **kwargs)


# (Not dask specific)
def path_to_sampled_path(path: str, *sample: float) -> str:
    return '-'.join([path, *map(str, sample)])


#
# progress (tdqm/dask/joblib)
#

import types
from typing import *

from dataclasses import dataclass, field
import joblib
from potoo.util import AttrContext
from tqdm import tqdm

from config import config

X = TypeVar('X')


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
@dataclass
class _progress_kwargs(AttrContext):
    default:  Optional[dict] = field(default_factory=lambda: config.progress_kwargs.get('default'))
    override: Optional[dict] = field(default_factory=lambda: config.progress_kwargs.get('override'))


# Workaround for @singleton (above)
progress_kwargs = _progress_kwargs()


def df_map_rows_progress(
    df: pd.DataFrame,
    f: Callable[['Row'], 'Row'],
    **kwargs,
) -> pd.DataFrame:
    return pd.DataFrame(
        index=df.index,  # Preserve index
        data=map_progress(
            f=lambda row: f(row),
            xs=(row for i, row in df.iterrows()),
            n=len(df),
            **kwargs,
        ),
    )


# TODO Think of a less confusing name?
def map_progress_df_rows(
    df: pd.DataFrame,
    f: Callable[['Row'], X],
    **kwargs,
) -> Iterable[X]:
    return map_progress(
        f=f,
        xs=df_rows(df),
        n=len(df),
        **kwargs,
    )


def map_progress(
    f: Callable[[X], X],
    xs: Iterable[X],
    desc: str = None,
    n: int = None,
    **_progress_kwargs,
) -> Iterable[X]:

    # Defaults
    if not n and hasattr(xs, '__len__'):
        n = len(xs)

    # Resolve progress_kwargs: .override -> **_progress_kwargs -> .default -> fallback default
    kwargs = (
        progress_kwargs.override or
        _progress_kwargs or
        progress_kwargs.default or
        dict(use='sync')  # Fallback default
    )
    kwargs = dict(kwargs)  # Copy so we can mutate

    # Delegate as per `use`
    use = kwargs.pop('use')
    return ({
        None: partial(_map_progress_sync, use_tqdm=False),
        'sync': _map_progress_sync,
        'dask': _map_progress_dask,
        'joblib': _map_progress_joblib,
    }[use])(
        f=f, xs=xs, desc=desc, n=n,
        **kwargs,
    )


def _map_progress_sync(
    f: Callable[[X], X],
    xs: Iterable[X],
    **kwargs,
) -> Iterable[X]:
    return list(_iter_progress_sync(
        map(f, xs),
        **kwargs,
    ))


def _iter_progress_sync(
    xs: Iterator[X],
    desc: str = None,
    n: int = None,
    use_tqdm=True,
) -> Iterator[X]:
    if use_tqdm:
        return tqdm(xs, total=n, desc=desc)
    else:
        return xs


# XXX iter_progress is incompat with dask-style par, which must eagerly unroll (cheap) xs to compute (heavy) map(f, xs)
# def iter_progress(xs: Iterator[X], **kwargs) -> Iterable[X]:
#     return map_progress(f=lambda x: x(), xs=(lambda x=x: x for x in xs), **kwargs)


def one_progress(
    f: Callable[[], X],
    **kwargs,
) -> X:
    """Draw a progress bar for a single item, for consistency with surrounding (many-element) progress bars"""

    # Don't default to dask/threads, since we don't benefit from threading overhead with one task
    if kwargs.get('use') == 'dask':
        kwargs.setdefault('scheduler', 'synchronous')

    [x] = map_progress(f=lambda _: f(), xs=[None], **kwargs)
    return x


#
# statsmodels
#

import statsmodels.api as sm
import statsmodels.formula.api as smf


def lm(*args, **kwargs):
    return smf.ols(*args, **kwargs).fit()


#
# PIL
#

import io

import PIL


def pil_img_save_to_bytes(img: PIL.Image.Image, format=None, **params) -> bytes:
    return bytes_from_file_write(lambda f: (
        img.save(f, format=format, **params)
    ))


def pil_img_open_from_bytes(b: bytes) -> PIL.Image.Image:
    return PIL.Image.open(io.BytesIO(b))  # (Infers format)


#
# audiosegment
#

import copy
import json
from typing import *

import audiosegment
import numpy as np
import pydub


def audio_eq(a: audiosegment.AudioSegment, b: audiosegment.AudioSegment) -> audiosegment.AudioSegment:
    """Equality on AudioSegment that takes metadata into account, unlike builtin == (surprise!)"""
    return all([
        a.name == b.name,
        a.sample_width == b.sample_width,
        a.frame_rate == b.frame_rate,
        a.channels == b.channels,
        a._data == b._data,
    ])


def audio_copy(audio: audiosegment.AudioSegment) -> audiosegment.AudioSegment:
    # (Surpringly hard to get right)
    audio_dict = audio.__dict__
    audio = copy.copy(audio)
    # [XXX No more nonstandard attrs woo! They never pickled correctly anyway!]
    # audio.__dict__.update(audio_dict)  # Copy non-standard attrs
    audio.seg = copy.copy(audio.seg)
    return audio


def audio_replace(audio: audiosegment.AudioSegment, **kwargs) -> audiosegment.AudioSegment:
    audio = audio_copy(audio)
    for k, v in kwargs.items():
        setattr(audio, k, v)
    return audio


def audio_with_numpy_array(
    audio: audiosegment.AudioSegment,
    f: Callable[[np.ndarray], np.ndarray],
) -> audiosegment.AudioSegment:
    return audio_replace(
        audio=audiosegment.from_numpy_array(
            nparr=f(audio.to_numpy_array()),
            framerate=audio.frame_rate,
        ),
        name=audio.name,
    )


def audio_hash(audio: audiosegment.AudioSegment) -> str:
    """
    A unique content-based id for an audiosegment.AudioSegment
    - Useful e.g. for caching operations on sliced audios, which don't otherwise have a reliable key
    """
    # Do hash(json(x)), since json(x) is typically way faster than pickle(x)
    return sha1hex(
        # str fields
        #   - 1-1 because json is delimited, and we name each field
        #   - Well defined because we sort the fields by name
        json_dumps_canonical(dict(
            name=audio.name,
            sample_width=audio.seg.sample_width,
            frame_rate=audio.seg.frame_rate,
            channels=audio.seg.channels,
        )),
        # bytes fields
        #   - 1-1 because there's only one bytes field (else the boundary would need to be delimited, somehow)
        audio.seg._data,
    )


def audio_concat(audios: audiosegment.AudioSegment, name: str = None) -> audiosegment.AudioSegment:
    """Concat audios safely, unlike builtin + which mutates (surprise!)"""

    # Preserve metadata from first audio
    [audio, *_] = audios
    ret = audio[:0]

    # Use given name, else concat (nonzero) names
    if name is not None:
        names = [name]
    else:
        names = [a.name for a in audios if len(a) > 0]
    ret.name = ' + '.join(names)

    # Concat _data (bytes)
    ret.seg._data = b''.join(
        a.seg._data for a in audios
    )

    return ret


def audio_pad(audio: audiosegment.AudioSegment, duration_s: float) -> audiosegment.AudioSegment:
    """Pad audio with duration_s of silence"""
    return audio_concat([audio, audiosegment.silent(
        duration=duration_s * 1000,
        frame_rate=audio.frame_rate,
    )])


def audio_bandpass_filter(audio: audiosegment.AudioSegment, **kwargs) -> audiosegment.AudioSegment:
    """Filter audio by a frequency band (bandpass filter)"""
    return audio_with_numpy_array(audio, lambda x: bandpass_filter(x, audio.frame_rate, **kwargs))


#
# bubo-features
#

import base64
import platform
import secrets
import textwrap
import types
from typing import *
import urllib.parse

import numpy as np
import matplotlib.pyplot as plt
import parse
from potoo.ipython import *
from potoo.pandas import *
from potoo.util import path_is_contained_by
import psutil
import pydub
import scipy
import yaml

from config import config
from constants import *
from datatypes import Audio


def print_sys_info():
    """Useful to put at the top of notebooks so you know which system it ran on (e.g. local dev vs. many-core remote)"""
    print(
        ''.join([
            yaml.safe_dump(default_flow_style=False, data=dict(
                platform=dict(platform.uname()._asdict()),
            )),
            yaml.safe_dump(default_flow_style=False, data=dict(
                cpu=psutil.cpu_count(),
                mem='%.0fg' % (psutil.virtual_memory().total / 1024**3),
                swap='%.0fg' % (psutil.swap_memory().total / 1024**3),
            )),
        ])
        .rstrip()
    )


def rec_str_line(rec, *_first, first=[], last=[], default=[
    'audio_id',
    ('recorded_at', lambda x: x.isoformat()),
    'species',
    ('duration_s', '%.1fs'),
]) -> str:
    """Interactive shorthand"""
    rec = rec.copy()
    rec['audio_id'] = rec.audio_id if 'audio_id' in rec else rec.name if isinstance(rec.name, str) else 'NO_AUDIO_ID'
    strs = []
    for field in [*_first, *first, *default, *last]:
        if not isinstance(field, tuple):
            field = (field, '%s')
        (k, f) = field
        if isinstance(f, str):
            f = lambda x, f=f: f % x
        strs.append(f(rec[k]))
    return '  '.join(strs)


def xc_rec_str_line(rec, *_first, first=[], last=[], default=[
    'xc_id',
    'species_subspecies',
    ('duration_s', '%.1fs'),
    'quality',
    'type',
    'country_locality',  # TODO country -> county_code
    # 'lat', 'lng',
    ('date', lambda x: x.date().isoformat()),
    'recordist_license_type',
]) -> str:
    """Interactive shorthand"""
    rec = rec.copy()
    # Ad-hoc formatting to make these easier to visually grok
    rec['xc_id'] = 'XC%s' % rec.get('xc_id', rec.name)  # Col else index
    rec['species_subspecies'] = '/'.join([rec.species, *([rec.subspecies] if rec.subspecies else [])])
    rec['country_locality'] = '/'.join([rec.country, *reversed(rec.locality.split(', '))])
    rec['recordist_license_type'] = '%s[%s]' % (rec.recordist, rec.license_type)
    return rec_str_line(rec, *_first, first=first, last=last, default=default)


def text_bar(
    size: float,
    max: float = None,
    norm: float = None,
    side: Union['left', 'right'] = 'left',
    full: str = '■',  # Good for drawing unbroken boxes
    empty: str = '—',
) -> str:
    """Draw a text bar (motivated by rendering distances in dfs)"""
    max = max or size
    if norm:
        size = size / max * norm
        max = norm
    fulls = full * int(round(size))
    empties = empty * int(round(max - size))
    if side == 'left':
        return fulls + empties
    else:
        return empties + fulls


# NOTE Thumbs are complete recs, so we can't just add a .thumb col to an existing recs...
def recs_thumb(recs, features, **kwargs) -> 'recs':
    return (recs
        .apply(axis=1, func=lambda rec: (
            rec_thumb(rec, features, **kwargs)
        ))
        # Restore col order
        .pipe(lambda df: df_reorder_cols(df, first=[c for c in recs.columns if c in df.columns]))
    )


def rec_thumb(*args, **kwargs) -> 'Recording':
    start_s, thumb = rec_thumb_with_start(*args, **kwargs)
    return thumb


def rec_thumb_with_start(
    rec,
    features,
    thumb_s=1,  # Duration of output thumb
    search_s=10,  # Duration of input rec to look for a thumb
    smooth_sigma_s=.25,  # σ of gaussian to use to smooth the audio signal
    verbose=False,  # Plot intermediate results
) -> (
    float,  # start_s
    'Recording',  # thumb_rec
):
    """Clip an informative thumbnail from a rec"""

    # Clip rec to prefix of duration search_s
    rec = features.slice_spectro(rec, 0, search_s)
    (f, t, S) = rec.spectro

    # Total power per time (x)
    x = S.sum(axis=0)

    # Smoothed power (y)
    i_per_s = 1 / (t[1] - t[0])  # Compute this consistently _across recs_, so we can e.g. grid thumbs together without hassle
    sigma_is = i_per_s * smooth_sigma_s
    y = scipy.ndimage.filters.gaussian_filter(x, sigma_is)

    # Find max smoothed power (highest mode)
    #   - Ignore the last thumb_s of y so we don't pick a partial thumb
    y[-int(thumb_s * i_per_s):] = 0
    max_i = np.argmax(y)

    # thumb is max smoothed power ± thumb_s/2
    thumb_start_s = max_i / i_per_s - thumb_s / 2
    thumb_start_s = np.clip(thumb_start_s, 0, rec.duration_s)  # Clip to a valid start_s
    thumb = features.slice_spectro(rec,
        thumb_start_s,
        thumb_start_s + thumb_s,
    )

    if verbose:
        plt.plot(x, color='grey')
        plt.plot(y, color='black')
        t = y.copy()
        thumb_start_i = int(thumb_start_s * i_per_s)
        t[np.arange(thumb_start_i)] = 0
        t[np.arange(thumb_start_i + int(i_per_s * thumb_s), len(t))] = 0
        plt.plot(t, color='red')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

    return (thumb_start_s, thumb)


def audio_abs_path(audio: Audio) -> Path:
    return audio_id_abs_path(audio.name)


def audio_id_abs_path(id: str) -> Path:
    """An Audio's path is its name relative to data_dir"""
    assert not Path(id).is_absolute()
    return Path(data_dir) / id


def strip_leading_cache_audio_dir(path: str) -> str:
    """Strip leading 'cache/audio/', if present"""
    rel_cache_audio_dir = Path(cache_audio_dir).relative_to(data_dir)
    if path_is_contained_by(path, rel_cache_audio_dir):
        path = str(Path(path).relative_to(rel_cache_audio_dir))
    return path


def audio_from_file_in_data_dir(path: str, **kwargs) -> Audio:
    """
    Like audio_from_file, except:
    - Allow input path to be relative to data_dir
    - Require input path to be under data_dir
    - Ensure output audio.name is relative to data, which all downstreams expect
    """
    if Path(path).is_absolute():
        assert path_is_contained_by(path, data_dir), f"path_is_contained_by({path!r}, {data_dir!r})"
    else:
        path = Path(data_dir) / path
    audio = audio_from_file(path, **kwargs)
    audio.name = str(Path(audio.name).relative_to(data_dir))
    assert not Path(audio.name).is_absolute()
    return audio


def audio_from_file(path: str, format: str = None, parameters: Iterable[str] = None, **kwargs) -> Audio:
    """
    Like audiosegment.from_file, except:
    - Interpret our .enc() audio ops to get the audio format, which we use in place of proper file extensions
    - Ensure we aren't passing format=None to pydub from_file (see below)
    - Ensure we are passing -nostdin to ffmpeg (see below)
    - Unpack ffmpeg error msgs from pydub exceptions, so that we can read them
    """

    # WARNING Don't pass format=None to pydub.AudioSegment.from_file if your file extension isn't real: things _mostly_
    # work, e.g. pydub/ffmpeg read the input file metadata to figure out the format, but then in another place pydub
    # uses the file ext to make a decision based on format, which (very quietly!) changes behavior, e.g.
    #   from_file(file='foo.mp3',      format=None) -> sample_fmt='fltp' and     is_format('mp3') -> pcm_s16le -> wav (bit=16)
    #   from_file(file='foo.enc(mp3)', format=None) -> sample_fmt='fltp' and not is_format('mp3') -> pcm_s32le -> wav (bit=32)
    format = format or audio_id_to_format(path)
    assert format, f"{format}"

    # WARNING Ensure `ffmpeg -nostdin` to avoid nondeterministic (!) bugs under `entr -r` (e.g. `bin/api-run-dev`),
    # where ffmpeg non-wav -> wav conversions return partial or no data, which causes pydub from_file to either
    # loudly fail, if no data from ffmpeg (the befuddling case), or _silently_ return _truncated_ audio, if partial data
    # from ffmpeg (the I-hope-you-noticed! case). This was pretty terrible to debug, so be-very-ware if you remove this
    # safeguard.
    #   - Docs for `ffmpeg -nostdin`: https://www.ffmpeg.org/ffmpeg-all.html#stdin-option
    #   - I guess caused by entr closing stdin on `-r`? https://bitbucket.org/eradman/entr/src/ee9e17b/entr.c#entr.c-403
    #   - ffmpeg error on no data: "Output file is empty, nothing was encoded"
    #   - pydub error on no data: "pydub.exceptions.CouldntDecodeError: Couldn't find data header in wav data"
    parameters = [
        *(parameters or []),
        '-nostdin',
    ]

    try:
        seg = pydub.AudioSegment.from_file(path, format=format, parameters=parameters, **kwargs)
    except Exception as e:

        # Parse ffmpeg error msgs out of pydub error msgs, else you're stuck with one big line of b'ffmpeg...\n...'
        if isinstance(e, pydub.exceptions.CouldntDecodeError):
            try:
                pydub_msg = e.args[0]
                (exit_code, ffmpeg_msg_as_bytes_repr) = parse.parse(
                    # Msg extracted from CouldntDecodeError(...) occurrences in pydub/audio_segment.py
                    'Decoding failed. ffmpeg returned error code: {:d}\n\nOutput from ffmpeg/avlib:\n\n{}',
                    pydub_msg,
                ).fixed
                ffmpeg_msg = eval(ffmpeg_msg_as_bytes_repr).decode()
                # Stop ignoring exceptions
            except:
                pass
            else:
                # Raise parsed exception
                raise type(e)(ffmpeg_msg)
        # Or if anything went wrong, re-raise original exception
        raise

    return audiosegment.AudioSegment(seg, path)


def recs_audio_persist(recs: 'RecordingDF', progress_kwargs=None, **kwargs) -> 'RecordingDF':
    return (recs
        .pipe(df_map_rows_progress, desc='rec_audio_persist',
            **(progress_kwargs or dict(use='dask', scheduler='threads')),
            f=partial(rec_audio_persist, **kwargs),
        )
    )


def rec_audio_persist(rec: 'Recording', **kwargs) -> 'Recording':
    assert rec.id == rec.audio.unbox.name
    rec = rec.copy()  # Copy so we can mutate
    rec.audio = box(audio_persist(rec.audio.unbox, **kwargs))
    if rec.audio.unbox:  # Skip if load_audio=False
        rec.id = rec.audio.unbox.name  # Propagate audio.name (= path) to rec.id
    return rec


def audio_persist(audio, load=None, load_audio=True, **audio_kwargs) -> Audio:
    """
    Code bottleneck to bridge from global utils (e.g. viz.plot_spectro, util.display_with_audio) into
    load.transcode_audio, which requires conjuring up a load instance, which we achieve in a hacky-but-acceptable way
    """
    # HACK Code smells abound. Probably some productive refactoring to do here, but low prio so far.

    # Conjure up a load instance, accepting overrides from the caller
    assert load not in [True, False],  "Oops, did you mean load_audio?"  # Confusing name collision
    if load is None:
        load = load_for_audio_persist(**audio_kwargs)

    # Transcode + persist
    #   - TODO Make Load._transcode_audio not private
    #   - FIXME This sometimes (^C?) creates (ffmpeg) and then barfs on (final audio_from_file_in_data_dir) empty files
    audio = load._transcode_audio(audio, load=load_audio)  # Confusing name collision
    if audio:  # Skip if load_audio=False
        assert audio_abs_path(audio).exists()
    return audio


def load_for_audio_persist(**audio_kwargs) -> 'Load':
    """The load instance for audio_persist, configured by config.audio.audio_persist"""
    # TODO Too decoupled from sg.load?
    #   - TODO Refactor modules to avoid util importing load, which creates an import cycle
    from load import Load  # Avoid cyclic imports
    return Load(**(audio_kwargs or config.audio.audio_persist.audio_kwargs))


def audio_add_ops(audio: Audio, *ops: str) -> Audio:
    """Add operation suffixes to the audio's id (= audio.name)"""
    return audio_replace(audio,
        name=audio_id_add_ops(audio.name, *ops),
    )


def audio_id_add_ops(id: str, *ops: str) -> str:
    """Add operation suffixes to an audio id (= audio.name)"""
    assert not isinstance(id, list), 'Oops, did you mean audio_id_add_ops(*id_ops)?'  # Catch an easy mistake

    # Simplify id (across both id and ops, else even basic simplifications won't work)
    input_id = id
    id = _audio_id_simplify(_audio_id_join_ops([id, *ops]))

    # If id changed then some downstream might write to it, so ensure it's properly housed under our cache/audio/ dir
    #   - If the new cache/audio/ path already exists, then assume it's semantically the same
    #   - If id == input_id, then assume no downstream will try to overwrite it (since it's semantically the same)
    if id != input_id:
        assert not Path(id).is_absolute()
        rel_cache_audio_dir = Path(cache_audio_dir).relative_to(data_dir)
        if not path_is_contained_by(id, rel_cache_audio_dir):
            id = str(rel_cache_audio_dir / id)

    return id


def _audio_id_join_ops(ops: List[str]) -> str:
    assert ops
    # Simplify the ops to avoid duplicated work
    ops = _audio_id_simplify_ops(ops)
    # Join the ops
    return '.'.join(ops)


def audio_id_split_ops(id: str) -> List[str]:
    """Split an audio id (= audio.name) into its operation parts"""
    # Simplification: conflate path vs. ext vs. ops, at the benefit of not having to invent a way to distinguish them
    #   - Property: _audio_id_join_ops(*audio_id_split_ops(id)) == id, if:
    #       - id ops are already simplified
    #   - Property: audio_id_add_ops(*audio_id_split_ops(id)) == id, if:
    #       - path_is_contained_by(id, rel_cache_audio_dir)
    #       - id ops are already simplified
    return id.split('.')


# XXX Make private, or something (not used)
def _audio_id_simplify(id: str) -> str:
    """Simplify an audio id to avoid duplicated work"""
    return _audio_id_join_ops(_audio_id_simplify_ops([*audio_id_split_ops(id)]))


def _audio_id_simplify_ops(ops: List[str]) -> List[str]:
    """Simplify audio id ops to avoid duplicated work"""

    def simplify_pair(a: 'op', b: 'op', A: 'op_type', B: 'op_type') -> Optional[List['op']]:
        """[*xs] if can simplify [a,b] -> [*xs], else None"""

        # PUNT Simplify .op().enc(wav) -> .op(), since Audio._data bytes in mem are always implicitly pcm/wav
        #   - Have
        #       - .enc(wav) anytime we write to file with format=wav
        #       - This is weird because .resample().enc(wav) "means" the same thing as .resample(), i.e. disk writes
        #         aren't semantically meaningful
        #   - Want
        #       - Add .enc(wav) only when it's semantically meaningful, i.e. only after reading a non-wav input
        #       - Simplification is currently the mechanism for rewriting id ops, and it doesn't currently know whether
        #         it's being used to construct the output file path or the input audio id, so achieving this would
        #         require introducing a way to distinguish those two uses
        #       - And we'd need to split audio id into two separate notions:
        #           - "File ids" like .enc(mp3) if load(format=mp3), and also .enc(wav) if load(format=wav)
        #           - "Memory ids" like .enc(mp3).enc(wav)
        #           - [Not totally clear to me...]
        #       - Any of which would require updating the various callers, which we don't have great test coverage for,
        #         so let's definitely, definitely punt on this for a while, because changing that area of the code is a
        #         pretty reliable way to burn up most of your week
        #   - Examples (have -> want)
        #       .wav                                      -> .wav
        #       .mp3.enc(wav)                             -> .mp3.enc(wav)
        #       .mp3.resample().enc(wav)                  -> .mp3.resample()
        #       .mp3.resample().enc(wav).slice().enc(wav) -> .mp3.resample().slice()
        #       .mp3.resample().enc(wav).slice().enc(mp3) -> .mp3.resample().slice().enc(mp3).enc(wav)

        # .wav.enc(wav) -> .wav
        #   - Can't do this with most input encodings, since we don't know the encoding params from the filename
        #   - But .wav's only encoding params are (hz,ch,bit), which we already control for
        if [a, b] == ['wav', 'enc(wav)']:
            return [a]

        # .enc(x).enc(x) -> .enc(x)
        #   - Assumes that all encodings are idempotent
        if [A, B] == ['enc', 'enc'] and a == b:
            return [a]

        # .resample(x).resample(x) -> .resample(x)
        if [A, B] == ['resample', 'resample'] and a == b:
            return [a]

        # .slice(p,q).slice(x,y) -> .slice(min(p+x,q),min(p+y,q))
        #   - Potentially unsafe because .slice().slice() means two slice operations have actually happeneded, different
        #     than e.g. .resample().resample() -> .resample() where our simplification informs the caller to skip the
        #     second resample, so let's omit this simplification until we have robust testing to validate that two slice
        #     operations actually produce the same output data as one combined slice operation
        #   - cf. the assert in Features._edit for another place affected by this decision
        # if [A, B] == ['slice', 'slice']:
        #     (p, q) = parse.parse('slice({:d},{:d})', a).fixed
        #     (x, y) = parse.parse('slice({:d},{:d})', b).fixed
        #     return ['slice(%d,%d)' % (min(p + x, q), min(p + y, q))]

    # Iterate until convergence
    ops = list(ops)
    prev_ops = None
    while ops != prev_ops:
        prev_ops = ops

        # Scan over each (contiguous) pair, delegating to simplify_pair
        #   - On first match, break and continue outer loop
        ops_with_types = list(zip(ops, (op.split('(')[0] for op in ops)))
        for i, ((a, A), (b, B)) in enumerate(zip(ops_with_types, ops_with_types[1:])):
            xs = simplify_pair(a, b, A, B)
            if xs is not None:
                ops = [*ops[:i], *xs, *ops[i + 2:]]  # Simplify [...,a,b,...] -> [...,*xs,...]
                break

    return ops


def audio_to_bytes(audio, **kwargs) -> bytes:
    audio = audio_persist(audio, **kwargs)
    return audio_id_to_bytes(audio.name)


def audio_id_to_bytes(id: str) -> bytes:
    with open(audio_id_abs_path(id), 'rb') as f:
        return f.read()


def audio_to_url(audio, url_type=None, **kwargs) -> str:
    audio = audio_persist(audio, **kwargs)
    if (url_type or config.audio.audio_to_url.url_type) == 'file':
        return 'file://%s' % urllib.parse.quote(str(audio_abs_path(audio)),
            safe='/,:()[] ',  # Cosmetic: exclude known-safe chars ('?' is definitely _not_ safe, not sure what else...)
        )
    elif (url_type or config.audio.audio_to_url.url_type) == 'data':
        return audio_bytes_to_data_url(
            audio_to_bytes(audio, **kwargs),
            mimetype=audio_to_mimetype(audio),
        )
    else:
        raise ValueError('Unexpected config.audio.audio_to_url.url_type: %s' % config.audio.audio_to_url.url_type)


def audio_bytes_to_data_url(audio_bytes: bytes, mimetype: str) -> str:
    return 'data:%(mimetype)s;base64,%(base64)s' % dict(
        mimetype=mimetype,
        base64=base64.b64encode(audio_bytes).decode('ascii'),
    )


def audio_to_html(audio, controls=None, preload=None, more_audio_attrs=None, **kwargs) -> str:
    audio = audio_persist(audio, **kwargs)
    return _audio_html(
        type=audio_to_mimetype(audio),
        src=audio_to_url(audio, **kwargs),
        controls=controls,
        preload=preload,
        more_audio_attrs=more_audio_attrs,
    )


def audio_bytes_to_html(audio_bytes: bytes, mimetype: str, **kwargs) -> str:
    return _audio_html(
        type=mimetype,
        src=audio_bytes_to_data_url(audio_bytes, mimetype),
        **kwargs,
    )


def _audio_html(type: str, src: str, controls=None, preload=None, more_audio_attrs=None) -> str:
    controls         = True   if controls         is None else controls
    preload          = 'none' if preload          is None else preload
    more_audio_attrs = ''     if more_audio_attrs is None else more_audio_attrs
    return dedent_and_strip('''
        <audio class="bubo-audio" %(controls)s preload="%(preload)s" %(more_audio_attrs)s>
            <source type="%(type)s" src="%(src)s" />
        </audio>
    ''') % dict(
        controls='controls' if controls else '',
        preload=preload,
        more_audio_attrs=more_audio_attrs,
        type=type,
        src=src,
    )


def audio_to_mimetype(audio: Audio) -> str:
    if not Path(audio_abs_path(audio)).exists():
        raise ValueError(f"audio must be persisted: {audio_abs_path(audio)}")
    return audio_id_to_mimetype(audio.name)


def audio_id_to_mimetype(id: str) -> str:
    return format_to_mimetype(audio_id_to_format(id))


def format_to_mimetype(format: str) -> str:
    # Let unknown formats fail with an informative KeyError
    return {
        # Audio
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'mp4': 'audio/mp4',  # (Collides with .mp4 video, but we don't care)
        # Image
        'png': 'image/png',
        # Add more as needed
    }[format]


def audio_id_to_format(id: str) -> str:
    id = str(id)  # In case id:Path
    last_op = audio_id_split_ops(id)[-1]
    # Convert e.g. 'enc(mp3,64k)' -> 'mp3', 'enc(wav)' -> 'wav'
    m = re.match(r'enc\(([^,]+)[^)]*\)', last_op)
    if m:
        (format,) = m.groups()
    else:
        # Else assume last_op is a file extension that we know
        format = last_op
    return format


def display_with_audio(x: 'Displayable', audio: 'Audio', **kwargs) -> 'Displayable':
    assert type(audio).__name__ == 'Audio'  # Avoid confusing with display_with_audio_html(x, audio_html)
    return display_with_audio_html(x,
        audio_html=audio_to_html(audio, **{
            'controls': False,  # No controls by default, but allow caller to override
            **kwargs,
        }),
    )


def display_with_audio_bytes(x: 'Displayable', audio_bytes: bytes, mimetype: str, **kwargs) -> 'Displayable':
    return display_with_audio_html(x,
        audio_html=audio_bytes_to_html(audio_bytes, mimetype=mimetype, **{
            'controls': False,  # No controls by default, but allow caller to override
            **kwargs,
        }),
    )


def display_with_style(x: 'Displayable', style_css: str) -> 'Displayable':
    """
    Wrap an (ipy) `display`-able with an inline style
    """
    if not style_css:
        return x
    else:
        x_html = ipy_formats_to_html(x)
        return HTML(dedent_and_strip('''
            <div>
                <style type="text/css">
                    %(style_css)s
                </style>
                %(x_html)s
            </div>
        ''') % dict(
            style_css=style_css,
            x_html=x_html,
        ))


def display_with_audio_html(x: 'Displayable', audio_html: str) -> 'Displayable':
    """
    Wrap an (ipy) `display`-able so that it plays the given audio on click
    - Click to toggle play/pause
    - Shift-click to seek back to the beginning
    """
    assert isinstance(audio_html, str)  # Avoid confusing with display_with_audio(x, audio)

    # Unpack x._display_with_audio_x (set below) if it exists, so that we're idempotent
    x = getattr(x, '_display_with_audio_x', x)

    # Make an HTML() that wraps x's html, audio's html, and a small amount of js for audio controls
    #   - This is an HTML() because it needs to be Displayable that emits a 'text/html' mimetype (e.g. not a
    #     Javascript()), else it will render as junk when used in a df_cell within a df, because df.to_html expects an
    #     html str from each df_cell
    x_html = ipy_formats_to_html(x)
    x_with_audio = HTML(dedent_and_strip('''
        <div class="bubo-audio-container">
            <div>
                <!-- Wrap in case x contains an audio elem, which would fool our selector below -->
                %(x_html)s
            </div>
            %(audio_html)s
            <script>

                // Local scope for isolation (e.g. so we can const)
                (() => {

                    // Get currentScript
                    const currentScript = (
                        document.currentScript || // Defined in browsers
                        document_currentScript    // Defined in notebooks (HACK manually provided by hydrogen-extras)
                    );

                    // WARNING To get a reference to our container, use a hard reference (currentScript.parentNode) instead
                    // of an element selector (e.g. document.querySelectorAll) since something in electron dynamically
                    // hides and reveals dom elements depending on the viewport (i.e. document scroll), and querying by
                    // document.querySelectorAll empirically only returned elements currently in view.
                    const container = currentScript.parentNode;
                    const [audio] = container.querySelectorAll(':scope > audio');

                    // Audio events
                    //  - Ref: https://developer.mozilla.org/en-US/docs/Web/Guide/Events/Media_events
                    const outlineInert = '';
                    container.style.outline = outlineInert;
                    const onAudioEvent = ev => {
                        if (!audio.paused && !audio.ended) {
                            // Playing
                            container.style.outline = '1px solid red';
                        } else if (!audio.ended && audio.currentTime > 0) {
                            // Paused but not reset
                            container.style.outline = '1px solid blue';
                        } else {
                            // Finished playing or reset while paused
                            container.style.outline = outlineInert;
                        }
                    };
                    audio.onplay   = onAudioEvent;
                    audio.onpause  = onAudioEvent;
                    audio.onended  = onAudioEvent;
                    audio.onseeked = onAudioEvent;

                    const forEachAudio = f => {
                        // (Is this at risk of the same document.querySelectorAll ghost problem described above?)
                        Array.from(document.getElementsByClassName('bubo-audio')).forEach(audio => {
                            if (audio.pause) { // Be robust to non-audio elems (does this still happen?)
                                f(audio);
                            }
                        });
                    };

                    // Audio behaviors
                    const resetAllPlayThis = () => {
                        forEachAudio(audio => { audio.pause(); audio.currentTime = 0; });
                        audio.play();
                    };
                    const resetAll = () => {
                        forEachAudio(audio => { audio.pause(); audio.currentTime = 0; });
                    };
                    const pauseAll = () => {
                        forEachAudio(audio => { audio.pause(); });
                    };
                    const pauseAllPlayThis = () => {
                        forEachAudio(audio => { audio.pause(); });
                        audio.play();
                    };

                    // Container events
                    const onContainerMouseEvent = ev => {
                        if (ev.type === 'click') {
                            if (!ev.shiftKey && audio.paused) {
                                resetAllPlayThis();
                            } else if (!ev.shiftKey && !audio.paused) {
                                resetAll();
                            } else if (ev.shiftKey && !audio.paused) {
                                pauseAll();
                            } else if (ev.shiftKey && audio.paused) {
                                pauseAllPlayThis();
                            }
                        } else if (ev.type === 'mouseover' && !ev.altKey && !ev.ctrlKey && ev.metaKey && !ev.shiftKey) {
                            // Like click on
                            resetAllPlayThis();
                        } else if (ev.type === 'mouseout' && !ev.altKey && !ev.ctrlKey && ev.metaKey && !ev.shiftKey) {
                            // Like click off
                            resetAll();
                        }
                    };
                    container.onclick     = onContainerMouseEvent;
                    container.onmouseover = onContainerMouseEvent;
                    container.onmouseout  = onContainerMouseEvent;

                })();

            </script>
        </div>
    ''') % dict(
        x_html=x_html,
        audio_html=audio_html,
    ))

    # Save so we can be idempotent (checked above)
    x_with_audio._display_with_audio_x = x

    return x_with_audio


# For potoo.ipython.df_cell
df_cell_spectros = lambda f, *args, **kwargs: lambda df: df_cell_display.many(f(df, *args, **kwargs, show=False))
df_cell_audios   = lambda df: df_cell_display.many(unbox_many(df.audio))
df_cell_textwrap = lambda col, width=70: lambda df: df[col].map(lambda x: df_cell_stack([
    subline
    for line in x.split('\n')
    for subline in textwrap.wrap(line, width)
]))
