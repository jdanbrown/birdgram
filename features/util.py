#
# Side effects
#

import warnings

# Suppress "FutureWarning: 'pandas.core' is private. Use 'pandas.Categorical'"
#   - https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)


#
# For export
#

from typing import Iterable, Iterator, List, Mapping, Tuple, TypeVar, Union

from attrdict import AttrDict
import PIL
from potoo import debug_print
from potoo.pandas import df_ensure, df_summary
from potoo.util import puts, singleton, strip_startswith, tap
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
import re
import os
import pickle
import random
import shlex
import textwrap
from typing import Callable, Optional, TypeVar, Union

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


def generator_to(agg):
    def decorator(f):
        @wraps(f)
        def g(*args, **kwargs):
            return agg(f(*args, **kwargs))
        return g
    return decorator


def glob_filenames_ensure_parent_dir(pattern: str) -> Iterable[str]:
    """
    Useful for globbing in gcsfuse, which requires dirs to exist as empty objects for its files to be listable
    """
    assert not re.match(r'.*[*?[].*/', pattern), \
        f'pattern[{pattern!r}] can only contain globs (*?[) in filenames, not dirnames'
    return glob.glob(ensure_parent_dir(pattern))


def ensure_parent_dir(path):
    mkdir_p(os.path.dirname(path))
    return path


def touch_file(path):
    ensure_parent_dir(path)
    open(path, 'w').close()
    return path


def mkdir_p(path):
    os.system('mkdir -p %s' % shlex.quote(str(path)))


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


def sha1hex(*xs: Union[bytes, str]) -> str:
    """sha1hex(x, y) == sha1hex(x + y)"""
    h = hashlib.sha1()
    for x in xs:
        if isinstance(x, str):
            x = x.encode()
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


# Kind of better than %memit...
@contextmanager
def print_mem_delta(desc='mem_delta', collect_before=False, collect_after=False, print=print):
    import gc
    import psutil
    from potoo.pretty import pformat
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
        diff = {k: '%s KB' % ((end[k] - start[k]) // 1024) for k in start.keys()}
        print('[%s] %s' % (desc, pformat(diff)))


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
# numpy
#

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
# pandas
#

from collections import OrderedDict
import tempfile
import time
from typing import Iterable, Iterator
import uuid

from dataclasses import dataclass
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
    engine='auto',  # .to_parquet + .read_parquet
    compression='default',  # .to_parquet
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
from typing import Callable, Iterable, Mapping, Optional, TypeVar

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


def _df_map_rows_progress_joblib(
    df: pd.DataFrame,
    f: Callable[['Row'], 'Row'],
    use_joblib=True,
    **kwargs,
):
    return pd.DataFrame(_map_progress_joblib(
        f=lambda row: f(row),
        xs=[row for i, row in df.iterrows()],
        **kwargs,
    ))


def _map_progress_joblib(
    f: Callable[[X], X],
    xs: Iterable[X],
    desc: str = None,  # TODO Display somewhere (like dask + tqdm)
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
from typing import Callable, Iterable, List, TypeVar, Union

from attrdict import AttrDict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from potoo.util import AttrContext, get_cols
import structlog

log = structlog.get_logger(__name__)

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

        def __init__(self, desc=None, **kwargs):
            super().__init__(**kwargs)
            self._terminal_cols = get_cols()
            self._desc = desc
            self._desc_pad = '%s: ' % self._desc if self._desc else ''
            # self._desc_pad = '%s:' % self._desc if self._desc else ''

        def _draw_bar(self, frac, elapsed):
            # (Mostly copy/pasted from ProgressBar)
            percent = int(100 * frac)
            elapsed = format_time(elapsed)
            msg_prefix = '\r%s[' % self._desc_pad
            # msg_prefix = '\r[%s' % self._desc_pad
            msg_suffix = '] | %3s%% Completed | %s' % (percent, elapsed)
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


def _df_map_rows_progress_dask(
    df: pd.DataFrame,
    f: Callable[['Row'], 'Row'],
    **kwargs,
):
    return pd.DataFrame(_map_progress_dask(
        f=lambda row: f(row),
        xs=[row for i, row in df.iterrows()],
        **kwargs,
    ))


def _map_progress_dask(
    f: Callable[[X], X],
    xs: Iterable[X],
    desc: str = None,
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
        import dask.bag
        # HACK dask.bag.from_sequence([pd.Series(...), ...]) barfs -- workaround by boxing it
        # HACK dask.bag.from_sequence([np.array(...), ...]) flattens the arrays -- workaround by boxing it
        # HACK Avoid other cases we haven't tripped over yet by boxing everything unconditionally
        wrap, unwrap = (lambda x: box(x)), (lambda x: x.unbox)
        with dask_progress(desc=desc):
            if not partition_size and not npartitions:
                xs = list(xs)
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
# sklearn / dask
#

import types
from typing import Callable, Iterable, TypeVar, Union

import joblib
from tqdm import tqdm

X = TypeVar('X')


def df_map_rows_progress(
    *args,
    use='sync',  # 'sync' | 'dask' | 'joblib'
    **kwargs,
) -> pd.DataFrame:
    return ({
        'sync': _df_map_rows_progress_sync,
        'dask': _df_map_rows_progress_dask,
        'joblib': _df_map_rows_progress_joblib,
    }[use])(*args, **kwargs)


def map_progress(
    *args,
    use='sync',  # 'sync' | 'dask' | 'joblib'
    **kwargs,
) -> Iterable[X]:
    return ({
        'sync': _map_progress_sync,
        'dask': _map_progress_dask,
        'joblib': _map_progress_joblib,
    }[use])(*args, **kwargs)


def _df_map_rows_progress_sync(
    df: pd.DataFrame,
    f: Callable[['Row'], 'Row'],
    **kwargs,
) -> pd.DataFrame:
    return pd.DataFrame(_map_progress_sync(
        f=lambda row: f(row),
        xs=(row for i, row in df.iterrows()),
        n=len(df),
        **kwargs,
    ))


def _map_progress_sync(
    f: Callable[[X], X],
    xs: Iterable[X],
    **kwargs,
) -> Iterable[X]:
    if hasattr(xs, '__len__'):
        kwargs.setdefault('n', len(xs))
    return list(iter_progress(
        map(f, xs),
        **kwargs,
    ))


def iter_progress(
    xs: Iterator[X],
    desc: str = None,
    n: int = None,
    use_tqdm=True,
) -> Iterator[X]:
    if use_tqdm:
        return tqdm(xs, total=n, desc=desc)
    else:
        return xs


#
# statsmodels
#

import statsmodels.api as sm
import statsmodels.formula.api as smf


def lm(*args, **kwargs):
    return smf.ols(*args, **kwargs).fit()


#
# audiosegment
#

import copy
import json
from typing import Callable

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
    audio.__dict__.update(audio_dict)  # Copy non-standard attrs (e.g. .path) [TODO Make our own Audio class to own .path]
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
        json.dumps(sort_keys=True, separators=(',', ':'), obj=dict(
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
from typing import Union
import urllib.parse

import numpy as np
import matplotlib.pyplot as plt
from potoo.ipython import *
from potoo.pandas import *
import psutil
import pydub
import scipy
import yaml

from config import config
from constants import *
from datatypes import Audio
from log import log  # For export [TODO Update callers]


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
    'basename',
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


# TODO Dedupe vs. Load._ergonomic_audio
def rec_audio_ensure_persisted(rec: 'Recording', **kwargs) -> 'Recording':
    assert rec.id == rec.audio.unbox.name  # TODO Is this actually always true? (Load._ergonomic_audio, ...where else?)
    rec = rec.copy()  # Copy so we can mutate
    rec.audio = box(audio_ensure_persisted(rec.audio.unbox, **kwargs))
    rec.path = rec.audio.unbox.path
    return rec


# TODO Dedupe vs. Load._ergonomic_audio
def audio_ensure_persisted(audio, **audio_export_kwargs) -> Audio:

    # TODO TODO Update defaults based on empirical measurements
    #   - [x] Measure sizes
    #   - [ ] Eval quality of the smaller sizes -- is 32k aac good enough quality for our needs?
    audio_export_kwargs = audio_export_kwargs or dict(
        # format='wav',
        # format='mp3', bitrate='32k',
        format='mp4', bitrate='32k', codec='aac',
        # format='mp4', bitrate='32k', codec='libfdk_aac',
    )

    abs_path = Path(audio_cache_path(audio, **audio_export_kwargs))
    audio = audio_replace(audio,
        path=str(abs_path.relative_to(data_dir)),
    )
    if not abs_path.exists():
        log.info(f'Persisting: {abs_path}')
        audio.export(ensure_parent_dir(abs_path), **audio_export_kwargs)

    return audio


def audio_cache_path(audio, **kwargs) -> str:
    return audio_cache_path_for_params(
        name=audio.name,
        frame_rate=audio.frame_rate,
        channels=audio.channels,
        sample_width=audio.sample_width,
        **kwargs,
    )


def audio_cache_path_for_params(
    name: str,  # (Assume: audio.name same as rec.id)
    frame_rate: int,
    channels: int,
    sample_width: int,
    # **audio_export_kwargs: Single point of control that enforces the subset of audio.export(**kwargs) we accept
    format: str,
    bitrate: str = None,
    codec: str = None,
) -> str:
    assert {
        'wav': not bitrate and not codec,
        'mp3': bitrate and not codec,
        'mp4': bitrate and codec,
    }[format], f'Invalid **audio_export_kwargs: {dict(format=format, bitrate=bitrate, codec=codec)}'
    params_id = '-'.join([
        f'{frame_rate}hz',
        f'{channels}ch',
        f'{sample_width * 8}bit',
        *({
            'wav': [],  # XXX Back compat
            # 'wav': [format],  # TODO After renaming cache dirs on local + remote
            'mp3': [format, bitrate],
            'mp4': [format, codec, bitrate],
        }[format]),
    ])
    ext = format  # Redundant, but helpful to external programs
    return f'{cache_dir}/{params_id}/{name}.{ext}'


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


def audio_to_bytes(audio, **kwargs) -> bytes:
    abs_path = Path(data_dir) / audio_ensure_persisted(audio, **kwargs).path
    with open(abs_path, 'rb') as f:
        return f.read()


def audio_to_url(audio, url_type=None, **kwargs) -> str:
    abs_path = Path(data_dir) / audio_ensure_persisted(audio, **kwargs).path
    if (url_type or config.audio_to_url.url_type) == 'file':
        return 'file://%s' % urllib.parse.quote(str(abs_path),
            safe='/,:()[] ',  # Cosmetic: exclude known-safe chars ('?' is definitely _not_ safe, not sure what else...)
        )
    elif (url_type or config.audio_to_url.url_type) == 'data':
        return 'data:%(mimetype)s;base64,%(base64)s' % dict(
            mimetype=audio_mimetype_for_path(abs_path),
            base64=base64.b64encode(audio_to_bytes(audio, **kwargs)).decode('ascii'),
        )
    else:
        raise ValueError('Unexpected config.audio_to_url.url_type: %s' % config.audio_to_url.url_type)


def audio_to_html(audio, controls=True, preload='none', more_audio_attrs='', **kwargs) -> str:
    audio = audio_ensure_persisted(audio, **kwargs)
    return dedent_and_strip('''
        <audio class="bubo-audio" %(controls)s preload="%(preload)s" %(more_audio_attrs)s>
            <source type="%(type)s" src="%(src)s" />
        </audio>
    ''') % dict(
        controls='controls' if controls else '',
        preload=preload,
        more_audio_attrs=more_audio_attrs,
        type=audio_mimetype_for_path(audio.path),
        src=audio_to_url(audio, **kwargs),
    )


def audio_mimetype_for_path(path) -> str:
    return {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.mp4': 'audio/mp4',
        # Add more as needed...
    }[Path(path).suffix]


def display_with_audio(x: 'Displayable', audio: 'Audio', **kwargs) -> 'Displayable':
    """
    Wrap an (ipy) `display`-able so that it plays the given audio on click
    - Click to toggle play/pause
    - Shift-click to seek back to the beginning
    """

    # Unpack x._display_with_audio_x (set below) if it exists, so that we're idempotent
    x = getattr(x, '_display_with_audio_x', x)

    # Make an HTML() that wraps x's html, audio's html, and a small amount of js for audio controls
    #   - This is an HTML() because it needs to be Displayable that emits a 'text/html' mimetype (e.g. not a
    #     Javascript()), else it will render as junk when used in a df_cell within a df, because df.to_html expects an
    #     html str from each df_cell
    x_html = ipy_formats_to_html(x)
    audio_html = audio_to_html(audio, **{
        'controls': False,  # No controls by default, but allow caller to override
        **kwargs,
    })
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
df_cell_audios = lambda df: df_cell_display.many(unbox_many(df.audio))
df_cell_textwrap = lambda col, width=70: lambda df: df[col].map(lambda x: df_cell_stack([
    subline
    for line in x.split('\n')
    for subline in textwrap.wrap(line, width)
]))
