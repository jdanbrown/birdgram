## Side effects

import warnings

# Suppress "FutureWarning: 'pandas.core' is private. Use 'pandas.Categorical'"
#   - https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)


## For export

from typing import Iterable, Iterator, List, Mapping, Tuple, TypeVar, Union

from attrdict import AttrDict
from itertools import *
from more_itertools import *
import PIL
from potoo.pandas import df_ensure, df_summary
from potoo.util import singleton, strip_startswith
import tqdm


## util

from contextlib import contextmanager
from functools import partial, wraps
import glob
import hashlib
import re
import os
import pickle
import random
import shlex
from typing import Optional, TypeVar, Union

X = TypeVar('X')


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
    os.system('mkdir -p %s' % shlex.quote(path))


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


def sha1hex(x: Union[bytes, str]) -> str:
    if isinstance(x, str):
        x = x.encode()
    return hashlib.sha1(x).hexdigest()


def enumerate_with_n(xs: Iterable[X]) -> (int, int, Iterator[X]):
    """Like enumerate(xs) but assume xs is materialized and include n = len(xs)"""
    n = len(xs)
    return ((i, n, x) for i, x in enumerate(xs))


## unix

import os


def ls(dir):
    return [
        os.path.join(dir, filename)
        for filename in os.listdir(dir)
    ]


## dataclasses

from collections import OrderedDict
import json
import sys
from typing import Iterable

from attrdict import AttrDict
import dataclasses
from potoo.util import get_cols


class DataclassUtil:
    """Things I wish all dataclasses had"""

    @classmethod
    def field_names(cls, **filters) -> Iterable[str]:
        return [
            x.name
            for x in dataclasses.fields(cls)
            if all(getattr(x, k) == v for k, v in filters.items())
        ]

    def replace(self, **kwargs) -> 'Self':
        return dataclasses.replace(self, **kwargs)

    def asdict(self) -> dict:
        """Convert to dict preserving field order, e.g. for df rows"""
        return OrderedDict(dataclasses.asdict(self))

    def asattr(self) -> AttrDict:
        return AttrDict(self.asdict())

    def __sizeof__(self):
        try:
            from dask.sizeof import sizeof
        except:
            sizeof = sys.getsizeof
        return sizeof(list(self.asdict().items()))


class DataclassConfig(DataclassUtil):
    """Expose dataclass fields via .config"""

    @property
    def config(self) -> AttrDict:
        return {
            k: v
            for k, v in self.asdict().items()
            if k not in (self.deps or {})
        }


## pandas

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


def df_rows(df) -> Iterator[Row]:
    return (row for i, row in df.iterrows())


def df_flatmap_list_col(df, col_name, col_f=lambda s: s):
    return (df
        .assign(**{col_name: col_f(df[col_name])})
        .pipe(df_flatmap, lambda row: [
            OrderedDict({**row, col_name: x})
            for x in row[col_name]
        ])
    )


@dataclass
class box:
    """Useful e.g. for putting iterables inside pd.Series/np.array"""
    unbox: any

    @classmethod
    def many(cls, xs):
        return [box(x) for x in xs]


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


## sklearn

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

X = TypeVar('X')


@singledispatch
def model_stats(model: Optional[sk.base.BaseEstimator], **kwargs) -> pd.DataFrame:
    return pd.DataFrame([
        dict(type=f'unknown:{type(model).__name__}'),
    ])


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


def _df_apply_progress_joblib(
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


## dask

import multiprocessing
from typing import Callable, Iterable, TypeVar

from attrdict import AttrDict
from dataclasses import dataclass
import pandas as pd
from potoo.util import AttrContext, get_cols

X = TypeVar('X')


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
@dataclass
class _dask_opts(AttrContext):
    override_use_dask: bool = None
    override_scheduler: bool = None


# Workaround for @singleton (above)
dask_opts = _dask_opts()


def _df_apply_progress_dask(
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
        from dask.diagnostics import ProgressBar
        # HACK dask.bag.from_sequence([pd.Series(...), ...]) barfs -- workaround by boxing it
        # HACK dask.bag.from_sequence([np.array(...), ...]) flattens the arrays -- workaround by boxing it
        # HACK Avoid other cases we haven't tripped over yet by boxing everything unconditionally
        wrap, unwrap = (lambda x: box(x)), (lambda x: x.unbox)
        with ProgressBar(width=get_cols() - 30):
            if not partition_size and not npartitions:
                xs = list(xs)
                (unit_sec, _) = timed(lambda: list(map(f, xs[:1])))
                npartitions = _npartitions_for_unit_sec(len(xs), unit_sec, **kwargs)
            return (dask.bag
                .from_sequence(map(wrap, xs), partition_size=partition_size, npartitions=npartitions)
                .map(unwrap)
                .map(f)
                .compute(get=dask_get_for_scheduler_name(scheduler, **(get_kwargs or {})))
            )


def _npartitions_for_unit_sec(n: int, unit_sec: float, target_sec_per_partition=2, min_npartitions_per_core=5) -> int:
    n_cores = multiprocessing.cpu_count()
    npartitions = int(n * unit_sec / target_sec_per_partition)  # Estimate from unit_sec
    npartitions = round(npartitions / n_cores) * n_cores  # Round to multiple of n_cores
    npartitions = max(npartitions, n_cores * min_npartitions_per_core)  # Min at n_cores * k (for small k)
    return npartitions


# Mimic http://dask.pydata.org/en/latest/scheduling.html
def dask_get_for_scheduler_name(scheduler, **kwargs):
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


## sklearn / dask

import types
from typing import Callable, Iterable, TypeVar, Union

import joblib
from tqdm import tqdm

X = TypeVar('X')


def df_apply_progress(
    *args,
    use='sync',  # 'sync' | 'dask' | 'joblib'
    **kwargs,
) -> pd.DataFrame:
    return ({
        'sync': _df_apply_progress_sync,
        'dask': _df_apply_progress_dask,
        'joblib': _df_apply_progress_joblib,
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


def _df_apply_progress_sync(
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
    return list(iter_progress(map(f, xs), **kwargs))


def iter_progress(
    xs: Iterator[X],
    n: int = None,
    use_tqdm=True,
) -> Iterator[X]:
    if use_tqdm:
        return tqdm(xs, total=n)
    else:
        return xs


## statsmodels

import statsmodels.api as sm
import statsmodels.formula.api as smf


def lm(*args, **kwargs):
    return smf.ols(*args, **kwargs).fit()


## bubo-features

import platform

import psutil
import yaml

from log import log  # For export [TODO Update callers]


def print_sys_info():
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
