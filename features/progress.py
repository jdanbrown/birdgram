from functools import partial
from pathlib import Path
import multiprocessing
import types
from typing import *

from attrdict import AttrDict
from dataclasses import dataclass, field
import joblib
import numpy as np
import pandas as pd
from potoo.util import AttrContext, get_cols, timed
from tqdm import tqdm

from config import config
from logging_ import _map_progress_log_time

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
    from util import df_rows  # Avoid cyclic import
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
    if not desc and hasattr(f, '__qualname__'):  # (e.g. partial(f) has no __name__ (or __qualname__))
        desc = f.__name__
    if not n and hasattr(xs, '__len__'):
        n = len(xs)

    # Resolve progress_kwargs: .override -> **_progress_kwargs -> .default -> fallback default
    kwargs = (
        progress_kwargs.override or
        _progress_kwargs or
        progress_kwargs.default or
        config.progress_kwargs  # Fallback default
    )
    kwargs = dict(kwargs)  # Copy so we can mutate

    # Delegate as per `use`
    use = kwargs.pop('use')
    return ({
        None: _map_progress_none,
        'log_time': _map_progress_log_time,
        'sync': _map_progress_sync,
        'dask': _map_progress_dask,
        'joblib': _map_progress_joblib,
    }[use])(
        f=f, xs=xs, desc=desc, n=n,
        **kwargs,
    )


def _map_progress_none(
    f: Callable[[X], X],
    xs: Iterable[X],
    desc: str = None,
    n: int = None,
) -> Iterable[X]:
    return list(map(f, xs))


def _map_progress_sync(
    f: Callable[[X], X],
    xs: Iterable[X],
    desc: str = None,
    n: int = None,
) -> Iterable[X]:
    return list(tqdm(
        map(f, xs), desc=desc, total=n,
    ))


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
# dask
#


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
    from util import box, coalesce  # Avoid cyclic import
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


#
# joblib
#


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
