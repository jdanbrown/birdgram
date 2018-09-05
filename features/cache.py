"""
Docs:
- https://pythonhosted.org/joblib/memory.html

Example usage:

    # Caches in <cache_dir>/joblib/module/submodule/foo/
    @cache.cache
    def foo(x, y):
        return z

    # Caches in <cache_dir>/joblib/module/submodule/foo/
    class Model:
        @staticmethod
        @cache_pure_method
        def foo(x, y):
            return z

    # Caches in <cache_dir>/joblib/module/submodule/Model.foo/
    class Model:
        @classmethod
        @cache_pure_method
        def foo(cls, x, y):
            return z

    # Caches in <cache_dir>/joblib/module/submodule/Model.foo/
    class Model:
        @cache_pure_method
        def foo(self, x, y):
            return z

    # Caches in <cache_dir>/joblib/module/submodule/lambda:foo/
    self.proj_skm_ = cache_lambda('foo', lambda x, y: z, x, y)

"""

from contextlib import contextmanager
from functools import partial, wraps
import uuid

import attr
from dataclasses import dataclass
from joblib import Memory
from joblib.memory import MemorizedFunc
from potoo.util import AttrContext

from constants import cache_dir
from log import log
from util import singleton

memory = Memory(
    cachedir=cache_dir,  # It adds its own joblib/ subdir
    invalidate_on_code_change=False,
    log=log.replace(
        # level='debug',  # Log '.'/'!' line on hit/miss [HACK HACK HACK]
        # level='info',  # Log '.'/'!' char on hit/miss [HACK HACK HACK]
        level='warn',  # Log nothing [FIXME -1 (log.char) is causing the 'IOStream.flush timed out' errors in remote kernels]
    ),
    verbose=0,  # Log nothing
    # verbose=1,  # Log cache miss
    # verbose=10,  # Log cache miss, cache hit [need >10 to log "Function has changed" before "Clearing cache"]
    # verbose=100,  # Log cache miss, cache hit, plus extra
)


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
@dataclass
class _cache_control(AttrContext):
    enabled: bool = True
    refresh: bool = False


# Workaround for @singleton (above)
cache_control = _cache_control()


# TODO This is becoming really hacky; consider making a separate api that reuses joblib storage but not joblib.Memory
# TODO Support @cache in addition to @cache() / @cache(...), like Memory.cache does
# TODO Include version in the function's dir name so it's easy to clean up old defunct versions that take up space
def cache(
    version=None,
    key=lambda *args, **kwargs: (args, kwargs),
    nocache=lambda *args, **kwargs: False,
    **kwargs,
):
    """
    Wrap Memory.cache to support version=... and key=..., and remove ignore=...
    """

    assert 'ignore' not in kwargs, 'Use key=... instead of ignore=...'

    def decorator(func):

        @wraps_workaround(func)
        def func_cached(cache_key, ignore):
            return func(*ignore['args'], **ignore['kwargs'])
        func_cached = memory.cache(func_cached, **kwargs, ignore=['ignore'])

        @wraps_workaround(func_cached)
        def g(*args, _nocache=False, **kwargs):
            ignore = dict(args=args, kwargs=kwargs)
            if not cache_control.enabled or nocache(*args, **kwargs) or _nocache:
                cache_key = None
                return func_cached.func(cache_key, ignore)
            else:
                cache_key = dict(version=version, key=key(*args, **kwargs))
                if cache_control.refresh:
                    _clear_result(func_cached, cache_key, ignore)
                return func_cached(cache_key, ignore)
        return g

    return decorator


def _clear_result(f: MemorizedFunc, *args, **kwargs):
    """Clear an individual MemorizedResult for a MemorizedFunc"""
    func_id, args_id = f._get_output_identifiers(*args, **kwargs)
    f.store_backend.clear_item([func_id, args_id])


def cache_lambda(func_name, f, *args, **kwargs):
    # For joblib.func_inspect.get_func_name
    f.__name__ = f'lambda:{func_name}'
    cache_args = kwargs.pop('cache_args', ())
    cache_kwargs = kwargs.pop('cache_kwargs', {})
    return cache(f, *cache_args, **cache_kwargs)(*args, **kwargs)


def cache_pure_method(m, *cache_args, **cache_kwargs):
    """See big caveats at https://pythonhosted.org/joblib/memory.html#gotchas"""
    @wraps_workaround(m)
    def g(self, *args, **kwargs):
        # lambda instead of partial to avoid "JobLibCollisionWarning: Cannot detect name collisions for function"
        f = lambda self, *args, **kwargs: m(self, *args, **kwargs)
        # For joblib.func_inspect.get_func_name
        f.__name__ = m.__qualname__
        f.__module__ = m.__module__
        return cache(f, *cache_args, **cache_kwargs)(self, *args, **kwargs)
    return g


def wraps_workaround(f):
    """
    Workaround a cloudpickle bug that causes joblib.Memory to lose the cached func __name__ inside dask processes
    - https://github.com/cloudpipe/cloudpickle/issues/177

    This workaround can be deleted when these asserts pass:

        import dask
        from joblib.func_inspect import *

        load = Load()

        # Works regardless of bug, since threads don't invoke cloudpickle
        assert dask.delayed(format_signature)(load._metadata).compute(get=dask_get_for_scheduler_name('threads')) == \
            ('load._metadata', '_metadata()')

        # Requires bugfix to work, since processes marshal using cloudpickle
        assert dask.delayed(format_signature)(load._metadata).compute(get=dask_get_for_scheduler_name('processes')) == \
            ('load._metadata', '_metadata()')

    """
    def decorator(g):
        g = wraps(f)(g)
        g.func_name = f.__name__
        return g
    return decorator
