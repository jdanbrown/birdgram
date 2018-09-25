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

from functools import partial, wraps
from typing import List
import uuid

import attr
from dataclasses import dataclass, field
from joblib import Memory
from joblib.memory import MemorizedFunc
from potoo.util import AttrContext

from config import config
from constants import cache_dir
from log import char_log
from util import singleton

memory = Memory(
    cachedir=cache_dir,  # It adds its own joblib/ subdir
    invalidate_on_code_change=False,
    log=char_log.replace(
        # level='debug',  # Log '.'/'!' line on hit/miss [HACK HACK HACK]
        # level='info',  # Log '.'/'!' char on hit/miss [HACK HACK HACK]
        # level='warn',  # Log nothing [FIXME -1 (char_log.char) is causing the 'IOStream.flush timed out' errors in remote kernels]
        level=config.cache.log_level,
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
    tags_refresh: List[str] = field(default_factory=lambda: [])
    tags_fail_on_miss: List[str] = field(default_factory=lambda: [])


# Workaround for @singleton (above)
cache_control = _cache_control()


def tags_match(a: List[str], b: List[str]) -> bool:
    """Tags match if they overlap (symmetric)"""
    return set(a) & set(b)


# TODO This is becoming really hacky; consider making a separate api that reuses joblib storage but not joblib.Memory
# TODO Support @cache in addition to @cache() / @cache(...), like Memory.cache does
def cache(
    version=None,
    key=lambda *args, **kwargs: (args, kwargs),
    nocache=lambda *args, **kwargs: False,
    norefresh=False,  # Don't refresh on cache_control(refresh=True)
    tag: str = None,  # Alias: tag=s -> tags=[s]
    tags: List[str] = [],
    **kwargs,
):
    """
    Wrap Memory.cache to support version=... and key=..., and remove ignore=...
    """

    # Params
    if tag:
        assert not tags, f"tag[{tag}], tags[{tags}]"
        tags = [tag]
        tag = None
    assert isinstance(tags, list), f"Expected List[str], got {type(tags).__name__}"
    assert 'ignore' not in kwargs, 'Use key=... instead of ignore=...'

    def decorator(func):

        @wraps_workaround(func)
        def func_cached(cache_key, ignore):
            return func(*ignore['args'], **ignore['kwargs'])
        func_cached._cache_version = version  # HACK HACK HACK Smuggle to MemorizedFunc.call, to include in dir name
        func_cached = memory.cache(func_cached, **kwargs, ignore=['ignore'])

        @wraps_workaround(func_cached)
        def g(*args, **kwargs):
            # WARNING Do _nocache here i/o arg list, else nonsensical errors with dask processes (but not threads/sync)
            _nocache = kwargs.pop('_nocache', False)
            ignore = dict(args=args, kwargs=kwargs)
            if not cache_control.enabled or nocache(*args, **kwargs) or _nocache:
                # Bypass cache
                cache_key = None
                return func_cached.func(cache_key, ignore)
            else:
                cache_key = dict(version=version, key=key(*args, **kwargs))
                if (
                    not norefresh and
                    (cache_control.refresh or tags_match(tags, cache_control.tags_refresh))
                ):
                    # Force refresh (= recompute and store)

                    # HACK HACK HACK Log to match joblib.memory.MemorizeFunc (cf. joblib/memory.py)
                    func_id, _ = func_cached._get_output_identifiers(cache_key, ignore)
                    memory.log.char('info', '»')
                    memory.log.char('debug', ' %s\n' % func_id)

                    out = func_cached.func(cache_key, ignore)
                    _store_result(out, func_cached, cache_key, ignore)
                else:
                    # Hit/miss as normal (via MemorizedFunc)

                    # Fail if cache miss and any func tags are in tags_fail_on_miss
                    if (
                        tags_match(tags, cache_control.tags_fail_on_miss) and
                        not _is_cached(func_cached, cache_key, ignore)
                    ):
                        raise ValueError('func[%s] cache miss (tags_fail_on_miss%s, tags%s)' % (
                            func.__qualname__, cache_control.tags_fail_on_miss, tags,
                        ))

                    out = func_cached(cache_key, ignore)
                return out
        return g

    return decorator


def _is_cached(f: MemorizedFunc, *args, **kwargs):
    func_id, args_id = f._get_output_identifiers(*args, **kwargs)
    return f.store_backend.contains_item([func_id, args_id])


def _store_result(out: any, f: MemorizedFunc, *args, **kwargs):
    func_id, args_id = f._get_output_identifiers(*args, **kwargs)
    f.store_backend.dump_item([func_id, args_id], out)


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
