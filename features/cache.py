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

import attr
from joblib import Memory

from constants import cache_dir
from util import singleton

memory = Memory(
    cachedir=cache_dir,  # It adds its own joblib/ subdir
    invalidate_on_code_change=False,
    # verbose=0,  # Log nothing
    # verbose=1,  # Log cache miss
    # verbose=10,  # Log cache miss, cache hit [need >10 to log "Function has changed" before "Clearing cache"]
    verbose=100,  # Log cache miss, cache hit, plus extra
)


# Use slots so that we fail loudly on unknown fields in `with cache_control(...)`
@attr.s(slots=True)
class _CacheControl:
    enabled = attr.ib(True)


cache_control_state = _CacheControl()


def cache_control(**kwargs):
    global cache_control_state
    orig = cache_control_state
    cache_control_state = _CacheControl(**kwargs)
    return _cache_control_gen(orig, **kwargs)


@contextmanager
def _cache_control_gen(orig, **kwargs):
    global cache_control_state
    try:
        yield
    finally:
        cache_control_state = orig


# TODO This is becoming really hacky; consider making a separate api that reuses joblib storage but not joblib.Memory
# TODO Support @cache in addition to @cache() / @cache(...), like Memory.cache does
def cache(
    version=None,
    key=lambda *args, **kwargs: (args, kwargs),
    **kwargs,
):
    """
    Wrap Memory.cache to support version=... and key=..., and remove ignore=...
    """

    assert 'ignore' not in kwargs, 'Use key=... instead of ignore=...'

    def decorator(func):

        @wraps(func)
        def func_cached(args, kwargs, cache_key):
            return func(*args, **kwargs)
        func_cached = memory.cache(func_cached, **kwargs, ignore=['args', 'kwargs'])

        @wraps(func_cached)
        def g(*args, **kwargs):
            if not cache_control_state.enabled:
                return func_cached.func(args, kwargs, None)
            else:
                return func_cached(args, kwargs, dict(
                    version=version,
                    key=key(*args, **kwargs),
                ))
        return g

    return decorator


def cache_lambda(func_name, f, *args, **kwargs):
    # For joblib.func_inspect.get_func_name
    f.__name__ = f'lambda:{func_name}'
    cache_args = kwargs.pop('cache_args', ())
    cache_kwargs = kwargs.pop('cache_kwargs', {})
    return cache(f, *cache_args, **cache_kwargs)(*args, **kwargs)


def cache_pure_method(m, *cache_args, **cache_kwargs):
    """See big caveats at https://pythonhosted.org/joblib/memory.html#gotchas"""
    @wraps(m)
    def g(self, *args, **kwargs):
        # lambda instead of partial to avoid "JobLibCollisionWarning: Cannot detect name collisions for function"
        f = lambda self, *args, **kwargs: m(self, *args, **kwargs)
        # For joblib.func_inspect.get_func_name
        f.__name__ = m.__qualname__
        f.__module__ = m.__module__
        return cache(f, *cache_args, **cache_kwargs)(self, *args, **kwargs)
    return g
