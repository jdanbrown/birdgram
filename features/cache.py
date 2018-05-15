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

from joblib import Memory

from constants import cache_dir

cache = Memory(
    cachedir=cache_dir,  # It adds its own joblib/ subdir
    # verbose=0,  # Log nothing
    # verbose=1,  # Log cache miss
    verbose=10,  # Log cache miss, cache hit
    # verbose=100,  # Log cache miss, cache hit, plus extra
)


def cache_lambda(func_name, f, *args, **kwargs):
    # For joblib.func_inspect.get_func_name
    f.__name__ = f'lambda:{func_name}'
    cache_args = kwargs.pop('cache_args', ())
    cache_kwargs = kwargs.pop('cache_kwargs', {})
    return cache.cache(f, *cache_args, **cache_kwargs)(*args, **kwargs)


def cache_pure_method(m, *cache_args, **cache_kwargs):
    """See big caveats at https://pythonhosted.org/joblib/memory.html#gotchas"""
    @wraps(m)
    def g(self, *args, **kwargs):
        # lambda instead of partial to avoid "JobLibCollisionWarning: Cannot detect name collisions for function"
        f = lambda self, *args, **kwargs: m(self, *args, **kwargs)
        # For joblib.func_inspect.get_func_name
        f.__name__ = m.__qualname__
        f.__module__ = m.__module__
        return cache.cache(f, *cache_args, **cache_kwargs)(self, *args, **kwargs)
    return g
