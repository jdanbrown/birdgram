import gzip as _gzip
import json

import numpy as np


def json_default_safe(x):
    try:
        import sklearn as sk
        BaseEstimator = sk.base.BaseEstimator
    except:
        BaseEstimator = None
    return (
        # Unwrap np scalar dtypes (e.g. np.int64 -> int) [https://stackoverflow.com/a/16189952/397334]
        np.asscalar(x) if isinstance(x, np.generic) else
        # HACK Avoid "circular reference detected" errors with sk estimators that contain other estimators (e.g. OneVsRestClassifier)
        x.get_params() if isinstance(x, BaseEstimator) else
        # Else return as is
        x
    )


def json_dump_safe(*args, **kwargs) -> str:
    """
    json.dump with a decoder that's safe for our common datatypes (e.g. np scalars)
    """
    return json.dump(*args, **kwargs, default=json_default_safe)


def json_dumps_safe(*args, **kwargs) -> str:
    """
    json.dumps with a decoder that's safe for our common datatypes (e.g. np scalars)
    """
    return json.dumps(*args, **kwargs, default=json_default_safe)


def json_dumps_canonical(obj: any, **kwargs) -> str:
    """
    Dump a canonical json representation of obj (e.g. suitable for use as a cache key)
    - json_dumps_canonical(dict(a=1, b=2)) == json_dumps_canonical(dict(b=2, a=1))
    """
    return json_dumps_safe(obj, sort_keys=True, separators=(',', ':'), **kwargs)


def json_sanitize(x: any) -> any:
    """
    Sanitize (deep) contents of x so it's json safe
    """
    return json.loads(json_dumps_safe(x))


def json_dump_path(obj: any, path: str, gzip: bool = False, **kwargs):
    """
    Shorthand for json_dump_safe to a path i/o a file object
    """
    _open = open if not gzip else _gzip.open
    with _open(path, 'wt') as f:
        json_dump_safe(obj, f, **kwargs)
        f.write('\n')
