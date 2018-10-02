import json

import numpy as np


def json_dumps_safe(*args, **kwargs) -> str:
    """
    json.dumps with a decoder that's safe for our common datatypes (e.g. np scalars)
    """
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


def json_sanitize(x: any) -> any:
    """
    Sanitize (deep) contents of x so it's json safe
    """
    return json.loads(json_dumps_safe(x))
