"""
API adapter for python-geohash
- Docs: https://github.com/hkwi/python-geohash/wiki/GeohashReference
- Interactive map: http://geohash.gofreerange.com/
- Interactive map: https://www.movable-type.co.uk/scripts/geohash.html
"""

from functools import partial
from typing import Iterable

from more_itertools import one, unique_everseen
import numpy as np

import geohash as _geohash
from util import generator_to, np_vectorize_asscalar

# Vectorize everything?
#   - For *_decode, np.vectorize barfs on multiple return values -- and adds no value on non-multiple input args
#       - e.g. np_vectorize_asscalar(int_decode, otypes=[np.float64, np.float64])


@partial(np_vectorize_asscalar, otypes=[str])
def str_encode(lat: float, lon: float, precision_bits: int) -> str:
    return str_at_prec(_geohash.encode(lat, lon, (precision_bits + 4) // 5), precision_bits)


def str_decode_origin(x: str) -> (float, float):
    """Decode (lat, lon) at bottom-left of geohash (independent of precision, since bottom-left is '<x>0...')"""
    # return _geohash.decode(x)  # Nope: this one returns the center, using an implicit precision=len(s)
    (lat, lon, _, _) = _geohash._geohash.decode(x)  # Have to reach into the native module to get the bottom-left
    return (lat, lon)


def str_at_prec(x: str, precision_bits: int) -> str:
    precision_bits_div = precision_bits // 5
    precision_bits_mod = precision_bits % 5
    keep_chars = x[: precision_bits_div]
    if precision_bits_mod == 0 or precision_bits_div >= len(x):
        truncate_char = ''
    else:
        [truncate_char] = x[precision_bits_div : precision_bits_div + 1]
        truncate_char = _int_to_char[int_at_prec(_char_to_int[truncate_char], precision_bits_mod, int_bits=5)]
    y = keep_chars + truncate_char
    extend_bits = precision_bits - len(y) * 5
    if extend_bits > 0:
        y += '0' * ((extend_bits + 4) // 5)
    return y


_int_to_char = '0123456789bcdefghjkmnpqrstuvwxyz'  # (Copied from error msg given by `_geohash.decode('a')`)
_char_to_int = {c: i for i, c in enumerate(_int_to_char)}


def str_expand(x: str, precision_bits: int) -> Iterable[str]:
    return [
        int_to_str(y, precision_bits)
        for y in int_expand(str_to_int(x, precision_bits), precision_bits)
    ]


def str_neighbors(x: str, precision_bits: int) -> Iterable[str]:
    """neighbors(x) + [x] = expand(x)"""
    x = str_at_prec(x, precision_bits)
    return [y for y in str_expand(x, precision_bits) if y != x]


def str_to_int(x: str, precision_bits: int) -> int:
    return int_encode(*str_decode_origin(x), precision_bits)


def int_to_str(x: int, precision_bits: int) -> str:
    return str_encode(*int_decode_origin(x), precision_bits)


# @partial(np_vectorize_asscalar, otypes=[np.uint64])  # TODO Generalize downstream code to not barf on np.uint64 (rely on unit tests)
def int_encode(lat: float, lon: float, precision_bits: int) -> int:
    return int_at_prec(_geohash.encode_uint64(lat, lon), precision_bits)


def int_decode_origin(x: int) -> (float, float):
    """Decode (lat, lon) at bottom-left of geohash (independent of precision, since bottom-left is '<x>0...')"""
    return _geohash.decode_uint64(x)  # No tricks here, this one returns the bottom-left (like the native module)


def int_at_prec(x: int, precision_bits: int, int_bits: int = 64) -> int:
    return x >> (int_bits - precision_bits) << (int_bits - precision_bits)


def int_expand(x: int, precision_bits: int) -> Iterable[int]:
    """Like geohash.expand_uint64, but enumerate all geohashes at precision instead of only returning [lo,hi) ranges"""

    # Special case: there are no 0-bit ints (would return [0] without this case)
    if precision_bits == 0:
        return []

    # Special case: _geohash.expand_uint64 fails to find neighbors when bits ≤ 2, so jump through bits+2 space instead
    elif precision_bits <= 2:
        return list(unique_everseen(
            int_at_prec(
                int(np.uint64(y) << np.uint64(2)),  # Bound left shift to uint64 (else negative ints)
                precision_bits,
            )
            for y in int_expand(x >> 2, precision_bits + 2)
        ))

    else:
        ret = []
        for lo, hi in _geohash.expand_uint64(x, precision_bits):
            if lo is None: lo = 0                    # Undo omission done by _geohash.expand_uint64
            if hi is None: hi = 0x10000000000000000  # Undo omission done by _geohash.expand_uint64
            coprecision_bits = 64 - precision_bits
            for y in range(lo >> coprecision_bits, hi >> coprecision_bits):
                y = int(np.uint64(y) << np.uint64(coprecision_bits))  # Bound left shift to uint64 (else negative ints)
                ret.append(y)
        return ret


def int_neighbors(x: int, precision_bits: int) -> Iterable[int]:
    """neighbors(x) + [x] = expand(x)"""
    x = int_at_prec(x, precision_bits)
    return [y for y in int_expand(x, precision_bits) if y != x]
