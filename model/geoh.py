"""
API adapter for python-geohash
- Docs: https://github.com/hkwi/python-geohash/wiki/GeohashReference
- Interactive map: http://geohash.gofreerange.com/
- Interactive map: https://www.movable-type.co.uk/scripts/geohash.html
"""

from functools import partial
from typing import Iterable, Mapping

from more_itertools import first, one, unique_everseen
import numpy as np
from tqdm import tqdm

import geohash as _geohash
from util import generator_to, np_vectorize_asscalar

# Vectorize everything?
#   - For *_decode, np.vectorize barfs on multiple return values -- and adds no value on non-multiple input args
#       - e.g. np_vectorize_asscalar(int_decode, otypes=[np.float64, np.float64])


# @partial(np_vectorize_asscalar, otypes=[str])  # TODO Disabled to avoid incidental complexities; restore when we care (rely on tests)
def str_encode(lat: float, lon: float, precision_bits: int) -> str:
    """_geohash.encode with precision_bits"""
    return str_at_prec(_geohash.encode(lat, lon, (precision_bits + 4) // 5), precision_bits)


# @partial(np_vectorize_asscalar, otypes=[np.uint64])  # TODO Generalize downstream code to not barf on np.uint64 (rely on tests)
def int_encode(lat: float, lon: float, precision_bits: int) -> int:
    """_geohash.encode_uint64 with precision_bits"""
    return int_at_prec(_geohash.encode_uint64(lat, lon), precision_bits)


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


def int_at_prec(x: int, precision_bits: int, int_bits: int = 64) -> int:
    return x >> (int_bits - precision_bits) << (int_bits - precision_bits)


def str_decode_origin(x: str) -> (float, float):
    """Decode (lat, lon) at bottom-left of geohash (independent of precision, since bottom-left is '<x>0...')"""
    # return _geohash.decode(x)  # Nope: this one returns the center, using an implicit precision=len(s)
    (lat, lon, _, _) = _geohash._geohash.decode(x)  # Have to reach into the native module to get the bottom-left
    return (lat, lon)


def int_decode_origin(x: int) -> (float, float):
    """Decode (lat, lon) at bottom-left of geohash (independent of precision, since bottom-left is '<x>0...')"""
    return _geohash.decode_uint64(x)  # No tricks here, this one returns the bottom-left (like the native module)


def str_expand(x: str, precision_bits: int) -> Iterable[str]:
    """_geohash.expand with precision_bits"""
    return [
        int_to_str(y, precision_bits)
        for y in int_expand(str_to_int(x, precision_bits), precision_bits)
    ]


def int_expand(x: int, precision_bits: int) -> Iterable[int]:
    """Like _geohash.expand_uint64, but enumerate all geohashes at precision instead of only returning [lo,hi) ranges"""

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


def str_neighbors(x: str, precision_bits: int) -> Iterable[str]:
    """_geohash.neighbors with precision_bits: neighbors(x) + [x] = expand(x)"""
    x = str_at_prec(x, precision_bits)
    return [y for y in str_expand(x, precision_bits) if y != x]


def int_neighbors(x: int, precision_bits: int) -> Iterable[int]:
    """_geohash.neighbors for ints: neighbors(x) + [x] = expand(x)"""
    x = int_at_prec(x, precision_bits)
    return [y for y in int_expand(x, precision_bits) if y != x]


def str_bbox(x: str, precision_bits: int) -> dict:
    """_geohash.bbox with precision_bits"""
    return int_bbox(str_to_int(x, precision_bits), precision_bits)


def int_bbox(x: int, precision_bits: int) -> dict:
    """_geohash.bbox for ints"""
    (lat_s, lon_w) = int_decode_origin(x)
    neighbors = [int_decode_origin(y) for y in int_neighbors(x, precision_bits)]
    lat_n = first([lat_n for lat_n, _ in neighbors if lat_n > lat_s] or [90])
    lon_e = first([lon_e for _, lon_e in neighbors if lon_e > lon_w] or [180])
    return dict(
        s=lat_s, w=lon_w,
        n=lat_n, e=lon_e,
    )


def str_to_int(x: str, precision_bits: int) -> int:
    return int_encode(*str_decode_origin(x), precision_bits)


def int_to_str(x: int, precision_bits: int) -> str:
    return str_encode(*int_decode_origin(x), precision_bits)


def all_geohashes_with_expand(precision_bits: int, progress=True) -> Mapping[str, Iterable[str]]:
    """
    Enumerate all geohash strs with their expands (= neighbors + self), at the given precision
    - Return a dict: geohash -> expand(geohash)
    - This approach is really dumb and slow, but works well enough for now
    - Approx runtimes:
        - precision_bits=1  -> len[   2] time[0s]
        - precision_bits=3  -> len[   8] time[0s]
        - precision_bits=5  -> len[  32] time[0s]
        - ...
        - precision_bits=13 -> len[  8k] time[1.2s]
        - precision_bits=15 -> len[ 32k] time[2.8s]
        - precision_bits=17 -> len[128k] time[11s]
        - precision_bits=19 -> len[512k] time[42s]
        - precision_bits=21 -> len[  2m] time[TODO (~160s)]
        - precision_bits=23 -> len[  8m] time[TODO (~640s)]
    """
    expands = {}
    to_expand = str_expand('0', precision_bits)  # '0' isn't necessarily at right precision, so don't include in output
    progress = progress and tqdm(total=2**precision_bits)
    while to_expand:
        x = to_expand.pop(0)
        if x not in expands:
            expands[x] = str_expand(x, precision_bits)
            to_expand.extend(expands[x])
            progress and progress.update()
    progress and progress.close()
    return expands
