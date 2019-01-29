"""
API adapter for python-geohash
- Docs: https://github.com/hkwi/python-geohash/wiki/GeohashReference
- Interactive map: http://geohash.gofreerange.com/
- Interactive map: https://www.movable-type.co.uk/scripts/geohash.html
"""

from functools import partial
import numpy as np
import pytest

from more_itertools import sliced
from toolz import valmap

import geoh


def approx_f(x, digits=6):
    return float('%%.%sf' % digits % x)


def approx_fs(xs, digits=6):
    if isinstance(xs, dict):
        return valmap(partial(approx_f, digits=digits), xs)
    else:
        return type(xs)(map(partial(approx_f, digits=digits), xs))


def int_to_bin_str(x, bits=64, **kwargs):
    _bin_str_space = lambda b, grouping=5, leading=0: ' '.join([*b[:leading], *sliced(b[leading:], grouping)])
    return _bin_str_space(np.binary_repr(int(x) >> (64 - bits), width=bits), **kwargs)


def unB(s):
    s = s.replace(' ', '')
    return eval('0b%s' % (s + ('0' * (64 - len(s)))))


def B(x, bits) -> str:
    if isinstance(x, str):
        x = unB(x)
    return int_to_bin_str(x, bits)


def test_str_to_int_and_int_to_str():
    out = [
        (64, '9q9pxg6r9bnd6', '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 01100 0011'),
        (64, '9b', '01001 01010', '9b00000000000', '01001 01010 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000 0000'),
        (60, '9b', '01001 01010', '9b0000000000',  '01001 01010 00000 00000 00000 00000 00000 00000 00000 00000 00000 00000'),
        (15, '9b', '01001 01010', '9b0',           '01001 01010 00000'),
        (11, '9b', '01001 01010', '9b0',           '01001 01010 0'),
        (10, '9b', '01001 01010', '9b',            '01001 01010'),
        (9,  '9b', '01001 01010', '9b',            '01001 0101'),
        (8,  '9b', '01001 01010', '98',            '01001 010'),
        (7,  '9b', '01001 01010', '98',            '01001 01'),
        (6,  '9b', '01001 01010', '90',            '01001 0'),
        (5,  '9b', '01001 01010', '9',             '01001'),
        (4,  '9b', '01001 01010', '8',             '0100'),
        (3,  '9b', '01001 01010', '8',             '010'),
        (2,  '9b', '01001 01010', '8',             '01'),
        (1,  '9b', '01001 01010', '0',             '0'),
        (0,  '9b', '01001 01010', '',              '0'),  # Empty str (s[:0]), zero int (x >> 64)
    ]
    # Expand (bits, s, i) -> (bits, s, i, s_out, i_out)
    out = [(*xs, *xs[1:3]) if len(xs) == 3 else xs for xs in out]
    assert [
        (bits, s, i, s_out, i_out)
        for bits, s, i, s_out, i_out in out
    ] == [
        (bits, s, i, geoh.int_to_str(unB(i), bits), B(geoh.str_to_int(s, bits), bits))
        for bits, s, i, s_out, i_out in out
    ]


def test_str_encode_and_at_prec():
    (lat, lon) = (37.9, -122)  # Mt. Diablo
    geohash = '9q9pxg6r9bnd6'
    out = [
        # Interactive map: http://geohash.gofreerange.com/
        (64, '9q9pxg6r9bnd6'),
        (63, '9q9pxg6r9bnd4'),
        (62, '9q9pxg6r9bnd0'),
        (61, '9q9pxg6r9bnd0'),
        (60, '9q9pxg6r9bnd'),
        (59, '9q9pxg6r9bnd'),
        (58, '9q9pxg6r9bnd'),
        (57, '9q9pxg6r9bn8'),
        (56, '9q9pxg6r9bn0'),
        (55, '9q9pxg6r9bn'),
        (54, '9q9pxg6r9bn'),
        (53, '9q9pxg6r9bn'),
        (52, '9q9pxg6r9bh'),
        (51, '9q9pxg6r9bh'),
        (50, '9q9pxg6r9b'),
        (49, '9q9pxg6r9b'),
        (48, '9q9pxg6r98'),
        (47, '9q9pxg6r98'),
        (46, '9q9pxg6r90'),
        (45, '9q9pxg6r9'),
        (44, '9q9pxg6r8'),
        (43, '9q9pxg6r8'),
        (42, '9q9pxg6r8'),
        (41, '9q9pxg6r0'),
        (40, '9q9pxg6r'),
        (39, '9q9pxg6q'),
        (38, '9q9pxg6n'),
        (37, '9q9pxg6h'),
        (36, '9q9pxg6h'),
        (35, '9q9pxg6'),
        (34, '9q9pxg6'),
        (33, '9q9pxg4'),
        (32, '9q9pxg0'),
        (31, '9q9pxg0'),
        (30, '9q9pxg'),
        (29, '9q9pxf'),
        (28, '9q9pxd'),
        (27, '9q9px8'),
        (26, '9q9px0'),
        (25, '9q9px'),
        (24, '9q9pw'),
        (23, '9q9pw'),
        (22, '9q9ps'),
        (21, '9q9ph'),
        (20, '9q9p'),
        (19, '9q9n'),
        (18, '9q9n'),
        (17, '9q9h'),
        (16, '9q9h'),
        (15, '9q9'),
        (14, '9q8'),
        (13, '9q8'),
        (12, '9q8'),
        (11, '9q0'),
        (10, '9q'),
        (9,  '9q'),
        (8,  '9n'),
        (7,  '9h'),
        (6,  '9h'),
        (5,  '9'),
        (4,  '8'),
        (3,  '8'),
        (2,  '8'),
        (1,  '0'),
        (0,  ''),
    ]
    assert out == [
        (bits, geoh.str_encode(lat, lon, bits))
        for bits, _ in out
    ]
    assert out == [
        (bits, geoh.str_at_prec(geohash, bits))
        for bits, _ in out
    ]


def test_str_at_prec_edge_cases():
    out = [
        # Interactive map: http://geohash.gofreerange.com/
        ('9q9pxg6r9bnd6', 71, '9q9pxg6r9bnd600'),
        ('9q9pxg6r9bnd6', 70, '9q9pxg6r9bnd60'),
        ('9q9pxg6r9bnd6', 69, '9q9pxg6r9bnd60'),
        ('9q9pxg6r9bnd6', 68, '9q9pxg6r9bnd60'),
        ('9q9pxg6r9bnd6', 67, '9q9pxg6r9bnd60'),
        ('9q9pxg6r9bnd6', 66, '9q9pxg6r9bnd60'),
        ('9q9pxg6r9bnd6', 65, '9q9pxg6r9bnd6'),
        ('9q9pxg6r9bnd6', 64, '9q9pxg6r9bnd6'),
        ('9q9pxg6r9bnd6', 0,  ''),
    ]
    assert out == [
        (geohash, bits, geoh.str_at_prec(geohash, bits))
        for geohash, bits, _ in out
    ]


def test_str_at_prec_enumerate_all_small_cases():
    xs = {
        1: ['0', 'h'],
        2: ['0', '8', 'h', 's'],
        3: ['0', '4', '8', 'd', 'h', 'n', 's', 'w'],
        4: ['0', '2', '4', '6', '8', 'b', 'd', 'f', 'h', 'k', 'n', 'q', 's', 'u', 'w', 'y'],
        5: [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
            'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        ],
    }
    out = [
        (0,  ['']),
        (1,  xs[1]),
        (2,  xs[2]),
        (3,  xs[3]),
        (4,  xs[4]),
        (5,  xs[5]),
        (6,  [c + x for c in xs[5] for x in xs[1]]),
        (7,  [c + x for c in xs[5] for x in xs[2]]),
        (8,  [c + x for c in xs[5] for x in xs[3]]),
        (9,  [c + x for c in xs[5] for x in xs[4]]),
        (10, [c + x for c in xs[5] for x in xs[5]]),
    ]
    assert out == [
        (bits, sorted(set(
            geoh.str_at_prec(x + y, bits)
            for x in geoh._int_to_char
            for y in geoh._int_to_char
        )))
        for bits, _ in out
    ]


def test_int_encode():
    (lat, lon) = (37.9, -122)  # Mt. Diablo
    out = [
        (64, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 01100 0011'),
        (63, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 01100 001'),
        (62, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 01100 00'),
        (61, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 01100 0'),
        (60, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 01100'),
        (59, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 0110'),
        (58, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 011'),
        (57, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 01'),
        (56, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100 0'),
        (55, '01001 10110 01001 10101 11101 01111 00110 10111 01001 01010 10100'),
        (10, '01001 10110'),
        (9,  '01001 1011'),
        (8,  '01001 101'),
        (7,  '01001 10'),
        (6,  '01001 1'),
        (5,  '01001'),
        (4,  '0100'),
        (3,  '010'),
        (2,  '01'),
        (1,  '0'),
        (0,  '0'),
    ]
    assert [
        (bits, B(i, bits))
        for bits, i in out
    ] == [
        (bits, B(geoh.int_encode(lat, lon, bits), bits))
        for bits, _ in out
    ]


def test_int_at_prec():
    out = [
        (64, '01001 10110 01001', '01001 10110 01001 00000 00000 00000 00000 00000 00000 00000 00000 00000 0000'),
        (16, '01001 10110 01001', '01001 10110 01001 0'),
        (15, '01001 10110 01001', '01001 10110 01001'),
        (8,  '01001 10110 01001', '01001 101'),
        (5,  '01001 10110 01001', '01001'),
        (1,  '01001 10110 01001', '01'),
        (1,  '01001 10110 01001', '0'),
        (0,  '01001 10110 01001', '0'),
    ]
    assert [
        (bits, i, B(j, bits))
        for bits, i, j in out
    ] == [
        (bits, i, B(geoh.int_at_prec(unB(i), bits), bits))
        for bits, i, j in out
    ]


def test_str_decode_origin():
    out = [
        # Interactive map: http://geohash.gofreerange.com/
        ('9q9pxg6r9bnd6', ( 37.900000, -122.000000)),
        ('9q9pxg6r9bnd4', ( 37.900000, -122.000000)),
        ('9q9pxg6r9bnd0', ( 37.900000, -122.000000)),
        ('9q9pxg6r9bnd',  ( 37.900000, -122.000000)),
        ('9q9pxg6r9bn8',  ( 37.900000, -122.000000)),
        ('9q9pxg6r9bn0',  ( 37.900000, -122.000001)),
        ('9q9pxg6r9bn',   ( 37.900000, -122.000001)),
        ('9q9pxg6r9bh',   ( 37.900000, -122.000003)),
        ('9q9pxg6r9b',    ( 37.900000, -122.000009)),
        ('9q9pxg6r98',    ( 37.900000, -122.000020)),
        ('9q9pxg6r90',    ( 37.900000, -122.000041)),
        ('9q9pxg6r9',     ( 37.900000, -122.000041)),
        ('9q9pxg6r8',     ( 37.900000, -122.000084)),
        ('9q9pxg6r0',     ( 37.899914, -122.000084)),
        ('9q9pxg6r',      ( 37.899914, -122.000084)),
        ('9q9pxg6q',      ( 37.899742, -122.000084)),
        ('9q9pxg6n',      ( 37.899742, -122.000427)),
        ('9q9pxg6h',      ( 37.899399, -122.000427)),
        ('9q9pxg6',       ( 37.898712, -122.000427)),
        ('9q9pxg4',       ( 37.897339, -122.000427)),
        ('9q9pxg0',       ( 37.897339, -122.003174)),
        ('9q9pxg',        ( 37.897339, -122.003174)),
        ('9q9pxf',        ( 37.891846, -122.003174)),
        ('9q9pxd',        ( 37.891846, -122.014160)),
        ('9q9px8',        ( 37.880859, -122.014160)),
        ('9q9px0',        ( 37.880859, -122.036133)),
        ('9q9px',         ( 37.880859, -122.036133)),
        ('9q9pw',         ( 37.880859, -122.080078)),
        ('9q9ps',         ( 37.880859, -122.167969)),
        ('9q9ph',         ( 37.792969, -122.167969)),
        ('9q9p',          ( 37.792969, -122.343750)),
        ('9q9n',          ( 37.617188, -122.343750)),
        ('9q9h',          ( 37.265625, -122.343750)),
        ('9q9',           ( 36.562500, -122.343750)),
        ('9q8',           ( 36.562500, -123.750000)),
        ('9q0',           ( 33.750000, -123.750000)),
        ('9q',            ( 33.750000, -123.750000)),
        ('9n',            ( 33.750000, -135.000000)),
        ('9h',            ( 22.500000, -135.000000)),
        ('9',             (  0.000000, -135.000000)),
        ('8',             (  0.000000, -180.000000)),
        ('0',             (-90.000000, -180.000000)),
        ('',              (-90.000000, -180.000000)),
    ]
    approx = lambda out: [(geohash, (approx_f(lat), approx_f(lon))) for geohash, (lat, lon) in out]
    assert approx(out) == approx([
        (geohash, geoh.str_decode_origin(geohash))
        for geohash, _ in out
    ])


def test_int_decode_origin():
    out = [
        (45, '9q9pxg6r9', '01001 10110 01001 10101 11101 01111 00110 10111 01001', ( 37.90, -122.00)),
        (40, '9q9pxg6r',  '01001 10110 01001 10101 11101 01111 00110 10111 00000', ( 37.90, -122.00)),
        (35, '9q9pxg6',   '01001 10110 01001 10101 11101 01111 00110 00000 00000', ( 37.90, -122.00)),
        (30, '9q9pxg',    '01001 10110 01001 10101 11101 01111 00000 00000 00000', ( 37.90, -122.00)),
        (25, '9q9px',     '01001 10110 01001 10101 11101 00000 00000 00000 00000', ( 37.88, -122.04)),
        (20, '9q9p',      '01001 10110 01001 10101 00000 00000 00000 00000 00000', ( 37.79, -122.34)),
        (15, '9q9',       '01001 10110 01001 00000 00000 00000 00000 00000 00000', ( 36.56, -122.34)),
        (10, '9q',        '01001 10110 00000 00000 00000 00000 00000 00000 00000', ( 33.75, -123.75)),
        (5,  '9',         '01001 00000 00000 00000 00000 00000 00000 00000 00000', (  0.00, -135.00)),
        (0,  '',          '00000 00000 00000 00000 00000 00000 00000 00000 00000', (-90.00, -180.00)),
    ]
    assert [
        (bits, s, i, approx_fs(digits=2, xs=(lat, lon)))
        for bits, s, i, (lat, lon) in out
    ] == [
        (bits, s, i, approx_fs(digits=2, xs=geoh.int_decode_origin(unB(i))))
        for bits, s, i, (lat, lon) in out
    ]


def test_str_expand_and_neighbors():
    out = [
        # Interactive map: http://geohash.gofreerange.com/
        ('9b',  15, '9b0', ['3xz', '3zb', '3zc', '98p', '98r', '9b1', '9b2', '9b3']),
        ('9b',  14, '9b0', ['3xy', '3zb', '3zf', '98n', '98q', '9b2', '9b4', '9b6']),
        ('9b',  13, '9b0', ['3xw', '3z8', '3zd', '98n', '98w', '9b4', '9b8', '9bd']),
        ('9b',  12, '9b0', ['3xs', '3z8', '3zs', '98h', '98s', '9b8', '9bh', '9bs']),
        ('9b',  11, '9b0', ['3xh', '3z0', '3zh', '98h', '99h', '9bh', '9c0', '9ch']),
        ('9b',  10, '9b',  ['3x',  '3z',  '6p',  '98',  '99',  '9c',  'd0',  'd1']),
        ('9b',  9,  '9b',  ['3w',  '3y',  '6n',  '98',  '9d',  '9f',  'd0',  'd4']),
        ('9b',  8,  '98',  ['3n',  '3w',  '6n',  '90',  '94',  '9d',  'd0',  'd4']),
        ('9b',  7,  '98',  ['3h',  '3s',  '6h',  '90',  '9h',  '9s',  'd0',  'dh']),
        ('9b',  6,  '90',  ['2h',  '3h',  '6h',  '80',  '8h',  '9h',  'd0',  'dh']),
        ('9b',  5,  '9',   ['2',   '3',   '6',   '8',   'b',   'c',   'd',   'f']),
        ('99',  9,  '98',  ['3q',  '3w',  '3y',  '92',  '96',  '9b',  '9d',  '9f']),
        ('9b',  4,  '8',   ['2',   '6',   'b',   'd',   'f',   'q',   'w',   'y']),
        ('9b',  3,  '8',   ['0',   '4',   'd',   'n',   'w']),
        ('9b',  2,  '8',   ['0',   'h',   's']),
        ('9b',  1,  '0',   ['h']),
        ('9b',  0,  '',    []),
    ]
    # Check neighbors(x) + [x] = expand(x)
    assert [
        (geohash, bits, geoh.str_at_prec(geohash, bits), sorted([
            *geoh.str_neighbors(geohash, bits),
            geoh.str_at_prec(geohash, bits),
        ]))
        for geohash, bits, geohash_at_prec, neighbors in out
        if bits > 0
    ] == [
        (geohash, bits, geoh.str_at_prec(geohash, bits), sorted(geoh.str_expand(geohash, bits)))
        for geohash, bits, geohash_at_prec, neighbors in out
        if bits > 0
    ]
    # Check out == expand
    assert [
        (geohash, bits, geohash_at_prec, sorted([*neighbors, *([geohash_at_prec] if bits > 0 else [])]))
        for geohash, bits, geohash_at_prec, neighbors in out
    ] == [
        (geohash, bits, geoh.str_at_prec(geohash, bits), sorted(geoh.str_expand(geohash, bits)))
        for geohash, bits, geohash_at_prec, neighbors in out
    ]
    # Check out == neighbors
    assert out == [
        (geohash, bits, geoh.str_at_prec(geohash, bits), sorted(geoh.str_neighbors(geohash, bits)))
        for geohash, bits, geohash_at_prec, neighbors in out
    ]


def test_int_expand_and_neighbors():
    out = [
        ('01001 10110',  5,  '01001', ['00010', '00011', '00110', '01000', '01010', '01011', '01100', '01110']),
        ('01001 10110',  4,  '0100',  ['0001', '0011', '0101', '0110', '0111', '1011', '1110', '1111']),
        ('01001 10110',  3,  '010',   ['000', '001', '011', '101', '111']),
        ('01001 10110',  2,  '01',    ['00', '10', '11']),
        ('01001 10110',  1,  '0',     ['1']),
        ('01001 10110',  0,  '0',     []),
    ]
    # Check neighbors(x) + [x] = expand(x)
    assert [
        (i, bits, B(geoh.int_at_prec(unB(i), bits), bits), sorted(B(j, bits) for j in [
            *geoh.int_neighbors(unB(i), bits),
            geoh.int_at_prec(unB(i), bits),
        ]))
        for i, bits, i_at_prec, neighbors in out
        if bits > 0
    ] == [
        (i, bits, B(geoh.int_at_prec(unB(i), bits), bits), sorted(B(j, bits) for j in geoh.int_expand(unB(i), bits)))
        for i, bits, i_at_prec, neighbors in out
        if bits > 0
    ]
    # Check out == expand
    assert [
        (i, bits, i_at_prec, sorted([*neighbors, *([i_at_prec] if bits > 0 else [])]))
        for i, bits, i_at_prec, neighbors in out
    ] == [
        (i, bits, B(geoh.int_at_prec(unB(i), bits), bits), sorted(B(j, bits) for j in geoh.int_expand(unB(i), bits)))
        for i, bits, i_at_prec, neighbors in out
    ]
    # Check out == neighbors
    assert out == [
        (i, bits, B(geoh.int_at_prec(unB(i), bits), bits), sorted(B(j, bits) for j in geoh.int_neighbors(unB(i), bits)))
        for i, bits, i_at_prec, neighbors in out
    ]


def test_bbox():
    out = [
        # Interactive map: http://geohash.gofreerange.com/
        (37.9, -122)
    ]


def test_bbox():
    out = [
        # Interactive map: http://geohash.gofreerange.com/

        # Mt. Diablo
        (32, '9q9pxg0', '01001 10110 01001 10101 11101 01111 00', dict(s= 37.90, w=-122.00, n= 37.90, e=-122.00)),
        (31, '9q9pxg0', '01001 10110 01001 10101 11101 01111 0',  dict(s= 37.90, w=-122.00, n= 37.90, e=-122.00)),
        (30, '9q9pxg',  '01001 10110 01001 10101 11101 01111',    dict(s= 37.90, w=-122.00, n= 37.90, e=-121.99)),
        (29, '9q9pxf',  '01001 10110 01001 10101 11101 0111',     dict(s= 37.89, w=-122.00, n= 37.90, e=-121.99)),
        (28, '9q9pxd',  '01001 10110 01001 10101 11101 011',      dict(s= 37.89, w=-122.01, n= 37.90, e=-121.99)),
        (27, '9q9px8',  '01001 10110 01001 10101 11101 01',       dict(s= 37.88, w=-122.01, n= 37.90, e=-121.99)),
        (26, '9q9px0',  '01001 10110 01001 10101 11101 0',        dict(s= 37.88, w=-122.04, n= 37.90, e=-121.99)),
        (25, '9q9px',   '01001 10110 01001 10101 11101',          dict(s= 37.88, w=-122.04, n= 37.92, e=-121.99)),
        (24, '9q9pw',   '01001 10110 01001 10101 1110',           dict(s= 37.88, w=-122.08, n= 37.92, e=-121.99)),
        (23, '9q9pw',   '01001 10110 01001 10101 111',            dict(s= 37.88, w=-122.08, n= 37.97, e=-121.99)),
        (22, '9q9ps',   '01001 10110 01001 10101 11',             dict(s= 37.88, w=-122.17, n= 37.97, e=-121.99)),
        (21, '9q9ph',   '01001 10110 01001 10101 1',              dict(s= 37.79, w=-122.17, n= 37.97, e=-121.99)),
        (20, '9q9p',    '01001 10110 01001 10101',                dict(s= 37.79, w=-122.34, n= 37.97, e=-121.99)),
        (19, '9q9n',    '01001 10110 01001 1010',                 dict(s= 37.62, w=-122.34, n= 37.97, e=-121.99)),
        (18, '9q9n',    '01001 10110 01001 101',                  dict(s= 37.62, w=-122.34, n= 37.97, e=-121.64)),
        (17, '9q9h',    '01001 10110 01001 10',                   dict(s= 37.27, w=-122.34, n= 37.97, e=-121.64)),
        (16, '9q9h',    '01001 10110 01001 1',                    dict(s= 37.27, w=-122.34, n= 37.97, e=-120.94)),
        (15, '9q9',     '01001 10110 01001',                      dict(s= 36.56, w=-122.34, n= 37.97, e=-120.94)),
        (14, '9q8',     '01001 10110 0100',                       dict(s= 36.56, w=-123.75, n= 37.97, e=-120.94)),
        (13, '9q8',     '01001 10110 010',                        dict(s= 36.56, w=-123.75, n= 39.38, e=-120.94)),
        (12, '9q8',     '01001 10110 01',                         dict(s= 36.56, w=-123.75, n= 39.38, e=-118.12)),
        (11, '9q0',     '01001 10110 0',                          dict(s= 33.75, w=-123.75, n= 39.38, e=-118.12)),
        (10, '9q',      '01001 10110',                            dict(s= 33.75, w=-123.75, n= 39.38, e=-112.50)),
        (9,  '9q',      '01001 1011',                             dict(s= 33.75, w=-123.75, n= 45.00, e=-112.50)),
        (8,  '9n',      '01001 101',                              dict(s= 33.75, w=-135.00, n= 45.00, e=-112.50)),
        (7,  '9h',      '01001 10',                               dict(s= 22.50, w=-135.00, n= 45.00, e=-112.50)),
        (6,  '9h',      '01001 1',                                dict(s= 22.50, w=-135.00, n= 45.00, e= -90.00)),
        (5,  '9',       '01001',                                  dict(s=  0.00, w=-135.00, n= 45.00, e= -90.00)),
        (4,  '8',       '0100',                                   dict(s=  0.00, w=-180.00, n= 45.00, e= -90.00)),
        (3,  '8',       '010',                                    dict(s=  0.00, w=-180.00, n= 90.00, e= -90.00)),
        (2,  '8',       '01',                                     dict(s=  0.00, w=-180.00, n= 90.00, e=   0.00)),
        (1,  '0',       '0',                                      dict(s=-90.00, w=-180.00, n= 90.00, e=   0.00)),

        # Map halves (big)
        (1,  '0',       '0',                                      dict(s=-90.00, w=-180.00, n= 90.00, e=   0.00)),
        (1,  'h',       '1',                                      dict(s=-90.00, w=   0.00, n= 90.00, e= 180.00)),

        # Map quarters (big)
        (2,  '0',       '00',                                     dict(s=-90.00, w=-180.00, n=  0.00, e=   0.00)),
        (2,  '8',       '01',                                     dict(s=  0.00, w=-180.00, n= 90.00, e=   0.00)),
        (2,  'h',       '10',                                     dict(s=-90.00, w=   0.00, n=  0.00, e= 180.00)),
        (2,  's',       '11',                                     dict(s=  0.00, w=   0.00, n= 90.00, e= 180.00)),

        # Map corners (tiny)
        (60, '00'*6,    ('00000 00000 '*6).rstrip(),              dict(s=-90.00, w=-180.00, n=-90.00, e=-180.00)),
        (60, 'bp'*6,    ('01010 10101 '*6).rstrip(),              dict(s= 90.00, w=-180.00, n= 90.00, e=-180.00)),
        (60, 'pb'*6,    ('10101 01010 '*6).rstrip(),              dict(s=-90.00, w= 180.00, n=-90.00, e= 180.00)),
        (60, 'zz'*6,    ('11111 11111 '*6).rstrip(),              dict(s= 90.00, w= 180.00, n= 90.00, e= 180.00)),

    ]
    # Validate str == int
    assert out == [
        (bits, geoh.int_to_str(unB(i), bits), B(geoh.str_to_int(s, bits), bits), bbox)
        for bits, s, i, bbox in out
    ]
    # Check str_bbox
    assert [
        (bits, s, i, approx_fs(digits=2, xs=bbox))
        for bits, s, i, bbox in out
    ] == [
        (bits, s, i, approx_fs(digits=2, xs=geoh.str_bbox(s, bits)))
        for bits, s, i, bbox in out
    ]
    # Check int_bbox (via str_to_int)
    assert [
        (bits, s, i, approx_fs(digits=2, xs=bbox))
        for bits, s, i, bbox in out
    ] == [
        (bits, s, i, approx_fs(digits=2, xs=geoh.int_bbox(unB(i), bits)))
        for bits, s, i, bbox in out
    ]
