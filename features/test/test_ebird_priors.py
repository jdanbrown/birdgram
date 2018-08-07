from datetime import date

import numpy as np
import pandas as pd
import pytest

from ebird_priors import *


def test_ebird_week():

    # int
    assert ebird_week(-1) == 47
    assert ebird_week(0) == 0
    assert ebird_week(1) == 1
    assert ebird_week(2) == 2
    assert ebird_week(30) == 30
    assert ebird_week(47) == 47
    assert ebird_week(48) == 0
    assert ebird_week(1234) == 1234 % 48

    # date
    assert [ebird_week(date(2010, 1, d)) for d in [
        1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28,
        29, 30, 31,
    ]] == [
        0,  0,  0,  0,  0,  0,  0,
        1,  1,  1,  1,  1,  1,  1,
        2,  2,  2,  2,  2,  2,  2,
        3,  3,  3,  3,  3,  3,  3,
        3,  3,  3,
    ]
    assert [ebird_week(date(2010, 2, d)) for d in [
        1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28,
    ]] == [
        4,  4,  4,  4,  4,  4,  4,
        5,  5,  5,  5,  5,  5,  5,
        6,  6,  6,  6,  6,  6,  6,
        7,  7,  7,  7,  7,  7,  7,
    ]
    assert [ebird_week(date(2010, 12, d)) for d in [
        1,  2,  3,  4,  5,  6,  7,
        8,  9,  10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28,
        29, 30, 31,
    ]] == [
        44, 44, 44, 44, 44, 44, 44,
        45, 45, 45, 45, 45, 45, 45,
        46, 46, 46, 46, 46, 46, 46,
        47, 47, 47, 47, 47, 47, 47,
        47, 47, 47,
    ]


def test_ebird_week_to_date():
    # Property: ebird_week(ebird_week_to_date(week)) == week
    for week in range(0, 48):
        assert ebird_week(ebird_week_to_date(week)) == week


def test_week_smooth():

    ebird_priors = partial(EbirdPriors, geohash_binwidth='12mi')  # Dummy param

    ep = ebird_priors(week_binwidth=1)
    out = [
        (42, 42, [41, 42]),
        (43, 43, [42, 43]),
        (44, 44, [43, 44]),
        (45, 45, [44, 45]),
        (46, 46, [45, 46]),
        (47, 47, [46, 47]),
        ( 0,  0, [ 0, 47]),  # week_smooth: Changes every 1 week
        ( 1,  1, [ 0,  1]),
        ( 2,  2, [ 1,  2]),
        ( 3,  3, [ 2,  3]),
        ( 4,  4, [ 3,  4]),
        ( 5,  5, [ 4,  5]),
    ]
    assert out == [
        (week, ep._week_bin(ebird_week_to_date(week)), ep._week_smooth(ebird_week_to_date(week)))
        for week, week_bin, week_smooth in out
    ]

    ep = ebird_priors(week_binwidth=2)
    out = [
        (42, 42, [40, 42]),
        (43, 42, [42, 44]),
        (44, 44, [42, 44]),
        (45, 44, [44, 46]),
        (46, 46, [44, 46]),
        (47, 46, [ 0, 46]),
        ( 0,  0, [ 0, 46]),
        ( 1,  0, [ 0,  2]),  # week_smooth: Changes every 2 weeks at bin midpoints
        ( 2,  2, [ 0,  2]),
        ( 3,  2, [ 2,  4]),
        ( 4,  4, [ 2,  4]),
        ( 5,  4, [ 4,  6]),
    ]
    assert out == [
        (week, ep._week_bin(ebird_week_to_date(week)), ep._week_smooth(ebird_week_to_date(week)))
        for week, week_bin, week_smooth in out
    ]

    ep = ebird_priors(week_binwidth=3)
    out = [
        (42, 42, [39, 42]),
        (43, 42, [39, 42]),
        (44, 42, [42, 45]),
        (45, 45, [42, 45]),
        (46, 45, [42, 45]),
        (47, 45, [ 0, 45]),
        ( 0,  0, [ 0, 45]),
        ( 1,  0, [ 0, 45]),
        ( 2,  0, [ 0,  3]),  # week_smooth: Changes every 3 weeks at bin midpoints
        ( 3,  3, [ 0,  3]),
        ( 4,  3, [ 0,  3]),
        ( 5,  3, [ 3,  6]),
    ]
    assert out == [
        (week, ep._week_bin(ebird_week_to_date(week)), ep._week_smooth(ebird_week_to_date(week)))
        for week, week_bin, week_smooth in out
    ]

    ep = ebird_priors(week_binwidth=4)
    out = [
        (42, 40, [40, 44]),
        (43, 40, [40, 44]),
        (44, 44, [40, 44]),
        (45, 44, [40, 44]),
        (46, 44, [ 0, 44]),
        (47, 44, [ 0, 44]),
        ( 0,  0, [ 0, 44]),
        ( 1,  0, [ 0, 44]),
        ( 2,  0, [ 0,  4]),  # week_smooth: Changes every 4 weeks at bin midpoints
        ( 3,  0, [ 0,  4]),
        ( 4,  4, [ 0,  4]),
        ( 5,  4, [ 0,  4]),
    ]
    assert out == [
        (week, ep._week_bin(ebird_week_to_date(week)), ep._week_smooth(ebird_week_to_date(week)))
        for week, week_bin, week_smooth in out
    ]

    ep = ebird_priors(week_binwidth=6)
    out = [
        (42, 42, [36, 42]),
        (43, 42, [36, 42]),
        (44, 42, [36, 42]),
        (45, 42, [ 0, 42]),
        (46, 42, [ 0, 42]),
        (47, 42, [ 0, 42]),
        ( 0,  0, [ 0, 42]),
        ( 1,  0, [ 0, 42]),
        ( 2,  0, [ 0, 42]),
        ( 3,  0, [ 0, 6]),  # week_smooth: Changes every 6 weeks at bin midpoints
        ( 4,  0, [ 0, 6]),
        ( 5,  0, [ 0, 6]),
    ]
    assert out == [
        (week, ep._week_bin(ebird_week_to_date(week)), ep._week_smooth(ebird_week_to_date(week)))
        for week, week_bin, week_smooth in out
    ]

    ep = ebird_priors(week_binwidth=12)
    out = [
        (42, 36, [ 0, 36]),  # week_smooth: Changes every 12 weeks at bin midpoints
        (43, 36, [ 0, 36]),
        (44, 36, [ 0, 36]),
        (45, 36, [ 0, 36]),
        (46, 36, [ 0, 36]),
        (47, 36, [ 0, 36]),
        ( 0,  0, [ 0, 36]),
        ( 1,  0, [ 0, 36]),
        ( 2,  0, [ 0, 36]),
        ( 3,  0, [ 0, 36]),
        ( 4,  0, [ 0, 36]),
        ( 5,  0, [ 0, 36]),
    ]
    assert out == [
        (week, ep._week_bin(ebird_week_to_date(week)), ep._week_smooth(ebird_week_to_date(week)))
        for week, week_bin, week_smooth in out
    ]

    ep = ebird_priors(week_binwidth=24)
    out = [
        (42, 24, [ 0, 24]),  # week_smooth: Changes every 24 weeks at bin midpoints -> all smooths are [0, 24]
        (43, 24, [ 0, 24]),
        (44, 24, [ 0, 24]),
        (45, 24, [ 0, 24]),
        (46, 24, [ 0, 24]),
        (47, 24, [ 0, 24]),
        ( 0,  0, [ 0, 24]),
        ( 1,  0, [ 0, 24]),
        ( 2,  0, [ 0, 24]),
        ( 3,  0, [ 0, 24]),
        ( 4,  0, [ 0, 24]),
        ( 5,  0, [ 0, 24]),
    ]
    assert out == [
        (week, ep._week_bin(ebird_week_to_date(week)), ep._week_smooth(ebird_week_to_date(week)))
        for week, week_bin, week_smooth in out
    ]

    ep = ebird_priors(week_binwidth=48)
    out = [
        (42,  0, [0]),  # week_smooth: Changes every 48 weeks at bin midpoints -> all smooths are [0]
        (43,  0, [0]),
        (44,  0, [0]),
        (45,  0, [0]),
        (46,  0, [0]),
        (47,  0, [0]),
        ( 0,  0, [0]),
        ( 1,  0, [0]),
        ( 2,  0, [0]),
        ( 3,  0, [0]),
        ( 4,  0, [0]),
        ( 5,  0, [0]),
    ]
    assert out == [
        (week, ep._week_bin(ebird_week_to_date(week)), ep._week_smooth(ebird_week_to_date(week)))
        for week, week_bin, week_smooth in out
    ]

    # Property: all week smoothings should be exactly 2 week bins, except for binwidth=48 which only has 1 bin
    out = [
        (week_binwidth, week, 2 if week_binwidth != 48 else 1)
        for week_binwidth in EbirdPriors._week_binwidths
        for week in range(0, 48)
    ]
    assert out == [
        (week_binwidth, week, len(ebird_priors(week_binwidth=week_binwidth)._week_smooth(ebird_week_to_date(week))))
        for week_binwidth, week, len_bins in out
    ]


def test_geohash_bin():

    ebird_priors = partial(EbirdPriors, week_binwidth=1)  # Dummy param
    (lat, lon) = (37.9, -122)  # Mt. Diablo

    # Interactive map: http://geohash.gofreerange.com/
    assert ebird_priors(geohash_binwidth='12407mi')._geohash_bin(lat, lon) == '0'
    assert ebird_priors(geohash_binwidth= '6204mi')._geohash_bin(lat, lon) == '8'
    assert ebird_priors(geohash_binwidth= '3102mi')._geohash_bin(lat, lon) == '9'
    assert ebird_priors(geohash_binwidth= '1551mi')._geohash_bin(lat, lon) == '9h'
    assert ebird_priors(geohash_binwidth=  '775mi')._geohash_bin(lat, lon) == '9q'
    assert ebird_priors(geohash_binwidth=  '388mi')._geohash_bin(lat, lon) == '9q0'
    assert ebird_priors(geohash_binwidth=  '194mi')._geohash_bin(lat, lon) == '9q8'
    assert ebird_priors(geohash_binwidth=   '97mi')._geohash_bin(lat, lon) == '9q9'
    assert ebird_priors(geohash_binwidth=   '48mi')._geohash_bin(lat, lon) == '9q9h'
    assert ebird_priors(geohash_binwidth=   '24mi')._geohash_bin(lat, lon) == '9q9n'
    assert ebird_priors(geohash_binwidth=   '12mi')._geohash_bin(lat, lon) == '9q9ph'
    assert ebird_priors(geohash_binwidth=    '6mi')._geohash_bin(lat, lon) == '9q9pw'
    assert ebird_priors(geohash_binwidth=    '3mi')._geohash_bin(lat, lon) == '9q9px'
    assert ebird_priors(geohash_binwidth=  '1.5mi')._geohash_bin(lat, lon) == '9q9px8'
    assert ebird_priors(geohash_binwidth=   '.8mi')._geohash_bin(lat, lon) == '9q9pxf'
    assert ebird_priors(geohash_binwidth=   '.4mi')._geohash_bin(lat, lon) == '9q9pxg0'


def test_geohash_smooth():

    ebird_priors = partial(EbirdPriors, week_binwidth=1)

    # Interactive map: http://geohash.gofreerange.com/
    (lat, lon) = (37.9, -122)  # Mt. Diablo
    assert ebird_priors(geohash_binwidth='.4mi')._geohash_bin(lat, lon) == '9q9pxg0'
    out = [
        ('12407mi', ['0',       'h']),                              #  1 bit
        ( '6204mi', ['0',       '4',       '8',       'd']),        #  3 bit
        ( '3102mi', ['8',       '9',       'b',       'c']),        #  5 bit
        ( '1551mi', ['9h',      '9s',      'c0',      'c8']),       #  7 bit
        (  '775mi', ['9h',      '9k',      '9n',      '9q']),       #  9 bit
        (  '388mi', ['9nh',     '9ph',     '9q0',     '9r0']),      # 11 bit
        (  '194mi', ['9q0',     '9q4',     '9q8',     '9qd']),      # 13 bit
        (   '97mi', ['9q8',     '9q9',     '9qb',     '9qc']),      # 15 bit
        (   '48mi', ['9q8s',    '9q9h',    '9qb8',    '9qc0']),     # 17 bit
        (   '24mi', ['9q9n',    '9q9q',    '9qc0',    '9qc2']),     # 19 bit
        (   '12mi', ['9q9ph',   '9q9r0',   '9qc0h',   '9qc20']),    # 21 bit
        (    '6mi', ['9q9pn',   '9q9pw',   '9q9r0',   '9q9r8']),    # 23 bit
        (    '3mi', ['9q9pr',   '9q9px',   '9q9r2',   '9q9r8']),    # 25 bit
        (  '1.5mi', ['9q9px8',  '9q9pxs',  '9q9r80',  '9q9r8h']),   # 27 bit
        (   '.8mi', ['9q9pxd',  '9q9pxf',  '9q9pxs',  '9q9pxu']),   # 29 bit
        (   '.4mi', ['9q9pxf0', '9q9pxfh', '9q9pxg0', '9q9pxgh']),  # 31 bit
    ]
    # Check out == _geohash_smooth
    assert out == [
        (binwidth, ebird_priors(geohash_binwidth=binwidth)._geohash_smooth(lat, lon))
        for binwidth, bins in out
    ]
    # Check bin == str_at_prec(bin) for each bin
    assert [
        (binwidth, bits, bins)
        for binwidth, bins in out
        for bits in [ebird_priors(geohash_binwidth=binwidth).geohash_precision_bits]
    ] == [
        (binwidth, bits, [geoh.str_at_prec(bin, bits) for bin in bins])
        for binwidth, bins in out
        for bits in [ebird_priors(geohash_binwidth=binwidth).geohash_precision_bits]
    ]
