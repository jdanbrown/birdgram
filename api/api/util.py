from collections import namedtuple
from haversine import haversine as _haversine
import os
import pipes
import re


class LonLat(namedtuple('LonLat_tuple', ['lon', 'lat'])):

    def __new__(cls, *args):
        types = [type(x) for x in args]
        if types == [str]:
            (s,) = args
            args = [float(x) for x in s.split(',')]
        return super().__new__(cls, *args)


test_lonlats = dict(
    home=LonLat(-122.440698, 37.741418),
)


def mkdir_p(dir):
    os.system("mkdir -p %s" % pipes.quote(dir))  # Don't error like os.makedirs


def flatten(xss):
    return [x for xs in xss for x in xs]


def haversine(lonlat_a: LonLat, lonlat_b: LonLat, km_or_miles='km'):
    return _haversine(
        (lonlat_a.lat, lonlat_a.lon),
        (lonlat_b.lat, lonlat_b.lon),
        miles={'km': False, 'miles': True}[km_or_miles],
    )


# TODO(dan) Submit PR to okunishinishi/python-stringcase
def snakecase(s: str) -> str:
    """
    Adapted from stringcase to fix `locID` -> `loc_i_d`:
    - https://github.com/okunishinishi/python-stringcase/blob/09557d4/stringcase.py
    - https://github.com/okunishinishi/python-stringcase/issues/4
    """
    s = re.sub(r'[-.\s]', '_', s)  # e.g. 'Foo-Bar Baz' -> 'Foo_Bar_Baz'
    s = re.sub(r'([A-Z][A-Z]+)(?=[A-Z])', r'\1_', s)  # e.g. 'OneTWOThreeFour' -> 'OneTWO_ThreeFour'
    s = re.sub(r'([^A-Z])(?=[A-Z])', r'\1_', s)  # e.g. 'OneTWO_ThreeFour' -> 'One_TWO_Three_Four'
    s = re.sub(r'_+', '_', s)  # e.g. 'Foo-Bar Baz' -> 'Foo_Bar_Baz' -> 'Foo__Bar__Baz' -> 'Foo_Bar_Baz'
    s = s.lower()  # e.g. 'One_TWO_Three_Four' -> 'one_two_three_four'
    return s
