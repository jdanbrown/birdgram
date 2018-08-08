import calendar
from collections import Counter
from dataclasses import dataclass
import datetime
from datetime import date
from functools import lru_cache
from typing import Iterable, List, Optional, Set, Tuple, Union

import matplotlib as mpl
from more_itertools import unique_everseen
import numpy as np
import pandas as pd
import parse
from potoo.pandas import df_reverse_cat
from potoo.plot import *
from potoo.util import tap
import sklearn as sk

import geoh
import metadata
from util import *


# https://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts
# https://help.ebird.org/customer/en/portal/articles/1210247-what-is-frequency-
n_ebird_weeks = 48
ebird_weeks = np.arange(n_ebird_weeks)
ebird_bar_widths = {
    .600: 9,
    .400: 8,
    .300: 7,
    .200: 6,
    .100: 5,
    .050: 4,
    .010: 3,
    .003: 2,
    .000: 1,
}


# Vectorized
def ebird_week(x: any) -> int:
    if hasattr(x, 'day') and hasattr(x, 'month') and hasattr(x, 'year'):  # HACK Find a simpler, more robust way
        return ebird_week_from_date(x)  # date | pd.Series.dt
    else:
        return ebird_week_normalize(x)  # int | np.ndarray[int]


# Vectorized
def ebird_week_from_date(date: date) -> int:
    """
    'Week' like ebird barchart weeks: 4 weeks per month, where the 4th week includes all days past 28
    - 0-indexed (0–47)
    """
    return (date.month - 1) * 4 + np.clip((date.day - 1) // 7, 0, 3)


# Vectorized
def ebird_week_normalize(week: int) -> int:
    """
    >>> ebird_week_normalize(0 + np.array([-1, 0, 1]))
    array([47,  0,  1])
    >>> ebird_week_normalize(1 + np.array([-1, 0, 1]))
    array([0,  1,  2])
    >>> ebird_week_normalize(47 + np.array([-1, 0, 1]))
    array([46, 47,  0])
    """
    return week % n_ebird_weeks


def ebird_week_to_date(week: int, year=2000) -> date:
    """
    Map an ebird week to an arbitrary date such that:
        ebird_week(ebird_week_to_date(week)) == week
    """
    return date(year, month=week // 4 + 1, day=week % 4 * 7 + 1)


def ebird_species_probs(
    priors_df: 'pd.DataFrame[loc_bin, date_bin, species, n_present, n]',
    groupby: List[str],
    agg: List[str],
    drop=True,
) -> 'pd.DataFrame[*groupby, n_present, n, p]':
    """Aggregate species probs"""
    return (priors_df

        # Aggregate n over agg neighborhood (e.g. (loc_bin, date_bin))
        .assign(n=lambda df: df.groupby(agg).n.first().sum())
        .pipe(lambda df: df if not drop else df.drop(columns=agg))

        # Aggregate n_present over agg neighborhood (e.g. (loc_bin, date_bin))
        #   - HACK .groupby on species:category produced bogus counts (e.g. TRBL 244 i/o 4); use species:str as a workaround
        .astype({
            **({} if 'species' in agg else {'species': 'str'}),
            **({} if 'date_bin' in agg else {'date_bin': 'int'}),
        })
        .pipe(lambda df: (df
            .groupby(observed=True, as_index=False, by=groupby)
            .agg({
                'n_present': 'sum',
                **{k: 'first' for k in df.columns if k not in ['n_present', *groupby]},
            })
        ))
        .astype({
            **({} if 'species' in agg else {'species': metadata.species.df.shorthand.dtype}),
            **({} if 'date_bin' in agg else {'date_bin': 'int'}),
        })

        # Compute p
        .assign(p=lambda df: df.n_present / df.n)

    )


# TODO How to accommodate ebird_priors.date_bin > 1w?
def plot_barchart(
    barcharts_df,
    downsample_sp=None,
    cols=None,
    width_per_col=4,
    aspect_per_col=1,
    random_state=0,
    debug=False,
) -> ggplot:

    # Defaults (which you'll usually want to tweak)
    downsample_sp = downsample_sp or barcharts_df.species.nunique()
    cols = cols or min(3, downsample_sp // 100 + 2)

    cmap = (
        # Debug: multi-color palette
        mpl.colors.ListedColormap(np.array(mpl.cm.tab10.colors)[[3, 1, 6, 4, 5, 7, 0, 9, 2]]) if debug else
        # Match color from ebird barcharts, for visual simplicity
        mpl.colors.ListedColormap(9 * ['#31c232'])
    )

    return (barcharts_df

        # Downsample by species (optional)
        .pipe(lambda df: df[df.species.isin(df.species
            .drop_duplicates().sample(replace=False, random_state=random_state, n=downsample_sp)
        )])

        # Add longhand, com_name (via join)
        .set_index('species', drop=False)
        .join(how='left', other=metadata.species.df.set_index('shorthand')[['longhand', 'com_name']])

        # Add w (ebird bar width) from p (prob)
        .assign(w=lambda df: df.p.map(ebird_bar_width))

        # Add facet_col
        .assign(facet_col=lambda df: (
            cols * df.species.cat.remove_unused_categories().cat.codes.astype(int) // df.species.nunique()
        ))

        # Debug: inspects
        .pipe(df_inspect, lambda df: () if not debug else (
            df_summary(df).T,
            df[:3],
            # Plot dist(p, w)
            (ggplot(df)
                + aes(x='p', fill='factor(w)') + geom_histogram(breaks=[*reversed(list(ebird_bar_widths)), 1])
                + theme_figsize(aspect=1/6) + xlim(0, 1) + scale_fill_cmap_d(cmap)
            ),
        ))

        # Plot
        .pipe(df_reverse_cat, 'species', 'longhand', 'com_name')
        .pipe(ggplot)
        + facet_wrap('facet_col', nrow=1, scales='free_y')
        + aes(x='date_bin', y='longhand', color='factor(w)', fill='factor(w)')
        + geom_point(aes(size='w'), shape='|', stroke=2)
        + scale_size_radius(
            range=(-1.5, 4),  # Min/max bar height for geom_point(size) (Default: (1, 6))
        )
        + scale_fill_cmap_d(cmap)
        + scale_color_cmap_d(cmap)
        + scale_x_continuous(limits=(0, 47),
            expand=(0, 1),  # Minimal expand (mul, add)
            labels=[5 * ' ' + x[0] for x in calendar.month_abbr[1:]] + [''],  # 12+1 month labels, on the major breaks
            breaks=np.arange(0, 47+4, 4) - .5,  # 13 grey major breaks, so we can label them with months
            minor_breaks=np.arange(0, 47+4, 4*3) - .5,  # 5 black minor breaks, to display over the major breaks
        )
        + theme(
            legend_position='none',
            axis_title=element_blank(),
            axis_text_x=element_text(size=7),
            axis_text_y=element_text(size=7),
            axis_ticks_major=element_blank(),
            panel_background=element_blank(),
            panel_grid_major_x=element_line(color='lightgrey', size=.5),  # Major/minor are inverted (see above)
            panel_grid_minor_x=element_line(color='black', size=.5),
            panel_grid_major_y=element_blank(),
            panel_grid_minor_y=element_blank(),
            strip_text=element_blank(),
            panel_spacing_x=2,
        )
        + theme_figsize(width=width_per_col * cols, aspect=aspect_per_col * cols)

    )


def ebird_bar_width(p: float, bar_widths=ebird_bar_widths) -> int:
    return next(v for k, v in bar_widths.items() if p >= k)


@dataclass
class SpeciesCounter(DataclassUtil):
    """
    A summable and memory-compact representation of species counts, e.g. stuff like:
          species  n_present  n    p
        0    CORA          4  5  0.8
        1    WREN          1  5  0.2
        2    CALT          3  5  0.6

    Specific design goal:
    - Suitable to use as the values of a (large) in-mem dict keyed by (date_bin, loc_bin), to represent binned ebird
      checklist counts

    Takeaways on cpu/mem tradeoffs for SpeciesCounter (based on notebooks/20180802_ebird_counters_comp.ipynb)
    - Use Counter instead of DF/Series or bounter
        - DF/Series are faster (~2x) to build than Counter
        - Counter is more compact (~2.75x) than DF/Series
        - Counter with int keys (.species cat codes) instead of str keys (.species) isn't significantly smaller (~7%)
        - Nix bounter: we're fine with a lossless representation (e.g. Counter), and tuning the fixed mem is nontrivial
          complexity
    - Estimate SpeciesCounter mem usage at ~300–400B each
        - Avg 331B per Counter with n=24845 (bias: first 3-4 weeks of Jan, US geos only)
    - Agg at predict time is fine
        - Adding a handful (~8) of Counters is fast (~5ms)
    """

    n: int  # Total num checklists
    sp: dict  # species -> n_present (num checklists species was present in)

    def with_sp(self, f):
        return self.replace(sp=f(self.sp))

    @property
    def df(self):
        """
        >>> SpeciesCounter(n=5, sp=dict(CALT=3, WREN=1, CORA=4)).df
          species  n_present  n    p
        0    CORA          4  5  0.8
        1    WREN          1  5  0.2
        2    CALT          3  5  0.6
        """
        return (
            pd.DataFrame(list(self.sp.items()), columns=['species', 'n_present'])
            .astype({'species': metadata.species.df.shorthand.dtype})
            .assign(
                n=self.n,
                p=lambda df: df.n_present / df.n,
            )
            .sort_values('species')
            .reset_index(drop=True)
        )

    def __add__(self, other):
        return type(self).sum([self, other])

    @classmethod
    def sum(cls, cs: Iterable['cls'], counter_cls=Counter, counter_update=lambda a, b: a.update(b)) -> 'cls':
        """
        >>> SpeciesCounter.sum([
        ...     SpeciesCounter(n=2, sp=dict(CORA=2, WREN=1)),
        ...     SpeciesCounter(n=3, sp=dict(CORA=2, CALT=3)),
        ... ])
        SpeciesCounter(
          n=5,
          sp={
            'CORA': 4,
            'WREN': 1,
            'CALT': 3
          }
        )
        """
        sp = counter_cls()
        for c in cs:
            counter_update(sp, c.sp)
        return cls(
            n=sum(c.n for c in cs),
            sp=dict(sp),
        )


@dataclass(frozen=True)  # frozen=True so we can hash, for @lru_cache
class EbirdPriors(DataclassEstimator):

    #
    # Params
    #

    # TODO Pick lightweight tuning params to get started
    #   - Query radius ~ binwidth, since query region is a rect with sides binwidth*2, approximately centered at the query point
    #
    # Theoretical mem ~ query radius (generated from code below):
    #           388mi     194mi      97mi      48mi      24mi      12mi       6mi       3mi     1.5mi      .8mi      .4mi
    #   48w  819.2 kB    3.3 MB   13.1 MB   52.4 MB  209.7 MB  838.9 MB    3.4 GB   13.4 GB   53.7 GB  214.7 GB  859.0 GB
    #   24w    1.6 MB    6.6 MB   26.2 MB  104.9 MB  419.4 MB    1.7 GB    6.7 GB   26.8 GB  107.4 GB  429.5 GB    1.7 TB
    #   16w    2.5 MB    9.8 MB   39.3 MB  157.3 MB  629.1 MB    2.5 GB   10.1 GB   40.3 GB  161.1 GB  644.2 GB    2.6 TB
    #   12w    3.3 MB   13.1 MB   52.4 MB  209.7 MB  838.9 MB    3.4 GB   13.4 GB   53.7 GB  214.7 GB  859.0 GB    3.4 TB
    #   8w     4.9 MB   19.7 MB   78.6 MB  314.6 MB    1.3 GB    5.0 GB   20.1 GB   80.5 GB  322.1 GB    1.3 TB    5.2 TB
    #   6w     6.6 MB   26.2 MB  104.9 MB  419.4 MB    1.7 GB    6.7 GB   26.8 GB  107.4 GB  429.5 GB    1.7 TB    6.9 TB
    #   4w     9.8 MB   39.3 MB  157.3 MB  629.1 MB    2.5 GB   10.1 GB   40.3 GB  161.1 GB  644.2 GB    2.6 TB   10.3 TB
    #   3w    13.1 MB   52.4 MB  209.7 MB  838.9 MB    3.4 GB   13.4 GB   53.7 GB  214.7 GB  859.0 GB    3.4 TB   13.7 TB
    #   2w    19.7 MB   78.6 MB  314.6 MB    1.3 GB    5.0 GB   20.1 GB   80.5 GB  322.1 GB    1.3 TB    5.2 TB   20.6 TB
    #   1w    39.3 MB  157.3 MB  629.1 MB    2.5 GB   10.1 GB   40.3 GB  161.1 GB  644.2 GB    2.6 TB   10.3 TB   41.2 TB
    #
    # Empirical mem ~ query radius:
    #   TODO Measure a few points to sketch out the surface
    #
    # Empirical time ~ query radius:
    #   TODO Measure a few points to sketch out the surface
    #
    """

    # Generate the table comment above (with color!)
    from ebird_priors import *
    from potoo.pandas import df_style_cell
    loc_bin_slice = slice(5, 20)
    loc_binwidths = np.array(list(EbirdPriors._loc_binwidths.keys()))[loc_bin_slice]
    loc_bin_ns = 2 ** np.array(list(EbirdPriors._loc_binwidths.values()))[loc_bin_slice]
    date_binwidths = [x for x in EbirdPriors._date_binwidths]
    date_bin_ns = 48 // np.array([EbirdPriors._date_binwidth_parse_weeks(x) for x in EbirdPriors._date_binwidths])
    (DF(columns=date_binwidths, index=loc_binwidths, data=np.vectorize(humanize.naturalsize)(
        400 * np.array(loc_bin_ns)[:, np.newaxis] * np.array(date_bin_ns)
    )).T
        .style.applymap(lambda x: df_style_cell(
            (lambda x: x[-1] in ['Bytes', 'kB'],       'color: #1f78b4'),  # dark blue
            (lambda x: x[-1] in ['MB'] and x[0] < 10,  'color: #a6cee3'),  # blue/green
            (lambda x: x[-1] in ['MB'] and x[0] < 100, 'color: #33a02c'),  # green
            (lambda x: x[-1] in ['MB'],                'color: #ffff99'),  # yellow
            (lambda x: x[-1] in ['GB'] and x[0] < 10,  'color: #ff7f00'),  # orange
            (lambda x: x[-1] in ['GB'] and x[0] < 100, 'color: #fb9a99'),  # orange/red/pink
            (lambda x: x[-1] in ['GB'],                'color: #e31a1c'),  # dark red
            (lambda x: x[-1] in ['TB'],                'color: #6a3d9a'),  # dark purple
        )(one((float(a), b) for a, b in [x.split()])))
    )

    """

    # Location bin size, specified by length of sides of a geohash square
    #   - We omit rectangular geohashes, e.g. 4+0 which is 24mi W x 12mi H
    #   - Precisions: 3+2 means geohash char precision 3 plus 2 (of 5) more bits (i.e. precision=3+1 minus 5-2 bits)
    loc_binwidth: str
    _loc_binwidths = {
        # Human-friendly str id -> num hi bits to keep in a uint64 geohash
        '12407mi':  1,  # n[  2 ], str_precision[0+1]
        '6204mi':   3,  # n[  8 ], str_precision[0+3]
        '3102mi':   5,  # n[ 32 ], str_precision[1+0]
        '1551mi':   7,  # n[128 ], str_precision[1+2]
        '775mi':    9,  # n[512 ], str_precision[1+4]
        '388mi':   11,  # n[  2k], str_precision[2+1]
        '194mi':   13,  # n[  8k], str_precision[2+3]
        '97mi':    15,  # n[ 32k], str_precision[3+0]
        '48mi':    17,  # n[128k], str_precision[3+2]
        '24mi':    19,  # n[512k], str_precision[3+4]
        '12mi':    21,  # n[  2m], str_precision[4+1]
        '6mi':     23,  # n[  8m], str_precision[4+3]
        '3mi':     25,  # n[ 32m], str_precision[5+0]
        '1.5mi':   27,  # n[128m], str_precision[5+2]
        '.8mi':    29,  # n[512m], str_precision[5+4]
        '.4mi':    31,  # n[  2b], str_precision[6+1]
        # .4mi is plenty small of a range, and 2b is plenty big to compute, so we'll stop here
    }

    @property
    def _loc_binwidth_geohash_precision_bits(self) -> int:
        return self._loc_binwidths[self.loc_binwidth]

    @property
    def _loc_binwidth_miles(self) -> float:
        return parse.parse('{:f}mi', self.loc_binwidth)[0]

    # Date bin size, specified by num ebird weeks
    #   - 48 ebird weeks per year (https://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts)
    #   - All factors of 48 are allowed
    #   - date_binwidth='1w' means 48 weeks per year, each 1w long
    #   - date_binwidth='48w' means 1 "week" per year, each 48w long
    date_binwidth: str
    _date_binwidths = ['48w', '24w', '16w', '12w', '8w', '6w', '4w', '3w', '2w', '1w']

    @property
    @lru_cache()
    def _date_binwidth_weeks(self) -> int:
        return self._date_binwidth_parse_weeks(self.date_binwidth)

    @classmethod
    def _date_binwidth_parse_weeks(cls, date_binwidth) -> int:
        return parse.parse('{:d}w', date_binwidth)[0]

    def __post_init__(self):
        """Validate params"""
        if self.loc_binwidth not in self._loc_binwidths:
            raise ValueError(f'Invalid loc_binwidth[{self.loc_binwidth}], must be one of: {list(self._loc_binwidths)}')
        if self.date_binwidth not in self._date_binwidths:
            raise ValueError(f'Invalid date_binwidth[{self.date_binwidth}], must be one of: {self._date_binwidths}')

    #
    # Loc/date binning and smoothing
    #

    # Types
    LocBin = str  # geohash
    DateBin = int  # ebird_week

    def _loc_smooth(self, lat: float, lon: float) -> Iterable[LocBin]:
        """
        "Smooth" a location into loc bins to avoid proximity to bin boundaries (to decrease variance)
        - Maps the loc to the 4 contiguous loc bins that best enclose it (i.e. with most balanced margins)
        """

        # Find neighborhood (≤9 bins) at next level of precision (p+2)
        p = self._loc_binwidth_geohash_precision_bits
        finer_p = p + 2  # p+2 because p+1 isn't square (because bits represent alternating lon/lat/lon/lat/...)
        finer_bin = self._loc_bin(lat, lon, finer_p)
        finer_bins = geoh.str_expand(finer_bin, finer_p)
        assert len(finer_bins) <= 9

        # Truncate those ≤9 bins back to our level of precision (p), collapsing them to ≤4 bins roughly centered at lat/lon
        bins = list(unique_everseen(geoh.str_at_prec(x, p) for x in finer_bins))
        assert len(bins) <= 4
        return bins

    def _loc_bin(self, lat: float, lon: float, precision_bits: int = None) -> LocBin:
        if precision_bits is None:
            precision_bits = self._loc_binwidth_geohash_precision_bits
        return geoh.str_encode(lat, lon, precision_bits)

    def _date_smooth(self, date: date) -> Iterable[DateBin]:
        """
        "Smooth" a date into date bins to avoid proximity to bin boundaries (to decrease variance)
        - Maps the date to the 2 contiguous date bins that best enclose it (i.e. with most balanced margins)
        """

        assert isinstance(date, datetime.date)
        week = ebird_week(date)
        date_bin = self._date_bin(date)

        # Boundary condition: if date_binwidth='1w' (smallest), measure day within week, else week within date_bin
        if self._date_binwidth_weeks == 1:
            closer_to_prev_bin = np.clip(date.day - 1, 0, 27) / 7 % 1. < .5
        else:
            closer_to_prev_bin = week / self._date_binwidth_weeks % 1. < .5

        bins = sorted({
            date_bin,
            ebird_week(date_bin + (-1 if closer_to_prev_bin else 1) * self._date_binwidth_weeks),
        })
        assert len(bins) == (2 if self.date_binwidth != '48w' else 1)
        return bins

    def _date_bin(self, date: date) -> DateBin:
        assert hasattr(date, 'day'), f'Expected date, got {type(date)}: {date}'
        assert 48 // self._date_binwidth_weeks * self._date_binwidth_weeks == 48, \
            f'date_binwidth[{self.date_binwidth}] must be factor of 48w'
        return ebird_week(date) // self._date_binwidth_weeks * self._date_binwidth_weeks

    #
    # fit/predict
    #

    # TODO Crib from notebook (the big, heavy ebird_priors pipeline)
    #   - Need to choose what stage of input to take (definitely after all the shell junk -- leave that in the notebook for now)
    def fit_df(self, x: ...) -> 'self':
        ...

    # XXX Nope, pipeline from raw-ish data
    # def fit_df(self, species_probs: pd.DataFrame) -> 'self':
    #     self._species_probs = species_probs

    # TODO Crib from notebook (ebird_priors lookup -> SpeciesCounter.sum)
    def predict_proba_one_df(self, lat: float, lon: float, date: date) -> pd.DataFrame:
        loc_bins = self._loc_smooth(lat, lon)
        date_bins = self._date_smooth(date)
        ...

    # TODO sk api
    def fit(self, X, y):
        ...

    # TODO sk api + batch prediction
    def predict_proba(self, X):
        ...
