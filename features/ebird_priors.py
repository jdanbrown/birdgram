import calendar
from datetime import date
from typing import List, Optional, Set, Tuple, Union

import geohash
import matplotlib as mpl
import numpy as np
import pandas as pd
from potoo.pandas import df_reverse_cat
from potoo.plot import *
from potoo.util import tap

import metadata
from util import *


# https://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts
# https://help.ebird.org/customer/en/portal/articles/1210247-what-is-frequency-
n_ebird_weeks = 48
ebird_weeks = np.arange(n_ebird_weeks) + 1
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
    if isinstance(x, date):
        return ebird_week_from_date(x)  # date | pd.Series.dt
    else:
        return ebird_week_normalize(x)  # int | np.ndarray[int]


# Vectorized
def ebird_week_from_date(date: date) -> int:
    """'Week' like ebird barchart weeks: 4 weeks per month, where the 4th week includes all days past 28"""
    return (
        (date.month - 1) * 4  # Week from month (4*[0,11])
        + np.clip(date.day // 7, 0, 3)  # Week of month ([0,3])
        + 1  # 0-indexed week to 1-indexed week ([0,47] -> [1,48])
    )


# Vectorized
def ebird_week_normalize(week: int) -> int:
    """
    >>> ebird_week_normalize(1 + np.array([-1, 0, 1]))
    array([48,  1,  2])
    >>> ebird_week_normalize(2 + np.array([-1, 0, 1]))
    array([1,  2,  3])
    >>> ebird_week_normalize(48 + np.array([-1, 0, 1]))
    array([47, 48,  1])
    """
    return (week - 1) % n_ebird_weeks + 1


# Vectorized versions of geohash.*
#   - https://github.com/hkwi/python-geohash/wiki/GeohashReference
geohash_encode        = np_vectorize_asscalar(geohash.encode,        otypes=[str])
geohash_encode_uint64 = np_vectorize_asscalar(geohash.encode_uint64, otypes=[np.uint64])
# XXX np.vectorize barfs on multiple return values, and adds no value on non-multiple input args
# geohash_decode        = np_vectorize_asscalar(geohash.decode,        otypes=[np.float64, np.float64])
# geohash_decode_uint64 = np_vectorize_asscalar(geohash.decode_uint64, otypes=[np.float64, np.float64])


def ebird_week_pm(
    radius: Union[int, Tuple[int, int]],
    weeks: Union[int, Iterable[int]],
) -> Set[int]:
    """Given weeks ± week_radius weeks"""
    weeks = np.array(list(weeks) if can_iter(weeks) else [weeks])
    (radius_l, radius_r) = (-radius, radius) if isinstance(radius, int) else radius
    deltas = np.arange(radius_l, radius_r + 1)
    return set(ebird_week(
        (weeks[:, np.newaxis] + deltas).reshape(-1)
    ))


def geohash_pm(
    radius: int,
    geohashes: Union[str, Iterable[str]],
) -> Set[str]:
    """Given geohashes + 8 neighbors, repeated radius many times"""
    geohashes = {geohashes} if isinstance(geohashes, str) else set(geohashes)
    expanded = set()
    for _ in range(radius):
        for g in geohashes - expanded:
            geohashes.update(geohash.neighbors(g))  # (.expand(g) = [g] + .neighbors(g))
            expanded.add(g)
    return geohashes


def ebird_species_probs(
    priors_df: 'pd.DataFrame[week, geohash4, species, n_present, n]',
    groupby: List[str],
    agg: List[str],
    drop=True,
) -> 'pd.DataFrame[*groupby, n_present, n, p]':
    """
    Aggregate species probs
    - python-geohash
        - https://github.com/hkwi/python-geohash/wiki
        - https://github.com/hkwi/python-geohash/wiki/GeohashReference
    - How big is a geohash?
        - https://www.movable-type.co.uk/scripts/geohash.html [sf: (38,-122)]
        - https://gis.stackexchange.com/questions/115280/what-is-the-precision-of-a-geohash
    - Misc
        - https://en.wikipedia.org/wiki/Geohash
        - https://www.elastic.co/guide/en/elasticsearch/reference/current/search-aggregations-bucket-geohashgrid-aggregation.html

    Geohash dims (height is constant, width varies by lat: max at equator, 0 at poles):
    - geohash3: (w, h) ≤ ( 97,  97) mi
    - geohash4: (w, h) ≤ ( 24,  12) mi
    - geohash5: (w, h) ≤ (3.0, 3.0) mi
    - geohash6: (w, h) ≤ (.76, .38) mi
    """
    return (priors_df

        # Aggregate n over agg neighborhood (e.g. (week, geohash))
        .assign(n=lambda df: df.groupby(agg).n.first().sum())
        .pipe(lambda df: df if not drop else df.drop(columns=agg))

        # Aggregate n_present over agg neighborhood (e.g. (week, geohash))
        #   - HACK .groupby on species:category produced bogus counts (e.g. TRBL 244 i/o 4); use species:str as a workaround
        .astype({
            **({} if 'species' in agg else {'species': 'str'}),
            **({} if 'week' in agg else {'week': 'int'}),
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
            **({} if 'week' in agg else {'week': 'int'}),
        })

        # Compute p
        .assign(p=lambda df: df.n_present / df.n)

    )


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
        + aes(x='week', y='longhand', color='factor(w)', fill='factor(w)')
        + geom_point(aes(size='w'), shape='|', stroke=2)
        + scale_size_radius(
            range=(-1.5, 4),  # Min/max bar height for geom_point(size) (Default: (1, 6))
        )
        + scale_fill_cmap_d(cmap)
        + scale_color_cmap_d(cmap)
        + scale_x_continuous(limits=(1, 48),
            expand=(0, 1),  # Minimal expand (mul, add)
            labels=[5 * ' ' + x[0] for x in calendar.month_abbr[1:]] + [''],  # 12+1 month labels, on the major breaks
            breaks=np.arange(1, 48+4, 4) - .5,  # 13 grey major breaks, so we can label them with months
            minor_breaks=np.arange(1, 48+4, 4*3) - .5,  # 5 black minor breaks, to display over the major breaks
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
