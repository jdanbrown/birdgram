## Notebook init

from notebooks import *

# NOTE Edit config.py to get num_recs=None, else you'll be restricted to ≤200 recs/sp
#   - Want: countries_k='na', com_names_k='us', num_recs=None,
#   - Not:  countries_k='na', com_names_k='us', num_recs=200,
assert config.server_globals.sg_load.xc_meta == {'countries_k': 'na', 'com_names_k': 'us', 'num_recs': None}

sg.init(None)  # Computes search_recs, if cache miss

search_recs = sg.search_recs

## Utils

def talk_recs_show(
    df,
    scale=None,
    order=[
        'xc_id', 'spectro_disp',
        'species', 'subspecies',
        'year', 'month_day', 'time',
        'type',
        'state', 'elevation', 'lat', 'lng', 'place',
        'remarks',
        'quality', 'recordist', 'background_species', 'date',
    ],
    drop=['license', 'bird_seen', 'playback_used'],
    astype={'year': int},
    replace={'subspecies': {'': '—'}},
    **kwargs,
):
    return (df
        # Drop any indexes (e.g. from filter/sort)
        .reset_index(drop=True)
        # Featurize
        .pipe(recs_featurize_spectro_disp, scale=scale)
        # View
        .pipe(recs_view_cols, append=order)  # append= to ensure all requested cols are included
        .pipe(df_reorder_cols, last=order)  # last= so that unknown cols show up loudly in the front
        .drop(columns=drop)
        .astype(astype)
        .replace(replace)
    )

# XXX Very slow to run over all 45k recs, do lazily instead
# recs = (search_recs
#     .pipe(df_inspect, lambda df: (df.shape,))
#     .pipe(talk_recs_show)
#     .pipe(df_inspect, lambda df: (df.shape,))
# )

def talk_recs_show_seasonal(
    species,
    filters=lambda df: [],
    bins=2,  # 2 | 4
    n=None,
    scale=None,
    scale_base=3.8,
    random_state=0,
):
    assert bins in [2, 4]
    if n is None:
        n = 50 // bins
    return (search_recs
        .pipe(df_inspect, lambda df: (df.shape,))
        # Filter
        [lambda df: reduce(lambda x, y: x & y, [
            df.species == species,
            *filters(df),
        ])]
        .pipe(df_inspect, lambda df: (df.shape,))
        # Sort
        # .sample(100, random_state=random_state)
        .sort_values(['month_day'], ascending=[False])
        # View
        .pipe(talk_recs_show,
            scale=scale or scale_base / bins,
        )
        [:1000]  # Safeguard: trim to a reasonable max (> any sp, but << all recs)
        # Bin by season (4 bins)
        #   - Boundaries based roughly on https://birdsna.org/Species-Account/bna/species/ruckin/breeding
        .assign(season=lambda df: df.month_day.str.split('-').str[0].map(lambda x: {
            '09': 'fall migration',
            '10': 'fall migration',
            '11': 'fall migration',
            '12': 'winter',
            '01': 'winter',
            '02': 'winter',
            '03': 'spring migration',
            '04': 'spring migration',
            '05': 'spring migration',
            '06': 'breeding',
            '07': 'breeding',
            '08': 'breeding',
        }.get(x)))
        .assign(season=lambda df: df.season.pipe(lambda s: s.pipe(as_ordered_cat, [
            'fall migration', 'winter', 'spring migration', 'breeding',
        ])))
        # Bin further
        #   - For bigger spectros (4->2 cols)
        .pipe(lambda df: df if bins == 4 else (df
            .replace({'season': {
                'spring migration': 'spring migration / breeding',
                'breeding':         'spring migration / breeding',
                'fall migration':   'fall migration / winter',
                'winter':           'fall migration / winter',
            }})
            .assign(season=lambda df: df.season.pipe(lambda s: s.pipe(as_ordered_cat, [
                'fall migration / winter', 'spring migration / breeding',
            ])))
        ))
        # Pivot by season (manually)
        .pipe(lambda df: pd.concat(axis=1, objs=[
            (df
                [df.season == season]
                [['spectro_disp']]
                .rename(columns={'spectro_disp': season})
                .sample(frac=1, random_state=random_state)  # Randomize per column (to avoid weird biases from incidental sorting above)
                .reset_index(drop=True)
            )
            for season in df.season.sort_values().unique()
        ]))
        .fillna('')
        .pipe(df_inspect, lambda df: (df.shape,))
        [:n]
        .pipe(df_inspect, lambda df: (df.shape,))
    )

def talk_wraa_calls():
    """All the damn wraa calls"""
    return (search_recs
        .pipe(df_inspect, lambda df: (df.shape,))
        # Filter
        [lambda df: reduce(lambda x, y: x | y, [

            (df.species == 'SPTO') & df.xc_id.isin([127012]),
            (df.species == 'EATO') & df.xc_id.isin([293823]),

            (df.species == 'HUVI') & df.xc_id.isin([297120]),
            # (df.species == 'HUVI') & df.xc_id.isin([348987]),
            (df.species == 'WAVI') & df.xc_id.isin([159366]),
            # (df.species == 'WAVI') & df.xc_id.isin([381527]),

            (df.species == 'HETH') & df.xc_id.isin([314303]),
            # (df.species == 'HETH') & df.xc_id.isin([131636]),

            (df.species == 'BEWR') & df.xc_id.isin([163209]),
            # (df.species == 'BEWR') & df.xc_id.isin([141349]),
            (df.species == 'HOWR') & df.xc_id.isin([265810]),

            (df.species == 'BANO') & df.xc_id.isin([294969]),  # Juv shriek [PFGBS]
            (df.species == 'GHOW') & df.xc_id.isin([154990]),  # Juv shriek [PFGBS]

            (df.species == 'BGGN') & df.xc_id.isin([376229]),
            # (df.species == 'BGGN') & df.xc_id.isin([81059]),
            (df.species == 'BCGN') & df.xc_id.isin([30087]),
            (df.species == 'BTGN') & df.xc_id.isin([253889]),
            (df.species == 'CAGN') & df.xc_id.isin([17808]),

            (df.species == 'LOSH') & df.xc_id.isin([255158]),
            # (df.species == 'LOSH') & df.xc_id.isin([255145]),
            (df.species == 'GGSH') & df.xc_id.isin([91968]),  # NOSH (Northern Shrike) used to be GGSH (Great Gray Shrike)

            (df.species == 'CASJ') & df.xc_id.isin([347904]),
            (df.species == 'STJA') & df.xc_id.isin([146610]),

        ])]
        .pipe(df_inspect, lambda df: (df.shape,))
        # View
        .pipe(talk_recs_show,
            scale=2.9,
        )
        .pipe(df_ordered_cats_like, species=metadata.ebird.df.shorthand)
        .sort_values(['species'])
    )
