## Notebook init

from notebooks import *

# NOTE Edit config.py to get num_recs=None, else you'll be restricted to ≤200 recs/sp
#   - Want: countries_k='na', com_names_k='us', num_recs=None,
#   - Not:  countries_k='na', com_names_k='us', num_recs=200,
assert config.server_globals.sg_load.xc_meta == {'countries_k': 'na', 'com_names_k': 'us', 'num_recs': None}

sg.init(None)  # Computes search_recs, if cache miss

search_recs = sg.search_recs

# XXX Very slow to run over all 45k recs, do lazily instead
# recs = (search_recs
#     .pipe(df_inspect, lambda df: (df.shape,))
#     .pipe(talk_recs_show)  # [Defined below]
#     .pipe(df_inspect, lambda df: (df.shape,))
# )

## Utils


def talk_hide_index(df) -> HTML:
    return display_with_style(df, style_css=lambda scoped_class: '''
        .%(scoped_class)s .dataframe tr th:first-child { display: none; }
    ''' % locals())


def talk_hide_columns(df) -> HTML:
    return display_with_style(df, style_css=lambda scoped_class: '''
        .%(scoped_class)s .dataframe thead { display: none; }
    ''' % locals())


def talk_hide_index_and_columns(df) -> HTML:
    return display_with_style(df, style_css=lambda scoped_class: '''
        .%(scoped_class)s .dataframe tr th:first-child { display: none; }
        .%(scoped_class)s .dataframe thead             { display: none; }
    ''' % locals())


def talk_show_refs(s: str) -> HTML:
    return HTML(
        '<div class="talk-refs text_cell_render">%s</div>' % '\n'.join([
            '<div>%s</div>' % line
            for line in s.split('\n')
        ]),
    )


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
        'species_com_name',
        'quality', 'recordist', 'background_species', 'date',
    ],
    append=[],
    drop=['license', 'bird_seen', 'playback_used'],
    astype={'year': int},
    replace={'subspecies': {'': '—'}},
):
    return (df
        # Drop any indexes (e.g. from filter/sort)
        .reset_index(drop=True)
        # Featurize
        .pipe(recs_featurize_spectro_disp, scale=scale)
        # View
        .pipe(recs_view_cols, append=[*order, *append])  # append= to ensure all requested cols are included
        .pipe(df_reorder_cols, last=order)  # last= so that unknown cols show up loudly in the front
        .drop(columns=drop)
        .astype(astype)
        .replace(replace)
    )


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
