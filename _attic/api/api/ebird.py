# Docs:
#   - https://confluence.cornell.edu/display/CLOISAPI/eBird+API+1.1
#   - https://confluence.cornell.edu/display/CLOISAPI/eBird-1.1-RecentNearbyObservations
#   - http://ebird.org/ebird/GuideMe?cmd=changeLocation
#   - http://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts

from datetime import date
from typing import Sequence

import pandas as pd
import pp
import structlog

from api.request import cached_request
from api.util import LonLat, haversine, snakecase, test_lonlats

log = structlog.get_logger(__name__)


def nearby_barcharts(
    lonlat: LonLat,
    num_nearby_hotspots: int = 10,  # Maybe a good idea to not allow any more than this
) -> pd.DataFrame:
    hotspot_df = nearby_hotspots(lonlat)
    return barcharts(
        loc_ids=hotspot_df.sort_values('dist_km')['loc_id'][:num_nearby_hotspots]
    )


def nearby_hotspots(
    lonlat: LonLat,
    dist_km: int = 50,  # Max: 500
    back_d: int = 30,
) -> pd.DataFrame:
    # https://confluence.cornell.edu/display/CLOISAPI/eBird-1.1-HotspotGeoReference
    df = df_from_ebird('http://ebird.org/ws1.1/ref/hotspot/geo', params=dict(
        lng=lonlat.lon,
        lat=lonlat.lat,
        dist=dist_km,
        back=back_d,
    ))
    # If response is empty then df will have no cols and .sort_values will fail; add dummy col to avoid this
    if 'dist_km' not in df:
        df['dist_km'] = None
    return df.sort_values('dist_km')


def barcharts(
    loc_ids: Sequence[str],
    begin_year: int = date.today().year - 11,  # -10y appears usefully less noisy than -5y
    end_year: int = date.today().year,
    begin_month: int = 1,
    end_month: int = 12,
) -> pd.DataFrame:
    """
    Aggregate payload down to one minimal barchart per species
    - pd.merge(raw_barcharts, nearby_hotspots) produces ~3MB per hotspot
    - Keep queries split by hotspot for cache locality
    """

    df = _raw_barcharts(
        loc_ids=loc_ids,
        begin_year=begin_year,
        end_year=end_year,
        begin_month=begin_month,
        end_month=end_month,
    )

    if df.empty:
        return df

    # What do all the columns mean?
    #   - Ref: http://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts
    #   - category? e.g.
    #       - 'species':8592, 'spuh':672, 'slash':288, 'hybrid':48
    #   - buckets vs. values_N? see "About the Frequencies" in the barcharts docs:
    #       - http://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts
    #       - values: num checklists reporting the species / num checklists
    #       - values_N: num checklists
    #       - buckets: non-linear f(values) in [-1, ..., 9] designed "to best show bird abundance at various scales"
    #   - Species names and codes? e.g.
    #       - speciesCode: 'doccor'
    #       - displayName: 'Double-crested Cormorant'
    #       - name: 'Double-crested Cormorant'
    #       - commonName: 'Double-crested Cormorant'
    #       - sciName: 'Phalacrocorax auritus'

    # Clean up columns
    df = (df
        .rename(columns={
            # Meaningful names for metrics
            'values': 'present_checklist_frac',
            'values_n': 'total_checklist_num',
            'buckets': 'freq_score',
        })
        .assign(
            present_checklist_num=lambda df: df['present_checklist_frac'] * df['total_checklist_num'],
        )
        .drop(axis=1, labels=[
            # Have to recompute after re-aggregating across hotspots
            'present_checklist_frac',
            'freq_score',
            # Junk, e.g. pretty_values='35%' when values=0.35
            'pretty_values',
        ])
    )

    # Aggregate metrics across hotspots
    df = (df
        .drop('loc_id', axis=1)
        .groupby(['species_code', 'display_name', 'name', 'common_name', 'sci_name', 'category', 'week'])
        .agg('sum')
        .reset_index()
    )

    # Re-divide present_checklist_frac, since we had to re-aggregate the base checklist counts
    df['present_checklist_frac'] = df.apply(axis=1, func=lambda x:
        -1 if x['total_checklist_num'] == 0 else x['present_checklist_num'] / x['total_checklist_num']
    )

    # Re-score freq, since we had to re-aggregate the base checklist counts
    df['freq_score'] = df.apply(axis=1, func=lambda x:
        freq_score(x['present_checklist_frac'], x['total_checklist_num'])
    )

    # Re-label with loc_ids
    df['loc_ids'] = ','.join(loc_ids)

    return df


def _raw_barcharts(
    loc_ids: Sequence[str],
    begin_year=date.today().year - 11,  # -10y appears usefully less noisy than -5y
    end_year=date.today().year,
    begin_month=1,
    end_month=12,
) -> pd.DataFrame:
    """
    Unofficial api:
    - http://ebird.org/ebird/barchart -> "Download Histogram Data" -> http://ebird.org/ebird/barchartData?...
    """

    if len(loc_ids) > 10:
        raise ValueError(f'len(loc_ids) > 10: loc_ids[{loc_ids}]')

    def _barchart(loc_id: str) -> pd.DataFrame:

        # http://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts
        #   "Since most months do not divide equally into 7-day periods, any remaining days are added to
        #   the last period. Thus, while the first, second, and third periods are all seven days, the
        #   final period ranges from seven to ten days, depending on if the month has 28, 29, 30, or 31
        #   days. The fact that the final period is consistently longer does not seem to bias the
        #   results strongly, but please do keep this in mind as you explore data using this tool."
        weeks = [
            'm%02d-w%d' % (month + 1, week + 1)
            for month in range(12)
            for week in range(4)
        ]

        # Unofficial api: http://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts
        #   - Cache key:
        #       - (lat,lon) -> k nearest hotspots -> stable cache key per "user session"
        #       - TODO Friendly cache hit rate: separate queries by hotspot + use ttl=1w or ttl=1mo
        df = df_from_ebird(
            'http://ebird.org/ebird/barchartData',
            params=dict(
                bmo=begin_month,
                emo=end_month,
                byr=begin_year,
                eyr=end_year,
                r=loc_id,  # (Accepts multiple as comma-sep string; max num is probably 10 like the other apis)
            ),
            rep_json_to_df=lambda rep_json: pd.concat(
                pd.DataFrame({
                    **x.get('taxon', {}),
                    **{k: v for k, v in x.items() if k != 'taxon'},
                    'week': weeks,
                })
                for x in rep_json['dataRows']
            ).reset_index(drop=True),
        )

        # Clean output
        df = df_clean(df)
        df['loc_id'] = loc_id

        return df

    dfs = [_barchart(loc_id) for loc_id in loc_ids]
    if dfs:
        return pd.concat(dfs)  # (Fails on empty list of dfs)
    else:
        return pd.DataFrame()


def freq_score(present_checklist_frac, total_checklist_num):
    """
    Reimplement the "About the Frequencies" table so we can re-aggregate our own barcharts:
    - http://help.ebird.org/customer/portal/articles/1010553-understanding-the-ebird-bar-charts
    """
    if   total_checklist_num    == 0:     return -1 # noqa
    elif present_checklist_frac == 0:     return 0  # noqa
    elif present_checklist_frac <  0.003: return 1  # noqa
    elif present_checklist_frac <  0.01:  return 2  # noqa
    elif present_checklist_frac <  0.05:  return 3  # noqa
    elif present_checklist_frac <  0.1:   return 4  # noqa
    elif present_checklist_frac <  0.2:   return 5  # noqa
    elif present_checklist_frac <  0.3:   return 6  # noqa
    elif present_checklist_frac <  0.4:   return 7  # noqa
    elif present_checklist_frac <  0.6:   return 8  # noqa
    else:                                 return 9  # noqa


def df_from_ebird(
    url: str,
    params={},
    rep_json_to_df=pd.DataFrame,
    method='GET',
):

    # Prepare input
    params.setdefault('fmt', 'json')
    params.setdefault('locale', 'en_US')

    # Query
    rep_json = cached_request(method, url, params=params).json()
    df = rep_json_to_df(rep_json)

    # Clean output
    df = df_clean(df)
    if 'lonlat' in df and {'lng', 'lat'}.issubset(params):
        df = df_with_dist(df, LonLat(params['lng'], params['lat']))

    return df


def df_clean(df):
    df = df.copy()
    df = df_normalize_col_casing(df)
    if {'lng', 'lat'}.issubset(df):
        df['lonlat'] = df[['lng', 'lat']].apply(lambda lng_lat: LonLat(*lng_lat), axis=1)
        df = df.drop(['lng', 'lat'], axis=1)
    return df


def df_normalize_col_casing(df):
    return df.rename(columns=snakecase)


def df_with_dist(df, lonlat: LonLat):
    df = df.copy()
    df['query_lonlat'] = [lonlat] * len(df)
    df['dist_km'] = df.apply(lambda x: haversine(x['lonlat'], x['query_lonlat']), axis=1)
    return df


if __name__ == '__main__':
    from api.app import new_app; new_app()
    pp(_raw_barcharts(['L5532282']))
    pp(barcharts(['L5532282']))
    pp(nearby_hotspots(test_lonlats['home']))
    pp(nearby_barcharts(test_lonlats['home']))
