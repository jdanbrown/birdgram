from collections import OrderedDict
from functools import lru_cache
import re
from typing import Optional

import pandas as pd

from constants import data_dir
from util import cache_to_file_forever, singleton

metadata_dir = f'{data_dir}/metadata'
ebird_taxa_path = f'{metadata_dir}/ebird-ws1.1-taxa-species.csv'


@singleton
class species:

    def __getitem__(self, x):
        """
        Simple and efficient (O(1)) lookup by any of:
        - shorthand
        - species_code
        - banding_code
        - com_name_code
        - sci_name_code
        - com_name
        - sci_name

        Example usage:
            >>> species['song sparrow']
            >>> species['song sparrow']['shorthand']
            >>> species['song sparrow', 'shorthand']
        """
        try:
            (query, key) = x
        except:
            (query, key) = (x, None)
        row = self._df_lookup.get(self._normalize_query(query or ''))
        if not row or not key:
            return row
        else:
            return row[key]

    @property
    @lru_cache()
    @cache_to_file_forever(f'{metadata_dir}/cache/_taxa')  # Avoid ~250ms (on a MBP)
    def df(self):
        """The full species df (based on http://ebird.org/ws1.1/ref/taxa/ebird?cat=species&fmt=csv)"""
        assert (self._raw_ebird_df.CATEGORY == 'species').all()
        return (self._raw_ebird_df
            .drop(columns=['CATEGORY'])
            .rename(columns={
                'SCIENTIFIC_NAME': 'sci_name',
                'COMMON_NAME': 'com_name',
                'TAXON_ID': 'taxon_id',
                'SPECIES_CODE': 'species_code',
                'CATEGORY': 'category',
                'TAXON_ORDER': 'taxon_order',
                'COM_NAME_CODES': 'com_name_codes',
                'SCI_NAME_CODES': 'sci_name_codes',
                'BANDING_CODES': 'banding_codes',
            })
            .pipe(self.add_shorthand_col)
        )

    @property
    def _raw_ebird_df(self):
        """The raw ebird species df (from http://ebird.org/ws1.1/ref/taxa/ebird?cat=species&fmt=csv)"""
        return pd.read_csv(ebird_taxa_path)

    def add_shorthand_col(self, df) -> pd.DataFrame:
        """Compute the unique 'shorthand' column and add it to the input df"""

        # Don't mutate the input
        df = df.copy()

        # Fix the order of the loop below where we assign shorthands, so that:
        #   - If a row has a banding_codes[0] then it will always be used (do them first)
        #   - If a row has com_name_codes then they will try to be used (do them second)
        #   - Include species_code in the sort so that our output is deterministic
        assert df.species_code.nunique() == len(df), "species_code isn't unique"
        df = (df
            .sort_values(['banding_codes', 'com_name_codes', 'species_code'], na_position='last')
            .reset_index(drop=True)  # Else .at[i, ...] will still follow the old ordering
        )

        shorthand_col = []
        used_shorthands = set()
        ifnull = lambda x, y: x if not pd.isnull(x) else y
        for i in range(len(df)):
            # Performance (~10K rows):
            #   - .at is noticeably faster than .loc
            #   - `i in range(...)` with `.at[i, col]` is noticeably faster than `i, row in .iterrows()` with `row[col]`
            candidates = list(filter(lambda x: x not in used_shorthands, [
                *ifnull(df.at[i, 'banding_codes'], '').split(),
                *ifnull(df.at[i, 'com_name_codes'], '').split(),
                df.at[i, 'species_code'],
            ]))
            assert candidates, 'species_code should always be a candidate since it should always be unique'
            shorthand = candidates[0]
            used_shorthands.add(shorthand)
            shorthand_col.append(shorthand)
        df['shorthand'] = shorthand_col

        assert df.shorthand.nunique() == len(df), "shorthand isn't unique"
        assert (df
            [lambda df: pd.notnull(df.banding_codes)]
            .pipe(lambda df: (df.shorthand == df.banding_codes.str.split().str[0]).all())
        ), "shorthand doesn't equal banding_codes[0] (where present)"

        return df

    @property
    @lru_cache()
    @cache_to_file_forever(f'{metadata_dir}/cache/_df_lookup')  # Avoid ~12s (on a MBP)
    def _df_lookup(self) -> dict:
        out = dict([
            (self._normalize_query(query), dict(row))
            for col in reversed([
                # High -> low prio (shorthand, species_code, ...)
                #   - e.g. SOSP -> Song Sparrow (banding_codes), not Socotra Sparrow or Somali Sparrow (com_name)
                'shorthand',  # First so that we always return shorthand as is
                'species_code',  # Second, even though it'd be safe to swap into first with shorthand
                'banding_codes',  # After species_code but before everything else
                'com_name_codes',  # e.g. SOSP -> Socotra Sparrow, Somali Sparrow
                'sci_name_codes',
                'com_name',
                'sci_name',
            ])
            for i, row in (self.df
                .fillna('')
                .assign(
                    banding_codes=lambda df: df.banding_codes.str.split(),
                    com_name_codes=lambda df: df.com_name_codes.str.split(),
                    sci_name_codes=lambda df: df.sci_name_codes.str.split(),
                )
                .iterrows()
            )
            for query in ([row[col]] if not isinstance(row[col], list) else row[col])
        ])
        assert out[self._normalize_query('SOSP')]['com_name'] == 'Song Sparrow'
        assert out[self._normalize_query('WIWA')]['com_name'] == "Wilson's Warbler"
        return out

    def _normalize_query(self, query: str) -> str:
        return query.lower().replace("'", '')
