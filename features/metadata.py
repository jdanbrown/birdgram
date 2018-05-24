from collections import OrderedDict
from functools import lru_cache
import re
from typing import List, Optional

import pandas as pd
from potoo.pandas import as_ordered_cat, df_transform_column_names

from constants import (
    data_dir,
    mul_species, mul_species_com_name, mul_species_species_code, mul_species_taxon_id,
    no_species, no_species_com_name, no_species_species_code, no_species_taxon_id,
    unk_species, unk_species_com_name, unk_species_species_code, unk_species_taxon_id,
)
from datatypes import Species
from util import cache_to_file_forever, df_reorder_cols, singleton

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
            (query, attr) = x
        except:
            (query, attr) = (x, None)
        if ',' in query:
            # Map queries containing ',' to "Muliple species"
            #   - TODO Figure out a way to handle multi-labeled data
            query = '_MUL'
        query = self._normalize_query(query or '')
        species = self._df_lookup.get(query)
        if not species or not attr:
            return species
        else:
            return getattr(species, attr)

    @property
    @lru_cache()
    @cache_to_file_forever(f'{metadata_dir}/cache/_taxa')  # Avoid ~250ms (on a MBP)
    def df(self):
        """The full species df (based on http://ebird.org/ws1.1/ref/taxa/ebird?cat=species&fmt=csv)"""

        synthetic_row = lambda species, com_name, species_code, taxon_id: dict(
            sci_name=com_name,
            com_name=com_name,
            taxon_id=taxon_id,
            species_code=species_code,
            taxon_order=species,
            com_name_codes=species,
            sci_name_codes=species,
            banding_codes=species,
        )

        assert (self._raw_ebird_df.CATEGORY == 'species').all()
        return (self._raw_ebird_df
            .drop(columns=[
                'CATEGORY',
            ])
            .rename(columns={
                'SCIENTIFIC_NAME': 'sci_name',
                'COMMON_NAME': 'com_name',
                'TAXON_ID': 'taxon_id',
                'SPECIES_CODE': 'species_code',
                'TAXON_ORDER': 'taxon_order',
                'COM_NAME_CODES': 'com_name_codes',
                'SCI_NAME_CODES': 'sci_name_codes',
                'BANDING_CODES': 'banding_codes',
            })
            # Add synthetic species
            .append(pd.DataFrame([
                # Unknown species (species present but not labeled)
                synthetic_row(unk_species, unk_species_com_name, unk_species_species_code, unk_species_taxon_id),
                # Multiple species [TODO Figure out a way to handle multi-labeled data]
                synthetic_row(mul_species, mul_species_com_name, mul_species_species_code, mul_species_taxon_id),
                # No species (no species present)
                synthetic_row(no_species, no_species_com_name, no_species_species_code, no_species_taxon_id),
            ]))
            .pipe(self.add_shorthand_col)
            .assign(
                longhand=lambda df: df.apply(axis=1, func=lambda row: f'{row.com_name} - {row.shorthand}'),
            )
            .assign(
                sci_name=lambda df: as_ordered_cat(df.sci_name),
                com_name=lambda df: as_ordered_cat(df.com_name),
                taxon_id=lambda df: as_ordered_cat(df.taxon_id),
                species_code=lambda df: as_ordered_cat(df.species_code),
                shorthand=lambda df: as_ordered_cat(df.shorthand),
                longhand=lambda df: as_ordered_cat(df.longhand),
            )
            .pipe(df_reorder_cols, first=[
                'shorthand',
                'longhand',
                'sci_name',
                'com_name',
                'taxon_id',
                'species_code',
                'taxon_order',
                'com_name_codes',
                'sci_name_codes',
                'banding_codes',
            ])
            .reset_index(drop=True)
        )

    @property
    def _raw_ebird_df(self):
        """The raw ebird species df (from http://ebird.org/ws1.1/ref/taxa/ebird?cat=species&fmt=csv)"""
        return pd.read_csv(
            ebird_taxa_path,
            dtype={
                'TAXON_ORDER': 'str',  # Not float
            },
        )

    def add_shorthand_col(self, df) -> pd.DataFrame:
        """Compute the unique 'shorthand' column and add it to the input df"""

        # Don't mutate the input
        input_df = df
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

        # Compute shorthand_col
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

        # Restore the order of the input df (e.g. taxo order from _raw_ebird_df)
        shorthand_df = df
        df = pd.merge(
            input_df,
            shorthand_df[['species_code', 'shorthand']],
            how='left',
            on='species_code',
        )

        # Integrity checks
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
            (self._normalize_query(query), Species(**row))
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
                .fillna(dict(
                    banding_codes='',
                    com_name_codes='',
                    sci_name_codes='',
                ))
                .assign(
                    banding_codes=lambda df: df.banding_codes.str.split(),
                    com_name_codes=lambda df: df.com_name_codes.str.split(),
                    sci_name_codes=lambda df: df.sci_name_codes.str.split(),
                )
                .iterrows()
            )
            for query in ([row[col]] if not isinstance(row[col], list) else row[col])
        ])
        assert out[self._normalize_query('SOSP')].com_name == 'Song Sparrow'
        assert out[self._normalize_query('WIWA')].com_name == "Wilson's Warbler"
        return out

    def _normalize_query(self, query: str) -> str:
        return query.lower().replace("'", '')


def sorted_species(species_queries: List[str], **kwargs) -> List[str]:
    """Sort a list of species query strings in taxo order"""
    return sorted(species_queries, **kwargs, key=lambda x: species[x].taxon_id)


@singleton
class xc_counts:
    """
    xeno-canto recordings counts
    - From https://docs.google.com/spreadsheets/d/1DNBi3jQ3NdFmexTfOOh_VmwqqHyBhEmh9kfVVbeRGK8/
    """

    @property
    def with_species(self):
        return pd.merge(
            species.df,
            self.df,
            how='left',
            on='species_code',
        )

    # FIXME Join on com_name after sci_name, to catch sci_names that don't match, e.g.
    #   - xc_counts: com_name='Orange-crowned Warbler', sci_name='Leiothlypis celata'
    #   - species:   com_name='Orange-crowned Warbler', sci_name='Oreothlypis celata'
    @property
    @lru_cache()
    def df(self):
        return (
            pd.merge(
                species.df,
                self._raw_df.drop(columns=['com_name']),
                how='left',
                on='sci_name',
            )
            [['species_code', 'n_recs', 'n_bg_recs']]
            .rename(columns={
                'n_recs': 'n_xc_recs',
                'n_bg_recs': 'n_xc_bg_recs',
            })
        )

    @property
    def _raw_df(self):
        return (pd.read_csv(f'{metadata_dir}/xeno-canto-rec-count-per-species.csv')
            # Lowercase col names so we don't break when we e.g. twiddle casing in the spreadsheet
            .pipe(df_transform_column_names, lambda c: c.lower())
            .drop(columns=[
                'unnamed: 5',
            ])
            .rename(columns={
                'common name': 'com_name',
                'scientific name': 'sci_name',
                'extinct?': 'extinct',
                '# recs': 'n_recs',
                '# bg recs': 'n_bg_recs',
                'species index': 'species_index',
            })
        )
