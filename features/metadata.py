from collections import OrderedDict
from functools import lru_cache
import re
from typing import Iterable, List, Optional

from more_itertools import unique_everseen
import pandas as pd
from potoo.pandas import as_ordered_cat, df_ordered_cat, df_reorder_cols, df_transform_column_names

from constants import (
    data_dir,
    mul_species, mul_species_com_name, mul_species_species_code, mul_species_taxon_id,
    no_species, no_species_com_name, no_species_species_code, no_species_taxon_id,
    unk_species, unk_species_com_name, unk_species_species_code, unk_species_taxon_id,
)
from datatypes import Species
from util import cache_to_file_forever, df_rows, singleton

metadata_dir = f'{data_dir}/metadata'
ebird_taxa_path = f'{metadata_dir}/ebird-ws1.1-taxa-species.csv'
ebird_clements_checklist_path = f'{metadata_dir}/eBird-Clements-v2018-integrated-checklist-August-2018.csv.gz'
df_cache_file = f'{metadata_dir}/cache/df-v0'


@singleton
class ebird:

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
        query = query or ''
        if ',' in query:
            # Map queries containing ',' to "Muliple species"
            #   - TODO Figure out a way to handle multi-labeled data
            query = '_MUL'
        query = self._normalize_query(query)
        species = self._df_lookup.get(query)
        if not species or not attr:
            return species
        else:
            return getattr(species, attr)

    def search(self, query: str) -> pd.DataFrame:
        tokens = query.lower().split()
        return (self.df
            [lambda df: (df
                .apply(axis=1, func=lambda row: (
                    all(
                        any(
                            t in x
                            for x in row.astype('str').str.lower()
                        )
                        for t in tokens
                    )
                ))
            )]
        )

    # HACK Ad-hoc function for constants.*_com_names_*
    #   - TODO Unify api with __getitem__
    #   - TODO Clean up along with xc.com_names_to_species
    def com_names_to_species(self, com_names: Iterable[str], check=True) -> Iterable[str]:
        manual_renames = {
            # Any com_name (e.g. a newer ebird taxo than self.df) -> com_name in self.df
            "Canada Jay": "Gray Jay",  # https://www.allaboutbirds.org/guide/Canada_Jay/
            "Cinnamon-rumped Seedeater": "White-collared Seedeater",  # https://ebird.org/camerica/news/taxonomy-update-central-america-2018
            "Morelet's Seedeater": "White-collared Seedeater",  # https://en.wikipedia.org/wiki/White-collared_seedeater
            "Mexican Duck": "Mallard",  # https://en.wikipedia.org/wiki/Mexican_duck
            "Leach's/Townsend's Storm-Petrel (dark-rumped)": "Leach's Storm-Petrel",  # https://en.wikipedia.org/wiki/Leach%27s_storm_petrel
            "Leach's/Townsend's Storm-Petrel (white-rumped)": "Leach's Storm-Petrel",  # https://en.wikipedia.org/wiki/Leach%27s_storm_petrel
            "Chiriqui Foliage-gleaner": "Buff-throated Foliage-gleaner",  # https://ebird.org/camerica/news/taxonomy-update-central-america-2018
            "Middle American Screech-Owl": "Vermiculated Screech-Owl",  # https://ebird.org/camerica/news/taxonomy-update-central-america-2018
            "Mistletoe Tyrannulet": "Paltry Tyrannulet",  # https://ebird.org/camerica/news/taxonomy-update-central-america-2018
            "Scarlet-rumped Tanager": "Passerini's Tanager",  # https://ebird.org/camerica/news/taxonomy-update-central-america-2018
        }
        com_names = set(com_names)
        com_names = {manual_renames.get(x, x) for x in com_names}
        res = self.df[['com_name', 'shorthand']][lambda df: df.com_name.isin(com_names)]
        unmatched = list(set(com_names) - set(res.com_name))
        if check and unmatched:
            raise ValueError('Unmatched com_names: %s' % unmatched)
        return res.shorthand.sort_values().tolist()

    @property
    @lru_cache()
    @cache_to_file_forever(df_cache_file)  # Avoid ~250ms (local MBP)
    def df(self):
        """The full species df (based on http://ebird.org/ws1.1/ref/taxa/ebird?cat=species&fmt=csv)"""

        synthetic_row = lambda species, com_name, species_code, taxon_id: dict(
            sci_name=com_name,
            com_name=com_name,
            species_code=species_code,
            taxon_order=species,
            taxon_id=taxon_id,
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
                'SPECIES_CODE': 'species_code',
                'TAXON_ORDER': 'taxon_order',
                'TAXON_ID': 'taxon_id',
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
                'species_code',
                'taxon_order',
                'taxon_id',
                'com_name_codes',
                'sci_name_codes',
                'banding_codes',
            ])
            .reset_index(drop=True)
            .pipe(self.add_species_group_cols)
        )

    @property
    def _raw_ebird_df(self):
        """The raw ebird species df (from http://ebird.org/ws1.1/ref/taxa/ebird?cat=species&fmt=csv)"""
        return pd.read_csv(
            ebird_taxa_path,
            dtype={
                'TAXON_ORDER': 'str',  # Not float [TODO Why did I decide str instead of float?]
            },
        )

    @property
    def _raw_ebird_clements_taxo_df(self):
        """The raw eBird/Clements Checklist taxonomy (from http://www.birds.cornell.edu/clementschecklist/download/)"""
        return (
            pd.read_csv(
                ebird_clements_checklist_path,
                encoding='latin1',  # utf8 barfs, latin1 seems to work better
                low_memory=False,  # Scan full file for dtype inference, else problems
            )
            .query("category == 'species'")
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

    def add_species_group_cols(self, species_df) -> pd.DataFrame:
        """Join [species_group, family, order] from the eBird/Clements taxo onto species_df"""

        # taxo_df has the species_group_cols we want to join onto species_df
        species_group_cols = ['species_group', 'family', 'order']
        taxo_df = (self._raw_ebird_clements_taxo_df
            # Simplify col names
            .rename(columns={
                'scientific name': 'sci_name',
                'English name': 'com_name',
                'eBird species group': 'species_group',
                'eBird species code 2018': 'species_code',
            })
        )

        # Manual joins for magics, e.g. _UNK, _MUL, _NON
        manual_magics = pd.DataFrame([
            dict(
                species_code=row.species_code,
                species_group=row.com_name,
                family=row.com_name,
                order=row.com_name,
            )
            for row in df_rows(species_df
                [lambda df: df.shorthand.str.startswith('_')]
            )
        ])

        # Manual joins for lumps/splits/renames
        manual_renames = {
            # col -> taxo_df str -> species_df str
            'com_name': {
                "Scarlet-rumped Tanager": "Cherrie's Tanager", # https://en.wikipedia.org/wiki/Cherrie%27s_tanager
                "Sunda Bush Warbler": "Timor Bush Warbler", # https://www.hbw.com/species/sunda-grasshopper-warbler-locustella-montis
                "Goldcrest": "Canary Islands Kinglet", # https://avibase.bsc-eoc.org/species.jsp?avibaseid=AC00D42656FE4E4D
                "Icterine Greenbul": "Liberian Greenbul", # https://en.wikipedia.org/wiki/Icterine_greenbul
                "Line-cheeked Spinetail": "Baron's Spinetail", # https://en.wikipedia.org/wiki/Line-cheeked_spinetail
                "Yellow-rumped Tinkerbird": "White-chested Tinkerbird", # https://en.wikipedia.org/wiki/White-chested_tinkerbird
                "Rufescent Screech-Owl": "Colombian Screech-Owl", # https://en.wikipedia.org/wiki/Rufescent_screech_owl
            },
        }

        # Add new species_group cols
        joined_df = (species_df

            # Prep for joins
            .assign(
                species_group=None,
                family=None,
                order=None,
            )

            # Join manual_magics
            .pipe(lambda df: (df
                .set_index('species_code')
                .join(how='left', other=(manual_magics
                    .set_index('species_code')[species_group_cols].rename(columns=lambda c: c + '_y')
                ))
                .assign(**{k: lambda df, k=k: df[k].combine_first(df[k + '_y']) for k in species_group_cols})
                .drop(columns=[c + '_y' for c in species_group_cols])
                .reset_index()
            ))

            # Join on species_code
            #   - Leaves 31 rows unmatched
            .pipe(lambda df: (df
                .set_index('species_code')
                .join(how='left', other=(taxo_df
                    .set_index('species_code')[species_group_cols].rename(columns=lambda c: c + '_y')
                ))
                .assign(**{k: lambda df, k=k: df[k].combine_first(df[k + '_y']) for k in species_group_cols})
                .drop(columns=[c + '_y' for c in species_group_cols])
                .reset_index()
                .astype({'species_code': species_df['species_code'].dtype})  # Restore category (if present)
            ))

            # Join unmatched rows on sci_name
            #   - Leaves 7/31 rows unmatched
            .pipe(lambda df: (df
                .set_index('sci_name')
                .join(how='left', other=(taxo_df
                    .set_index('sci_name')[species_group_cols].rename(columns=lambda c: c + '_y')
                ))
                .assign(**{k: lambda df, k=k: df[k].combine_first(df[k + '_y']) for k in species_group_cols})
                .drop(columns=[c + '_y' for c in species_group_cols])
                .reset_index()
                .astype({'sci_name': species_df['sci_name'].dtype})  # Restore category (if present)
            ))

            # Join unmatched rows manually, via com_name
            #   - Leaves 0/7 rows unmatched
            #   - Joining on com_name without manual_renames matches no new rows; we join on com_name only to express the rename mapping
            .pipe(lambda df: (df
                .set_index('com_name')
                .join(how='left', other=(taxo_df
                    .replace(manual_renames)
                    .set_index('com_name')[species_group_cols].rename(columns=lambda c: c + '_y')
                ))
                .assign(**{k: lambda df, k=k: df[k].combine_first(df[k + '_y']) for k in species_group_cols})
                .drop(columns=[c + '_y' for c in species_group_cols])
                .reset_index()
                .astype({'com_name': species_df['com_name'].dtype})  # Restore category (if present)
            ))

            # Add ordered categories for species_group_cols
            .pipe(df_ordered_cat, **{
                k: lambda df, k=k: list(unique_everseen(df.sort_values('species_code')[k]))
                for k in species_group_cols
            })

            # Order cols
            .pipe(df_reorder_cols, first=species_df.columns, last=species_group_cols)

        )

        # Integrity checks
        assert {joined_df[c].isnull().sum() for c in species_group_cols} == {0}

        return joined_df

    @property
    @lru_cache()
    @cache_to_file_forever(f'{df_cache_file}._df_lookup')  # Avoid ~12s (local MBP)
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


# Back compat [TODO Remove all old usage of metadata.species]
species = ebird


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
