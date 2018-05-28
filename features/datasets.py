from collections import OrderedDict
from functools import lru_cache
import re

import pandas as pd
from potoo.pandas import df_reorder_cols

from constants import data_dir, mul_species, no_species, unk_species
import metadata
from util import singleton

datasets = {
    'recordings': 'recordings/*',
    'peterson-field-guide': 'peterson-field-guide/*/audio/*',
    'birdclef-2015': 'birdclef-2015/organized/wav/*',
    'warblrb10k': 'dcase-2018/warblrb10k_public_wav/*',
    'ff1010bird': 'dcase-2018/ff1010bird_wav/*',
    'nips4b': 'nips4b/all_wav/*',
    'mlsp-2013': 'mlsp-2013/mlsp_contest_dataset/essential_data/src_wavs/*',
}


def metadata_from_audio(id: str, dataset: str) -> dict:
    id_parts = id.split('/')
    basename = id_parts[-1]
    species_query = None
    if dataset == 'peterson-field-guide':
        species_query = id.split('/')[1]
    elif dataset == 'recordings':
        m = re.match(r'^([A-Z]{4}) ', basename)
        species_query = m.groups()[0] if m else unk_species
    elif dataset == 'mlsp-2013':
        train_labels = mlsp2013.train_labels_for_filename.get(
            basename,
            [unk_species],  # If missing it's an unlabeled test rec
        )
        species_query = ','.join(sorted(train_labels)) if train_labels else no_species
        # TODO Generalize species[species_query] to work on multi-label species (e.g. 'SOSP,WIWA')
        #   - Works fine for now because it passes through queries it doesn't understand, and these are already codes
        # species = ','.join(sorted(train_labels)) if train_labels else no_species
        # species_longhand = species
        # species_com_name = species
    species = metadata.species[species_query] or metadata.species[unk_species]
    return OrderedDict(
        species=species.shorthand,
        species_longhand=species.longhand,
        species_com_name=species.com_name,
        species_query=species_query,
        basename=basename,
    )


@singleton
class mlsp2013:

    def __init__(self):
        self.dir = f'{data_dir}/mlsp-2013'

    @property
    @lru_cache()
    def labels(self):
        pass

    @property
    @lru_cache()
    def rec_id2filename(self):
        return pd.read_csv(f'{self.dir}/mlsp_contest_dataset/essential_data/rec_id2filename.txt')

    @property
    @lru_cache()
    def sample_submission(self):
        return pd.read_csv(f'{self.dir}/mlsp_contest_dataset/essential_data/sample_submission.csv')

    @property
    @lru_cache()
    def species_list(self):
        return pd.read_csv(f'{self.dir}/mlsp_contest_dataset/essential_data/species_list.txt')

    @property
    @lru_cache()
    def rec_labels_test_hidden(self):
        # Has variable numbers of columns (multiple labels per rec_id), so parse it manually
        with open(f'{self.dir}/mlsp_contest_dataset/essential_data/rec_labels_test_hidden.txt') as f:
            return (
                pd.DataFrame(line.rstrip().split(',', 1) for line in f.readlines())
                .T.set_index(0).T  # Pull first row into df col names
            )

    @property
    def test_recs(self):
        return self.rec_labels_test_hidden[lambda df: df['[labels]'] == '?'][['rec_id']]

    @property
    def _train_labels_raw(self):
        return self.rec_labels_test_hidden[lambda df: df['[labels]'] != '?']

    @property
    @lru_cache()
    def train_labels(self):
        return (self._train_labels_raw
            .astype({'rec_id': 'int'})
            .fillna({'[labels]': '-1'})
            .set_index('rec_id')['[labels]']
            .map(lambda s: [int(x) for x in s.split(',') if x != ''])
            .apply(pd.Series).unstack()  # flatmap
            .reset_index(level=0, drop=True)  # Drop 'level' index
            .sort_index().reset_index()  # Sort and reset 'rec_id' index
            .rename(columns={0: 'class_id'})
            .dropna()
            .merge(self.species_list, how='left', on='class_id').drop(columns=['class_id'])
            .merge(self.rec_id2filename, how='left', on='rec_id')
            .pipe(df_reorder_cols, first=['rec_id', 'filename'])
        )

    @property
    @lru_cache()
    def train_labels_for_filename(self) -> dict:
        return (mlsp2013.train_labels
            .groupby('filename')['code']
            .apply(lambda s: [x for x in s if pd.notnull(x)])
            .pipe(dict)
        )
