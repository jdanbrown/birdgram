from collections import OrderedDict
from functools import lru_cache
import glob
import re

from attrdict import AttrDict
from dataclasses import dataclass
import joblib
import pandas as pd
from potoo.pandas import as_ordered_cat, df_reorder_cols

from constants import data_dir, mul_species, no_species, unk_species, unk_species_species_code
from datatypes import Recording, RecordingDF
import metadata
from util import *

DATASETS = {
    'recordings': 'recordings/*',
    'peterson-field-guide': 'peterson-field-guide/*/audio/*',
    'birdclef-2015': 'birdclef-2015/organized/wav/*',
    'warblrb10k': 'dcase-2018/warblrb10k_public_wav/*',
    'ff1010bird': 'dcase-2018/ff1010bird_wav/*',
    'nips4b': 'nips4b/all_wav/*',
    'mlsp-2013': 'mlsp-2013/mlsp_contest_dataset/essential_data/src_wavs/*',
}


def metadata_from_dataset(id: str, dataset: str) -> AttrDict:
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
    return AttrDict(
        # TODO De-dupe these with Load.METADATA
        species=species.shorthand,
        species_longhand=species.longhand,
        species_com_name=species.com_name,
        species_query=species_query,
        basename=basename,
    )


#
# xeno-canto (xc)
#


@singleton
class xc:

    # TODO Download audio file from metadata
    # TODO Scrape more (non-audio) metadata from .url (e.g. https://www.xeno-canto.org/417638):
    #   - remarks
    #   - background species

    dir: str = f'{data_dir}/xc'
    metadata_dir: str = f'{dir}/metadata'

    @property
    def unsaved_ids(self) -> 'pd.Series[int]':
        id = self.metadata.id.astype('int')  # str -> int
        return pd.Series(sorted(set(range(id.max())) - set(id)))
        # return set(range(id.max())), set(id)

    @property
    @lru_cache()
    def metadata(self) -> 'XCDF':
        """Load all saved metadata from fs, keeping the latest observed metadata record per XC id"""
        metadata_paths_best_last = sorted(glob.glob(f'{self.metadata_dir}/*.pkl'))
        return (
            pd.concat([joblib.load(path) for path in metadata_paths_best_last])
            .drop_duplicates('id', keep='last')
            .sort_values('id', ascending=False)
            .reset_index(drop=True)  # Index is uninformative
        )

    def save_metadata(self, df: 'XCDF', name: str):
        nonce = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        path = f'{self.metadata_dir}/{nonce}-{name}.pkl'
        assert not os.path.exists(path)
        log('xc.save_metadata', **{'path': path, 'len(df)': len(df)})
        joblib.dump(df, ensure_parent_dir(path))

    def query_and_save(self, query, **kwargs) -> 'XCDF':
        df = self.query(query, **kwargs)
        self.save_metadata(df, name=query)
        return df

    def query(self, *args, **kwargs) -> 'XCDF':
        return XCDF(self._query(*args, **kwargs))

    def _query(self, query: str, page_limit: int = None) -> Iterator[AttrDict]:
        log('xc:query', query=query, page_limit=page_limit)
        page = 1
        while True:
            # Request
            rep_http = requests.get('https://www.xeno-canto.org/api/2/recordings', params=dict(
                page=page,
                query=query,
            ))
            if rep_http.status_code != 200:
                raise Exception('xeno-canto response: status_code[%s], content[%s]' % (rep_http.status_code, rep_http.text))
            rep = AttrDict(rep_http.json())
            log('xc_query:page', query=query, **{k: v for k, v in rep.items() if k not in ['recordings']})
            # Yield recs
            for rec in rep.recordings:
                yield AttrDict(rec)
            # Next page
            if page >= min(rep.numPages, page_limit or np.inf):
                break
            page += 1


@dataclass
class XCMetadata(DataclassConversions):
    """xeno-canto recording, lightly adapted from the XC api [https://www.xeno-canto.org/article/153]"""
    species_code: str  # Synthetic FK to metadata.species
    id: str
    sci_name: str
    com_name: str
    subspecies: str
    recordist: str
    country: str
    locality: str
    lat: str
    lng: str
    type: str
    quality: str
    date: pd.Timestamp
    time: str  # '?' for unknown
    download: str
    license: str
    url: str


# TODO Should this be idempotent? (currently assumes input is shaped like the XC api)
def XCDF(*args, **kwargs) -> pd.DataFrame:
    """xeno-canto df"""
    import metadata  # Lazy import to avoid recursive module load
    quality_cats = ['A', 'B', 'C', 'D', 'E', 'no score']
    return (
        pd.DataFrame(*args, **kwargs)
        # Remap XC cols to be more ergonomic
        .rename(columns={
            'en': 'com_name',
            'ssp': 'subspecies',
            'rec': 'recordist',
            'cnt': 'country',
            'loc': 'locality',
            'q': 'quality',
            'file': 'download',
            'lic': 'license',
        })
        .assign(
            sci_name=lambda df: df['gen'].str.cat(df['sp'], sep=' '),
        )
        .drop(columns=[
            'gen', 'sp',  # -> sci_name
        ])
        # Add species_code as FK to metadata.species (ebird taxo)
        .assign(species_code=lambda df: df.apply(axis=1, func=lambda row: (
            metadata.species[row.sci_name, 'species_code'] or
            metadata.species[row.com_name, 'species_code'] or
            unk_species_species_code
        )))
        # Create category cols
        .assign(
            # Reuse species_code cat from metadata.species.df
            #   - Drop unused cats since we never care about all potential cats at once
            species_code=lambda df: (df
                ['species_code'].astype('str').astype(metadata.species.df['species_code'].dtype)
                .cat.remove_unused_categories()
            ),
            # Sort sci_name to match species_code
            #   - Drop unused cats since we never care about all potential cats at once
            sci_name=lambda df: (df
                ['sci_name'].pipe(
                    as_ordered_cat,
                    list(df[['species_code', 'sci_name']].drop_duplicates().sort_values('species_code')['sci_name'])
                )
                .cat.remove_unused_categories()
            ),
            # Sort com_name to match species_code
            #   - Drop unused cats since we never care about all potential cats at once
            com_name=lambda df: (df
                ['com_name'].pipe(
                    as_ordered_cat,
                    list(df[['species_code', 'com_name']].drop_duplicates().sort_values('species_code')['com_name'])
                )
                .cat.remove_unused_categories()
            ),
            # Don't drop unused cats, since we do care to think about them all at once (6 total)
            quality=lambda df: df['quality'].pipe(as_ordered_cat, quality_cats),
        )
        # Order cols like XC fields
        .pipe(df_reorder_cols, first=[x.name for x in dataclasses.fields(XCMetadata)])
    )


#
# birdclef-2015
#


@singleton
class birdclef2015:

    # TODO TODO Finish fleshing this out (20180530_dataset_birdclef.ipynb + 20180524_eval_birdclef.ipynb)

    def xml_data(self, recs: RecordingDF) -> pd.DataFrame:
        return pd.DataFrame([self.xml_dict_for_rec(rec) for rec in df_rows(recs)])

    def xml_dict_for_rec(self, rec: Recording) -> dict:

        # wav_path -> (<Audio>, 'train'|'test')
        wav_path = os.path.join(data_dir, rec.path)
        xml_path_prefix = wav_path.replace('/wav/', '/xml/').replace('.wav', '')
        train_path = glob.glob(f'{xml_path_prefix}-train.xml')
        test_path = glob.glob(f'{xml_path_prefix}-test.xml')
        assert bool(train_path) != bool(test_path), \
            f'Failed to find train_path[{train_path}] xor test_path[{test_path}] for wav_path[{wav_path}]'
        [xml_path], train_test = (train_path, 'train') if train_path else (test_path, 'test')
        with open(xml_path) as f:
            audio_elem = ET.fromstring(f.read())
        assert audio_elem.tag == 'Audio'

        # (<Audio>, 'train'|'test') -> xml_dict
        xml_dict = {
            self._snakecase_xml_key(e.tag): e.text.strip() if e.text else e.text
            for e in audio_elem
        }

        return xml_dict

    def _snakecase_xml_key(self, key: str) -> str:
        key = stringcase.snakecase(key)
        key = {
            # Patch up weird cases
            'author_i_d': 'author_id',  # Oops: 'AuthorID' became 'author_i_d'
        }.get(key, key)
        return key


@dataclass
class Birdclef2015Rec(DataclassConversions):
    """birdclef2015 recording"""
    media_id: int
    class_id: str
    vernacular_names: str
    family: str
    order: str
    genus: str
    species: str
    sub_species: str
    background_species: str
    author_id: str
    author: str
    elevation: int
    locality: str
    latitude: float
    longitude: float
    content: str
    quality: int
    date: str
    time: str
    comments: str
    file_name: str
    year: str


#
# mlsp-2013
#


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
