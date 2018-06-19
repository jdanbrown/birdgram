from collections import OrderedDict
from functools import lru_cache
import glob
from pathlib import Path
import re
from typing import Optional

from attrdict import AttrDict
from dataclasses import dataclass, field
import joblib
import pandas as pd
from potoo.pandas import as_ordered_cat, df_ordered_cat, df_reorder_cols

from cache import cache
import constants
from constants import data_dir, mul_species, no_species, unk_species, unk_species_species_code
from datatypes import Recording, RecordingDF
import metadata
from util import *

DATASETS = {
    'recordings': 'recordings/*',
    'peterson-field-guide': 'peterson-field-guide/*/audio/*',
    'xc': 'xc/data/*/*/*.mp3',
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
    elif dataset == 'xc':
        [_lit_xc, _lit_data, species_query, _id, _lit_audio] = id_parts
        assert (_lit_xc, _lit_data, _lit_audio) == ('xc', 'data', 'audio')
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


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
# @singleton
@dataclass(frozen=True)  # frozen=True so we can hash, for @lru_cache
class XC(DataclassUtil):

    # TODO Scrape more (non-audio) metadata from downloaded html (e.g. https://www.xeno-canto.org/417638):
    #   - remarks
    #   - background species
    #   - duration

    dir: Path = Path(f'{constants.data_dir}/xc')
    data_dir: Path = Path(f'{dir}/data')
    metadata_dir: Path = Path(f'{dir}/metadata/v1')

    @property
    def unsaved_ids(self) -> 'pd.Series[int]':
        id = self.metadata.id
        return pd.Series(sorted(set(range(id.max())) - set(id)))

    @property
    @lru_cache()
    @cache(version=3, key=lambda self: (
        self,
        # TODO Re-enable after we stop downloading all the time, since cache miss is slow (~45s)
        #   - In the meantime, manually bump version after big jumps in downloads
        # self._data_paths,
    ))
    def metadata(self) -> 'XCDF':
        """Make a full XCDF by joining _metadata + downloaded_ids"""
        return (self._metadata
            .set_index('id')
            .join(self.downloaded_ids.set_index('id'), how='left')
            .reset_index()
            .fillna({'downloaded': False})
            .pipe(XCDF)
        )

    @property
    @cache(version=3, key=lambda self: self)
    def _metadata(self) -> "pd.DataFrame['id': int, ...]":
        """Load all saved metadata from fs, keeping the latest observed metadata record per XC id"""
        metadata_paths_best_last = sorted(glob.glob(f'{self.metadata_dir}/*.pkl'))
        return (
            pd.concat([joblib.load(path) for path in metadata_paths_best_last])
            .drop_duplicates('id', keep='last')
            .astype({'id': 'int'})  # Back compat for when 'id' used to be a str [Safe to XXX after next save_metadata]
            .sort_values('id', ascending=False)
            .reset_index(drop=True)  # Index is junk
        )

    @property
    @lru_cache()
    def metadata_by_id(self) -> 'XCDF':
        """For fast lookup by id (e.g. xc.metadata_by_id.loc[id])"""
        return self.metadata.set_index('id')

    @property
    def downloaded_ids(self) -> "pd.DataFrame['id': int, 'downloaded': bool]":
        return (
            pd.DataFrame(XCResource.downloaded_ids(self._data_paths), columns=['id'])
            .assign(downloaded=True)
        )

    @property
    def _data_paths(self) -> Iterable[Path]:
        return list(self.data_dir.glob('*/*/*'))  # {data_dir}/{species}/{id}/{filename}

    def metadata_for_id(self, id: int) -> 'XCMetadata':
        assert isinstance(id, int)
        metadata = self.metadata_by_id.loc[id]
        assert isinstance(metadata, pd.Series), f'Index id[{id}] not unique: {metadata}'
        return XCMetadata(**{
            k: v
            for k, v in dict(metadata
                .copy()  # Else .set_value warns about trying to "set on a copy of a slice from a DataFrame"
                .set_value('id', metadata.name)  # Reset index on the series
            ).items()
            if k in XCMetadata.field_names(init=True)
        })

    def query_and_save(self, query, **kwargs) -> 'XCDF':
        xcdf = self.query(query, **kwargs)
        self.save_metadata(xcdf, name=query)
        return xcdf

    def query(self, *args, **kwargs) -> 'XCDF':
        return XCDF(self._query(*args, **kwargs))

    def save_metadata(self, xcdf: 'XCDF', name: str):
        nonce = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        path = f'{self.metadata_dir}/{nonce}-{name}.pkl'
        assert not os.path.exists(path)
        log('xc.save_metadata', **{'path': path, 'len(xcdf)': len(xcdf)})
        # TODO Persist xcdf._raw_df instead of xcdf, so that we can iterate on the XCDF transform without redownloading
        #   - Making this change will require either a redownload or some back-compat logic at load time (in .metadata)
        joblib.dump(xcdf, ensure_parent_dir(path))

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
                raise Exception('xeno-canto response: status_code[%s], content[%s]' % (
                    rep_http.status_code, rep_http.text,
                ))
            rep = AttrDict(rep_http.json())
            log('xc_query:page', query=query, **{k: v for k, v in rep.items() if k not in ['recordings']})
            # Yield recs
            for rec in rep.recordings:
                yield AttrDict(rec)
            # Next page
            if page >= min(rep.numPages, page_limit or np.inf):
                break
            page += 1


# Workaround for @singleton (above)
xc = XC()


@dataclass
class XCMetadata(DataclassUtil):
    """xeno-canto recording, lightly adapted from the XC api [https://www.xeno-canto.org/article/153]"""

    downloaded: bool
    species: str  # Synthetic FK to metadata.species (species -> shorthand)
    id: int
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
    license_type: str = field(default=None, init=False)  # Derived field
    license_detail: str = field(default=None, init=False)  # Derived field
    license: str
    url: str
    download: str

    # TODO De-dupe with XCDF, which repeats this as df operations to avoid a very slow bottleneck
    def __post_init__(self):
        license_parts = self.license.split('//creativecommons.org/licenses/')[-1].rstrip('/')
        self.license_type = license_parts.split('/', 1)[0]
        self.license_detail = license_parts.split('/', 1)[-1]

    @classmethod
    def for_id(cls, id: int) -> 'cls':
        """Lookup the XCResource for a given id"""
        assert isinstance(id, int)
        return xc.metadata_for_id(id)


@dataclass
class XCResource(DataclassUtil):

    species: str
    id: int
    downloaded: Optional[bool]

    # HACK Optional fields to avoid refactoring scrape_xc.py
    _metadata: AttrDict = field(default_factory=lambda: None, init=True, repr=False, compare=False)

    @property
    def dir(self) -> Path:
        return xc.data_dir.joinpath(self.species, str(self.id))

    @property
    def html_path(self) -> Path:
        return self.dir.joinpath('page.html')

    @property
    def audio_path(self) -> Path:
        return self.dir.joinpath('audio.mp3')

    @property
    def html_url(self):
        return f'https://www.xeno-canto.org/{self.id}'

    @property
    def audio_url(self):
        return f'https://www.xeno-canto.org/{self.id}/download'

    @classmethod
    def for_id(cls, id: int) -> 'cls':
        """Lookup the XCResource for a given id"""
        assert isinstance(id, int)
        return cls.from_metadata(XCMetadata.for_id(id))

    @classmethod
    def from_metadata(cls, metadata: XCMetadata) -> 'cls':
        """Convert an XCMetadata to an XCResource (no lookup required)"""
        return cls(
            species=metadata.species,
            id=metadata.id,
            downloaded=metadata.downloaded,
            _metadata=metadata,  # HACK Optional field to avoid refactoring scrape_xc.py
        )

    @classmethod
    def downloaded_ids(cls, data_paths: Iterator[Path]) -> Iterable[int]:
        """
        Map the set of downloaded data_paths to the set of id's that are (fully) downloaded
        - Needed by xc, but we own this concern to encapsulate that "downloaded" means "page.html and audio.mp3 exist"
        """
        data_paths_parts = (
            pd.DataFrame(
                (p.relative_to(xc.data_dir).parts for p in data_paths),
                columns=['species', 'id', 'filename'],
            )
            .astype({'id': 'int'})
        )
        return sorted(
            id
            for id, g in data_paths_parts.groupby('id')
            if set(g.filename) >= {'page.html', 'audio.mp3'}
        )


# TODO Maintain .id as an index? (Example use case: xc.metadata_by_id)
def XCDF(xcdf, *args, **kwargs) -> pd.DataFrame:
    """
    Make a xeno-canto df from any inputs that are valid for pd.DataFrame
    - Idempotent: XCDF(XCDF(...)) == XCDF(...)
    """

    # raw -> pseudo-XCDF
    #   - Not idempotent (e.g. col renames)
    if not (
        isinstance(xcdf, pd.DataFrame) and
        set(xcdf.columns) >= set(XCMetadata.field_names(init=True))  # init=True to exclude derived fields
    ):

        # Stash the raw data so that we can persist it instead of the transformed data (in xc.save_metadata), so that we can
        # iterate on the XCDF transform without redownloading
        #   - TODO Persist _raw_df, not xcdf, so that we can iterate on this transformation without redownloading
        _raw_df = pd.DataFrame(xcdf, *args, **kwargs)
        xcdf = _raw_df.copy()
        xcdf._raw_df = _raw_df

        # Transform raw -> XCDF
        xcdf = (xcdf
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
                sci_name=lambda df: df['gen'].str.cat(df['sp'], sep=' '),  # sci_name = '{genus} {species}'
            )
            .drop(columns=[
                'gen', 'sp',  # -> sci_name
            ])
        )

    import metadata  # Lazy import to avoid recursive module load
    quality_cats = ['A', 'B', 'C', 'D', 'E', 'no score']

    # pseudo-XCDF -> XCDF
    #   - Idempotent (i.e. must map an XCDF to itself) [TODO Write some tests for this]
    return (xcdf
        # Add 'species' FK to metadata.species.df.shorthand (ebird taxo)
        .assign(species=lambda df: df.apply(axis=1, func=lambda row: (
            metadata.species[row.sci_name, 'shorthand'] or
            metadata.species[row.com_name, 'shorthand'] or
            unk_species_species_code
        )))
        # Cleaning
        .assign(date=lambda df: df.date if df.date.dtype.name == 'datetime64[ns]' else (df.date
            .str.replace('-00', '-01')  # Coerce e.g. '2003-03-00' -> '2003-03-01'
            .pipe(pd.to_datetime, errors='coerce')  # Coerce remaining bad data to NaT (e.g. '1018-05-17')
        ))
        # Types
        .astype({
            'id': 'int',  # Else surprising sorting
            'lat': 'float',
            'lng': 'float',
        })
        # Derived cols from XCMetadata (before creating category cols, so that derived fields are available)
        # .apply(axis=1, func=lambda row: pd.Series(XCMetadata(**row).asdict()))  # XXX Very slow bottleneck
        .assign(
            # TODO De-dupe with XCMetadata.__post_init__, which is where I wish this could live as non-df operations
            _license_parts=lambda df: df.license.str.split('//creativecommons.org/licenses/', 1).str[-1].str.rstrip('/'),
            license_type=lambda df: df._license_parts.str.split('/', 1).str[0],
            license_detail=lambda df: df._license_parts.str.split('/', 1).str[-1],
        )
        .drop(columns=[
            '_license_parts',
        ])
        # Category cols
        .pipe(df_ordered_cat,
            quality=quality_cats,
            license_type=lambda df: sorted(df.license_type.unique()),
            license_detail=lambda df: sorted(df.license_detail.unique()),
        )
        .assign(
            # Reuse species (= ebird 'shorthand') cat from metadata.species.df
            #   - Drop unused cats since we never care about all potential cats at once
            species=lambda df: (df
                ['species'].astype('str').astype(metadata.species.df['shorthand'].dtype)
                .cat.remove_unused_categories()
            ),
            # Sort sci_name to match species
            #   - Drop unused cats since we never care about all potential cats at once
            sci_name=lambda df: (df
                ['sci_name'].pipe(
                    as_ordered_cat,
                    list(df[['species', 'sci_name']].drop_duplicates().sort_values('species')['sci_name'])
                )
                .cat.remove_unused_categories()
            ),
            # Sort com_name to match species
            #   - Drop unused cats since we never care about all potential cats at once
            com_name=lambda df: (df
                ['com_name'].pipe(
                    as_ordered_cat,
                    list(df[['species', 'com_name']].drop_duplicates().sort_values('species')['com_name'])
                )
                .cat.remove_unused_categories()
            ),
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
class Birdclef2015Rec(DataclassUtil):
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
