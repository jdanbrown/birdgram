from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Optional, Union

from attrdict import AttrDict
from dataclasses import dataclass, field
import joblib
from more_itertools import ilen
import pandas as pd
import parse
from potoo.pandas import as_ordered_cat, df_ordered_cat, df_reorder_cols
from potoo.util import or_else, strip_startswith
import requests
import requests_html
from tqdm import tqdm

from cache import cache
import constants
from constants import code_dir, data_dir, mul_species, no_species, unk_species, unk_species_species_code
from dataset.birdclef2015 import *  # For export
from dataset.mlsp2013 import *  # For export
from datatypes import Recording, RecordingDF
from log import log
import metadata
from metadata import ebird  # For export
from util import *

DATASETS = {
    'recordings': dict(
        # ≥.5g
        root='recordings',
        audio_glob='*.wav',
    ),
    'peterson-field-guide': dict(
        # ≥.3g
        root='peterson-field-guide',
        audio_glob='*/audio/*.mp3',
    ),
    'xc': dict(
        # ≥98g
        root='xc',
        audio_glob='data/*/*/*.mp3',
    ),
    # [Moved to external drive to save space]
    # 'birdclef-2015': dict(
    #     # 94g
    #     root='birdclef-2015',
    #     audio_glob='organized/wav/*.wav',
    # ),
    # 'warblrb10k': dict(
    #     # 6.7g
    #     root='dcase-2018/warblrb10k_public_wav',
    #     audio_glob='*.wav',
    # ),
    # 'ff1010bird': dict(
    #     # 6.4g
    #     root='dcase-2018/ff1010bird_wav',
    #     audio_glob='*.wav',
    # ),
    # 'nips4b': dict(
    #     # 1.1g
    #     root='nips4b',
    #     audio_glob='all_wav/*.wav',
    # ),
    # 'mlsp-2013': dict(
    #     # 1.3g
    #     root='mlsp-2013',
    #     audio_glob='mlsp_contest_dataset/essential_data/src_wavs/*.wav',
    # ),
}


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
@dataclass
class _audio_path_files(DataclassUtil):
    """Manage the data/**/audio-paths.jsonl files"""

    @generator_to(pd.DataFrame)
    def list(self, *datasets: str) -> Iterator[dict]:
        """List the data/**/audio-paths.jsonl files"""
        for dataset, config in DATASETS.items():
            if not datasets or dataset in datasets:
                root = Path(data_dir) / config['root']
                audio_paths_path = root / 'audio-paths.jsonl'
                with audio_paths_path.open() as f:
                    lines = ilen(line for line in f if line)
                yield dict(
                    updated_at=pd.to_datetime(None if not audio_paths_path.exists() else audio_paths_path.stat().st_mtime * 1e9),
                    dataset=dataset,
                    path=str(audio_paths_path.relative_to(code_dir)),
                    lines=lines,
                )

    def update(self, *datasets: str):
        """Call this to update the data/**/audio-paths.jsonl files, which determine which audio paths get loaded"""
        for dataset, config in DATASETS.items():
            if not datasets or dataset in datasets:
                root = Path(data_dir) / config['root']
                audio_paths_path = root / 'audio-paths.jsonl'
                log.info('Globbing... %s' % (root / config['audio_glob']).relative_to(code_dir))
                audio_paths = pd.DataFrame([
                    dict(path=str(path.relative_to(root)))
                    for path in tqdm(unit=' paths', iterable=root.glob(config['audio_glob']))
                ])
                audio_paths.to_json(audio_paths_path, orient='records', lines=True)
                log.info('Wrote: %s (%s paths)' % (audio_paths_path.relative_to(code_dir), len(audio_paths)))

    def read(self, dataset: str) -> Iterable[str]:
        """Read audio paths from data/**/audio-paths.jsonl"""
        config = DATASETS[dataset]
        root = Path(data_dir) / config['root']
        audio_paths_path = root / 'audio-paths.jsonl'
        if not audio_paths_path.exists():
            raise ValueError(f'File not found: {audio_paths_path}')
        audio_paths = pd.read_json(audio_paths_path, lines=True)
        return [root / p for p in audio_paths['path']]


# Workaround for @singleton (above)
audio_path_files = _audio_path_files()


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
        [_const_xc, _const_data, species_query, _id, *_rest] = id_parts
        assert (_const_xc, _const_data) == ('xc', 'data')
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


def com_names_to_species(metadata_name: str, com_names: Iterable[str], **kwargs) -> Iterable[str]:
    metadata = {
        'xc': xc,
        'ebird': ebird,
    }[metadata_name]
    return metadata.com_names_to_species(com_names, **kwargs)


#
# xeno-canto (xc)
#   - TODO Refactor to split this out into a new module dataset.xc
#       - Break cycle: xc._audio_paths (should move to dataset.xc) currently depends on audio_path_files (should stay here)
#


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
@dataclass(frozen=True)  # frozen=True so we can hash, for @lru_cache
class _xc(DataclassUtil):

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
    @cache(version=7, key=lambda self: (self, self._audio_paths_hash))
    def metadata(self) -> 'XCDF':
        """Make a full XCDF by joining _metadata + downloaded_ids"""
        return (self._metadata
            .set_index('id')
            .join(self.downloaded_ids.set_index('id'), how='left')
            .join(self.downloaded_page_metadata.set_index('id'), how='left')
            .reset_index()
            .fillna({'downloaded': False})
            .pipe(XCDF)
        )

    @property
    @cache(version=3, key=lambda self: self)
    def _metadata(self) -> "pd.DataFrame['id': int, ...]":
        """Load all saved metadata from fs, keeping the latest observed metadata record per XC id"""
        metadata_paths_best_last = sorted(glob_filenames_ensure_parent_dir(f'{self.metadata_dir}/*.pkl'))
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
            pd.DataFrame(columns=['id'], data=XCResource.downloaded_ids(self._audio_paths))
            .assign(downloaded=True)
        )

    @property
    @lru_cache()
    def _audio_paths(self) -> Iterable[Path]:
        return audio_path_files.read('xc')

    @property
    def _audio_paths_hash(self) -> str:
        return sha1hex(json.dumps([str(x) for x in self._audio_paths]))  # Way faster than joblib.dump

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

    # HACK Ad-hoc function for constants.*_com_names_*
    #   - TODO Clean up along with ebird.com_names_to_species
    def com_names_to_species(self, com_names: Iterable[str], check=True) -> Iterable[str]:
        com_names = set(com_names)
        res = self.metadata[['com_name', 'species']][lambda df: df.com_name.isin(com_names)]
        unmatched = list(set(com_names) - set(res.com_name))
        if check and unmatched:
            raise ValueError('Unmatched com_names: %s' % unmatched)
        return res.species.sort_values().tolist()

    # TODO Clean up along with {xc,ebird}.com_names_to_species
    @property
    @lru_cache()
    def com_name_to_species_dict(self) -> dict:
        return (xc.metadata
            [['com_name', 'species']]
            .drop_duplicates()  # TODO Assert no duplicate keys flow into the dict
            .pipe(lambda df: {
                row.com_name: row.species
                for row in df_rows(df)
            })
        )

    def background_to_background_species(self, background: Iterable[str]) -> Iterable[str]:
        """For self.downloaded_page_metadata.background"""
        return [
            species if not pd.isnull(species) else com_name
            for name in background
            for com_name in [name.split(' (')[0]]  # "com_name (sci_name)" | "com_name" | free form
            for species in [self.com_name_to_species_dict.get(com_name)]
        ]

    @property
    def downloaded_page_metadata(self) -> "pd.DataFrame['id': int, ...]":
        return self.downloaded_page_metadata_full[[
            'id',
            # "Remarks from the Recordist"
            'remarks',
            'bird_seen',
            'playback_used',
            # "Recording data"
            'elevation',
            'background',
            # "Sound characteristics"
            'volume',
            'speed',
            'pitch',
            'length',
            'number_of_notes',
            'variable',
            # "Audio file properties"
            # 'length',  # FIXME Oops, during parsing this field gets overwritten by 'length' from "Sound characteristics"
            'channels',
            'sampling_rate',
            'bitrate_of_mp3',
        ]]

    @property
    @cache(version=0, key=lambda self: (self, self._audio_paths_hash))
    def downloaded_page_metadata_full(self) -> "pd.DataFrame['id': int, ...]":
        audio_paths = self._audio_paths
        page_paths = [p.parent / 'page.html' for p in audio_paths]
        return (
            pd.DataFrame(
                map_progress(self._parse_page_metadata, page_paths,
                    use='dask',
                    partition_size=100,
                    scheduler='processes',      # For 2000 recs: uncached[10s], cached[1.6s]
                    # scheduler='threads',      # For 2000 recs: uncached[22s], cached[2.7s]
                    # scheduler='synchronous',  # For 2000 recs: uncached[30s], cached[2.4s]
                ),
            )
            [lambda df: sorted(df.columns)]
            .pipe(df_reorder_cols, first=['id'])
        )

    @cache(version=1, key=lambda self, path: (self, path))
    def _parse_page_metadata(self, path: Path) -> dict:
        assert isinstance(path, Path)  # Avoid messing with str vs. Path to simplify cache key

        # For logging
        relpath = path.relative_to(xc.data_dir)

        # For integrity checks
        (_path_species, path_id, _path_filename) = relpath.parts
        path_id = int(path_id)

        # Read file
        with open(path) as f:
            html = f.read()

        # Prep output
        ret = dict()
        ret['id'] = path_id  # Ensure id is always populated, else nulls (which means nan's and float instead of int)
        ret['_raw'] = dict()

        # Noop if page is empty (else requests_html.HTML() barfs)
        if not html:
            log.warn(f'Skipping empty page.html[{relpath}]')
            return ret

        page = requests_html.HTML(url=path, html=html)

        # Parse: id, com_name, sci_name
        title = page.find('meta[property="og:title"]', first=True)
        if title:
            title = title.attrs.get('content')
        ret['_raw']['title'] = title
        if not title:
            page_id = None
            ret['com_name'] = None
            ret['sci_name'] = None
        else:
            (page_id, com_name, sci_name) = parse.parse('XC{} {} ({})', title).fixed
            page_id = int(page_id)
            ret['com_name'] = com_name
            ret['sci_name'] = sci_name

        if page_id != path_id:
            log.warn(f"Skipping malformed page.html[{relpath}]: page_id[{page_id}] != path_id[{path_id}]")
            return ret

        # Parse: remarks, bird_seen, playback_used
        #   - Ref: https://www.xeno-canto.org/upload/1/2
        #   - Examples:
        #       - '' [https://www.xeno-canto.org/420291]
        #       - '\n\nbird-seen:no\n\nplayback-used:no' [https://www.xeno-canto.org/413790]
        #       - 'About 20ft away in sagebrush steppe.\n\nbird-seen:yes\n\nplayback-used:no' [https://www.xeno-canto.org/418018]
        description = page.find('meta[property="og:description"]', first=True)
        if description:
            description = description.attrs.get('content')
        ret['_raw']['description'] = description
        if not description:
            ret['remarks'] = ''  # The branch below produces '' instead of None, so we also produce '' for consistency
            ret['bird_seen'] = None
            ret['playback_used'] = None
        else:
            lines = description.split('\n')
            keys = ['bird-seen', 'playback-used']
            for k in keys:
                ret[k.replace('-', '_')] = or_else(None, lambda: first(
                    parse.parse('%s:{}' % k, line)[0]
                    for line in lines
                    if line.startswith('%s:' % k)
                ))
            ret['remarks'] = '\n'.join(
                line
                for line in lines
                if not any(
                    line.startswith('%s:' % k)
                    for k in keys
                )
            ).strip()

        # Parse: all key-value pairs from #recording-data
        #   - (Thanks XC for structuring this so well!)
        recording_data = {
            k.lower().replace(' ', '_'): v
            for tr in page.find('#recording-data .key-value tr')
            for [k, v, *ignore] in [[td.text for td in tr.find('td')]]
        }
        ret['_raw']['recording_data'] = recording_data
        ret.update(recording_data)

        # Clean up fields
        #   - background is worth attempting here
        ret['background'] = [
            x
            for x in ret['background'].split('\n')
            for x in [x.strip()]
            if x != 'none'
        ]
        #   - But don't touch the rest of these so we don't mess anything up (they'd be easy for a motivated user)
        # ret['latitude'] = or_else(None, lambda: float(ret['latitude']))
        # ret['longitude'] = or_else(None, lambda: float(ret['longitude']))
        # ret['elevation'] = or_else(None, lambda: parse.parse('{:g} m', ret['elevation'])[0])
        # ret['sampling_rate'] = or_else(None, lambda: parse.parse('{:g} (Hz)', ret['sampling_rate'])[0])
        # ret['bitrate_of_mp3'] = or_else(None, lambda: parse.parse('{:g} (bps)', ret['bitrate_of_mp3'])[0])
        # ret['channels'] = or_else(None, lambda: parse.parse('{:g} (bps)', ret['channels'])[0])

        return ret


# Workaround for @singleton (above)
xc = _xc()


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

    @property
    def dir(self) -> Path:
        return xc.data_dir / self.species / str(self.id)

    @property
    def html_path(self) -> Path:
        return self.dir / 'page.html'

    @property
    def audio_path(self) -> Path:
        return self.dir / 'audio.mp3'

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
        )

    @classmethod
    def downloaded_ids(cls, audio_paths: Iterator[Path]) -> Iterable[int]:
        """
        Map the set of downloaded audio_paths to the set of id's that are (fully) downloaded
        - Needed by xc, but we own this concern to encapsulate that "downloaded" means "page.html and audio.mp3 exist"
        """
        audio_paths_parts = (
            pd.DataFrame(
                (p.relative_to(xc.data_dir).parts for p in audio_paths),
                columns=['species', 'id', 'filename'],
            )
            .astype({'id': 'int'})
        )
        return (audio_paths_parts
            # Audio is the completion marker for all the files (it's downloaded last), so we only need to check for it
            [lambda df: df.filename == 'audio.mp3']
            .id
            .values
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
# HACK TODO Hastily factored out of notebooks; merge into xc above, after cleaning up
#


def load_xc_recs(
    projection: 'Projection',
    countries_k: str,
    com_names_k: str,
    recs_at_least: int,
    num_species: int,
    num_recs: int,
) -> DF:
    xc_meta, recs_stats = _load_xc_meta(countries_k, com_names_k, recs_at_least, num_species, num_recs)
    xc_raw_recs = _xc_meta_to_xc_raw_recs(projection.features.load, xc_meta)
    _inspect_xc_raw_recs(xc_raw_recs)
    xc_recs = _xc_raw_recs_to_xc_recs(projection, xc_raw_recs)
    _inspect_xc_recs(xc_recs)
    return xc_recs, recs_stats


def _load_xc_meta(
    countries_k: str,
    com_names_k: str,
    recs_at_least: int,
    num_species: int,
    num_recs: int,
) -> DF:
    log.info('[1/3 fast] Filtering xc.metadata -> xc_meta...', **{
        '(countries_k, com_names_k)': (countries_k, com_names_k),
        '(recs_at_least, num_species, num_recs)': (recs_at_least, num_species, num_recs),
    })
    # Load xc_meta (fast)
    #   1. countries: Filter recs to these countries
    #   2. species: Filter recs to these species
    #   3. recs_at_least: Filter species to those with at least this many recs
    #   4. num_species: Sample this many of the species
    #   5. num_recs: Sample this many recs per species
    get_recs_stats = lambda df: dict(sp=df.species.nunique(), recs=len(df))
    puts_stats = lambda desc: partial(tap, f=lambda df: print('%-15s %12s (sp/recs)' % (desc, '%(sp)s/%(recs)s' % get_recs_stats(df))))
    xc_meta = (xc.metadata
        .pipe(puts_stats('all'))
        # 1. countries: Filter recs to these countries
        [lambda df: df.country.isin(constants.countries[countries_k])]
        .pipe(puts_stats('countries'))
        # 2. species: Filter recs to these species
        [lambda df: df.species.isin(com_names_to_species(*com_names[com_names_k]))]
        .pipe(puts_stats('species'))
        # Omit not-downloaded recs (should be few within the selected countries)
        [lambda df: df.downloaded]
        .pipe(puts_stats('(downloaded)'))
        # Remove empty cats for perf
        .pipe(df_remove_unused_categories)
        # 3. recs_at_least: Filter species to those with at least this many recs
        [lambda df: df.species.isin(df.species.value_counts()[lambda s: s >= recs_at_least].index)]
        .pipe(puts_stats('recs_at_least'))
        # 4. num_species: Sample this many of the species
        [lambda df: df.species.isin(df.species.drop_duplicates().pipe(lambda s: s.sample(n=min(len(s), num_species), random_state=0)))]
        .pipe(puts_stats('num_species'))
        # 5. num_recs: Sample this many recs per species
        #   - Remove empty cats else .groupby fails on empty groups
        .pipe(df_remove_unused_categories)
        .groupby('species').apply(lambda g: g.sample(n=min(len(g), num_recs), random_state=0))
        .pipe(puts_stats('num_recs'))
        # Drop species with <2 recs, else StratifiedShuffleSplit complains (e.g. 'TUVU')
        [lambda df: df.species.isin(df.species.value_counts()[lambda s: s >= 2].index)]
        .pipe(puts_stats('recs ≥ 2'))
        # Clean up for downstream
        .pipe(df_remove_unused_categories)
    )
    _recs_stats = get_recs_stats(xc_meta)
    recs_stats = ', '.join(['%s[%s]' % (k, v) for k, v in _recs_stats.items()])
    return (xc_meta, recs_stats)


def _xc_meta_to_xc_raw_recs(
    load: 'Load',
    xc_meta: DF,
    xc_paths_dump_path='/tmp/xc_paths',  # When uncached, helpful to run load.recs in a terminal (long running and verbose)
) -> DF:
    log.info('[2/3 slower] Loading xc_meta -> xc_raw_recs (.audio, more metadata)...')
    xc_paths = [
        ('xc', f'{data_dir}/xc/data/{row.species}/{row.id}/audio.mp3')
        for row in df_rows(xc_meta)
    ]
    if xc_paths_dump_path:
        joblib.dump(xc_paths, xc_paths_dump_path)
    xc_raw_recs = (
        load.recs(paths=xc_paths)
        .assign(
            # TODO Push upstream
            xc_id=lambda df: df.id.str.split('/').str[3].astype(int),
        )
        .set_index('xc_id')
        .join(how='left', other=(xc_meta
            .set_index('id')
            .drop(columns=['species', 'sci_name', 'com_name'])
            # TODO Push upstream
            .assign(
                state_only=lambda df: df.locality.str.split(', ').str[-1],
                place_only=lambda df: df.locality.str.split(', ').str[:-1].str.join(', '),
                state=lambda df: df.state_only.astype(str) + ', ' + df.country.astype(str),
                place=lambda df: df.place_only.astype(str) + ', ' + df.state.astype(str),
                year=lambda df: df.date.dt.year,
                month=lambda df: df.date.dt.month,
                month_day=lambda df: df.date.dt.strftime('%m-%d'),
                hour=lambda df: (df.time
                    .str.split(':').str[0].str.slice(0, 2)
                    .pipe(pd.to_numeric, errors='coerce')  # Invalid -> nan
                    .map(lambda x: x if 0 <= x < 24 else None)
                    # TODO Handle 'pm' (actually, just look at the .value_counts() and rewrite this as a testable function)
                ),
                background_species=lambda df: df.background.map(xc.background_to_background_species),
                n_background_species=lambda df: df.background_species.str.len(),
            )
            # Maybe push upstream?
            .assign(
                place_only_stack=lambda df: df.place_only.str.split(', ').map(df_cell_stack),
                state_only_stack=lambda df: df.state_only.str.split(', ').map(df_cell_stack),
                place_stack=lambda df: df.place.str.split(', ').map(df_cell_stack),
                state_stack=lambda df: df.state.str.split(', ').map(df_cell_stack),
                background_species_stack=lambda df: df.background_species.map(df_cell_stack),
                remarks_stack=df_cell_textwrap('remarks', 40),  # Make this one easy for the user to redo, to change width
            )
        ))
    )
    return xc_raw_recs


def _inspect_xc_raw_recs(xc_raw_recs: DF) -> DF:

    # TODO Useful enough to keep this in here? Or defer to notebook? -- which means it wouldn't display until after the slow stuff...
    log.info('Inspect xc_raw_recs')
    display(
        df_summary(xc_raw_recs).T,
        df_value_counts(xc_raw_recs, limit=30, dropna=False, exprs=[
            'species',
            'subspecies',
            'country',
            'state',
            ('quality', dict(sort_values=True)),
            'type',
            ('(duration_s//30)*30', dict(sort_values=True)),
            'recordist',
            ('year', dict(sort_values=True, ascending=False)),
            ('month', dict(sort_values=True)),
            'hour',
            'place',
            'n_background_species',
            'bird_seen',
            'playback_used',
            'elevation',
            'volume',
            'speed',
            'pitch',
            'length',
            'number_of_notes',
            'variable',
            'channels',
            'sampling_rate',
            'bitrate_of_mp3',
        ]),
        (xc_raw_recs
            .sample(n=min(10, len(xc_raw_recs)), random_state=0)
            .sort_values('species')
            [lambda df: [c for c in df.columns if not c.endswith('_stack')]]  # Save vertical space by cutting repetition
        ),
    )

    # Cheap plot: species counts
    log.info('Inspect xc_raw_recs: species counts (cheap plot)')
    display(xc_raw_recs
        .species_longhand.value_counts().sort_index()
        .reset_index().rename(columns={'index': 'species_longhand', 'species_longhand': 'num_recs'})
        .assign(num_recs=lambda df: df.num_recs.map(lambda n: '%s /%s' % ('•' * int(n / df.num_recs.max() * 60), df.num_recs.max())))
    )


def _xc_raw_recs_to_xc_recs(
    projection: 'Projection',
    xc_raw_recs: DF,
) -> DF:
    log.info('[3/3 slowest] Featurizing xc_raw_recs -> xc_recs (.audio, .feat, .spectro)...')
    # Featurize: .audio, .feat, .spectro (slowest)
    #   - NOTE .spectro is heavy: 3.1gb for 2167 dan4 recs
    xc_recs = (xc_raw_recs
        # .audio
        .assign(audio=lambda df: projection.features.load.audio(df, scheduler='threads'))
        # .feat
        .pipe(projection.transform)
        # .spectro
        .pipe(_recs_add_spectro, projection.features, cache=True)
        .pipe(df_reorder_cols, last=['audio', 'feat', 'spectro'])
    )
    assert {'audio', 'feat', 'spectro'} <= set(xc_recs.columns)
    return xc_recs


def _recs_add_spectro(recs, features, **kwargs) -> 'recs':
    """Featurize: .spectro (slow)"""
    # Cache control is knotty here: _spectro @cache is disabled to avoid disk blow up on xc, but we'd benefit from it for recordings
    #   - But the structure of the code makes it very tricky to enable @cache just for _spectro from one caller and not the other
    #   - And the app won't have the benefit of caching anyway, so maybe punt and ignore?
    return (recs
        .assign(spectro=lambda df: features.spectro(df, scheduler='threads', **kwargs))  # threads >> sync, procs
    )


def _inspect_xc_recs(xc_recs: DF):
    log.info('Inspect xc_recs')
    display(df_value_counts(xc_recs, limit=25, dropna=False, exprs=[
        'species',
        'subspecies',
        'country',
        'state',
        ('quality', dict(sort_values=True)),
        'type',
        ('(duration_s//30)*30', dict(sort_values=True)),
        ('year', dict(sort_values=True, ascending=False)),
        ('month', dict(sort_values=True)),
        ('hour', dict(sort_values=True)),
        'n_background_species',
    ]))


# XC cols that are interesting for EDA (for easy projection)
xc_eda_cols = [
    'species',
    'subspecies',
    'quality',
    'duration_s',
    'type',
    'state',
    'lat',
    'lng',
    'year',
    'month_day',
    'hour',
    'time',
    'license_type',
    'recordist',
    'elevation',
    'bird_seen',
    'playback_used',
    'background_species',
    'remarks',
]
xc_eda_cols_stack = [
    {
        'state': 'state_stack',
        'background_species': 'background_species_stack',
        'remarks': 'remarks_stack',
    }.get(x, x)
    for x in xc_eda_cols
]


def xc_to_handtype_fwf_df(df: pd.DataFrame, cols=['xc_id', 'handtype']) -> pd.DataFrame:
    """For hand-labeling song types (motivated by notebooks/app_ideas_5)"""
    if df.index.name is not None:
        df = df.reset_index()
    if 'handtype' not in df.columns:
        df = df.assign(handtype='')
    return (df
        [cols]
        .reset_index().rename(columns={'index': 'i'})
        .pipe(df_to_fwf_df)
    )


def load_xc_handtype(relpath_glob: str = '*.fwf', **kwargs) -> pd.DataFrame:
    """For hand-labeling song types (motivated by notebooks/app_ideas_5)"""
    df = (
        pd.concat([
            pd_read_fwf(
                path,
                widths='infer',
                na_filter=False,  # Map missing values to '' instead of np.nan
            )
            for path in glob.glob(f'{hand_labels_dir}/xc/{relpath_glob}')
        ])
        [['xc_id', 'handtype']]  # Drop non-hand-label cols, which we only included for context to the human editing the file
        .drop_duplicates()  # Tolerate non-conflicting dupes in the input files
        .set_index('xc_id')  # For easy join
        .pipe(df_col_map, handtype=lambda s: s.split(','))  # .handtype is multi-valued
        # .pipe(df_col_color_d, handtype=mpl_cmap_concat('Set1', 'Set2'))  [XXX Blood everywhere, defer to downstream]
    )
    # Ensure no conflicting dupes in the input files
    assert len(df) == df.reset_index().xc_id.nunique(), "Oops, non-unique labels; please resolve"
    return df
