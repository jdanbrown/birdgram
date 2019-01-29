from dataclasses import dataclass
import pandas as pd

from constants import *
from datatypes import Recording, RecordingDF
from util import *


@singleton
class birdclef2015:

    # TODO Finish fleshing this out (20180530_dataset_birdclef.ipynb + 20180524_eval_birdclef.ipynb)

    def xml_data(self, recs: RecordingDF) -> pd.DataFrame:
        return pd.DataFrame([self.xml_dict_for_rec(rec) for rec in df_rows(recs)])

    def xml_dict_for_rec(self, rec: Recording) -> dict:

        # wav_path -> (<Audio>, 'train'|'test')
        wav_path = os.path.join(data_dir, rec.path)
        xml_path_prefix = wav_path.replace('/wav/', '/xml/').replace('.wav', '')
        train_path = glob_filenames_ensure_parent_dir(f'{xml_path_prefix}-train.xml')
        test_path = glob_filenames_ensure_parent_dir(f'{xml_path_prefix}-test.xml')
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
