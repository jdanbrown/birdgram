from typing import List, Tuple, Union

import audiosegment
from dataclasses import dataclass
import numpy as np


@dataclass
class Recording:
    dataset: str
    species: str
    species_longhand: str
    species_com_name: str
    species_query: str
    basename: str
    name: str
    duration_s: float
    samples_mb: float
    samples_n: int
    audio: audiosegment.AudioSegment


RecOrAudioOrSignal = Union[
    Recording,  # rec as Recording/attrs
    dict,  # rec as dict
    audiosegment.AudioSegment,  # audio
    Tuple[np.array, int],  # (x, sample_rate)
    np.array,  # x where sample_rate=standard_sample_rate_hz
]


@dataclass
class Species:
    sci_name: str
    com_name: str
    taxon_id: str
    species_code: str
    taxon_order: str
    com_name_codes: List[str]
    sci_name_codes: List[str]
    banding_codes: List[str]
    shorthand: str
    longhand: str
