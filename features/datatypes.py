from typing import List, Tuple, Union

import audiosegment
from dataclasses import dataclass
import numpy as np


@dataclass
class Recording:
    dataset: str
    species: str
    species_query: str
    basename: str
    name: str
    duration_s: float
    samples_mb: float
    samples_n: int
    samples: int
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
    banding_codes: List[str]
    com_name: str
    com_name_codes: List[str]
    sci_name: str
    sci_name_codes: List[str]
    shorthand: str
    species_code: str
    taxon_id: str
    taxon_order: any  # TODO Is this a float or a str?
