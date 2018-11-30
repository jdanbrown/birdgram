from contextlib import contextmanager
from pathlib import Path

from constants import birdgram_ios_tests_data_dir
from json_ import json_dump_path
from util import ensure_parent_dir


@contextmanager
def test_for_swift(name):
    """
    Generate test data for swift tests
    - Examples in Birdgram/ios/Tests/*.ipynb
    """
    # HACK Control-flow shenanigans, because python [how to simplify with coroutines?]
    class Dump(Exception):
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    try:
        yield (name, Dump)
    except Dump as dump:
        json_dump_path(
            obj=dump.kwargs,
            path=ensure_parent_dir(str(Path(birdgram_ios_tests_data_dir) / name) + '.json'),
        )
