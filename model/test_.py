from contextlib import contextmanager
from pathlib import Path

from cache import cache_control
from constants import mobile_ios_tests_data_dir
from json_ import json_dump_path
from util import ensure_parent_dir


@contextmanager
def test_for_swift(name):
    """
    Generate test data for swift tests
    - Examples in mobile/ios/Tests/*.ipynb
    """
    # Disable cache for tests
    with cache_control(enabled=False):
        # HACK Dump/dump: control-flow shenanigans, because python [any simpler with coroutines?]
        class Dump(Exception):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        try:
            yield (name, Dump)
        except Dump as dump:
            json_dump_path(
                obj=dump.kwargs,
                path=ensure_parent_dir(str(Path(mobile_ios_tests_data_dir) / name) + '.json'),
            )
