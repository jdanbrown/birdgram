from datetime import datetime
import json
import logging
import sys
import threading
from typing import Optional, Union

import crayons
from dataclasses import dataclass
from potoo.util import AttrContext, singleton
import yaml


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#
#   @singleton
#   class foo: pass
#   cloudpickle.dump(foo)  # Fails with "can't pickle _thread._local objects"
#
#   class foo: pass
#   foo = foo()
#   cloudpickle.dump(foo)  # Fails with "can't pickle _thread._local objects"
#
#   class Foo: pass
#   foo = Foo()
#   cloudpickle.dump(foo)  # Ok! Use this as a workaround.
#
@dataclass
class Log(AttrContext):
    """Simple, ad-hoc logging specialized for interactive usage"""

    # TODO Throw this away and migrate to real logging! (e.g. structlog)
    #   - Features I hesitate to throw away:
    #       - Human-readable one-kwargs-per-line formatting -- should be able to replicate in structlog
    #       - log.char -- might have to give this up one...

    Level = int
    LevelLike = Union[str, Level]
    level: LevelLike = 'debug'

    _state: Optional[Union['line', 'char']] = None
    _lock = threading.RLock()  # Synchronize char vs. line printing (works with threads but not processes)

    # Back compat
    def __call__(self, *args, **kwargs):
        self.log(logging.DEBUG, *args, **kwargs)

    def debug (self, *args, **kwargs): self.log(logging.DEBUG, *args, **kwargs)
    def info  (self, *args, **kwargs): self.log(logging.INFO,  *args, **kwargs)
    def warn  (self, *args, **kwargs): self.log(logging.WARN,  *args, **kwargs)
    def error (self, *args, **kwargs): self.log(logging.ERROR, *args, **kwargs)

    def log(self, level: LevelLike, event: str, **kwargs):
        if self._to_level(level) >= self._to_level(self.level):
            with self._lock:
                self._to_state('line')
                t = datetime.utcnow().isoformat()
                t = t[:23]  # Trim micros, keep millis
                t = t.split('T')[-1]  # Trim date for now, since we're primarily interactive usage
                # Display timestamp + event on first line
                print('[%s] %-5s  %s' % (t, self._format_level(level), event))
                # Display each (k,v) pair on its own line, indented
                for k, v in kwargs.items():
                    v_yaml = yaml.safe_dump(json.loads(json.dumps(v)), default_flow_style=True, width=1e9)
                    v_yaml = v_yaml.split('\n')[0]  # Handle documents ([1] -> '[1]\n') and scalars (1 -> '1\n...\n')
                    print('  %s: %s' % (k, v_yaml))

    # HACK HACK HACK Motivated by the '•'/'!' logging in our hacky fork of joblib.memory.MemorizedFunc
    def char(self, level: LevelLike, char: str):
        if self._to_level(level) >= self._to_level(self.level):
            with self._lock:
                self._to_state('char')
                print(char, end='', flush=True)

    # WARNING This gets really janky and surprising, e.g.
    #   - Across processes, log instances have separate state
    #   - Across notebook cells, state='line' will unnecessarily print() to flush an arbitrarily far-away state='char'
    #   - Across notebook cells with multiprocessing, many procs can unnecessarily print() to flush a far-away char :/
    def _to_state(self, state: Union['line', 'char']):
        assert state is not None, 'Not allowed to transition back to None'
        if (self._state, state) == ('char', 'line'):
            print()
        self._state = state

    def _to_level(self, level: LevelLike) -> Level:
        if isinstance(level, self.Level):
            return level
        elif isinstance(level, str):
            return getattr(logging, level.upper())
        else:
            raise ValueError(f"Can't convert to Level: {level}")

    def _format_level(self, level: LevelLike, color=None, color_bold=True) -> str:
        color = color if color is not None else sys.stdout.isatty()
        (name, _color) = {
            logging.DEBUG: ('DEBUG', 'blue'),
            logging.INFO:  ('INFO',  'green'),
            logging.WARN:  ('WARN',  'yellow'),
            logging.ERROR: ('ERROR', 'red'),
        }[self._to_level(level)]
        if color:
            name = getattr(crayons, _color)(name, bold=color_bold)
        return name


# Workaround for @singleton (above)
log = Log()
