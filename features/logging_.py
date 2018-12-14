from collections import OrderedDict
from datetime import datetime
from functools import partial
import inspect
import json
import logging
import logging.config
import os
from pathlib import Path
import sys
from typing import *

import crayons
from potoo import debug_print
from potoo.util import context_onexit, format_duration, generator_to, timer_start
import structlog
import oyaml as yaml  # Drop-in replacement that preserves dict ordering (for BuboRenderer)

from config import config
from json_ import json_dumps_safe, json_sanitize

log = structlog.get_logger(__name__)

logging_yaml_path = Path(__file__).parent / 'logging.yaml'


def init_logging(logging_yaml=None):

    # Docs
    #   - https://structlog.readthedocs.io/en/stable/getting-started.html
    #   - http://structlog.readthedocs.io/en/stable/standard-library.html
    logging.config.dictConfig(load_logging_dict(logging_yaml))
    structlog.configure(
        logger_factory            = structlog.stdlib.LoggerFactory(),
        wrapper_class             = structlog.stdlib.BoundLogger,
        context_class             = structlog.threadlocal.wrap_dict(OrderedDict),
        cache_logger_on_first_use = True,
        processors                = [
            # structlog.stdlib.filter_by_level,             # Done in logging.yaml
            # structlog.stdlib.add_logger_name,             # Done in logging.yaml
            # structlog.stdlib.add_log_level,               # Done in logging.yaml
            # structlog.processors.TimeStamper(fmt="iso"),  # Done in logging.yaml
            structlog.stdlib.PositionalArgumentsFormatter(),  # Support e.g. log.info('msg: %s %s', x, y)
            structlog.processors.StackInfoRenderer(),       # If stack_info=True, render caller's stack as 'stack' field (structured)
            # structlog.processors.format_exc_info,         # If exc_info=(True|e), render as 'exception' field (structured)
            structlog.processors.ExceptionPrettyPrinter(),  # If exc_info=(True|e), print traceback to stdout (friendly)
            structlog.processors.UnicodeDecoder(),
            # structlog.processors.JSONRenderer(serializer=json_dumps_safe),  # We use BuboRenderer instead (for now)
            BuboRenderer(),
        ],
    )

    # Log that logging is configured
    #   - WARNING If you log before logging is configured you'll get a default config that you can't undo
    log.info(logging_yaml=str(logging_yaml))


def log_levels(levels: dict):
    """Set levels when called, and reset if called as a context manager"""

    @generator_to(dict)
    def set_levels(levels: dict) -> dict:
        for name, level in levels.items():
            logger = logging.getLogger(name)
            yield (name, logger.level)
            logger.setLevel(level)

    # Set levels when called
    orig_levels = set_levels(levels)
    # Reset levels if called as a context manager
    return context_onexit(set_levels, orig_levels)


def load_logging_dict(logging_yaml=None) -> dict:
    """Shared between here + gunicorn_config.py (via bin/api-run-prod)"""
    logging_yaml = (
        logging_yaml or
        os.environ.get('LOGGING_YAML') or
        logging_yaml_path
    )
    with open(logging_yaml) as f:
        return json_sanitize(yaml.safe_load(f.read()))  # json to sanitize whatever wacky stuff yaml gives us


class BuboRenderer:
    """
    A basic alternative to structlog.processors.JSONRenderer/KeyValueRenderer
    - Good for human readability
    - Less good for structured processing
    """

    # TODO Decouple from logging.yaml [by writing a BuboFormatter]
    def __call__(self, logger, name, event_dict):
        event = event_dict.pop('event', None)
        data = None if not event_dict else (
            # *['%s=%s' % (k, v) for k, v in event_dict.items()]
            # *['%s:%s' % (k, json_dumps_safe(v)) for k, v in event_dict.items()]
            # json_dumps_safe(event_dict)
            yaml.safe_dump(json_sanitize(event_dict), default_flow_style=True, width=1e9).rstrip()
        )
        msg = ' '.join(x for x in [event, data] if x is not None)
        return msg


class BuboFilter(logging.Filter):

    def filter(self, record):
        """
        Format record with conditional stuff and color, in the spirit of:
            format: '%(levelname)-8s [%(asctime)s.%(msecs)03d] [%(process)5d]%(lineno)4d %(name)s/%(funcName)s: %(msg)s'
            datefmt: '%Y-%m-%dT%H:%M:%S'
        """

        # Get caller
        (caller, module) = get_caller()  # (See detailed comments in get_caller)
        record.pathname = caller.frame.f_code.co_filename
        record.lineno = caller.frame.f_lineno
        record.funcName = caller.frame.f_code.co_name
        record.stack_info = None  # TODO Capture this (mimic logging.Logger.findCaller) [high complexity, low prio]

        # msg, args
        #   - HACK `msg % args` is done downstream by Formatter, but we do it in advance to compute name_funcName_message
        #   - https://docs.python.org/3/library/logging.html
        if record.args:
            record.msg = record.msg % record.args
            record.args = ()  # Unset args so that Formatter's operation is a noop

        # name_funcName_message
        #   - TODO Omit funcName from deps: add include/exclude args (in logging.yaml) to control when to include funcName
        #   - TODO Code org: write a BuboFormatter to own this concern (and also the logging.yaml format string)
        record.name_funcName_message = (
            '%(name)s/%(funcName)s: %(msg)s' if record.msg else
            '%(name)s/%(funcName)s'
        ) % dict(record.__dict__)

        # levelname
        record.levelname = color(
            x='%-8s' % record.levelname,  # Format before color since ansi chars mess up the width adjustment
            color={
                'DEBUG':    'blue',
                'INFO':     'green',
                'WARN':     'yellow',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'magenta',
            }.get(record.levelname),
        )

        # timestamp, asctime, datefmt
        #   - [Why is record.asctime absent here, even though it's present in logging.yaml format?]
        record.datefmt = config.logging.datefmt  # Switches on $BUBO_ROLE (in config.py)
        record.asctime = datetime.fromtimestamp(record.created).strftime(record.datefmt)
        record.timestamp = color('black', '[%s.%03d]' % (record.asctime, record.msecs))

        # process
        record.process = color('black', '[%5d]' % record.process)

        # lineno
        record.lineno = color('black', '%4d' % record.lineno)

        return True


def get_caller():
    # HACK Find the caller's frame (mimic logging.Logger.{_log,findCaller,makeRecord})
    #   - The builtin logging library supports `funcName` in LogRecord format strings, but it stops working when you
    #     use structlog _and_ set a log level via logging config (e.g. fileConfig('logging.yaml')), because in that
    #     case funcName always reports `_proxy_to_logger` (from structlog.stdlib) instead of the actual caller
    #   - Here we redo the effort and explicitly filter out any callers under the 'logging' and 'structlog' modules
    #   - (Fingers crossed that this doesn't introduce _other_ bugs...)
    for i, caller in enumerate(inspect.stack(context=0)):
        # Skip frames from logging modules (including our own frame)
        module = inspect.getmodule(caller.frame)
        if module is not None and module.__name__.split('.')[0] in [
            'logging',
            'structlog',
            # HACK HACK So that log_time* find the real caller's frame
            'logging_',    # bubo.logging_, for log_time* (which are defined in this module)
            'contextlib',  # For log_time_context
            'pandas',      # For `.pipe(log_time_df, ...)`
            'progress',    # For _map_progress_log_time_all
        ]:
            continue
        break
    return (caller, module)


def color(color: str, x: any, bold=True) -> str:
    s = str(x)
    if sys.stdout.isatty() and color is not None:
        s = getattr(crayons, color)(s, bold=bold)
    return s


#
# TODO Where should these log_time* utils live?
#   - HACK HACK Putting them here is our current approach to finding the caller's funcName in BuboFilter.filter
#

from contextlib import contextmanager
from functools import wraps
from typing import *

from potoo.util import timer_start


def log_time_df(df: 'pd.DataFrame', f: Callable[['pd.DataFrame', '...'], 'X'], *args, **kwargs) -> 'X':
    return log_time(f, df, *args, **kwargs)


def log_time(f: Callable[['...'], 'X'], *args, desc=None, log=None, **kwargs) -> 'X':
    desc = desc or f.__qualname__
    with log_time_context(desc=desc, log=log):
        return f(*args, **kwargs)


@contextmanager
def log_time_context(desc=None, report: Callable[[], any] = None, log=None):
    log = log or get_log_as_caller()
    timer = timer_start()
    log.debug('%s...' % (f'{desc} ' if desc else ''))
    try:
        yield
    finally:
        report = [
            *([str(report())] if report else []),
            '%.3fs' % timer.time(),
        ]
        log.info('%s[%s]' % (f'{desc} ' if desc else '', ', '.join(report)))


def get_log_as_caller():
    (_caller, module) = get_caller()
    return structlog.get_logger(module.__name__ if module else '[none]')


# TODO How to make this report the call site's lineno (and funcName and module/logger name)?
#   - Passing log=log fixes record.name, but not .lineno + .funcName
#   - Example stack: https://gist.github.com/jdanbrown/b9944d7a131fe7faa5585c6064cdacb8
# def log_time_deco(f_or_desc=None, f=None, desc=None, log=None):
#     if callable(f_or_desc):
#         f = f_or_desc
#     else:
#         desc = f_or_desc
#     def decorator(f):
#         @wraps(f)
#         def g(*args, **kwargs):
#             with log_time_context(desc=desc, log=log):
#                 return f(*args, **kwargs)
#         return g
#     return decorator(f) if f else decorator


# Dispatch from util.map_progress
#   - HACK HACK In module logging_ instead of util so it can find the caller's frame (see above)
def _map_progress_log_time_all(
    f: Callable[['X'], 'X'],
    xs: Iterable['X'],
    desc: str = None,
    n: int = None,
) -> Iterable['X']:
    """log_time around whole loop"""
    desc = (
        None      if desc is None and n is None else
        f'({n})'  if desc is None else
        f'{desc}' if n is None else
        f'{desc} ({n})'
    )
    with log_time_context(desc=desc):
        return list(map(f, xs))


# Dispatch from util.map_progress
#   - HACK HACK In module logging_ instead of util so it can find the caller's frame (see above)
def _map_progress_log_time_each(
    f: Callable[['X'], 'X'],
    xs: Iterable['X'],
    desc: str = None,
    n: int = None,
) -> Iterable['X']:
    """log_time around each element in loop"""
    def desc_i(i):
        return (
            None                  if desc is None and n is None else
            f'({i+1}/{n})'        if desc is None else
            f'{desc} ({i+1}/?)'   if n is None else
            f'{desc} ({i+1}/{n})'
        )
    if n is None:
        report = None
    else:
        timer = timer_start()
        def report(i):
            t = timer.time()
            t_tot = t / (i + 1) * n
            t_rem = t_tot - t
            format = lambda t: format_duration(int(t))
            return '-%s/%s' % (format(t_rem), format(t_tot))
    ys = []  # HACK HACK Instead of @generator_to, so we can find the caller's frame (see above)
    log.info(desc_i(-1))  # Log '(0/n)' at the beginning
    for i, x in enumerate(xs):
        with log_time_context(desc=desc_i(i), report=report and partial(report, i)):
            ys.append(f(x))
    return ys
