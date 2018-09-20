from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime
import inspect
import json
import logging
import logging.config
import os
from pathlib import Path
import sys

import crayons
from potoo.util import context_onexit, generator_to
import structlog
import oyaml as yaml  # Drop-in replacement that preserves dict ordering (for BuboRenderer)

from config import config

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
            structlog.stdlib.PositionalArgumentsFormatter(),  # Support for e.g. log.info('msg: %s %s', x, y)
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # structlog.processors.JSONRenderer(serializer=json.dumps),  # We use BuboRenderer instead (for now)
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
        return json.loads(json.dumps(yaml.safe_load(f.read())))  # json to sanitize whatever wacky stuff yaml gives us


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
            # *['%s:%s' % (k, json.dumps(v)) for k, v in event_dict.items()]
            # json.dumps(event_dict)
            yaml.safe_dump(json.loads(json.dumps(event_dict)), default_flow_style=True, width=1e9).rstrip()
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

        # HACK Find the caller's pathname/lineno/funcName (mimic logging.Logger.{_log,findCaller,makeRecord})
        #   - The builtin logging library supports `funcName` in LogRecord format strings, but it stops working when you
        #     use structlog _and_ set a log level via logging config (e.g. fileConfig('logging.yaml')), because in that
        #     case funcName always reports `_proxy_to_logger` (from structlog.stdlib) instead of the actual caller
        #   - Here we redo the effort and explicitly filter out any callers under the 'logging' and 'structlog' modules
        #   - (Fingers crossed that this doesn't introduce _other_ bugs...)
        for i, caller in enumerate(inspect.stack(context=0)):
            # Skip our own frame
            if i == 0:
                continue
            # Skip frames from logging and structlog
            module = inspect.getmodule(caller.frame)
            if module is not None and module.__name__.split('.')[0] in ['logging', 'structlog']:
                continue
            break

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


def color(color: str, x: any, bold=True) -> str:
    s = str(x)
    if sys.stdout.isatty() and color is not None:
        s = getattr(crayons, color)(s, bold=bold)
    return s
