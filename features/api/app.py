from collections import OrderedDict
import inspect
import json
import logging
import logging.config
import os
from pathlib import Path
import subprocess

import cloudpickle
from dataclasses import dataclass
from flask import Flask
from potoo.ipython import ipy_install_mock_get_ipython
import structlog
import oyaml as yaml  # Drop-in replacement that preserves dict ordering (for BuboRenderer)

import api.routes
from api.server_globals import sg
from cache import memory
from config import config

log = structlog.get_logger(__name__)


def create_app(
    # config_yaml=Path(__file__).parent.parent / os.environ.get('CONFIG_YAML', 'config-dev.yaml'),
):

    check_deps()
    init_cloudpickle()
    init_logging()
    init_potoo()
    memory.log.level = 'debug'  # Verbose cache logging for api (but keep quiet for notebooks)

    app = Flask(__name__)

    # Config [TODO -> config-*.yaml]
    # app.config['EXPLAIN_TEMPLATE_LOADING'] = True  # Debug
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for /static (for computed responses, see api.routes)
    app.config['PDB'] = os.environ.get('PDB')

    # log.info('load_config', config_yaml=config_yaml)
    # with open(config_yaml) as f:
    #     app.config.from_mapping(yaml.load(f).get('flask', {}))

    app.register_blueprint(api.routes.bp)
    sg.init(app)
    ipy_install_mock_get_ipython()

    return app


def check_deps():
    """Raise if any deps aren't found"""
    shell = lambda cmd: subprocess.check_call(cmd, shell=True)
    shell('ffmpeg -version >/dev/null 2>&1')  # Is ffmpeg installed?
    shell('ffmpeg -version 2>&1 | grep libmp3lame >/dev/null')  # Was ffmpeg built with libmp3lame?


def init_cloudpickle():
    """Add cloudpickle dispatches"""
    # Fix structlog loggers to be cloudpickle-able [like https://github.com/cloudpipe/cloudpickle/pull/96]
    #   - TODO Open a cloudpickle issue [just requires making a small repro to illustrate the problem]
    def save_structlog_BoundLoggerLazyProxy(self, obj):
        self.save_reduce(structlog.get_logger, obj._logger_factory_args, obj=obj)
    def save_structlog_BoundLoggerBase(self, obj):
        # TODO Add support for structlog logger.bind [how does it even work?]
        raise ValueError("TODO Add support for structlog logger.bind [how does it even work?]")
    cloudpickle.CloudPickler.dispatch[structlog._config.BoundLoggerLazyProxy] = save_structlog_BoundLoggerLazyProxy
    cloudpickle.CloudPickler.dispatch[structlog.BoundLoggerBase] = save_structlog_BoundLoggerBase


def load_logging_dict(logging_yaml=None) -> dict:
    """Shared between here + gunicorn_config.py (via bin/api-run-prod)"""
    logging_yaml = (
        logging_yaml or
        os.environ.get('LOGGING_YAML') or
        Path(__file__).parent.parent / 'logging.yaml'
    )
    with open(logging_yaml) as f:
        return json.loads(json.dumps(yaml.safe_load(f.read())))  # json to sanitize whatever wacky stuff yaml gives us


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

        # HACK Find the caller's funcName
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
        record.funcName = caller.function

        # HACK `msg % args` is done downstream by Formatter, but we do it in advance to compute name_funcName_message
        #   - https://docs.python.org/3/library/logging.html
        record.msg = record.msg % record.args
        record.args = ()  # Unset args so that Formatter's operation is a noop

        # TODO Omit funcName from deps: add include/exclude args (in logging.yaml) to control when to include funcName
        # TODO Code org: write a BuboFormatter to own this concern (and also the logging.yaml format string)
        record.name_funcName_message = (
            '%(name)s/%(funcName)s: %(msg)s' if record.msg else
            '%(name)s/%(funcName)s'
        ) % dict(record.__dict__)

        return True


def init_potoo():
    """cf. notebooks/__init__.py"""

    from potoo.python import ensure_python_bin_dir_in_path, install_sigusr_hooks
    # ensure_python_bin_dir_in_path()
    # install_sigusr_hooks()

    from potoo.pandas import set_display_on_sigwinch, set_display
    # set_display_on_sigwinch()
    set_display()

    from potoo.ipython import disable_special_control_backslash_handler, set_display_on_ipython_prompt, ipy_formats
    # disable_special_control_backslash_handler()
    # set_display_on_ipython_prompt()
    ipy_formats.set()

    from potoo.plot import plot_set_defaults
    plot_set_defaults()
