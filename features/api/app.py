from collections import OrderedDict
import json
import logging.config
import os.path
import subprocess

from flask import Flask
from potoo.ipython import ipy_install_mock_get_ipython
import structlog
import yaml

import api.routes
from api.server_globals import sg
from cache import memory
from config import config

log = structlog.get_logger(__name__)


def create_app(
    # config_yaml = os.path.join(os.path.dirname(__file__), '..', os.environ.get('CONFIG_YAML', 'config-dev.yaml')),
):

    check_deps()
    init_logging()
    init_potoo()
    memory.log.level = 'debug'  # Verbose cache logging for api (but keep quiet for notebooks)

    app = Flask(__name__)

    # Config [TODO -> config-*.yaml]
    # app.config['EXPLAIN_TEMPLATE_LOADING'] = True  # Debug

    # log.info('load_config', config_yaml=config_yaml)
    # with open(config_yaml) as f:
    #     app.config.from_mapping(yaml.load(f).get('flask', {}))

    app.register_blueprint(api.routes.bp)
    sg.init(app)
    ipy_install_mock_get_ipython()

    return app


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


def init_logging(
    logging_conf = os.path.join(os.path.dirname(__file__), '../logging.conf'),
):

    # http://structlog.readthedocs.io/en/stable/standard-library.html
    structlog.configure(
        processors = [
            structlog.stdlib.filter_by_level,
            # structlog.stdlib.add_logger_name,             # In logging.conf
            # structlog.stdlib.add_log_level,               # In logging.conf
            # structlog.processors.TimeStamper(fmt="iso"),  # In logging.conf
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(serializer=json.dumps),
        ],
        context_class             = structlog.threadlocal.wrap_dict(OrderedDict),
        logger_factory            = structlog.stdlib.LoggerFactory(),
        wrapper_class             = structlog.stdlib.BoundLogger,
        cache_logger_on_first_use = True,
    )

    logging.config.fileConfig(logging_conf)
    log.info('init_logging', logging_conf=logging_conf)  # (Can't log before logging is configured)


def check_deps():
    """Raise if any deps aren't found"""
    shell = lambda cmd: subprocess.check_call(cmd, shell=True)
    shell('ffmpeg -version >/dev/null 2>&1')  # Is ffmpeg installed?
    shell('ffmpeg -version 2>&1 | grep libmp3lame >/dev/null')  # Was ffmpeg built with libmp3lame?
