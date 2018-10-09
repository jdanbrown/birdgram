from collections import OrderedDict
import logging.config
import os.path

from flask import Flask
import potoo.pandas
import simplejson
import structlog
import yaml

import api.routes

log = structlog.get_logger(__name__)


def new_app(
    config_yaml = os.path.join(os.path.dirname(__file__), '..', os.environ.get('CONFIG_YAML', 'config-dev.yaml')),
):

    init_logging()
    potoo.pandas.set_display()

    app = Flask(__name__)

    log.info('load_config', config_yaml=config_yaml)
    with open(config_yaml) as f:
        app.config.from_mapping(yaml.load(f).get('flask', {}))

    app.register_blueprint(api.routes.bp)

    return app


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
            structlog.processors.JSONRenderer(serializer=simplejson.dumps),
        ],
        context_class             = structlog.threadlocal.wrap_dict(OrderedDict),
        logger_factory            = structlog.stdlib.LoggerFactory(),
        wrapper_class             = structlog.stdlib.BoundLogger,
        cache_logger_on_first_use = True,
    )

    logging.config.fileConfig(logging_conf)
    log.info('init_logging', logging_conf=logging_conf)  # (Can't log before logging is configured)
