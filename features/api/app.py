import os
from pathlib import Path

from flask import Flask
from potoo.ipython import ipy_install_mock_get_ipython
import structlog

import api.routes
from api.server_globals import sg
from logging_ import init_logging
from cache import memory

log = structlog.get_logger(__name__)


def create_app(
    # config_yaml=Path(__file__).parent.parent / os.environ.get('CONFIG_YAML', 'config-dev.yaml'),
):

    init_check_deps()
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

    # HACK Make get_ipython() work, e.g. for potoo.ipython.ipy_formats, which we hijack for api responses 🙈
    ipy_install_mock_get_ipython()

    return app
