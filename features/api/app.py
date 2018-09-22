import os
from pathlib import Path
import textwrap

from flask import Flask
import json
from potoo.ipython import ipy_install_mock_get_ipython
import structlog
import yaml

import api.routes
from api.server_globals import sg
from cache import memory
from config import config
from inits import *
from logging_ import init_logging
from util import *

log = structlog.get_logger(__name__)


def create_app(
    # config_yaml=Path(__file__).parent.parent / os.environ.get('CONFIG_YAML', 'config-dev.yaml'),
):

    init_logging()
    init_check_deps()
    init_cloudpickle()
    init_potoo()

    app = Flask(__name__)

    # Config [TODO -> config-*.yaml]
    # app.config['EXPLAIN_TEMPLATE_LOADING'] = True  # Debug
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for /static (for computed responses, see api.routes)
    app.config['BUBO_PDB'] = os.environ.get('BUBO_PDB')

    # log.info('load_config', config_yaml=config_yaml)
    # with open(config_yaml) as f:
    #     app.config.from_mapping(yaml.load(f).get('flask', {}))

    # Log config (great for api logs, too noisy for notebook)
    #   - Pretty print so we can actually read it, violating the one-line-per-log-event principle
    log.info('Config:')
    print(textwrap.indent(prefix='  ', text=(  # Indent under the log line
        yaml.safe_dump(default_flow_style=False, width=1e9,  # Yaml for high SNR
            data=json.loads(json_dumps_safe(config)),  # Json cleanse to strip nonstandard data structures for yaml
        ).rstrip('\n')
    )))

    app.register_blueprint(api.routes.bp)
    sg.init(app)

    # HACK Make get_ipython() work, e.g. for potoo.ipython.ipy_formats, which we hijack for api responses 🙈
    ipy_install_mock_get_ipython()

    return app
