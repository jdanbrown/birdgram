# Before `import config`
import os; os.environ['BUBO_ROLE'] = 'notebook'

import calendar
from collections import *
import contextlib
import copy
from datetime import datetime, date, timedelta
import fnmatch
import glob
import gzip
import importlib
import inspect
import json
import math
import os
import os.path
import parse
from pathlib import Path
import pdb
import re
import shlex
import string
import sys
import subprocess
import tempfile
import time
from typing import *

from attrdict import AttrDict
import dask
import dask.array as da
import dask.dataframe as dd
# import dateparser  # Slow and currently unused
from IPython.display import display
from itertools import *
import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
from more_itertools import *
import numpy as np
import pandas as pd
import PIL
import potoo
from potoo.debug_print import *
from potoo.ipython import *
from potoo.numpy import *
from potoo.pandas import *
from potoo.plot import *
from potoo.pretty import *
from potoo.pytest import *
from potoo.util import *
import pprint
import requests
import rpy2
import rpy2.robjects as robjects
import scipy
import scipy.stats as stats
import sklearn as sk
import statsmodels.api as sm
import statsmodels.formula.api as smf
import toolz
from tqdm import tqdm
import yaml
import xgboost as xgb

from api.recs import *
from api.server_globals import *
from cache import *
from config import config
from constants import *
from datasets import *
from ebird_priors import *
from features import *
import geoh
from load import *
import metadata
from notebooks.app_ux import *
from proc_stats import *
from sk_hack import *
from sk_util import *
from sp14.model import *
from util import *
from viz import *
from xgb_sklearn_hack import *
from xgb_util import *

#
# Mimic the relevant bits of ~/.pythonrc
#   - cf. api.app.init_potoo
#

if ipy:
    ipy.magic('load_ext autoreload')
    ipy.magic('autoreload 2')

if ipy:
    ipy.display_formatter.formatters['text/plain'].for_type_by_name(
        'matplotlib.axes._subplots', 'AxesSubplot', lambda x, p, cycle: (p.text(repr(x)), plt.show()),
    )

from potoo.python import *
ensure_python_bin_dir_in_path()
# install_sigusr_hooks()

from potoo.pandas import *
# set_display_on_sigwinch()
set_display()

from potoo.ipython import *
# disable_special_control_backslash_handler()
# set_display_on_ipython_prompt()
ipy_formats.set()

from potoo.plot import *
plot_set_defaults()

if ipy:
    import potoo.default_magic_magic
    ipy.magic('load_ext potoo.default_magic_magic')

from potoo.ipython import *
gc_on_ipy_post_run_cell()

#
# Mimic api.app.create_app
#

from api.app import *
check_deps()
# init_logging()  # TODO After migrating log.log to real logging

#
# TODO Move these up into potoo or ~/.pythonrc
#

if ipy:
    ipy.run_cell_magic('capture', '', '%xmode plain')
