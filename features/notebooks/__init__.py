from collections import *
import contextlib
from datetime import datetime, date, timedelta
import glob
import gzip
import importlib
import inspect
import json
import os
import os.path
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
from potoo.ipython import *
from potoo.pandas import *
from potoo.plot import *
from potoo.pretty import *
from potoo.pytest import *
from potoo.util import *
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

from cache import *
from constants import *
from datasets import *
from features import *
from load import *
from proc_stats import *
from sk_hack import *
from sk_util import *
from sp14.model import *
from util import *
from viz import *

##
# Mimic the relevant bits of ~/.pythonrc

if ipy:
    ipy.magic('load_ext autoreload')
    ipy.magic('autoreload 2')

if ipy:
    ipy.display_formatter.formatters['text/plain'].for_type_by_name(
        'matplotlib.axes._subplots', 'AxesSubplot', lambda x, p, cycle: (p.text(repr(x)), plt.show()),
    )

from potoo.ipython import *
# set_display_on_ipython_prompt()
ipy_formats.set()

from potoo.pandas import *
set_display()

from potoo.plot import *
plot_set_defaults()

if ipy:
    import potoo.default_magic_magic
    ipy.magic('load_ext potoo.default_magic_magic')

from potoo.ipython import *
gc_on_ipy_post_run_cell()

##
# TODO Move these up into potoo or ~/.pythonrc

if ipy:
    ipy.run_cell_magic('capture', '', '%xmode plain')
