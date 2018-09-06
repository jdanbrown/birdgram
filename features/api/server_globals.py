from dataclasses import dataclass
import numpy as np
from potoo.dataclasses import DataclassUtil
from potoo.util import singleton
import structlog

from attrdict import AttrDict
from cache import cache
from datasets import load_xc_meta
from sp14.model import Search

log = structlog.get_logger(__name__)


# "Server globals", abbreviated `sg`
#   - Purpose: read-only state that's initialized once during app creation, and lives until the process exits
#   - Different than "application globals" (http://flask.pocoo.org/docs/1.0/api/#application-globals)
#   - Different than "application context" (http://flask.pocoo.org/docs/1.0/appcontext/)
#   - Different than "request context" (http://flask.pocoo.org/docs/1.0/reqcontext/)
#   - Different than "session" (http://flask.pocoo.org/docs/1.0/api/#sessions)
#
# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
class _sg:

    def init(self, app):
        if not hasattr(self, '_init_done'):
            log.info('init')
            self.__dict__.update(sg_load.load())
            log.info('init:done')
            self._init_done = True


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
@dataclass
class _sg_load(DataclassUtil):

    # Config:

    # search
    experiment_id = 'comp-l1-l2-na-ca'
    cv_str = 'split_i=0,train=34875,test=331,classes=331'
    search_params_str = 'n_species=331,n_recs=1.0'
    classifier_str = 'cls=ovr-logreg_ovr,solver=liblinear,C=0.001,class_weight=balanced'
    random_state = 0
    fix_missing_skm_projection_id = 'peterson-v0-26bae1c'

    # xc_meta
    (countries_k, com_names_k, num_recs) = (None, None,   None)  # All xc.metadata
    # (countries_k, com_names_k, num_recs) = (None, None,   10)    # XXX Faster dev
    # (countries_k, com_names_k, num_recs) = (None, 'dan5', None)  # XXX Faster dev
    # (countries_k, com_names_k, num_recs) = (None, 'dan5', 10)    # XXX Faster dev

    # load*: Maintain tight and well-defined input/output relationships so we can cache (without headaches)
    #   - e.g. any app config we depend on should be explicitly surfaced as function params

    @classmethod
    def load(self):
        # Split load* by method so we can easily inspect data volume from @cache dir sizes
        #   - Also helps isolate cache refresh
        return dict(
            **self.load_search(),
            **self.load_xc_meta(),
        )

    @classmethod
    @cache(version=0)
    def load_search(self):
        log.info('load:search')
        x = AttrDict()
        x.search = Search.load_v0(
            experiment_id=self.experiment_id,
            cv_str=self.cv_str,
            search_params_str=self.search_params_str,
            classifier_str=self.classifier_str,
            random_state=self.random_state,
            fix_missing_skm_projection_id=self.fix_missing_skm_projection_id
        )
        x.projection = x.search.projection
        x.features = x.projection.features
        x.load = x.features.load
        return dict(x)

    @classmethod
    @cache(version=0)
    def load_xc_meta(self):
        log.info('load:xc_meta')
        x = AttrDict()
        x.xc_meta, _recs_stats = load_xc_meta(
            countries_k=self.countries_k,
            com_names_k=self.com_names_k,
            recs_at_least=None,
            num_species=None,
            num_recs=self.num_recs,
            drop_recs_lt_2=False,
        )
        return dict(x)


# Workaround for @singleton (above)
sg_load = _sg_load()
sg = _sg()
