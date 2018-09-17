from dataclasses import dataclass
import numpy as np
from potoo.dataclasses import DataclassUtil
from potoo.util import singleton
import structlog

from attrdict import AttrDict
from cache import cache
from config import config
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
            log.info()
            self.__dict__.update(sg_load.load())
            log.info('done')
            self._init_done = True

    # WARNING Dask pickles funcs even when scheduler != 'processes'
    #   - Huge bottleneck if you reference large objects, e.g. sg.*
    #   - e.g. dask.bag.map -> bag_map -> dask.base.tokenize -> normalize_function -> _normalize_function -> pickle.dumps(func)
    #   - Looks like it does it to compute the task id, so it's probably essential behavior that's hard to change

    # To avoid this, we define custom pickling so that we aren't a ~250mb bottleneck for callers who try to pickle us
    #   - TODO Be robust to changes in sg_load: e.g. write out its params and reconstruct from those params
    #       - I've deferred this just because I recall the sg/sg_load singleton dance being tricky and subtle wrt. flask
    #         app initialization, and I don't remember the pitfalls that I was dancing around at the time. Buyer beware.
    def __getstate__(self):
        return {}
    def __setstate__(self, state):
        self.init()


# WARNING @singleton breaks cloudpickle in a very strange way because it "rebinds" the class name:
#   - See details in util.Log
@dataclass(frozen=True)  # Mutation isn't intended, and it isn't safe for client code that pickles/unpickles sg
class _sg_load(DataclassUtil):

    # Config:

    # search
    experiment_id                 : str = config.server_globals.sg_load.experiment_id
    cv_str                        : str = config.server_globals.sg_load.cv_str
    search_params_str             : str = config.server_globals.sg_load.search_params_str
    classifier_str                : str = config.server_globals.sg_load.classifier_str
    random_state                  : str = config.server_globals.sg_load.random_state
    fix_missing_skm_projection_id : str = config.server_globals.sg_load.fix_missing_skm_projection_id

    # xc_meta
    countries_k : str = config.server_globals.sg_load.countries_k
    com_names_k : str = config.server_globals.sg_load.com_names_k
    num_recs    : int = config.server_globals.sg_load.num_recs

    # load*: Maintain tight and well-defined input/output relationships so we can cache (without headaches)
    #   - e.g. any app config we depend on should be explicitly surfaced as function params

    def load(self):
        # Split load* by method so we can easily inspect data volume from @cache dir sizes
        #   - Also helps isolate cache refresh
        return dict(
            **self.load_search(),
            **self.load_xc_meta(),
        )

    @cache(version=2, key=lambda self: self)
    def load_search(self):
        log.info()
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

    @cache(version=3, key=lambda self: self)
    def load_xc_meta(self):
        log.info()
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
