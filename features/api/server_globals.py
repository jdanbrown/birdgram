from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np
import pandas as pd
from potoo import debug_print
from potoo.dataclasses import DataclassUtil
from potoo.util import singleton, timed
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

    def init(self, app, reload=False, **kwargs):
        if not hasattr(self, '_init_done') or reload:
            log.info()
            sg_load.load(self, **kwargs)
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
    config: dict = field(default_factory=lambda: config.server_globals.sg_load)

    # load*: Maintain tight and well-defined input/output relationships so we can cache (without headaches)
    #   - e.g. any app config we depend on should be explicitly surfaced as function params

    def load(self, sg, skip=[]):
        # Split load* by method so we can easily inspect data volume from @cache dir sizes
        #   - Also helps isolate cache refresh
        for f in [
            self.load_search,
            self.load_xc_meta,
            self.load_feat_info,    # Implicit deps: feat_info   -> sg.search
            self.load_search_recs,  # Implicit deps: search_recs -> sg.xc_meta, sg.feat_info(->sg.search)
        ]:
            if f in skip:
                log.warn('%s [skipped]' % f.__name__)
            else:
                log.debug('%s...' % f.__name__)
                elapsed_s, _ = timed(lambda: (
                    sg.__dict__.update(f())
                ))
                log.info('%s (took %.3fs)' % (f.__name__, elapsed_s))

    @cache(version=2, tags='sg', key=lambda self: self)
    def load_search(self) -> dict:
        log.info()
        x = AttrDict()
        x.search = Search.load_v0(**self.config.search)
        x.projection = x.search.projection
        x.features = x.projection.features
        x.load = x.features.load
        return dict(x)

    @cache(version=4, tags='sg', key=lambda self, **kwargs: (self, kwargs),
        norefresh=True,  # Very slow and rarely worth refreshing [TODO Push this norefresh down closer to the root slowness]
    )
    def load_xc_meta(self, **kwargs) -> dict:
        log.info()
        x = AttrDict()
        x.xc_meta, _recs_stats = load_xc_meta(
            countries_k=self.config.xc_meta.countries_k,
            com_names_k=self.config.xc_meta.com_names_k,
            recs_at_least=None,
            num_species=None,
            num_recs=self.config.xc_meta.num_recs,
            drop_recs_lt_2=False,
            **kwargs,
        )
        return dict(x)

    # No @cache/@lru_cache here because callers do their own caching
    def load_feat_info(self) -> dict:
        from api.recs import get_feat_info  # Lazy import to avoid cycles [TODO Refactor to avoid cycle]
        log.info()
        return dict(
            feat_info=get_feat_info(),  # (Does its own caching)
        )

    # No @cache/@lru_cache here because callers do their own caching
    def load_search_recs(self) -> dict:
        from api.recs import get_search_recs  # Lazy import to avoid cycles [TODO Refactor to avoid cycle]
        log.info()
        return dict(
            search_recs=get_search_recs(),  # (Does its own caching)
        )


# Workaround for @singleton (above)
sg_load = _sg_load()
sg = _sg()
