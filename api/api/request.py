import requests
import structlog
from datetime import datetime

from api import cache


log = structlog.get_logger(__name__)


def request(method, url, **kwargs):
    log.debug('http_request', method=method, url=url, **kwargs)
    rep = requests.request(method, url, **kwargs)
    log.info('http_response', method=method, url=url, **kwargs, rep=dict(
        status_code = rep.status_code,
        reason      = rep.reason,
        headers     = rep.headers,
    ))
    rep.raise_for_status()
    return rep


def cached_request(method, url, **kwargs):
    key = dict(
        date   = datetime.utcnow().date().isoformat(),  # TODO Expose to caller
        method = method,
        url    = url,
        **kwargs
    )
    rep = cache.get_or_put(key, lambda: request(method, url, **kwargs))
    assert isinstance(rep, requests.models.Response), {'type': type(rep), 'rep': rep}
    return rep
