import hashlib
import json
import requests
import structlog

from api import cache


log = structlog.get_logger(__name__)


def request(method, url, **kw):
    log.debug('http_request', method=method, url=url, **kw)
    rep = requests.request(method, url, **kw)
    log.info('http_response', method=method, url=url, **kw, rep=dict(
        status_code = rep.status_code,
        reason      = rep.reason,
        headers     = rep.headers,
    ))
    rep.raise_for_status()
    return rep


def cached_request(method, url, **kw):
    key_json = json.dumps(
        sort_keys  = True,
        separators = (',', ':'),
        obj        = dict(
            method = method,
            url    = url,
            **kw
        ),
    )
    key = hashlib.sha1(key_json.encode('utf8')).hexdigest()
    rep = cache.get_or_put(key, lambda: request(method, url, **kw))
    assert isinstance(rep, requests.models.Response), {'type': type(rep), 'rep': rep}
    return rep
