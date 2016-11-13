import os.path
import pickle
import structlog

from api.util import mkdir_p


log       = structlog.get_logger(__name__)
cache_dir = '/tmp/bubo-api-cache'


def get_or_put(k, f, dumps=pickle.dumps, loads=pickle.loads):
    v = get(k, loads)
    if v is not None:
        return v
    else:
        v = f()
        put(k, v, dumps)
        return v


def get(k, loads=pickle.loads):
    try:
        with open(os.path.join(cache_dir, k), 'rb') as f:
            v = f.read()
        log.info('cache_get_hit', k=k)
    except OSError:
        log.info('cache_get_miss', k=k)
        return None
    else:
        return loads(v)


def put(k, v, dumps=pickle.dumps):
    mkdir_p(cache_dir)
    with open(os.path.join(cache_dir, k), 'wb') as f:
        f.write(dumps(v))
    log.info('cache_put', k=k)
