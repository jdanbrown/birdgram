import os.path
import pickle
import structlog

from api.util import mkdir_p


log       = structlog.get_logger(__name__)
cache_dir = '/tmp/bubo-api-cache'


def get_or_put(key, value_f, dumps=pickle.dumps, loads=pickle.loads):
    value = get(key, loads)
    if value is not None:
        return value
    else:
        value = value_f()
        put(key, value, dumps)
        return value


def get(key, loads=pickle.loads):
    try:
        with open(os.path.join(cache_dir, key), 'rb') as f:
            value = f.read()
        log.info('cache_hit', key=key)
    except OSError:
        log.info('cache_miss', key=key)
        return None
    else:
        return loads(value)


def put(key, value, dumps=pickle.dumps):
    mkdir_p(cache_dir)
    with open(os.path.join(cache_dir, key), 'wb') as f:
        f.write(dumps(value))
    log.info('cache_write', key=key)
