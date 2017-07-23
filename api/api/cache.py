import hashlib
import json
import os.path
import pickle
import structlog

from api.util import mkdir_p


log       = structlog.get_logger(__name__)
cache_dir = '/tmp/bubo-api-cache'


def get_or_put(key_obj, value_f, dumps=pickle.dumps, loads=pickle.loads):
    value = get(key_obj, loads)
    if value is not None:
        return value
    else:
        value = value_f()
        put(key_obj, value, dumps)
        return value


def get(key_obj, loads=pickle.loads):
    key = key_for_obj(key_obj)
    try:
        with open(os.path.join(cache_dir, key), 'rb') as f:
            value = f.read()
        log.info('cache_hit', key=key, key_obj=key_obj)
    except OSError:
        log.info('cache_miss', key=key, key_obj=key_obj)
        return None
    else:
        return loads(value)


def put(key_obj, value, dumps=pickle.dumps):
    key = key_for_obj(key_obj)
    mkdir_p(cache_dir)
    with open(os.path.join(cache_dir, key), 'wb') as f:
        f.write(dumps(value))
    log.info('cache_write', key=key, key_obj=key_obj)


def key_for_obj(key_obj: any) -> str:
    key_json = json.dumps(
        sort_keys  = True,
        separators = (',', ':'),
        obj        = key_obj,
    )
    key = hashlib.sha1(key_json.encode('utf8')).hexdigest()
    return key
