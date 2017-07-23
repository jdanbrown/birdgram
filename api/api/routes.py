# Docs:
#   - http://flask.pocoo.org/docs/0.11/quickstart/

import structlog
from flask import Blueprint, redirect, request
from flask.json import jsonify

from api import ebird
from api.util import LonLat

log = structlog.get_logger(__name__)
bp  = Blueprint('routes', __name__)


@bp.route('/')
def root():
    return redirect('/health')


@bp.route('/health')
def health():
    return jsonify('healthy')


@bp.route('/debug/error')
def error():
    raise Exception('oops')


@bp.route('/nearby_hotspots', methods=['GET', 'OPTIONS'])
def nearby_hotspots():
    kwargs = request.args.to_dict()
    kwargs['lonlat'] = LonLat(kwargs['lonlat'])
    return jsonify_df(ebird.nearby_hotspots(**kwargs))


@bp.route('/barcharts', methods=['GET', 'OPTIONS'])
def barcharts():
    kwargs = request.args.to_dict()
    kwargs['loc_ids'] = [kwargs.pop('loc_id')]
    return jsonify_df(ebird.barcharts(**kwargs))


@bp.route('/nearby_barcharts', methods=['GET', 'OPTIONS'])
def nearby_barcharts():
    kwargs = request.args.to_dict()
    kwargs['lonlat'] = LonLat(kwargs['lonlat'])
    return jsonify_df(ebird.nearby_barcharts(**kwargs))


def jsonify_df(df):
    return jsonify(df.to_dict(orient='records'))


# TODO
# def parse_request_args(func, request_args: dict) -> dict:
#     return {k: v for k, v in request_args.items()}
#
#     for k, typ in func.__annotations__.items():
#         if k != 'return':
#             if typ == Sequence[str]:
#                 pass
