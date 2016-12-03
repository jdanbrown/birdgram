# Docs:
#   - http://flask.pocoo.org/docs/0.11/quickstart/


import structlog
from flask import json, Blueprint, redirect
from flask.json import jsonify


from api import ebird


log = structlog.get_logger(__name__)
bp  = Blueprint('routes', __name__)


@bp.route('/')
def root():
    return redirect('/health')


@bp.route('/health')
def health():
    return jsonify('healthy')


@bp.route('/error')
def error():
    raise Exception('oops')


@bp.route('/barchart', methods=['GET'])
def barchart():
    return jsonify({
        k: json.loads(v.to_json())
        for k,v in ebird.barchart().items()
    })
