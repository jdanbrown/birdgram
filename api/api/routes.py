# Docs:
#   - http://flask.pocoo.org/docs/0.11/quickstart/

import inspect
import traceback
from typing import Callable, Sequence

from flask import Blueprint, redirect, request
from flask.json import jsonify
import structlog

from api import compspectro, ebird

log = structlog.get_logger(__name__)
bp  = Blueprint('routes', __name__, static_folder='static')


#
# Routes
#


@bp.route('/')
def root():
    return redirect('/health')


@bp.route('/health')
def health():
    return jsonify('healthy')


@bp.route('/debug/error')
def error():
    raise Exception('oops')


@bp.route('/focus-birds/v0', methods=['GET'])
def focus_birds_v0():
    return jsonify(with_parsed_request_args(
        compspectro.focus_birds_v0,
        request.args,
    ))


@bp.route('/nearby_barcharts', methods=['GET'])
def nearby_barcharts():
    return jsonify_df(with_parsed_request_args(
        ebird.nearby_barcharts,
        request.args,
        checks=dict(
            num_nearby_hotspots=('<= 10', lambda x: x <= 10),
        ),
    ))


@bp.route('/nearby_hotspots', methods=['GET'])
def nearby_hotspots():
    return jsonify_df(with_parsed_request_args(
        ebird.nearby_hotspots,
        request.args,
        checks=dict(
            dist_km=('<= 500', lambda x: x <= 500),
        ),
    ))


@bp.route('/barcharts', methods=['GET'])
def barcharts():
    # Enforce only one loc_id, to push costs of accidental api abuse back to caller
    return jsonify_df(with_parsed_request_args(
        ebird.barcharts,
        request.args,
        overrides=dict(
            loc_ids=lambda kwargs: [kwargs['loc_id']],
            loc_id=lambda kwargs: None,  # Drop
        ),
    ))


#
# Util
#


def with_parsed_request_args(
    func: Callable[[str], any],
    request_args: dict,
    overrides: dict = {},
    checks: dict = {},
) -> any:
    return func(**_parse_request_args(
        func,
        request_args,
        overrides,
        checks,
    ))


def _parse_request_args(
    func: Callable[[str], any],
    request_args: dict,
    overrides: dict = {},
    checks: dict = {},
) -> dict:
    """
    Compute parsed args for func but don't call func, for easier testing
    """

    kwargs = request_args.to_dict()  # MultiMap -> dict
    kwargs_in = kwargs.copy()  # Preserve initial state to avoid side effects

    # 1. Parse args explicitly via user override
    for k, parse in overrides.items():
        try:
            v = overrides[k](kwargs_in)
        except Exception as e:
            raise ResponseStatusException(400, 'Invalid param', param=k, error=e)
        if v is None:
            del kwargs[k]
        else:
            kwargs[k] = v

    # 2. Parse resulting args implicitly via function signature
    for k, v_str in kwargs.items():
        try:
            v = _parse_request_arg(func, k, v_str)
        except Exception as e:
            raise ResponseStatusException(400, 'Invalid param', param=k, value=v_str, error=e)
        kwargs[k] = v

    # 3. Validate parsed values via user checks
    for k, (check, is_valid) in checks.items():
        if k in kwargs:
            if not is_valid(kwargs[k]):
                raise ResponseStatusException(400, 'Invalid param', param=k, value=kwargs[k], check=check)

    # Return 400 if kwargs don't match func signature
    _validate_kwargs(func, **kwargs)

    return kwargs


def _parse_request_arg(func, k: str, v: str) -> any:
    typ = inspect.getfullargspec(func).annotations.get(k)
    if typ is None:
        parse = lambda x: x
    elif typ == Sequence[str]:
        parse = lambda x: x
    # elif typ == ...
    #    ...
    else:
        parse = typ
    return parse(v)


def _validate_kwargs(func, **kwargs):
    argspec = inspect.getfullargspec(func)
    argspec_defaults = argspec.defaults or []
    required_args = argspec.args[:-len(argspec_defaults)]
    optional_args = dict(zip(argspec.args[-len(argspec_defaults):], argspec_defaults))
    if not set(required_args).issubset(set(kwargs)):
        raise ResponseStatusException(400, 'Extraneous params',
            required=required_args,
            optional=optional_args,
            given=kwargs,
        )
    elif not set(kwargs).issubset(set(required_args) | set(optional_args)):
        raise ResponseStatusException(400, 'Missing params',
            required=required_args,
            optional=optional_args,
            given=kwargs,
        )


def jsonify_df(df):
    return jsonify(df.to_dict(orient='records'))


class ResponseStatusException(Exception):
    def __init__(self, status_code: int, reason: str, error: BaseException = None, **data):
        self.status_code = status_code
        self.payload = dict(
            reason=reason,
            data=data,
        )
        self.error = error
        if error is not None:
            self.payload['error'] = f'{type(error).__name__}: {error}'


@bp.errorhandler(ResponseStatusException)
def exception_to_response_with_status(e: ResponseStatusException):
    if e.error is not None:
        log.debug('ResponseStatusException_with_error', e=e)
        traceback.print_exception(None, e, e.__traceback__)
    return jsonify(e.payload), e.status_code
