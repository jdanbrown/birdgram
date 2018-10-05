import inspect
import numbers
import pdb
import traceback
from typing import Callable, Sequence
from urllib.parse import urlencode, urlsplit

import flask
from flask import current_app as app  # How to access app in a blueprint [https://stackoverflow.com/a/38262792/397334]
from flask import Blueprint, Markup, redirect, render_template, render_template_string, request, Response
from flask.json import jsonify
from more_itertools import partition
from potoo.ipython import ipy_formats, ipy_formats_to_html
from potoo.util import ensure_endswith, strip_endswith
import structlog
from toolz import compose, valfilter, valmap
import werkzeug
from werkzeug.datastructures import MultiDict
from werkzeug.urls import Href, URL, url_parse, url_unparse

import api.recs
from api.util import *
from logging_ import *
from util import *

log = structlog.get_logger(__name__)
bp = Blueprint('routes', __name__, static_folder='api/static')


#
# Routes
#


@bp.route('/')
def root():
    return redirect('/recs/xc/species')


@bp.route('/health')
def health():
    return jsonify('healthy')


@bp.route('/debug/error')
def error():
    raise Exception('oops')


@bp.route('/api/recs/xc/meta', methods=['GET'])
def api_recs_xc_meta():
    return jsonify_df(with_parsed_request_args(
        api.recs.xc_meta,
        request.args,
    ))


@bp.route('/recs/xc/species', methods=['GET'])
def recs_xc_species():
    return htmlify_df('recs_xc_species.html.j2', with_parsed_request_args(
        api.recs.xc_species_html,
        request.args,
    ))


@bp.route('/recs/xc/similar', methods=['GET'])
def recs_xc_similar():
    return htmlify_df('recs_xc_similar.html.j2', with_parsed_request_args(
        api.recs.xc_similar_html,
        request.args,
    ))


#
# Template defs: req_*
#


@bp.app_template_global()
def is_dev() -> bool:
    return bool(req_query_get('dev'))


@bp.app_template_global()
def req_url_replace(**fields) -> str:
    return url_replace(request.url, **fields)


@bp.app_template_global()
def req_path_with_query_params(**query_params) -> str:
    return href(request.path)(**valfilter(lambda v: v is not None, {
        **request.args,
        **query_params,
    }))


@bp.app_template_global()
def req_path_with_zoom(zoom_inc: Optional[int]) -> str:
    return req_path_with_query_params(**_zoom_params(zoom_inc))


def _zoom_params(
    zoom_inc: Union['-1', None, '1'],  # None to reset zoom
) -> dict:
    assert zoom_inc in [-1, None, 1]
    curr_scale  = coalesce(or_else(None, lambda: float(request.args.get('scale'))),  api.recs.defaults['scale'])
    curr_n_recs = coalesce(or_else(None, lambda: float(request.args.get('n_recs'))), api.recs.defaults['n_recs'])
    curr_scale  = np.clip(curr_scale,  1, 5)     # TODO De-dupe with api.recs
    curr_n_recs = np.clip(curr_n_recs, 0, None)  # TODO De-dupe with api.recs
    scale = (
        np.clip(curr_scale + zoom_inc, 1, 5) if zoom_inc is not None else
        api.recs.defaults['scale']
    )
    clean_float_as_str = lambda x: strip_endswith(str(x), '.0')
    return dict(
        scale  = clean_float_as_str(scale),
        n_recs = clean_float_as_str(curr_n_recs * curr_scale / scale)  # float i/o int so zooming in/out doesn't truncate
    )


@bp.app_template_global()
def req_path_with_toggle_dev() -> str:
    dev = coalesce(or_else(None, lambda: int(request.args.get('dev'))), api.recs.defaults['dev'])
    return req_path_with_query_params(
        dev=int(not dev) or None,  # None to omit 'dev' query params instead of '&dev=0'
    )


@bp.app_template_global()
def req_query_get(k: str, default_default: any = None) -> str:
    return request.args.get(k, api.recs.defaults.get(k) or default_default)


@bp.app_template_global()
def req_switch_host_host() -> dict:
    if urlsplit(request.url).netloc == config.hosts.prod:
        return config.hosts.local
    else:
        return config.hosts.prod


@bp.app_template_global()
def req_switch_host_text() -> dict:
    if urlsplit(request.url).netloc == config.hosts.prod:
        return 'R'
    else:
        return 'L'


@bp.app_template_global()
def req_href(path=None, *args, **kwargs) -> Href:
    """Like href, except default path to request.path, and merge kwargs with request.args"""
    path = path or request.path
    h = href(path, *args, **kwargs)
    def f(_kwargs=None, **kwargs):
        assert bool(_kwargs) != bool(kwargs)
        kwargs = _kwargs or kwargs
        return h(**{
            **request.args,
            **kwargs,  # User kwargs override keys in request.args
        })
    return f


#
# Template defs: *
#


@bp.app_template_global()
def href(*args, safe=True, **kwargs) -> Href:
    """
    Usage:
        <a href="{{ href('/path')(a=1, b=2) }}">                    = <a href="/path?a=1&b=2">
        <a href="{{ href('/path')(a=1, b=2) | forceescape }}">      = <a href="/path?a=1&amp;b=2">
        <a href="{{ href('/path', safe=False)(a=1, b=2) }}">        = <a href="/path?a=1&amp;b=2">
        <a href="{{ href('/path', safe=False)(a=1, b=2) | safe }}"> = <a href="/path?a=1&b=2">
    """
    h = Href(*args, **kwargs)
    if safe:
        h = compose(Markup, h)
    return h


#
# Cross cutting (bp.*)
#


# Disable caching for computed responses
#   - For /static, see config['SEND_FILE_MAX_AGE_DEFAULT'] in create_app
@bp.after_request
def add_headers_no_caching(rep):
    rep.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    rep.headers['Pragma'] = 'no-cache'
    rep.headers['Expires'] = '0'
    return rep


@bp.errorhandler(Exception)
def handle_exception(e):

    # Break into debugger if $BUBO_PDB is truthy
    #   - But not if stdin isn't a tty, e.g. under `entr -r` (which closes stdin for subprocs)
    if app.config['BUBO_PDB'] and sys.stdin.isatty():
        pdb.post_mortem(e.__traceback__)

    # Translate all exceptions to an http response
    #   - XXX Disables the flask browser debugger, which seems to be the best way to debug
    # log.debug('handle_exception', e=e)
    # traceback.print_exception(None, e, e.__traceback__)
    # if not isinstance(e, ApiError):
    #     e = ApiError(status_code=500, msg=str(e))

    # Translate ApiError's into http responses
    if isinstance(e, ApiError):
        rep = jsonify({'msg': e.msg, **valmap(str, e.kwargs)})
        rep.status_code = e.status_code
        return rep
    # Translate RedirectError's into http 3xx responses
    #   - HACK Signaling redirects via exception is messy and encourages bad code structure; refactor callers and kill this
    elif isinstance(e, RedirectError):
        rep = Response()
        rep.status_code = e.status_code
        rep.headers['Location'] = e.location
        return rep
    else:
        raise e


#
# Util
#


def url_replace(url: str, **kwargs) -> str:
    # werkzeug.urls is great: http://werkzeug.pocoo.org/docs/0.14/urls/
    return url_parse(url).replace(**kwargs).to_url()


def with_parsed_request_args(
    func: Callable[[str], any],
    request_args: dict,
    overrides: dict = {},
    checks: dict = {},  # TODO Started using more lightweight require(...) directly in api.recs
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
            raise ApiError(400, 'Invalid param', param=k, error=e)
        if v is None:
            del kwargs[k]
        else:
            kwargs[k] = v

    # 2. Parse resulting args implicitly via function signature
    for k, v_str in kwargs.items():
        try:
            v = _parse_request_arg(func, k, v_str)
        except Exception as e:
            raise ApiError(400, 'Invalid param', param=k, value=v_str, error=e)
        kwargs[k] = v

    # 3. Validate parsed values via user checks
    for k, (check, is_valid) in checks.items():
        if k in kwargs:
            if not is_valid(kwargs[k]):
                raise ApiError(400, 'Invalid param', param=k, value=kwargs[k], check=check)

    # Return 400 if kwargs don't match func signature
    _validate_kwargs(func, **kwargs)

    return kwargs


def _parse_request_arg(func, k: str, v: str) -> any:
    param = inspect.signature(func).parameters.get(k)
    typ = param and param.annotation
    if typ is None:
        parse = lambda x: x
    elif typ == Sequence[str]:  # TODO Sequence[X], generically
        parse = lambda x: x.split(',') if x else []
    elif issubclass(typ, numbers.Number):
        parse = lambda x: None if x == '' else typ(x)
    # elif typ == ...
    #    ...
    else:
        parse = lambda x: typ(x)
    return parse(v)


def _validate_kwargs(func, **kwargs):
    params = inspect.signature(func).parameters
    (required_args, optional_args) = partition(lambda p: p.default != inspect.Signature.empty, params.values())
    required_args = [p.name for p in required_args]
    optional_args = {p.name: p.default for p in optional_args}
    if set(kwargs) - {*required_args, *optional_args}:
        # Redirect instead of hard error on extra params
        #   - Still provides (subtle) feedback to user by changing the url
        #   - Smooths the jumps between linked pages (e.g. /species <-> /similar) by silently dropping irrelevant params
        raise RedirectError(302, Href(request.base_url)(MultiDict([
            (k, v)
            for k, v in request.args.items(multi=True)  # Preserve repeated keys from MultiDict
            if k in {*required_args, *optional_args}
        ])))
    elif set(required_args) - set(kwargs):
        raise ApiError(400, 'Missing params',
            required=required_args,
            optional=optional_args,
            given=kwargs,
        )


def jsonify_df(df) -> Response:
    return jsonify(df.to_dict(orient='records'))


def htmlify_df(template: str, df, render_df_html=True) -> str:
    with log_time_context('ipy_formats'):
        # df_html = ipy_formats_to_html(df)  # XXX This doesn't know df_cell's
        df_html = ipy_formats._format_df(df, mimetype='text/html',  # HACK Promote _format_df from private (and rename?)
            index=False,
            # header=False,  # Keep: helpful to include headers on the table, for now
        )
    with log_time_context('render (slow ok)'):
        # Slow: df_html has a bunch of inline img/audio data urls
        #   - Don't fix this for web only, since faster flask templates aren't likely to be critical path for mobile
        if render_df_html:
            # Render inline templates in the df html str
            df_html = render_template_string(df_html)
        return htmlify_html(
            template=template,
            body_html=df_html,
        )


def htmlify_html(template: str, body_html: str) -> str:
    return render_template(template,
        body_html=body_html,
        # (More stuff will end up here...)
    )


# XXX Replaced by ApiError [anything here we should crib before deleting it?]
#
# class ResponseStatusException(Exception):
#     def __init__(self, status_code: int, reason: str, error: BaseException = None, **data):
#         self.status_code = status_code
#         self.payload = dict(
#             reason=reason,
#             data=data,
#         )
#         self.error = error
#         if error is not None:
#             self.payload['error'] = f'{type(error).__name__}: {error}'
#
#
# @bp.errorhandler(ResponseStatusException)
# def exception_to_response_with_status(e: ResponseStatusException):
#     if e.error is not None:
#         log.debug('ResponseStatusException_with_error', e=e)
#         traceback.print_exception(None, e, e.__traceback__)
#     return jsonify(e.payload), e.status_code
