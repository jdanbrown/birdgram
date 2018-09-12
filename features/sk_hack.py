"""
sklearn "hacks": things I copied and pasted and tweaked from sklearn because it was more pragmatic than (a) working
around the desired tweaks or (b) rewriting it entirely myself.

See sk_util for more modular, responsible utils for sklearn.
"""

from collections import defaultdict
import copy
from datetime import datetime
from functools import partial
import glob
import inspect
from itertools import product
import os
from pathlib import Path
import re
import textwrap
import uuid
import warnings

import humanize
import joblib
from more_itertools import flatten, unique_everseen
import numpy as np
from potoo.util import timed
from scipy.stats import rankdata
import sh
import structlog
import yaml

from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection._search import GridSearchCV
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import six
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import indexable
from sklearn.utils.deprecation import DeprecationDict
from sklearn.metrics.scorer import _check_multimetric_scoring

from cache import cache
from proc_stats import ProcStats
from sk_util import joblib_dumps
from util import box, ensure_parent_dir, enumerate_with_n, model_stats, sha1hex

log = structlog.get_logger(__name__)


@cache(version=4)
def _fit_and_score_cached(*args, **kwargs):
    """
    Like _fit_and_score except:
    - Cached
    - Returns estimator if return_estimator=True (default: False, like the other _fit_and_score args)
    - Logs
    """

    extra_metrics = kwargs.pop('extra_metrics', None)
    recompute_extra_metrics = kwargs.pop('recompute_extra_metrics', False)
    return_estimator = kwargs.pop('return_estimator', False)
    (i, n) = kwargs.pop('i_n', (None, None))
    split_i = kwargs.pop('split_i', None)
    proc_stats_interval = kwargs.pop('proc_stats_interval', 1.)  # sec
    artifacts = kwargs.pop('artifacts', {})
    if artifacts:
        # Fail fast on required fields
        assert artifacts.get('save') or artifacts.get('reuse'), f'Expected save and/or reuse: artifacts[{artifacts}]'
        artifacts_dir = artifacts['dir']
        if artifacts.get('reuse'):
            assert artifacts['reuse'] == artifacts['experiment_id']
        experiment_id = artifacts['experiment_id']

    a = inspect.signature(_fit_and_score).bind(*args, **kwargs).arguments
    estimator = a['estimator']
    X = a['X']
    y = a['y']
    train = a['train']
    test = a['test']
    parameters = a['parameters']

    # model_id
    estimator_for_model_id = clone(estimator).set_params(**parameters)  # Like _fit_and_score
    model_id = (
        re.sub(r'[^a-zA-Z0-9._=,"\'()]+', '-',
            re.sub(r'[[:]', '=',
                re.sub(r'[] ]', '',
                    # TODO Simplify: merge this defunct log_details into the above re.sub's to cancel them out
                    ', '.join([
                        f'split_i[{split_i}]',
                        f'train[{len(train)}]',
                        f'test[{len(test)}]',
                        f'classes[{len(np.unique(y))}]',
                        f'estimator[{estimator_for_model_id}]',
                    ])
                )
            )
        )
    )

    # Logging strs
    log_name = '_fit_and_score: n[%s/%s], model_id[%s]' % (i + 1, n, model_id)

    if artifacts:
        artifact_model_dir = f'{artifacts_dir}/{experiment_id}/{model_id}'
        artifact_model_dir_done = f'{artifact_model_dir}/DONE'

    # HACK Hard-code slots for extra_metrics and estimator
    #   - TODO Change ret list->dict so we can avoid this brittleness
    EXTRA_METRICS_I = 5
    ESTIMATOR_I = 6

    # Compute model
    compute = not (artifacts.get('reuse') and os.path.exists(artifact_model_dir_done))
    if compute:

        # Skip uncached model, if requested
        if artifacts.get('skip_compute_if_missing'):
            log.info('%s: skip_compute_if_missing' % log_name)
            return None

        log.info('%s: fit...' % log_name)
        # Mem overhead of proc_stats: ~370b/poll (interval=1s -> ~1.3mb/hr = ~30mb/d)
        with ProcStats(interval=proc_stats_interval) as proc_stats:
            _, ret = timed(_fit_and_score, *args, **kwargs, finally_=lambda elapsed_s, _: (
                log.info('%s: fit[%.3fs]' % (log_name, elapsed_s))
            ))

        # HACK ("Hard-code slots", above)
        ret.append(None)  # extra_metrics
        ret.append(None)  # estimator

    # Load model from artifacts (if they exist)
    else:

        # Load ret
        ret_path = f'{artifact_model_dir}/ret.pkl'
        _, ret = timed(joblib.load, ret_path, finally_=lambda elapsed_s, _: (
            log.info('%s: reuse.load[%.3fs, %s]: ret' % (
                log_name,
                elapsed_s,
                humanize.naturalsize(os.stat(ret_path).st_size),
            ))
        ))

        # HACK ("Hard-code slots", above)
        if len(ret) < EXTRA_METRICS_I + 1:
            ret.append(None)
        if len(ret) < ESTIMATOR_I + 1:
            ret.append(None)

        # Load estimator (if requested)
        if not (return_estimator or recompute_extra_metrics):
            estimator = None
        else:
            estimator_path = f'{artifact_model_dir}/estimator.pkl'
            _, estimator = timed(joblib.load, estimator_path, finally_=lambda elapsed_s, _: (
                log.info('%s: reuse.load[%.3fs, %s]: estimator' % (
                    log_name,
                    elapsed_s,
                    humanize.naturalsize(os.stat(estimator_path).st_size),
                ))
            ))

        # HACK Needed for extra_metrics, below
        #   - TODO Consolidate this concern: currently split across user extra_metrics and `if compute`, above
        proc_stats = ret[EXTRA_METRICS_I].get('proc_stats')

    # HACK ("Hard-code slots", above)
    assert len(ret) == ESTIMATOR_I + 1

    # Recompute extra_metrics so that reuse + return_estimator allows us to add/change extra_metrics without retraining
    if extra_metrics and estimator:

        X_train, y_train = _safe_split(estimator, X, y, train)  # Copied from _fit_and_score
        X_test, y_test = _safe_split(estimator, X, y, test, train)  # Copied from _fit_and_score
        _locals = locals()
        def compute_extra_metrics():
            ret[EXTRA_METRICS_I] = {
                # eval instead of lambda because lambda's (and def's) surprisingly don't bust cache when they change
                metric_name: eval(metric_expr, None, _locals)
                for metric_name, metric_expr in extra_metrics.items()
            }
        sizeof_pickle = lambda x: len(joblib_dumps(x))
        log.info('%s: extra_metrics... %s' % (log_name, list(extra_metrics.keys())))
        timed(compute_extra_metrics, finally_=lambda elapsed_s, _: (
            log.info('%s: extra_metrics[%.3fs, %s]: %s' % (
                log_name,
                elapsed_s,
                humanize.naturalsize(sizeof_pickle(ret[EXTRA_METRICS_I])),
                list(extra_metrics.keys()),
            ))
        ))

    if return_estimator:
        # Treat estimator as source of truth in case they disagree (like e.g. extra_metrics)
        ret[ESTIMATOR_I] = estimator or ret[ESTIMATOR_I]
    else:
        # Force-exclude estimator, e.g. if ret.pkl already included it
        ret[ESTIMATOR_I] = None

    # Save ret.pkl + estimator.pkl (if needed)
    if artifacts.get('save') and (compute or recompute_extra_metrics):
        artifacts_to_save = artifacts['save']
        if artifacts_to_save is True:
            artifacts_to_save = ['ret', 'estimator']
        if isinstance(artifacts_to_save, list):
            artifacts_to_save = {k: k for k in artifacts_to_save}
        _locals = locals()
        for artifact_name, artifact_expr in artifacts_to_save.items():
            def save():
                value = eval(artifact_expr, None, _locals)
                if artifact_name == 'ret':
                    # Always ret.pkl without estimator to avoid double-saving huge estimators (even if no estimator.pkl)
                    value = copy.copy(value)
                    value[ESTIMATOR_I] = None
                pkl_path = ensure_parent_dir(f'{artifact_model_dir}/{artifact_name}.pkl')
                # Atomic file write (via tmp + atomic rename)
                #   - Don't use /tmp: if it's on a different fs then os.rename will fail. Write in the same dir instead.
                #   - And don't use shutil.move (from /tmp), since it doesn't do an atomic write
                tmp_path = f'{pkl_path}.tmp'
                joblib.dump(value, tmp_path)
                os.rename(tmp_path, pkl_path)
                return os.stat(pkl_path).st_size
            log.info('%s: save: %s...' % (log_name, artifact_name))
            timed(save, finally_=lambda elapsed_s, pkl_size: (
                log.info('%s: save[%.3fs, %s]: %s' % (
                    log_name, elapsed_s, pkl_size and humanize.naturalsize(pkl_size), artifact_name,
                ))
            ))
        Path(ensure_parent_dir(artifact_model_dir_done)).touch()

    return ret


class GridSearchCVCached(GridSearchCV):
    """
    Like GridSearchCV:
    - Methods below are copied from sklearn.model_selection._search (0.19.1)
    - Changes marked with "HACK(db): ..."
    """

    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        fit_params=None,
        n_jobs=1,
        iid=True,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch='2*n_jobs',
        error_score='raise',
        return_train_score='warn',
        extra_metrics=None,  # HACK(db): Added
        return_estimator=False,  # HACK(db): Added
        artifacts=None,  # HACK(db): Added
        **fit_and_score_kwargs,  # HACK(db): Added
    ):
        # HACK(db): Method body isn't copied (it just calls super and sets an attr)
        super().__init__(
            estimator,
            param_grid,
            scoring,
            fit_params,
            n_jobs,
            iid,
            refit,
            cv,
            verbose,
            pre_dispatch,
            error_score,
            return_train_score,
        )
        self.extra_metrics = extra_metrics
        self.return_estimator = return_estimator
        self.artifacts = artifacts
        self.fit_and_score_kwargs = fit_and_score_kwargs

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)
        # Regenerate parameter iterable for each fit
        candidate_params = list(self._get_param_iterator())
        n_candidates = len(candidate_params)
        if self.verbose > 0:
            print("Fitting {0} folds for each of {1} candidates, totalling"
                  " {2} fits".format(n_splits, n_candidates,
                                     n_candidates * n_splits))

        base_estimator = clone(self.estimator)
        pre_dispatch = self.pre_dispatch

        # HACK(db): Added logging for self.artifacts
        if self.verbose > 0 and self.artifacts:
            if self.artifacts.get('reuse'):
                self.artifacts['experiment_id'] = self.artifacts['reuse']
            else:
                # Stamp experiment_id once for all calls to _fit_and_score_cached
                if not self.artifacts.get('experiment_id'):
                    self.artifacts['experiment_id'] = '-'.join([
                        datetime.utcnow().strftime('%Y%m%d-%H%M%S'),
                        sha1hex(str(uuid.uuid4()))[:7],
                    ])
            log.info('artifacts', **self.artifacts)

        # HACK(db): Reformatted
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch,
        )(
            # HACK(db): Changed _fit_and_score -> _fit_and_score_cached
            delayed(_fit_and_score_cached)(
                clone(base_estimator), X, y, scorers, train,
                test, self.verbose, parameters,
                fit_params=fit_params,
                return_train_score=self.return_train_score,
                return_n_test_samples=True,
                return_times=True, return_parameters=False,
                extra_metrics=self.extra_metrics,  # HACK(db): Added
                return_estimator=self.return_estimator,  # HACK(db): Added
                error_score=self.error_score,
                i_n=(i, n),  # HACK(db): Added
                split_i=split_i,  # HACK(db): Added
                artifacts=self.artifacts,  # HACK(db): Added
                **self.fit_and_score_kwargs,  # HACK(db): Added
            )
            # HACK(db): Added i, n, split_i
            for i, n, (parameters, (split_i, (train, test))) in enumerate_with_n(list(product(
                candidate_params,
                enumerate(cv.split(X, y, groups)),  # HACK(db): Added enumerate for split_i
            )))
        )

        # HACK(db): Added None filter to out to allow _fit_and_score_cached to return None (for skip_compute_if_missing)
        #   - Must also recompute n_candidates
        skip_ix = np.array([x is None for x in out])
        out = list(np.array(out)[~skip_ix])
        if not out:
            raise ValueError('skip_compute_if_missing skipped all models (%s -> %s)' % (n_candidates, len(out)))
        candidate_params = list(np.array(candidate_params)[~skip_ix])
        n_candidates = len(candidate_params)

        # HACK(db): Added extra_metrics + return_estimator , and refactored `outs` deconstruction to be more modular
        outs = list(zip(*out))
        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            train_score_dicts = outs.pop(0)
        test_score_dicts = outs.pop(0)
        test_sample_counts = outs.pop(0)
        fit_time = outs.pop(0)
        score_time = outs.pop(0)
        extra_metrics = outs.pop(0)
        estimator = outs.pop(0)
        assert outs == []
        if not self.extra_metrics:
            assert set(extra_metrics) == {None}
        if not self.return_estimator:
            assert set(estimator) == {None}

        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        test_scores = _aggregate_score_dicts(test_score_dicts)
        if self.return_train_score:
            train_scores = _aggregate_score_dicts(train_score_dicts)

        # TODO: replace by a dict in 0.21
        results = (DeprecationDict() if self.return_train_score == 'warn'
                   else {})

        # HACK(db): Added dtype, for override
        def _store(key_name, array, weights=None, splits=False, rank=False, dtype=np.float64):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.

            if not np.issubdtype(dtype, np.number):
                # HACK(db): Added box/unbox to protect np.array values inside of array
                array = [box(x) for x in array]
            array = np.array(array, dtype=dtype).reshape(n_candidates, n_splits)
            if not np.issubdtype(dtype, np.number):
                # HACK(db): Added box/unbox to protect np.array values inside of array
                array = [[x.unbox for x in y] for y in array]
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s" % (split_i, key_name)] = (
                        # HACK(db): Added list vs. np.array branching in case array contains irregular np.array's
                        [x[split_i] for x in array] if not np.issubdtype(dtype, np.number) else
                        array[:, split_i]
                    )

            # HACK(db): Disabled mean/std/rank altogether, to allow irregular-sized array (e.g. variable num classes)
            #   - TODO(db): Update the refit logic below, since it reads from mean_*/rank_*
            # HACK(db): Added condition on issubdtype, to allow non-numeric dtype overrides
            if np.issubdtype(dtype, np.number):
                array_means = np.average(array, axis=1, weights=weights)
                results['mean_%s' % key_name] = array_means
                # Weighted std is not directly available in numpy
                array_stds = np.sqrt(np.average((array -
                                                 array_means[:, np.newaxis]) ** 2,
                                                axis=1, weights=weights))
                results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        # HACK(db): Added splits=True to return the full distribution of fit/score times instead of just (mean, std)
        _store('fit_time', fit_time, splits=True)
        _store('score_time', score_time, splits=True)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray,
                                            np.empty(n_candidates,),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        # HACK(db): Added
        if self.return_estimator:
            _store('estimator', estimator, splits=True, dtype='object')

        # NOTE test_sample counts (weights) remain the same for all candidates
        test_sample_counts = np.array(test_sample_counts[:n_splits],
                                      dtype=np.int)
        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=test_sample_counts if self.iid else None)
            if self.return_train_score:
                prev_keys = set(results.keys())
                _store('train_%s' % scorer_name, train_scores[scorer_name],
                       splits=True)

                if self.return_train_score == 'warn':
                    for key in set(results.keys()) - prev_keys:
                        message = (
                            'You are accessing a training score ({!r}), '
                            'which will not be available by default '
                            'any more in 0.21. If you need training scores, '
                            'please set return_train_score=True').format(key)
                        # warn on key access
                        results.add_warning(key, message, FutureWarning)

        # HACK(db): Added
        if self.extra_metrics:
            for metric_name in unique_everseen(flatten(d.keys() for d in extra_metrics)):
                _store(metric_name, [d[metric_name] for d in extra_metrics], splits=True, dtype='object')

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = candidate_params[self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][
                self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
