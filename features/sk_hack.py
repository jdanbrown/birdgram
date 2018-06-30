"""
sklearn "hacks": things I copied and pasted and tweaked from sklearn because it was more pragmatic than (a) working
around the desired tweaks or (b) rewriting it entirely myself.

See sk_util for more modular, responsible utils for sklearn.
"""

from collections import defaultdict
from functools import partial
import inspect
from itertools import product
import warnings

from more_itertools import flatten, unique_everseen
import numpy as np
from potoo.util import timed
from scipy.stats import rankdata

from sklearn.base import is_classifier, clone
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
from log import log
from util import box


@cache(version=4)
def _fit_and_score_cached(*args, **kwargs):
    """
    Like _fit_and_score except:
    - Cached
    - Returns estimator if return_estimator=True (default: False, like the other _fit_and_score args)
    - Logs
    """

    extra_metrics = kwargs.pop('extra_metrics', None)
    return_estimator = kwargs.pop('return_estimator', False)
    split_i = kwargs.pop('split_i')

    a = inspect.signature(_fit_and_score).bind(*args, **kwargs).arguments
    estimator = a['estimator']
    X = a['X']
    y = a['y']
    train = a['train']
    test = a['test']
    parameters = a['parameters']

    log_estimator = clone(estimator).set_params(**parameters)  # Like _fit_and_score
    log_details = ', '.join([
        f'split_i[{split_i}]',
        f'train[{len(train)}]',
        f'test[{len(test)}]',
        f'classes[{len(np.unique(y))}]',
        f'estimator[{log_estimator}]',
    ])
    log.info('_fit_and_score... %s' % (log_details))
    _, ret = timed(_fit_and_score, *args, **kwargs, finally_=lambda elapsed_s: (
        log.info('_fit_and_score[%.3fs]: %s' % (elapsed_s, log_details))
    ))

    if extra_metrics:
        X_train, y_train = _safe_split(estimator, X, y, train)  # Copied from _fit_and_score
        X_test, y_test = _safe_split(estimator, X, y, test, train)  # Copied from _fit_and_score
        ret.append({
            # eval instead of lambda because lambda's (and def's) surprisingly don't bust cache when they change
            metric_name: eval(metric_expr, None, dict(
                estimator=estimator,
                i_train=train,
                X_train=X_train,
                y_train=y_train,
                i_test=test,
                X_test=X_test,
                y_test=y_test,
            ))
            for metric_name, metric_expr in extra_metrics.items()
        })

    if return_estimator:
        ret.append(estimator)

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
                split_i=split_i,  # HACK(db): Added
            )
            for parameters, (split_i, (train, test)) in product(  # HACK(db): Added split_i
                candidate_params,
                enumerate(cv.split(X, y, groups)),  # HACK(db): Added enumerate for split_i
            )
        )

        # HACK(db): Added extra_metrics + return_estimator , and refactored `outs` deconstruction to be more modular
        outs = list(zip(*out))
        # if one choose to see train score, "out" will contain train score info
        if self.return_train_score:
            train_score_dicts = outs.pop(0)
        test_score_dicts = outs.pop(0)
        test_sample_counts = outs.pop(0)
        fit_time = outs.pop(0)
        score_time = outs.pop(0)
        if self.extra_metrics:
            extra_metrics = outs.pop(0)
        if self.return_estimator:
            estimators = outs.pop(0)
        assert outs == []

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
            results['estimators'] = list(estimators)

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
