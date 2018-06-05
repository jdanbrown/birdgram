import joblib
from more_itertools import flatten
import numpy as np
import sklearn as sk

from cache import cache


class GridSearchCVMultiCached(sk.model_selection.GridSearchCV):
    # Inherit so we can reuse the constructor's long param list, which we can't *args,**kwargs because of .get_params()

    def fit(self, X, y=None) -> 'Self':
        n_jobs_upper = -self.n_jobs if self.n_jobs < 0 else 1
        n_jobs_lower = self.n_jobs if self.n_jobs >= 0 else 1
        self.cvs_ = joblib.Parallel(n_jobs=n_jobs_upper)(
            joblib.delayed(self._fit_cached_pickleable)(
                sk.model_selection.GridSearchCV(**{
                    **self.get_params(deep=False),
                    'param_grid': {k: [v] for k, v in params.items()},
                    'n_jobs': n_jobs_lower,
                }),
                X,
                y,
            )
            for params in sk.model_selection.ParameterGrid(self.param_grid)
        )
        self.cv_results_ = {
            k: np.concatenate([cv.cv_results_[k] for cv in self.cvs_])
            for k in set(flatten(cv.cv_results_.keys() for cv in self.cvs_))
        }
        return self

    def _fit_cached_pickleable(self, *args, **kwargs):
        return self._fit_cached(*args, **kwargs)

    @cache(version=0, key=lambda self, cv, X, y: (cv, X, y))
    def _fit_cached(self, cv, X, y):
        return cv.fit(X, y)
