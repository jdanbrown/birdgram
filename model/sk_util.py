import io

import joblib
import numpy as np
import pandas as pd
import sklearn as sk


def sk_dir_attrs(x):
    """Like dir(x), but subset to sk 'attributes' [http://scikit-learn.org/dev/glossary.html#term-attribute]"""
    return [k for k in dir(x) if k.endswith('_') and not k.startswith('_')]


def sk_dirs_attrs(x):
    """Like dirs(x), but subset to sk 'attributes' [http://scikit-learn.org/dev/glossary.html#term-attribute]"""
    return {k: getattr(x, k) for k in sk_dir_attrs(x)}


def joblib_dumps(x: any, *args, **kwargs) -> bytes:
    f = io.BytesIO()
    joblib.dump(x, f, *args, **kwargs)
    return f.getvalue()


def joblib_loads(b: bytes, *args, **kwargs) -> any:
    return joblib.load(io.BytesIO(b), *args, **kwargs)


def confusion_matrix_prob_df(y_true, y_prob, classes) -> 'pd.DataFrame[prob @ (n_species, n_species)]':
    return pd.DataFrame(
        data=confusion_matrix_prob(y_true, y_prob, classes),
        index=pd.Series(classes, name='true'),
        columns=pd.Series(classes, name='pred'),
    )


def confusion_matrix_prob(
    y_true: 'np.ndarray[p]',
    y_prob: 'np.ndarray[p, c]',  # Like from .predict_proba
    classes: 'np.ndarray[c]',
) -> 'np.ndarray[prob @ (c, c)]':
    """Like sk.metrics.confusion_matrix but adds all .predict_proba results instead of just one .predict result"""
    classes = np.array(classes)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    (C,) = classes.shape
    (P,) = y_true.shape
    assert y_prob.shape == (P, C)
    class_i = {c: i for i, c in enumerate(classes)}
    M = np.zeros((C, C))
    for c, y_prob in zip(y_true, y_prob):
        M[class_i[c]] += y_prob
    assert M.shape == (C, C)
    return M


# Like sk.metrics.ranking.coverage_error, except:
#   - Return one coverage error per sample (instead of returning the mean)
#   - If y_true isn't in y_score then fill with np.inf (instead of 0, which is really confusing)
def coverage_errors(
    y_true: 'np.ndarray[bool @ n]',
    y_score: 'np.ndarray[float @ n]',
    fill=np.inf,
) -> 'np.ndarray[np.float @ n]':  # float to represent Union[int, np.inf]

    # Validate inputs
    y_true = sk.utils.check_array(y_true, ensure_2d=False)
    y_score = sk.utils.check_array(y_score, ensure_2d=False)
    sk.utils.check_consistent_length(y_true, y_score)
    y_type = sk.utils.multiclass.type_of_target(y_true)
    if y_type != 'multilabel-indicator':
        raise ValueError('%s format is not supported' % y_type)
    if y_true.shape != y_score.shape:
        raise ValueError('y_true and y_score have different shape')

    y_score_mask = np.ma.masked_array(y_score, mask=np.logical_not(y_true))
    y_min_relevant = y_score_mask.min(axis=1).reshape((-1, 1))
    coverage = (y_score >= y_min_relevant).sum(axis=1)
    coverage = coverage.astype(float).filled(fill)
    return coverage
