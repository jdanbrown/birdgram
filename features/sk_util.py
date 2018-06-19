import io

import joblib
import numpy as np
import pandas as pd

from sk_hack import *  # For export


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
