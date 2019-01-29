import numpy as np
import pytest

from sk_util import confusion_matrix_prob, coverage_errors


def test_confusion_matrix_proba():
    np.testing.assert_array_equal(
        confusion_matrix_prob(
            y_true=['b', 'b', 'a'],
            y_prob=[
                [.5, .5],
                [.1, .9],
                [.8, .2],
            ],
            classes=['a', 'b'],
        ),
        np.array([
            [.8, .2],
            [.6, 1.4],
        ]),
    )


def test_coverage_error():
    y_true = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],  # True class not present (e.g. after downsampling classes)
    ])
    y_score = np.array([
        [.2, .6, .2],
        [.5, .3, .3],
        [.1, .9, .0],
        [.4, .1, .3],
    ])
    np.testing.assert_array_equal(coverage_errors(y_true, y_score), [
        1,
        3,
        2,
        np.inf,
    ])
