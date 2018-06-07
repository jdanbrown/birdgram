import numpy as np
import pytest

from sk_util import confusion_matrix_prob


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
