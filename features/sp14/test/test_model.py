import numpy as np
import pytest

from sp14.model import *


def make_model(**kwargs):
    return Model(
        verbose_params=False,  # Too noisy for tests
        **kwargs,
    )


def test__patches():

    S = np.array([
        # f=3, t=5
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [5, 6, 7, 8, 9],
    ])

    model = make_model(patch_length=1)
    [patches] = model._patches([(None, None, S)])
    np.testing.assert_array_equal(patches, np.array([
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [5, 6, 7, 8, 9],
    ]))

    model = make_model(patch_length=2)
    [patches] = model._patches([(None, None, S)])
    np.testing.assert_array_equal(patches, np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]))

    model = make_model(patch_length=3)
    [patches] = model._patches([(None, None, S)])
    np.testing.assert_array_equal(patches, np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
    ]))

    model = make_model(patch_length=4)
    [patches] = model._patches([(None, None, S)])
    np.testing.assert_array_equal(patches, np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9],
    ]))

    model = make_model(patch_length=5)
    [patches] = model._patches([(None, None, S)])
    np.testing.assert_array_equal(patches, np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
        [3],
        [4],
        [5],
        [6],
        [7],
        [5],
        [6],
        [7],
        [8],
        [9],
    ]))
