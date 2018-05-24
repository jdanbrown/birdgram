from addict import Dict  # TODO Switch back to AttrDict since Dict().x doesn't fail, and PR an AttrDict.to_dict()
import numpy as np
import pytest
import sklearn.utils

from sp14.model import *


def make_model(**kwargs):
    return Model(
        verbose_config=False,  # Too noisy for tests
        **kwargs,
    )


def test_model():

    recs_to_X = lambda recs: [Recording(**row) for i, row in recs.iterrows()]
    recs_to_y = lambda recs: np.array(recs.species)

    recs_paths = load_recs_paths(['peterson-field-guide'])[:10]
    recs = load_recs_data(recs_paths)

    train_n, test_n = 5, 5
    recs_shuf = sklearn.utils.shuffle(recs, random_state=0)
    recs_train, recs_test = recs_shuf[:train_n], recs_shuf[train_n : train_n + test_n]
    recs_train_X, recs_test_X = recs_to_X(recs_train), recs_to_X(recs_test)
    recs_train_y, recs_test_y = recs_to_y(recs_train), recs_to_y(recs_test)

    model = make_model()
    model.fit_proj(recs_train_X);
    model.fit_class(recs_train_X, recs_train_y);
    model.predict(recs_test_X, 'classes')
    model.predict(recs_test_X, 'kneighbors')


def test__patches():

    S = np.array([
        # f=3, t=5
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [5, 6, 7, 8, 9],
    ])

    [patches] = Model._patches_from_spectros([(None, None, S)], patch_length=1)
    np.testing.assert_array_equal(patches, np.array([
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [5, 6, 7, 8, 9],
    ]))

    [patches] = Model._patches_from_spectros([(None, None, S)], patch_length=2)
    np.testing.assert_array_equal(patches, np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]))

    [patches] = Model._patches_from_spectros([(None, None, S)], patch_length=3)
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

    [patches] = Model._patches_from_spectros([(None, None, S)], patch_length=4)
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

    [patches] = Model._patches_from_spectros([(None, None, S)], patch_length=5)
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


def test__transform_proj():
    proj_skm = Dict(transform=lambda X: -X)
    patches = [
        # f*p = 3, t variable per patch
        np.array([
            [0, 1],
            [1, 2],
            [2, 3],
        ]),
        np.array([
            [2, 4, 5],
            [3, 5, 6],
            [4, 6, 7],
        ]),
        np.array([
            [6],
            [7],
            [8],
        ]),
    ]
    projs = Model._transform_proj(proj_skm, patches)
    for proj, expected in zip(projs, [
        np.array([
            [-0, -1],
            [-1, -2],
            [-2, -3],
        ]),
        np.array([
            [-2, -4, -5],
            [-3, -5, -6],
            [-4, -6, -7],
        ]),
        np.array([
            [-6],
            [-7],
            [-8],
        ]),
    ]):
        np.testing.assert_array_equal(proj, expected)
