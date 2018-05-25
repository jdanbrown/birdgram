from attrdict import AttrDict
import numpy as np
import pytest
import sklearn.utils

from sp14.model import *


def test_model():

    recs_paths = load_recs_paths(['peterson-field-guide'])[:10]
    recs = load_recs_data(recs_paths)

    # Basic features
    features = Features()
    recs['spectro'] = features.spectros(recs)
    recs['patches'] = features.patches(recs)

    # Fit projection, add learned features
    train_projection_n = 10
    _shuf = sklearn.utils.shuffle(recs, random_state=0)
    recs_train_projection = _shuf[:train_projection_n]
    projection = Projection()
    projection.fit(recs_train_projection)
    recs['feat'] = projection.transform(recs)

    # Fit search
    train_n, test_n = 5, 5
    _shuf = sklearn.utils.shuffle(recs, random_state=0)
    recs_train, recs_test = _shuf[:train_n], _shuf[train_n : train_n + test_n]
    search = Search()
    search.fit(recs_train)

    # Predict
    search.predict(recs_test, 'classes')
    search.predict(recs_test, 'kneighbors')

    # TODO Eval search
    # search.confusion_matrix(recs_test)
    # search.coverage_errors(recs_test)


def test__patches():

    S = np.array([
        # f=3, t=5
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [5, 6, 7, 8, 9],
    ])

    [patches] = Features._patches_from_spectros([(None, None, S)], patch_length=1)
    np.testing.assert_array_equal(patches, np.array([
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [5, 6, 7, 8, 9],
    ]))

    [patches] = Features._patches_from_spectros([(None, None, S)], patch_length=2)
    np.testing.assert_array_equal(patches, np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]))

    [patches] = Features._patches_from_spectros([(None, None, S)], patch_length=3)
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

    [patches] = Features._patches_from_spectros([(None, None, S)], patch_length=4)
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

    [patches] = Features._patches_from_spectros([(None, None, S)], patch_length=5)
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
    skm = AttrDict(transform=lambda X: -X)
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
    projs = Projection._projs(patches, skm)
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
