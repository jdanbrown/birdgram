from attrdict import AttrDict
import numpy as np
import pandas as pd
import pytest
import sklearn.utils

from cache import cache_control
from load import Load
from sp14.model import Features, Projection, Search


@pytest.fixture(autouse=True, scope='module')
def disable_cache():
    cache_control(enabled=False)


def test_model():

    load = Load()
    recs = load.recs(['peterson-field-guide'])[:10]

    # Basic features
    features = Features()
    recs = features.transform(recs)

    # Fit projection, add learned features
    train_projection_n = 10
    _shuf = sklearn.utils.shuffle(recs, random_state=0)
    recs_train_projection = _shuf[:train_projection_n]
    projection = Projection()
    projection.fit(recs_train_projection)
    recs = projection.transform(recs)

    # Fit search
    train_n, test_n = 5, 5
    _shuf = sklearn.utils.shuffle(recs, random_state=0)
    recs_train, recs_test = _shuf[:train_n], _shuf[train_n : train_n + test_n]
    search = Search(projection=projection)
    search.fit(recs_train)

    # Predict
    search.species(recs_test)
    search.species_probs(recs_test)
    search.similar_recs(recs_test, 3)

    # Eval
    search.confusion_matrix(recs_test)
    # search.coverage_error(recs_test, by='species')  # FIXME sklearn coverage_error fails if y_true only has one class


def test__patches():

    S = np.array([
        # f=3, t=5
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [5, 6, 7, 8, 9],
    ])
    rec = pd.Series(dict(
        id='id',
        dataset='dataset',
        spectro=(None, None, S),
    ))

    patches = Features(patch_length=1)._patches(rec)
    np.testing.assert_array_equal(patches, np.array([
        [1, 2, 3, 4, 5],
        [3, 4, 5, 6, 7],
        [5, 6, 7, 8, 9],
    ]))

    patches = Features(patch_length=2)._patches(rec)
    np.testing.assert_array_equal(patches, np.array([
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ]))

    patches = Features(patch_length=3)._patches(rec)
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

    patches = Features(patch_length=4)._patches(rec)
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

    patches = Features(patch_length=5)._patches(rec)
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
    recs = pd.DataFrame([
        dict(
            id='id',
            dataset='dataset',
            patches=a_patches,
        )
        for a_patches in patches
    ])

    projection = Projection()
    projection.skm_ = skm
    proj = projection.proj(recs)
    for a_proj, expected in zip(proj, [
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
        np.testing.assert_array_equal(a_proj, expected)
