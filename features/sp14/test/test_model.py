from attrdict import AttrDict
import numpy as np
import pandas as pd
import pytest
import sklearn.utils

from cache import cache_control
from load import Load
from sp14.model import Features, Projection, Search, SearchEvals
from util import *


@pytest.fixture(autouse=True, scope='module')
def disable_cache():
    cache_control(enabled=False)


def test_model_pipeline():

    # Estimators
    search = Search()
    projection = search.projection
    features = projection.features
    load = features.load

    # The high-level pipeline, skipping the low-level steps
    recs = load.recs(['peterson-field-guide'])[:3]
    recs = features.transform(recs)
    projection.fit(recs)
    recs = projection.transform(recs)
    search.fit(recs)

    # Predict
    search.species(recs)
    search.species_probs(recs)
    search.similar_recs(recs, 3)

    # Eval
    search.confusion_matrix(recs)
    # search.coverage_error(recs, by='species')  # FIXME sklearn coverage_error fails if y_true only has one class


def test_model_pipeline_steps():

    # Estimators
    search = Search()
    projection = search.projection
    features = projection.features
    load = features.load

    # All the load steps
    recs = load.recs(['peterson-field-guide'])[:3]
    _ = recs.assign(
        audio=load.audio,
    )

    # All the features steps
    recs = features.transform(recs)
    _ = recs.assign(
        spectro=features.spectro,
        patches=features.patches,
    )

    # All the projection steps
    projection.fit(recs)
    recs = projection.transform(recs)
    _ = recs.assign(
        proj=projection.proj,
        agg=projection.agg,
        feat=projection.feat,
    )

    # All the search steps
    search.fit(recs)


@pytest.mark.slow
@pytest.mark.parametrize('cache_enabled,cache_refresh', [
    (False, False),
    (True,  False),
    (True,  True),
])
@pytest.mark.parametrize('override_scheduler', [
    'processes',
    'threads',
    'synchronous',
])
def test_model_pipeline_steps_with_cache_and_dask(override_scheduler, cache_enabled, cache_refresh):
    with dask_opts(override_scheduler=override_scheduler):
        with cache_control(enabled=cache_enabled, refresh=cache_refresh):
            test_model_pipeline_steps()


def test_features_patches():

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


def test_projection_proj():

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


def test_coverage_errors():
    search_evals = SearchEvals(
        classes=np.array(['a', 'b', 'c']),
        y=np.array([
            'b',
            'b',
            'a',
            'z',
        ]),
        y_scores=np.array([
            [.2, .6, .2],
            [.5, .3, .3],
            [.9, .1, .0],
            [.4, .1, .3],
        ])
    )
    np.testing.assert_array_equal(search_evals.coverage_errors(), [
        1,
        3,
        1,
        np.inf,
    ])
