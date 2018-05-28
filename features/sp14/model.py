"""
Config
    |                             | defaults      | [SP14]         | [SBF16]       |
    |-----------------------------|---------------|----------------|---------------|
    | rec_sample_rate             | 22050         | 44100          | 22050         |
    | spectro_f_min               | 1000          | 500            | 2000          |
    |   f_max                     | 11025         | 22050          | 11025         |
    | spectro_f_bins (f)          | 40            | 40             | 40            |
    | spectro_hop_length          | 256 (12ms)    | 1024 (23ms)    | 32 (1.5ms)    |
    | spectro_frame_length        | 512 (23ms)    | 1024 (23ms)    | 256 (12ms)    |
    |   frame_overlap             | .5            | 0              | .875          |
    |   frames/s (t/s)            | 86            | 43             | 689           |
    | spectro_frame_window        | hann          | hamming        | hann          |
    | norm                        | [TODO]        | RMS+median     | [TODO]        |
    | patch_length (p)            | 4 (46ms)      | ~4 (~93ms)     | ~16 (~22ms)   |
    | proj_skm_variance_explained | .99           | —              | .99           |
    | proj_skm_k (k)              | 500           | 500            | ~512          |
    | agg_funs                    | μ,σ,max       | μ,σ            | ~μ,σ,max      |
    |   a                         | 3             | 2              | ~3            |
    |   features                  | 1500          | 1000           | ~1536         |

Pipeline
    | rec     | (samples,) | (22050/s,)   | (44100/s)     | (22050/s,)
    | spectro | (f, t)     | (40, 86/s)   | (40, 43/s)    | (40, 689/s)
    | patch   | (f*p, t)   | (40*4, 86/s) | (40*~4, 43/s) | (40*~16, 689/s)
    | proj    | (k, t)     | (500, 86/s)  | (500, 43/s)   | (~512, 689/s)
    | agg     | (k, a)     | (500, 3)     | (500, 2)      | (~512, ~3)
    | feat    | (k*a,)     | (1500,)      | (1000,)       | (~1536,)
"""

import copy
from datetime import datetime
import itertools
import json
from typing import Iterable, Union

from attrdict import AttrDict
import dataclasses
from dataclasses import dataclass, field
import joblib
import numpy as np
from potoo.util import round_sig
import sklearn
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
import yaml

from cache import cache, cache_lambda, cache_pure_method
from constants import data_dir
from datasets import *
from datatypes import RecordingDF
from features import *
from load import *
import metadata
from metadata import sorted_species
from sp14.skm import SKM
from util import *
from viz import *


def to_X(recs: RecordingDF) -> Iterable[AttrDict]:
    return [AttrDict(row) for i, row in recs.iterrows()]


def to_y(recs: RecordingDF) -> np.ndarray:
    return np.array(list(recs.species))


class _Base:

    @property
    def config(self) -> AttrDict:
        return dataclasses.asdict(self)


@dataclass
class Features(_Base):

    sample_rate: int = 22050
    f_min: int = 1000
    f_bins: int = 40
    hop_length: int = 256
    frame_length: int = 512
    frame_window: str = 'hann'
    patch_length: int = 4

    @property
    def spectro_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'sample_rate',
            'f_min',
            'f_bins',
            'hop_length',
            'frame_length',
            'frame_window',
        ]})

    @property
    def patch_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'patch_length',
        ]})

    # .audio -> .spectro -> .patches
    #   - TODO TODO transform vs. patches?
    def transform(self, recs: RecordingDF) -> RecordingDF:
        """.patches <- .spectro <- .audio"""
        # TODO TODO Don't load .spectro/.audio when .patches missing but all are cached
        if 'patches' not in recs:
            recs = recs.copy()
            if 'spectro' not in recs:
                if 'audio' not in recs:
                    recs = recs_load_audio(recs)
                recs['spectro'] = self.spectro(recs)
            recs['patches'] = self.patches(recs)
            recs = RecordingDF(recs)
        return recs

    # Consumes .audio on cache miss
    def spectro(self, recs: RecordingDF) -> Iterable[Melspectro]:
        """.spectro <- .audio"""
        return recs.spectro if 'spectro' in recs else self._spectros(recs)

    # Consumes .spectro on cache miss
    def patches(self, recs: RecordingDF) -> Iterable[np.ndarray]:
        """.patches <- .spectro"""
        return recs.patches if 'patches' in recs else self._patches(recs)

    # Consumes .audio on cache miss
    def _spectros(self, recs: RecordingDF) -> Iterable[Melspectro]:
        """
        rec (samples,) -> spectro (f,t)
          - f: freq indexes (Hz), mel-scaled
          - t: time indexes (s)
          - S: log power (f x t): log(X**2) where X is the (energy) unit of the audio signal
        """
        log('Features.spectros:recs', **{
            'len(recs)': len(recs),
            'sum(duration_h)': round_sig(recs.duration_s.sum() / 3600, 3),
            'sum(samples_mb)': round_sig(recs.samples_mb.sum(), 3),
            'sum(samples_n)': int(recs.samples_n.sum()),
        })
        spectros = [self._spectro(rec) for i, rec in recs.iterrows()]
        log('Features.spectros:spectros', **{
            '(f, sum(t))': (one({x.S.shape[0] for x in spectros}), sum(x.S.shape[1] for x in spectros)),
        })
        return spectros

    @cache(version=0, verbose=0, key=lambda self, rec: (rec.id, self.spectro_config))
    def _spectro(self, rec: Recording) -> Melspectro:
        config = self.spectro_config
        (_rec, audio, _x, _sample_rate) = unpack_rec(rec.audio)
        assert audio.frame_rate == config.sample_rate, 'Expected %s, got %s' % (config.sample_rate, audio)
        # TODO Filter by config.f_min
        #   - In Melspectro, try librosa.filters.mel(..., fmin=..., fmax=...) and see if that does what we want...
        spectro = Melspectro(
            audio,
            nperseg=config.frame_length,
            overlap=1 - config.hop_length / config.frame_length,
            window=config.frame_window,
            n_mels=config.f_bins,
        )
        spectro = self._spectro_denoise(spectro)
        return spectro

    # TODO Recompute denoised audio to match denoised spectro
    def _spectro_denoise(self, spectro: Melspectro) -> Melspectro:
        # Like [SP14]
        spectro = spectro.norm_rms()
        spectro = spectro.clip_below_median_per_freq()
        return spectro

    # Consumes .spectro on cache miss
    def _patches(self, recs: RecordingDF) -> Iterable[np.ndarray]:
        """rec (samples,) -> spectro (f,t) -> patch (f*p,t)"""
        log('Features.patches:recs', **{
            'len(recs)': len(recs),
            'sum(duration_h)': round_sig(recs.duration_s.sum() / 3600, 3),
            'sum(samples_mb)': round_sig(recs.samples_mb.sum(), 3),
            'sum(samples_n)': int(recs.samples_n.sum()),
            '(f, sum(t))': (one({x.S.shape[0] for x in recs.spectro}), sum(x.S.shape[1] for x in recs.spectro)),
        })
        patches = [self._patches_from_spectro(rec) for i, rec in recs.iterrows()]
        log('Features.patches:patches', **{
            '(f*p, sum(t))': (one({p.shape[0] for p in patches}), sum(p.shape[1] for p in patches)),
        })
        return patches

    @cache(version=0, verbose=0, key=lambda self, rec: (rec.id, self.patch_config))
    def _patches_from_spectro(self, rec: Recording) -> np.ndarray:
        """spectro (f,t) -> patch (f*p,t)"""
        config = self.patch_config
        (f, t, S) = rec.spectro
        return np.array([
            S[:, i:i + config.patch_length].flatten()
            for i in range(S.shape[1] - (config.patch_length - 1))
        ]).T


@dataclass
class Projection(_Base):

    k: int = 500
    variance_explained: float = .99  # (SKM default: .99)
    do_pca: bool = True              # (SKM default: True)
    pca_whiten: bool = True          # (SKM default: True)
    standardize: bool = False        # (SKM default: False)
    normalize: bool = False          # (SKM default: False)
    agg_funs: Iterable[str] = field(default_factory=lambda: [
        'mean', 'std', 'max',
    ])

    @property
    def skm_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'normalize',
            'standardize',
            'pca_whiten',
            'do_pca',
            'variance_explained',
            'k',
        ]})

    @property
    def agg_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'agg_funs',
        ]})

    @classmethod
    def load(cls, id: str) -> 'Projection':
        log('Projection.load', path=self._save_path(id))
        return joblib.load(self._save_path(id))

    def save(self, id: str):
        log('Projection.save', path=self._save_path(id))
        joblib.dump(self, self._save_path(id))

    @classmethod
    def _save_path(cls, id: str):
        return f'{data_dir}/models/projection/{id}.pkl'

    #
    # Instance methods (stateful, unsafe to cache)
    #

    @consumes_cols('patches')
    def fit(self, recs: RecordingDF) -> 'Projection':
        """patch (f*p,t) -> ()"""
        patches = list(recs.patches)
        log('Projection.fit:patches', **{
            'patches dims': (len(patches), one(set(p.shape[0] for p in patches)), sum(p.shape[1] for p in patches)),
        })
        self.skm_ = self._fit(patches, **self.skm_config)
        log('Projection.fit:skm.fit', **{
            'skm.pca.components_.shape': self.skm_.pca.components_.shape,
            'skm.D.shape': self.skm_.D.shape,
        })
        return self

    @consumes_cols('patches')
    def transform(self, recs: RecordingDF) -> np.ndarray:
        """patch (f*p,t) -> proj (k,t) -> agg (k,a) -> feat (k*a,)"""
        return self.feat(recs)

    @consumes_cols('patches')
    def feat(self, recs: RecordingDF) -> np.ndarray:
        """patch (f*p,t) -> proj (k,t) -> agg (k,a) -> feat (k*a,)"""
        patches = list(recs.patches)
        return self._transform(patches, self.skm_, self.agg_config)

    @consumes_cols('patches')
    def projs(self, recs: RecordingDF) -> np.ndarray:
        """patch (f*p,t) -> proj (k,t)"""
        patches = list(recs.patches)
        return self._projs(patches, self.skm_)

    #
    # Class methods (pure, safe to cache)
    #

    @classmethod
    @cache(version=0)
    def _fit(cls, patches, **skm_config) -> BaseEstimator:
        """patch (f*p,t) -> ()"""
        skm = SKM(**skm_config)
        skm_X = np.concatenate(patches, axis=1)  # (Entirely an skm.fit concern)
        log('Projection._fit:skm_X', **{
            'skm_X.shape': skm_X.shape,
        })
        skm.fit(skm_X)  # TODO Get sklearn to show verbose progress during .fit
        if not skm.do_pca:
            skm.pca.components_ = np.eye(skm.D.shape[0])
        return skm

    @classmethod
    def _transform(cls, patches, skm, agg_config):
        """patch (f*p,t) -> proj (k,t) -> agg (k,a) -> feat (k*a,)"""
        projs = cls._projs(patches, skm)
        aggs = cls._aggs(projs, **agg_config)
        return cls._feats_from_aggs(aggs)

    @classmethod
    @generator_to(list)
    def _projs(cls, patches, skm):
        """patch (f*p,t) -> proj (k,t)"""
        # Concat patches (along t) so we can skm.transform them all at once
        concat_patches = np.concatenate(patches, axis=1)
        concat_projs = skm.transform(concat_patches)
        # Unconcat projs (along t) to match patches
        t_shapes = [p.shape[1] for p in patches]
        t_starts = [0] + list(np.cumsum(t_shapes)[:-1])
        for patch, t_shape, t_start in zip(patches, t_shapes, t_starts):
            # Each proj should have the same t_shape and t_start as its patch
            proj = concat_projs[:, t_start:t_start + t_shape]
            yield proj

    @classmethod
    @generator_to(list)
    def _aggs(cls, projs, agg_funs):
        """proj (k,t) -> agg (k,a)"""
        for proj in projs:
            yield pd.DataFrame(OrderedDict({
                agg_fun: {
                    'mean':     lambda X: np.mean(X, axis=1),
                    'std':      lambda X: np.std(X, axis=1),
                    'min':      lambda X: np.min(X, axis=1),
                    'max':      lambda X: np.max(X, axis=1),
                    'median':   lambda X: np.median(X, axis=1),
                    'skewness': lambda X: scipy.stats.skew(X, axis=1),
                    'kurtosis': lambda X: scipy.stats.kurtosis(X, axis=1),
                    'dmean':    lambda X: np.mean(np.diff(X, axis=1), axis=1),
                    'dstd':     lambda X: np.std(np.diff(X, axis=1), axis=1),
                    'dmean2':   lambda X: np.mean(np.diff(np.diff(X, axis=1), axis=1), axis=1),
                    'dstd2':    lambda X: np.std(np.diff(np.diff(X, axis=1), axis=1), axis=1),
                }[agg_fun](proj)
                for agg_fun in agg_funs
            }))

    @classmethod
    @generator_to(list)
    def _feats_from_aggs(cls, aggs):
        """agg (k,a) -> feat (k*a,)"""
        for agg in aggs:
            yield agg.T.values.flatten()

    #
    # Plotting
    #

    def plot_proj_centroids(self, **kwargs):
        """Viz the projection patch centroids (pca -> skm)"""
        skm = self.skm_
        proj = (skm.D.T @ skm.pca.components_).T  # The total projection: pca -> skm.D
        self.plot_patches(proj, **kwargs)

    def plot_patches(self, patches, **kwargs):
        """Viz a set of patches (f*p, n) as a grid that matches figsize"""
        plot_patches(
            patches,
            self.config.patch_config.spectro_config.f_bins,
            self.config.patch_config.patch_length,
            **kwargs,
        )


@dataclass
class Search(_Base):

    n_neighbors: int = 3

    @property
    def knn_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'n_neighbors',
        ]})

    #
    # Instance methods (stateful, unsafe to cache)
    #

    # TODO Clean up naming: feats (feature vectors to fit, i.e. X) vs. to_X(recs) (instances to recall in similar_recs)

    @consumes_cols('feat')
    def fit(self, recs: RecordingDF) -> 'Search':
        """feat (k*a,) -> ()"""
        self.feats_ = np.array(list(recs.feat))  # Feature vectors to fit
        assert self.feats_.shape[0] == len(recs)  # knn.fit(X) wants X.shape == (n_samples, n_features)
        self.fit_recs_ = to_X(recs)  # Instances to recall (in predict)
        self.fit_classes_ = to_y(recs)
        log('Search.fit:feats', **{
            'recs': len(recs),
            '(n, f*p)': self.feats_.shape,
        })
        self.knn_ = self._fit(self.feats_, self.fit_classes_, **self.knn_config)
        log('Search.fit:knn', **{
            'knn.get_params': self.knn_.get_params(),
            'knn.classes_': sorted_species(self.knn_.classes_.tolist()),
            'knn.classes_.len': len(self.knn_.classes_),
        })
        return self

    @consumes_cols('feat')
    def species(self, recs: RecordingDF) -> Iterable['species']:
        """feat (k*a,) -> species (1,)"""

        # Unpack inputs (i.e. split here if we need caching)
        feats = np.array(list(recs.feat))

        # Predict: .predict
        #   - Preserve recs.index in output's index, for easy joins
        species = pd.DataFrame(index=recs.index, data={
            'species_pred': self.knn_.predict(feats),
        })
        log('Search.species', **{
            'recs': len(recs),
            '(n, k*a)': feats.shape,
            'species': len(species),
        })

        # Join test labels back onto the predictions, if given
        if 'species' in recs:
            species.insert(0, 'species_true', to_y(recs))

        return species

    @consumes_cols('feat')
    def species_probs(self, recs: RecordingDF) -> pd.DataFrame:
        """feat (k*a,) -> species (len(fit_classes_),)"""

        # Unpack inputs (i.e. split here if we need caching)
        feats = np.array(list(recs.feat))
        knn = self.knn_

        # Predict: .predict_proba
        #   - Preserve recs.index in output's index, for easy joins
        proba = knn.predict_proba(feats)
        species_probs = pd.DataFrame(index=recs.index, data=[
            {i: [p, c] for i, (c, p) in enumerate(sorted(dict(row).items(), key=lambda x: (-x[1], x[0])))}
            for i, row in pd.DataFrame(proba, columns=knn.classes_).iterrows()
        ])
        log('Search.species_probs', **{
            'recs': len(recs),
            '(n, k*a)': feats.shape,
            'knn.n_neighbors': self.knn_.n_neighbors,
            'species_probs': species_probs.shape,
        })

        # Join test labels back onto the predictions, if given
        if 'species' in recs:
            species_probs.insert(0, 'species_true', to_y(recs))

        return species_probs

    @consumes_cols('feat')
    def similar_recs(self, recs: RecordingDF, similar_n) -> pd.DataFrame:
        """feat (k*a,) -> rec (similar_n,)"""

        # Unpack inputs (i.e. split here if we need caching)
        feats = np.array(list(recs.feat))
        knn = self.knn_
        fit_recs = self.fit_recs_
        fit_classes = self.fit_classes_

        # Search: .kneighbors
        #   - Preserve recs.index in output's index, for easy joins
        (dists, fit_is) = knn.kneighbors(feats, n_neighbors=similar_n)
        similar_recs = pd.DataFrame(index=recs.index, data=[
            [
                [fit_i, dist, fit_classes[fit_i]]
                for fit_i, dist in zip(fit_is[i], dists[i])
            ]
            for i in range(len(fit_is))
        ])
        log('Search.similar_recs', **{
            'recs': len(recs),
            '(n, k*a)': feats.shape,
            'similar_recs': similar_recs.shape,
        })

        # Join test labels back onto the predictions, if given
        if 'species' in recs:
            similar_recs.insert(0, 'species_true', to_y(recs))

        return similar_recs

    @consumes_cols('species')
    def confusion_matrix(self, recs: RecordingDF) -> 'pd.DataFrame[count @ (n_species, n_species)]':
        species = self.species(recs)
        y_true = species.species_true
        y_pred = species.species_pred
        labels = sorted_species(set(y_true) | set(y_pred))
        M = pd.DataFrame(
            data=sklearn.metrics.confusion_matrix(y_true, y_pred, labels),
            index=pd.Series(labels, name='true'),
            columns=pd.Series(labels, name='pred'),
        )
        log('Search.confusion_matrix', **{
            'M': M.shape,
        })
        return M

    def coverage_error(self, recs: RecordingDF, by: str) -> 'pd.DataFrame[by, coverage_error]':
        if recs[by].dtype.name == 'category':
            recs[by] = recs[by].cat.remove_unused_categories()  # Else groupby includes all categories
        return (recs
            .groupby(by)
            .apply(lambda g: self._coverage_error(g))
            .reset_index()
            .rename(columns={0: 'coverage_error'})
        )

    @consumes_cols('species')
    def _coverage_error(self, recs: RecordingDF) -> float:
        y_true = np.array([
            (self.knn_.classes_ == rec.species).astype('int')
            for i, rec in recs.iterrows()
        ])
        y_score = self.knn_.predict_proba(list(recs.feat))
        return sklearn.metrics.coverage_error(y_true, y_score)

    #
    # Class methods (pure, safe to cache)
    #

    @classmethod
    @cache(version=0)
    def _fit(cls, feats, classes, **knn_config) -> BaseEstimator:
        """feat (k*a,) -> ()"""
        knn = KNeighborsClassifier(**knn_config)
        return knn.fit(feats, classes)

    #
    # Plotting
    #

    @consumes_cols('species')
    def plot_confusion_matrix(self, recs: RecordingDF, **kwargs):
        confusion_matrix = self.confusion_matrix(recs)
        plot_confusion_matrix(
            confusion_matrix.as_matrix(),
            labels=list(confusion_matrix.index),
            **kwargs,
        )


# TODO Update for refactor: figure out how to split this up
def _print_config(config):
    if config.verbose_config:
        _g = lambda x: '%.3g' % x
        _samples_s = config.patch_config.spectro_config.sample_rate
        _f = config.patch_config.spectro_config.f_bins
        _hop = config.patch_config.spectro_config.hop_length
        _frame = config.patch_config.spectro_config.frame_length
        _t_s = _samples_s / _hop
        _p = config.patch_config.patch_length
        _k = config.proj_skm_config.k
        _a = len(config.agg_config.funs)
        _ka = _k * _a
        log('init:config', **{
            'rec_sample_rate': '%s Hz' % _samples_s,
            'spectro_f_min': '%s Hz' % config.patch_config.spectro_config.f_min,
            '  f_max': '%s Hz' % (_samples_s // 2),
            'spectro_f_bins (f)': '%s freq bins' % _f,
            'spectro_hop_length': '%s samples (%s ms)' % (_hop, _g(1 / _samples_s * _hop * 1000)),
            'spectro_frame_length': '%s samples (%s ms)' % (_frame, _g(1 / _samples_s * _frame * 1000)),
            '  frame_overlap': '%s%% overlap (%s samples)' % (_g(100 * (1 - _hop / _frame)), _frame // 2),
            '  frames/s (t/s)': '%s samples/s' % _g(_t_s),
            'spectro_frame_window': repr(config.patch_config.spectro_config.frame_window),
            'norm': '[TODO]',  # TODO
            'patch_length (p)': '%s frames (%s ms)' % (_p, _g(_p * 1 / _samples_s * _hop * 1000)),
            'proj_skm_variance_explained': '%s%% variance' % _g(100 * config.proj_skm_config.variance_explained),
            'proj_skm_k': '%s clusters' % _k,
            'agg_funs': repr(config.agg_config.funs),
            '  a': '%s aggs' % _a,
            '  features': '%s features' % _ka,
            'class_knn_n_neighbors': config.class_knn_config.n_neighbors,
        })
        log('init:pipeline', **{
            # (Gross spacing hacks to make stuff align)
            'spectro': f'(f, t)   ({_f}, {_g(_t_s)}/s)',
            'patch  ': f'(f*p, t) ({_f}*{_p}, {_g(_t_s)}/s)',
            'proj   ': f'(k, t)   ({_k}, {_g(_t_s)}/s)',
            'agg    ': f'(k, a)   ({_k}, {_a})',
            'feat   ': f'(k*a,)   ({_ka},)',
        })


@singleton
@dataclass
class log:

    verbose: bool = True

    def __call__(self, event, **kwargs):
        """
        Simple, ad-hoc logging specialized for interactive usage
        """
        if self.verbose:
            t = datetime.utcnow().isoformat()
            t = t[:23]  # Trim micros, keep millis
            t = t.split('T')[-1]  # Trim date for now, since we're primarily interactive usage
            # Display timestamp + event on first line
            print('[%s] %s' % (t, event))
            # Display each (k,v) pair on its own line, indented
            for k, v in kwargs.items():
                v_yaml = yaml.safe_dump(json.loads(json.dumps(v)), default_flow_style=True, width=1e9)
                v_yaml = v_yaml.split('\n')[0]  # Handle documents ([1] -> '[1]\n') and scalars (1 -> '1\n...\n')
                print('  %s: %s' % (k, v_yaml))
