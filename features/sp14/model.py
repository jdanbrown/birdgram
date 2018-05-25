import copy
from datetime import datetime
import itertools
import json
from typing import Iterable, Union

from attrdict import AttrDict
import dataclasses
from dataclasses import dataclass, field
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import yaml

from cache import cache, cache_lambda, cache_pure_method
from datasets import *
from features import *
from load import *
import metadata
from sp14.skm import SKM
from util import *
from viz import *


# TODO Re-home this docstring
"""
Config
    |                             | defaults      | [SP14]         | [SBF16]       |
    |-----------------------------|---------------|----------------|---------------|
    | rec_sample_rate             | 22050         | 44100          | 22050
    | spectro_f_min               | 1000          | 500            | 2000
    |   f_max                     | 11025         | 22050          | 11025
    | spectro_f_bins (f)          | 40            | 40             | 40
    | spectro_hop_length          | 256 (12ms)    | 1024 (23ms)    | 32 (1.5ms)
    | spectro_frame_length        | 512 (23ms)    | 1024 (23ms)    | 256 (12ms)
    |   frame_overlap             | .5            | 0              | .875
    |   frames/s (t/s)            | 86            | 43             | 689
    | spectro_frame_window        | hann          | hamming        | hann
    | norm                        | [TODO]        | RMS+median     | [TODO]
    | patch_length (p)            | 4 (46ms)      | ~4 (~93ms)     | ~16 (~22ms)
    | proj_skm_variance_explained | .99           | —              | .99
    | proj_skm_k (k)              | 500           | 500            | ~512
    | agg_funs                    | μ,σ,max       | μ,σ            | ~μ,σ,max
    |   a                         | 3             | 2              | ~3
    |   features                  | 1500          | 1000           | ~1536

Pipeline
    | rec     | (samples,) | (22050/s,)   | (44100/s)     | (22050/s,)
    | spectro | (f, t)     | (40, 86/s)   | (40, 43/s)    | (40, 689/s)
    | patch   | (f*p, t)   | (40*4, 86/s) | (40*~4, 43/s) | (40*~16, 689/s)
    | proj    | (k, t)     | (500, 86/s)  | (500, 43/s)   | (~512, 689/s)
    | agg     | (k, a)     | (500, 3)     | (500, 2)      | (~512, ~3)
    | feat    | (k*a,)     | (1500,)      | (1000,)       | (~1536,)
"""


def to_X(recs: pd.DataFrame) -> List[AttrDict]:
    return [AttrDict(row) for i, row in recs.iterrows()]


def to_y(recs: pd.DataFrame) -> np.ndarray:
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

    #
    # Instance methods (stateful)
    #

    def spectros(self, recs: pd.DataFrame) -> Iterable[Melspectro]:
        assert 'audio' in recs
        return self._spectros(to_X(recs), **self.spectro_config)

    def patches(self, recs: pd.DataFrame) -> Iterable[np.ndarray]:
        assert 'spectro' in recs
        return self._patches(to_X(recs), **self.patch_config)

    #
    # Class methods (not stateful)
    #

    @classmethod
    def _spectros(cls, recs, **spectro_config) -> List[Melspectro]:
        """
        rec (samples,) -> spectro (f,t)
          - f: freq indexes (Hz), mel-scaled
          - t: time indexes (s)
          - S: log power (f x t): log(X**2) where X is the (energy) unit of the audio signal
        """
        assert all('audio' in rec for rec in recs)
        # Cache per dataset group
        #   - Sort by (dataset, name) -> apply _spectros_cache_block per group -> unsort back to original order
        i_recs_sorted = sorted(enumerate(recs), key=lambda i_rec: (i_rec[1].dataset, i_rec[1].name))
        recs_sorted = [rec for i, rec in i_recs_sorted]
        unsort = {i_orig: i_sorted for i_sorted, (i_orig, rec) in enumerate(i_recs_sorted)}
        spectros_sorted = list(flatten(
            cls._spectros_cache_block(
                # Strip recs to audios, to eliminate spurious information in cache keys (e.g. species_query)
                [rec.audio for rec in recs_sorted_for_dataset],
                **spectro_config,
            )
            for dataset, recs_sorted_for_dataset in itertools.groupby(recs_sorted, lambda rec: rec.dataset)
        ))
        spectros = [spectros_sorted[unsort[i_orig]] for i_orig, rec in enumerate(recs)]
        return spectros

    @classmethod
    @cache(version=0)
    def _spectros_cache_block(cls, audios, **spectro_config) -> Iterable[Melspectro]:
        return [cls._spectro(audio, **spectro_config) for audio in audios]

    @classmethod
    def _spectro(cls, audio, sample_rate, f_min, frame_length, hop_length, frame_window, f_bins) -> Melspectro:
        (_rec, audio, _x, _sample_rate) = unpack_rec(audio)
        assert audio.frame_rate == sample_rate, 'Expected %s, got %s' % (sample_rate, audio)
        # TODO Filter by f_min
        #   - In Melspectro, try librosa.filters.mel(..., fmin=..., fmax=...) and see if that does what we want...
        spectro = Melspectro(
            audio,
            nperseg=frame_length,
            overlap=1 - hop_length / frame_length,
            window=frame_window,
            n_mels=f_bins,
        )
        spectro = cls._spectro_denoise(spectro)
        return spectro

    # TODO Recompute denoised audio to match denoised spectro
    @classmethod
    def _spectro_denoise(cls, spectro: Melspectro) -> Melspectro:
        # Like [SP14]
        spectro = spectro.norm_rms()
        spectro = spectro.clip_below_median_per_freq()
        return spectro

    @classmethod
    def _patches(cls, recs, patch_length):
        """rec (samples,) -> spectro (f,t) -> patch (f*p,t)"""
        assert all('spectro' in rec for rec in recs)
        log('Features.patches:recs', **{
            'len(recs)': len(recs),
            'duration_s': [r.duration_s for r in recs],
            'sum(duration_s)': sum(r.duration_s for r in recs),
            '(samples,)': [int(r.audio.frame_count()) for r in recs],
            'sum(samples)': sum(int(r.audio.frame_count()) for r in recs),
        })
        patches = cls._patches_from_spectros([rec.spectro for rec in recs], patch_length)
        log('Features.patches:patches', **{
            '(f*p, t)': [p.shape for p in patches],
            '(f*p, sum(t))': (one({p.shape[0] for p in patches}), sum(p.shape[1] for p in patches)),
        })
        return patches

    @classmethod
    @cache(version=0)
    @generator_to(list)
    def _patches_from_spectros(cls, spectros: List[Melspectro], patch_length):
        """spectro (f,t) -> patch (f*p,t)"""
        log('Features.patches:spectros', **{
            '(f, t)': [x.S.shape for x in spectros],
            '(f, sum(t))': (one({x.S.shape[0] for x in spectros}), sum(x.S.shape[1] for x in spectros)),
        })
        for spectro in spectros:
            (f, t, S) = spectro
            patch = np.array([
                S[:, i:i + patch_length].flatten()
                for i in range(S.shape[1] - (patch_length - 1))
            ]).T
            yield patch


@dataclass
class Projection(_Base):

    skm_config: AttrDict = field(default_factory=lambda: AttrDict(
        variance_explained=.99,
        k=500,
    ))
    agg_funs: List[str] = field(default_factory=lambda: [
        'mean', 'std', 'max',
    ])

    @property
    def agg_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'agg_funs',
        ]})

    #
    # Instance methods (stateful)
    #

    def fit(self, recs: pd.DataFrame):
        """patch (f*p,t) -> ()"""
        assert 'patches' in recs
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

    def transform(self, recs: pd.DataFrame) -> np.ndarray:
        """patch (f*p,t) -> proj (k,t) -> agg (k,a) -> feat (k*a,)"""
        assert 'patches' in recs
        patches = list(recs.patches)
        return self._transform(patches, self.skm_, self.agg_config)

    def projs(self, recs: pd.DataFrame) -> np.ndarray:
        """patch (f*p,t) -> proj (k,t)"""
        assert 'patches' in recs
        patches = list(recs.patches)
        return self._projs(patches, self.skm_)

    #
    # Class methods (not stateful)
    #

    @classmethod
    @cache(version=0)
    def _fit(cls, patches, **skm_config):
        """patch (f*p,t) -> ()"""
        skm = SKM(**skm_config)
        skm_X = np.concatenate(patches, axis=1)  # (Entirely an skm.fit concern)
        log('Projection._fit:skm_X', **{
            'skm_X.shape': skm_X.shape,
        })
        skm.fit(skm_X)
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

    knn_config: AttrDict = field(default_factory=lambda: AttrDict(
        n_neighbors=3,
    ))

    #
    # Instance methods (stateful)
    #

    # TODO Straighten out feats (feature vectors to fit, i.e. X) vs. to_X(recs) (instances to recall)

    def fit(self, recs: pd.DataFrame):
        """feat (k*a,) -> ()"""
        assert 'feat' in recs
        self.feats_ = np.array(list(recs.feat))  # Feature vectors to fit
        self.recs_ = to_X(recs)  # Instances to recall (in predict)
        self.classes_ = to_y(recs)
        log('Search.fit:feats', **{
            '(f*p, t)': [f.shape for f in self.feats_],
        })
        self.knn_ = self._fit(self.feats_, self.classes_, **self.knn_config)
        log('Search.fit:knn', **{
            'knn.get_params': self.knn_.get_params(),
            'knn.classes_': self.knn_.classes_.tolist(),
        })
        return self

    def predict(self, recs: pd.DataFrame, type: Union['classes', 'kneighbors']) -> pd.DataFrame:
        """feat (k*a,) -> preds"""
        assert 'feat' in recs
        return self._predict(
            np.array(list(recs.feat)),
            type,
            self.knn_,
            self.recs_,
            self.classes_,
        )

    def test(self, recs: pd.DataFrame, type: Union['classes', 'kneighbors']) -> pd.DataFrame:
        """Predict, and then add test labels back to the predictions"""
        assert 'feat' in recs
        return pd.concat(axis=1, objs=[
            pd.DataFrame({'y': to_y(recs)}),
            self.predict(recs, type),
        ]).T

    #
    # Class methods (not stateful)
    #

    @classmethod
    @cache(version=0)
    def _fit(cls, feats, classes, **knn_config):
        """feat (k*a,) -> ()"""
        knn = KNeighborsClassifier(**knn_config)
        return knn.fit(feats, classes)

    @classmethod
    def _predict(cls, feats, type, knn, knn_recs, knn_classes) -> pd.DataFrame:
        """feat (k*a,) -> class | neighbor"""
        if type == 'classes':
            return cls._predict_class_classes(feats, knn)  # (knn.predict_proba)
        elif type == 'kneighbors':
            return cls._predict_class_kneighbors(feats, knn, knn_recs, knn_classes)  # (knn.kneighbors)
        else:
            raise ValueError(f"type[{type}] must be one of: 'classes', 'kneighbors'")

    @classmethod
    def _predict_class_classes(cls, feats, knn) -> pd.DataFrame:
        """feat (k*a,) -> class (test_n, class_n)"""
        classes = knn.classes_
        proba = knn.predict_proba(feats)
        classes_df = pd.DataFrame([
            {i: [p, c] for i, (c, p) in enumerate(sorted(dict(row).items(), key=lambda x: (-x[1], x[0])))}
            for i, row in pd.DataFrame(proba, columns=classes).iterrows()
        ])
        log('Search.predict:classes', **{
            # TODO Put something useful here that isn't too big
        })
        return classes_df

    @classmethod
    def _predict_class_kneighbors(cls, feats, knn, knn_recs, knn_classes) -> pd.DataFrame:
        """feat (k*a,) -> neighbor (test_n, train_n)"""
        knn_recs = knn_recs
        knn_classes = knn_classes
        (dists, fit_is) = knn.kneighbors(feats, n_neighbors=len(knn_recs))
        kneighbors_df = pd.DataFrame([
            [
                [fit_i, dist, knn_classes[fit_i]]
                for fit_i, dist in zip(fit_is[i], dists[i])
            ]
            for i in range(len(fit_is))
        ])
        log('Search.predict:kneighbors', **{
            # TODO Put something useful here that isn't too big
        })
        return kneighbors_df


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
