import copy
from datetime import datetime
import itertools
import json
from typing import Iterable, Union

from addict import Dict  # TODO Switch back to AttrDict since Dict().x doesn't fail, and PR an AttrDict.to_dict()
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


class Model:
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

    #
    # Instance methods (stateful)
    #

    def __init__(
        self,
        rec_sample_rate=22050,
        spectro_f_min=1000,
        spectro_f_bins=40,
        spectro_hop_length=256,
        spectro_frame_length=512,
        spectro_frame_window='hann',
        patch_length=4,
        proj_skm_variance_explained=.99,
        proj_skm_k=500,
        proj_skm_config={},
        agg_funs=['mean', 'std', 'max'],
        class_knn_n_neighbors=3,
        verbose_config=False,
    ):
        self.config = Dict(
            patch_config=dict(
                spectro_config=dict(
                    sample_rate=rec_sample_rate,
                    f_min=spectro_f_min,
                    f_bins=spectro_f_bins,
                    hop_length=spectro_hop_length,
                    frame_length=spectro_frame_length,
                    frame_window=spectro_frame_window,
                ),
                patch_length=patch_length,
            ),
            proj_skm_config={
                'variance_explained': proj_skm_variance_explained,
                'k': proj_skm_k,
                **proj_skm_config,
            },
            agg_config=dict(
                funs=agg_funs,
            ),
            class_knn_config=dict(
                n_neighbors=class_knn_n_neighbors,
            ),
            verbose_config=verbose_config,
        )
        self._print_config(self.config)

    def fit_proj(self, recs: pd.DataFrame):
        """rec (samples,) -> spectro (f,t) -> patch (f*p,t) -> [skm.fit]"""
        self.proj_recs_ = self.to_X(recs)
        self.proj_patches_ = self._patches(self.proj_recs_, **self.config.patch_config)
        self.proj_skm_ = SKM(**self.config.proj_skm_config)
        skm_X = np.concatenate(self.proj_patches_, axis=1)  # (Entirely an skm.fit concern)
        self._log('fit_proj:skm_X', **{
            'skm_X.shape': skm_X.shape,
        })
        self.proj_skm_ = self._skm_fit(self.proj_skm_, skm_X)
        if not self.config.proj_skm_config.get('do_pca', True):
            self.proj_skm_.pca.components_ = np.eye(self.proj_skm_.D.shape[0])
        self._log('fit_proj:skm.fit', **{
            'skm.pca.components_.shape': self.proj_skm_.pca.components_.shape,
            'skm.D.shape': self.proj_skm_.D.shape,
        })
        return self

    def fit_class(self, recs: pd.DataFrame):
        """patch (f*p,t) -> [skm.transform] -> proj (k,t) -> agg (k,a) -> feat (k*a,) -> [knn.fit]"""
        self.class_recs_ = self.to_X(recs)
        self.class_classes_ = self.to_y(recs)
        self.class_patches_ = self._patches(self.class_recs_, **self.config.patch_config)
        self.class_feats_ = self._feats(self.class_patches_, self.proj_skm_, self.config.agg_config)  # (skm.transform)
        self._log('fit_class:knn_Xy', **{
            '(f*p, t)': [p.shape for p in self.class_patches_],
        })
        self.class_knn_ = (
            KNeighborsClassifier(**self.config.class_knn_config)
            .fit(self.class_feats_, self.class_classes_)
        )
        self._log('fit_class:knn', **{
            'knn.get_params': self.class_knn_.get_params(),
            'knn.classes_': self.class_knn_.classes_.tolist(),
        })
        return self

    def predict(self, recs: pd.DataFrame, type: Union['classes', 'kneighbors']) -> pd.DataFrame:
        """
        rec (samples,) -> spectro (f,t) -> patch (f*p,t) -> [skm.transform]
            -> proj (k,t) -> agg (k,a) -> feat (k*a,) -> [knn.predict]
        """
        return self._predict(
            self.to_X(recs),
            type,
            self.proj_skm_,
            self.class_knn_,
            self.class_recs_,
            self.class_classes_,
            self.config.patch_config,
            self.config.agg_config,
        )

    def test(self, recs: pd.DataFrame, type: Union['classes', 'kneighbors']) -> pd.DataFrame:
        """Predict, and then add test labels back to the predictions"""
        return pd.concat(axis=1, objs=[
            pd.DataFrame({'y': self.to_y(recs)}),
            self.predict(recs, type),
        ]).T

    #
    # User-friendly instance methods that we don't rely on internally
    #

    def spectros(self, recs: pd.DataFrame) -> Iterable[Melspectro]:
        return self._spectros(self.to_X(recs), **self.config.patch_config.spectro_config)

    def patches(self, recs: pd.DataFrame) -> Iterable[np.ndarray]:
        return self._patches(self.to_X(recs), **self.config.patch_config)

    #
    # Class methods (not stateful)
    #

    @classmethod
    def to_X(cls, recs: pd.DataFrame) -> List[Recording]:
        return df_to_recs(recs)

    @classmethod
    def to_y(cls, recs: pd.DataFrame) -> np.ndarray:
        return np.array(recs.species)

    @classmethod
    def _print_config(cls, config):
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
            cls._log('init:config', **{
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
            cls._log('init:pipeline', **{
                # (Gross spacing hacks to make stuff align)
                'spectro': f'(f, t)   ({_f}, {_g(_t_s)}/s)',
                'patch  ': f'(f*p, t) ({_f}*{_p}, {_g(_t_s)}/s)',
                'proj   ': f'(k, t)   ({_k}, {_g(_t_s)}/s)',
                'agg    ': f'(k, a)   ({_k}, {_a})',
                'feat   ': f'(k*a,)   ({_ka},)',
            })

    @classmethod
    def _patches(cls, recs, spectro_config, patch_length):
        """rec (samples,) -> spectro (f,t) -> patch (f*p,t)"""
        cls._log('patches:recs', **{
            'len(recs)': len(recs),
            'duration_s': [r.duration_s for r in recs],
            'sum(duration_s)': sum(r.duration_s for r in recs),
            '(samples,)': [int(r.audio.frame_count()) for r in recs],
            'sum(samples)': sum(int(r.audio.frame_count()) for r in recs),
        })
        patches = cls._patches_from_recs(recs, spectro_config, patch_length)
        cls._log('patches:patches', **{
            '(f*p, t)': [p.shape for p in patches],
            '(f*p, sum(t))': (one({p.shape[0] for p in patches}), sum(p.shape[1] for p in patches)),
        })
        return patches

    @classmethod
    @cache(version=0)
    def _patches_from_recs(cls, recs, spectro_config, patch_length):
        """rec (samples,) -> spectro (f,t) -> patch (f*p,t)"""
        spectros = cls._spectros(recs, **spectro_config)
        cls._log('patches:spectros', **{
            '(f, t)': [x.S.shape for x in spectros],
            '(f, sum(t))': (one({x.S.shape[0] for x in spectros}), sum(x.S.shape[1] for x in spectros)),
        })
        patches = cls._patches_from_spectros(spectros, patch_length)
        return patches

    @classmethod
    def _spectros(cls, recs, **spectro_config) -> List[Melspectro]:
        """
        rec (samples,) -> spectro (f,t)
          - f: freq indexes (Hz), mel-scaled
          - t: time indexes (s)
          - S: log power (f x t): log(X**2) where X is the (energy) unit of the audio signal
        """
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

    # TODO Recompute audio after filtering spectro (e.g. playing audio should sound denoised)
    @classmethod
    def _spectro_denoise(cls, spectro: Melspectro) -> Melspectro:
        # Like [SP14]
        spectro = spectro.norm_rms()
        spectro = spectro.clip_below_median_per_freq()
        return spectro

    @classmethod
    @cache(version=0)
    @generator_to(list)
    def _patches_from_spectros(cls, spectros, patch_length):
        """spectro (f,t) -> patch (f*p,t)"""
        for spectro in spectros:
            (f, t, S) = spectro
            patch = np.array([
                S[:, i:i + patch_length].flatten()
                for i in range(S.shape[1] - (patch_length - 1))
            ]).T
            yield patch

    @classmethod
    @cache(version=0)
    def _skm_fit(cls, skm, X, **kwargs):
        """Pure wrapper around skm.fit for caching"""
        skm = copy.copy(skm)
        skm.fit(X, **kwargs)
        return skm

    @classmethod
    def _predict(cls, recs, type, proj_skm, class_knn, class_recs, class_classes, patch_config, agg_config) -> pd.DataFrame:
        """
        rec (samples,) -> spectro (f,t) -> patch (f*p,t) -> [skm.transform]
            -> proj (k,t) -> agg (k,a) -> feat (k*a,) -> [knn.predict]
        """
        # _transform_proj: recs -> aggs
        patches = cls._patches(recs, **patch_config)
        feats = cls._feats(patches, proj_skm, agg_config)  # (skm.transform)
        # _predict_class_*: feats -> classes/kneighbors
        return cls._predict_class(feats, type, class_knn, class_recs, class_classes)

    @classmethod
    def _predict_class(cls, feats, type, class_knn, class_recs, class_classes) -> pd.DataFrame:
        if type == 'classes':
            return cls._predict_class_classes(feats, class_knn)  # (knn.predict_proba)
        elif type == 'kneighbors':
            return cls._predict_class_kneighbors(feats, class_knn, class_recs, class_classes)  # (knn.kneighbors)
        else:
            raise ValueError(f"type[{type}] must be one of: 'classes', 'kneighbors'")

    @classmethod
    def _predict_class_classes(cls, feats, class_knn) -> pd.DataFrame:
        """agg (k,a) -> class (test_n, class_n)"""
        classes = class_knn.classes_
        proba = class_knn.predict_proba(feats)
        classes_df = pd.DataFrame([
            {i: [p, c] for i, (c, p) in enumerate(sorted(dict(row).items(), key=lambda x: (-x[1], x[0])))}
            for i, row in pd.DataFrame(proba, columns=classes).iterrows()
        ])
        cls._log('predict:classes', **{
            # TODO Put something useful here that isn't too big
        })
        return classes_df

    @classmethod
    def _predict_class_kneighbors(cls, feats, class_knn, class_recs, class_classes) -> pd.DataFrame:
        """agg (k,a) -> neighbor (test_n, train_n)"""
        fit_recs = class_recs
        fit_classes = class_classes
        (dists, fit_is) = class_knn.kneighbors(feats, n_neighbors=len(fit_recs))
        kneighbors_df = pd.DataFrame([
            [
                [fit_i, dist, fit_classes[fit_i]]
                for fit_i, dist in zip(fit_is[i], dists[i])
            ]
            for i in range(len(fit_is))
        ])
        cls._log('predict:kneighbors', **{
            # TODO Put something useful here that isn't too big
        })
        return kneighbors_df

    @classmethod
    def _feats(cls, patches, proj_skm, agg_config):
        """patch (f*p,t) -> [skm.transform] -> agg (k,a) -> feat (k*a,)"""
        projs = cls._transform_proj(proj_skm, patches)  # (skm.transform)
        aggs = cls._aggs(projs, **agg_config)
        feats = cls._feats_from_aggs(aggs)
        return feats

    @classmethod
    @generator_to(list)
    def _transform_proj(cls, proj_skm, patches):
        """patch (f*p,t) -> proj (k,t)"""
        # Concat patches (along t) so we can skm.transform them all at once
        concat_patches = np.concatenate(patches, axis=1)
        concat_projs = proj_skm.transform(concat_patches)
        # Unconcat projs (along t) to match patches
        t_shapes = [p.shape[1] for p in patches]
        t_starts = [0] + list(np.cumsum(t_shapes)[:-1])
        for patch, t_shape, t_start in zip(patches, t_shapes, t_starts):
            # Each proj should have the same t_shape and t_start as its patch
            proj = concat_projs[:, t_start:t_start + t_shape]
            yield proj

    @classmethod
    @generator_to(list)
    def _aggs(cls, projs, funs):
        """proj (k,t) -> agg (k,a)"""
        for proj in projs:
            yield pd.DataFrame(OrderedDict({
                fun: {
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
                }[fun](proj)
                for fun in funs
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
        skm = self.proj_skm_
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

    #
    # Logging
    #

    @classmethod
    def _log(cls, event, **kwargs):
        """
        Simple, ad-hoc logging specialized for interactive usage
        """
        if cls.verbose:
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

    verbose = True
