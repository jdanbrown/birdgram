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
    | patches | (f*p, t)   | (40*4, 86/s) | (40*~4, 43/s) | (40*~16, 689/s)
    | proj    | (k, t)     | (500, 86/s)  | (500, 43/s)   | (~512, 689/s)
    | agg     | (k, a)     | (500, 3)     | (500, 2)      | (~512, ~3)
    | feat    | (k*a,)     | (1500,)      | (1000,)       | (~1536,)
"""

import copy
from datetime import datetime
from frozendict import frozendict
from functools import partial
import hashlib
import itertools
import json
from typing import Iterable, Mapping, Optional, Union
import uuid

from attrdict import AttrDict
import dask
from dataclasses import dataclass, field
import joblib
import lightgbm as lgb  # For Search.cls
import numpy as np
from potoo.numpy import np_sample, np_sample_stratified
from potoo.util import round_sig
import sklearn as sk
import sklearn.ensemble
import sklearn.multiclass
import sklearn.pipeline
import sklearn.naive_bayes
import yaml
import xgboost as xgb

from cache import cache
from constants import data_dir
from datasets import *
from datatypes import Audio, RecordingDF
from features import *
from load import *
import metadata
from metadata import sorted_species
from sk_util import *
from sp14.skm import SKM
from util import *
from viz import *
import xgb_sklearn_hack


@dataclass
class Features(DataclassConfig):

    # Dependencies [TODO Simplify with dataclasses.InitVar?]
    load: Load = Load()

    # Config
    sample_rate: int = 22050
    f_min: int = 1000
    f_bins: int = 40
    hop_length: int = 256
    frame_length: int = 512
    frame_window: str = 'hann'
    patch_length: int = 4

    @property
    def deps(self) -> AttrDict:
        return AttrDict({k: getattr(self, k) for k in [
            'load',
        ]})

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

    # Optimal performance varies with cache hit/miss rate:
    #   - 100% cache miss: row-major-synchronous[7.9s], row-major-threads[8.3s], col-major-defaults[5.2s]
    #   - 100% cache hit:  row-major-synchronous[.61s], row-major-threads[.39s], col-major-defaults[1.2s]
    #   - By default we choose col-major-defaults
    #   - When the user wants to assume near-100% cache hits, they can manually `dask_opts`/`features.patches`
    @short_circuit(lambda self, recs, **kwargs: recs if 'patches' in recs else None)
    def transform(self, recs: RecordingDF, drop_spectro=True) -> RecordingDF:
        """Adds .patches (f*p,t)"""
        # Performance (100 peterson recs):
        #   - Row-major: defaults[5.5s], no_dask[5.6s], synchronous[5.8s], threads[6.5s], processes[?]
        #   - Col-major: defaults[3.1s], no_dask[5.4s], synchronous[5.9s], threads[6.0s], processes[?]
        #   - Bottlenecks (col-major no_dask):
        #          ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #             100    1.068    0.011    2.707    0.027 spectral.py:1681(_fft_helper)
        #             100    1.022    0.010    1.022    0.010 {built-in method numpy.fft.fftpack_lite.rfftf}
        #            4496    0.590    0.000    0.590    0.000 {built-in method numpy.core.multiarray.array}
        #          237664    0.349    0.000    0.349    0.000 {method 'flatten' of 'numpy.ndarray' objects}
        #             100    0.286    0.003    3.318    0.033 spectral.py:563(spectrogram)
        return RecordingDF(recs
            # Compute col-major instead of row-major, since they each perform best with different schedulers
            .assign(spectro=self.spectro)  # Fastest with 'threads'
            .assign(patches=self.patches)  # Fastest with 'synchronous'
            # Drop .spectro by default to minimize mem usage (pass drop_spectro=True, or read the rows from cache)
            .drop(columns=['spectro'] if drop_spectro else [])
        )

    @short_circuit(lambda self, recs: recs.get('patches'))
    def patches(self, recs: RecordingDF) -> Column['np.ndarray[(f*p,t)]']:
        """.patches (f*p,t) <- .spectro (f,t)"""
        log.debug('Features.patches:in', **{
            'len(recs)': len(recs),
            'len(recs) per dataset': recs.get('dataset', pd.Series()).value_counts().to_dict(),
            'sum(duration_h)': round_sig(sum(recs.get('duration_s', [])) / 3600, 3),
            'sum(samples_mb)': round_sig(sum(recs.get('samples_mb', [])), 3),
            'sum(samples_n)': int(sum(recs.get('samples_n', []))),
            '(f, sum(t))': recs.get('spectro', pd.Series()).pipe(lambda xs: (
                list({x.S.shape[0] for x in xs}),
                sum(x.S.shape[1] for x in xs),
            ))
        })
        # Performance (100 peterson recs):
        #   - Scheduler: no_dask[.94s], synchronous[.97s], threads[4.2s], processes[~100s]
        #   - Bottlenecks (no_dask):
        #          ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #          237664    0.397    0.000    0.397    0.000 {method 'flatten' of 'numpy.ndarray' objects}
        #         546/542    0.219    0.000    0.219    0.000 {built-in method numpy.core.multiarray.array}
        #             100    0.153    0.002    0.550    0.005 model.py:170(<listcomp>)
        #             100    0.069    0.001    0.862    0.009 model.py:163(_patches)
        #               1    0.042    0.042    0.944    0.944 <string>:1(<module>)
        patches = map_progress(self._patches, df_rows(recs), use='dask', scheduler='synchronous')
        log.debug('Features.patches:out', **{
            '(f*p, sum(t))': (one({p.shape[0] for p in patches}), sum(p.shape[1] for p in patches)),
        })
        return patches

    @short_circuit(lambda self, recs, **kwargs: recs.get('spectro'))
    def spectro(self, recs: RecordingDF, **kwargs) -> Column[Melspectro]:
        """.spectro (f,t) <- .audio (samples,)"""
        log.debug('Features.spectros:in', **{
            'len(recs)': len(recs),
            'len(recs) per dataset': recs.get('dataset', pd.Series()).value_counts().to_dict(),
            'sum(duration_h)': round_sig(sum(recs.get('duration_s', [])) / 3600, 3),
            'sum(samples_mb)': round_sig(sum(recs.get('samples_mb', [])), 3),
            'sum(samples_n)': int(sum(recs.get('samples_n', []))),
        })
        # Performance (100 peterson recs):
        #   - Scheduler: no_dask[4.4s], synchronous[4.7s], threads[2.2s], processes[~130s]
        #   - Bottlenecks (no_dask):
        #          ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #             100    1.031    0.010    2.633    0.026 spectral.py:1681(_fft_helper)
        #             100    0.996    0.010    0.996    0.010 {built-in method numpy.fft.fftpack_lite.rfftf}
        #       3843/3839    0.333    0.000    0.334    0.000 {built-in method numpy.core.multiarray.array}
        #             100    0.285    0.003    3.228    0.032 spectral.py:563(spectrogram)
        #             100    0.284    0.003    0.381    0.004 signaltools.py:2464(detrend)
        spectros = map_progress(self._spectro, df_rows(recs), **{
            **dict(
                use='dask', scheduler='threads',
            ),
            **kwargs,
        })
        log.debug('Features.spectros:out', **{
            '(f, sum(t))': (one({x.S.shape[0] for x in spectros}), sum(x.S.shape[1] for x in spectros)),
        })
        return spectros

    @short_circuit(lambda self, rec: rec.get('patches'))
    # @cache(version=0, key=lambda self, rec: (rec.id, self.patch_config, self.spectro_config, self.deps))  # TODO After birdclef
    def _patches(self, rec: Row) -> 'np.ndarray[(f*p, t)]':
        """spectro (f,t) -> patch (f*p,t)"""
        (f, t, S) = self._spectro(rec)  # Cached
        p = self.patch_config.patch_length
        return np.array([
            S[:, i:i+p].flatten()
            for i in range(S.shape[1] - (p - 1))
        ]).T

    @short_circuit(lambda self, rec: rec.get('spectro'))
    # @cache(version=0, key=lambda self, rec: (rec.id, self.spectro_config, self.deps))  # TODO After birdclef
    def _spectro(self, rec: Row) -> Melspectro:
        """
        .spectro (f,t) <- .audio (samples,)
          - f: freq indexes (Hz), mel-scaled
          - t: time indexes (s)
          - S: log power (f x t): log(X**2) where X is the (energy) unit of the audio signal
        """
        audio = self.load._audio(rec)  # Pull
        c = self.spectro_config
        (_rec, audio, _x, _sample_rate) = unpack_rec(audio)
        assert audio.frame_rate == c.sample_rate, 'Unsupported sample_rate[%s != %s] for audio[%s]' % (
            audio.frame_rate, c.sample_rate, audio,
        )
        # TODO Filter by c.f_min
        #   - In Melspectro, try librosa.filters.mel(..., fmin=..., fmax=...) and see if that does what we want...
        spectro = Melspectro(
            audio,
            nperseg=c.frame_length,
            overlap=1 - c.hop_length / c.frame_length,
            window=c.frame_window,
            n_mels=c.f_bins,
        )
        spectro = self._spectro_denoise(spectro)
        return spectro

    # TODO Add .audio_denoised so we can hear what the denoised spectro sounds like
    def _spectro_denoise(self, spectro: Melspectro) -> Melspectro:
        """Denoise like [SP14]"""
        spectro = spectro.norm_rms()
        spectro = spectro.clip_below_median_per_freq()
        return spectro


@dataclass
class Projection(DataclassConfig):

    # Dependencies [TODO Simplify with dataclasses.InitVar?]
    features: Features = Features()

    # Config
    skm_fit_max_t: int = 600_000
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
    def deps(self) -> AttrDict:
        return AttrDict({k: getattr(self, k) for k in [
            'features',
        ]})

    @property
    def safety_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'skm_fit_max_t',
        ]})

    @property
    def skm_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'k',
            'variance_explained',
            'do_pca',
            'pca_whiten',
            'standardize',
            'normalize',
        ]})

    @property
    def agg_config(self) -> AttrDict:
        return AttrDict({k: v for k, v in self.config.items() if k in [
            'agg_funs',
        ]})

    @classmethod
    def load(cls, id: str, **override_attrs) -> 'cls':
        log.debug('Projection.load', path=cls._save_path(id))
        # Save/load the attrs, not the instance, so we don't preserve class objects with outdated code
        saved_attrs = joblib.load(cls._save_path(id))
        projection = Projection()
        projection.__dict__.update(saved_attrs)
        projection.__dict__.update(override_attrs)
        return projection

    def save(self, basename: str):
        nonce = sha1hex(str(uuid.uuid4()))[:7]
        model_id = f'{basename}-{nonce}'
        self._save(model_id)
        return model_id

    def _save(self, id: str):
        log.debug('Projection.save', path=self._save_path(id))
        # Save/load the attrs, not the instance, so we don't preserve class objects with outdated code
        joblib.dump(self.__dict__, self._save_path(id))

    @classmethod
    def _save_path(cls, id: str):
        return ensure_parent_dir(f'{data_dir}/models/projection/{id}.pkl')

    def fit(self, recs: RecordingDF) -> 'self':
        """skm_ <- .patch (f*p,t)"""
        recs = self.features.transform(recs)  # Pull
        log.debug('Projection.fit:in', **{
            'patches': (
                len(recs.patches),
                one(set(p.shape[0] for p in recs.patches)),
                sum(p.shape[1] for p in recs.patches),
            ),
        })
        self.skm_ = self._fit(recs)
        log.debug('Projection.fit:out', **{
            'skm.pca.components_': self.skm_.pca.components_.shape,
            'skm.D': self.skm_.D.shape,
        })
        return self

    @cache(version=0, verbose=100, key=lambda self, recs: (recs.id, self.skm_config, self.deps))
    def _fit(self, recs: RecordingDF) -> SKM:
        """skm <- .patch (f*p,t)"""
        skm = SKM(**self.skm_config)
        skm_X = np.concatenate(list(recs.patches), axis=1)  # (Entirely an skm.fit concern)
        log.debug('Projection._fit:in', **{
            'skm_X.shape': skm_X.shape,
        })
        if skm_X.shape[1] > self.safety_config.skm_fit_max_t:
            raise ValueError('Maybe not memory safe: skm_X t[%s] > skm_fit_max_t[%s]' % (
                skm_X.shape[1], self.safety_config.skm_fit_max_t,
            ))
        skm.fit(skm_X)  # TODO Get sklearn to show verbose progress during .fit
        if not skm.do_pca:
            skm.pca.components_ = np.eye(skm.D.shape[0])
        return skm

    @short_circuit(lambda self, recs, **kwargs: recs if 'feat' in recs else None)
    def transform(self, recs: RecordingDF, **kwargs) -> RecordingDF:
        """Adds .feat (k*a,)"""
        # Performance (300 peterson recs):
        #   - Row-major: no_dask[4.1s], synchronous[4.4s], threads[3.5s], processes[?]
        #   - Col-major: no_dask[4.8s], synchronous[5.4s], threads[3.9s], processes[?]
        #   - Bottlenecks (row-major no_dask):
        #          ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #             600    1.340    0.002    1.340    0.002 {built-in method numpy.core.multiarray.dot}
        #            1508    0.805    0.001    0.805    0.001 {method 'reduce' of 'numpy.ufunc' objects}
        #             300    0.777    0.003    1.039    0.003 _methods.py:86(_var)
        #             300    0.292    0.001    0.699    0.002 base.py:99(transform)
        #         900/300    0.235    0.000    4.004    0.013 cache.py:99(func_cached)
        with dask_opts(**{
            **dict(
                # FIXME Threading causes hangs during heavy (xc 13k) cache hits, going with processes to work around
                # override_scheduler='synchronous',  # 21.8s (16 cpu, 100 xc recs)  # Fastest for cache hits, by a lot
                # override_scheduler='threads',      # 15.1s (16 cpu, 100 xc recs)
                # override_scheduler='processes',    # 8.4s (16 cpu, 100 xc recs)  # TODO Commented while debugging learning_curve
                override_scheduler='processes',    # 8.4s (16 cpu, 100 xc recs)  # TODO Uncommented after encountering hangs again
            ),
            **kwargs,
        }):
            return RecordingDF(recs
                # Compute row-major with the 'threads' scheduler, since all cols perform best with 'threads'
                .assign(feat=self.feat)
            )

    # Cache to speed up the dev loop in e.g. compare_classifiers_xc
    #   - Low cost to reset the cache since _feat cache hits are fast (~10s for ~13k recs)
    @short_circuit(lambda self, recs: recs.get('feat'))
    @cache(version=0, key=lambda self, recs: (recs.id, self.agg_config, self.skm_config, self.deps))
    def feat(self, recs: RecordingDF) -> Column['np.ndarray[(k*a,)]']:
        """feat (k*a,) <- .agg (k,a)"""
        # Performance (600 peterson recs):
        #   - Scheduler: no_dask[.37s], synchronous[.72s], threads[.60s], processes[>200s]
        #   - Bottlenecks (no_dask):
        #          ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #       3030/3027    0.067    0.000    0.068    0.000 {built-in method numpy.core.multiarray.array}
        #             600    0.046    0.000    0.046    0.000 model.py:376(<listcomp>)
        #       15000/600    0.026    0.000    0.076    0.000 dataclasses.py:1014(_asdict_inner)
        #           79220    0.020    0.000    0.037    0.000 {built-in method builtins.isinstance}
        #           12600    0.012    0.000    0.018    0.000 copy.py:132(deepcopy)
        feat = map_progress(self._feat, df_rows(recs), use='dask', scheduler='threads')
        return feat

    @short_circuit(lambda self, recs: recs.get('agg'))
    def agg(self, recs: RecordingDF) -> Column['np.ndarray[(k,a)]']:
        """agg (k,a) <- .proj (k,t)"""
        # Performance (600 peterson recs):
        #   - Scheduler: no_dask[7.5s], synchronous[4.7s], threads[2.2s], processes[serdes-error...]
        #   - Bottlenecks (no_dask):
        #          ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #            2410    1.994    0.001    1.994    0.001 {method 'reduce' of 'numpy.ufunc' objects}
        #             600    1.744    0.003    2.425    0.004 _methods.py:86(_var)
        #             600    0.424    0.001    2.849    0.005 _methods.py:133(_std)
        #       15000/600    0.032    0.000    0.088    0.000 dataclasses.py:1014(_asdict_inner)
        #           82820    0.023    0.000    0.042    0.000 {built-in method builtins.isinstance}
        agg = map_progress(self._agg, df_rows(recs), use='dask', scheduler='threads')
        return agg

    @short_circuit(lambda self, recs: recs.get('proj'))
    def proj(self, recs: RecordingDF) -> Column['np.ndarray[(k,t)]']:
        """proj (k,t) <- .patch (f*p,t)"""
        # Performance (100 peterson recs):
        #   - Scheduler: no_dask[2.3s], synchronous[1.6s], threads[1.2s], processes[>1000s]
        #   - Bottlenecks (no_dask):
        #          ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        #             200    1.236    0.006    1.236    0.006 {built-in method numpy.core.multiarray.dot}
        #             110    0.591    0.005    0.591    0.005 {method 'reduce' of 'numpy.ufunc' objects}
        #             100    0.185    0.002    1.052    0.011 base.py:99(transform)
        #               1    0.163    0.163    2.260    2.260 <string>:1(<module>)
        #             100    0.018    0.000    2.056    0.021 skm.py:433(transform)
        proj = map_progress(self._proj, df_rows(recs), use='dask', scheduler='threads')
        return proj

    @short_circuit(lambda self, rec: rec.get('feat'))
    @cache(version=0, key=lambda self, rec: (rec.id, self.agg_config, self.skm_config, self.deps))
    def _feat(self, rec: Row) -> 'np.ndarray[(k*a,)]':
        """feat (k*a,) <- .agg (k,a)"""
        agg = self._agg(rec)
        # Deterministic and unsurprising order for feature vectors: follow the order of agg_config.agg_funs
        feat = np.array([
            x
            for agg_fun in self.agg_config.agg_funs
            for x in agg[agg_fun]
        ])
        return feat

    @short_circuit(lambda self, rec: rec.get('agg'))
    @cache(version=0, key=lambda self, rec: (rec.id, self.agg_config, self.skm_config, self.deps))
    def _agg(self, rec: Row) -> Mapping['a', 'np.ndarray[(k,)]']:
        """agg (k,a) <- .proj (k,t)"""
        proj = self._proj(rec)
        return {
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
            for agg_fun in self.agg_config.agg_funs
        }

    @short_circuit(lambda self, rec: rec.get('proj'))
    # @cache(version=0, key=lambda self, rec: (rec.id, self.skm_config, self.deps))  # TODO After birdclef
    def _proj(self, rec: Row) -> 'np.ndarray[(k,t)]':
        """proj (k,t) <- .patch (f*p,t)"""
        patches = self.features._patches(rec)  # Pull
        return self.skm_.transform(patches)

    # This is faster than row-at-a-time _proj by ~2x, but isn't mem safe (but would be easy to make mem safe)
    #   - TODO Consider how to allow bulk col-at-a-time operations
    @generator_to(list)
    def _proj_bulk(self, recs: RecordingDF) -> Column['np.ndarray[(k,t)]']:
        """proj (k,t) <- .patch (f*p,t)"""
        # TODO Chunk up recs to make this mem safe
        for recs in [recs]:
            patches = self.features.patches(recs)  # Pull
            # Concat patches (along t) so we can skm.transform them all at once
            concat_patches = np.concatenate(list(patches), axis=1)
            concat_projs = self.skm_.transform(concat_patches)
            # Unconcat proj (along t) to match patches
            t_shapes = [p.shape[1] for p in patches]
            t_starts = [0] + list(np.cumsum(t_shapes)[:-1])
            for patch, t_shape, t_start in zip(patches, t_shapes, t_starts):
                # Each proj should have the same t_shape and t_start as its patch
                proj = concat_projs[:, t_start:t_start + t_shape]
                yield proj

    def plot_proj_centroids(self, **kwargs):
        """Viz the projection patch centroids (pca -> skm)"""
        skm = self.skm_
        proj = (skm.D.T @ skm.pca.components_).T  # The total projection: pca -> skm.D
        self.plot_patches(proj, **kwargs)

    def plot_patches(self, patches, **kwargs):
        """Viz a set of patches (f*p, n) as a grid that matches figsize"""
        plot_patches(
            patches,
            self.features.spectro_config.f_bins,
            self.features.patch_config.patch_length,
            **kwargs,
        )


@dataclass
class Search(DataclassConfig, sk.base.BaseEstimator, sk.base.ClassifierMixin):

    #
    # sklearn boilerplate [TODO Move into mixin]
    #   - http://scikit-learn.org/dev/developers/contributing.html#rolling-your-own-estimator
    #

    def get_params(self, deep=True):
        return self.config

    def set_params(self, **params):
        self.__dict__.update(params)
        return self

    # TODO Make this a passing test
    #   - Fix: `projection: Projection` isn't a model param (detected because it's not a primitive type)
    # sk.utils.estimator_checks.check_estimator(Search)

    _X_col = 'feat'
    _X_dtype = np.float32  # TODO Bubble this up through Projection (default is np.float64, which we don't need)
    _y_col = 'species'
    _y_dtype = None

    @classmethod
    def df(cls, X: Iterable, y: Iterable) -> pd.DataFrame:
        return pd.DataFrame({
            cls._X_col: [np.array(x) for x in X],  # HACK Avoid error "Data must be 1-dimensional"
            cls._y_col: y,
        })

    @classmethod
    @requires_cols(_X_col)
    def X(cls, df: pd.DataFrame) -> np.ndarray:
        return np.array(list(df[cls._X_col]), dtype=cls._X_dtype)

    @classmethod
    @requires_cols(_y_col)
    def y(cls, df: pd.DataFrame) -> np.ndarray:
        return np.array(list(df[cls._y_col]), dtype=cls._y_dtype)

    @classmethod
    @requires_cols(_X_col, _y_col)
    def Xy(cls, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return (cls.X(df), cls.y(df))

    #
    # Config
    #   - TODO Pull up agg_funs from Projection (and add a self.transform to populate .feat)
    #

    # HACK Expose training set adjustments as model params, so the analysis code can think only in terms of model params
    #   - n_species: Train on fewer classes -- "learning difficulty?"
    #   - n_recs: Train on fewer instances per class (i.e. stratified) -- "learning curve"
    n_species: Optional[Union[int, float]] = None
    n_recs: Optional[Union[int, float]] = None

    # Params for classifier, e.g.
    #   - classifier: str = 'cls: knn, n_neighbors: 3'
    #   - classifier: str = 'cls: svm, random_state: 0, probability: true, kernel: rbf, C: 10'  # Like [SBF16]
    #   - classifier: str = 'cls: rf, random_state: 0, criterion: entropy, n_estimators: 200'  # Like [SP14]
    classifier: str = None

    # For n_species/n_recs, as well as self.classifier params (when appropriate)
    #   - TODO Fork RandomState before passing off to others (e.g. RandomState(0) -> .randint() -> .randint() -> ...)
    random_state: int = 0

    # verbose: bool = True  # Like GridSearchCV [TODO Hook this up to log / TODO Think through caching implications]

    # Dependencies [TODO Simplify with dataclasses.InitVar?]
    projection: Projection = field(default=Projection(), repr=False)  # repr=False? Convenient for watching cv go by...

    @property
    def deps(self) -> AttrDict:
        return AttrDict({k: getattr(self, k) for k in [
            'projection',
        ]})

    @property
    def classifier_config(self) -> AttrDict:
        return AttrDict(yaml.safe_load('{%s}' % self.classifier))

    #
    # Fit
    #

    def fit_recs(self, recs: RecordingDF) -> 'self':
        """feat (k*a,) -> ()"""
        recs = self.projection.transform(recs)  # Pull
        return self.fit(*self.Xy(recs))

    def fit(self, X, y) -> 'self':
        """feat (k*a,) -> ()"""
        # log.info(f'Search.fit... classifier[{self.classifier}]')
        if self.n_species is not None:
            orig_classes = sk.utils.multiclass.unique_labels(y)
            sub_classes = np_sample(
                orig_classes,
                # **{'n' if isinstance(self.n_species, int) else 'frac': self.n_species},  # FIXME n_species > len(orig_classes)
                **(
                    # Have to min(..., len(orig_classes)) in case the test/train split lost classes (e.g. with few obs)
                    dict(n=min(self.n_species, len(orig_classes))) if isinstance(self.n_species, int) else
                    dict(frac=self.n_species)
                ),
                random_state=self.random_state,
            )
            i = np.isin(y, sub_classes)
            X, y = X[i], y[i]
        if self.n_recs is not None:
            X, y = np_sample_stratified(
                X, y,
                **{'n' if isinstance(self.n_recs, int) else 'frac': self.n_recs},
                random_state=self.random_state,
            )
        sk.utils.validation.check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_ = sk.utils.multiclass.unique_labels(y)
        log.debug('Search.fit:in', **{
            'recs': len(X),
            '(n, f*p)': X.shape,
        })
        self.classifier_ = self._fit_pure(X, y, self.classes_)
        log.debug('Search.fit:out', **{
            'classifier.get_params': self.classifier_.get_params(),
            'classifier.classes_': sorted_species(self.classifier_.classes_.tolist()),
            'classifier.classes_.len': len(self.classifier_.classes_),
        })
        # log.info(f'Search.fit. classifier[{self.classifier}]')
        return self

    @cache(version=4, key=lambda self, X, y, classes: (
        self.classifier_config, {k: v for k, v in self.config.items() if k != 'classifier'},  # Prefer classifier dict over str
        X, y, classes,
    ))
    def _fit_pure(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> sk.base.BaseEstimator:
        """classifier <- .feat (k*a,)"""
        classifier = self._make_classifier(
            **self.classifier_config,
            n_features=X.shape[1],  # For xgb/lgb
            n_classes=len(classes),  # For xgb/lgb
        )
        return classifier.fit(X, y)

    def _make_classifier(
        self,
        cls,
        standardize=None,
        multiclass=None,
        n_features=None,  # For xgb/lgb
        n_classes=None,  # For xgb/lgb
        **kwargs,
    ) -> sk.base.BaseEstimator:

        # Expand complex shorthands
        if cls.startswith('std-'):
            [standardize, cls] = cls.split('-', 1)
        if cls.startswith('ovr-'):
            [multiclass, cls] = cls.split('-', 1)

        # Define constructor shorthands
        nb = sk.naive_bayes.GaussianNB
        knn = sk.neighbors.KNeighborsClassifier
        logreg_ovr = partial(sk.linear_model.LogisticRegression,
            random_state=self.random_state,
            multi_class='ovr',  # Default
            n_jobs=-1,  # Use all cores (default: 1)
        )
        # NOTE logreg_multi is single core (only ovr is multicore)
        logreg_multi = partial(logreg_ovr,
            multi_class='multinomial',
        )
        sgdlog = partial(
            sk.linear_model.SGDClassifier,
            loss='log',
            penalty='l2',
            max_iter=1000,  # Match defaults coming in 0.21
            tol=1e-3,  # Match defaults coming in 0.21
            random_state=self.random_state,
            n_jobs=-1,  # Use all cores (default: 1)
        )
        svm = partial(sk.svm.SVC,
            probability=True,
            random_state=self.random_state,
            kernel='rbf',  # Default
        )
        rf = partial(sk.ensemble.RandomForestClassifier,
            # criterion='gini',  # Default
            random_state=self.random_state,
            n_jobs=-1,  # Use all cores (default: 1)
        )
        xgb_rf = partial(self._xgb_rf_classifier,
            n_features=n_features,
            n_classes=n_classes,
            random_state=self.random_state,
            n_jobs=-1,  # Use all cores (default: 1)
            # FIXME silent=False causes annoying atom notifications instead of ipykernel stdout/stderr (because subprocess?)
            silent=True,
        )
        lgbm = partial(lgb.LGBMClassifier,
            random_state=self.random_state,
            # TODO Flesh out like xgb
        )

        # Eval classifier
        classifier = eval(cls)(**kwargs)

        # Add multiclass wrappers
        if multiclass:
            assert not isinstance(classifier, xgb.XGBClassifier), 'xgb is already structured as OVR internally'
            if 'n_jobs' in classifier.get_params():
                # OVR does its own par, and setting n_jobs=1 avoids O(classes) warnings
                classifier.set_params(n_jobs=1)
            classifier = {
                'ovr': sk.multiclass.OneVsRestClassifier(
                    estimator=classifier,
                    n_jobs=-1,  # Use all cores (default: 1) [TODO Should this and rf n_jobs both be -1 together?]
                ),
            }[multiclass]

        # Add preprocessing steps
        if standardize:
            # WARNING logreg with std does worse than logreg without std -- which is very counterintuitive
            #   - See notebooks/20180717_logreg_std_does_worse_(weird...).ipynb
            #   - TODO Why is this?
            classifier = sk.pipeline.make_pipeline(sk.preprocessing.StandardScaler(), classifier)

        return classifier

    def _xgb_rf_classifier(
        self,
        n_features,
        n_classes,
        **kwargs,
    ):
        # Based on 20180710_xgb_train_eval_example.ipynb
        return xgb_sklearn_hack.XGBClassifier(**{
            **dict(

                # Multiclass
                #   - softmax: .predict will return (len(X_test),) like sk .predict
                #   - softprob: .predict will return (len(X_test), n_classes) like sk .predict_proba
                #       - https://github.com/dmlc/xgboost/issues/101
                #   - Both require num_class
                objective='multi:softprob',
                num_class=n_classes,

                # Random forest
                learning_rate=1,  # Don't down-weight each round's trees, since there's only 1 round
                subsample=1 - 1 / np.e,  # Approximate the instance exposure (no replacement) for bootstrap (replacement)
                colsample_bylevel=np.sqrt(n_features) / n_features,  # sqrt(p) for classification, p/3 for regression
                train_kwargs=dict(  # HACK Non-standard arg added in xgb_sklearn_hack
                    num_boost_round=1,
                ),

                # Tree tuning
                max_depth=0,  # TODO Does 0 actually work as "no limit"? -- the notebook results don't support this
                #   - Default: 6
                #   - 0 means no limit
                min_child_weight=0,
                #   - Default: 1
                tree_method='exact',  # TODO Switch back to 'auto' after we figure out how to make xgb rf's work correctly
                #   - Default: 'auto'
                #   - 'exact' | 'approx' | 'hist'
                #   - Quick takeaways
                #       - 'exact'/'approx' take comparable duration, 'hist' takes _way_ longer
                #       - Stick with 'exact'/'auto' for now

            ),
            **kwargs,
        })

    #
    # Predict
    #

    @requires_cols('feat')
    def species(self, recs: RecordingDF) -> Column['species']:
        """feat (k*a,) -> species (1,)"""

        # Unpack inputs (i.e. split here if we need caching)
        X = self.X(recs)
        y = self.y(recs)
        classifier_ = self.classifier_

        # Predict: .predict
        #   - Preserve recs.index in output's index, for easy joins
        species = pd.DataFrame(index=recs.index, data={
            'species_pred': classifier_.predict(X),
        })
        # log.debug('Search.species', **{
        #     'recs': len(recs),
        #     '(n, k*a)': X.shape,
        #     'species': len(species),
        # })

        # Join test labels back onto the predictions, if given
        if 'species' in recs:
            species.insert(0, 'species_true', y)

        return species

    # TODO Add args to limit results: top_k, at_least_p
    @requires_cols('feat')
    def species_probs(self, recs: RecordingDF) -> pd.DataFrame:
        """feat (k*a,) -> species (len(y_),)"""

        # Unpack inputs (i.e. split here if we need caching)
        X = self.X(recs)
        y = self.y(recs)
        classifier_ = self.classifier_

        # Predict: .predict_proba
        #   - Preserve recs.index in output's index, for easy joins
        proba = classifier_.predict_proba(X)
        species_probs = pd.DataFrame(index=recs.index, data=[
            {i: [p, c] for i, (c, p) in enumerate(sorted(dict(row).items(), key=lambda x: (-x[1], x[0])))}
            for i, row in pd.DataFrame(proba, columns=classifier_.classes_).iterrows()
        ])
        # log.debug('Search.species_probs', **{
        #     'recs': len(recs),
        #     '(n, k*a)': X.shape,
        #     # 'knn.n_neighbors': knn_.n_neighbors,
        #     'species_probs': species_probs.shape,
        # })

        # Join test labels back onto the predictions, if given
        if 'species' in recs:
            species_probs.insert(0, 'species_true', y)

        return species_probs

    # TODO Rethink this for non-knn classifiers
    @requires_cols('feat')
    def similar_recs(self, recs: RecordingDF, similar_n) -> pd.DataFrame:
        """feat (k*a,) -> rec (similar_n,)"""

        # Unpack inputs (i.e. split here if we need caching)
        X = self.X(recs)
        y = self.y(recs)
        y_ = self.y_
        knn_ = self.knn_

        # Search: .kneighbors
        #   - Preserve recs.index in output's index, for easy joins
        (dists, fit_is) = knn_.kneighbors(X, n_neighbors=similar_n)
        similar_recs = pd.DataFrame(index=recs.index, data=[
            [
                [fit_i, dist, y_[fit_i]]
                for fit_i, dist in zip(fit_is[i], dists[i])
            ]
            for i in range(len(fit_is))
        ])
        # log.debug('Search.similar_recs', **{
        #     'recs': len(recs),
        #     '(n, k*a)': X.shape,
        #     'similar_recs': similar_recs.shape,
        # })

        # Join test labels back onto the predictions, if given
        if 'species' in recs:
            similar_recs.insert(0, 'species_true', y)

        return similar_recs

    #
    # Score
    #

    # For sk: score(X, y)
    def score(self, X, y) -> float:
        """Goodness-of-fit measure (higher is better)"""
        # log.info(f'Search.score... classifier[{self.classifier}]')
        score = SearchEvals(search=self, X=X, y=y).score()
        # log.info(f'Search.score. classifier[{self.classifier}]')
        return score


@dataclass(init=False, repr=False)
class SearchEvals(DataclassUtil):
    """
    One of:
    - SearchEvals(search=..., recs=...)                     # Does search.predict_proba
    - SearchEvals(search=..., i=..., X=..., y=...)          # Does search.predict_proba
    - SearchEvals(i=..., y=..., classes=..., y_scores=...)  # Doesn't predict
    """

    # The minimal data we need for evals
    i: np.ndarray = None
    y: np.ndarray
    classes: np.ndarray
    y_scores: np.ndarray

    # TODO Need to port to sk Pipeline so that both test/train data are filtered the same, else n_species gets way
    # too gnarly in the cv_results code since y_test doesn't know it should be filtered
    #   - But! here's a hack to avoid it in the short term (fingers crossed it doesn't introduce non-obvious bugs...)
    drop_missing_classes_for_n_species: bool

    def __init__(
        self,
        search=None,
        recs=None,
        i=None,
        X=None,
        y=None,
        classes=None,
        y_scores=None,
        drop_missing_classes_for_n_species=False,
    ):

        # Check inputs
        defined = lambda *args: all(x is not None for x in args)
        assert 1 == sum([
            defined(search, recs),
            defined(search, X, y),
            defined(y, classes, y_scores),
        ])

        # Percolate
        if defined(search, recs):
            i = np.arange(len(recs))
            X, y = search.Xy(recs)
        if defined(search, X, y):
            y_scores = search.classifier_.predict_proba(X)  # Predict!
            classes = search.classifier_.classes_
        assert defined(y, classes, y_scores)

        # HACK See comment above
        if drop_missing_classes_for_n_species:
            # classes is post-n_species but y and y_scores aren't. Update them to match.
            j = np.isin(y, classes)
            y = y[j]
            y_scores = y_scores[j]

        # Store
        self.i = i
        self.y = y
        self.classes = classes
        self.y_scores = y_scores
        self.drop_missing_classes_for_n_species = drop_missing_classes_for_n_species

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join([
            f'{k}[{none_or(v, lambda v: v.shape)}]'
            for k, v in self.asdict().items()
            if k not in [
                'drop_missing_classes_for_n_species',
            ]
        ]))

    def score(self, **kwargs) -> float:
        """Goodness-of-fit measure (higher is better)"""
        return -self.coverage_error(**kwargs)

    def coverage_error(self, agg=np.median) -> float:
        """Predict and measure aggregate coverage error"""
        return agg(self.coverage_errors())

    def coverage_errors(self) -> 'np.ndarray[int @ len(X)]':
        """Predict and measure each instance's coverage error (avoid sk's internal np.mean so we can do our own agg)"""
        return coverage_errors(
            y_true=(self.classes == self.y[:, None]),
            y_score=np.nan_to_num(self.y_scores, 0),  # In case a predict_proba returns nan (e.g. OneVsRestClassifier)
        )

    @classmethod
    @requires_cols('feat', 'species')
    def coverage_errors_recs(cls, search, recs: RecordingDF) -> RecordingDF:
        """.coverage_errors + assign to a col in recs"""
        return recs.assign(coverage_error=cls(search=search, recs=recs).coverage_errors())

    @classmethod
    @requires_cols('feat', 'species')
    def coverage_error_by(cls, search, recs: RecordingDF, by: str, **kwargs) -> "pd.DataFrame[by, 'coverage_error']":
        """Predict and measure coverage error, grouped by recs[by]"""
        if recs[by].dtype.name == 'category':
            recs = recs.assign(**{by: recs[by].cat.remove_unused_categories()})  # Else groupby includes all categories
        return (recs
            .groupby(by)
            .apply(lambda g: cls(search=search, recs=g.assign(**{by: g.name})).coverage_error(**kwargs))
            .reset_index()
            .rename(columns={0: 'coverage_error'})
        )

    def confusion_matrix_prob_df(self) -> 'pd.DataFrame[prob @ (n_species, n_species)]':
        return confusion_matrix_prob_df(self.y, self.y_scores, self.classes)

    def confusion_matrix_prob(self) -> 'np.ndarray[prob @ (c, c)]':
        return confusion_matrix_prob(self.y, self.y_scores, self.classes)


@model_stats.register(Search)
def _(search, **kwargs) -> pd.DataFrame:
    return (
        model_stats(search.classifier_, **kwargs)
        .assign(type=lambda df: 'search/' + df.type)
    )


def to_X(recs: RecordingDF) -> Iterable[AttrDict]:
    return [AttrDict(row) for i, row in recs.iterrows()]


def to_y(recs: RecordingDF) -> np.ndarray:
    return np.array(list(recs.species))


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
        log.debug('init:config', **{
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
        log.debug('init:pipeline', **{
            # (Gross spacing hacks to make stuff align)
            'spectro': f'(f, t)   ({_f}, {_g(_t_s)}/s)',
            'patch  ': f'(f*p, t) ({_f}*{_p}, {_g(_t_s)}/s)',
            'proj   ': f'(k, t)   ({_k}, {_g(_t_s)}/s)',
            'agg    ': f'(k, a)   ({_k}, {_a})',
            'feat   ': f'(k*a,)   ({_ka},)',
        })
