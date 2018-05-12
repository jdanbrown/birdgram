from datetime import datetime

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skm import SKM
import yaml

from util import *


class Model:
    """
    Params
        |                      | defaults      | [SP14]         | [SBF16]       |
        |----------------------|---------------|----------------|---------------|
        | rec_sample_rate      | 22050         | 44100          | 22050
        | spectro_f_min        | 1000          | 500            | 2000
        |   f_max              | 11025         | 22050          | 11025
        | spectro_f_bins (f)   | 40            | 40             | 40
        | spectro_hop_length   | 256 (12ms)    | 1024 (23ms)    | 32 (1.5ms)
        | spectro_frame_length | 512 (23ms)    | 1024 (23ms)    | 256 (12ms)
        |   frame_overlap      | .5            | 0              | .875
        |   frames/s (t/s)     | 86            | 43             | 689
        | spectro_frame_window | hann          | hamming        | hann
        | norm                 | [TODO]        | RMS+median     | [TODO]
        | patch_length (p)     | 4 (46ms)      | ~4 (~93ms)     | ~16 (~22ms)
        | proj_skm_pca_var     | .99           | —              | .99
        | proj_skm_k (k)       | 500           | 500            | ~512
        | agg_funs             | μ,σ,max       | μ,σ            | ~μ,σ,max
        |   a                  | 3             | 2              | ~3
        |   features           | 1500          | 1000           | ~1536

    Pipeline
        | rec     | (samples,) | (22050/s,)   | (44100/s)     | (22050/s,)
        | spectro | (f, t)     | (40, 86/s)   | (40, 43/s)    | (40, 689/s)
        | patch   | (f*p, t)   | (40*4, 86/s) | (40*~4, 43/s) | (40*~16, 689/s)
        | proj    | (k, t)     | (500, 86/s)  | (500, 43/s)   | (~512, 689/s)
        | agg     | (k, a)     | (500, 3)     | (500, 2)      | (~512, ~3)
        | feat    | (k*a,)     | (1500,)      | (1000,)       | (~1536,)
    """

    def __init__(
        self,
        rec_sample_rate=22050,
        spectro_f_min=1000,
        spectro_f_bins=40,
        spectro_hop_length=256,
        spectro_frame_length=512,
        spectro_frame_window='hann',
        patch_length=4,
        proj_skm_pca_var=.99,
        proj_skm_k=500,
        agg_funs=['mean', 'std', 'max'],
        class_knn_k=3,
        verbose=True,
        verbose_params=True,
    ):
        self.__dict__.update(locals())
        self._print_params()

    def _print_params(self):
        if self.verbose_params:
            _g = lambda x: '%.3g' % x
            _samples_s = self.rec_sample_rate
            _f = self.spectro_f_bins
            _hop = self.spectro_hop_length
            _frame = self.spectro_frame_length
            _t_s = _samples_s / _hop
            _p = self.patch_length
            _k = self.proj_skm_k
            _a = len(self.agg_funs)
            _ka = _k * _a
            self._log('init:params', **{
                'rec_sample_rate': '%s Hz' % _samples_s,
                'spectro_f_min': '%s Hz' % self.spectro_f_min,
                '  f_max': '%s Hz' % (_samples_s // 2),
                'spectro_f_bins (f)': '%s freq bins' % _f,
                'spectro_hop_length': '%s samples (%s ms)' % (_hop, _g(1 / _samples_s * _hop * 1000)),
                'spectro_frame_length': '%s samples (%s ms)' % (_frame, _g(1 / _samples_s * _frame * 1000)),
                '  frame_overlap': '%s%% overlap (%s samples)' % (_g(100 * (1 - _hop / _frame)), _frame // 2),
                '  frames/s (t/s)': '%s samples/s' % _g(_t_s),
                'spectro_frame_window': repr(self.spectro_frame_window),
                'norm': '[TODO]',  # TODO
                'patch_length (p)': '%s frames (%s ms)' % (_p, _g(_p * 1 / _samples_s * _hop * 1000)),
                'proj_skm_pca_var': '%s%% variance' % _g(100 * self.proj_skm_pca_var),
                'proj_skm_k': '%s clusters' % _k,
                'agg_funs': repr(self.agg_funs),
                '  a': '%s aggs' % _a,
                '  features': '%s features' % _ka,
                'class_knn_k': self.class_knn_k,
            })
            self._log('init:pipeline', **{
                # (Gross spacing hacks to make stuff align)
                'spectro': f'(f, t)   ({_f}, {_g(_t_s)}/s)',
                'patch  ': f'(f*p, t) ({_f}*{_p}, {_g(_t_s)}/s)',
                'proj   ': f'(k, t)   ({_k}, {_g(_t_s)}/s)',
                'agg    ': f'(k, a)   ({_k}, {_a})',
                'feat   ': f'(k*a,)   ({_ka},)',
            })

    def fit_proj(self, recs):
        """rec (samples,) -> spectro (f,t) -> patch (f*p,t) -> [skm.fit]"""
        self.proj_recs_ = recs
        self.proj_patches_ = self.patches(self.proj_recs_)
        self.proj_skm_ = SKM(k=self.proj_skm_k)
        skm_X = np.concatenate(self.proj_patches_, axis=1)  # (Entirely an skm.fit concern)
        self._log('fit_proj:skm_X', **{
            'skm_X.shape': skm_X.shape,
        })
        self.proj_skm_.fit(skm_X)
        self._log('fit_proj:skm.fit', **{
            'skm.pca.components_.shape': self.proj_skm_.pca.components_.shape,
            'skm.D.shape': self.proj_skm_.D.shape,
        })
        return self

    def fit_class(self, recs, classes):
        """patch (f*p,t) -> [skm.transform] -> proj (k,t) -> agg (k,a) -> feat (k*a,) -> [knn.fit]"""
        self.class_recs_ = recs
        self.class_classes_ = classes
        self.class_patches_ = self.patches(self.class_recs_)
        self.class_feats_ = self.feats(self.class_patches_)  # (skm.transform)
        self._log('fit_class:knn_Xy', **{
            '(f*p, t)': [p.shape for p in self.class_patches_],
        })
        self.class_knn_ = KNeighborsClassifier(self.class_knn_k).fit(self.class_feats_, self.class_classes_)
        self._log('fit_class:knn', **{
            'knn.get_params': self.class_knn_.get_params(),
            'knn.classes_': self.class_knn_.classes_.tolist(),
        })
        return self

    def predict(self, recs, type) -> pd.DataFrame:
        """
        rec (samples,) -> spectro (f,t) -> patch (f*p,t) -> [skm.transform]
            -> proj (k,t) -> agg (k,a) -> feat (k*a,) -> [knn.predict]
        """
        # _transform_proj: recs -> aggs
        self.predict_patches_ = self.patches(recs)
        self.predict_feats_ = self.feats(self.predict_patches_)  # (skm.transform)
        # _predict_class_*: feats -> classes/kneighbors
        return self._predict_class(feats, type)

    def _predict_class(self, feats, type) -> pd.DataFrame:
        if type == 'classes':
            return self._predict_class_classes(self.predict_feats_)  # (knn.predict_proba)
        elif type == 'kneighbors':
            return self._predict_class_kneighbors(self.predict_feats_)  # (knn.kneighbors)
        else:
            raise ValueError(f"type[{type}] must be one of: 'classes', 'kneighbors'")

    def _predict_class_classes(self, feats) -> pd.DataFrame:
        """agg (k,a) -> class (test_n, class_n)"""
        proba = self.class_knn_.predict_proba(feats)
        classes = self.class_knn_.classes_
        classes_df = pd.DataFrame([
            {i: [k, v] for i, (k, v) in enumerate(sorted(dict(row).items(), key=lambda x: (-x[1], x[0])))}
            for i, row in pd.DataFrame(proba, columns=classes).iterrows()
        ])
        self._log('predict:classes', **{
            # TODO Put something useful here that isn't too big
        })
        return classes_df

    def _predict_class_kneighbors(self, feats) -> pd.DataFrame:
        """agg (k,a) -> neighbor (test_n, train_n)"""
        fit_recs = self.class_recs_
        fit_classes = self.class_classes_
        (dists, fit_is) = self.class_knn_.kneighbors(feats, n_neighbors=len(fit_recs))
        kneighbors_df = pd.DataFrame([
            [
                [fit_i, dist, fit_classes[fit_i]]
                for fit_i, dist in zip(fit_is[i], dists[i])
            ]
            for i in range(len(fit_is))
        ])
        self._log('predict:kneighbors', **{
            # TODO Put something useful here that isn't too big
        })
        return kneighbors_df

    def patches(self, recs):
        """rec (samples,) -> spectro (f,t) -> patch (f*p,t)"""
        self._log('patches:recs', **{
            'len(recs)': len(recs),
            '(samples,)': [int(r.audio.frame_count()) for r in recs],
        })
        spectros = self._spectros(recs)
        self._log('patches:spectros', **{
            '(f, t)': [x.S.shape for x in spectros],
        })
        patches = self._patches(spectros)
        self._log('patches:patches', **{
            '(f*p, t)': [p.shape for p in patches],
        })
        return patches

    def feats(self, patches):
        """patch (f*p,t) -> [skm.transform] -> agg (k,a) -> feat (k*a,)"""
        projs = self._transform_proj(patches)  # (skm.transform)
        aggs = self._aggs(projs)
        feats = self._feats(aggs)
        return feats

    @generator_to(list)
    def _spectros(self, recs):
        """
        rec (samples,) -> spectro (f,t)
          - f: freq indexes (Hz), mel-scaled
          - t: time indexes (s)
          - S: log power (f x t): log(X**2) where X is the (energy) unit of the audio signal
        """
        for rec in recs:
            (rec, _audio, _x, _sample_rate) = unpack_rec(rec)
            assert rec.audio.frame_rate == self.rec_sample_rate, 'Expected %s, got %s' % (self.rec_sample_rate, rec)
            yield Melspectro(
                rec,
                nperseg=self.spectro_frame_length,
                overlap=1 - self.spectro_hop_length / self.spectro_frame_length,
                window=self.spectro_frame_window,
                n_mels=self.spectro_f_bins,
            )

    @generator_to(list)
    def _patches(self, spectros):
        """spectro (f,t) -> patch (f*p,t)"""
        for spectro in spectros:
            (f, t, S) = spectro
            patch = np.array([
                S[:, i:i + self.patch_length].flatten()
                for i in range(S.shape[1] - (self.patch_length - 1))
            ]).T
            yield patch

    def _transform_proj(self, patches):
        """patch (f*p,t) -> proj (k,t)"""
        projs = [self.proj_skm_.transform(p) for p in patches]
        return projs

    @generator_to(list)
    def _aggs(self, projs):
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
                for agg_fun in self.agg_funs
            }))

    @generator_to(list)
    def _feats(self, aggs):
        """agg (k,a) -> feat (k*a,)"""
        for agg in aggs:
            yield agg.T.values.flatten()

    def _log(self, event, **kwargs):
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
                v_yaml = yaml.safe_dump(v, default_flow_style=True, width=1e9)
                v_yaml = v_yaml.split('\n')[0]  # Handle documents ([1] -> '[1]\n') and scalars (1 -> '1\n...\n')
                print('  %s: %s' % (k, v_yaml))
