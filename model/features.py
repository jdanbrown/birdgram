from collections import OrderedDict
from contextlib import ExitStack
import copy
import os.path
import re
import sys
from typing import Any, NewType, Optional, Tuple, Union

import audiosegment
import dataclasses
from dataclasses import dataclass
from functools import partial
import librosa
import matplotlib as mpl
import numpy as np
import pandas as pd
import potoo.plot
from potoo.plot import show_img
import pydub
import scipy
import structlog
from typing import List

from constants import cache_dir, data_dir, default_log_ylim_min_hz, standard_sample_rate_hz
from datatypes import Audio, Recording, RecOrAudioOrSignal
import metadata
from util import *

log = structlog.get_logger(__name__)


def df_to_recs(df: pd.DataFrame) -> List[Recording]:
    return [
        Recording(**{
            k: v
            for k, v in dict(row).items()
            if k in [x.name for x in dataclasses.fields(Recording)]
        })
        for row in df_rows(df)
    ]


def unpack_rec(rec_or_audio_or_signal: RecOrAudioOrSignal) -> (
    Optional[Recording],  # rec if input was rec else None
    'Box[Audio]',  # box(audio)
    Audio,  # audio
    np.array,  # x
    int,  # sample_rate
):
    """
    Allow user to pass in a variety of input types
    - Returns (rec, audio), where rec is None if no Recording was provided
    """
    v = rec_or_audio_or_signal
    rec = audio = x = sample_rate = None

    # audio as boxed audio
    if isinstance(v, box):
        v = v.unbox

    # rec as Recording/attrs/Series
    if not isinstance(v, (dict, Audio, np.ndarray, tuple)):
        if isinstance(v, pd.Series):
            v = dict(v)
        else:
            v = v.asdict()

    # rec as dict
    if isinstance(v, dict):
        rec = Recording(**v)
    # audio
    elif isinstance(v, Audio):
        audio = v
    # (x, sample_rate)
    elif isinstance(v, tuple):
        (x, sample_rate) = v
    # x where sample_rate=standard_sample_rate_hz
    elif isinstance(v, np.ndarray):
        (x, sample_rate) = (v, standard_sample_rate_hz)

    audio = (
        rec.audio.unbox if rec and hasattr(rec.audio, 'unbox') else
        rec.audio if rec else
        audio if audio is not None else  # Careful: bool(Audio) isn't reliable for ~0s audios
        audiosegment.from_numpy_array(x, framerate=sample_rate)
    )
    x = audio.to_numpy_array()
    sample_rate = audio.frame_rate

    if sample_rate != standard_sample_rate_hz:
        log.warn(f'Nonstandard sample_rate[{sample_rate}] != standard[{standard_sample_rate_hz}] for audio[{audio}]')

    return (rec, audio, x, sample_rate)


def plt_audio_signal(audio: Audio, **kwargs):
    plt_signal(audio.to_numpy_array(), audio.frame_rate, **kwargs)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%ds'))


class HasPlotAudioTF:

    def _plot_audio_tf(
        self,
        powscale=lambda x: x,
        raw=False,
        yscale='linear',
        ylim=None,  # Hz
        cmap=None,
        ax=None,
        fancy=True,
        audio=False,  # XXX Back compat
        show_marginals=True,
        show_spines=False,
        yticks=None,
        xformat=lambda x: '%ds' % x,
        yformat=lambda y: ('%.1f' % (y / 1000)).rstrip('0').rstrip('.') + 'K',  # Compact (also, 'K' instead of 'KHz')
        labelsize=8,
        figsize=None,
        show=False,
        **kwargs,
    ):

        if raw:
            fancy = False
            figsize = None
        assert not (fancy and ax), f"Can't supply both fancy[{fancy}] and ax[{ax}]"

        with ExitStack() as stack:
            if figsize:
                stack.enter_context(potoo.plot.figsize(**figsize))

            if not raw:

                if fancy and show_marginals:
                    fig = plt.figure()
                    gs = mpl.gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[30, 1], height_ratios=[1, 8], wspace=0, hspace=0)
                else:
                    # Don't interfere with the existing figure/axis
                    fig = plt.gcf()
                    gs = None

                # Setup plot for time-freq spectro (big central plot)
                if fancy and show_marginals:
                    ax_tf = fig.add_subplot(gs[-1, 0])
                else:
                    ax_tf = ax or plt.gca()

            # Unpack attrs
            (f, t, S) = self  # Delegate to __iter__, e.g. for Mfcc

            # Scale S power, if requested
            S = powscale(S)

            # Compute marginals
            S_f = S.mean(axis=1)
            S_t = S.mean(axis=0)

            if raw:
                # Output image resolution is 1-1 with input array shape, but we can't add axis labels and figure titles
                raw_kwargs = raw if isinstance(raw, dict) else {}
                show_img(S, origin='lower', **raw_kwargs)
            else:

                # Plot time-freq spectro (big central plot)
                ax_tf.pcolormesh(
                    # Extend (t,f) by one element each to avoid dropping the last row and col from S (fence-post problem)
                    #   - https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolor.html
                    list(t) + [t[-1] + (t[-1] - t[-2])],
                    list(f) + [f[-1] + (f[-1] - f[-2])],
                    S,
                    cmap=cmap,
                )
                if yscale:
                    yscale = yscale if isinstance(yscale, dict) else dict(value=yscale)
                    ax_tf.set_yscale(**yscale)
                if ylim:
                    ax_tf.set_ylim(ylim)
                if yticks is not None:
                    if ylim:
                        yticks = [y for y in yticks if coalesce(ylim[0], -np.inf) <= y <= coalesce(ylim[1], np.inf)]
                    ax_tf.set_yticks(yticks)
                if labelsize:
                    ax_tf.tick_params(labelsize=labelsize)
                if not show_spines:
                    [s.set_visible(False) for s in ax_tf.spines.values()]

                if fancy and show_marginals:

                    # Marginal time (top edge)
                    ax_t = fig.add_subplot(gs[0, 0])
                    ax_t.step(t, S_t, 'k', linewidth=.5)
                    ax_t.set_xlim(t[0], t[-1])  # (Off by one, but good enough) [repro: try plt.step with few y bins]
                    # ax_t.set_ylim(S_t.min(), S_t.max())  # Makes the highest bar edge disappear
                    ax_t.set_xticks([])
                    ax_t.set_yticks([])
                    ax_t.axis('off')

                    # Marginal freq (right edge)
                    ax_f = fig.add_subplot(gs[-1, -1])
                    ax_f.plot(S_f, f, 'k', linewidth=.5)
                    ax_f.set_ylim(f[0], f[-1])  # (Off by one, but good enough) [repro: try plt.step with few y bins]
                    # ax_f.set_xlim(S_f.min(), S_f.max())  # Makes the highest bar edge disappear
                    if yscale:
                        # ax_f.set_yscale(ax_tf.get_yscale())  # Doesn't work, use **yscale instead
                        ax_f.set_yscale(**yscale)
                    ax_f.set_xticks([])
                    ax_f.set_yticks([])
                    ax_f.axis('off')

                    # Match lims across marginal time/freq plots
                    [ax_t_min, ax_t_max] = ax_t.get_ylim()
                    [ax_f_min, ax_f_max] = ax_f.get_xlim()
                    ax_t_f_lim = [min(ax_t_min, ax_f_min), max(ax_t_max, ax_f_max)]
                    ax_t.set_ylim(ax_t_f_lim)
                    ax_f.set_xlim(ax_t_f_lim)

                if xformat: ax_tf.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos=None: xformat(x)))
                if yformat: ax_tf.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos=None: yformat(y)))

            # Avoid forcing a plt.show() if we have nothing to show (e.g. audio), else take care to show everything cleanly
            if show or figsize or fancy or (audio and not raw):
                plt.show()
            if audio:
                assert False, "TODO Refactor spectro.plot -> rec.plot, since spectro no longer knows .audio"  # TODO
                # display(self.audio)


class SpectroLike:

    def replace(self, **kwargs) -> 'SpectroLike':
        other = copy.copy(self)
        other.__dict__.update(kwargs)
        return other

    def __iter__(self):
        """For unpacking, e.g. (f, t, S) = FooSpectro(...)"""
        return iter([self.f, self.t, self.S])

    def __sizeof__(self):
        return sys.getsizeof(self.__dict__) + sum(
            sys.getsizeof(k) + sys.getsizeof(v)
            for k, v in self.__dict__.items()
            if k != 'self'
        )

    # TODO Recompute rec.audio after denoising rec.spectro? (e.g. playing audio should sound denoised)
    def norm_rms(self) -> 'SpectroLike':
        """Normalize by RMS (like "normalize a spectro by its RMS energy" from [SP14])"""
        S = self.S
        S = S / np.sqrt((S ** 2).mean())
        return self.replace(S=S)

    # TODO Recompute rec.audio after denoising rec.spectro? (e.g. playing audio should sound denoised)
    def clip_below_median_per_freq(self) -> 'SpectroLike':
        """For each freq bin (row), subtract the median and then zero out negative values"""
        S = self.S
        S = (S.T - np.median(S, axis=1)).T
        S = np.clip(S, 0, None)
        return self.replace(S=S)


class Spectro(HasPlotAudioTF, SpectroLike):

    def __init__(
        self,
        rec_or_audio_or_signal,
        nperseg=1024,  # Samples per stft segment
        overlap=0.75,  # Fraction of nperseg samples that overlap between segments
        load=None,  # XXX Back compat: old .pkl's have 'load' in spectro_kwargs, and try to pass it back here. Accept and drop.
        **kwargs,  # Passthru to scipy.signal.spectrogram
    ):
        """
        Compute the (real, power) spectrogram of an audio signal
        - spectro(x) = |STFT(x)|**2
        - Real (|-|), not complex (because humans can't hear complex phase)
        - Power (**2), not energy

        Creates attributes:
        - f: freq indexes (Hz)
        - t: time indexes (s)
        - S: power (f x t): X**2 where X is the (energy) unit of the audio signal
        with shapes:
        - f: len(f) ≈ nperseg/2
        - t: len(t) ≈ len(x)/stride, where stride = nperseg*(1-overlap)
        - S: S.shape = (len(f), len(t))
        """

        (_rec, _audio, x, sample_rate) = unpack_rec(rec_or_audio_or_signal)

        (f, t, S) = scipy.signal.spectrogram(x, sample_rate, **{
            'window': 'hann',
            'nperseg': nperseg,
            'noverlap': int(overlap * nperseg),
            'scaling': 'spectrum',  # Return units X**2 ('spectrum'), not units X**2/Hz ('density')
            'mode': 'magnitude',  # Return |STFT(x)**2|, not STFT(x)**2 (because "humans can't hear complex phase")
            **kwargs,
        })

        # Store
        self.f = f
        self.t = t
        self.S = S

    def plot(self, **kwargs):
        self._plot_audio_tf(**{
            'powscale': np.log1p,  # Spectrogram (S) is defined as linear scale, but plot it as log scale by default
            **kwargs,
        })


class Melspectro(HasPlotAudioTF, SpectroLike):

    def __init__(
        self,
        rec_or_audio_or_signal,
        nperseg=1024,  # Samples per stft segment
        overlap=0.75,  # Fraction of nperseg samples that overlap between segments [TODO Fix when < .5 (see below)]
        mels_div=2,  # Increase to get fewer freq bins (unsafe to decrease) [TODO Understand better]
        n_mels=None,  # Specify directly, instead of mels_div
        load=None,  # XXX Back compat: old .pkl's have 'load' in spectro_kwargs, and try to pass it back here. Accept and drop.
        **kwargs,  # Passthru to scipy.signal.spectrogram
    ):
        """
        Compute the mel spectrogram of an audio signal:
        - Take the (normal) power spectrogram (i.e. S = |STFT(x)|**2)
        - Log-transform the freq axis from linear scale to (approximately) mel scale, using a mel filter bank
        - Log-transform the powers
        - (Optional) Subtract mean per freq band to reduce noise

        Creates attributes:
        - f: freq indexes (Hz), mel-scaled
        - t: time indexes (s)
        - S: log power (f x t): log(X**2) where X is the (energy) unit of the audio signal
        with shapes:
        - f: len(f) ≈ n_mels or nperseg/2/mels_div
        - t: len(t) ≈ len(x)/hop_length, where hop_length = nperseg*(1-overlap)
        - S: S.shape = (len(f), len(t))

        When to use melspectro vs. mfcc (from [2]):
        - "tl;dr: Use Mel-scaled filter banks [i.e. melspectro] if the machine learning algorithm is not susceptible to
          highly correlated input. Use MFCCs if the machine learning algorithm is susceptible to correlated input."

        References:
        1. https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
        2. http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
        3. §12.5.7 of "Text-to-Speech Synthesis" (2009) [https://books.google.com/books?isbn=0521899273]
        """

        (_rec, _audio, _x, sample_rate) = unpack_rec(rec_or_audio_or_signal)

        # TODO Why do we match librosa.feature.melspectrogram when overlap>=.5 but not <.5?
        if overlap < .5:
            log.warn(f"Melspectro gives questionable output when overlap[{overlap}] < .5 (doesn't match librosa)")

        self.melspectro_kwargs = {
            'nperseg': nperseg,
            'overlap': overlap,
            'mels_div': mels_div,
            'n_mels': n_mels,
            **kwargs,
        }

        # Start with a normal spectro
        self.spectro_kwargs = {
            'nperseg': nperseg,
            'overlap': overlap,
            'scaling': 'spectrum',
            'mode': 'magnitude',
            **kwargs,
        }
        (f, t, S) = self.to_normal_spectro(rec_or_audio_or_signal=rec_or_audio_or_signal)

        # HACK Apply unknown transforms to match librosa.feature.melspectrogram
        #   - TODO Figure out why these are required to match output
        #   - And keep in mind that we currently match only when overlap>=.5 but not <.5
        S = S * (nperseg // 2)  # No leads on this one...
        S = S**2  # Like energy->power, but spectro already gives us power instead of energy...

        # Linear freq -> mel-scale freq
        n_mels = n_mels or nperseg // 2 // mels_div  # TODO mels_div should depend on sample_rate [assumes default rate]
        mel_basis = librosa.filters.mel(sample_rate, n_fft=nperseg, n_mels=n_mels)
        S = np.dot(mel_basis, S)

        # Linear power -> log power
        S = librosa.power_to_db(S)

        # Mel-scale f to match S[i]
        f = librosa.mel_frequencies(n_mels, f.min(), f.max())

        # Store
        self.f = f
        self.t = t
        self.S = S

    # FIXME Oops, we removed load_audio_as_rec() when we removed spectro.audio. Figure out how to handle this now.
    def reparam(self, rec_or_audio_or_signal=None, **kwargs):
        return type(self)(rec_or_audio_or_signal if rec_or_audio_or_signal is not None else self.load_audio_as_rec(), **{
            **self.melspectro_kwargs,
            **kwargs,
        })

    # FIXME Oops, we removed load_audio_as_rec() when we removed spectro.audio. Figure out how to handle this now.
    def to_normal_spectro(self, rec_or_audio_or_signal=None, **kwargs):
        return Spectro(rec_or_audio_or_signal if rec_or_audio_or_signal is not None else self.load_audio_as_rec(), **{
            **self.spectro_kwargs,
            **kwargs,
        })

    def slice(self, start_s: float = None, end_s: float = None) -> 'Melspectro':
        if start_s is None: start_s = 0
        if end_s is None: end_s = np.inf
        (ix,) = np.where((self.t >= start_s) & (self.t < end_s))
        new = copy.deepcopy(self)  # Deep copy so that deep mutation is safe (e.g. self.spectro_kwargs[...][...] = ...)
        new.t = self.t[ix]
        new.t = new.t - new.t[0] + self.t[0]  # Rebase new.t to match self.t [TODO Should self.t[0] be nonzero? Always 0.012...]
        new.S = self.S[:, ix]
        return new

    def plot(self, **kwargs):
        self._plot_audio_tf(**{
            # Mel-scale the y-axis
            #   - Required even though Melspectro already mel-scales the S freq axis, else plt scales it back to linear
            #   - https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_yscale.html#matplotlib.axes.Axes.set_yscale
            #   - https://matplotlib.org/api/_as_gen/matplotlib.pyplot.yscale.html
            'yscale': dict(
                value='symlog', basey=1000, linthreshy=1000, linscaley=.1,
                subsy=[1000],  # Hack to hide minor tick marks
            ),
            'yticks': [0, 1000, 2000, 4000, 8000],
            **kwargs,
        })

    def plot_normal(self, **kwargs):
        """Plot normal spectro"""
        self.to_normal_spectro().plot(**kwargs)


    def plots(self, **kwargs):
        """Plot normal spectro + melspectro"""
        kwargs.setdefault('show', True)
        self.plot_normal(**{**kwargs, 'audio': False})
        self.plot(**kwargs)


# TODO Basically works, but I left some loose ends
class Mfcc(HasPlotAudioTF):

    def __init__(
        self,
        rec_or_audio_or_signal,
        nperseg=1024,  # Samples per stft segment
        overlap=0.75,  # Fraction of nperseg samples that overlap between segments
        mels_div=2,  # Increase to get fewer freq bins (unsafe to decrease) [TODO Understand better]
        first_n_mfcc=None,  # Default: all mfcc's (= len(f) = nperseg/2/mels_div)
        std=True,  # Whether to standardize the quefrency slices (rows) of M
        dct_type=2,  # TODO librosa uses 2 (FT) but isn't 3 (IFT) more technically correct?
        dct_norm='ortho',
        **kwargs,  # Passthru to scipy.signal.spectrogram
    ) -> (
        'q',  # quefrency indexes (s)
        't',  # time indexes (s)
        'M',  # [What are the units here?]
    ):
        """
        Compute the MFCCs of an audio signal:
        - Take the mel spectrogram (i.e. S = melspectro(x))
        - Map each time's spectrum to a cepstrum by taking the IDCT (inverse DCT, i.e. DCT type 3), as if it were a signal
        - (Optional) Subtract mean per coefficient to reduce noise

        Returns:
        - S.shape = (len(q), len(t))
        - len(t) ≈ len(x)/stride, where stride = nperseg*(1-overlap)
        - len(q) ≈ first_n_mfcc, else nperseg/2/mels_div

        When to use melspectro vs. mfcc (from [2]):
        - "tl;dr: Use Mel-scaled filter banks [i.e. melspectro] if the machine learning algorithm is not susceptible to
          highly correlated input. Use MFCCs if the machine learning algorithm is susceptible to correlated input."

        References:
        1. https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
        2. http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
        3. §12.5.7 of "Text-to-Speech Synthesis" (2009) [https://books.google.com/books?isbn=0521899273]
        """

        (f, t, S) = Melspectro(rec_or_audio_or_signal, **{
            'nperseg': nperseg,
            'overlap': overlap,
            'mels_div': mels_div,
            **kwargs,
        })

        if first_n_mfcc is None:
            first_n_mfcc = len(f)

        # M = np.dot(librosa.filters.dct(first_n_mfcc, len(f)), S)  # XXX
        M = scipy.fftpack.dct(S, axis=0, type=dct_type, norm=dct_norm)[:first_n_mfcc]
        # Quefrency units are time (lag?) with values 1/f
        #   - http://rug.mnhn.fr/seewave/HTML/MAN/ceps.html -- does just 1/f
        #   - http://azimadli.com/vibman/cepstrumterminology.htm
        #   - TODO But how do we encode them without making plt.pcolormesh scale non-linearly? Stick with arange for now...
        #       - Use plt.imshow instead of plt.pcolormesh? imshow works with plt.show(), it doesn't replace it
        #           - https://matplotlib.org/tutorials/introductory/images.html
        #           - https://matplotlib.org/gallery/images_contours_and_fields/contour_image.html
        #           - https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html
        # q = f[:first_n_mfcc]
        q = np.arange(first_n_mfcc)

        # Standardize: (x - μ) / σ
        if std:
            M = (M - M.mean(axis=1)[:, np.newaxis]) / M.std(axis=1)[:, np.newaxis]

        # Store
        self.q = q
        self.t = t
        self.M = M

    def __iter__(self):
        """For unpacking, e.g. (q, t, M) = Mfcc(...)"""
        return iter([self.q, self.t, self.M])

    def plot(self, **kwargs):
        self._plot_audio_tf(**{
            # Mel-scale the y-axis
            #   - Required even though Melspectro already mel-scales the S freq axis, else plt scales it back to linear
            #   - https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_yscale.html#matplotlib.axes.Axes.set_yscale
            #   - https://matplotlib.org/api/_as_gen/matplotlib.pyplot.yscale.html
            'yscale': dict(value='symlog', basey=2, linthreshy=1024, linscaley=.5),
            # TODO Figure out correct values/labels for y-axis (see comments above)
            'yformat': lambda y: y,
            **kwargs,
        })


# TODO
def plt_compare_spec_mel_mfcc(rec_or_audio_or_signal):

    Spectro(rec_or_audio_or_signal).plot(audio=False)
    plt.show()

    Melspectro(rec_or_audio_or_signal).plot(audio=False)
    plt.show()

    Mfcc(rec_or_audio_or_signal).plot(audio=False)
    plt.show()

    (_rec, _audio, x, sample_rate) = unpack_rec(rec_or_audio_or_signal)
    mfccs = librosa.feature.mfcc(x.astype(float), sample_rate, n_mfcc=4)
    for i in range(mfccs.shape[0]):
        mfccs[i] = (mfccs[i] - mfccs[i].mean()) / mfccs[i].std()
    plt.pcolormesh(mfccs)
    plt.gca().yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%9d'))  # Hack to align x-axis with spectro
    plt.show()


# TODO Eventually
def ambiguity_function_via_spectro(x):
    """
    A fast ambiguity function for signal x via 2D-FT on its (STFT) spectrogram:
        A_x(ν,τ) = FT_{t->ν}(IFT_{τ<-f}(S_x(t,f)))

    This approach mimics QTFD relationship between the ambiguity function and WVD, which is slow to compute:
        A_z(ν,τ) = FT_{t->ν}(IFT_{τ<-f}(W_z(t,f)))
    """
    pass  # TODO WIP in plot_a_few_tfds.ipynb
