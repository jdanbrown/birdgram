import logging

log = logging.getLogger(__name__)


## bubo-features

from typing import Any, NewType, Optional, Tuple, Union

import attr
import audiosegment
from functools import partial
import librosa
import matplotlib as mpl
import numpy as np
import pydub
import scipy

recordings_dir = '/Users/danb/hack/bubo/data/recordings-new'
data_dir = '/Users/danb/hack/bubo/data'
peterson_dir = f'{data_dir}/peterson-field-guide'

standard_sample_rate_hz = 22050  # Can resolve 11025Hz (by Nyquist), which most/all birds are below
default_log_ylim_min_hz = 512  # Most/all birds are above 512Hz (but make sure to clip noise below 512Hz)

Recording = attr.make_class('Recording', ['audio', 'source', 'species_code', 'title'])
RecOrAudioOrSignal = Union[
    Recording,  # rec as Recording/attrs
    dict,  # rec as dict
    audiosegment.AudioSegment,  # audio
    Tuple[np.array, int],  # (x, sample_rate)
    np.array,  # x where sample_rate=standard_sample_rate_hz
]


def _unpack_input(rec_or_audio_or_signal: RecOrAudioOrSignal) -> (
    Optional[Recording],  # rec if input was rec else None
    audiosegment.AudioSegment,  # audio
    np.array,  # x
    int,  # sample_rate
):
    """
    Allow user to pass in a variety of input types
    - Returns (rec, audio), where rec is None if no Recording was provided
    """
    v = rec_or_audio_or_signal
    rec = audio = x = sample_rate = None

    # rec as Recording/attrs
    if not isinstance(v, (dict, audiosegment.AudioSegment, np.ndarray, tuple)):
        v = {a.name: getattr(v, a.name) for a in Recording.__attrs_attrs__}

    # rec as dict
    if isinstance(v, dict):
        rec = Recording(**v)
    # audio
    elif isinstance(v, audiosegment.AudioSegment):
        audio = v
    # (x, sample_rate)
    elif isinstance(v, tuple):
        (x, sample_rate) = v
    # x where sample_rate=standard_sample_rate_hz
    elif isinstance(v, np.ndarray):
        (x, sample_rate) = (v, standard_sample_rate_hz)

    audio = (
        rec.audio if rec else
        audio if audio is not None else  # Careful: bool(audiosegment.AudioSegment) isn't reliable for ~0s audios
        audiosegment.from_numpy_array(x, framerate=sample_rate)
    )
    x = audio.to_numpy_array()
    sample_rate = audio.frame_rate

    if sample_rate != standard_sample_rate_hz:
        log.warn(f'Nonstandard sample_rate[{sample_rate}] != standard[{standard_sample_rate_hz}] for audio[{audio}]')

    return (rec, audio, x, sample_rate)


def spectro(
    rec_or_audio_or_signal,
    nperseg=1024,  # Samples per stft segment
    overlap=0.75,  # Fraction of nperseg samples that overlap between segments
    **kwargs,  # Passthru to scipy.signal.spectrogram
) -> (
    'f',  # freq indexes (Hz)
    't',  # time indexes (s)
    'S',  # power (f x t): X**2 where X is the (energy) unit of the audio signal
):
    """
    Compute the (real, power) spectrogram of an audio signal
    - spectro(x) = |STFT(x)|**2
    - Real (|-|), not complex (because humans can't hear complex phase)
    - Power (**2), not energy

    Returns:
    - S.shape = (len(f), len(t))
    - len(t) ≈ len(x)/stride, where stride = nperseg*(1-overlap)
    - len(f) ≈ nperseg/2
    """

    (rec, audio, x, sample_rate) = _unpack_input(rec_or_audio_or_signal)
    (f, t, S) = scipy.signal.spectrogram(x, sample_rate, **{
        'window': 'hann',
        'nperseg': nperseg,
        'noverlap': int(overlap * nperseg),
        'scaling': 'spectrum',  # Return units X**2 ('spectrum'), not units X**2/Hz ('density')
        'mode': 'magnitude',  # Return |STFT(x)**2|, not STFT(x)**2 (because humans can't hear complex phase)
        **kwargs,
    })
    return (f, t, S)


def melspectro(
    rec_or_audio_or_signal,
    nperseg=1024,  # Samples per stft segment
    overlap=0.75,  # Fraction of nperseg samples that overlap between segments
    mels_div=2,  # Increase to get fewer freq bins (unsafe to decrease) [TODO Understand better]
    **kwargs,  # Passthru to scipy.signal.spectrogram
) -> (
    'f',  # freq indexes (Hz), mel-scaled
    't',  # time indexes (s)
    'S',  # log power (f x t): log(X**2) where X is the (energy) unit of the audio signal
):
    """
    Compute the mel spectrogram of an audio signal:
    - Take the (normal) power spectrogram (i.e. S = |STFT(x)|**2)
    - Log-transform the freq axis from linear scale to (approximately) mel scale, using a mel filter bank
    - Log-transform the powers
    - (Optional) Subtract mean per freq band to reduce noise

    Returns:
    - S.shape = (len(f), len(t))
    - len(t) ≈ len(x)/stride, where stride = nperseg*(1-overlap)
    - len(f) ≈ nperseg/2/mels_div

    When to use melspectro vs. mfcc (from [2]):
    - "tl;dr: Use Mel-scaled filter banks [i.e. melspectro] if the machine learning algorithm is not susceptible to
      highly correlated input. Use MFCCs if the machine learning algorithm is susceptible to correlated input."

    References:
    1. https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
    2. http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    3. §12.5.7 of "Text-to-Speech Synthesis" (2009) [https://books.google.com/books?isbn=0521899273]
    """

    (rec, audio, x, sample_rate) = _unpack_input(rec_or_audio_or_signal)

    # TODO Why do we match librosa.feature.melspectrogram when overlap>=.5 but not <.5?
    if overlap < .5:
        log.warn(f"melspectro gives questionable output when overlap[{overlap}] < .5 (i.e. doesn't match librosa)")

    # Normal spectro
    (f, t, S) = spectro(audio, **{
        'nperseg': nperseg,
        'overlap': overlap,
        'scaling': 'spectrum',
        'mode': 'magnitude',
        **kwargs,
    })

    # HACK Apply unknown transforms to match librosa.feature.melspectrogram
    #   - TODO Figure out why these are required to match output
    #   - And keep in mind that we currently match only when overlap>=.5 but not <.5
    S = S * (nperseg // 2)  # No leads on this one...
    S = S**2  # Like energy->power, but spectro already gives us power instead of energy...

    # Linear freq -> mel-scale freq
    n_mels = nperseg // 2 // mels_div  # TODO mels_div should depend on sample_rate [assumes standard rate]
    mel_basis = librosa.filters.mel(sample_rate, n_fft=nperseg, n_mels=n_mels)
    S = np.dot(mel_basis, S)

    # Linear power -> log power
    S = librosa.power_to_db(S)

    # Mel-scale f like S
    f = librosa.mel_frequencies(n_mels, f.min(), f.max())

    return (f, t, S)


# TODO
def mfcc(
    rec_or_audio_or_signal,
    nperseg=1024,  # Samples per stft segment
    overlap=0.75,  # Fraction of nperseg samples that overlap between segments
    mels_div=2,  # Increase to get fewer freq bins (unsafe to decrease) [TODO Understand better]
    std=True,  # Whether to standardize the quefrency slices (rows) of M
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
    - len(q) ≈ nperseg/2/mels_div

    When to use melspectro vs. mfcc (from [2]):
    - "tl;dr: Use Mel-scaled filter banks [i.e. melspectro] if the machine learning algorithm is not susceptible to
      highly correlated input. Use MFCCs if the machine learning algorithm is susceptible to correlated input."

    References:
    1. https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
    2. http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    3. §12.5.7 of "Text-to-Speech Synthesis" (2009) [https://books.google.com/books?isbn=0521899273]
    """
    (rec, audio, x, sample_rate) = _unpack_input(rec_or_audio_or_signal)

    (f, t, S) = melspectro(audio, **{
        'nperseg': nperseg,
        'overlap': overlap,
        'mels_div': mels_div,
        **kwargs,
    })

    # n_mfcc = len(f)  # TODO Some function of mels_div
    n_mfcc = 4
    M = np.dot(librosa.filters.dct(n_mfcc, len(f)), S);  plt.title('librosa dct filter bank (DCT type 2, not inverse)')
    # M = scipy.fftpack.dct(S, axis=0, norm=None, n=n_mfcc, type=2);  plt.title('scipy DCT (type 2)')
    # M = scipy.fftpack.dct(S, axis=0, norm=None, n=n_mfcc, type=3);  plt.title('scipy IDCT (type 3)')
    # M = np.abs(scipy.fftpack.ifft(S, axis=0, n=n_mfcc));  plt.title('scipy IFFT')
    print(M.shape)
    # q = f  # TODO ?
    # q = np.arange(M.shape[0] + 1)
    q = np.arange(M.shape[0])

    # Standardize ((x - μ) / σ)
    if std:
        M = (M - M.mean(axis=1)[:, np.newaxis]) / M.std(axis=1)[:, np.newaxis]

    return (q, t, M)


def plt_audio_signal(audio: audiosegment.AudioSegment, **kwargs):
    plt_signal(audio.to_numpy_array(), audio.frame_rate, **kwargs)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%ds'))


def plt_spectro(*args, **kwargs):
    _plt_spectro_func(*args, **{
        'spectro_func': kwargs.pop('spectro', spectro),
        'powscale': np.log1p,
        **kwargs,
    })


def plt_melspectro(*args, **kwargs):
    _plt_spectro_func(*args, **{
        'spectro_func': kwargs.pop('melspectro', melspectro),
        # Mel-scale the y-axis
        #   - Required even though melspectro already mel-scales the S freq axis, else plt scales it back to linear
        #   - https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.set_yscale.html#matplotlib.axes.Axes.set_yscale
        #   - https://matplotlib.org/api/_as_gen/matplotlib.pyplot.yscale.html
        'yscale': dict(value='symlog', basey=2, linthreshy=1024, linscaley=.5),
        **kwargs,
    })


def plt_mfcc(*args, **kwargs):
    plt_melspectro(*args, **{
        'spectro_func': kwargs.pop('mfcc', mfcc),
        'yformat': lambda y: y,
        **kwargs,
    })


def _plt_spectro_func(
    rec_or_audio_or_signal,
    spectro_func,
    powscale=lambda x: x,
    yscale='linear',
    ylim=None,  # Hz
    cmap=None,
    ax=None,
    fancy=True,
    show_audio=True,
    show_marginals=True,
    show_title=True,
    show_spines=False,
    xformat=lambda x: '%ds' % x,
    yformat=lambda y: '%.0fKiHz' % int(y / 1024),
    # yformat=lambda y: '%.0fHz' % y,
    fontsize=10,
    labelsize=8,
    **kwargs,
):

    assert not (fancy and ax), f"Can't supply both fancy[{fancy}] and ax[{ax}]"
    (rec, audio, x, sample_rate) = _unpack_input(rec_or_audio_or_signal)

    (f, t, S) = spectro_func(audio, **kwargs)
    S = powscale(S)
    S_f = S.mean(axis=1)
    S_t = S.mean(axis=0)

    if fancy and show_marginals:
        fig = plt.figure()
        gs = mpl.gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[30, 1], height_ratios=[1, 8], wspace=0, hspace=0)
    else:
        # Don't interfere with the existing figure/axis
        fig = None
        gs = None

    # Time-freq spectro (big central plot)
    if fancy:
        ax_tf = fig.add_subplot(gs[-1, 0])
    else:
        ax_tf = ax or plt.gca()
    ax_tf.pcolormesh(
        # Extend (t,f) by one element each to avoid dropping the last row and col from S (fence-post problem)
        #   - https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolor.html
        list(t) + [t[-1] + (t[-1] - t[-2])], list(f) + [f[-1] + (f[-1] - f[-2])],
        S,
        cmap=cmap,
    )
    if yscale:
        yscale = yscale if isinstance(yscale, dict) else dict(value=yscale)
        ax_tf.set_yscale(**yscale)
    if ylim:
        ax_tf.set_ylim(ylim)
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

    if fancy and show_title and rec:
        # Put title at the bottom (as ax_tf xlabel) because fig.suptitle messes up vspacing with different figsizes
        ax_tf.set_xlabel(f'{rec.source}/{rec.species_code}/{rec.title}', fontsize=fontsize)
        # fig.suptitle(f'{rec.source}/{rec.species_code}/{rec.title}', fontsize=fontsize)  # XXX Poop on this

    if xformat: ax_tf.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos=None: xformat(x)))
    if yformat: ax_tf.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos=None: yformat(y)))

    if fancy and show_audio:
        plt.show()  # Flush plot else audio displays first
        display(audio)


# TODO
def plt_compare_spec_mel_mfcc(rec_or_audio_or_signal):
    (rec, audio, x, sample_rate) = _unpack_input(rec_or_audio_or_signal)

    x = audio.to_numpy_array()
    sample_rate = audio.frame_rate

    plt_spectro(x)
    plt.show()

    plt_melspectro(audio0, show_audio=False)
    plt.show()

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


## matplotlib

import matplotlib.pyplot as plt
import numpy as np


def plt_signal(y: np.array, x_scale: float = 1, show_ydtype=False, show_yticks=False):
    # Performance on ~1.1M samples:
    #   - ggplot+geom_line / qplot is _really_ slow (~30s)
    #   - df.plot is decent (~800ms)
    #   - plt.plot is fastest (~550ms)
    plt.plot(
        np.arange(len(y)) / x_scale,
        y,
    )
    if not show_yticks:
        # Think in terms of densities and ignore the scale of the y-axis
        plt.yticks([])
    if show_ydtype:
        # But show the representation dtype so the user can stay aware of overflow and space efficiency
        plt.ylabel(y.dtype.type.__name__)
        if np.issubdtype(y.dtype, np.integer):
            plt.ylim(np.iinfo(y.dtype).min, np.iinfo(y.dtype).max)


## pandas

from typing import List

import pandas as pd


def df_reorder_cols(df: pd.DataFrame, first: List[str] = [], last: List[str] = []) -> pd.DataFrame:
    first_last = set(first) | set(last)
    return df.reindex(columns=first + [c for c in df.columns if c not in first_last] + last)


## shell

import os


def ls(dir):
    return [
        os.path.join(dir, filename)
        for filename in os.listdir(dir)
    ]


## General

import os
import shlex


def ensure_parent_dir(path):
    mkdir_p(os.path.dirname(path))
    return path


def mkdir_p(path):
    os.system('mkdir -p %s' % shlex.quote(path))


def puts(x):
    print(x)
    return x
