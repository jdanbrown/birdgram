import dask
import dask.bag
# import ggplot as gg
import numpy as np
import os
import pandas as pd
from IPython.display import display
import random
import scipy
import scipy.signal as sig
import time
import traceback
import wavio

# # XXX dev: repl workflow
# exec('; '.join(['import %s; reload(%s)' % (m, m) for m in [
#     'bubo.util',
#     'potoo.dynvar',
#     'potoo.mpl_backend_xee',
# ]]))

import potoo.mpl_backend_xee
# from bubo.util import caffe_root, plot_img, plot_gg, show_shapes, show_tuple_tight, tmp_rcParams
# from bubo.util import gg_layer, gg_xtight, gg_ytight, gg_tight
from bubo.util import plot_img, shell, singleton, puts

#
# Util
#

def with_progress(f, total_i, start_time, prob_print, i, *args):
    if random.random() < .25:
        elapsed_i   = i + 1
        elapsed_s   = int((time.time() - start_time) * 10) / 10.
        remaining_s = float('inf') if elapsed_i == 0 else int((float(total_i - elapsed_i) * (float(elapsed_s) / elapsed_i)) * 10) / 10.
        print('[%ss remaining, %s/%s]' % (remaining_s, elapsed_i, total_i))
    return f(*args)
    # XXX WHY ARE WE SWALLOWING THE ERRORS AND THEN NOT PRINTING THEM LATER?!
    # try:
    #     return f(*args)
    # except Exception as e:
    #     return dict(args=args, i=i, e=e, e_tb=traceback.format_exc())

#
# Spectrum / spectrogram
#

dir   = 'data/recordings'
#dir   = 'data/nips4b-wavs'
#dir   = 'data/mlsp-2013-wavs'
#dir   = 'data/warblr-2016-ff1010bird-wavs'
#dir   = 'data/warblr-2016-warblrb10k-wavs'
#dir   = 'data/birdclef-2015-wavs'
files = os.listdir(dir)
paths = sorted([os.path.join(dir, x) for x in files if x.endswith('.wav')])
display(paths)

## XXX Hand-picked examples
#paths = [
#    #'data/recordings/cal towhee.wav',
#    #'data/recordings/Recording 0001.wav',
#    #'data/recordings/white crowned sparrow (2).wav',
#    #'data/recordings/crow, not phoebe, hummingbird.wav',
#    #'data/recordings/chickadee funny noise, other chirp song sparrow.wav',
#    'data/mlsp-2013-wavs/PC10_20090513_054500_0010.wav',
#    #'data/mlsp-2013-wavs/PC10_20090513_054500_0020.wav',
#    #'data/mlsp-2013-wavs/PC10_20090606_054500_0020.wav',
#]

def spec_path(wav_path):
    return wav_path.replace('data/', 'data/spec/', 1) + '.png'

def make_spec(wav_path):

    # Use wavio because scipy.io.wavfile.read(wav_path) spin-hangs on metadata chunks
    #   - Also prints warning "WavFileWarning: Chunk (non-data) not understood, skipping it."
    #   - http://stackoverflow.com/a/14348968/397334
    wav = wavio.read(wav_path)
    (w_shape0, w_shape1) = wav.data.shape # Not sure why it's 2D; assume it's (N,1) and squash to 1D
    assert w_shape1 == 1
    signal      = wav.data.reshape((w_shape0,))
    sample_rate = wav.rate

    # Spectrogram
    #   - Save .png alongside .wav (scipy.sig.spectrogram + plot_img)
    #   - Make img from scipy.sig.spectrogram using matplotlib.pyplot.specgram impl as reference:
    #   - TODO Continue diffing with plt.specgram to make sure we aren't doing anything dumb:
    #       - https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/pyplot.py#L3205
    #       - https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/axes/_axes.py#L7090
    #       - (Sxx, f, t, im) = plt.specgram(signal, NFFT=nperseg, noverlap=noverlap, sides='onesided')
    nperseg  = 256
    noverlap = nperseg // 2
    (f, t, Sxx) = sig.spectrogram(
        x               = signal,
        fs              = sample_rate,
        nperseg         = nperseg,
        noverlap        = noverlap,
        return_onesided = True,           # (Real valued, two-sided is symmetric)
        #window         = ('tukey', .25), # Default
        window          = 'hann',         # Default for plt.specgram
    )
    img = np.flipud(Sxx)
    img = np.log10(img)
    img = scipy.misc.imresize(img, (400, 800))
    wav_spec_path = spec_path(wav_path)
    print(f'Rendering: {wav_spec_path} ...')
    with potoo.mpl_backend_xee.override_fig_path(wav_spec_path):
        print('foo')  # XXX
        plot_img(img)
        print('foo')  # XXX

print('Filtering %s total wav paths to find missing specs...' % len(paths))
paths_missing = (dask.bag.from_sequence(paths)
    .filter(lambda x: not os.path.exists(spec_path(x)))
    .compute()
)
print('Making %s/%s missing specs...' % (len(paths_missing), len(paths)))
errs = (dask.bag
    .from_sequence(enumerate(paths_missing))
    .map(lambda i_x: with_progress(make_spec, len(paths_missing), time.time(), .25, *i_x))
    .filter(lambda x: x is not None)
    .compute()
)
errs

## Signal
#signal_df      = pd.DataFrame()
#signal_df['t'] = np.arange(len(signal)) / float(len(signal))
#signal_df['y'] = signal
#plot_gg(gg_layer(
#    gg.ggplot(signal_df, gg.aes(x='t', y='y')),
#    gg.geom_line(size=.1),
#    gg_tight(),
#    gg.ggtitle('signal: %s' % wav_path),
#))

## Spectrum
#spectrum_df      = pd.DataFrame()
#spectrum_df['Y'] = np.fft.fft(signal_df['y']) / len(signal_df['y'])                # fft (with 1/n normalization)
#spectrum_df['f'] = np.arange(len(spectrum_df['Y'])) / float(len(spectrum_df['Y'])) # freq
#spectrum_df['a'] = np.abs(spectrum_df['Y'])                                        # amplitude
#spectrum_df['p'] = np.abs(spectrum_df['Y'])**2                                     # power
#spectrum_h_df    = spectrum_df[:len(spectrum_df)/2]                                # positive half (real signal -> hermitian spectrum)
#plot_gg(gg_layer(
#    gg.ggplot(spectrum_h_df, gg.aes(x='f', y='a')),
#    gg.geom_point(size=10, alpha=.2),
#    gg_tight(),
#    gg.ggtitle('spectrum: %s' % wav_path),
#))
