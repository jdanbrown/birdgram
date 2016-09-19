import caffe
from contextlib import contextmanager
import ggplot as gg
import ipdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pprint import pprint
import pywt
import scipy.io.wavfile as wav
import scipy.signal as sig

# XXX dev: repl workflow
exec '; '.join(['import %s; reload(%s)' % (m, m) for m in [
    'bubo.util',
    'bubo.dynvar',
    'bubo.mpl_backend_xee',
]])

import bubo.mpl_backend_xee
from bubo.util import caffe_root, plot_img, plot_gg, show_shapes, show_tuple_tight, tmp_rcParams
from bubo.util import gg_layer, gg_xtight, gg_ytight, gg_tight
from bubo.util import shell, singleton, puts

#
# Paths
#

dir   = 'data/recordings'
files = os.listdir(dir)
paths = sorted([os.path.join(dir, x) for x in files if x.endswith('.wav')])
pprint(paths)

#
# Wavelets: non-working attempts...
#   - Illustration of various wavelets:
#       - http://wavelets.pybytes.com/wavelet/dmey/
#   - Pretty plot of a wavelet transform:
#       - https://docs.obspy.org/tutorial/code_snippets/continuous_wavelet_transform.html
#   - Example of what bird call transformed through wavelets might should look like:
#       - https://www.researchgate.net/publication/26620207_Wavelets_in_Recognition_of_Bird_Sounds
#   - Other informative papers:
#       - http://www.ncbi.nlm.nih.gov/pmc/articles/PMC4728069/
#       - http://arxiv.org/abs/1311.4764
#   - http://pythonhosted.org/PyGASP/
#       - https://bitbucket.org/bowmanat/pygasp/src/master/pygasp/dwt.py
#

path = '%(dir)s/white crowned sparrow (2).wav' % locals()

from pygasp.dwt import dwt

# Very slow...
#rate, signal = wav.read(path)
#signal_dwt = dwt.dwt(signal, wav='db3', levels=5, mode='zpd')
#signal_dwt

# Never terminates...
#dwt.scalogram(signal_dwt)

# http://stackoverflow.com/questions/16482166/basic-plotting-of-wavelet-analysis-output-in-matplotlib
# - Maybe also try https://github.com/nigma/pywt/blob/master/demo/wp_scalogram.py
path = '%(dir)s/white crowned sparrow (2).wav' % locals()
def f0():

    import pylab
    import scipy.io.wavfile as wavfile

    # Find the highest power of two less than or equal to the input.
    def lepow2(x):
        return 2 ** pylab.floor(pylab.log2(x))

    # Make a scalogram given an MRA tree.
    def scalogram(data):
        bottom = 0

        vmin = min(map(lambda x: min(abs(x)), data))
        vmax = max(map(lambda x: max(abs(x)), data))

        pylab.gca().set_autoscale_on(False)

        for row in range(0, len(data)):
            scale = 2.0 ** (row - len(data))

            pylab.imshow(
                pylab.array([abs(data[row])]),
                interpolation = 'nearest',
                vmin = vmin,
                vmax = vmax,
                extent = [0, 1, bottom, bottom + scale])

            bottom += scale

    # Load the signal, take the first channel, limit length to a power of 2 for simplicity.
    rate, signal = wavfile.read(path)
    signal = signal[0:lepow2(len(signal))]
    tree = pywt.wavedec(signal, 'db10')

    #pylab.gray()
    scalogram(tree)
    pylab.show()

#f0()

# http://www.pybytes.com/pywavelets/ref/dwt-discrete-wavelet-transform.html
#   - http://wavelets.pybytes.com/wavelet/dmey/
#   - TODO Plot individual wavelets:
#       - http://stackoverflow.com/questions/1094655/wavelet-plot-with-python-libraries
#cA, cD = pywt.dwt(signal, pywt.Wavelet('db1'))
#print cA
#print cD
#print len(signal), len(cA), len(cD)

if False:
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html
    # http://docs.scipy.org/doc/scipy/reference/signal.html
    path = '%(dir)s/white crowned sparrow (2).wav' % locals()
    print path
    sample_rate, signal = wav.read(path)
    signal              = signal[450000:480000]
    cwtmatr             = sig.cwt(signal, sig.ricker, widths=np.arange(1, 31))
    print cwtmatr.shape
    plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.show()

#
# Spectrum / spectrogram
#

# XXX Hand-picked examples
paths = [
    #'%(dir)s/Recording 0001.wav' % locals(),
    '%(dir)s/white crowned sparrow (2).wav' % locals(),
    #'%(dir)s/crow, not phoebe, hummingbird.wav' % locals(), # TODO Pegs python/docker cpu, doesn't terminate...
    #'%(dir)s/chickadee funny noise, other chirp song sparrow.wav' % locals(), # TODO Pegs python/docker cpu, doesn't terminate...
]

for i, path in enumerate(paths):

    print '(%s/%s) %s' % (i+1, len(paths), path)
    sample_rate, signal = wav.read(path)

    # Spectrogram
    with tmp_rcParams({
        'image.cmap': 'jet' # Undo grayscale from ~/.matplotlib/matplotlibrc
    }):

        ## Save as fig
        #plt.figure(figsize = (10, 5))
        #plt.xlabel('time')
        #plt.ylabel('freq')
        #signal_len      = 8192
        #segment_len     = 256
        #segment_overlap = signal_len/64
        #plt.specgram(signal, NFFT=segment_len, noverlap=segment_overlap, sides='onesided')
        ##cb = plt.colorbar(); cb.set_label('amplitude')
        #plt.autoscale(tight=True)
        #plt.tight_layout()
        #plt.show()

        # Save .png alongside .wav
        #   - TODO For wavs.html, how to kill all padding so that spec fills all the img pixels?
        #       - TODO plt.specgram -> http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
        fig = plt.figure(figsize = (10, 5))
        plt.axes(frameon=False) # Killed by plt.specgram...
        signal_len      = 8192
        segment_len     = 256
        segment_overlap = signal_len/64
        plt.specgram(signal, NFFT=segment_len, noverlap=segment_overlap, sides='onesided')
        plt.autoscale(tight=True)
        plt.tight_layout(pad=0) # (Negative numbers do do something...)
        plt.axis('off')
        #plt.show() # XXX Save as fig for faster dev
        with bubo.mpl_backend_xee.override_fig_path(path + '.spec.png'): # TODO
            plt.show()

    ## Signal
    #signal_df      = pd.DataFrame()
    #signal_df['t'] = np.arange(len(signal)) / float(len(signal))
    #signal_df['y'] = signal
    #plot_gg(gg_layer(
    #    gg.ggplot(signal_df, gg.aes(x='t', y='y')),
    #    gg.geom_line(size=.1),
    #    gg_tight(),
    #    gg.ggtitle('signal: %s' % path),
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
    #    gg.ggtitle('spectrum: %s' % path),
    #))

