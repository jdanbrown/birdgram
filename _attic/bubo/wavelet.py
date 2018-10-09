#
# cf. stft.py -- extracted from there, might benefit from re-syncing
#

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
import scipy
import scipy.io.wavfile as wav
import scipy.signal as sig

# XXX dev: repl workflow
exec '; '.join(['import %s; reload(%s)' % (m, m) for m in [
    'bubo.util',
    'potoo.dynvar',
    'potoo.mpl_backend_xee',
]])

import potoo.mpl_backend_xee
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
