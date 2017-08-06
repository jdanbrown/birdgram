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

# XXX dev: repl workflow
exec '; '.join(['import %s; reload(%s)' % (m, m) for m in [
    'bubo.util',
    'potoo.dynvar',
    'potoo.mpl_backend_xee',
]])

from bubo.util import caffe_root, plot_img, plot_gg, show_shapes, show_tuple_tight
from bubo.util import gg_layer, gg_xtight, gg_ytight, gg_tight
from bubo.util import shell, singleton, puts

#
# Signal / spectrum
#

# Signal (time domain)
signal      = pd.DataFrame()
Fs          = 150.0                            # sampling rate
Ts          = 1.0/Fs                           # sampling interval
signal['t'] = np.arange(0, 1, Ts)              # time vector
ff          = 5                                # frequency of the signal
signal['y'] = np.sin(2*np.pi*ff * signal['t']) # the signal
n           = len(signal['y'])                 # length of the signal

# Spectrum (freq domain)
spectrum       = pd.DataFrame()
k              = np.arange(n)
T              = n/Fs
spectrum['f']  = k/T                       # frequency range
spectrum['Y']  = np.fft.fft(signal['y'])/n # fft (with 1/n normalization)
spectrum['a']  = np.abs(spectrum['Y'])     # amplitude spectrum
spectrum['p']  = np.abs(spectrum['Y'])**2  # power spectrum
spectrum_h     = spectrum[:n/2]            # positive half (real signal -> hermitian spectrum)

plot_gg(gg_layer(
    gg.ggplot(signal, gg.aes(x='t', y='y')),
    gg.geom_point(),
))

plot_gg(gg_layer(
    gg.ggplot(spectrum_h, gg.aes(x='f', y='a')),
    gg.geom_point(),
))
