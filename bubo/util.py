from contextlib import contextmanager
from copy import deepcopy
import ggplot as gg
import matplotlib as mpl
import matplotlib.image
import matplotlib.pyplot as plt
import os
import sys

import bubo.mpl_backend_xee

caffe_root = 'caffe-root'

def shell(cmd):
    print >>sys.stderr, 'shell: cmd[%s]' % cmd
    status = os.system(cmd)
    if status != 0:
        raise Exception('Exit status[%s] from cmd[%s]' % (status, cmd))

def singleton(cls):
    return cls()

def show_tuple_tight(xs):
    return '(%s)' % ','.join(map(str, xs))

def puts(x):
    print x
    return x

class gg_xtight(object):
    def __init__(self, margin=0.05):
        self.margin = margin
    def __radd__(self, g):
        g          = deepcopy(g)
        xs         = g.data[g._aes['x']]
        lims       = [xs.min(), xs.max()]
        margin_abs = float(self.margin) * (lims[1] - lims[0])
        g.xlimits  = [xs.min() - margin_abs, xs.max() + margin_abs]
        return g

class gg_ytight(object):
    def __init__(self, margin=0.05):
        self.margin = margin
    def __radd__(self, g):
        g          = deepcopy(g)
        ys         = g.data[g._aes['y']]
        lims       = [ys.min(), ys.max()]
        margin_abs = float(self.margin) * (lims[1] - lims[0])
        g.ylimits  = [ys.min() - margin_abs, ys.max() + margin_abs]
        return g

class gg_tight(object):
    def __init__(self, margin=0.05):
        self.margin = margin
    def __radd__(self, g):
        return g + gg_xtight(self.margin) + gg_ytight(self.margin)

def gg_layer(*args):
    'More uniform syntax than \ and + for many-line layer addition'
    return reduce(lambda a,b: a + b, args)

class gg_theme_keep_defaults_for(gg.theme_gray):
    def __init__(self, *rcParams):
        super(gg_theme_keep_defaults_for, self).__init__()
        for x in rcParams:
            del self._rcParams[x]

def plot_gg(ggplot):
    ggplot += gg_theme_keep_defaults_for('figure.figsize')
    with bubo.mpl_backend_xee.basename_suffix(ggplot.title):
        repr(ggplot) # (Over)optimized for repl/notebook usage (repr(ggplot) = ggplot.make(); plt.show())
    #return ggplot # Don't return to avoid plotting a second time if repl/notebook

def plot_img(data, basename_suffix=''):
    with bubo.mpl_backend_xee.basename_suffix(basename_suffix):
        return bubo.mpl_backend_xee.imsave_xee(data)

def plot_img_via_imshow(data):
    'Makes lots of distorted pixels, huge PITA, use imsave/plot_img instead'
    (h,w) = data.shape[:2] # (h,w) | (h,w,3)
    dpi   = 100
    k     = 1 # Have to scale this up to ~4 to avoid distorted pixels
    with tmp_rcParams({
        'image.interpolation': 'nearest',
        'figure.figsize':      puts((w/float(dpi)*k, h/float(dpi)*k)),
        'savefig.dpi':         dpi,
    }):
        img = plt.imshow(data)
        img.axes.get_xaxis().set_visible(False) # Don't create padding for axes
        img.axes.get_yaxis().set_visible(False) # Don't create padding for axes
        plt.axis('off')                         # Don't draw axes
        plt.tight_layout(pad=0)                 # Don't add padding
        plt.show()

@contextmanager
def tmp_rcParams(kw):
    _save_mpl = mpl.RcParams(**mpl.rcParams)
    _save_plt = mpl.RcParams(**plt.rcParams)
    try:
        mpl.rcParams.update(kw)
        plt.rcParams.update(kw) # TODO WHY ARE THESE DIFFERENT
        yield
    finally:
        mpl.rcParams = _save_mpl
        plt.rcParams = _save_plt

# Gross and horrible, but useful to help me understand how np.array shapes change as we transform them
@singleton
class show_shapes:

    def __init__(self):
        self.shapes = None

    @contextmanager
    def bracketing(self, desc, data, disable=False):
        if disable:
            yield
        else:
            self.shapes = []
            self.shapes.append(self._format_shape(data.shape))
            try:
                yield
                print '%s.show_shapes:%s' % (desc, self._show_compressed_shapes(1, self._compress_seq_pairs(self.shapes)))
            finally:
                self.shapes = None

    def __call__(self, desc, data):
        if self.shapes is not None:
            self.shapes.append('%-20s%s' % (self._format_shape(data.shape), desc))
        return data

    def _format_shape(self, shape):
        return '(%s)' % ','.join(map(str, shape))

    # type compressed_shape = str | (compressed_shape, int) | [compressed_shape, ...]
    def _show_compressed_shapes(self, level, compressed_shapes):
        if isinstance(compressed_shapes, str):
            return compressed_shapes
        elif isinstance(compressed_shapes, tuple):
            (n,x) = compressed_shapes
            s     = self._show_compressed_shapes(level, x)
            return s if n == 1 else '%s%s%s' % (n, s, '  '*(level-1))
        else:
            return self._show_shapes(level, [self._show_compressed_shapes(level+1, x) for x in compressed_shapes])

    def _show_shapes(self, level, shapes):
        return ''.join(['\n%s%s' % ('  '*level, x) for x in shapes])

    # Algorithmically easy and sufficient for our use cases so far; generalize only as necessary
    #   - returns compressed_shape
    def _compress_seq_pairs(self, xs):
        ys = []
        i  = 0
        while i < len(xs):
            n_seq_pairs = 1
            j = i + 2
            while j+1 < len(xs) and (xs[i], xs[i+1]) == (xs[j], xs[j+1]):
                n_seq_pairs += 1
                j += 2
            if n_seq_pairs == 1:
                ys.append((1, (xs[i])))
                i += 1
            else:
                ys.append((n_seq_pairs, ([xs[i], xs[i+1]])))
                i += n_seq_pairs * 2
        return ys
