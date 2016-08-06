# coding=utf8
# Based on https://github.com/oselivanov/matplotlib_iterm2/blob/master/matplotlib_iterm2/backend_iterm2.py

"""
Based on https://github.com/oselivanov/matplotlib_iterm2/blob/master/matplotlib_iterm2/backend_iterm2.py

Test:
    $ MPLBACKEND=module://bubo.mpl_backend_xee ipy -c 'from pylab import *; plot([1,2,3]); show()'
"""

# http://matplotlib.org/devel/coding_guide.html#developing-a-new-backend
# http://matplotlib.org/users/customizing.html

from datetime import datetime
import os.path
import platform
import subprocess
import sys

from matplotlib._pylab_helpers import Gcf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.figure import Figure
import matplotlib.image
import matplotlib.pyplot
from PIL import Image
import re

from bubo.dynvar import DynVar

# TODO Where to put these? Any way to get custom keys into matplotlibrc?
#   - https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/rcsetup.py#L885
#   - https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/__init__.py#L1119
_rcParams = {
    'xee.path':                     'data/figs',
    'xee.show_via':                 'savefig', # savefig | canvas
    'xee.platform.Darwin.open_cmd': 'open -g -a Xee³ %(fig_path)s',
    'xee.platform.Linux.open_cmd':  '',
}

def show():
    for manager in Gcf.get_all_fig_managers():
        manager.show()
        Gcf.destroy(manager.num) # Else every call to figure() will add an extra image produced by every future show()

def new_figure_manager(num, *args, **kwargs):
    return FigureManagerXee(
        FigureCanvasAgg(kwargs.pop('FigureClass', Figure)(*args, **kwargs)),
        num,
    )

class FigureManagerXee(FigureManagerBase):

    def show(self):
        fig_path = new_fig_path()
        print 'mpl_backend_xee: %s' % fig_path
        if _rcParams['xee.show_via'] == 'savefig':
            self._show_via_savefig(fig_path)
        else:
            self._show_via_canvas(fig_path)
        open_fig(fig_path)

    # Configure via `savefig.*` rcParams: http://matplotlib.org/users/customizing.html
    def _show_via_savefig(self, fig_path):
        matplotlib.pyplot.savefig(fig_path)

    # Configure via `figure.*` rcParams: http://matplotlib.org/users/customizing.html
    def _show_via_canvas(self, fig_path):
        self.canvas.draw()
        (w,h) = (int(self.canvas.get_renderer().width), int(self.canvas.get_renderer().height))
        img   = Image.frombuffer('RGBA', (w,h), self.canvas.buffer_rgba(), 'raw', 'RGBA', 0, 1)
        img.save(fig_path)

    # TODO Maybe bad to call multiple times?
    def close(self):
        Gcf.destroy(self.num)

FigureManager = FigureManagerXee

basename_suffix = DynVar(None)

def new_fig_path():
    figs_dir = _rcParams['xee.path']
    os.system("mkdir -p '%s'" % figs_dir) # Don't error like os.makedirs
    return os.path.join(
        figs_dir,
        '%s.png' % '-'.join(filter(lambda x: x, [
            'fig',
            re.sub('[:.-]', '', datetime.utcnow().isoformat()),
            re.sub('[\s/]', '-', (basename_suffix.value() or '')).lower(),
        ]))
    )

def open_fig(fig_path):
    open_cmd = _rcParams.get('xee.platform.%s.open_cmd' % platform.system())
    if open_cmd:
        cmd = open_cmd % locals()
        try:
            subprocess.call(cmd, shell=True)
        except Exception, e:
            print >>sys.stderr, '[%s] Failed to run cmd[%s]: %s' % (__name__, cmd, e)

# TODO Sometimes trims last col, e.g. data.shape[(309,253)] produces an image with WxH[252x309]
def imsave_xee(data):
    fig_path = new_fig_path()
    print 'img_backend_xee: %s' % fig_path
    img = matplotlib.image.imsave(fig_path, data)
    open_fig(fig_path)
    return img
