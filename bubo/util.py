import matplotlib.pyplot as plt
import os
import sys

caffe_root = 'caffe-root'

def shell(cmd):
    print >>sys.stderr, 'shell: cmd[%s]' % cmd
    status = os.system(cmd)
    if status != 0:
        raise Exception('Exit status[%s] from cmd[%s]' % (status, cmd))

def plt_show():
    plt.tight_layout()
    plt.show()
