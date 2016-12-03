import os
import pipes
import pp as _pp


get_cols = lambda: int(os.popen('stty size').read().split()[1])
pp       = lambda x: _pp         (x, width=get_cols(), indent=2)
pformat  = lambda x: _pp.pformat (x, width=get_cols(), indent=2)


def mkdir_p(dir):
    os.system("mkdir -p %s" % pipes.quote(dir))  # Don't error like os.makedirs


def flatten(xss):
    return [x for xs in xss for x in xs]
