from contextlib import contextmanager

from potoo.plot import *
from potoo.util import *
from potoo.util import singleton


caffe_root = 'caffe-root'


def show_tuple_tight(xs):
    return '(%s)' % ','.join(map(str, xs))


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
                print('%s.show_shapes:%s' % (desc, self._show_compressed_shapes(1, self._compress_seq_pairs(self.shapes))))
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
            (n, x) = compressed_shapes
            s      = self._show_compressed_shapes(level, x)
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
