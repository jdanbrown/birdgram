"""
Slight variant of https://github.com/dask/zict/blob/master/zict/buffer.py
"""

# Copyright (c) 2018 Matthew Rocklin, Dan Brown
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of toolz nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

from itertools import chain

from zict.common import ZictBase, close
from zict.lru import LRU


class PersistentBuffer(ZictBase):
    """
    A write-through variant of zict.Buffer that uses slow as a persistence layer:
    - All writes write to both fast and slow
    - Nothing deletes from slow except for __delitem__

    See zict.Buffer for more details.
    """

    def __init__(self, fast, slow, n, weight=lambda k, v: 1,
                 evict_fast_callbacks=None, load_fast_callbacks=None):
        self.fast = LRU(n, fast, weight=weight, on_evict=[self.evict_fast])
        self.slow = slow
        self.n = n
        self.weight = weight
        if callable(evict_fast_callbacks):
            evict_fast_callbacks = [evict_fast_callbacks]
        if callable(load_fast_callbacks):
            load_fast_callbacks = [load_fast_callbacks]
        self.evict_fast_callbacks = evict_fast_callbacks or []
        self.load_fast_callbacks = load_fast_callbacks or []

    def evict_fast(self, key, value):
        for cb in self.evict_fast_callbacks:
            cb(key, value)

    def load_fast(self, key):
        value = self.slow[key]
        if self.weight(key, value) <= self.n:
            self.fast[key] = value
        for cb in self.load_fast_callbacks:
            cb(key, value)
        return value

    def __getitem__(self, key):
        if key in self.fast:
            return self.fast[key]
        elif key in self.slow:
            return self.load_fast(key)
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        self.slow[key] = value
        if self.weight(key, value) <= self.n:
            self.fast[key] = value

    def __delitem__(self, key):
        if key in self.fast:
            del self.fast[key]
        if key in self.slow:
            del self.slow[key]
        if key not in self.fast and key not in self.slow:
            raise KeyError(key)

    def keys(self):
        return chain(self.fast.keys(), self.slow.keys())

    def values(self):
        return chain(self.fast.values(), self.slow.values())

    def items(self):
        return chain(self.fast.items(), self.slow.items())

    def __len__(self):
        return len(self.fast) + len(self.slow)

    def __iter__(self):
        return chain(iter(self.fast), iter(self.slow))

    def __contains__(self, key):
        return key in self.fast or key in self.slow

    def __str__(self):
        return 'PersistentBuffer<%s, %s>' % (str(self.fast), str(self.slow))

    __repr__ = __str__

    def flush(self):
        self.fast.flush()
        self.slow.flush()

    def close(self):
        close(self.fast)
        close(self.slow)
