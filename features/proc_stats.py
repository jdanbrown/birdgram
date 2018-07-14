from collections import OrderedDict
import contextlib
import gc
import os
import threading
import time
from typing import Callable, Iterable, List

from attrdict import AttrDict
from dataclasses import dataclass, field
import pandas as pd
from plotnine import *
from potoo.util import or_else
import psutil

from util import DataclassUtil, none_or, X


@dataclass(repr=False)
class ProcStats(DataclassUtil):

    interval: float
    stats: pd.DataFrame = None

    def __enter__(self) -> 'ProcStats':
        self.start()
        return self

    def __exit__(self, *exc_details):
        self.stop()

    def start(self):
        """
        Mem overhead: ~370B per sample
        - Methodology: sum(len(pickle.dumps(x)) for x in proc_stats._stats) / len(proc_stats._stats)
        """
        proc = psutil.Process(os.getpid())
        # GC once before we start to calibrate mem usage
        gc.collect()
        self._stats = []
        self._poll = Poll(
            interval=self.interval,
            run=lambda: self._stats.extend((
                oneshot(p, lambda p: dict(
                    time=pd.Timestamp.now(),
                    role=role,
                    pid=p.pid,
                    ppid=p.ppid(),
                    cpu_user=p.cpu_times().user,
                    cpu_system=p.cpu_times().system,
                    **{'mem_' + k: v for k, v in p.memory_info()._asdict().items()},
                    mem_pct_rss=p.memory_percent('rss'),
                    io_rn=or_else(None, lambda: p.io_counters().read_count),
                    io_wn=or_else(None, lambda: p.io_counters().write_count),
                    io_rb=or_else(None, lambda: p.io_counters().read_bytes),
                    io_wb=or_else(None, lambda: p.io_counters().write_bytes),
                    **{'ctx_' + k: v for k, v in p.num_ctx_switches()._asdict().items()},
                    fds=p.num_fds(),
                    threads=p.num_threads(),
                ))
                for role, p in [
                    ('parent', proc),
                    *[('child', p) for p in proc.children(recursive=True)],
                ]
            )),
        )
        self._poll.start()

    def stop(self):
        self._poll.stop()
        self._poll = None  # Avoid pickling Poll.run, which will fail on e.g. lambdas
        self.stats = self._freeze(self._stats)
        self._stats = None

    def _freeze(self, stats: Iterable[dict]) -> pd.DataFrame:
        """Convert .stats from List[dict] to df, because I don't know an efficient way to accumulate a df directly"""
        return pd.DataFrame(
            OrderedDict(d)  # To preserve dict key ordering for pandas
            for d in stats
        )

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join([
            'interval=%r' % self.interval,
            'stats[%s]' % none_or(self.stats, len),
        ]))


def oneshot(proc: psutil.Process, f: Callable[[psutil.Process], X]) -> X:
    """Process.oneshot as an expression instead of a statement"""
    with proc.oneshot():
        return f(proc)


@dataclass
class Poll(DataclassUtil):

    interval: float
    run: Callable[[], None]

    _thread = None
    _stop = None

    def __enter__(self) -> 'Poll':
        self.start()
        return self

    def __exit__(self, *exc_details):
        self.stop()

    def start(self) -> 'self':
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        return self

    def stop(self) -> 'self':
        self._stop.set()
        self._thread.join()
        self._stop = None
        self._thread = None
        return self

    def _run(self):
        while True:
            if self._stop.is_set():
                break
            self.run()
            time.sleep(self.interval)
