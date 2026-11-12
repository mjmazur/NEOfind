"""Run independent CPU work (one function over many items) on several cores.

``parallel_map(fn, items, threads)`` returns ``[fn(x) for x in items]``, in
order, computed by forked worker processes. ``fn`` may be a closure: the
workers are forked after it is set, so they inherit it (and everything it
refers to -- the frames, the evaluator) without copying or pickling; only the
items' indices go out and the results come back pickled. Each result is
computed exactly as it would be in this process, so the output is the same
as the serial loop's.

Workers must not touch the GPU (a forked child can't use its parent's CUDA
context); the track fits this is for are CPU only.
"""

import multiprocessing as mp
import os
import signal
import warnings

# Below this many items the serial loop is quicker than starting workers.
MIN_ITEMS = 8

_job = None  # (fn, items) inherited by the forked workers


def _init_worker():
    # A worker inherits the parent's Python signal handlers (the campaign's
    # SIGTERM handler only sets a stop flag): with them an idle worker would
    # outlive a SIGTERM and the parent wait for it forever.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def _call(i):
    fn, items = _job
    return fn(items[i])


def n_workers(threads: int = 0) -> int:
    return threads if threads > 0 else (os.cpu_count() or 1)


def parallel_map(fn, items, threads: int = 0):
    items = list(items)
    procs = min(n_workers(threads), len(items))
    if procs < 2 or len(items) < MIN_ITEMS or "fork" not in mp.get_all_start_methods():
        return [fn(x) for x in items]
    global _job
    _job = (fn, items)
    try:
        with warnings.catch_warnings():
            # Python 3.12+ warns about fork() in a process with threads (CuPy,
            # numba): the children only run numpy/scipy code.
            warnings.simplefilter("ignore", DeprecationWarning)
            pool = mp.get_context("fork").Pool(procs, initializer=_init_worker)
        try:
            out = pool.map(_call, range(len(items)), chunksize=1)
            pool.close()  # workers finish and exit by themselves
            pool.join()
            return out
        except BaseException:
            pool.terminate()
            raise
    finally:
        _job = None
