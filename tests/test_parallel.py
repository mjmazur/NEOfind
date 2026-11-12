import numpy as np

from neodetect.parallel import parallel_map


def test_parallel_map_matches_serial_and_keeps_order():
    data = np.random.default_rng(0).normal(size=(50, 30))  # inherited by the forked workers

    def fn(i):  # a closure: not picklable, reached through fork
        return float(np.sort(data[i]).cumsum()[-1]), i

    items = list(range(50))
    assert parallel_map(fn, items, threads=4) == [fn(i) for i in items]
    assert parallel_map(fn, items[:3], threads=4) == [fn(i) for i in items[:3]]  # serial path
    assert parallel_map(fn, [], threads=4) == []


def test_parallel_map_returns_with_a_python_sigterm_handler_installed():
    """The campaign installs a SIGTERM handler that only sets a flag; workers
    inheriting it once made the pool wait forever for an idle worker."""
    import signal
    stop = []
    old = signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    old_alarm = signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError("pool hung")))
    signal.alarm(60)
    try:
        for _ in range(5):
            assert parallel_map(lambda i: i * i, list(range(40)), threads=8) == [i * i for i in range(40)]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGTERM, old)
    assert not stop
