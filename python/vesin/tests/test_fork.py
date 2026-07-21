import multiprocessing as mp
import sys
import threading

import numpy as np
import pytest

import vesin


def _compute():
    """Compute neighbors"""
    rng = np.random.default_rng(42)
    points = rng.uniform(0, 20, size=(200, 3)).astype(np.float64)
    box = 20.0 * np.eye(3, dtype=np.float64)

    nl = vesin.NeighborList(
        cutoff=2.5,
        full_list=True,
        algorithm="cell_list",
    )
    nl.compute(
        points=points,
        box=box,
        periodic=[True, True, True],
        quantities="ij",
    )


@pytest.mark.timeout(10)
def test_fork_during_calculations():
    """Regression test: fork safety of the thread pool"""

    if sys.platform == "win32":
        pytest.skip("fork is not available on Windows")

    mp.set_start_method("fork", force=True)

    # Load the library and create the thread pool before starting the hammer
    # thread / forking. This ensures `pthread_atfork` handlers are registered.
    _compute()

    # Keep computing on the main process until `stop` is set. This forces the
    # calculation to create and hold mutex like for a large scale data loader
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            _compute()

    thread = threading.Thread(target=hammer, daemon=True)
    thread.start()

    # Fork and compute in child
    proc = mp.Process(target=_compute)
    proc.start()
    try:
        proc.join()

        stop.set()
        thread.join(1)

        if proc.exitcode != 0:
            pytest.fail(f"Child process exited with code {proc.exitcode}")

    finally:
        if proc.is_alive():
            proc.kill()
            proc.join(1)

        if thread.is_alive():
            stop.set()
            thread.join(1)
