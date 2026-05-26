import numpy as np
import pytest

from _utils import TEST_SYSTEMS, algorithm_is_applicable
from vesin import NeighborList


cp = pytest.importorskip("cupy")

try:
    cp.cuda.Device(0).compute_capability
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA is not available", allow_module_level=True)


def sort_neighbors(i, j, S, D, d):
    ijS = np.concatenate((i.reshape(-1, 1), j.reshape(-1, 1), S), axis=1)
    sort_indices = np.lexsort(np.flip(ijS, axis=1).T)
    return (
        i[sort_indices],
        j[sort_indices],
        S[sort_indices],
        D[sort_indices],
        d[sort_indices],
    )


def compute_neighbors(points, box, periodic, cutoff, full_list, algorithm, sorted):
    calculator = NeighborList(
        cutoff=cutoff,
        full_list=full_list,
        sorted=sorted,
        algorithm=algorithm,
    )
    return calculator.compute(
        points=points,
        box=box,
        periodic=periodic,
        quantities="ijSDd",
    )


@pytest.mark.parametrize("full_list", [False, True])
@pytest.mark.parametrize("system", TEST_SYSTEMS, ids=[s.name for s in TEST_SYSTEMS])
@pytest.mark.parametrize("cutoff", [3.0, 5.0, 7.0, 10.0])
@pytest.mark.parametrize("gpu_algorithm", ["cell_list", "brute_force"])
@pytest.mark.parametrize("sorted", [False, True])
@pytest.mark.parametrize("transform", ["none", "rotate", "rotate_and_translate"])
def test_gpu_matches_cpu(system, cutoff, full_list, gpu_algorithm, sorted, transform):
    if not algorithm_is_applicable(system, cutoff, gpu_algorithm):
        return

    if transform == "none":
        pass
    elif transform == "rotate":
        system = system.transform(translate=False)
    elif transform == "rotate_and_translate":
        system = system.transform(translate=True)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    cpu_i, cpu_j, cpu_S, cpu_D, cpu_d = compute_neighbors(
        points=system.points,
        box=system.box,
        periodic=system.periodic,
        cutoff=cutoff,
        full_list=full_list,
        algorithm="cell_list",
        sorted=sorted,
    )
    cpu_i, cpu_j, cpu_S, cpu_D, cpu_d = sort_neighbors(
        cpu_i,
        cpu_j,
        cpu_S,
        cpu_D,
        cpu_d,
    )

    gpu_i, gpu_j, gpu_S, gpu_D, gpu_d = compute_neighbors(
        points=cp.asarray(system.points, dtype=cp.float64),
        box=cp.asarray(system.box, dtype=cp.float64),
        periodic=cp.asarray(system.periodic),
        cutoff=cutoff,
        full_list=full_list,
        algorithm=gpu_algorithm,
        sorted=sorted,
    )

    gpu_i, gpu_j, gpu_S, gpu_D, gpu_d = sort_neighbors(
        cp.asnumpy(gpu_i),
        cp.asnumpy(gpu_j),
        cp.asnumpy(gpu_S),
        cp.asnumpy(gpu_D),
        cp.asnumpy(gpu_d),
    )

    message = (
        f"Neighbors do not match between CPU and GPU for {system.name}, "
        f"transformed with {system.transform_summary}."
    )

    assert np.array_equal(cpu_i, gpu_i), message
    assert np.array_equal(cpu_j, gpu_j), message
    assert np.array_equal(cpu_S, gpu_S), message
    assert np.allclose(cpu_D, gpu_D), message
    assert np.allclose(cpu_d, gpu_d), message
