import numpy as np
import pytest

from _utils import (
    TEST_CUTOFFS,
    TEST_SYSTEMS,
    algorithm_is_applicable,
    canonicalize_half_pairs,
)
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
@pytest.mark.parametrize("cutoff", TEST_CUTOFFS)
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
    if not full_list:
        cpu_i, cpu_j, cpu_S, cpu_D, cpu_d = canonicalize_half_pairs(
            cpu_i, cpu_j, cpu_S, cpu_D, cpu_d
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

    gpu_i, gpu_j, gpu_S, gpu_D, gpu_d = (
        cp.asnumpy(gpu_i),
        cp.asnumpy(gpu_j),
        cp.asnumpy(gpu_S),
        cp.asnumpy(gpu_D),
        cp.asnumpy(gpu_d),
    )

    if not full_list:
        gpu_i, gpu_j, gpu_S, gpu_D, gpu_d = canonicalize_half_pairs(
            gpu_i, gpu_j, gpu_S, gpu_D, gpu_d
        )

    gpu_i, gpu_j, gpu_S, gpu_D, gpu_d = sort_neighbors(
        gpu_i, gpu_j, gpu_S, gpu_D, gpu_d
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


@pytest.mark.parametrize("system", TEST_SYSTEMS, ids=[s.name for s in TEST_SYSTEMS])
@pytest.mark.parametrize("gpu_algorithm", ["cell_list", "brute_force", "verlet"])
def test_gpu_padded_matches_cpu(system, gpu_algorithm):
    cutoff = 5.0
    if gpu_algorithm == "verlet":
        algorithm, skin = "cell_list", 0.5
    else:
        algorithm, skin = gpu_algorithm, 0.0

    if not algorithm_is_applicable(system, cutoff, algorithm):
        return

    cpu = NeighborList(cutoff=cutoff, full_list=True, padded_to=256)
    cpu_i, cpu_j, cpu_S, cpu_D, cpu_d = sort_neighbors(
        *cpu.compute(system.points, system.box, system.periodic, "ijSDd")
    )

    gpu = NeighborList(
        cutoff=cutoff,
        full_list=True,
        padded_to=256,
        algorithm=algorithm,
        skin=skin,
    )
    # run twice to also go through the cached Verlet candidates
    for _ in range(2):
        outputs = gpu.compute(
            points=cp.asarray(system.points, dtype=cp.float64),
            box=cp.asarray(system.box, dtype=cp.float64),
            periodic=cp.asarray(system.periodic),
            quantities="ijSDd",
        )
        outputs = [cp.asnumpy(output) for output in outputs]

        # each point gets exactly `padded_to` consecutive entries
        n_points = len(system.points)
        assert np.array_equal(outputs[0], np.repeat(np.arange(n_points), 256))

        gpu_i, gpu_j, gpu_S, gpu_D, gpu_d = sort_neighbors(*outputs)

        message = f"padded neighbors do not match between CPU and GPU for {system.name}"
        assert np.array_equal(cpu_i, gpu_i), message
        assert np.array_equal(cpu_j, gpu_j), message
        assert np.array_equal(cpu_S, gpu_S), message
        assert np.allclose(cpu_D, gpu_D), message
        assert np.allclose(cpu_d, gpu_d), message
