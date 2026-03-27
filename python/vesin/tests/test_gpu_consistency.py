import os
from dataclasses import dataclass

import ase.io
import numpy as np
import pytest

from vesin import NeighborList


cp = pytest.importorskip("cupy")

try:
    cp.cuda.Device(0).compute_capability
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA is not available", allow_module_level=True)


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@dataclass(frozen=True)
class SystemForTests:
    name: str
    points: np.ndarray
    box: np.ndarray
    periodic: tuple[bool, bool, bool]


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


def compute_neighbors(points, box, periodic, cutoff, full_list, algorithm):
    calculator = NeighborList(
        cutoff=cutoff,
        full_list=full_list,
        sorted=False,
        algorithm=algorithm,
    )
    return calculator.compute(
        points=points,
        box=box,
        periodic=periodic,
        quantities="ijSDd",
    )


def compare_cpu_gpu(
    case: SystemForTests, cutoff: float, full_list: bool, gpu_algorithm: str
):
    cpu_i, cpu_j, cpu_S, cpu_D, cpu_d = compute_neighbors(
        points=case.points,
        box=case.box,
        periodic=case.periodic,
        cutoff=cutoff,
        full_list=full_list,
        algorithm="cell_list",
    )
    cpu_i, cpu_j, cpu_S, cpu_D, cpu_d = sort_neighbors(
        cpu_i,
        cpu_j,
        cpu_S,
        cpu_D,
        cpu_d,
    )

    gpu_i, gpu_j, gpu_S, gpu_D, gpu_d = compute_neighbors(
        points=cp.asarray(case.points, dtype=cp.float64),
        box=cp.asarray(case.box, dtype=cp.float64),
        periodic=cp.asarray(case.periodic),
        cutoff=cutoff,
        full_list=full_list,
        algorithm=gpu_algorithm,
    )

    gpu_i, gpu_j, gpu_S, gpu_D, gpu_d = sort_neighbors(
        cp.asnumpy(gpu_i),
        cp.asnumpy(gpu_j),
        cp.asnumpy(gpu_S),
        cp.asnumpy(gpu_D),
        cp.asnumpy(gpu_d),
    )

    assert np.array_equal(cpu_i, gpu_i)
    assert np.array_equal(cpu_j, gpu_j)
    assert np.array_equal(cpu_S, gpu_S)
    assert np.allclose(cpu_D, gpu_D)
    assert np.allclose(cpu_d, gpu_d)


def polymer_chain() -> SystemForTests:
    # Carbon backbone with realistic C-C distances for a simple 1D polymer-like chain.
    spacing = 1.54
    points = np.array(
        [[index * spacing, 0.0, 0.0] for index in range(8)], dtype=np.float64
    )
    box = np.diag([8 * spacing, 18.0, 18.0]).astype(np.float64)
    return SystemForTests(
        name="polymer_chain_1d",
        points=points,
        box=box,
        periodic=(True, False, False),
    )


def graphene_sheet() -> SystemForTests:
    # 4x4 graphene supercell in the xy plane with vacuum along z.
    a1 = np.array([2.46, 0.0, 0.0])
    a2 = np.array([1.23, 2.13042249, 0.0])
    basis = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.23, 0.71014083, 0.0]),
    ]

    points = []
    for i in range(4):
        for j in range(4):
            origin = i * a1 + j * a2
            for atom in basis:
                points.append(origin + atom)

    return SystemForTests(
        name="graphene_sheet_2d",
        points=np.asarray(points, dtype=np.float64),
        box=np.array(
            [
                [9.84, 0.0, 0.0],
                [4.92, 8.52168996, 0.0],
                [0.0, 0.0, 18.0],
            ],
            dtype=np.float64,
        ),
        periodic=(True, True, False),
    )


def diamond_crystal() -> SystemForTests:
    atoms = ase.io.read(f"{CURRENT_DIR}/data/diamond.xyz")
    atoms = atoms.repeat((2, 2, 2))
    rng = np.random.default_rng(1234)
    positions = np.asarray(atoms.positions, dtype=np.float64).copy()
    positions += rng.normal(scale=1.0, size=positions.shape)
    return SystemForTests(
        name="diamond_crystal_3d",
        points=positions,
        box=np.asarray(atoms.cell[:], dtype=np.float64),
        periodic=tuple(bool(value) for value in atoms.pbc),
    )


def naphthalene_cluster() -> SystemForTests:
    atoms = ase.io.read(f"{CURRENT_DIR}/data/naphthalene.xyz")
    return SystemForTests(
        name="naphthalene_cluster_0d",
        points=np.asarray(atoms.positions, dtype=np.float64),
        box=np.diag([30.0, 30.0, 30.0]).astype(np.float64),
        periodic=(False, False, False),
    )


SYSTEMS_FOR_TESTS = [
    naphthalene_cluster(),
    polymer_chain(),
    graphene_sheet(),
    diamond_crystal(),
]

CUTOFFS = [3.0, 5.0, 7.0, 10.0]


def min_periodic_box_dimension(case: SystemForTests) -> float | None:
    periodic_lengths = [
        float(np.linalg.norm(case.box[index]))
        for index, is_periodic in enumerate(case.periodic)
        if is_periodic
    ]
    if len(periodic_lengths) == 0:
        return None
    return min(periodic_lengths)


def gpu_algorithm_is_applicable(
    case: SystemForTests,
    cutoff: float,
    gpu_algorithm: str,
) -> bool:
    if gpu_algorithm == "cell_list":
        return True

    min_periodic_dim = min_periodic_box_dimension(case)
    if min_periodic_dim is None:
        return True

    return cutoff <= min_periodic_dim / 2.0


@pytest.mark.parametrize("full_list", [False, True])
@pytest.mark.parametrize(
    "system", SYSTEMS_FOR_TESTS, ids=[case.name for case in SYSTEMS_FOR_TESTS]
)
@pytest.mark.parametrize("cutoff", CUTOFFS)
@pytest.mark.parametrize("gpu_algorithm", ["cell_list", "brute_force"])
def test_gpu_matches_cpu_for_fixed_systems(
    system, cutoff, full_list, gpu_algorithm, monkeypatch
):
    if not gpu_algorithm_is_applicable(system, cutoff, gpu_algorithm):
        return

    monkeypatch.setenv("VESIN_CUDA_MAX_PAIRS_PER_POINT", "4096")
    compare_cpu_gpu(system, cutoff, full_list, gpu_algorithm)
