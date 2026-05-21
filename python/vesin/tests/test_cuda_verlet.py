"""Tests for Verlet caching with CUDA arrays."""

import pytest

from vesin import NeighborList


cp = pytest.importorskip("cupy")

try:
    cp.cuda.Device(0).compute_capability
except cp.cuda.runtime.CUDARuntimeError:
    pytest.skip("CUDA is not available", allow_module_level=True)


def test_cuda_pair_crossing_cutoff_inside_skin_is_reported():
    nl = NeighborList(cutoff=1.0, full_list=False, skin=0.4, algorithm="cell_list")
    box = cp.asarray(10.0 * cp.eye(3), dtype=cp.float64)

    pos1 = cp.asarray([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]], dtype=cp.float64)
    i1, j1 = nl.compute(pos1, box, periodic=False, quantities="ij")
    assert (
        list(zip(cp.asnumpy(i1).tolist(), cp.asnumpy(j1).tolist(), strict=True)) == []
    )

    pos2 = cp.asarray([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]], dtype=cp.float64)
    i2, j2, d2 = nl.compute(pos2, box, periodic=False, quantities="ijd")

    assert list(zip(cp.asnumpy(i2).tolist(), cp.asnumpy(j2).tolist(), strict=True)) == [
        (0, 1)
    ]
    assert cp.asnumpy(d2).tolist() == pytest.approx([0.95])


def test_cuda_verlet_sorted_output_is_sorted_by_first_pair_index():
    nl = NeighborList(
        cutoff=2.0, full_list=True, sorted=True, skin=0.5, algorithm="cell_list"
    )
    box = cp.asarray(8.0 * cp.eye(3), dtype=cp.float64)
    positions = cp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 4.0, 4.0],
            [4.8, 4.0, 4.0],
        ],
        dtype=cp.float64,
    )

    nl.compute(positions, box, periodic=False, quantities="ij")
    positions = positions + cp.asarray(
        [
            [0.01, 0.0, 0.0],
            [0.00, 0.01, 0.0],
            [0.00, 0.0, 0.01],
            [0.01, 0.01, 0.0],
            [0.00, 0.01, 0.01],
        ],
        dtype=cp.float64,
    )

    i, _ = nl.compute(positions, box, periodic=False, quantities="ij")

    i = cp.asnumpy(i)
    assert (i == sorted(i.tolist())).all()


def _neighbors_as_set(i, j, shifts):
    shifts = cp.asnumpy(shifts) if isinstance(shifts, cp.ndarray) else shifts
    return set(
        zip(
            i.tolist(),
            j.tolist(),
            shifts[:, 0].tolist(),
            shifts[:, 1].tolist(),
            shifts[:, 2].tolist(),
            strict=True,
        )
    )


def test_cuda_verlet_matches_cpu_stateless_over_short_trajectory():
    box = 5.0 * cp.eye(3, dtype=cp.float64)
    cpu_box = cp.asnumpy(box)
    positions = cp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5],
        ],
        dtype=cp.float64,
    )

    gpu_nl = NeighborList(cutoff=3.0, full_list=True, skin=0.5, algorithm="cell_list")
    cpu_nl = NeighborList(cutoff=3.0, full_list=True, algorithm="cell_list")

    for step in range(6):
        cpu_positions = cp.asnumpy(positions)
        cpu_i, cpu_j, cpu_S = cpu_nl.compute(
            cpu_positions,
            cpu_box,
            periodic=True,
            quantities="ijS",
        )

        gpu_i, gpu_j, gpu_S = gpu_nl.compute(
            positions,
            box,
            periodic=True,
            quantities="ijS",
        )

        assert _neighbors_as_set(
            cp.asnumpy(gpu_i), cp.asnumpy(gpu_j), gpu_S
        ) == _neighbors_as_set(
            cpu_i,
            cpu_j,
            cpu_S,
        )

        positions = positions + cp.asarray(
            [
                [0.01 * (step + 1), 0.0, 0.0],
                [0.0, 0.01 * (step + 1), 0.0],
                [0.0, 0.0, 0.01 * (step + 1)],
                [0.005 * (step + 1), 0.005 * (step + 1), 0.0],
            ],
            dtype=cp.float64,
        )


def test_cuda_verlet_matches_cpu_with_triclinic_periodic_shifts():
    box = cp.asarray(
        [
            [4.0, 0.0, 0.0],
            [0.6, 4.2, 0.0],
            [0.3, 0.4, 4.5],
        ],
        dtype=cp.float64,
    )
    cpu_box = cp.asnumpy(box)
    positions = cp.asarray(
        [
            [0.10, 0.10, 0.10],
            [3.85, 0.25, 0.20],
            [0.25, 3.95, 0.35],
            [0.30, 0.45, 4.10],
            [2.10, 2.00, 2.20],
        ],
        dtype=cp.float64,
    )

    gpu_nl = NeighborList(cutoff=1.0, full_list=True, skin=0.35, algorithm="cell_list")
    cpu_nl = NeighborList(cutoff=1.0, full_list=True, algorithm="cell_list")

    for step in range(4):
        cpu_positions = cp.asnumpy(positions)
        cpu_i, cpu_j, cpu_S = cpu_nl.compute(
            cpu_positions,
            cpu_box,
            periodic=True,
            quantities="ijS",
        )

        gpu_i, gpu_j, gpu_S = gpu_nl.compute(
            positions,
            box,
            periodic=True,
            quantities="ijS",
        )

        assert _neighbors_as_set(
            cp.asnumpy(gpu_i), cp.asnumpy(gpu_j), gpu_S
        ) == _neighbors_as_set(
            cpu_i,
            cpu_j,
            cpu_S,
        )

        positions = positions + cp.asarray(
            [
                [0.010 * (step + 1), 0.000, 0.000],
                [0.000, 0.008 * (step + 1), 0.000],
                [0.000, 0.000, 0.006 * (step + 1)],
                [0.004 * (step + 1), 0.004 * (step + 1), 0.000],
                [0.000, 0.003 * (step + 1), 0.002 * (step + 1)],
            ],
            dtype=cp.float64,
        )
