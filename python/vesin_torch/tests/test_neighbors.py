from itertools import product
from typing import List, Union

import numpy as np
import pytest
import torch

from vesin.torch import NeighborList


def run_check_neighbors(
    points,
    box,
    periodic,
    cutoff,
    full_list,
    quantities,
    expected_outputs,
    device="cpu",
    rtol=1e-6,
    atol=1e-6,
):
    points = torch.tensor(points, dtype=torch.float64, device=device)
    box = torch.tensor(box, dtype=torch.float64, device=device)

    calculator = NeighborList(cutoff=cutoff, full_list=full_list)
    outputs = calculator.compute(points, box, periodic, quantities)

    index = {q: i for i, q in enumerate(quantities)}
    for q, expected in expected_outputs.items():
        actual = outputs[index[q]].cpu()
        expected = torch.tensor(expected, dtype=torch.float64)
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, msg=f"Mismatch in {q}"
        )


@pytest.mark.parametrize(
    "device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("full_list", [False, True])
def test_large_box_small_cutoff(device, full_list):
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [-6.0, 0.0, 0.0],
            [-6.0, -2.0, 0.0],
            [-6.0, 0.0, -2.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    box = torch.tensor(
        [
            [54.0, 0.0, 0.0],
            [0.0, 54.0, 0.0],
            [0.0, 0.0, 54.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    calculator = NeighborList(cutoff=2.1, full_list=full_list)

    i, j, d = calculator.compute(points, box, periodic=True, quantities="ijd")

    pairs = torch.stack((i, j), dim=1)
    sort_idx = torch.argsort(pairs[:, 0] * (i.max() + 1) + pairs[:, 1])

    # Apply sort
    i = i[sort_idx]
    j = j[sort_idx]
    d = d[sort_idx]

    # Convert to plain Python lists for easy matching
    actual_pairs = sorted(zip(i.tolist(), j.tolist(), strict=True))
    actual_dists = [d.item() for d in d]

    if full_list:
        expected_pairs = sorted(
            [(0, 1), (0, 2), (1, 0), (2, 0), (3, 4), (3, 5), (4, 3), (5, 3)]
        )
        expected_dists = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    else:
        expected_pairs = sorted([(0, 1), (0, 2), (3, 4), (3, 5)])
        expected_dists = [2.0, 2.0, 2.0, 2.0]
    # Check pairs
    assert actual_pairs == expected_pairs, (
        f"Expected pairs {expected_pairs}, got {actual_pairs}"
    )

    # Check distances approximately
    for actual, expected in zip(actual_dists, expected_dists, strict=True):
        assert abs(actual - expected) < 1e-8, (
            f"Expected distance {expected}, got {actual}"
        )


def test_errors():
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    box = torch.zeros((3, 3), dtype=torch.float64)

    calculator = NeighborList(cutoff=2.8, full_list=True)

    message = "only float64 dtype is supported in vesin"
    with pytest.raises(ValueError, match=message):
        calculator.compute(
            points.to(torch.float32),
            box.to(torch.float32),
            periodic=False,
            quantities="ij",
        )

    message = "expected `points` and `box` to have the same dtype, got Double and Float"
    with pytest.raises(ValueError, match=message):
        calculator.compute(
            points,
            box.to(torch.float32),
            periodic=False,
            quantities="ij",
        )

    message = "expected `points` and `box` to have the same device, got cpu and meta"
    with pytest.raises(ValueError, match=message):
        calculator.compute(
            points,
            box.to(device="meta"),
            periodic=False,
            quantities="ij",
        )

    message = "unexpected character in `quantities`: Q"
    with pytest.raises(ValueError, match=message):
        calculator.compute(
            points,
            box,
            periodic=False,
            quantities="ijQ",
        )

    message = "device meta is not supported in vesin"
    with pytest.raises(RuntimeError, match=message):
        calculator.compute(
            points.to(device="meta"),
            box.to(device="meta"),
            periodic=False,
            quantities="ij",
        )


@pytest.mark.parametrize("quantities", ["ijS", "D", "d", "ijSDd"])
def test_all_alone_no_neighbors(quantities):
    points = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64)
    box = torch.eye(3, dtype=torch.float64)

    calculator = NeighborList(cutoff=0.1, full_list=True)
    outputs = calculator.compute(points, box, True, quantities)

    if "ij" in quantities:
        assert list(outputs[quantities.index("i")].shape) == [0]
        assert list(outputs[quantities.index("j")].shape) == [0]

    if "S" in quantities:
        assert list(outputs[quantities.index("S")].shape) == [0, 3]

    if "D" in quantities:
        assert list(outputs[quantities.index("D")].shape) == [0, 3]
        assert not outputs[quantities.index("D")].requires_grad

    if "d" in quantities:
        assert list(outputs[quantities.index("d")].shape) == [0]
        assert not outputs[quantities.index("d")].requires_grad

    points.requires_grad_(True)
    box.requires_grad_(True)
    outputs = calculator.compute(points, box, True, quantities)

    if "ij" in quantities:
        assert list(outputs[quantities.index("i")].shape) == [0]
        assert list(outputs[quantities.index("j")].shape) == [0]

    if "S" in quantities:
        assert list(outputs[quantities.index("S")].shape) == [0, 3]

    if "D" in quantities:
        assert list(outputs[quantities.index("D")].shape) == [0, 3]
        assert outputs[quantities.index("D")].requires_grad

    if "d" in quantities:
        assert list(outputs[quantities.index("d")].shape) == [0]
        assert outputs[quantities.index("d")].requires_grad


@pytest.mark.parametrize(
    "device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
)
def test_mixed_periodic_inputs(device):
    periodic = torch.tensor([True, False, False], dtype=torch.bool)
    cutoff = 0.5
    # To understand this test, only focus on the first and second axis (x and y coords)
    # 1) Since only the first periodic boundary is enabled,
    #    the first and second point are only 0.2 away.
    # 2) Notice that the first and third point are 0.8 away.
    #    However, if periodicity is enabled on the second axis, then
    #    they would only be 0.1 away and would be considered neighbors.
    points = torch.tensor(
        [
            [0.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=torch.float64,
    )
    box = torch.eye(3, dtype=torch.float64)

    calculator = NeighborList(cutoff=cutoff, full_list=False)
    i, j, shifts, distances, _vectors = calculator.compute(
        points.to(device), box.to(device), periodic.to(device), "ijSdD"
    )

    torch.testing.assert_close(
        i.to(device), torch.tensor([0], dtype=torch.int64, device=device)
    )
    torch.testing.assert_close(
        j.to(device), torch.tensor([1], dtype=torch.int64, device=device)
    )
    torch.testing.assert_close(
        shifts.to(device), torch.tensor([[-1, 0, 0]], dtype=torch.int32, device=device)
    )
    torch.testing.assert_close(
        distances.to(device), torch.tensor([0.2], dtype=torch.float64, device=device)
    )


@pytest.mark.parametrize(
    "periodic",
    [torch.tensor(p, dtype=torch.bool) for p in product([False, True], repeat=3)],
)
@pytest.mark.parametrize(
    "device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize(
    "positions",
    [
        np.array(
            [
                [0.05, 0.05, 0.05],
                [0.95, 0.05, 0.05],
                [0.05, 0.95, 0.05],
                [0.05, 0.05, 0.95],
                [0.95, 0.95, 0.05],
                [0.95, 0.05, 0.95],
                [0.05, 0.95, 0.95],
                [0.95, 0.95, 0.95],
                [0.45, 0.45, 0.45],
                [0.55, 0.55, 0.55],
            ],
            dtype=np.float64,
        ),
        # A more comprehensive test with many more atoms
        np.random.default_rng(0).random((100, 3)),
    ],
)
def test_mixed_periodic_inputs_many_atoms(periodic, device, positions):
    """Larger test with many atoms to ensure that neighbors are computed correctly.
    Uses ASE as a reference."""
    import ase
    from ase.neighborlist import neighbor_list

    cutoff = 0.35
    cell = np.eye(3, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)

    def _sorted_neighbor_results(
        i, j, shifts, vectors, distances
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return deterministic neighbor outputs sorted by (i, j, shift)."""
        i = np.asarray(i, dtype=np.int64).reshape(-1)
        if i.size == 0:
            return (
                np.empty((0, 5), dtype=np.int64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        j = np.asarray(j, dtype=np.int64).reshape(-1)
        shifts = np.asarray(shifts, dtype=np.int64).reshape(-1, 3)
        vectors = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)

        pairs = np.column_stack((i, j, shifts))
        order = np.lexsort(pairs.T)
        return pairs[order], vectors[order], distances[order]

    atoms = ase.Atoms(
        symbols=["H"] * len(positions),
        positions=positions,
        cell=cell,
        pbc=np.asarray(periodic, dtype=bool),
    )

    expected_pairs, expected_vectors, expected_distances = _sorted_neighbor_results(
        *neighbor_list("ijSDd", atoms, cutoff)
    )

    calculator = NeighborList(cutoff=cutoff, full_list=True)
    points = torch.tensor(positions, dtype=torch.float64, device=device)
    box = torch.tensor(cell, dtype=torch.float64, device=device)
    outputs = calculator.compute(points, box, periodic, "ijSDd")

    actual_pairs, actual_vectors, actual_distances = _sorted_neighbor_results(
        *[out.detach().cpu().numpy() for out in outputs]
    )

    assert np.array_equal(actual_pairs, expected_pairs)
    # Disabled since the vectors are different from ASE
    # assert np.allclose(actual_vectors, expected_vectors)
    assert np.allclose(actual_distances, expected_distances)


class NeighborListWrap:
    def __init__(self, cutoff: float, full_list: bool):
        self._c = NeighborList(cutoff=cutoff, full_list=full_list)

    def compute(
        self,
        points: torch.Tensor,
        box: torch.Tensor,
        periodic: Union[bool, torch.Tensor],
        quantities: str,
        copy: bool,
    ) -> List[torch.Tensor]:
        return self._c.compute(
            points=points,
            box=box,
            periodic=periodic,
            quantities=quantities,
            copy=copy,
        )


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: NeighborListWrap) -> NeighborListWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)
