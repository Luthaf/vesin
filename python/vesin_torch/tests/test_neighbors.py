from typing import List

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
    assert len(i) == len(j)
    assert len(i) == len(d)
    assert len(i) == (8 if full_list else 4)

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


class NeighborListWrap:
    def __init__(self, cutoff: float, full_list: bool):
        self._c = NeighborList(cutoff=cutoff, full_list=full_list)

    def compute(
        self,
        points: torch.Tensor,
        box: torch.Tensor,
        periodic: bool,
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
