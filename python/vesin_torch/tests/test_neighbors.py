from typing import List

import pytest
import torch

from vesin.torch import NeighborList


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def test_errors():
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    box = torch.zeros((3, 3), dtype=torch.float64)

    calculator = NeighborList(cutoff=2.8, full_list=True)

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
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_alone_no_neighbors(quantities, dtype):
    points = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=dtype)
    box = torch.eye(3, dtype=dtype)

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
    def __init__(self, cutoff: float, full_list: bool, sorted: bool):
        self._c = NeighborList(cutoff=cutoff, full_list=full_list, sorted=sorted)

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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype(dtype, device):
    box = torch.eye(3, dtype=dtype, device=device) * 3.0
    points = torch.rand((100, 3), dtype=dtype, device=device) * 3.0

    calculator = NeighborList(cutoff=1, full_list=True)
    i, j, s, D, d = calculator.compute(points, box, True, "ijSDd")

    assert i.dtype == torch.int64
    assert j.dtype == torch.int64
    assert s.dtype == torch.int32
    assert D.dtype == dtype
    assert d.dtype == dtype
