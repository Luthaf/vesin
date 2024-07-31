from typing import List

import pytest
import torch

from vesin.torch import NeighborList


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
