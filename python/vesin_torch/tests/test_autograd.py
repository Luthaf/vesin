import pytest
import torch

from vesin.torch import NeighborList


@pytest.mark.parametrize("full_list", [True, False])
@pytest.mark.parametrize("requires_grad", [(True, True), (False, True), (True, False)])
@pytest.mark.parametrize("quantities", ["ijS", "D", "d", "ijSDd"])
def test_autograd(full_list, requires_grad, quantities):
    torch.manual_seed(0xDEADBEEF)

    points_fractional = torch.rand((34, 3), dtype=torch.float64)
    box = torch.diag(5 * torch.rand(3, dtype=torch.float64))
    box += torch.rand((3, 3), dtype=torch.float64)

    points = points_fractional @ box

    points.requires_grad_(requires_grad[0])
    box.requires_grad_(requires_grad[1])

    calculator = NeighborList(cutoff=7.8, full_list=full_list)

    def compute(points, box):
        results = calculator.compute(points, box, periodic=True, quantities=quantities)
        return results

    torch.autograd.gradcheck(compute, (points, box), fast_mode=True)
