import pytest
import torch
from metatensor.torch.atomistic import NeighborListOptions, System

from vesin.torch.metatensor import NeighborList


def test_errors():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.float64)
    cell = 4 * torch.eye(3, dtype=torch.float64)
    system = System(
        positions=positions,
        cell=cell,
        pbc=torch.ones(3, dtype=bool),
        types=torch.tensor([6, 8]),
    )

    options = NeighborListOptions(cutoff=3.5, full_list=True, strict=True)
    calculator = NeighborList(options, length_unit="A")

    system.pbc[0] = False
    message = (
        "vesin currently does not support mixed periodic and non-periodic "
        "boundary conditions"
    )
    with pytest.raises(NotImplementedError, match=message):
        calculator.compute(system)


def test_script():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.float64)
    cell = 4 * torch.eye(3, dtype=torch.float64)
    system = System(
        positions=positions,
        cell=cell,
        pbc=torch.ones(3, dtype=bool),
        types=torch.tensor([6, 8]),
    )

    options = NeighborListOptions(cutoff=3.5, full_list=True, strict=True)
    calculator = torch.jit.script(NeighborList(options, length_unit="A"))
    calculator.compute(system)


def test_backward():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.float64, requires_grad=True
    )
    cell = (4 * torch.eye(3, dtype=torch.float64)).clone().requires_grad_(True)
    system = System(
        positions=positions,
        cell=cell,
        pbc=torch.ones(3, dtype=bool),
        types=torch.tensor([6, 8]),
    )

    options = NeighborListOptions(cutoff=3.5, full_list=True, strict=True)
    calculator = NeighborList(options, length_unit="A")
    neighbors = calculator.compute(system)

    value = ((neighbors.values) ** 2).sum() * torch.linalg.det(cell)
    value.backward()

    # check there are gradients, and they are not zero
    assert positions.grad is not None
    assert cell.grad is not None
    assert torch.linalg.norm(positions.grad) > 0
    assert torch.linalg.norm(cell.grad) > 0
