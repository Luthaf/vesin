import torch
from metatomic.torch import NeighborListOptions, System

from vesin.metatomic import NeighborList


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
    calculator = torch.jit.script(
        NeighborList(options, length_unit="A", torchscript=True)
    )
    calculator.compute(system)
