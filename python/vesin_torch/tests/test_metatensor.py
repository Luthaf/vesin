from typing import Dict, List, Optional

import pytest
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from vesin.torch.metatensor import NeighborList, compute_requested_neighbors


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


class InnerModule(torch.nn.Module):
    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [NeighborListOptions(cutoff=3.4, full_list=False, strict=True)]


class OuterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = InnerModule()

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [NeighborListOptions(cutoff=5.2, full_list=True, strict=False)]

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        return {}


def test_model():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.float64, requires_grad=True
    )
    cell = (4 * torch.eye(3, dtype=torch.float64)).clone().requires_grad_(True)
    pbc = torch.ones(3, dtype=bool)
    types = torch.tensor([6, 8])
    systems = [
        System(positions=positions, cell=cell, pbc=pbc, types=types),
        System(positions=positions, cell=cell, pbc=pbc, types=types),
    ]

    # Using a "raw" model
    model = OuterModule()
    compute_requested_neighbors(
        systems=systems, system_length_unit="A", model=model, model_length_unit="A"
    )

    for system in systems:
        all_options = system.known_neighbor_lists()
        assert len(all_options) == 2
        assert all_options[0].requestors() == ["OuterModule"]
        assert all_options[0].cutoff == 5.2
        assert all_options[1].requestors() == ["OuterModule.inner"]
        assert all_options[1].cutoff == 3.4

    # Using a MetatensorAtomisticModel model
    capabilities = ModelCapabilities(
        length_unit="A",
        interaction_range=6.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    model = MetatensorAtomisticModel(model.eval(), ModelMetadata(), capabilities)
    compute_requested_neighbors(
        systems=System(positions=positions, cell=cell, pbc=pbc, types=types),
        system_length_unit="A",
        model=model,
    )

    for system in systems:
        all_options = system.known_neighbor_lists()
        assert len(all_options) == 2
        assert all_options[0].requestors() == ["OuterModule"]
        assert all_options[0].cutoff == 5.2
        assert all_options[1].requestors() == ["OuterModule.inner"]
        assert all_options[1].cutoff == 3.4

    message = (
        "the given `model_length_unit` \\(nm\\) does not match the model "
        "capabilities \\(A\\)"
    )
    with pytest.raises(ValueError, match=message):
        compute_requested_neighbors(
            systems=System(positions=positions, cell=cell, pbc=pbc, types=types),
            system_length_unit="A",
            model=model,
            model_length_unit="nm",
        )
