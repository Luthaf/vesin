from typing import Dict, List, Optional

import pytest


torch = pytest.importorskip("torch")
metatomic = pytest.importorskip("metatomic")

from metatensor.torch import Labels, TensorMap  # noqa: E402
from metatomic.torch import (  # noqa: E402
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from vesin.metatomic import (  # noqa: E402
    NeighborList,
    compute_requested_neighbors,
    compute_requested_neighbors_from_options,
    neighbor_lists_for_model,
)


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


class NLModule(torch.nn.Module):
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
    ) -> None:
        compute_requested_neighbors_from_options(
            systems=systems,
            options=self.requested_neighbor_lists(),
            system_length_unit="A",
            check_consistency=True,
        )


def test_model():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.4]], dtype=torch.float64, requires_grad=True
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
    message = (
        "`compute_requested_neighbors` is deprecated and will be removed in a future "
        "version. Please use `neighbor_lists_for_model` to get the calculators and "
        "call them directly."
    )
    with pytest.warns(UserWarning, match=message):
        compute_requested_neighbors(
            systems=systems,
            system_length_unit="A",
            model=model,
            model_length_unit="A",
            check_consistency=True,
        )

    for system in systems:
        all_options = system.known_neighbor_lists()
        assert len(all_options) == 2
        assert all_options[0].requestors() == ["OuterModule"]
        assert all_options[0].cutoff == 5.2
        assert all_options[1].requestors() == ["OuterModule.inner"]
        assert all_options[1].cutoff == 3.4

    # Using a AtomisticModel
    capabilities = ModelCapabilities(
        length_unit="A",
        interaction_range=6.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    model = AtomisticModel(model.eval(), ModelMetadata(), capabilities)
    with pytest.warns(UserWarning, match=message):
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


def test_get_requested_neighbor_calculators():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.4]], dtype=torch.float64, requires_grad=True
    )
    cell = (4 * torch.eye(3, dtype=torch.float64)).clone().requires_grad_(True)
    pbc = torch.ones(3, dtype=bool)
    types = torch.tensor([6, 8])

    # Using a raw model
    model = OuterModule()
    calculators = neighbor_lists_for_model(
        model=model,
        system_length_unit="A",
        model_length_unit="A",
    )

    assert len(calculators) == 2
    assert calculators[0].options.cutoff == 5.2
    assert calculators[1].options.cutoff == 3.4

    # Using an AtomisticModel
    capabilities = ModelCapabilities(
        length_unit="A",
        interaction_range=6.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic_model = AtomisticModel(model.eval(), ModelMetadata(), capabilities)
    calculators = neighbor_lists_for_model(
        model=atomistic_model,
        system_length_unit="A",
    )

    assert len(calculators) == 2

    # Reuse calculators across multiple calls, without passing model
    for _ in range(3):
        system = System(positions=positions, cell=cell, pbc=pbc, types=types)
        for calculator in calculators:
            calculator.add_neighbor_list(system)

        all_options = system.known_neighbor_lists()
        assert len(all_options) == 2
        assert all_options[0].cutoff == 5.2
        assert all_options[1].cutoff == 3.4


def test_torchscriptability():
    torch.jit.script(NLModule())
