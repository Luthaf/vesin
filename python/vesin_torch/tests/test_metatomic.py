from typing import List

import torch
from metatensor.torch import TensorBlock
from metatomic.torch import NeighborListOptions, System

from vesin.metatomic import NeighborList, neighbor_lists_for_model


class SerializedNeighborListModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        options = NeighborListOptions(cutoff=3.5, full_list=False, strict=True)
        self.calculator = NeighborList(options, length_unit="A", torchscript=True)

    def forward(self, system: System) -> TensorBlock:
        return self.calculator.compute(system)


def test_script(tmp_path):
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.float64)
    cell = 4 * torch.eye(3, dtype=torch.float64)
    system = System(
        positions=positions,
        cell=cell,
        pbc=torch.ones(3, dtype=bool),
        types=torch.tensor([6, 8]),
    )

    module = torch.jit.script(SerializedNeighborListModule())
    expected = module(system)

    path = tmp_path / "calculator.pt"
    module.save(str(path))
    loaded = torch.jit.load(str(path))
    actual = loaded(system)

    assert torch.equal(actual.values, expected.values)
    assert actual.samples == expected.samples
    assert actual.components == expected.components
    assert actual.properties == expected.properties


class CustomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.options = NeighborListOptions(cutoff=3.5, full_list=False, strict=True)

    def forward(self, system: System) -> TensorBlock:
        return system.get_neighbor_list(self.options)

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.options]


class ModelThatDoesItsOwnNeighborCalculation(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.calculators = neighbor_lists_for_model(
            model=self.model,
            system_length_unit="A",
            model_length_unit="A",
            torchscript=True,
        )

    def forward(self) -> TensorBlock:
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]], dtype=torch.float64
        )
        cell = 4 * torch.eye(3, dtype=torch.float64)
        system = System(
            positions=positions,
            cell=cell,
            pbc=torch.ones(3, dtype=torch.bool),
            types=torch.tensor([6, 8]),
        )

        for calculator in self.calculators:
            calculator.add_neighbor_list(system)

        return self.model.forward(system)


def test_script_for_model(tmp_path):
    model = CustomModel()
    wrapper = ModelThatDoesItsOwnNeighborCalculation(model)

    wrapper = torch.jit.script(wrapper)
    expected = wrapper()

    path = tmp_path / "calculator.pt"
    wrapper.save(str(path))
    loaded = torch.jit.load(str(path))
    actual = loaded()

    assert torch.equal(actual.values, expected.values)
    assert actual.samples == expected.samples
    assert actual.components == expected.components
    assert actual.properties == expected.properties
