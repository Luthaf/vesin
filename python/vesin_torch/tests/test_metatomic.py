import torch
from metatensor.torch import TensorBlock
from metatomic.torch import NeighborListOptions, System

from vesin.metatomic import NeighborList


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
    assert actual.samples.names == expected.samples.names
    assert torch.equal(actual.samples.values, expected.samples.values)
    assert len(actual.components) == len(expected.components)
    for actual_component, expected_component in zip(
        actual.components, expected.components, strict=True
    ):
        assert actual_component.names == expected_component.names
        assert torch.equal(actual_component.values, expected_component.values)
    assert actual.properties.names == expected.properties.names
    assert torch.equal(actual.properties.values, expected.properties.values)
