import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vesin_torch import NeighborList


TORCH_VERSION = tuple(int(v) for v in torch.__version__.split(".")[:2])

pytestmark = pytest.mark.skipif(
    TORCH_VERSION < (2, 7),
    reason="AOTInductor supports TorchBind objects since torch 2.7",
)

if sys.platform == "win32":
    # Never emit OpenMP-parallel kernels in the AOTI models, see
    # https://github.com/pytorch/pytorch/issues/198205
    torch._inductor.config.cpp.min_chunk_size = 2**31 - 1


class Energy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.neighbors = NeighborList(cutoff=3.0, full_list=False, skin=0.5)

    def forward(
        self, points: torch.Tensor, box: torch.Tensor, return_forces: bool
    ) -> (torch.Tensor, torch.Tensor):
        if return_forces:
            points = points.detach().requires_grad_(True)

        (distances,) = self.neighbors.compute(points, box, True, "d")
        energy = (1.0 / distances).sum()

        if return_forces:
            (gradient,) = torch.autograd.grad(energy, points)
            return energy, -gradient
        else:
            return energy, torch.empty((0, 3), dtype=points.dtype, device=points.device)


def _system(n_points):
    generator = torch.Generator().manual_seed(0xDEADBEEF + n_points)
    length = (n_points / 0.05) ** (1 / 3)
    points = torch.rand(n_points, 3, dtype=torch.float64, generator=generator)
    return points * length, length * torch.eye(3, dtype=torch.float64)


def test_energy(tmp_path):
    """Compute the energy, handling dynamic shapes"""
    model = Energy()
    points, box = _system(n_points=40)
    exported = torch.export.export(
        model,
        (points, box, False),
        dynamic_shapes=({0: torch.export.Dim("n_points", min=0)}, None, None),
    )
    path = torch._inductor.aoti_compile_and_package(
        exported, package_path=str(tmp_path / "energy.pt2")
    )
    loaded = torch._inductor.aoti_load_package(path)
    exported = exported.module()

    for n_points in [40, 200]:
        points, box = _system(n_points)
        expected_energy, _ = model(points, box, return_forces=False)
        # run multiple times to check that state re-use does not crash
        for _ in range(2):
            energy, forces = loaded(points, box, False)
            assert torch.allclose(energy, expected_energy)
            assert forces.shape == (0, 3)

            energy, forces = exported(points, box, False)
            assert torch.allclose(energy, expected_energy)
            assert forces.shape == (0, 3)


def test_energy_and_forces_inside(tmp_path):
    """Compute the forces inside of the exported graph, using make_fx."""
    model = Energy()
    points, box = _system(n_points=40)

    # make_fx includes the backward pass in the graph, which allows to export the
    # calculation of the forces.
    traced = make_fx(model, tracing_mode="symbolic")(points, box, True)
    exported = torch.export.export(
        traced,
        (points, box, True),
        dynamic_shapes=({0: torch.export.Dim("n_points", min=0)}, None, None),
    )
    path = torch._inductor.aoti_compile_and_package(
        exported, package_path=str(tmp_path / "energy-forces.pt2")
    )
    loaded = torch._inductor.aoti_load_package(path)
    exported = exported.module()

    for n_points in [40, 200]:
        points, box = _system(n_points)
        expected_energy, expected_forces = model(points, box, return_forces=True)
        for _ in range(2):
            energy, forces = loaded(points, box, True)
            assert torch.allclose(energy, expected_energy)
            assert torch.allclose(forces, expected_forces)

            # return_forces=False does not work, the control flow is baked inside the
            # graph with tracing_mode="symbolic", but tracing_mode="symbolic" is
            # required to make the backward pass differentiable.
            energy, forces = loaded(points, box, False)
            assert torch.allclose(energy, expected_energy)
            assert forces.shape == (n_points, 3)  # FIXME, this should be (0, 3)

            energy, forces = exported(points, box, True)
            assert torch.allclose(energy, expected_energy)
            assert torch.allclose(forces, expected_forces)

            with pytest.raises(AssertionError, match="Guard failed: arg2_1 == True"):
                exported(points, box, False)


def test_energy_and_forces_outside(tmp_path):
    """
    Compute the forces outside of the exported graph, using autograd.

    This does not work with AOTI, only torch.export in Python
    """
    model = Energy()
    points, box = _system(n_points=40)
    points.requires_grad_(True)
    exported = torch.export.export(
        model,
        (points, box, False),
        dynamic_shapes=({0: torch.export.Dim("n_points", min=0)}, None, None),
    )

    exported = exported.module()

    for n_points in [40, 200]:
        points, box = _system(n_points)
        points.requires_grad_(True)
        expected_energy, _ = model(points, box, return_forces=False)
        # run multiple times to check that state re-use does not crash
        for _ in range(2):
            energy, forces = exported(points, box, False)
            assert torch.allclose(energy, expected_energy)
            assert forces.shape == (0, 3)

            energy.backward()
            assert points.grad is not None


def test_export_save_load(tmp_path):
    points, box = _system(40)
    model = Energy()
    expected, _ = model(points, box, False)

    exported = torch.export.export(model, (points, box, False))
    states = [
        spec.target
        for spec in exported.graph_signature.input_specs
        if spec.kind == torch.export.graph_signature.InputKind.CUSTOM_OBJ
    ]
    assert len(states) == 1

    path = tmp_path / "model.pt2"
    torch.export.save(exported, str(path))
    loaded = torch.export.load(str(path))

    # loading the model creates a new state
    state = loaded.constants[states[0]]
    assert isinstance(state, torch.ScriptObject)
    assert state is not model.neighbors._state

    energy, _ = loaded.module()(points, box, False)
    assert torch.allclose(energy, expected)
