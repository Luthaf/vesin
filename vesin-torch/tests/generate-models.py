"""
Generate the models used by the C++ tests of vesin-torch, together with reference
values computed in Python:

- ``energy.pt2``: AOTInductor package computing the energy of a pair potential
- ``energy-forces.pt2``: AOTInductor package computing the energy and the forces,
  with the backward pass included in the graph by ``make_fx``
- ``energy.pt``: TorchScript version of the energy model
- ``reference.pt``: TorchScript module storing the inputs, energies and forces
  for multiple systems as buffers.

Usage: python generate-models.py <output-dir>
"""

import os
import sys

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vesin_torch import NeighborList


TORCH_VERSION = tuple(int(v) for v in torch.__version__.split(".")[:2])

if sys.platform == "win32":
    # Never emit OpenMP-parallel kernels in the AOTI models, see
    # https://github.com/pytorch/pytorch/issues/198205
    torch._inductor.config.cpp.min_chunk_size = 2**31 - 1


class Energy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.neighbors = NeighborList(cutoff=3.0, full_list=False, skin=0.5)

    def forward(self, points: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        (distances,) = self.neighbors.compute(points, box, True, "d")
        return (1.0 / distances).sum()


def system(n_points):
    generator = torch.Generator().manual_seed(0xDEADBEEF + n_points)
    length = (n_points / 0.05) ** (1 / 3)
    points = torch.rand(n_points, 3, dtype=torch.float64, generator=generator)
    return points * length, length * torch.eye(3, dtype=torch.float64)


def energy_and_forces(model):
    def function(points, box):
        points = points.detach().requires_grad_(True)
        energy = model(points, box)
        (gradient,) = torch.autograd.grad(energy, points)
        return energy.detach(), -gradient

    return function


class Reference(torch.nn.Module):
    def __init__(self, references):
        super().__init__()
        self.n_references = len(references)
        for i, (points, box, energy, forces) in enumerate(references):
            self.register_buffer(f"points_{i}", points)
            self.register_buffer(f"box_{i}", box)
            self.register_buffer(f"energy_{i}", energy)
            self.register_buffer(f"forces_{i}", forces)

    def forward(self) -> int:
        return self.n_references


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = Energy()
    torch.jit.script(model).save(os.path.join(output_dir, "energy.pt"))

    # AOTInductor supports TorchBind objects since torch 2.7
    if TORCH_VERSION >= (2, 7):
        points, box = system(n_points=40)
        dynamic_shapes = ({0: torch.export.Dim("n_points", min=2)}, None)

        exported = torch.export.export(
            model, (points, box), dynamic_shapes=dynamic_shapes
        )
        torch._inductor.aoti_compile_and_package(
            exported, package_path=os.path.join(output_dir, "energy.pt2")
        )

        # make_fx includes the backward pass in the graph, which allows to export
        # the calculation of the forces
        function = energy_and_forces(model)
        traced = make_fx(function, tracing_mode="symbolic")(points, box)
        exported = torch.export.export(
            traced, (points, box), dynamic_shapes=dynamic_shapes
        )
        torch._inductor.aoti_compile_and_package(
            exported, package_path=os.path.join(output_dir, "energy-forces.pt2")
        )

    # use a new model to compute the reference values, so they do not depend on the
    # state of the model used above
    reference_model = Energy()
    references = []
    for n_points in [(40), (250), (1000)]:
        points, box = system(n_points)
        energy, forces = energy_and_forces(reference_model)(points, box)
        references.append((points, box, energy, forces))

    reference = torch.jit.script(Reference(references))
    reference.save(os.path.join(output_dir, "reference.pt"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    main(sys.argv[1])
