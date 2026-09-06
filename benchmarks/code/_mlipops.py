import importlib.metadata

import numpy as np
import torch

from ._base import NeighborList


class MLipopsNeighborList(NeighborList):
    name = "mlipops"
    devices = ("cpu", "cuda")

    @classmethod
    def version(cls):
        return importlib.metadata.version("mlipops")

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device)
        if np.any(atoms.cell.lengths() < 2 * cutoff):
            raise ValueError(f"{self.name} can not run for this super cell")

        import mlipops

        self.calculator = mlipops.NeighborList(
            cutoff=cutoff,
            include_symmetric=full_list,
            padding=False,
            device=device,
        )
        self.positions = torch.tensor(atoms.positions, device=device)
        if np.all(atoms.pbc):
            self.box_vectors = torch.tensor(atoms.cell[:], device=device)
        else:
            self.box_vectors = None

    def run(self):
        # mlipops caches the result and skips the computation if it is called
        # again with the exact same positions tensor; clone the positions here so
        # each call triggers an actual neighbor search
        pairs = self.calculator(self.positions.clone(), self.box_vectors)
        return int(pairs.shape[0])
