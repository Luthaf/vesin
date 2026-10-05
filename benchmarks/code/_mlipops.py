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
        import mlipops

        super().__init__(atoms, cutoff, full_list, device)

        self.calculator = mlipops.NeighborList(
            cutoff=cutoff,
            include_symmetric=full_list,
            padding=False,
            device=device,
        )

        self.positions = torch.tensor(atoms.positions, device=device)

        if np.any(atoms.pbc) and not np.all(atoms.pbc):
            raise NotImplementedError(
                "mlipops only supports fully periodic or fully non-periodic structures"
            )

        if np.all(atoms.pbc):
            face_distances = 1.0 / np.linalg.norm(atoms.cell.reciprocal(), axis=1)
            if np.any(face_distances < 2 * cutoff):
                raise ValueError("mlipops can not run for this super cell")

            self.box_vectors = torch.tensor(atoms.cell[:], device=device)
        else:
            self.box_vectors = None

    def run(self):
        # mlipops caches the result and skips the computation if it is called
        # again with the exact same positions tensor; clone the positions here so
        # each call triggers an actual neighbor search
        pairs = self.calculator(self.positions.clone(), self.box_vectors)
        return int(pairs.shape[0])
