import torch

import vesin

from ._base import NeighborList


class VesinNeighborList(NeighborList):
    name = "vesin"
    devices = ("cpu", "cuda")

    @classmethod
    def version(cls):
        return vesin.__version__

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device)

        self.calculator = vesin.NeighborList(
            cutoff=cutoff,
            full_list=full_list,
            sorted=False,
        )
        if device == "cpu":
            self.positions = atoms.positions
            self.cell = atoms.cell[:]
            self.pbc = atoms.pbc
        else:
            self.positions = torch.tensor(atoms.positions, device=device)
            self.cell = torch.tensor(atoms.cell[:], device=device)
            self.pbc = torch.tensor(atoms.pbc, device=device)

    def run(self):
        i, j, S, d = self.calculator.compute(
            self.positions, self.cell, self.pbc, quantities="ijSd", copy=False
        )
        return len(i)
