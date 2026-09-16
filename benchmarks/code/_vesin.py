import torch

import vesin

from ._base import NeighborList


class VesinNeighborListBase(NeighborList):
    devices = ("cpu", "cuda")

    @classmethod
    def version(cls):
        return vesin.__version__

    def __init__(self, atoms, cutoff, full_list, device, sorted, skin):
        super().__init__(atoms, cutoff, full_list, device)

        self.calculator = vesin.NeighborList(
            cutoff=cutoff,
            full_list=full_list,
            sorted=sorted,
            skin=skin,
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


class VesinNeighborList(VesinNeighborListBase):
    name = "vesin"

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device, sorted=False, skin=0.0)


class VesinVerletNeighborList(VesinNeighborListBase):
    name = "vesin/verlet"

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(
            atoms, cutoff, full_list, device, sorted=False, skin=0.2 * cutoff
        )


class VesinSortedNeighborList(VesinNeighborListBase):
    name = "vesin/sorted"

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device, sorted=True, skin=0.0)
