import torch

import vesin

from ._base import NeighborList


class VesinNeighborListBase(NeighborList):
    devices = ("cpu", "cuda")

    @classmethod
    def version(cls):
        return vesin.__version__

    def __init__(
        self, atoms, cutoff, full_list, device, sorted, skin, algorithm="auto"
    ):
        super().__init__(atoms, cutoff, full_list, device)

        self.calculator = vesin.NeighborList(
            cutoff=cutoff,
            full_list=full_list,
            sorted=sorted,
            skin=skin,
            algorithm=algorithm,
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
    run_by_default = False

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(
            atoms, cutoff, full_list, device, sorted=False, skin=0.2 * cutoff
        )


class VesinSortedNeighborList(VesinNeighborListBase):
    name = "vesin/sorted"
    run_by_default = False

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device, sorted=True, skin=0.0)


class VesinVerletSortedNeighborList(VesinNeighborListBase):
    name = "vesin/verlet-sorted"
    run_by_default = False

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(
            atoms, cutoff, full_list, device, sorted=True, skin=0.2 * cutoff
        )


# Forcing one algorithm or the other measures what `algorithm="auto"` has to
# choose between. Neither is meant to win everywhere: the brute force kernels
# refuse to run past their point and periodic image limits, and those runs show
# up as missing points rather than as timings.
class VesinBruteForceNeighborList(VesinNeighborListBase):
    name = "vesin/brute-force"
    run_by_default = False

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(
            atoms,
            cutoff,
            full_list,
            device,
            sorted=False,
            skin=0.0,
            algorithm="brute_force",
        )


class VesinCellListNeighborList(VesinNeighborListBase):
    name = "vesin/cell-list"
    run_by_default = False

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(
            atoms,
            cutoff,
            full_list,
            device,
            sorted=False,
            skin=0.0,
            algorithm="cell_list",
        )
