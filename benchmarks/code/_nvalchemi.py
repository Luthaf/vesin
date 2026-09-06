import torch

from ._base import NeighborList


class NValchemiNeighborList(NeighborList):
    name = "nvalchemi"
    devices = ("cpu", "cuda")

    @classmethod
    def version(cls):
        import nvalchemiops

        return nvalchemiops.__version__

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device)

        self.positions = torch.tensor(atoms.positions, device=device)
        self.cell = torch.tensor(atoms.cell[:], device=device).unsqueeze(0)
        self.pbc = torch.tensor(atoms.pbc, device=device)

    def run(self):
        from nvalchemiops.torch.neighbors import (
            neighbor_list as nvalchemi_neighbor_list,
        )

        indices, counts, deltas = nvalchemi_neighbor_list(
            self.positions,
            self.cutoff,
            cell=self.cell,
            pbc=self.pbc,
            half_fill=not self.full_list,
        )
        return int(counts.sum())
