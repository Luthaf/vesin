import torch

from ._base import NeighborList


class VesinTorchNeighborList(NeighborList):
    name = "vesin-torch"
    devices = ("cpu", "cuda")
    run_by_default = False

    @classmethod
    def version(cls):
        import vesin_torch

        return vesin_torch.__version__

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        import vesin_torch

        super().__init__(atoms, cutoff, full_list, device)

        self.calculator = vesin_torch.NeighborList(cutoff=cutoff, full_list=full_list)
        self.positions = torch.tensor(atoms.positions, device=device)
        self.cell = torch.tensor(atoms.cell[:], device=device)
        self.pbc = torch.tensor(atoms.pbc, device=device)

    def run(self):
        i, j, S, d = self.calculator.compute(
            self.positions, self.cell, self.pbc, quantities="ijSd", copy=False
        )
        return len(i)
