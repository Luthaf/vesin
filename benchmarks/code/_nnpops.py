import numpy as np
import torch

from ._base import NeighborList


class NNPOpsNeighborList(NeighborList):
    name = "nnpops"
    devices = ("cpu", "cuda")
    full_list = (False,)

    @classmethod
    def version(cls):
        import NNPOps

        return NNPOps.__version__

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device)
        if full_list:
            raise NotImplementedError(
                "NNPOps only supports half lists (full_list=False)"
            )
        if np.any(atoms.cell.lengths() < 2 * cutoff):
            raise ValueError(f"{self.name} can not run for this super cell")

        self.positions = torch.tensor(atoms.positions, device=device)
        if np.all(atoms.pbc):
            self.box_vectors = torch.tensor(atoms.cell[:], device=device)
        else:
            self.box_vectors = None

    def run(self):
        import NNPOps.neighbors

        neighbors, deltas, distances, count = NNPOps.neighbors.getNeighborPairs(
            self.positions, cutoff=self.cutoff, box_vectors=self.box_vectors
        )
        # NNPOps returns an unordered half list where pairs beyond the cutoff are
        # marked with NaN distances; only count the pairs within the cutoff
        return int(torch.isfinite(distances).sum())
