import numpy as np

from ._base import NeighborList


class PymatgenNeighborList(NeighborList):
    name = "pymatgen"
    devices = ("cpu",)
    full_list = (True,)

    @classmethod
    def version(cls):
        import pymatgen.core

        return pymatgen.core.__version__

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device)
        if not full_list:
            raise NotImplementedError("pymatgen only supports full_list=True")
        if not np.all(atoms.pbc):
            raise NotImplementedError("pymatgen only supports periodic structures")

        import pymatgen.core

        self.structure = pymatgen.core.Structure(
            atoms.cell[:],
            atoms.numbers,
            atoms.positions,
            coords_are_cartesian=True,
        )

    def run(self):
        i, j, S, d = self.structure.get_neighbor_list(self.cutoff)
        return len(i)
