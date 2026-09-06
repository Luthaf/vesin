import numpy as np

from ._base import NeighborList


class SislNeighborList(NeighborList):
    name = "sisl"
    devices = ("cpu",)
    full_list = (True,)

    @classmethod
    def version(cls):
        import sisl

        return sisl.__version__

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device)
        if not full_list:
            raise NotImplementedError("sisl only supports full_list=True")

        import sisl

        system = sisl.Geometry.new.ase(atoms)
        periodic = np.nonzero(atoms.pbc)[0]
        if len(periodic) > 0:
            system = system.translate2uc(axes=periodic)

        self.system = system

    def run(self):
        import sisl

        finder = sisl.geom.NeighborFinder(self.system, R=self.cutoff)
        neighbors = finder.find_neighbors()
        return len(neighbors.i)
