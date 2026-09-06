from ._base import NeighborList


class MatscipyNeighborList(NeighborList):
    name = "matscipy"
    devices = ("cpu",)
    full_list = (True,)

    @classmethod
    def version(cls):
        import matscipy

        return matscipy.__version__

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device)
        if not full_list:
            raise NotImplementedError("matscipy only supports full_list=True")

        self.cutoff = float(cutoff)

    def run(self):
        import matscipy.neighbours

        i, j, S, d = matscipy.neighbours.neighbour_list(
            cutoff=self.cutoff,
            positions=self.atoms.positions,
            cell=self.atoms.cell,
            pbc=self.atoms.pbc,
            quantities="ijSd",
        )
        return len(i)
