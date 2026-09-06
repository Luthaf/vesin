from ._base import NeighborList


class ASENeighborList(NeighborList):
    name = "ase"
    devices = ("cpu",)
    full_list = (True,)

    @classmethod
    def version(cls):
        import ase

        return ase.__version__

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        super().__init__(atoms, cutoff, full_list, device)
        if not full_list:
            raise NotImplementedError("ASE only supports full_list=True")

        self.cutoff = float(cutoff)

    def run(self):
        import ase.neighborlist

        i, j, S, d = ase.neighborlist.neighbor_list("ijSd", self.atoms, self.cutoff)
        return len(i)
