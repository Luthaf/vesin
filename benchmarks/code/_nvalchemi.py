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

        from nvalchemiops.torch.neighbors import neighbor_list, prepare_neighbor_list

        self._neighbor_list = neighbor_list

        torch_device = torch.device(device)
        self.positions = torch.tensor(atoms.positions, device=torch_device)
        self.cell = torch.tensor(atoms.cell[:], device=torch_device).unsqueeze(0)
        pbc = torch.tensor(atoms.pbc, device=torch_device)

        # Resolve the strategy and allocate its storage once.
        n_atoms = self.positions.shape[0]
        volume = abs(float(torch.det(self.cell[0])))
        density = n_atoms / volume if volume > 0 else None

        def prepare(max_neighbors=None):
            return prepare_neighbor_list(
                self.positions,
                cutoff,
                self.cell,
                pbc,
                half_fill=not full_list,
                atomic_density=density,
                max_neighbors=max_neighbors,
            )

        self.state = prepare()
        self._neighbor_list(self.positions, cell=self.cell, state=self.state)
        if device == "cuda":
            torch.cuda.synchronize()

        # resize if we needed more neighbors for this system
        needed = int(self.state.num_neighbors.max())
        if needed > self.state.neighbor_matrix.shape[1]:
            self.state = prepare(max_neighbors=needed + 16)

    def run(self):
        _, counts, _ = self._neighbor_list(
            self.positions, cell=self.cell, state=self.state
        )
        # int() syncs, which is what makes the CUDA timings meaningful:
        # benchmark() uses time.time() and never calls torch.cuda.synchronize().
        return int(counts.sum())
