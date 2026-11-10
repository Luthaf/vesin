import os

import numpy as np

from ._base import NeighborList


class LAMMPSNeighborList(NeighborList):
    """Neighbor list built by LAMMPS, driven through its Python/C library interface.

    LAMMPS only builds neighbor lists on behalf of a pair style (or fix/compute) that
    requests one, so this uses ``pair_style zero`` which computes nothing and only
    requests the list. ``run`` then times a single MD step with a forced reneighboring,
    i.e. rebuilding the ghost atoms (periodic images) and the binned neighbor list.

    LAMMPS does not produce distances or cell shifts: periodic images are represented
    by ghost atoms, so this does not do the same amount of work as implementations
    returning ``ijSd``.
    """

    name = "lammps"
    devices = ("cpu",)
    full_list = (True, False)

    @classmethod
    def version(cls):
        import lammps

        lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
        version = lmp.version()
        lmp.close()
        return str(version)

    def __init__(self, atoms, cutoff, full_list, device="cpu"):
        import lammps

        super().__init__(atoms, cutoff, full_list, device)

        pbc = np.asarray(atoms.pbc, dtype=bool)
        cell = atoms.cell[:]
        positions = atoms.positions

        # LAMMPS uses a "restricted triclinic" cell, with a along x, b in the xy plane
        # and c with a positive z component. Rotate the system to match.
        if np.any(pbc):
            a, b, c = cell
            lx = np.linalg.norm(a)
            xy = np.dot(b, a) / lx
            ly = np.sqrt(np.dot(b, b) - xy**2)
            xz = np.dot(c, a) / lx
            yz = (np.dot(b, c) - xy * xz) / ly
            lz = np.sqrt(np.dot(c, c) - xz**2 - yz**2)
            lammps_cell = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])

            fractional = positions @ np.linalg.inv(cell)
            fractional[:, pbc] %= 1.0
            positions = fractional @ lammps_cell
        else:
            lx = ly = lz = xy = xz = yz = 0.0

        lo = np.zeros(3)
        hi = np.array([lx, ly, lz])
        # tilt factors involving each axis, LAMMPS only allows them along periodic axes
        tilts = [(xy, xz), (xy, yz), (xz, yz)]
        for axis in range(3):
            if pbc[axis]:
                continue
            if any(abs(t) > 1e-12 for t in tilts[axis]):
                raise NotImplementedError(
                    "LAMMPS requires the cell to be orthogonal along non-periodic axes"
                )

            # use a shrink-wrapped boundary around the atoms
            lo[axis] = positions[:, axis].min() - 1.0
            hi[axis] = positions[:, axis].max() + 1.0

        boundary = " ".join("p" if p else "s" for p in pbc)
        # `package omp N` flags all neighbor list requests as threaded, independently
        # of the pair style. LAMMPS defaults to a single thread when OMP_NUM_THREADS
        # is not set (so `package omp 0` would be single threaded), while vesin
        # defaults to one thread per core: pass the same number of threads explicitly.
        try:
            n_threads = int(os.environ.get("OMP_NUM_THREADS", ""))
        except ValueError:
            n_threads = 0
        if n_threads <= 0:
            n_threads = os.cpu_count()

        self.lmp = lammps.lammps(
            cmdargs=[
                "-log",
                "none",
                "-screen",
                "none",
                "-nocite",
                "-pk",
                "omp",
                str(n_threads),
            ]  # fmt: skip
        )
        # LAMMPS silently ignores the thread count if it was compiled without OpenMP,
        # which is the case of the PyPI wheels on macOS. Use benchmarks/setup.sh.
        if self.lmp.extract_setting("nthreads") != n_threads:
            raise RuntimeError(
                f"LAMMPS is not using {n_threads} threads, was it built with OpenMP?"
            )
        self.lmp.commands_string(f"""
            units metal
            atom_style atomic
            atom_modify map array
            boundary {boundary}
            region box prism {lo[0]} {hi[0]} {lo[1]} {hi[1]} {lo[2]} {hi[2]} \
                {xy} {xz} {yz} units box
            create_box 1 box
            mass 1 1.0
        """)

        n_atoms = len(atoms)
        created = self.lmp.create_atoms(
            n_atoms,
            list(range(1, n_atoms + 1)),
            [1] * n_atoms,
            positions.reshape(-1).tolist(),
        )
        if created != n_atoms:
            raise RuntimeError(f"LAMMPS created {created} atoms, expected {n_atoms}")

        # with no skin, the list contains exactly the pairs within the cutoff
        list_kind = "full" if full_list else ""
        self.lmp.commands_string(f"""
            pair_style zero {cutoff} {list_kind}
            pair_coeff * *
            neighbor 0.0 bin
            # `every 1 delay 0 check no` forces a full reneighboring on every step
            neigh_modify every 1 delay 0 check no one 10000 page 200000
            thermo_style custom step
            run 0
        """)

        self.list_index = self.lmp.find_pair_neighlist("zero")
        if self.list_index < 0:
            raise RuntimeError("could not find the LAMMPS neighbor list")

        self.n_pairs = None

    def run(self):
        self.lmp.command("run 1 pre no post no")

        if self.n_pairs is None:
            # counting goes through one Python call per atom, only do it once
            n_pairs = 0
            for element in range(self.lmp.get_neighlist_size(self.list_index)):
                _, numneigh, _ = self.lmp.get_neighlist_element_neighbors(
                    self.list_index, element
                )
                n_pairs += numneigh
            self.n_pairs = n_pairs

        return self.n_pairs
