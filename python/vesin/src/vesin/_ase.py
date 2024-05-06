try:
    import ase
except ImportError:
    ase = None


from ._neighbors import NeighborList


def ase_neighbor_list(quantities, a, cutoff, self_interaction=False):
    """
    Wrapper around vesin's `NeighborList`

    """
    if ase is None:
        raise ImportError("")

    if self_interaction:
        raise ValueError("self_interaction=True is not implemented")

    if not isinstance(cutoff, float):
        raise ValueError("only a single float cutoff is supported")

    if not isinstance(a, ase.Atoms):
        raise TypeError("TODO")

    if a.pbc[0] and a.pbc[1] and a.pbc[2]:
        periodic = True
    elif not a.pbc[0] and not a.pbc[1] and not a.pbc[2]:
        periodic = False
    else:
        raise ValueError(
            "different periodic boundary conditions on different axis "
            "are not supported"
        )

    nl = NeighborList(cutoff=cutoff, full_list=True)
    return nl.compute(
        points=a.positions,
        box=a.cell[:],
        periodic=periodic,
        quantities=quantities,
        copy=True,
    )
