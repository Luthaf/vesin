try:
    import ase
except ImportError:
    ase = None


from ._neighbors import NeighborList


def ase_neighbor_list(quantities, a, cutoff, self_interaction=False, max_nbins=0):
    """
    This is a thin wrapper around :py:class:`NeighborList`, providing the same API as
    :py:func:`ase.neighborlist.neighbor_list`.

    It is intended as a drop-in replacement for the ASE function, but only supports a
    subset of the functionality. Notably, the following is not supported:

    - ``self_interaction=True``
    - :py:class:`ase.Atoms` with mixed periodic boundary conditions
    - giving ``cutoff`` as a dictionary

    :param quantities: quantities to output from the neighbor list. Supported are
        ``"i"``, ``"j"``, ``"d"``, ``"D"``, and ``"S"`` with the same meaning as in ASE.
    :param a: :py:class:`ase.Atoms` instance
    :param cutoff: cutoff radius for the neighbor list
    :param self_interaction: Should an atom be considered its own neighbor? Default:
        False
    :param max_nbins: for ASE compatibility, ignored by this implementation
    """
    if ase is None:
        raise ImportError("could not import ase, this function requires ase")

    if self_interaction:
        raise ValueError("self_interaction=True is not implemented")

    if not isinstance(cutoff, float):
        raise ValueError("only a single float cutoff is supported")

    if not isinstance(a, ase.Atoms):
        raise TypeError(f"`a` should be ase.Atoms, got {type(a)} instead")

    if a.pbc[0] and a.pbc[1] and a.pbc[2]:
        periodic = True
    elif not a.pbc[0] and not a.pbc[1] and not a.pbc[2]:
        periodic = False
    else:
        raise ValueError(
            "different periodic boundary conditions on different axis are not supported"
        )

    # sorted=True and full_list=True since that's what ASE does
    calculator = NeighborList(cutoff=cutoff, full_list=True, sorted=True)
    return calculator.compute(
        points=a.positions,
        box=a.cell[:],
        periodic=periodic,
        quantities=quantities,
        copy=True,
    )
