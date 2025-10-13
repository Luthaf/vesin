from typing import List, Union

import torch


class NeighborList:
    """A neighbor list calculator that can be used with TorchScript."""

    def __init__(self, cutoff: float, full_list: bool):
        """
        :param cutoff: spherical cutoff for this neighbor list
        :param full_list: should we return each pair twice (as ``i-j`` and ``j-i``) or
            only once
        """
        self._c = torch.classes.vesin._NeighborList(cutoff=cutoff, full_list=full_list)

    def compute(
        self,
        points: torch.Tensor,
        box: torch.Tensor,
        periodic: Union[bool, torch.Tensor],
        quantities: str,
        copy: bool = True,
    ) -> List[torch.Tensor]:
        """
        Compute the neighbor list for the system defined by ``positions``, ``box``, and
        ``periodic``; returning the requested ``quantities``.

        ``quantities`` can contain any combination of the following values:

        - ``"i"`` to get the index of the first point in the pair
        - ``"j"`` to get the index of the second point in the pair
        - ``"P"`` to get the indexes of the two points in the pair simultaneously
        - ``"S"`` to get the periodic shift of the pair
        - ``"d"`` to get the distance between points in the pair
        - ``"D"`` to get the distance vector between points in the pair

        :param points: positions of all points in the system
        :param box: bounding box of the system
        :param periodic: should we use periodic boundary conditions?
        :param quantities: quantities to return, defaults to "ij"
        :param copy: should we copy the returned quantities, defaults to ``True``.
            Setting this to ``False`` might be a bit faster, but the returned tensors
            are view inside this class, and will be invalidated whenever this class is
            garbage collected or used to run a new calculation.

        :return: list of :py:class:`torch.Tensor` as indicated by ``quantities``.
        """

        if isinstance(periodic, bool):
            periodic = torch.tensor([periodic, periodic, periodic], dtype=torch.bool)

        return self._c.compute(
            points=points,
            box=box,
            periodic=periodic,
            quantities=quantities,
            copy=copy,
        )
