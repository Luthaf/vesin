from typing import List, Union

import torch
from metatensor.torch import Labels, TensorBlock
from metatomic.torch import NeighborListOptions, System, register_autograd_neighbors

from .. import NeighborList as NeighborListNumpy


try:
    from vesin.torch import NeighborList as NeighborListTorch

except ImportError:

    class NeighborListTorch:
        def __init__(self, cutoff: float, full_list: bool):
            raise ValueError("torchscript=True requires `vesin-torch` as a dependency")

        def compute(
            self,
            points: torch.Tensor,
            box: torch.Tensor,
            periodic: Union[bool, torch.Tensor],
            quantities: str,
            copy: bool = True,
        ) -> List[torch.Tensor]:
            raise ValueError("torchscript=True requires `vesin-torch` as a dependency")


class NeighborList:
    """
    A neighbor list calculator that can be used with metatomic's models.

    The main difference with the other calculators is the automatic handling of
    different length unit between what the model expects and what the ``System`` are
    using.
    """

    def __init__(
        self,
        options: NeighborListOptions,
        length_unit: str,
        torchscript: bool = False,
        check_consistency: bool = False,
    ):
        """
        :param options: :py:class:`metatomic.torch.NeighborListOptions` defining the
            parameters of the neighbor list
        :param length_unit: unit of length used for the systems data
        :param torchscript: whether this function should be compatible with TorchScript
            or not. If ``True``, this requires installing the ``vesin-torch`` package.
        :param check_consistency: whether to run additional checks on the neighbor list
            validity

        Example
        -------

        >>> from vesin.metatomic import NeighborList
        >>> from metatomic.torch import System, NeighborListOptions
        >>> import torch
        >>> system = System(
        ...     positions=torch.eye(3).requires_grad_(True),
        ...     cell=4 * torch.eye(3).requires_grad_(True),
        ...     types=torch.tensor([8, 1, 1]),
        ...     pbc=torch.ones(3, dtype=bool),
        ... )
        >>> options = NeighborListOptions(cutoff=4.0, full_list=True, strict=False)
        >>> calculator = NeighborList(options, length_unit="Angstrom")
        >>> neighbors = calculator.compute(system)
        >>> neighbors
        TensorBlock
            samples (18): ['first_atom', 'second_atom', 'cell_shift_a', 'cell_shift_b', 'cell_shift_c']
            components (3): ['xyz']
            properties (1): ['distance']
            gradients: None
        <BLANKLINE>
        >>>
        >>> # The returned TensorBlock can then be registered with the system
        >>> system.add_neighbor_list(options, neighbors)
        """  # noqa: E501

        self.options = options
        self.length_unit = length_unit
        self.check_consistency = check_consistency

        if torch.jit.is_scripting() or torchscript:
            self._nl = NeighborListTorch(
                cutoff=self.options.engine_cutoff(self.length_unit),
                full_list=self.options.full_list,
            )
        else:
            self._nl = NeighborListNumpy(
                cutoff=self.options.engine_cutoff(self.length_unit),
                full_list=self.options.full_list,
            )

        # cached Labels
        self._components = Labels("xyz", torch.tensor([[0], [1], [2]]))
        self._properties = Labels("distance", torch.tensor([[0]]))

    def compute(self, system: System, copy: bool = True) -> TensorBlock:
        """
        Compute the neighbor list for the given :py:class:`metatomic.torch.System`.

        :param system: a :py:class:`metatomic.torch.System` containing data about a
            single structure. If the positions or cell of this system require gradients,
            the neighbors list values computational graph will be set accordingly.

            The positions and cell need to be in the length unit defined for this
            :py:class:`NeighborList` calculator.
        :param copy: whether to copy the neighbor list values before returning. If
            ``False``, the neighbor list will contain a pointer to memory allocated
            inside the calculator, saving memory and removing one copy. In this case,
            the user MUST ensure that the calculator is kept alive at least as long as
            the neighbor list is used.
        """

        points = system.positions.detach()
        box = system.cell.detach()

        # computes neighbor list
        (P, S, D) = self._nl.compute(
            points=points,
            box=box,
            periodic=system.pbc,
            quantities="PSD",
            copy=copy,
        )
        P = torch.as_tensor(P, dtype=torch.int32)
        S = torch.as_tensor(S, dtype=torch.int32)
        D = torch.as_tensor(D)

        self._components = self._components.to(device=D.device)
        self._properties = self._properties.to(device=D.device)

        # converts to a suitable TensorBlock format
        neighbors = TensorBlock(
            D.reshape(-1, 3, 1),
            samples=Labels(
                names=[
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                values=torch.hstack([P, S]),
                assume_unique=True,
            ),
            components=[self._components],
            properties=self._properties,
        )

        register_autograd_neighbors(
            system, neighbors, check_consistency=self.check_consistency
        )

        return neighbors

    def add_neighbor_list(
        self,
        systems: Union[System, List[System]],
        copy: bool = True,
    ):
        """
        Compute the neighbor list for all the given systems and add it to them.

        :param systems: a system or a list of systems for which to compute and add the
            neighbor list. If the positions or cell of these systems require gradients,
            the neighbors list values computational graph will be set accordingly.

            The positions and cell need to be in the length unit defined for this
            :py:class:`NeighborList` calculator.
        :param copy: whether to copy the neighbor list values before adding them to the
            system. If ``False``, the neighbor list will contain a pointer to memory
            allocated inside the calculator, saving memory and removing one copy. In
            this case, the user MUST ensure that the calculator is kept alive at least
            as long as the neighbor list is used. ``copy=False`` is only supported when
            computing neighbor list for a single system.
        """

        if isinstance(systems, list):
            if len(systems) > 1 and not copy:
                raise ValueError(
                    "`copy=False` is not supported when computing neighbor lists "
                    "for multiple systems with the same calculator"
                )
        else:
            systems = [systems]

        for system in systems:
            neighbors = self.compute(system, copy=copy)
            system.add_neighbor_list(self.options, neighbors)
