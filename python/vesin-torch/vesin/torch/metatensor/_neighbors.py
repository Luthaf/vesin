import torch

from .. import NeighborList as NeighborListTorch


try:  # only define a metatensor adapter if metatensor is available
    from metatensor.torch import Labels, TensorBlock
    from metatensor.torch.atomistic import NeighborListOptions, System

    _HAS_METATENSOR = True
except ModuleNotFoundError:
    _HAS_METATENSOR = False

    class Labels:
        pass

    class TensorBlock:
        pass

    class System:
        pass

    class NeighborListOptions:
        pass


class NeighborList:
    """
    A neighbor list calculator that can be used with metatensor's atomistic models.

    The main difference with the other calculators is the automatic handling of
    different length unit between what the model expects and what the ``System`` are
    using.

    .. seealso::

        The :py:func:`vesin.torch.metatensor.compute_requested_neighbors` function can
        be used to automatically compute and store all neighbor lists required by a
        given model.
    """

    def __init__(self, options: NeighborListOptions, length_unit: str):
        """
        :param options: :py:class:`metatensor.torch.atomistic.NeighborListOptions`
            defining the parameters of the neighbor list
        :param length_unit: unit of length used for the systems data

        Example
        -------

        >>> from vesin.torch.metatensor import NeighborList
        >>> from metatensor.torch.atomistic import System, NeighborListOptions
        >>> import torch
        >>> system = System(
        ...     positions=torch.eye(3).requires_grad_(True),
        ...     cell=4 * torch.eye(3).requires_grad_(True),
        ...     types=torch.tensor([8, 1, 1]),
        ...     pbc=torch.ones(3, dtype=bool),
        ... )
        >>> options = NeighborListOptions(cutoff=4.0, full_list=True)
        >>> calculator = NeighborList(options, length_unit="Angstrom")
        >>> neighbors = calculator.compute(system)
        >>> neighbors
        TensorBlock
            samples (54): ['first_atom', 'second_atom', 'cell_shift_a', 'cell_shift_b',
        ...    'cell_shift_c']
            components (3): ['xyz']
            properties (1): ['distance']
            gradients: None
        >>> # The returned TensorBlock can then be registered with the system
        >>> system.add_neighbor_list(options, neighbors)
        """

        if not torch.jit.is_scripting():
            if not _HAS_METATENSOR:
                raise ModuleNotFoundError(
                    "`vesin.metatensor` requires the `metatensor-torch` package"
                )
        self.options = options
        self.length_unit = length_unit
        self._nl = NeighborListTorch(
            cutoff=self.options.engine_cutoff(self.length_unit),
            full_list=self.options.full_list,
        )

        # cached Labels
        self._components = [Labels("xyz", torch.tensor([[0], [1], [2]]))]
        self._properties = Labels(["distance"], torch.tensor([[0]]))

    def compute(self, system: System) -> TensorBlock:
        """
        Compute the neighbor list for the given
        :py:class:`metatensor.torch.atomistic.System`.

        :param system: a :py:class:`metatensor.torch.atomistic.System` containing the
            data about a structure. If the positions or cell of this system require
            gradients, the neighbors list values computational graph will be set
            accordingly.

            The positions and cell need to be in the length unit defined for this
            :py:class`NeighborList` calculator.
        """

        # move to float64, as vesin only works in torch64
        points = system.positions.to(torch.float64)
        box = system.cell.to(torch.float64)
        if torch.all(system.pbc):
            periodic = True
        elif not torch.any(system.pbc):
            periodic = False
        else:
            raise NotImplementedError(
                "vesin currently does not support mixed periodic and non-periodic "
                "boundary conditions"
            )

        # computes neighbor list
        (P, S, D) = self._nl.compute(
            points=points, box=box, periodic=periodic, quantities="PSD", copy=True
        )

        # converts to a suitable TensorBlock format
        neighbors = TensorBlock(
            D.reshape(-1, 3, 1).to(system.positions.dtype),
            samples=Labels(
                names=[
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                values=torch.hstack([P, S]),
            ),
            components=self._components,
            properties=self._properties,
        )

        return neighbors
