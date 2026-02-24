from typing import List, Optional, Union

import torch
from metatomic.torch import (
    AtomisticModel,
    ModelInterface,
    NeighborListOptions,
    System,
)

from ._neighbors import NeighborList


def compute_requested_neighbors(
    systems: Union[List[System], System],
    system_length_unit: str,
    model: Union[AtomisticModel, ModelInterface],
    model_length_unit: Optional[str] = None,
    check_consistency: bool = False,
):
    """
    Compute all neighbors lists requested by the ``model`` through
    ``requested_neighbor_lists()`` member functions, and store them inside all the
    ``systems``.

    .. seealso::

        :py:func:`vesin.metatomic.compute_requested_neighbors_from_options` which 
        is compatible with TorchScript and can be used inside a model.

    :param systems: Single system or list of systems for which we need to compute the
        neighbor lists that the model requires.
    :param system_length_unit: unit of length used by the data in ``systems``
    :param model: :py:class:`AtomisticModel` or any ``torch.nn.Module`` following the
        :py:class:`ModelInterface`
    :param model_length_unit: unit of length used by the model, optional. This is only
        required when giving a raw model instead of a :py:class:`AtomisticModel`.
    :param check_consistency: whether to run additional checks on the neighbor lists
        validity
    """

    if isinstance(model, AtomisticModel):
        if model_length_unit is not None:
            if model.capabilities().length_unit != model_length_unit:
                raise ValueError(
                    f"the given `model_length_unit` ({model_length_unit}) does not "
                    f"match the model capabilities ({model.capabilities().length_unit})"
                )

        all_options = model.requested_neighbor_lists()
    elif isinstance(model, torch.nn.Module):
        if model_length_unit is None:
            raise ValueError(
                "`model_length_unit` parameter is required when not "
                "using AtomisticModel"
            )

        all_options = []
        _get_requested_neighbor_lists(
            model, model.__class__.__name__, all_options, model_length_unit
        )

    compute_requested_neighbors_from_options(
        systems, all_options, system_length_unit, check_consistency
    )


def _get_requested_neighbor_lists(
    module: torch.nn.Module,
    module_name: str,
    requested: List[NeighborListOptions],
    length_unit: str,
):
    """
    Recursively extract the requested neighbor lists from a non-exported metatomic
    model.
    """
    if hasattr(module, "requested_neighbor_lists"):
        for new_options in module.requested_neighbor_lists():
            new_options.add_requestor(module_name)

            already_requested = False
            for existing in requested:
                if existing == new_options:
                    already_requested = True
                    for requestor in new_options.requestors():
                        existing.add_requestor(requestor)

            if not already_requested:
                if new_options.length_unit not in ["", length_unit]:
                    raise ValueError(
                        f"NeighborsListOptions from {module_name} already have a "
                        f"length unit ('{new_options.length_unit}') which does not "
                        f"match the model length units ('{length_unit}')"
                    )

                new_options.length_unit = length_unit
                requested.append(new_options)

    for child_name, child in module.named_children():
        _get_requested_neighbor_lists(
            module=child,
            module_name=module_name + "." + child_name,
            requested=requested,
            length_unit=length_unit,
        )


def compute_requested_neighbors_from_options(
    systems: List[System],
    options: List[NeighborListOptions],
    system_length_unit: str,
    check_consistency: bool,
) -> None:
    """
    Compute all neighbors lists requested by the ``options``, and store them inside all
    the ``systems``.

    :param systems: list of systems for which we need to compute the
        neighbor lists that required by the ``options``.
    :param options: list of :py:class:`NeighborListOptions`
    :param system_length_unit: unit of length used by the data in ``systems``
    :param check_consistency: whether to run additional checks on the neighbor lists
        validity
    """

    if not isinstance(systems, list):
        systems = [systems]

    for option in options:
        calculator = NeighborList(
            option,
            system_length_unit,
            check_consistency=check_consistency,
        )

        for system in systems:
            neighbors = calculator.compute(system)
            system.add_neighbor_list(option, neighbors)
