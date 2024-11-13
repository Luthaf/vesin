from typing import List, Optional, Union

import torch

from ._neighbors import NeighborList


try:
    from metatensor.torch.atomistic import (
        MetatensorAtomisticModel,
        ModelInterface,
        NeighborListOptions,
        System,
    )

    _HAS_METATENSOR = True
except ModuleNotFoundError:
    _HAS_METATENSOR = False

    class System:
        pass

    class MetatensorAtomisticModel:
        pass


def compute_all_neighbors(
    systems: Union[List[System], System],
    system_length_unit: str,
    model: Union[MetatensorAtomisticModel, ModelInterface],
    model_length_unit: Optional[str] = None,
):
    """
    Compute all neighbors lists required by the model, and store them inside all the
    systems.

    :param systems: Single system or list of systems for which we need to compute the
        neighbor lists that the model requires.
    :param system_length_unit: unit of length used by the data in ``systems``
    :param model: :py:class:`MetatensorAtomisticModel` or any ``torch.nn.Module``
        following the :py:class:`ModelInterface`
    :param model_length_unit: unit of length used by the model. This is only required
        when giving a raw model instead of a :py:class:`MetatensorAtomisticModel`.
    """

    if isinstance(model, MetatensorAtomisticModel):
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
                "using MetatensorAtomisticModel"
            )

        all_options = []
        _get_requested_neighbor_lists(
            model, model.__class__.__name__, all_options, model_length_unit
        )

    if not isinstance(systems, list):
        systems = [systems]

    for options in all_options:
        calculator = NeighborList(options, system_length_unit)
        for system in systems:
            neighbors = calculator.compute(system)
            system.add_neighbor_list(options, neighbors)


def _get_requested_neighbor_lists(
    module: torch.nn.Module,
    module_name: str,
    requested: List[NeighborListOptions],
    length_unit: str,
):
    """
    Recursively extract the requested neighbor lists from a non-exported metatensor
    atomistic model.
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
