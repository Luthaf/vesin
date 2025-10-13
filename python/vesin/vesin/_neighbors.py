import ctypes
from ctypes import ARRAY, POINTER
from typing import List, Union

import numpy as np
import numpy.typing as npt

from ._c_api import VesinCPU, VesinDevice, VesinNeighborList, VesinOptions
from ._c_lib import _get_library


class NeighborList:
    """
    A neighbor list calculator.
    """

    def __init__(self, cutoff: float, full_list: bool, sorted: bool = False):
        """
        :param cutoff: spherical cutoff for this neighbor list
        :param full_list: should we return each pair twice (as ``i-j`` and ``j-i``) or
            only once
        :param sorted: Should vesin sort the returned pairs in lexicographic order
            (sorting both ``i`` and then ``j`` at constant ``i``)?
        """
        self._lib = _get_library()
        self.cutoff = float(cutoff)
        self.full_list = bool(full_list)
        self.sorted = bool(sorted)

        self._neighbors = VesinNeighborList()

    def __del__(self):
        if hasattr(self, "_lib") and hasattr(self, "_neighbors"):
            self._lib.vesin_free(self._neighbors)

    def compute(
        self,
        points: "npt.ArrayLike",
        box: "npt.ArrayLike",
        periodic: "Union[bool, npt.ArrayLike]",
        quantities: str = "ij",
        copy=True,
    ) -> List[np.ndarray]:
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

        :param points: positions of all points in the system (this can be anything that
            can be converted to a numpy array)
        :param box: bounding box of the system (this can be anything that can be
            converted to a numpy array)
        :param periodic: per-axis periodic boundary condition mask
        :param quantities: quantities to return, defaults to "ij"
        :param copy: should we copy the returned quantities, defaults to ``True``.
            Setting this to ``False`` might be a bit faster, but the returned arrays are
            view inside this class, and will be invalidated whenever this class is
            garbage collected or used to run a new calculation.

        :return: tuple of arrays as indicated by ``quantities``.
        """
        points = np.asarray(points, dtype=np.float64)
        box = np.asarray(box, dtype=np.float64)
        periodic = np.asarray(periodic, dtype=bool)
        if periodic.ndim == 0:
            periodic = [periodic.item(), periodic.item(), periodic.item()]

        if box.shape != (3, 3):
            raise ValueError("`box` must be a 3x3 matrix")

        Vector = ARRAY(ctypes.c_double, 3)
        box = ARRAY(Vector, 3)(
            Vector(box[0][0], box[0][1], box[0][2]),
            Vector(box[1][0], box[1][1], box[1][2]),
            Vector(box[2][0], box[2][1], box[2][2]),
        )
        periodic = ARRAY(ctypes.c_bool, 3)(periodic[0], periodic[1], periodic[2])

        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError("`points` must be a nx3 array")

        options = VesinOptions()
        options.cutoff = self.cutoff
        options.full = self.full_list
        options.sorted = self.sorted
        options.return_shifts = "S" in quantities
        options.return_distances = "d" in quantities
        options.return_vectors = "D" in quantities

        error_message = ctypes.c_char_p()
        status = self._lib.vesin_neighbors(
            points.ctypes.data_as(POINTER(ARRAY(ctypes.c_double, 3))),
            points.shape[0],
            box,
            periodic,
            VesinDevice(VesinCPU, 0),
            options,
            self._neighbors,
            error_message,
        )

        if status != 0:
            raise RuntimeError(error_message.value.decode("utf8"))

        # create numpy arrays for the data
        n_pairs = self._neighbors.length
        if n_pairs == 0:
            pairs = np.empty((0, 2), dtype=ctypes.c_size_t)
            shifts = np.empty((0, 3), dtype=ctypes.c_int32)
            distances = np.empty((0,), dtype=ctypes.c_double)
            vectors = np.empty((0, 3), dtype=ctypes.c_double)
        else:
            ptr = ctypes.cast(self._neighbors.pairs, POINTER(ctypes.c_size_t))
            pairs = np.ctypeslib.as_array(ptr, shape=(n_pairs, 2))

            if "S" in quantities:
                ptr = ctypes.cast(self._neighbors.shifts, POINTER(ctypes.c_int32))
                shifts = np.ctypeslib.as_array(ptr, shape=(n_pairs, 3))

            if "d" in quantities:
                ptr = ctypes.cast(self._neighbors.distances, POINTER(ctypes.c_double))
                distances = np.ctypeslib.as_array(ptr, shape=(n_pairs,))

            if "D" in quantities:
                ptr = ctypes.cast(self._neighbors.vectors, POINTER(ctypes.c_double))
                vectors = np.ctypeslib.as_array(ptr, shape=(n_pairs, 3))

        # assemble output

        data = []
        for quantity in quantities:
            if quantity == "P":
                if copy:
                    data.append(pairs.copy())
                else:
                    data.append(pairs)

            if quantity == "i":
                if copy:
                    data.append(pairs[:, 0].copy())
                else:
                    data.append(pairs[:, 0])

            elif quantity == "j":
                if copy:
                    data.append(pairs[:, 1].copy())
                else:
                    data.append(pairs[:, 1])

            elif quantity == "S":
                if copy:
                    data.append(shifts.copy())
                else:
                    data.append(shifts)

            elif quantity == "d":
                if copy:
                    data.append(distances.copy())
                else:
                    data.append(distances)

            elif quantity == "D":
                if copy:
                    data.append(vectors.copy())
                else:
                    data.append(vectors)

        return tuple(data)
