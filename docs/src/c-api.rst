.. _c-api:

C API reference
===============

Vesin's C API is defined in the ``vesin.h`` header. The main function is
:c:func:`vesin_neighbors`, which runs a neighbors list calculation.

.. doxygenfunction:: vesin_neighbors

.. doxygenfunction:: vesin_free

.. doxygenstruct:: VesinNeighborList

.. doxygenstruct:: VesinOptions

.. doxygenstruct:: VesinDevice

.. doxygenenum:: VesinDeviceKind

.. doxygenenum:: VesinAlgorithm


Verlet caching
--------------

Vesin supports displacement-based Verlet caching to avoid redundant spatial
searches in MD simulations. Set ``options.skin > 0`` when calling
:c:func:`vesin_neighbors` to enable it.

When enabled, the first call performs a full spatial search with
``cutoff + skin`` and caches the pair topology. Subsequent calls reuse the
cached topology as long as no atom has displaced by more than ``skin / 2``
from its reference position. On reuse, only distance vectors are recomputed
from current positions (O(N_pairs) instead of a full O(N) spatial search).

The Verlet state is stored in ``neighbors->opaque`` and managed automatically.
Call ``vesin_free`` as usual to release all resources including the cached state.

Changes that trigger a full rebuild:

- Any atom moving more than ``skin / 2``
- Number of atoms changing
- Box dimensions changing (tolerance 1e-12)
- Periodicity changing
