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
