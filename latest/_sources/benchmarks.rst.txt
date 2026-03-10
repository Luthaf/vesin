.. _benchmarks:

Benchmarks
==========

Here are the result of a benchmark of multiple neighbor list implementations.
The benchmark runs on multiple super-cell of diamond carbon, up to 30'000 atoms,
with multiple cutoffs, and using either CPU or CUDA hardware.

The results below are for an AMD 3955WX CPU and an NVIDIA 4070 Ti SUPER GPU; if
you want to run it on your own system, the corresponding script is in vesin's
`GitHub repository <bench-script_>`_.

.. _bench-script: https://github.com/Luthaf/vesin/blob/main/benchmarks/benchmark.py

.. figure:: benchmark.png
    :align: center

    Speed comparison between multiple neighbor list implementations: vesin, `ase
    <https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html>`_, `matscipy
    <http://libatoms.github.io/matscipy/tools/neighbour_list.html>`_, `pymatgen
    <https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.IStructure.get_neighbor_list>`_,
    `torch_nl <https://github.com/felixmusil/torch_nl/>`_, and `NNPOps
    <https://github.com/openmm/NNPOps/>`_.

    Missing points indicate that a specific code could not run the calculation
    (for example, NNPOps requires the cell to be twice the cutoff in size, and
    can't run with large cutoffs and small cells).
