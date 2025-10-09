Vesin: we are all neighbors
===========================

.. |occ| image:: /static/images/Occitan.png
    :width: 18px

.. |arp| image:: /static/images/Arpitan.png
    :width: 18px

.. |lomb| image:: /static/images/Lombardy.png
    :width: 18px

.. |cat| image:: /static/images/Catalan.png
    :width: 18px

.. list-table::
    :align: center
    :widths: auto
    :header-rows: 1

    - * English 🇺🇸⁠/⁠🇬🇧
      * Occitan |occ|
      * French 🇫🇷
      * Arpitan |arp|
      * Gallo‑Italic |lomb|
      * Catalan |cat|
      * Spanish 🇪🇸
      * Italian 🇮🇹
    - * neighbo(u)r
      * vesin
      * voisin
      * vesin
      * visin
      * veí
      * vecino
      * vicino


``vesin`` is a lightweight neighbor list calculator for molecular systems and
three-dimensional graphs. It is written in C++ and can be used as a standalone
library from C or Python. ``vesin`` is designed to be :ref:`fast <benchmarks>`
and easy to use.

Installation
------------

.. tab-set::

    .. tab-item:: Python
        :sync: python

        You can install the code with ``pip``:

        .. code-block:: bash

            pip install vesin

        **TorchScript:**

        The TorchScript bindings can be installed with:

        .. code-block:: bash

            pip install vesin[torch]

    .. tab-item:: C/C++ (CMake)
        :sync: cxx

        If you use CMake as your build system, the simplest thing to do is to
        add https://github.com/Luthaf/vesin to your project.

        .. code-block:: cmake

            # assuming the code is in the `vesin/` directory (for example using
            # git submodule)
            add_subdirectory(vesin)

            target_link_libraries(your-target vesin)

        Alternatively, you can use CMake's `FetchContent
        <https://cmake.org/cmake/help/latest/module/FetchContent.html>`_ module
        to automatically download the code:

        .. code-block:: cmake

            include(FetchContent)
            FetchContent_Declare(
                vesin
                GIT_REPOSITORY https://github.com/Luthaf/vesin.git
            )

            FetchContent_MakeAvailable(vesin)

            target_link_libraries(your-target vesin)

        **TorchScript:**

        To make the TorchScript version of the library available to CMake as
        well, you should set the ``VESIN_TORCH`` option to ``ON``. If you are
        using ``add_subdirectory(vesin)``:

        .. code-block:: cmake

            set(VESIN_TORCH ON CACHE BOOL "Build the vesin_torch library")

            add_subdirectory(vesin)

            target_link_libraries(your-target vesin_torch)

        And if you are using ``FetchContent``:

        .. code-block:: cmake

            set(VESIN_TORCH ON CACHE BOOL "Build the vesin_torch library")

            # like above
            FetchContent_Declare(...)
            FetchContent_MakeAvailable(...)

            target_link_libraries(your-target vesin_torch)

    .. tab-item:: C/C++ (single file build)

        We support merging all files in the vesin library to a single one that
        can then be included in your own project and built with the same build
        system as the rest of your code.

        You can generate this single file to build with the following commands:

        .. code-block:: bash

            git clone https://github.com/Luthaf/vesin.git
            cd vesin
            python create-single-cpp.py

        Then you'll need to copy both ``include/vesin.h`` and
        ``vesin-single-build.cpp`` in your project and configure your build
        system accordingly.

        **TorchScript:**

        The TorchScript API does not support single file build, please use one
        of the CMake options instead.


    .. tab-item:: C/C++ (global installation)

        You can build and install vesin in some global location (referred to as
        ``$PREFIX`` below), and then use the right compiler flags to give this
        location to your compiler. In this case, compilation of ``vesin`` and
        your code happen separately.

        .. code-block:: bash

            git clone https://github.com/Luthaf/vesin.git
            cd vesin
            mkdir build && cd build
            cmake -DCMAKE_INSTALL_PREFIX=$PREFIX <other options> ..
            cmake --install .

        You can then compile your code, adding ``$PREFIX/include`` to the
        compiler include path, ``$PREFIX/lib`` to the linker library path; and
        linking to vesin (typically with ``-lvesin``). If you are building vesin
        as a shared library, you'll also need to define ``VESIN_SHARED`` as a
        preprocessor constant (``-DVESIN_SHARED`` when compiling the code).

        Some relevant cmake options you can customize:

        +------------------------------+--------------------------------------------------+------------------------------------------------+
        | Option                       | Description                                      | Default                                        |
        +==============================+==================================================+================================================+
        | ``CMAKE_BUILD_TYPE``         | Type of build: Debug or Release                  | Release                                        |
        +------------------------------+--------------------------------------------------+------------------------------------------------+
        | ``CMAKE_INSTALL_PREFIX``     | Prefix where the library will be installed       | ``/usr/local``                                 |
        +------------------------------+--------------------------------------------------+------------------------------------------------+
        | ``BUILD_SHARED_LIBS``        | Default to building and installing a shared      | OFF                                            |
        |                              | library instead of a static one                  |                                                |
        +------------------------------+--------------------------------------------------+------------------------------------------------+
        | ``VESIN_INSTALL``            | Should CMake install vesin library and headers   | | ON when building vesin directly              |
        |                              |                                                  | | OFF when including vesin in another project  |
        +------------------------------+--------------------------------------------------+------------------------------------------------+
        | ``VESIN_TORCH``              | Build (and install if ``VESIN_INSTALL=ON``) the  | OFF                                            |
        |                              | vesin_torch library                              |                                                |
        +------------------------------+--------------------------------------------------+------------------------------------------------+

        **TorchScript:**

        Set ``VESIN_TORCH`` to ``ON`` to build and install the TorchScript
        bindings.

        You can then compile your code, adding ``$PREFIX/include`` to the
        compiler include path, ``$PREFIX/lib`` to the linker library path; and
        linking to vesin_torch (typically with ``-lvesin_torch``).

        You'll need to also add to the include and linker path the path to the
        same torch installation that was used to build the library.


Usage example
-------------

.. tab-set::

    .. tab-item:: Python
        :sync: python

        .. py:currentmodule:: vesin

        There are two ways to use vesin from Python, you can use the
        :py:class:`NeighborList` class:

        .. code-block:: Python

            import numpy as np
            from vesin import NeighborList

            # positions can be anything compatible with numpy's ndarray
            positions = [
                (0, 0, 0),
                (0, 1.3, 1.3),
            ]
            box = 3.2 * np.eye(3)

            calculator = NeighborList(cutoff=4.2, full_list=True)
            i, j, S, d = calculator.compute(
                points=points,
                box=box,
                periodic=True,
                quantities="ijSd"
            )

        Alternatively, you can use the :py:func:`ase_neighbor_list` function,
        which mimics the API of :py:func:`ase.neighborlist.neighbor_list`:

        .. code-block:: Python

            import ase
            from vesin import ase_neighbor_list

            atoms = ase.Atoms(...)

            i, j, S, d = ase_neighbor_list("ijSd", atoms, cutoff=4.2)


    .. tab-item:: C and C++
        :sync: cxx

        .. code-block:: c++

            #include <string.h>
            #include <stdio.h>
            #include <stdlib.h>

            #include <vesin.h>

            int main() {
                // points can be any pointer to `double[3]`
                double points[][3] = {
                    {0, 0, 0},
                    {0, 1.3, 1.3},
                };
                size_t n_points = 2;

                // box can be any `double[3][3]` array
                double box[3][3] = {
                    {3.2, 0.0, 0.0},
                    {0.0, 3.2, 0.0},
                    {0.0, 0.0, 3.2},
                };
                bool periodic = true;

                // calculation setup
                VesinOptions options;
                options.cutoff = 4.2;
                options.full = true;

                // decide what quantities should be computed
                options.return_shifts = true;
                options.return_distances = true;
                options.return_vectors = false;

                VesinNeighborList neighbors;
                memset(&neighbors, 0, sizeof(VesinNeighborList));

                const char* error_message = NULL;
                int status = vesin_neighbors(
                    points, n_points, box, periodic,
                    VesinCPU, options,
                    &neighbors,
                    &error_message,
                );

                if (status != EXIT_SUCCESS) {
                    fprintf(stderr, "error: %s\n", error_message);
                    return 1;
                }

                // use neighbors as needed
                printf("we have %d pairs\n", neighbors.length);

                vesin_free(&neighbors);

                return 0;
            }

    .. tab-item:: TorchScript Python

        The entry point for the TorchScript API is the
        :py:class:`vesin.torch.NeighborList` class in Python, and the
        corresponding :cpp:class:`vesin_torch::NeighborListHolder` class in C++;
        both modeled after the standard Python API. For Python, the class is
        available in the ``vesin.torch`` module.

        In both cases, the code is integrated with PyTorch autograd framework,
        meaning if the ``points`` or ``box`` argument have
        ``requires_grad=True``, then the ``d`` (distances) and ``D`` (distance
        vectors) outputs will be integrated to the computational graph.

        .. code-block:: Python

            import torch
            from vesin.torch import NeighborList

            positions = torch.tensor(
                [[0.0, 0.0, 0.0],
                 [0.0, 1.3, 1.3]],
                dtype=torch.float64,
                requires_grad=True,
            )
            box = 3.2 * torch.eye(3, dtype=torch.float64)

            calculator = NeighborList(cutoff=4.2, full_list=True)
            i, j, S, d = calculator.compute(
                points=points,
                box=box,
                periodic=True,
                quantities="ijSd"
            )

    .. tab-item:: TorchScript C++

        The entry point for the TorchScript API is the
        :py:class:`vesin.torch.NeighborList` class in Python, and the
        corresponding :cpp:class:`vesin_torch::NeighborListHolder` class in C++;
        both modeled after the standard Python API. For C++, the class is
        available in the ``vesin_torch.hpp`` header.

        In both cases, the code is integrated with PyTorch autograd framework,
        meaning if the ``points`` or ``box`` argument have
        ``requires_grad=True``, then the ``d`` (distances) and ``D`` (distance
        vectors) outputs will be integrated to the computational graph.

        .. code-block:: C++

            #include <torch/torch.h>

            #include <vesin_torch.hpp>

            int main() {
                auto options = torch::TensorOptions().dtype(torch::kFloat64);
                auto positions = torch.tensor(
                    {{0.0, 0.0, 0.0},
                     {0.0, 1.3, 1.3}},
                    options
                );
                positions.requires_grad_(true);

                auto box = 3.2 * torch.eye(3, options);

                auto calculator = torch::make_intrusive<NeighborListHolder>(
                    /*cutoff=*/ 4.2,
                    /*full_list=*/ true
                );

                calculator.
                auto outputs = calculator.compute(
                    /*points=*/ points,
                    /*box=*/ box,
                    /*periodic=*/ true,
                    /*quantities=*/ "ijSd",
                    /*copy=*/ true,
                );

                auto i = outputs[0];
                auto j = outputs[1];
                auto S = outputs[2];
                auto d = outputs[3];

                // ...
            }


API Reference
-------------

.. toctree::
    :maxdepth: 1
    :hidden:

    Vesin <self>

.. toctree::
    :maxdepth: 1

    python-api
    torch-api
    c-api
    fortran-api
    metatomic


.. toctree::
    :maxdepth: 1
    :hidden:

    benchmarks
