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


.. |logo-c| image:: /static/images/logo-c.png
    :height: 1.2em

.. |logo-cxx| image:: /static/images/logo-cxx.png
    :height: 1.2em

.. |logo-fortran| image:: /static/images/logo-fortran.png
    :height: 1.1em

.. |logo-python| image:: /static/images/logo-python.png
    :height: 1.2em

.. |logo-cuda| image:: /static/images/logo-cuda.png
    :height: 1.2em

.. |logo-torch| image:: /static/images/logo-torch.png
    :height: 1.2em

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

    .. tab-item:: |logo-python| Python
        :sync: python

        You can install the code with ``pip``:

        .. code-block:: bash

            pip install vesin

        |logo-torch| **TorchScript:**

        The TorchScript bindings can be installed with:

        .. code-block:: bash

            pip install vesin[torch]

    .. tab-item:: |logo-c| |logo-cxx| |logo-fortran| CMake
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

        |logo-torch| **TorchScript:**

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

        |logo-fortran| **Fortran:**

        To make the fortran bindings of the library available to CMake, you
        should set the ``VESIN_FORTRAN`` option to ``ON``.

        .. code-block:: cmake

            set(VESIN_FORTRAN ON CACHE BOOL "Build the vesin_fortran library")

            add_subdirectory(vesin)
            # or use fetch_content
            FetchContent_xxx(...)


    .. tab-item:: |logo-c| |logo-cxx| Single file

        We support merging all files in the vesin library to a single one that
        can then be included in your own project and built with the same build
        system as the rest of your code.

        You can download this single file from the `github releases
        <https://github.com/Luthaf/vesin/releases>`_, or generate it yourself
        with the following commands:

        .. code-block:: bash

            git clone https://github.com/Luthaf/vesin.git
            cd vesin
            python create-single-cpp.py

        Then you'll need to copy both ``include/vesin.h`` and
        ``vesin-single-build.cpp`` in your project and configure your build
        system accordingly.

        You should define ``VESIN_SHARED`` as a preprocessor constant
        (``-DVESIN_SHARED``) when compiling the code if you want to build vesin
        as a shared library.

        .. important::

            The **TorchScript** API is not supported by the single file file
            build. If you need this feature, please use one of the CMake
            options instead.


        |logo-cuda| **CUDA:**

        You can define ``VESIN_DEFAULT_CUDA_MAX_PAIRS_PER_POINT`` to set the
        default maximum number of pairs per point for the CUDA version of vesin.
        If this is not defined, it will default to 256.

    .. tab-item:: |logo-c| |logo-cxx| |logo-fortran| Global Install

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
        | ``VESIN_INSTALL``            | Should CMake install vesin library and headers   | ON when building vesin directly                |
        |                              |                                                  | OFF when including vesin in another project    |
        +------------------------------+--------------------------------------------------+------------------------------------------------+
        | ``VESIN_TORCH``              | Build (and install if ``VESIN_INSTALL=ON``) the  | OFF                                            |
        |                              | vesin_torch library                              |                                                |
        +------------------------------+--------------------------------------------------+------------------------------------------------+
        | ``VESIN_FORTRAN``            | Build (and install if ``VESIN_INSTALL=ON``) the  | OFF                                            |
        |                              | vesin_fortran library                            |                                                |
        +------------------------------+--------------------------------------------------+------------------------------------------------+


        |logo-torch| **TorchScript:**

        Set ``VESIN_TORCH`` to ``ON`` to build and install the TorchScript
        bindings.

        You can then compile your code, adding ``$PREFIX/include`` to the
        compiler include path, ``$PREFIX/lib`` to the linker library path; and
        linking to ``vesin_torch`` (typically with ``-lvesin_torch``).

        You'll need to also add to the include and linker path the path to the
        same ``libtorch`` installation that was used to build the library.

        |logo-fortran| **Fortran:**

        Set ``VESIN_FORTRAN`` to ``ON`` to build and install the Fortran
        bindings.

        You can then compile your code, adding ``$PREFIX/include`` to the
        compiler include path, ``$PREFIX/lib`` to the linker library path; and
        linking to ``vesin_fortran`` (typically with ``-lvesin_fortran``).


Usage example
-------------

.. tab-set::

    .. tab-item:: |logo-python| Python
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

        :py:func:`NeighborList.compute` accepts any object that can be converted
        to a numpy array for the ``points``, ``box`` and ``periodic`` arguments
        (including lists or tuples). The output arrays are numpy arrays as well.

        We also support `CuPy`_ arrays if you have CuPy installed, and will then
        run the calculation on the GPU. In this case, the output arrays will be
        CuPy arrays as well.

        .. _CuPy: https://cupy.dev/

        Alternatively, you can use the :py:func:`ase_neighbor_list` function,
        which mimics the API of :py:func:`ase.neighborlist.neighbor_list`:

        .. code-block:: Python

            import ase
            from vesin import ase_neighbor_list

            atoms = ase.Atoms(...)

            i, j, S, d = ase_neighbor_list("ijSd", atoms, cutoff=4.2)


    .. tab-item:: |logo-c| |logo-cxx| C and C++
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
                options.sorted = false;
                options.algorithm = VesinAutoAlgorithm;

                // decide what quantities should be computed
                options.return_shifts = true;
                options.return_distances = true;
                options.return_vectors = false;

                VesinNeighborList neighbors;
                memset(&neighbors, 0, sizeof(VesinNeighborList));

                const char* error_message = NULL;
                int status = vesin_neighbors(
                    points,
                    n_points,
                    box,
                    periodic,
                    {VesinCPU, 0},
                    options,
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

        |logo-cuda| **CUDA:**

        To use the CUDA version of vesin, you'll need to allocate the data for
        ``points``, ``box`` and ``periodic`` as device pointers, and then call

        .. code-block:: c++

            int device_id = 0; // choose your GPU device id
            VesinDevice cuda_device = {VesinCUDA, device_id};

            int status = vesin_neighbors(
                d_points,    // device pointer
                n_points,
                d_box,       // device pointer
                d_periodic,  // device pointer
                cuda_device,
                options,
                &neighbors,
                &error_message,
            );

        The data in the ``neighbors`` structure will be directly allocated on
        the device.

    .. tab-item:: |logo-fortran| Fortran

        The fortran bindings provide a module named ``vesin`` which contains the
        ``NeighborList`` type.

        .. code-block:: fortran

            program main
                use vesin, only: NeighborList

                implicit none

                real :: points(:,:)
                real :: box(3,3)
                integer :: i, ierr
                type(NeighborList) :: neighbor_list

                ! define some points positions and box
                points = reshape([                      &
                    0.0_real64, 0.0_real64, 0.0_real64, &
                    0.0_real64, 1.3_real64, 1.3_real64  &
                ], [2, 3])

                box = reshape([                         &
                    3.2_real64, 0.0_real64, 0.0_real64, &
                    0.0_real64, 3.2_real64, 0.0_real64, &
                    0.0_real64, 0.0_real64, 3.2_real64  &
                ], [3, 3])

                ! initialize `neighbor_list` with some options
                neighbor_list = NeighborList(cutoff=4.2, full=.true., sorted=.true.)

                ! run the calculation
                call neighbor_list%compute(points, box, periodic=.true., status=ierr)
                if (ierr /= 0) then
                    write(*, *) neighbor_list%errmsg
                    stop
                end if

                write(*,*) "we got ", neighbor_list%length, "pairs"
                do i=1,neighbor_list%length
                    write(*, *) " - ", i, ":", neighbor_list%pairs(:, i)
                end do

                ! release allocated memory
                call neighbor_list%free()
                deallocate(positions)
            end program main


    .. tab-item:: |logo-torch| TorchScript

        The entry point for the TorchScript API is the
        :py:class:`vesin.torch.NeighborList` class in Python, and the
        corresponding :cpp:class:`vesin_torch::NeighborListHolder` class in C++;
        both modeled after vesin's Python API.

        In both cases, the code is integrated with PyTorch autograd framework,
        meaning if the ``points`` or ``box`` argument have
        ``requires_grad=True``, then the ``d`` (distances) and ``D`` (distance
        vectors) outputs will be integrated to the computational graph.

        |logo-python| **Python:**

        For Python, the ``NeighborList`` class is available in the
        ``vesin.torch`` module.

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

        |logo-cxx| **C++:**

        For C++, the class is available in the ``vesin_torch.hpp`` header.

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
    citation
