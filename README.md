# Vesin: fast neighbor lists for atomistic systems

![Tests](https://img.shields.io/github/check-runs/Luthaf/vesin/main?logo=github&label=tests)

This is a work in progress!

## Python API

### Installation

```bash
pip install git+https://github.com/luthaf/vesin
```

### Usage

Generic interface:

```py
from vesin import NeighborList
import numpy as np

positions = [
    (0, 0, 0),
    (0, 1.3, 1.3),
]
box = 3.2 * np.eye(3)

nl = NeighborList(cutoff=4.2, full_list=True)
i, j, S, d = nl.compute(
    points=points,
    box=box,
    periodic=True,
    quantities="ijSd"
)
```

[ASE](https://wiki.fysik.dtu.dk/ase/) interface:

```py
from vesin import ase_neighbor_list
import ase

atoms = ase.Atoms(...)

i, j, S, d = ase_neighbor_list("ijSd", atoms, cutoff=4.2)
```

### Benchmarks

Benchmark result for increasingly large diamond supercells, on Apple M1 Max CPU.
You can run this benchmark on your system with th script at
`benchmarks/benchmark.py`.

![Benchmarks](./docs/src/benchmark.png)


## C/C++ API

### Installation

```bash
git clone git+https://github.com/luthaf/vesin
cd vesin

mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX ..
cmake --build .
cmake --install .
```

Alternatively, run

```bash
./create-single-cpp.py
```

Copy `vesin-single-build.cpp` and `include/vesin.h` in your project, and compile
the code in C++17 or higher mode. In this case, you'll need to define
`VESIN_SHARED` whenever you include the header to use the code as a shared
library, and additionally define `VESIN_EXPORTS` when building the shared
library itself. If you are using the code as a static library, you don't have to
do anything.

### Usage

Compile with `-I $INSTALL_PREFIX/include`, and link with `-L $INSTALL_PREFIX/lib
-lvesin`.


```c
#include <string.h>
#include <stdio.h>

#include <vesin.h>

int main() {
    // data
    double points[][3] = {
        {0, 0, 0},
        {0, 1.3, 1.3},
    };
    size_t n_points = 2;

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
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = false;

    VesinNeighbors neighbors;
    memset(&neighbors, 0, sizeof(VesinNeighbors));

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

    vesin_free(&neighbors);

    return 0;
}
```
