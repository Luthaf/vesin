# Vesin: fast neighbor lists for atomistic systems

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](http://luthaf.fr/vesin/)
![Tests](https://img.shields.io/github/check-runs/Luthaf/vesin/main?logo=github&label=tests)

| US English 🇺🇸 | UK English 🇬🇧 | Occitan <img src="./docs/src/static/images/Occitan.png" width=18> | French 🇫🇷 | Gallo‑Italic <img src="./docs/src/static/images/Lombardy.png" width=18> | Catalan <img src="./docs/src/static/images/Catalan.png" width=18> | Spanish 🇪🇸 | Italian 🇮🇹 |
|---------------|---------------|----------|-----------|--------------|---------|------------|------------|
| neighbor      | neighbour     | vesin    | voisin    | visin        | veí     | vecino     | vicino     |



Vesin is a C library that computes neighbor lists for atomistic system, and tries
to be fast and easy to use. We also provide a Python package to call the C
library.

### Installation

To use the code from Python, you can install it with `pip`:

```
pip install git+https://github.com/luthaf/vesin
```

See the [documentation](https://luthaf.fr/vesin/latest/index.html#installation)
for more information on how to install the code to use it from C or C++.

### Usage instruction

You can either use the `NeighborList` calculator class:

```py
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
```

We also provide a function with drop-in compatibility to ASE's neighbor list:

```py
import ase
from vesin import ase_neighbor_list

atoms = ase.Atoms(...)

i, j, S, d = ase_neighbor_list("ijSd", atoms, cutoff=4.2)
```

See the [documentation](https://luthaf.fr/vesin/latest/c-api.html) for more
information on how to use the code from C or C++.

### Benchmarks

You can find below benchmark result for increasingly large diamond supercells,
on Apple M1 Max CPU. You can run this benchmark on your system with the script
at `benchmarks/benchmark.py`.

![Benchmarks](./docs/src/benchmark.png)

## License

Vesin is is distributed under the [3 clauses BSD license](LICENSE). By
contributing to this code, you agree to distribute your contributions under the
same license.
