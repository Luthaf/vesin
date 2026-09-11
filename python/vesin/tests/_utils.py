import os
from dataclasses import dataclass

import ase.io
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def periodic_face_distances(system) -> np.ndarray:
    """Distances between opposite faces of the lattice made of the *periodic*
    box vectors, i.e. how far apart its lattice planes are along each direction.
    Empty for a fully non-periodic system."""
    box = np.asarray(system.box, dtype=np.float64)
    periodic = [index for index, value in enumerate(system.periodic) if value]
    if len(periodic) == 0:
        return np.zeros(0)

    # the columns of the pseudo-inverse are the reciprocal vectors, which works
    # for one and two periodic directions as well as three
    return 1.0 / np.linalg.norm(np.linalg.pinv(box[periodic]), axis=0)


def algorithm_is_applicable(
    system,
    cutoff: float,
    gpu_algorithm: str,
) -> bool:
    if gpu_algorithm == "cell_list" or gpu_algorithm == "auto":
        return True
    elif gpu_algorithm == "brute_force":
        face_distances = periodic_face_distances(system)
        return bool(np.all(cutoff / face_distances < 40))
    else:
        raise ValueError(f"Invalid GPU algorithm {gpu_algorithm}")


def _random_transform_matrix() -> tuple[np.ndarray, bool]:
    rng = np.random.default_rng()
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))

    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1

    inverted = bool(rng.integers(2))
    if inverted:
        rotation = -rotation

    return rotation


def _random_translation() -> np.ndarray:
    rng = np.random.default_rng()
    return rng.uniform(-100.0, 100.0, size=3)


def _format_transformation(
    *,
    transform: np.ndarray | None,
    translation: np.ndarray | None,
) -> str:
    if transform is None and translation is None:
        return ""

    transform_text = np.array2string(
        transform,
        precision=17,
        suppress_small=False,
    )
    message = f"random transform matrix:\n{transform_text}"

    if translation is None:
        return message

    translation_text = np.array2string(
        translation,
        precision=17,
        suppress_small=False,
    )
    return f"{message}\nrandom translation vector:\n{translation_text}"


@dataclass(frozen=True)
class SystemForTests:
    name: str
    transform_summary: str
    points: np.ndarray
    box: np.ndarray
    periodic: tuple[bool, bool, bool]

    def __str__(self) -> str:
        return self.name

    def transform(self, translate: bool = False) -> "SystemForTests":
        transform = _random_transform_matrix()
        translation = _random_translation() if translate else None
        points = self.points @ transform.T
        if translation is not None:
            points = points + translation

        transform_summary = _format_transformation(
            transform=transform,
            translation=translation,
        )

        return SystemForTests(
            name=self.name,
            transform_summary=transform_summary,
            points=points,
            box=self.box @ transform.T,
            periodic=self.periodic,
        )


def polymer_chain() -> SystemForTests:
    # Carbon backbone with realistic C-C distances for a simple 1D polymer-like chain.
    spacing = 1.54
    points = np.array(
        [[index * spacing, 0.0, 0.0] for index in range(8)], dtype=np.float64
    )
    box = np.diag([8 * spacing, 18.0, 18.0]).astype(np.float64)
    return SystemForTests(
        name="polymer_chain",
        transform_summary="nothing",
        points=points,
        box=box,
        periodic=(True, False, False),
    )


def graphene_sheet() -> SystemForTests:
    # 4x4 graphene supercell in the xy plane with vacuum along z.
    a1 = np.array([2.46, 0.0, 0.0])
    a2 = np.array([1.23, 2.13042249, 0.0])
    basis = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.23, 0.71014083, 0.0]),
    ]

    points = []
    for i in range(4):
        for j in range(4):
            origin = i * a1 + j * a2
            for atom in basis:
                points.append(origin + atom)

    return SystemForTests(
        name="graphene_sheet_2d",
        transform_summary="nothing",
        points=np.asarray(points, dtype=np.float64),
        box=np.array(
            [
                [9.84, 0.0, 0.0],
                [4.92, 8.52168996, 0.0],
                [0.0, 0.0, 18.0],
            ],
            dtype=np.float64,
        ),
        periodic=(True, True, False),
    )


def diamond_crystal() -> SystemForTests:
    atoms = ase.io.read(f"{CURRENT_DIR}/data/diamond.xyz")
    atoms = atoms.repeat((2, 2, 2))
    rng = np.random.default_rng(1234)
    positions = np.asarray(atoms.positions, dtype=np.float64).copy()
    positions += rng.normal(scale=1.0, size=positions.shape)
    return SystemForTests(
        name="diamond_crystal_3d",
        transform_summary="nothing",
        points=positions,
        box=np.asarray(atoms.cell[:], dtype=np.float64),
        periodic=tuple(bool(value) for value in atoms.pbc),
    )


def naphthalene_cluster() -> SystemForTests:
    atoms = ase.io.read(f"{CURRENT_DIR}/data/naphthalene.xyz")
    return SystemForTests(
        name="naphthalene_cluster_non_periodic",
        transform_summary="nothing",
        points=np.asarray(atoms.positions, dtype=np.float64),
        box=np.diag([30.0, 30.0, 30.0]).astype(np.float64),
        periodic=(False, False, False),
    )


def atoms_outside_box() -> SystemForTests:
    # Triclinic system where a few atoms have been moved outside of the box by a whole
    # number of box vectors, along both periodic and non-periodic directions.
    atoms = ase.io.read(f"{CURRENT_DIR}/data/carbon.xyz")
    atoms = atoms.repeat((3, 3, 3))

    positions = np.asarray(atoms.positions, dtype=np.float64).copy()
    box = np.asarray(atoms.cell[:], dtype=np.float64)

    positions[3] += box[0]
    positions[10] -= 2 * box[1]
    positions[20] += box[2] - box[0]

    return SystemForTests(
        name="atoms_outside_box",
        transform_summary="nothing",
        points=positions,
        box=box,
        periodic=(True, False, True),
    )


def non_periodic_points_outside_box() -> SystemForTests:
    # Non-periodic system where the points are outside of the box: the box should be
    # completely ignored when computing the neighbors.
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.1, 1.2, 1.3],
            [9.5, 0.5, 0.5],
            [10.1, 0.5, 0.5],
            [-3.2, 4.7, -1.8],
            [5.0, 2.0, 0.5],
        ],
        dtype=np.float64,
    )

    return SystemForTests(
        name="non_periodic_points_outside_box",
        transform_summary="nothing",
        points=points,
        box=np.diag([1.1, 1.2, 1.3]).astype(np.float64),
        periodic=(False, False, False),
    )


def thin_film() -> SystemForTests:
    # Periodic in all three directions, but much thinner than the cutoff along
    # x: the search has to reach many periodic images in that one direction and
    # only one or two in the others.
    #
    # The thickness must not divide any of TEST_CUTOFFS, or an atom lands
    # exactly on the cutoff from its own periodic image and every backend gets
    # to disagree about it in the last bit. 0.6 did exactly that: 5 * 0.6 = 3.0.
    thickness = 0.57
    lengths = np.array([thickness, 11.0, 11.0])
    rng = np.random.default_rng(7)
    return SystemForTests(
        name="thin_film_anisotropic",
        transform_summary="nothing",
        points=(rng.random((6, 3)) * lengths).astype(np.float64),
        box=np.diag(lengths).astype(np.float64),
        periodic=(True, True, True),
    )


def issue_153() -> SystemForTests:
    return SystemForTests(
        name="issue_153",
        transform_summary="nothing",
        points=np.asarray([[0.0, 2.5, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        box=np.diag([4.0, 4.5, 0.0]).astype(np.float64),
        periodic=(True, True, False),
    )


def system_from_xyz(name):
    atoms = ase.io.read(f"{CURRENT_DIR}/data/{name}.xyz")
    return SystemForTests(
        name=name,
        transform_summary="nothing",
        points=atoms.positions,
        box=atoms.cell[:],
        periodic=atoms.pbc,
    )


# Cutoffs the test suites sweep over.
TEST_CUTOFFS = [3.0, 5.0, 7.0, 10.0]

TEST_SYSTEMS = [
    system_from_xyz("water"),
    system_from_xyz("diamond"),
    system_from_xyz("naphthalene"),
    system_from_xyz("carbon"),
    system_from_xyz("slab"),
    system_from_xyz("Cd2I4O12"),
    system_from_xyz("rotated_box"),
    naphthalene_cluster(),
    polymer_chain(),
    graphene_sheet(),
    diamond_crystal(),
    atoms_outside_box(),
    non_periodic_points_outside_box(),
    thin_film(),
    issue_153(),
]


# Half-list pair direction is not part of the contract: both cell lists pick
# which atom of a pair comes out as `i` from the offset between the two cells,
# so it depends on the grid -- and the grid depends on the cutoff (Verlet builds
# its candidates at `cutoff + skin`) and on the backend (the CPU puts one cell
# per cutoff, CUDA picks several). The same physical pair can therefore come out
# as `(i, j, S)` from one call and `(j, i, -S)` from another.
#
# These two apply the same canonicalization, one to (i, j, shift) tuples and one
# to whole arrays; keep them in agreement.
def canonical_half_pair(i, j, sx, sy, sz):
    """Canonicalize one half-list pair given as a tuple of scalars."""
    if i > j:
        return (j, i, -sx, -sy, -sz)
    if i == j:
        # `i > j` cannot separate a pair of an atom with its own periodic
        # image, so those are canonicalized on the shift instead
        total = sx + sy + sz
        if total < 0 or (total == 0 and (sz < 0 or (sz == 0 and sy < 0))):
            return (i, j, -sx, -sy, -sz)
    return (i, j, sx, sy, sz)


def canonicalize_half_pairs(i, j, S, D, d):
    """Canonicalize half-list pairs given as arrays, negating `S` and `D` with
    the swap and leaving `d` alone. The inputs are not modified."""
    i, j, S, D = i.copy(), j.copy(), S.copy(), D.copy()

    swap = i > j

    total = S.sum(axis=1)
    swap |= (i == j) & (
        (total < 0)
        | ((total == 0) & ((S[:, 2] < 0) | ((S[:, 2] == 0) & (S[:, 1] < 0))))
    )

    i[swap], j[swap] = j[swap], i[swap]
    S[swap] = -S[swap]
    D[swap] = -D[swap]

    return i, j, S, D, d
