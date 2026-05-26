import os
from dataclasses import dataclass

import ase.io
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def _min_periodic_box_dimension(system) -> float | None:
    periodic_lengths = [
        float(np.linalg.norm(system.box[index]))
        for index, is_periodic in enumerate(system.periodic)
        if is_periodic
    ]
    if len(periodic_lengths) == 0:
        return None
    return min(periodic_lengths)


def algorithm_is_applicable(
    system,
    cutoff: float,
    gpu_algorithm: str,
) -> bool:
    if gpu_algorithm == "cell_list":
        return True

    min_periodic_dim = _min_periodic_box_dimension(system)
    if min_periodic_dim is None:
        return True

    return cutoff <= min_periodic_dim / 2.0


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
        transform_summary="",
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
        transform_summary="",
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
        transform_summary="",
        points=positions,
        box=np.asarray(atoms.cell[:], dtype=np.float64),
        periodic=tuple(bool(value) for value in atoms.pbc),
    )


def naphthalene_cluster() -> SystemForTests:
    atoms = ase.io.read(f"{CURRENT_DIR}/data/naphthalene.xyz")
    return SystemForTests(
        name="naphthalene_cluster_non_periodic",
        transform_summary="",
        points=np.asarray(atoms.positions, dtype=np.float64),
        box=np.diag([30.0, 30.0, 30.0]).astype(np.float64),
        periodic=(False, False, False),
    )


def issue_153() -> SystemForTests:
    return SystemForTests(
        name="issue_153",
        transform_summary="",
        points=np.asarray([[0.0, 2.5, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
        box=np.diag([4.0, 4.5, 0.0]).astype(np.float64),
        periodic=(True, True, False),
    )


def system_from_xyz(name):
    atoms = ase.io.read(f"{CURRENT_DIR}/data/{name}.xyz")
    return SystemForTests(
        name=name,
        transform_summary="",
        points=atoms.positions,
        box=atoms.cell[:],
        periodic=atoms.pbc,
    )


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
    issue_153(),
]
