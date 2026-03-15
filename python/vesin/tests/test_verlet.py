"""Tests for Verlet caching via NeighborList with skin > 0."""

import math

import numpy as np
import pytest

from vesin import NeighborList


def _stateless_nl(positions, box, periodic, cutoff, full_list):
    """Compute a stateless neighbor list (skin=0) for reference."""
    nl = NeighborList(cutoff=cutoff, full_list=full_list)
    i, j, S = nl.compute(
        points=np.array(positions, dtype=np.float64),
        box=np.array(box, dtype=np.float64),
        periodic=periodic,
        quantities="ijS",
    )
    return set(zip(i.tolist(), j.tolist(), *[S[:, k].tolist() for k in range(3)]))


class TestVerletBasic:
    def test_first_call_produces_pairs(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            dtype=np.float64,
        )
        box = 10.0 * np.eye(3)

        nl = NeighborList(cutoff=3.0, full_list=True, skin=0.5)
        i, j, S = nl.compute(positions, box, periodic=True, quantities="ijS")
        assert len(i) > 0

    def test_matches_stateless(self):
        """Verlet NL must contain all pairs from stateless NL."""
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        box = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]

        ref = _stateless_nl(positions, box, True, 3.0, True)

        nl = NeighborList(cutoff=3.0, full_list=True, skin=0.5)
        i, j, S = nl.compute(
            points=np.array(positions, dtype=np.float64),
            box=np.array(box, dtype=np.float64),
            periodic=True,
            quantities="ijS",
        )
        verlet = set(
            zip(i.tolist(), j.tolist(), *[S[:, k].tolist() for k in range(3)])
        )

        for p in ref:
            assert p in verlet, f"Missing pair {p}"

    def test_half_list_matches_stateless(self):
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
        box = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]

        ref = _stateless_nl(positions, box, True, 3.0, False)

        nl = NeighborList(cutoff=3.0, full_list=False, skin=0.5)
        i, j, S = nl.compute(
            points=np.array(positions, dtype=np.float64),
            box=np.array(box, dtype=np.float64),
            periodic=True,
            quantities="ijS",
        )
        verlet = set(
            zip(i.tolist(), j.tolist(), *[S[:, k].tolist() for k in range(3)])
        )

        for p in ref:
            assert p in verlet


class TestVerletCaching:
    def test_small_movement_reuses_cache(self):
        """After small displacement, pair count should stay stable."""
        nl = NeighborList(cutoff=3.0, full_list=True, skin=1.0)
        box = 10.0 * np.eye(3)

        pos1 = np.array(
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [1.5, 1.5, 0.0]],
            dtype=np.float64,
        )
        i1, j1 = nl.compute(pos1, box, periodic=True, quantities="ij")
        n1 = len(i1)

        # Move atoms by less than skin/2 = 0.5
        pos2 = np.array(
            [[0.1, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.6, 0.0], [1.5, 1.4, 0.0]],
            dtype=np.float64,
        )
        i2, j2 = nl.compute(pos2, box, periodic=True, quantities="ij")

        # Verify correctness
        ref = _stateless_nl(pos2.tolist(), box.tolist(), True, 3.0, True)
        i2_list, j2_list = i2.tolist(), j2.tolist()
        # All reference pairs must be present
        for p in ref:
            found = False
            for k in range(len(i2_list)):
                if i2_list[k] == p[0] and j2_list[k] == p[1]:
                    found = True
                    break
            assert found, f"Missing pair {p}"

    def test_repeated_same_positions(self):
        """Repeated calls with identical positions should produce identical results."""
        nl = NeighborList(cutoff=3.0, full_list=True, skin=0.5)
        box = 10.0 * np.eye(3)
        pos = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )

        i1, j1 = nl.compute(pos, box, periodic=True, quantities="ij")
        n1 = len(i1)

        for _ in range(5):
            i2, j2 = nl.compute(pos, box, periodic=True, quantities="ij")
            assert len(i2) == n1


class TestVerletTrajectory:
    def test_md_trajectory_correctness(self):
        """Over a short trajectory, Verlet output must always be a superset
        of stateless output."""
        positions = [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5],
        ]
        box = [[5, 0, 0], [0, 5, 0], [0, 0, 5]]

        nl = NeighborList(cutoff=3.0, full_list=True, skin=0.5)

        n_steps = 15
        for step in range(n_steps):
            pos = np.array(positions, dtype=np.float64)
            box_arr = np.array(box, dtype=np.float64)

            i, j, S = nl.compute(pos, box_arr, periodic=True, quantities="ijS")
            verlet = set(
                zip(
                    i.tolist(),
                    j.tolist(),
                    *[S[:, k].tolist() for k in range(3)],
                )
            )

            ref = _stateless_nl(positions, box, True, 3.0, True)
            for p in ref:
                assert p in verlet, f"Step {step}: missing pair {p}"

            # Small perturbation
            dx = 0.03 * math.sin(step * 1.1)
            dy = 0.03 * math.sin(step * 1.3 + 1.0)
            dz = 0.03 * math.sin(step * 1.7 + 2.0)
            for ii in range(len(positions)):
                positions[ii][0] += dx * (ii + 1)
                positions[ii][1] += dy * (ii + 1)
                positions[ii][2] += dz * (ii + 1)


class TestVerletNonPeriodic:
    def test_non_periodic(self):
        positions = [
            [0.134, 1.282, 1.701],
            [-0.273, 1.026, -1.471],
            [1.922, -0.124, 1.900],
            [1.400, -0.464, 0.480],
            [0.149, 1.865, 0.635],
        ]
        box = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        ref = _stateless_nl(positions, box, False, 3.42, False)

        nl = NeighborList(cutoff=3.42, full_list=False, skin=0.5)
        i, j, S = nl.compute(
            points=np.array(positions, dtype=np.float64),
            box=np.array(box, dtype=np.float64),
            periodic=False,
            quantities="ijS",
        )
        verlet = set(
            zip(i.tolist(), j.tolist(), *[S[:, k].tolist() for k in range(3)])
        )

        assert verlet == ref
