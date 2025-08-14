import torch

from vesin.torch import NeighborList


def test_large_box_small_cutoff(device, full_list):
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [-6.0, 0.0, 0.0],
            [-6.0, -2.0, 0.0],
            [-6.0, 0.0, -2.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    box = torch.tensor(
        [
            [54.0, 0.0, 0.0],
            [0.0, 54.0, 0.0],
            [0.0, 0.0, 54.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    calculator = NeighborList(cutoff=2.1, full_list=full_list)

    quantities = "ijdS"  # i,j for indices, d for distances
    i, j, dists, shifts = calculator.compute(
        points, box, periodic=True, quantities=quantities
    )

    print(dists)

    pairs = torch.stack((i, j), dim=1)

    sort_idx = torch.argsort(pairs[:, 0] * (i.max() + 1) + pairs[:, 1])

    # Apply sort
    i = i[sort_idx]
    j = j[sort_idx]
    dists = dists[sort_idx]

    # Convert to plain Python lists for easy matching
    actual_pairs = sorted(zip(i.tolist(), j.tolist()))
    actual_dists = [d.item() for d in dists]

    if full_list:
        expected_pairs = sorted(
            [
                (0, 1),
                (0, 2),
                (1, 0),
                (2, 0),
                (3, 4),
                (3, 5),
                (4, 3),
                (5, 3),
            ]
        )
        expected_dists = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    else:
        expected_pairs = sorted(
            [
                (0, 1),
                (0, 2),
                (3, 4),
                (3, 5),
            ]
        )
        expected_dists = [2.0, 2.0, 2.0, 2.0]
    # Check pairs
    assert actual_pairs == expected_pairs, (
        f"Expected pairs {expected_pairs}, got {actual_pairs}"
    )

    # Check distances approximately
    for actual, expected in zip(actual_dists, expected_dists):
        assert abs(actual - expected) < 1e-8, (
            f"Expected distance {expected}, got {actual}"
        )


test_large_box_small_cutoff(device="cuda", full_list=False)
