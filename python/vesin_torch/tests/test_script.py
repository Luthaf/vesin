import copy
import pickle

import torch

from vesin_torch import NeighborList


def _system():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.6, 0.6, 0.0], [1.2, 0.0, 0.0], [1.8, 0.6, 0.0]],
        dtype=torch.float64,
    )
    box = 4.0 * torch.eye(3, dtype=torch.float64)
    return points, box


def test_pickle():
    calculator = NeighborList(
        cutoff=1.1,
        full_list=False,
        sorted=True,
        algorithm="cell_list",
        skin=0.1,
        n_threads=2,
    )
    loaded = pickle.loads(pickle.dumps(calculator))

    assert calculator.cutoff == loaded.cutoff
    assert calculator.full_list == loaded.full_list
    assert calculator.sorted == loaded.sorted
    assert calculator.algorithm == loaded.algorithm
    assert calculator.skin == loaded.skin
    assert calculator.n_threads == loaded.n_threads

    points, box = _system()
    for a, e in zip(
        calculator.compute(points, box, True, "PSdD"),
        loaded.compute(points, box, True, "PSdD"),
        strict=True,
    ):
        assert torch.equal(a, e)


def test_deepcopy():
    calculator = NeighborList(
        cutoff=1.1,
        full_list=False,
        sorted=True,
        algorithm="cell_list",
        skin=0.1,
        n_threads=2,
    )
    copied = copy.deepcopy(calculator)

    # the copy has its own state
    assert copied._state is not calculator._state

    assert calculator.cutoff == copied.cutoff
    assert calculator.full_list == copied.full_list
    assert calculator.sorted == copied.sorted
    assert calculator.algorithm == copied.algorithm
    assert calculator.skin == copied.skin
    assert calculator.n_threads == copied.n_threads

    points, box = _system()
    for a, e in zip(
        calculator.compute(points, box, True, "PSdD"),
        copied.compute(points, box, True, "PSdD"),
        strict=True,
    ):
        assert torch.equal(a, e)


def test_torch_save(tmp_path):
    calculator = NeighborList(
        cutoff=1.1,
        full_list=False,
        sorted=True,
        algorithm="cell_list",
        skin=0.1,
        n_threads=2,
    )

    path = tmp_path / "calculator.pt"
    torch.save(calculator, path)
    loaded = torch.load(path, weights_only=False)

    assert calculator.cutoff == loaded.cutoff
    assert calculator.full_list == loaded.full_list
    assert calculator.sorted == loaded.sorted
    assert calculator.algorithm == loaded.algorithm
    assert calculator.skin == loaded.skin
    assert calculator.n_threads == loaded.n_threads

    points, box = _system()
    for a, e in zip(
        calculator.compute(points, box, True, "PSdD"),
        loaded.compute(points, box, True, "PSdD"),
        strict=True,
    ):
        assert torch.equal(a, e)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calculator = NeighborList(
            cutoff=1.1,
            full_list=False,
            sorted=True,
            algorithm="cell_list",
            skin=0.1,
            n_threads=2,
        )

    def forward(self, points: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        (distances,) = self.calculator.compute(points, box, True, "d")
        return distances.sum()


def test_script_save_load(tmp_path):
    model = Model()
    scripted = torch.jit.script(model)

    # the scripted module must keep the calculator alive after the Python
    # object is garbage collected
    del model

    points, box = _system()
    expected = scripted(points, box)

    path = tmp_path / "model.pt"
    scripted.save(str(path))
    loaded = torch.jit.load(str(path))

    assert torch.equal(loaded(points, box), expected)

    points.requires_grad_(True)
    loaded(points, box).backward()
    assert points.grad is not None
