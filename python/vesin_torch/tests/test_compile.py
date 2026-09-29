import re

import pytest
import torch
from torch.export import Dim

from vesin_torch import NeighborList


TORCH_VERSION = tuple(int(v) for v in torch.__version__.split(".")[:2])

pytestmark = pytest.mark.skipif(
    TORCH_VERSION < (2, 5), reason="torch.compile support requires torch 2.5"
)


@pytest.mark.parametrize("backend", ["eager", "inductor", "aot_eager"])
@pytest.mark.parametrize("full_list", [True, False])
def test_compile_forward_no_graph_break(full_list, backend):
    calculator = NeighborList(cutoff=1.1, full_list=full_list)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calculator = calculator

        def forward(self, points, box):
            (distances,) = self.calculator.compute(
                points=points,
                box=box,
                periodic=torch.tensor(True),
                quantities="d",
            )
            return distances.sum()

    model = Model()
    compiled = torch.compile(model, backend=backend, fullgraph=True)

    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.6, 0.6, 0.0], [1.2, 0.0, 0.0], [1.8, 0.6, 0.0]],
        dtype=torch.float64,
    )
    box = 4.0 * torch.eye(3, dtype=torch.float64)
    expected = model(points, box)
    actual = compiled(points, box)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("backend", ["eager", "inductor", "aot_eager"])
def test_compile_forward_padded(backend):
    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=3)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calculator = calculator

        def forward(self, points, box):
            (i, j, distances) = self.calculator.compute(
                points=points,
                box=box,
                periodic=torch.tensor(True),
                quantities="ijd",
            )
            return i, j, distances.sum()

    model = Model()
    compiled = torch.compile(model, backend=backend, fullgraph=True)

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
    )
    box = torch.diag(torch.tensor([54.0, 54.0, 54.0], dtype=torch.float64))

    expected_i, expected_j, expected = model(points, box)
    actual_i, actual_j, actual = compiled(points, box)

    assert torch.allclose(actual_i, expected_i)
    assert torch.allclose(actual_j, expected_j)
    assert torch.allclose(actual, expected)

    # every point is padded to exactly `padded_to` outgoing pairs; the padding
    # pairs are `(i, -1)` pseudo-pairs with zero distance
    assert len(actual_i) == 18
    counts = torch.bincount(actual_i.to(torch.int64), minlength=6)
    assert torch.all(counts == 3)

    padding = actual_j == -1
    assert int(torch.sum(padding).item()) == 10
    assert torch.all(actual_j[padding] == -1)


@pytest.mark.parametrize("backend", ["eager", "inductor", "aot_eager"])
@pytest.mark.parametrize("full_list", [True, False])
def test_compile_backward_outside_model(full_list, backend):
    calculator = NeighborList(cutoff=1.1, full_list=full_list)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calculator = calculator

        def forward(self, points, box):
            (distances,) = self.calculator.compute(
                points=points,
                box=box,
                periodic=torch.tensor(True),
                quantities="d",
            )
            return distances.sum()

    model = Model()
    compiled = torch.compile(model, backend=backend, fullgraph=True)

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.6, 0.0],
            [1.2, 0.0, 0.0],
            [1.8, 0.6, 0.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    box = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    expected = model(points, box)
    expected.backward()
    expected_points_grad = points.grad.clone()
    expected_box_grad = box.grad.clone()

    points.grad = None
    box.grad = None
    assert points.requires_grad
    assert box.requires_grad

    actual = compiled(points, box)
    actual.backward()

    assert torch.allclose(actual, expected.detach())
    assert torch.allclose(points.grad, expected_points_grad)
    assert torch.allclose(box.grad, expected_box_grad)


@pytest.mark.skipif(
    not hasattr(torch._dynamo.config, "trace_autograd_ops"),
    reason="tracing torch.autograd.grad with torch.compile requires trace_autograd_ops",
)
@pytest.mark.parametrize("backend", ["eager", "inductor", "aot_eager"])
@pytest.mark.parametrize("full_list", [True, False])
def test_compile_backward_inside_model(full_list, backend):
    calculator = NeighborList(cutoff=1.1, full_list=full_list)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calculator = calculator

        def forward(self, points, box):
            points = points.detach().requires_grad_(True)
            box = box.detach().requires_grad_(True)

            (distances,) = self.calculator.compute(
                points=points,
                box=box,
                periodic=torch.tensor(True),
                quantities="d",
            )
            energy = distances.sum()

            points_grad, box_grad = torch.autograd.grad(energy, [points, box])
            return energy.detach(), points_grad, box_grad

    model = Model()

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.6, 0.0],
            [1.2, 0.0, 0.0],
            [1.8, 0.6, 0.0],
        ],
        dtype=torch.float64,
    )
    box = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float64,
    )

    expected = model(points, box)

    # torch.compile needs this option to include `torch.autograd.grad` in the graph
    with torch._dynamo.config.patch(trace_autograd_ops=True):
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        actual = compiled(points, box)

    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert torch.allclose(actual_value, expected_value)


def _system(n_points):
    indices = torch.arange(n_points, dtype=torch.float64) * 0.6
    points = torch.stack([indices, torch.zeros(n_points), torch.zeros(n_points)], dim=1)
    box = 100.0 * torch.eye(3, dtype=torch.float64)
    return points, box


@pytest.mark.parametrize("padded_to", [None, 3])
def test_export_dynamic_shapes(padded_to):
    """
    Check that when using a padded neighbor list, the output shapes are correctly
    marked as padded_to * n_points, and not left as a dynamic value.
    """
    calculator = NeighborList(cutoff=1.1, full_list=True, padded_to=padded_to)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calculator = calculator

        def forward(self, points, box):
            (i, j, distances) = self.calculator.compute(
                points=points,
                box=box,
                periodic=torch.tensor(True),
                quantities="ijd",
            )
            return i, j, distances

    model = Model()

    points, box = _system(4)
    exported = torch.export.export(
        model,
        (points, box),
        dynamic_shapes=({0: Dim("n_points")}, None),
    )
    graph = exported.module()

    pairs_shape = None
    for node in exported.graph.nodes:
        if node.op == "call_function" and "vesin.neighbors" in str(node.target):
            pairs_shape = node.meta["val"][0].shape[0]
    assert pairs_shape is not None

    if padded_to is None:
        # the number of pairs is data-dependent, the fake implementation marks
        # it as a dynamic size unrelated to the inputs (unbacked symint)
        assert re.fullmatch(r"u\d+", str(pairs_shape))
    else:
        # padding makes the output size input-dependent: exactly `padded_to *
        # n_points` pairs, expressed in terms of the `points` dynamic size
        # (backed symint)
        assert re.fullmatch(rf"{padded_to}\*s\d+", str(pairs_shape))

    for n_points in [4, 8]:
        points, box = _system(n_points)
        expected_i, expected_j, expected_distances = model(points, box)
        actual_i, actual_j, actual_distances = graph(points, box)

        assert torch.equal(actual_i, expected_i)
        assert torch.equal(actual_j, expected_j)
        assert torch.equal(actual_distances, expected_distances)

        if padded_to is not None:
            assert actual_i.shape == (padded_to * n_points,)
