import pytest
import torch

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
