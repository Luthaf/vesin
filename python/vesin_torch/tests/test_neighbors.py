import re
from typing import List

import pytest
import torch

from vesin_torch import NeighborList


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


def test_errors():
    points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64)
    box = torch.zeros((3, 3), dtype=torch.float64)

    calculator = NeighborList(cutoff=2.8, full_list=True)

    message = "expected `points` and `box` to have the same device, got cpu and meta"
    with pytest.raises(ValueError, match=message):
        calculator.compute(
            points,
            box.to(device="meta"),
            periodic=False,
            quantities="ij",
        )

    message = "unexpected character in `quantities`: Q"
    with pytest.raises(ValueError, match=message):
        calculator.compute(
            points,
            box,
            periodic=False,
            quantities="ijQ",
        )

    message = "device meta is not supported in vesin"
    with pytest.raises(RuntimeError, match=message):
        calculator.compute(
            points.to(device="meta"),
            box.to(device="meta"),
            periodic=False,
            quantities="ij",
        )

    message = "n_threads must be zero or a positive integer"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=2.8, full_list=True, n_threads=-1)


@pytest.mark.parametrize("quantities", ["ijS", "D", "d", "ijSDd"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_alone_no_neighbors(quantities, dtype):
    points = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=dtype)
    box = torch.eye(3, dtype=dtype)

    calculator = NeighborList(cutoff=0.1, full_list=True)
    outputs = calculator.compute(points, box, True, quantities)

    if "ij" in quantities:
        assert list(outputs[quantities.index("i")].shape) == [0]
        assert list(outputs[quantities.index("j")].shape) == [0]

    if "S" in quantities:
        assert list(outputs[quantities.index("S")].shape) == [0, 3]

    if "D" in quantities:
        assert list(outputs[quantities.index("D")].shape) == [0, 3]
        assert not outputs[quantities.index("D")].requires_grad

    if "d" in quantities:
        assert list(outputs[quantities.index("d")].shape) == [0]
        assert not outputs[quantities.index("d")].requires_grad

    points.requires_grad_(True)
    box.requires_grad_(True)
    outputs = calculator.compute(points, box, True, quantities)

    if "ij" in quantities:
        assert list(outputs[quantities.index("i")].shape) == [0]
        assert list(outputs[quantities.index("j")].shape) == [0]

    if "S" in quantities:
        assert list(outputs[quantities.index("S")].shape) == [0, 3]

    if "D" in quantities:
        assert list(outputs[quantities.index("D")].shape) == [0, 3]
        assert outputs[quantities.index("D")].requires_grad

    if "d" in quantities:
        assert list(outputs[quantities.index("d")].shape) == [0]
        assert outputs[quantities.index("d")].requires_grad


class NeighborListWrap:
    def __init__(
        self,
        cutoff: float,
        full_list: bool,
        sorted: bool,
        algorithm: str,
        skin: float,
        n_threads: int,
    ):
        self._c = NeighborList(
            cutoff=cutoff,
            full_list=full_list,
            sorted=sorted,
            algorithm=algorithm,
            skin=skin,
            n_threads=n_threads,
        )

    def compute(
        self,
        points: torch.Tensor,
        box: torch.Tensor,
        periodic: bool,
        quantities: str,
        copy: bool,
    ) -> List[torch.Tensor]:
        return self._c.compute(
            points=points,
            box=box,
            periodic=periodic,
            quantities=quantities,
            copy=copy,
        )


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: NeighborListWrap) -> NeighborListWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype(dtype, device):
    box = torch.eye(3, dtype=dtype, device=device) * 3.0
    points = torch.rand((100, 3), dtype=dtype, device=device) * 3.0

    calculator = NeighborList(cutoff=1, full_list=True)
    i, j, s, D, d = calculator.compute(points, box, True, "ijSDd")

    assert i.dtype == torch.int64
    assert j.dtype == torch.int64
    assert s.dtype == torch.int32
    assert D.dtype == dtype
    assert d.dtype == dtype


def test_pad_neighbors():
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
        device="cpu",
    )
    box = torch.diag(
        torch.tensor([54.0, 54.0, 54.0], dtype=torch.float64, device="cpu")
    )

    calculator = NeighborList(cutoff=2.1, full_list=True, padded_to=3)
    i, j, S, D, d = calculator.compute(points, box, True, "ijSDd")

    assert len(i) == 18
    assert len(j) == 18

    counts = torch.bincount(i.to(torch.int64), minlength=6)
    assert torch.all(counts == 3)

    padding = j == -1
    assert int(torch.sum(padding).item()) == 10
    assert torch.all(S[padding] == 0)
    assert torch.all(D[padding] == 0)
    assert torch.all(d[padding] == 0)


@pytest.mark.parametrize("device", DEVICES)
def test_padded_neighbors_autograd(device):
    points = torch.tensor(
        [[0.1, 0.2, 0.3], [2.8, 0.1, 0.2], [1.5, 1.5, 1.5]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    box = (torch.eye(3, dtype=points.dtype, device=device) * 3).requires_grad_()
    reference = NeighborList(cutoff=0.8, full_list=True)
    padded = NeighborList(cutoff=0.8, full_list=True, padded_to=3)
    j, vectors, distances = padded.compute(points, box, True, "jDd")
    ref_vectors, ref_distances = reference.compute(points, box, True, "Dd")
    assert torch.count_nonzero(j == -1) == 7
    assert torch.count_nonzero(vectors[j == -1]) == 0
    assert torch.count_nonzero(distances[j == -1]) == 0

    def derivatives(vectors, distances):
        energy = vectors.square().sum() + distances.sum()
        forces = torch.autograd.grad(energy, (points, box), create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in forces), (points, box)
        )
        return (*forces, *second)

    for actual, expected in zip(
        derivatives(vectors, distances),
        derivatives(ref_vectors, ref_distances),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_pad_neighbors_errors():
    message = "`padded_to != None` requires `full_list = true`"
    with pytest.raises(ValueError, match=message):
        _ = NeighborList(cutoff=2.0, full_list=False, padded_to=1)

    message = "`padded_to` must be positive or None, got -1"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=1.0, full_list=True, padded_to=-1)

    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
        device="cpu",
    )
    box = torch.eye(3, dtype=torch.float64, device="cpu")

    nl = NeighborList(cutoff=1.5, full_list=True, padded_to=1)
    message = "point 0 has more neighbors than the requested `padded_to` (1)"
    with pytest.raises(RuntimeError, match=re.escape(message)):
        nl.compute(points, box, periodic=False)

    # the same calculator can still be used after an error
    i, j = nl.compute(points[:1], box, periodic=False)
    assert i.tolist() == [0]
    assert j.tolist() == [-1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("full_list", [True, False])
def test_fixed_n_pairs(device, full_list):
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
    box = torch.diag(
        torch.tensor([54.0, 54.0, 54.0], dtype=torch.float64, device=device)
    )

    calculator = NeighborList(
        cutoff=2.1, full_list=full_list, sorted=True, fixed_n_pairs=20
    )
    i, j, S, D, d = calculator.compute(points, box, True, "ijSDd")

    assert i.shape == (20,)
    assert j.shape == (20,)
    assert S.shape == (20, 3)
    assert D.shape == (20, 3)
    assert d.shape == (20,)

    # the actual pairs come first, followed by padding
    n_pairs = 8 if full_list else 4
    padding = j == -1
    assert not torch.any(padding[:n_pairs])
    assert torch.all(padding[n_pairs:])

    assert torch.all(i[padding] == -1)
    assert torch.all(S[padding] == 0)
    assert torch.all(D[padding] == 0)
    assert torch.all(d[padding] == 0)

    # actual pairs should match the calculation without padding
    reference = NeighborList(cutoff=2.1, full_list=full_list, sorted=True)
    ref_i, ref_j, ref_S, ref_D, ref_d = reference.compute(points, box, True, "ijSDd")

    # the order of pairs is not deterministic on CUDA. There are no periodic
    # images in this box, so `(i, j)` uniquely identifies a pair.
    order = torch.argsort(i[:n_pairs] * len(points) + j[:n_pairs])
    ref_order = torch.argsort(ref_i * len(points) + ref_j)

    assert torch.equal(i[:n_pairs][order], ref_i[ref_order])
    assert torch.equal(j[:n_pairs][order], ref_j[ref_order])
    assert torch.equal(S[:n_pairs][order], ref_S[ref_order])
    assert torch.equal(D[:n_pairs][order], ref_D[ref_order])
    assert torch.equal(d[:n_pairs][order], ref_d[ref_order])

    # exactly enough room for all pairs
    calculator = NeighborList(
        cutoff=2.1, full_list=full_list, sorted=True, fixed_n_pairs=n_pairs
    )
    i, j = calculator.compute(points, box, True, "ij")
    order = torch.argsort(i * len(points) + j)
    assert torch.equal(i[order], ref_i[ref_order])
    assert torch.equal(j[order], ref_j[ref_order])


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_n_pairs_autograd(device):
    points = torch.tensor(
        [[0.1, 0.2, 0.3], [2.8, 0.1, 0.2], [1.5, 1.5, 1.5]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    box = (torch.eye(3, dtype=points.dtype, device=device) * 3).requires_grad_()
    reference = NeighborList(cutoff=0.8, full_list=True)
    padded = NeighborList(cutoff=0.8, full_list=True, fixed_n_pairs=10)
    i, j, vectors, distances = padded.compute(points, box, True, "ijDd")
    ref_vectors, ref_distances = reference.compute(points, box, True, "Dd")
    assert torch.count_nonzero(i == -1) == 8
    assert torch.count_nonzero(j == -1) == 8
    assert torch.count_nonzero(vectors[j == -1]) == 0
    assert torch.count_nonzero(distances[j == -1]) == 0

    def derivatives(vectors, distances):
        energy = vectors.square().sum() + distances.sum()
        forces = torch.autograd.grad(energy, (points, box), create_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in forces), (points, box)
        )
        return (*forces, *second)

    for actual, expected in zip(
        derivatives(vectors, distances),
        derivatives(ref_vectors, ref_distances),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_fixed_n_pairs_errors():
    message = "`fixed_n_pairs` must be positive or None, got 0"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=1.0, full_list=True, fixed_n_pairs=0)

    message = "`padded_to` and `fixed_n_pairs` can not be used together"
    with pytest.raises(ValueError, match=message):
        NeighborList(cutoff=1.0, full_list=True, padded_to=3, fixed_n_pairs=10)

    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    box = torch.eye(3, dtype=torch.float64)

    nl = NeighborList(cutoff=1.5, full_list=True, fixed_n_pairs=4)
    message = (
        "the pre-allocated output arrays can hold 4 pairs, "
        "but this calculation needs 6 pairs"
    )
    with pytest.raises(RuntimeError, match=message):
        nl.compute(points, box, periodic=False)

    # the same calculator can still be used after an error
    i, j = nl.compute(points[:1], box, periodic=False)
    assert i.tolist() == [-1, -1, -1, -1]
    assert j.tolist() == [-1, -1, -1, -1]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_n_pairs(device, requires_grad):
    # each point has 2 neighbors, for 6 pairs in total
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
        device=device,
        requires_grad=requires_grad,
    )
    box = torch.eye(3, dtype=torch.float64, device=device)

    def check_n_pairs(n, expected):
        assert n.device == points.device
        assert n.shape == ()
        assert n.dtype == torch.int64
        assert n.item() == expected

    nl = NeighborList(cutoff=1.5, full_list=True)
    i, n = nl.compute(points, box, False, "in")
    assert i.shape == (6,)
    check_n_pairs(n, 6)

    nl = NeighborList(cutoff=1.5, full_list=True, fixed_n_pairs=10)
    i, n = nl.compute(points, box, False, "in")
    assert i.shape == (10,)
    check_n_pairs(n, 6)

    nl = NeighborList(cutoff=1.5, full_list=True, padded_to=4)
    i, n = nl.compute(points, box, False, "in")
    assert i.shape == (12,)
    check_n_pairs(n, 12)

    # without enough room, the calculation does not fail and only returns padding
    nl = NeighborList(cutoff=1.5, full_list=True, fixed_n_pairs=4)
    P, S, D, d, n = nl.compute(points, box, False, "PSDdn")
    check_n_pairs(n, 6)
    assert torch.all(P == -1)
    assert torch.all(S == 0)
    assert torch.all(D == 0)
    assert torch.all(d == 0)

    if requires_grad:
        # gradients of the padding are zero
        d.sum().backward()
        assert torch.all(points.grad == 0)

    # it still fails if the number of pairs is not requested
    message = (
        "the pre-allocated output arrays can hold 4 pairs, "
        "but this calculation needs 6 pairs"
    )
    with pytest.raises(RuntimeError, match=message):
        nl.compute(points, box, False, "P")

    nl = NeighborList(cutoff=1.5, full_list=True, padded_to=1)
    P, n = nl.compute(points, box, False, "Pn")
    check_n_pairs(n, 6)
    assert P.shape == (3, 2)
    assert torch.all(P == -1)

    # using `n` to get a large enough `padded_to`
    nl = NeighborList(cutoff=1.5, full_list=True, padded_to=n.item() // len(points))
    j, n = nl.compute(points, box, False, "jn")
    check_n_pairs(n, 6)
    assert torch.all(j >= 0)


def test_padded_outputs_are_independent():
    # with padding, vesin writes directly in tensors allocated for each call
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    box = torch.eye(3, dtype=torch.float64)
    nl = NeighborList(cutoff=1.5, full_list=True, sorted=True, padded_to=4)

    P_1, d_1 = nl.compute(points, box, False, "Pd", copy=False)
    expected = d_1.clone()

    P_2, d_2 = nl.compute(1.05 * points, box, False, "Pd", copy=False)
    assert P_1.data_ptr() != P_2.data_ptr()
    assert d_1.data_ptr() != d_2.data_ptr()

    assert torch.equal(d_1, expected)
    assert torch.allclose(d_2, 1.05 * expected)


# vesin-torch can only access non-default CUDA streams with torch 2.11 or later
TORCH_VERSION = tuple(int(v) for v in torch.__version__.split(".")[:2])
TORCH_HAS_CUDA_STREAMS = TORCH_VERSION >= (2, 11) and torch.cuda.is_available()


def random_system(device, scale=1.0):
    generator = torch.Generator().manual_seed(0xDEADBEEF)
    points = 5.0 * torch.rand((100, 3), dtype=torch.float64, generator=generator)
    box = 5.0 * torch.eye(3, dtype=torch.float64)
    return scale * points.to(device), box.to(device)


def sort_pairs(i, j, S):
    return sorted(zip(i.tolist(), j.tolist(), map(tuple, S.tolist()), strict=True))


def delayed_copy(points):
    """
    Copy ``points`` into a new zero-initialized tensor, after making the current
    stream wait for a while. Any code running on another stream will see the zeros
    instead of the points.
    """
    lazy_points = torch.zeros_like(points)
    torch.cuda.current_stream(points.device).wait_stream(
        torch.cuda.default_stream(points.device)
    )

    # This should be long enough for the kernels to run on the wrong stream before the
    # data is ready, if it ignores the current stream.
    torch.cuda._sleep(200_000_000)
    lazy_points.copy_(points)
    return lazy_points


@pytest.mark.skipif(not TORCH_HAS_CUDA_STREAMS, reason="CUDA streams not available")
@pytest.mark.parametrize("algorithm", ["brute_force", "cell_list"])
def test_cuda_stream(algorithm):
    points, box = random_system("cuda")
    calculator = NeighborList(cutoff=2.0, full_list=True, algorithm=algorithm)
    expected = sort_pairs(*calculator.compute(points, box, True, "ijS"))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        lazy_points = delayed_copy(points)
        i, j, S = calculator.compute(lazy_points, box, True, "ijS")
        # use the outputs on the same stream, without explicit synchronization
        actual = sort_pairs(i, j, S)

    assert actual == expected


@pytest.mark.skipif(not TORCH_HAS_CUDA_STREAMS, reason="CUDA streams not available")
@pytest.mark.parametrize("skin", [0.0, 0.5])
def test_cuda_multiple_streams(skin):
    reference = NeighborList(cutoff=2.0, full_list=True)
    calculator = NeighborList(cutoff=2.0, full_list=True, skin=skin)

    # use the same calculator on different streams, one after the other. The
    # points move a bit between calls, to check the Verlet cache when skin > 0.
    for step in range(4):
        points, box = random_system("cuda", scale=1.0 + 0.02 * step)
        expected = sort_pairs(*reference.compute(points, box, True, "ijS"))

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            lazy_points = delayed_copy(points)
            actual = sort_pairs(*calculator.compute(lazy_points, box, True, "ijS"))

        assert actual == expected

    # back on the default stream
    points, box = random_system("cuda")
    expected = sort_pairs(*reference.compute(points, box, True, "ijS"))
    actual = sort_pairs(*calculator.compute(points, box, True, "ijS"))
    assert actual == expected
