"""Independent numerical oracles, with executable deliberate-fault witnesses.

Adaptive rectangular precision is in CDF index space, not physical space.
Delaunay/Sibson have linear precision inside the hull; KNNBarycentric only
inside its selected triangle. Wendland KNN has neither global linear precision
nor interpolatory node values. Neighbor-switch continuity is not asserted for
the KNN families. No magnification/area implementation is used here.
"""

import importlib.util
import os
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import Delaunay
from scipy.optimize import brentq
from scipy.special import ndtr

import autoarray as aa
from autoarray.inversion.mesh.interpolator import rectangular as adaptive
from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
    InterpolatorRectangularUniform,
)

FAMILIES = [
    "uniform",
    "adaptive",
    "adaptive_kernel",
    "delaunay",
    "sibson",
    "knn",
    "knn_barycentric",
]


def record_evidence(message, *args):
    if os.environ.get("PYAUTO_NUMERICS_AUDIT_EVIDENCE") == "1":
        print(message % args)


@contextmanager
def fault_detected():
    """Keep fault witnesses silent normally; opt-in evidence is captured with -s."""
    with pytest.raises(AssertionError) as failure:
        yield
    record_evidence("Deliberate fault rejected: %s", failure.value)


class Grid:
    def __init__(self, values):
        self.array = np.asarray(values)
        self.over_sampled = self


def lattice(n):
    axis = np.linspace(-1, 1, n)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([y.ravel(), x.ravel()])


def tables(family, nodes, query, n):
    if family in ("knn", "knn_barycentric"):
        if importlib.util.find_spec("jax") is None:
            pytest.skip("production KNN implementation requires optional jax")
    if family.startswith("adaptive"):
        mesh_cls = (
            aa.mesh.RectangularRTUAdaptDensity
            if family == "adaptive_kernel"
            else aa.mesh.RectangularBilinearAdaptDensity
        )
        mesh = mesh_cls(shape=(n, n), respect_small_datasets=False)
        data_grid = Grid(nodes)
        data_grid.over_sampled = Grid(query)
        interpolator = mesh.interpolator_cls(
            mesh=mesh,
            mesh_grid=Grid(nodes),
            data_grid=data_grid,
            mesh_weight_map=None,
            **mesh.interpolator_kwargs,
        )
    elif family == "uniform":
        interpolator = InterpolatorRectangularUniform(
            mesh=SimpleNamespace(shape=(n, n)),
            mesh_grid=Grid(nodes),
            data_grid=Grid(query),
        )
    else:
        mesh_name = {
            "delaunay": "Delaunay",
            "sibson": "DelaunayNN",
            "knn": "KNearestNeighbor",
            "knn_barycentric": "KNNBarycentric",
        }[family]
        mesh = getattr(aa.mesh, mesh_name)(pixels=len(nodes))
        interpolator = mesh.interpolator_cls(
            mesh=mesh, mesh_grid=Grid(nodes), data_grid=Grid(query)
        )
    mappings, _, weights = interpolator._mappings_sizes_weights
    return np.asarray(mappings), np.asarray(weights)


def evaluate(mappings, weights, values):
    return np.sum(values[mappings.clip(min=0)] * weights, axis=1)


def assert_precision(family, nodes, query, n, corrupt=False):
    mappings, weights = tables(family, nodes, query, n)
    assert np.isfinite(weights).all()
    assert ((mappings >= -1) & (mappings < len(nodes))).all()
    if corrupt:
        # Preserve partition of unity while corrupting corner/weight pairing.
        weights = np.roll(weights, 1, axis=1)
    if family.startswith("adaptive"):
        mu, scale = nodes.mean(0), nodes.std(0).min()
        forward, _ = (
            adaptive.create_transforms((nodes - mu) / scale, mesh_pixels=n, xp=np)
            if family == "adaptive_kernel"
            else adaptive.create_transforms_rank((nodes - mu) / scale, xp=np)
        )
        expected = (n - 3) * forward((query - mu) / scale) + 1
        coordinate = np.column_stack([n - np.arange(n * n) // n, np.arange(n * n) % n])
        # Guards can appear in a zero-weight stencil but cannot contribute.
        live = weights != 0
        row, col = n - mappings // n, mappings % n
        flat_row = mappings // n
        assert ((flat_row[live] >= 2) & (flat_row[live] <= n - 1)).all()
        assert ((row[live] >= 1) & (row[live] <= n - 2)).all()
        assert ((col[live] >= 1) & (col[live] <= n - 2)).all()
    else:
        expected, coordinate = query, nodes
    reconstructed = np.column_stack(
        [evaluate(mappings, weights, coordinate[:, d]) for d in (0, 1)]
    )
    np.testing.assert_allclose(reconstructed, expected, atol=2e-10, rtol=0)


@pytest.mark.parametrize("family", FAMILIES[:-2] + ["knn_barycentric"])
def test_linear_precision_and_pairing_fault(family):
    # A triangle containing all queries makes KNNBarycentric's applicability explicit.
    if family == "knn_barycentric":
        nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
        query = np.array([[-0.7, -0.6], [-0.2, -0.5], [-0.4, -0.1]])
        n = 3
    else:
        n, nodes = 9, lattice(9)
        query = np.random.default_rng(8).uniform(-0.75, 0.75, (31, 2))
        if family.startswith("adaptive"):
            query = np.vstack(
                [query, [[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]]]
            )
    assert_precision(family, nodes, query, n)
    with fault_detected():
        assert_precision(family, nodes, query, n, corrupt=True)


@pytest.mark.parametrize(
    "family,axis",
    [
        (family, axis)
        for family in ("uniform", "adaptive", "adaptive_kernel")
        for axis in (0, 1)
    ]
    + [
        ("delaunay", None),
        pytest.param(
            "sibson",
            None,
            marks=pytest.mark.xfail(
                strict=True,
                reason="PyAutoArray#610: Sibson interior-edge fallback is discontinuous",
            ),
        ),
    ],
)
def test_exact_nextafter_boundary_continuity_and_locator_fault(
    family, axis, monkeypatch
):
    n, nodes = 9, lattice(9)
    if family.startswith("adaptive"):
        # Isolate integer discretization from the approximate CDF inverse.
        monkeypatch.setattr(
            adaptive, "_transforms_from", lambda *args, **kw: (lambda q: q, lambda q: q)
        )
        index = np.full((n - 4, 2), 3.37)
        index[:, axis] = np.arange(2, n - 2)
        query = nodes.std(0).min() * (index - 1) / (n - 3) + nodes.mean(0)
        normal = np.zeros_like(query)
        normal[:, axis] = 1
    elif family in ("delaunay", "sibson"):
        tri = Delaunay(nodes)
        edges = sorted(
            {
                tuple(sorted((a, b)))
                for simplex in tri.simplices
                for a, b in zip(simplex, np.roll(simplex, 1))
            }
        )
        edges = [
            (a, b) for a, b in edges if (np.abs((nodes[a] + nodes[b]) / 2) < 0.9).all()
        ]
        query = np.array([(nodes[a] + nodes[b]) / 2 for a, b in edges])
        tangent = np.array([nodes[b] - nodes[a] for a, b in edges])
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
    else:
        query = np.full((7, 2), 0.137)
        query[:, axis] = np.linspace(-0.75, 0.75, 7)
        normal = np.zeros_like(query)
        normal[:, axis] = 1
    # Edge-normal movement actually crosses diagonal simplex boundaries.
    # Finite steps complement ULP steps (qhull deliberately tolerates ULPs).
    probes = np.stack(
        [
            query - 1e-8 * normal,
            np.nextafter(query, query - normal),
            query,
            np.nextafter(query, query + normal),
            query + 1e-8 * normal,
        ],
        axis=1,
    ).reshape(-1, 2)
    mappings, weights = tables(family, nodes, probes, n)
    assert np.isfinite(weights).all()
    assert ((mappings >= -1) & (mappings < n * n)).all()
    values = np.sin(np.arange(n * n) * 0.71)
    result = evaluate(mappings, weights, values).reshape(-1, 5)
    broken = mappings.copy()
    broken[3::5] = (broken[3::5] + n) % (n * n)  # incorrect one-row cell locator
    with fault_detected():
        np.testing.assert_allclose(
            evaluate(broken, weights, values).reshape(-1, 5)[:, 3],
            result[:, 2],
            atol=2e-10,
            rtol=0,
        )
    for column in (1, 3):
        np.testing.assert_allclose(result[:, column], result[:, 2], atol=2e-10, rtol=0)
    for column in (0, 4):
        # Mesh spacing .25 and nodal range2 bound a finite-step change.
        np.testing.assert_allclose(result[:, column], result[:, 2], atol=2e-6, rtol=0)


def smooth(q):
    return np.sin(0.8 * q[:, 0]) + np.cos(1.1 * q[:, 1]) + 0.2 * q[:, 0] * q[:, 1]


@pytest.mark.parametrize("family", FAMILIES)
def test_supported_region_refinement_and_frozen_mesh_fault(family):
    # Queries are concentrated in the central data-supported region.
    query = np.random.default_rng(19).uniform(-0.55, 0.55, (41, 2))

    def error_at(n):
        nodes = lattice(n)
        if family.startswith("adaptive"):
            axis = np.linspace(-1, 1, 1600)
            nodes = np.column_stack(
                [axis, axis[np.random.default_rng(70).permutation(len(axis))]]
            )
        mappings, weights = tables(family, nodes, query, n)
        if family.startswith("adaptive"):
            # Independent inverse of the fixed, untied empirical rank data.
            index = np.arange(n * n)
            u = (np.column_stack([n - index // n, index % n]) - 1) / (n - 3)
            ranks = np.arange(1, len(nodes) + 1) / (len(nodes) + 1)
            physical = np.column_stack(
                [np.interp(u[:, d], ranks, np.sort(nodes[:, d])) for d in (0, 1)]
            )
            if family == "adaptive_kernel":
                for d in (0, 1):
                    lo, hi = nodes[:, d].min(), nodes[:, d].max()
                    h = (hi - lo) / n
                    raw = lambda z: ndtr((z - nodes[:, d]) / h).mean()
                    a, b = raw(lo), raw(hi)
                    for unit in np.unique(u[:, d]):
                        clipped = np.clip(unit, 0, 1)
                        value = (
                            lo
                            if clipped == 0
                            else (
                                hi
                                if clipped == 1
                                else brentq(
                                    lambda z: (raw(z) - a) / (b - a) - clipped, lo, hi
                                )
                            )
                        )
                        physical[u[:, d] == unit, d] = value
        else:
            physical = nodes
        return np.sqrt(
            np.mean(
                (evaluate(mappings, weights, smooth(physical)) - smooth(query)) ** 2
            )
        )

    errors = [error_at(n) for n in (8, 16, 32)]
    record_evidence("Supported-region RMS %s: %s", family, errors)
    # KNNBarycentric may clip a non-enclosing triangle: require convergence,
    # not a fictitious global second-order rate. Likewise kernel KNN.
    ratio = 0.85 if family.startswith("knn") else 0.6
    assert errors[1] < ratio * errors[0], errors
    assert errors[2] < ratio * errors[1], errors
    # Inject a mesh-builder fault: every requested refinement reuses n=8.
    frozen = [error_at(8) for requested_n in (8, 16, 32)]
    with fault_detected():
        assert frozen[1] < ratio * frozen[0]


@pytest.mark.parametrize("n", [16, 32, 64])
def test_cdf_knot_refinement_and_inverse_discretization_fault(n):
    points = np.random.default_rng(22).normal(size=(200, 2))
    u = np.linspace(0.04, 0.96, n)[:, None] * np.ones((1, 2))
    errors = []
    for knots in (None, 256, 1024):
        options = {} if knots is None else {"n_knots": knots}
        forward, inverse = adaptive.create_transforms(
            points, mesh_pixels=n, xp=np, **options
        )
        errors.append(np.max(np.abs(forward(inverse(u)) - u)) * (n - 3))
    record_evidence("CDF index roundtrip n=%s knots=64/256/1024: %s", n, errors)
    assert errors[1] < errors[0] / 4, errors
    assert errors[2] < errors[1] / 4, errors
    with fault_detected():
        # Quantize inverse queries to the default table: larger tables cannot
        # repair a discretization that throws away their extra resolution.
        quantized = np.round(u * 63) / 63
        bad = np.max(np.abs(forward(inverse(quantized)) - u)) * (n - 3)
        assert bad < errors[1] / 4


def test_uniform_control_bit_identical_under_adaptive_pairing_fault(monkeypatch):
    nodes, query = lattice(8), np.random.default_rng(41).uniform(-0.6, 0.6, (23, 2))
    before = tables("uniform", nodes, query, 8)
    original = adaptive.adaptive_rectangular_mappings_weights_via_interpolation_from

    def mirrored(*args, **kwargs):
        m, w = original(*args, **kwargs)
        return m, w[:, [2, 3, 0, 1]]

    monkeypatch.setattr(
        adaptive,
        "adaptive_rectangular_mappings_weights_via_interpolation_from",
        mirrored,
    )
    with fault_detected():
        assert_precision("adaptive", nodes, query, 8)
    after = tables("uniform", nodes, query, 8)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(before, after))
    with fault_detected():
        assert before[1].tobytes() == after[1][:, [2, 3, 0, 1]].tobytes()


@pytest.mark.xfail(
    strict=True,
    reason="PyAutoArray#609: partial final KNN block omits/mislabels tail nodes",
)
def test_knn_partial_final_block_matches_independent_neighbor_oracle():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("production KNN implementation requires optional jax")
    from autoarray.inversion.mesh.interpolator.knn import get_interpolation_weights

    nodes = np.column_stack([np.arange(130, dtype=float), np.zeros(130)])
    query = nodes[-2:]
    squared = ((query[:, None] - nodes[None]) ** 2).sum(2)
    # Deliberately truncate the point set to the first full block: exact tail
    # nodes disappear even though the nearest-distance oracle is unchanged.
    truncated_nearest = np.argmin(squared[:, :128], axis=1)
    with fault_detected():
        np.testing.assert_array_equal(truncated_nearest, [128, 129])
    mappings, _, distance = get_interpolation_weights(nodes, query, 3, 1.5)
    # Tie-independent assertion: nearest distance is zero and is paired with
    # the exact queried tail vertex (128 and 129).
    np.testing.assert_array_equal(np.asarray(mappings)[:, 0], [128, 129])
    np.testing.assert_allclose(
        np.asarray(distance),
        np.sqrt(np.take_along_axis(squared, np.asarray(mappings), axis=1)),
        atol=1e-9,
        rtol=0,
    )


def test_knn_full_block_neighbors_kernel_and_pairing_fault():
    nodes = lattice(8)
    query = np.random.default_rng(81).uniform(-0.7, 0.7, (17, 2))
    mappings, weights = tables("knn", nodes, query, 8)
    distance = np.sqrt(((query[:, None] - nodes[None]) ** 2).sum(2))
    expected = np.argsort(distance, axis=1)[:, :10]
    np.testing.assert_array_equal(mappings, expected)
    d = np.take_along_axis(distance, expected, axis=1)
    r = np.sqrt(d * d + 1e-20) / (d[:, -1, None] * 1.5 + 1e-10)
    raw = np.where(r < 1, (1 - r) ** 6 * (35 * r * r + 18 * r + 3), 0)
    oracle = raw / (raw.sum(1, keepdims=True) + 1e-10)
    np.testing.assert_allclose(weights, oracle, atol=1e-11, rtol=0)
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-9, rtol=0)
    with fault_detected():
        np.testing.assert_allclose(
            np.roll(weights, 1, axis=1), oracle, atol=1e-11, rtol=0
        )


@pytest.mark.xfail(
    strict=True,
    reason="PyAutoArray#610: exact interior edge uses triangle weights instead of Sibson weights",
)
def test_sibson_square_center_matches_independent_symmetry_oracle():
    nodes = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    values = nodes[:, 0] * nodes[:, 1]
    # Relevant broken variant: substitute the two diagonal endpoints for the
    # four equal natural neighbors. Unity and coordinate precision both hold.
    diagonal_weights = np.array([0.5, 0.0, 0.0, 0.5])
    with fault_detected():
        np.testing.assert_allclose(diagonal_weights @ values, 0.0, atol=1e-12, rtol=0)
    mappings, weights = tables("sibson", nodes, np.array([[0.0, 0.0]]), 2)
    effective = np.zeros(4)
    np.add.at(effective, mappings[0].clip(min=0), weights[0])
    # Fourfold square symmetry fixes every weight independently to one fourth.
    np.testing.assert_allclose(effective, 0.25, atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        evaluate(mappings, weights, values), 0.0, atol=1e-12, rtol=0
    )


@pytest.mark.xfail(
    strict=True,
    reason="PyAutoArray#610: near-edge cancellation clips negative weights after normalization",
)
def test_sibson_near_edge_preserves_partition_after_positive_filter():
    # A normalize-then-clip mutation reproduces the defect mechanism while
    # leaving the intended constant-field oracle unchanged.
    signed = np.array([-0.1, 0.55, 0.55])
    normalized = signed / signed.sum()
    broken = np.where(normalized > 0, normalized, 0)
    with fault_detected():
        np.testing.assert_allclose(broken.sum(), 1.0, atol=1e-12, rtol=0)
    nodes = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    query = np.array([[-1e-8, 0.0], [1e-8, 0.0]])
    mappings, weights = tables("sibson", nodes, query, 2)
    np.testing.assert_allclose(weights.sum(1), 1.0, atol=1e-12, rtol=0)
    reconstructed = np.column_stack(
        [evaluate(mappings, weights, nodes[:, d]) for d in (0, 1)]
    )
    np.testing.assert_allclose(reconstructed, query, atol=1e-12, rtol=0)


def dense_kernel_cdf(points, query, n):
    """Independent defining mixture, no production forward/inverse functions."""
    lo, hi = points.min(0), points.max(0)
    h = (hi - lo) / n
    raw = lambda q: ndtr((q[:, None, :] - points[None, :, :]) / h).mean(1)
    a, b = raw(lo[None, :]), raw(hi[None, :])
    return np.clip((raw(query) - a) / (b - a), 0, 1)


@pytest.mark.parametrize("axis", [0, 1])
def test_actual_kernel_clip_plateaus_exact_nextafter_and_cell_fault(axis):
    n, nodes = 16, lattice(9)
    centers = np.full((2, 2), 0.137)
    centers[:, axis] = [-1.0, 1.0]
    direction = np.zeros_like(centers)
    direction[:, axis] = 1
    probes = np.stack(
        [
            centers - 1e-9 * direction,
            np.nextafter(centers, centers - direction),
            centers,
            np.nextafter(centers, centers + direction),
            centers + 1e-9 * direction,
        ],
        axis=1,
    ).reshape(-1, 2)
    mappings, weights = tables("adaptive_kernel", nodes, probes, n)
    coordinate = np.column_stack([n - np.arange(n * n) // n, np.arange(n * n) % n])
    expected = (n - 3) * dense_kernel_cdf(nodes, probes, n) + 1
    actual = np.column_stack(
        [evaluate(mappings, weights, coordinate[:, d]) for d in (0, 1)]
    )
    np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)
    np.testing.assert_allclose(actual[2::5, axis], [1.0, n - 2.0], atol=2e-10, rtol=0)
    for side in (0, 1):
        sample = actual.reshape(2, 5, 2)[side]
        np.testing.assert_allclose(
            sample, np.broadcast_to(sample[2], sample.shape), atol=2e-7, rtol=0
        )
    # A plateau-only one-row cell assignment corruption is invisible to a
    # generic random-query precision test; target the exact saturated rows.
    broken = mappings.copy()
    broken[2::5] = (broken[2::5] + n) % (n * n)
    with fault_detected():
        reconstructed = np.column_stack(
            [evaluate(broken, weights, coordinate[:, d]) for d in (0, 1)]
        )
        np.testing.assert_allclose(reconstructed, expected, atol=2e-10, rtol=0)


def test_adaptive_geometry_guard_nodes_and_edge_order_fault():
    n, nodes = 16, lattice(9)
    mesh = aa.mesh.RectangularRTUAdaptDensity(
        shape=(n, n), respect_small_datasets=False
    )
    interp = mesh.interpolator_cls(
        mesh=mesh,
        mesh_grid=Grid(lattice(n)),
        data_grid=Grid(nodes),
        mesh_weight_map=None,
        **mesh.interpolator_kwargs,
    )
    mappings, _, weights = interp._mappings_sizes_weights
    active = np.unique(mappings[weights > 0])
    assert np.array_equal(np.unique(active // n), np.arange(2, n))
    assert np.array_equal(np.unique(active % n), np.arange(1, n - 1))
    # Zeroed perimeter pixels are a boundary condition, not the inactive guard
    # set: positive support overlaps the last row only. Preserve that fact
    # without claiming the intended inversion boundary condition is correct.
    overlap = np.intersect1d(active, mesh.zeroed_pixels)
    assert len(overlap) > 0
    np.testing.assert_array_equal(overlap // n, np.full(len(overlap), n - 1))
    rows = np.arange(n + 1)
    edge_u = np.column_stack([(n - rows - 0.5) / (n - 3), (rows - 1.5) / (n - 3)])
    knots = np.column_stack([np.linspace(-1, 1, mesh.n_knots)] * 2)
    knot_cdf = dense_kernel_cdf(nodes, knots, n)
    expected = np.column_stack(
        [np.interp(edge_u[:, d], knot_cdf[:, d], knots[:, d]) for d in (0, 1)]
    )
    edges = np.asarray(interp.mesh_geometry.edges_transformed)
    np.testing.assert_allclose(edges, expected, atol=2e-12, rtol=0)
    with fault_detected():
        # Inject a one-row ordering error in the returned geometry edges.
        np.testing.assert_allclose(
            np.roll(edges, 1, axis=0), expected, atol=2e-12, rtol=0
        )
