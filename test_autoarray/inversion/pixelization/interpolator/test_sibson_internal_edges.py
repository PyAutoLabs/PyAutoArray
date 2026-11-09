"""
PyAutoArray#610: Sibson (natural-neighbour) weights at and near internal Delaunay edges.

The weights are computed from the Bowyer-Watson cavity boundary, so a query on or near an internal edge is
regular. These tests check the three production bodies (the NumPy reference, numba and JAX) against an
independent oracle that builds the Voronoi cells by half-plane clipping and shares no code with production.
"""

import importlib.util

import numpy as np
import pytest
from scipy.spatial import Delaunay

from autoarray.inversion.mesh.interpolator import sibson
from autoarray.inversion.mesh.interpolator import _sibson_numba

requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None, reason="JAX path requires jax"
)


# ---------------------------------------------------------------------------------------------------------------
# Independent oracle: Voronoi cells by half-plane clipping (Sutherland-Hodgman); no production code.
# ---------------------------------------------------------------------------------------------------------------


def _clip(polygon, through, normal):
    kept = []
    for i in range(len(polygon)):
        p, q = polygon[i], polygon[(i + 1) % len(polygon)]
        dp, dq = normal @ (p - through), normal @ (q - through)
        if dp <= 0:
            kept.append(p)
        if dp * dq < 0:
            kept.append(p + (q - p) * (dp / (dp - dq)))
    return kept


def _area(polygon):
    if len(polygon) < 3:
        return 0.0
    c = polygon[0]
    return 0.5 * sum(
        (polygon[i] - c)[0] * (polygon[i + 1] - c)[1]
        - (polygon[i] - c)[1] * (polygon[i + 1] - c)[0]
        for i in range(1, len(polygon) - 1)
    )


def oracle_weights(points, query, box=1.0e4):
    """Sibson weights: area of (new cell of query) intersected with (old cell of each vertex), normalised."""
    corners = [(-box, -box), (box, -box), (box, box), (-box, box)]
    new_cell = [np.array(c, dtype=float) for c in corners]
    for p in points:
        new_cell = _clip(new_cell, (query + p) / 2, p - query)
    areas = {}
    for i, v in enumerate(points):
        cell = new_cell
        for k, p in enumerate(points):
            if k != i:
                cell = _clip(cell, (v + p) / 2, p - v)
            if len(cell) < 3:
                break
        area = _area(cell)
        if area > 0:
            areas[i] = area
    total = sum(areas.values())
    return {k: a / total for k, a in areas.items()}


# ---------------------------------------------------------------------------------------------------------------
# Production backends on explicit tables.
# ---------------------------------------------------------------------------------------------------------------


def _neighbors_from(simplices):
    """qhull convention: neighbors[t, j] is the triangle across the edge opposite vertex j (-1 on the hull)."""
    edge_owner = {}
    for t, s in enumerate(simplices):
        for j in range(3):
            edge = frozenset((int(s[(j + 1) % 3]), int(s[(j + 2) % 3])))
            edge_owner.setdefault(edge, []).append((t, j))
    neighbors = -np.ones_like(simplices)
    for owners in edge_owner.values():
        if len(owners) == 2:
            (t0, j0), (t1, j1) = owners
            neighbors[t0, j0], neighbors[t1, j1] = t1, t0
    return neighbors


def _locate(points, simplices, query):
    for t, s in enumerate(simplices):
        a, b, c = points[s]
        m = np.array([b - a, c - a]).T
        u, v = np.linalg.solve(m, query - a)
        if u >= -1e-12 and v >= -1e-12 and u + v <= 1 + 1e-12:
            return t
    return -1


def tables(points, query, simplices=None):
    if simplices is None:
        triangulation = Delaunay(points)
        simplices = triangulation.simplices.astype(np.int32)
        neighbors = triangulation.neighbors.astype(np.int32)
        seeds = triangulation.find_simplex(query).astype(np.int32)
    else:
        simplices = np.asarray(simplices, dtype=np.int32)
        neighbors = _neighbors_from(simplices).astype(np.int32)
        seeds = np.array([_locate(points, simplices, q) for q in query], np.int32)
    count = simplices.shape[0]
    simplices_padded = -np.ones((2 * len(points), 3), np.int32)
    neighbors_padded = -np.ones_like(simplices_padded)
    simplices_padded[:count] = simplices
    neighbors_padded[:count] = neighbors
    fallback = np.argmin(((query[:, None] - points[None]) ** 2).sum(-1), axis=1)
    return dict(
        query_points=query,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=neighbors_padded,
        simplex_indexes=seeds,
        outside_fallback_indexes=fallback.astype(np.int32),
    )


def run(backend, points, query, simplices=None):
    kwargs = tables(np.asarray(points, float), np.asarray(query, float), simplices)
    circles = sibson.delaunay_circumcircles_from(
        kwargs["points"], kwargs["simplices_padded"], xp=np
    )
    caps = dict(max_cavity_triangles=32, max_neighbors=32)
    if backend == "numpy":
        return sibson._sibson_mappings_weights_numpy_reference(
            **kwargs, circumcircles=circles, **caps
        )
    if backend == "numba":
        return sibson._sibson_mappings_weights_numba(
            **kwargs, circumcircles=circles, **caps
        )
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    jkwargs = {k: jnp.asarray(v) for k, v in kwargs.items()}
    outputs = jax.jit(
        lambda d: sibson.sibson_mappings_weights_from_tables(**d, **caps, xp=jnp)
    )(jkwargs)
    return tuple(np.asarray(o) for o in outputs)


def rows(outputs):
    mappings, sizes, weights = outputs[0], outputs[1], outputs[2]
    return [
        {int(mappings[i, k]): float(weights[i, k]) for k in range(sizes[i])}
        for i in range(len(sizes))
    ]


def max_difference(a, b):
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


BACKENDS = ["numpy", "numba", pytest.param("jax", marks=requires_jax)]

SQUARE = np.array(
    [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
    + [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]]
)


def _square_simplices(diagonal, clockwise=False):
    """The inner square split along one of its two diagonals, plus the outer ring, as explicit triangles."""
    inner = [[0, 3, 1], [0, 2, 3]] if diagonal == "0-3" else [[0, 2, 1], [2, 3, 1]]
    ring = [
        [4, 5, 2],
        [4, 2, 0],
        [5, 6, 3],
        [5, 3, 2],
        [6, 7, 1],
        [6, 1, 3],
        [7, 4, 0],
        [7, 0, 1],
    ]
    simplices = np.array(inner + ring)
    u = SQUARE[simplices[:, 1]] - SQUARE[simplices[:, 0]]
    v = SQUARE[simplices[:, 2]] - SQUARE[simplices[:, 0]]
    signed = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    want_negative = clockwise
    flip = (signed < 0) != want_negative
    simplices[flip] = simplices[flip][:, [0, 2, 1]]
    return simplices


# ---------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("diagonal", ["0-3", "1-2"])
@pytest.mark.parametrize("clockwise", [False, True])
def test__square_centre__quarter_weights_for_either_diagonal_and_orientation(
    backend, diagonal, clockwise
):
    """Fourfold symmetry fixes all four weights to 1/4, whichever diagonal the triangulation picked and
    whichever way its triangles are wound: the query lies exactly on that internal diagonal.
    """
    simplices = _square_simplices(diagonal, clockwise=clockwise)
    (row,) = rows(run(backend, SQUARE, [[0.0, 0.0]], simplices))

    assert set(row) == {0, 1, 2, 3}
    for weight in row.values():
        assert weight == pytest.approx(0.25, abs=1.0e-14)


@pytest.mark.parametrize("backend", BACKENDS)
def test__regular_hexagon_centre__sixth_weights(backend):
    angles = np.arange(6) * np.pi / 3
    points = np.vstack(
        [
            np.column_stack([np.cos(angles), np.sin(angles)]),
            4 * np.column_stack([np.cos(angles + 0.3), np.sin(angles + 0.3)]),
        ]
    )
    (row,) = rows(run(backend, points, [[0.0, 0.0]]))

    assert set(row) == set(range(6))
    for weight in row.values():
        assert weight == pytest.approx(1.0 / 6.0, abs=1.0e-13)


def _asymmetric_fixtures():
    rng = np.random.default_rng(610)
    random_mesh = np.vstack(
        [rng.uniform(-1, 1, (40, 2)), [[-4, -4], [4, -4], [4, 4], [-4, 4]]]
    )
    rectangle = np.array(
        [[x, y] for x in (-2.0, 0.0, 2.0) for y in (-1.0, 0.0, 1.0)]
        + [[-6, -6], [6, -6], [6, 6], [-6, 6]]
    )
    kite = np.array(
        [
            [0.0, 1.0],
            [0.6, 0.0],
            [0.0, -2.0],
            [-0.6, 0.0],
            [-5, -5],
            [5, -5],
            [5, 5],
            [-5, 5],
        ]
    )
    return {
        "random": (random_mesh, rng.uniform(-0.8, 0.8, (25, 2))),
        "rectangle_2to1": (rectangle, rng.uniform(-1.8, 1.8, (15, 2)) * [1.0, 0.5]),
        "kite": (kite, np.array([[0.05, 0.1], [0.0, 0.0], [-0.2, -0.7], [0.1, -1.2]])),
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fixture", ["random", "rectangle_2to1", "kite"])
def test__asymmetric_queries__match_independent_oracle(backend, fixture):
    points, query = _asymmetric_fixtures()[fixture]

    for q, row in zip(query, rows(run(backend, points, query))):
        assert max_difference(row, oracle_weights(points, q)) < 1.0e-10


def _internal_edge_midpoints(points):
    triangulation = Delaunay(points)
    seen, midpoints = set(), []
    for t, s in enumerate(triangulation.simplices):
        for j in range(3):
            if triangulation.neighbors[t, j] < 0:
                continue
            a, b = s[(j + 1) % 3], s[(j + 2) % 3]
            if frozenset((a, b)) in seen:
                continue
            seen.add(frozenset((a, b)))
            pa, pb = points[a], points[b]
            normal = np.array([-(pb - pa)[1], (pb - pa)[0]])
            midpoints.append(((pa + pb) / 2, normal / np.linalg.norm(normal)))
    return midpoints


def _lattice_and_random():
    axis = np.linspace(-1, 1, 5)
    lattice = np.array(
        [[x, y] for x in axis for y in axis]
    ) + 1.0e-3 * np.random.default_rng(1).normal(size=(25, 2))
    rng = np.random.default_rng(611)
    return {
        "lattice": np.vstack([lattice, [[-4, -4], [4, -4], [4, 4], [-4, 4]]]),
        "random": np.vstack(
            [rng.uniform(-1, 1, (30, 2)), [[-4, -4], [4, -4], [4, 4], [-4, 4]]]
        ),
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mesh", ["lattice", "random"])
def test__internal_edges__continuous_from_both_sides(backend, mesh):
    """At every internal edge midpoint, finite and nextafter offsets on either side differ from the on-edge
    row by at most L * offset: continuity (the former code jumped by up to 0.027 at 1e-8).
    """
    points = _lattice_and_random()[mesh]
    midpoints = _internal_edge_midpoints(points)
    offsets = [1.0e-2, 1.0e-6, 1.0e-8, 1.0e-12]

    queries = []
    for midpoint, normal in midpoints:
        queries.append(midpoint)
        for offset in offsets:
            queries += [midpoint + offset * normal, midpoint - offset * normal]
        queries += [
            np.nextafter(midpoint, midpoint + normal),
            np.nextafter(midpoint, midpoint - normal),
        ]
    results = rows(run(backend, points, np.array(queries)))

    stride = 1 + 2 * len(offsets) + 2
    lipschitz = 200.0
    for m in range(len(midpoints)):
        on_edge = results[m * stride]
        probes = results[m * stride + 1 : (m + 1) * stride]
        for i, offset in enumerate(offsets):
            for probe in probes[2 * i : 2 * i + 2]:
                assert max_difference(probe, on_edge) <= lipschitz * offset + 1.0e-12
        for probe in probes[-2:]:
            assert max_difference(probe, on_edge) <= 1.0e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test__affine_precision_partition_and_support(backend):
    """Linear precision, partition of unity, non-negativity and natural-neighbour support -- including the
    near-edge queries where the former code's weights summed to 1.12."""
    points = _lattice_and_random()["random"]
    rng = np.random.default_rng(612)
    queries = list(rng.uniform(-0.9, 0.9, (40, 2)))
    for midpoint, normal in _internal_edge_midpoints(points)[:15]:
        queries += [midpoint, midpoint + 1.0e-8 * normal, midpoint - 1.0e-8 * normal]
    queries = np.array(queries)

    for q, row in zip(queries, rows(run(backend, points, queries))):
        assert sum(row.values()) == pytest.approx(1.0, abs=1.0e-12)
        assert min(row.values()) >= 0.0
        reconstructed = sum(w * points[k] for k, w in row.items())
        np.testing.assert_allclose(reconstructed, q, atol=1.0e-11, rtol=0)
        assert set(row) <= set(oracle_weights(points, q))


def _hull_points():
    return np.random.default_rng(613).uniform(-1, 1, (30, 2))


def _hull_edges(points):
    triangulation = Delaunay(points)
    for t, s in enumerate(triangulation.simplices):
        for j in range(3):
            if triangulation.neighbors[t, j] < 0:
                yield s[(j + 1) % 3], s[(j + 2) % 3], s[j]


@pytest.mark.parametrize("backend", BACKENDS)
def test__hull_edges__linear_on_the_edge_and_continuous_from_inside(backend):
    """On a convex-hull edge Sibson coordinates are linear between its endpoints; approaching the edge from
    inside they converge to that limit at O(distance)."""
    points = _hull_points()
    distances = [1.0e-4, 1.0e-8, 1.0e-12]
    cases, queries = [], []
    for a, b, opposite in _hull_edges(points):
        on_edge = points[a] + 0.37 * (points[b] - points[a])
        inward = points[opposite] - on_edge
        inward /= np.linalg.norm(inward)
        cases.append((a, b))
        queries.append(on_edge)
        queries += [on_edge + d * inward for d in distances]
    results = rows(run(backend, points, np.array(queries)))

    stride = 1 + len(distances)
    for c, (a, b) in enumerate(cases):
        linear = {int(a): 0.63, int(b): 0.37}
        on_edge = results[c * stride]
        assert max_difference(on_edge, linear) <= 1.0e-12
        for d, row in zip(distances, results[c * stride + 1 : (c + 1) * stride]):
            assert max_difference(row, linear) <= 200.0 * d + 1.0e-12


@pytest.mark.parametrize("backend", BACKENDS)
def test__outside_hull__nearest_vertex_fallback_unchanged(backend):
    points = _hull_points()
    query = np.array([[3.0, 0.2], [-2.5, -2.5]])
    nearest = np.argmin(((query[:, None] - points[None]) ** 2).sum(-1), axis=1)

    for n, row in zip(nearest, rows(run(backend, points, query))):
        assert row == {int(n): 1.0}


def _parity_queries():
    points = _lattice_and_random()["random"]
    queries = [q for q, _ in _internal_edge_midpoints(points)]
    queries += list(np.random.default_rng(614).uniform(-1.2, 1.2, (40, 2)))
    return points, np.array(queries)


@requires_jax
def test__numpy_numba_and_jax_jit__agree():
    """One algorithm, three bodies: identical mappings / sizes / flags and weights to 1e-12."""
    points, queries = _parity_queries()
    reference = run("numpy", points, queries)
    for backend in ("numba", "jax"):
        outputs = run(backend, points, queries)
        for index in (0, 1, 3, 4, 5):
            np.testing.assert_array_equal(np.asarray(outputs[index]), reference[index])
        np.testing.assert_allclose(outputs[2], reference[2], rtol=1.0e-12, atol=1.0e-12)


def test__numba_area_tolerances_match_reference_module():
    assert _sibson_numba.ZERO_AREA_TOLERANCE == sibson.SIBSON_ZERO_AREA_TOLERANCE
    assert (
        _sibson_numba.NEGATIVE_AREA_TOLERANCE == sibson.SIBSON_NEGATIVE_AREA_TOLERANCE
    )


def _exact_lattice(n=9):
    axis = np.linspace(-1.0, 1.0, n)
    return np.array([[x, y] for x in axis for y in axis]), axis[1] - axis[0]


def test__delaunay_nn_magnification_areas__integrate_sibson_basis():
    """
    `MeshGeometryDelaunayNN.areas_for_magnification` integrates the Sibson basis functions. On an exact
    lattice (every cell cocircular) symmetry fixes each interior vertex's integral to the cell area h^2
    (which the corrected internal-edge weights now honour), the areas tile the hull, and they integrate a
    linear field exactly up to quadrature error.
    """
    from autoarray.inversion.mesh.mesh_geometry.delaunay import (
        basis_integral_areas_from,
        basis_quadrature_points_from,
    )

    nodes, h = _exact_lattice()
    points = basis_quadrature_points_from(nodes)
    mappings, _, weights = sibson.sibson_mappings_weights_from(nodes, points)
    areas = basis_integral_areas_from(nodes, mappings, weights)

    assert areas.sum() == pytest.approx(4.0, rel=1.0e-12)

    interior = np.all(np.abs(nodes) < 1.0 - 1.5 * h, axis=1)
    np.testing.assert_allclose(areas[interior], h * h, rtol=2.0e-3)

    linear = 0.3 + 1.7 * nodes[:, 0] - 0.4 * nodes[:, 1]
    assert np.sum(areas * linear) == pytest.approx(4.0 * 0.3, rel=1.0e-3)


def test__delaunay_nn_interpolator__uses_sibson_basis_geometry():
    import autoarray as aa
    from autoarray.inversion.mesh.mesh_geometry.delaunay import (
        MeshGeometryDelaunayNN,
    )

    nodes, _ = _exact_lattice(5)
    mesh_grid = aa.Grid2DIrregular(values=nodes[:, ::-1])
    data_grid = aa.Grid2D.uniform(shape_native=(4, 4), pixel_scales=0.3)
    mesh = aa.mesh.DelaunayNN(pixels=nodes.shape[0])
    interpolator = mesh.interpolator_cls(
        mesh=mesh, mesh_grid=mesh_grid, data_grid=data_grid
    )

    assert isinstance(interpolator.mesh_geometry, MeshGeometryDelaunayNN)
    assert interpolator.mesh_geometry.areas_for_magnification.sum() == pytest.approx(
        4.0, rel=1.0e-12
    )
