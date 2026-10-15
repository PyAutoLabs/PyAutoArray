"""Parity of the numba Sibson kernel against the per-query NumPy reference.

``sibson_numba_kernel`` is a port of ``_sibson_single_from_tables``; the
integer outputs and diagnostics must agree exactly and the weights to rounding
(the reference groups equal vertex ids with an unstable ``argsort``, so the
order contributions are summed in can differ).
"""

import numpy as np
import pytest

pytest.importorskip("numba")

from autoarray.inversion.mesh.interpolator import sibson
from autoarray.inversion.mesh.interpolator.delaunay import (
    pix_indexes_delaunay_walk_from,
    scipy_delaunay_tri_only,
)


def _tables(points, query):
    simplices, neighbors, vertex_simplex = scipy_delaunay_tri_only(points)
    delaunay_mappings, simplex_indexes = pix_indexes_delaunay_walk_from(
        query_points=query,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        vertex_simplex=vertex_simplex,
        xp=np,
        return_simplex_indexes=True,
    )
    return dict(
        query_points=query,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        simplex_indexes=simplex_indexes,
        outside_fallback_indexes=delaunay_mappings[:, 0],
        circumcircles=sibson.delaunay_circumcircles_from(points, simplices, xp=np),
    )


def _reference_and_numba(points, query, max_cavity_triangles=32, max_neighbors=32):
    kwargs = _tables(points, query)
    kwargs.update(
        max_cavity_triangles=max_cavity_triangles, max_neighbors=max_neighbors
    )
    reference = sibson._sibson_mappings_weights_numpy_reference(**kwargs)
    numba_outputs = sibson._sibson_mappings_weights_numba(**kwargs)
    return reference, numba_outputs


def _assert_parity(reference, numba_outputs):
    names = ("mappings", "sizes", "weights", "cavity_sizes", "overflow", "degenerate")
    assert len(reference) == len(numba_outputs) == 6
    for name, expected, actual in zip(names, reference, numba_outputs):
        assert expected.dtype == actual.dtype, name
        assert expected.shape == actual.shape, name
        if name == "weights":
            assert np.array_equal(np.isnan(expected), np.isnan(actual))
            np.testing.assert_allclose(
                actual, expected, rtol=1.0e-12, atol=1.0e-12, equal_nan=True
            )
        else:
            assert np.array_equal(expected, actual), name


@pytest.mark.parametrize(
    "seed, point_count, query_count, query_extent",
    [
        (0, 20, 200, 1.0),
        (1, 300, 1500, 1.3),
        (2, 100, 600, 0.8),
        (3, 1000, 1500, 1.2),
    ],
)
def test__random_meshes__match_reference(seed, point_count, query_count, query_extent):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.0, 1.0, size=(point_count, 2))
    query = rng.uniform(-query_extent, query_extent, size=(query_count, 2))
    # Queries exactly on vertices and on edge midpoints of the mesh.
    simplices, _, _ = scipy_delaunay_tri_only(points)
    simplices = simplices[(simplices >= 0).all(axis=1)][:20]
    edge_midpoints = 0.5 * (points[simplices[:, 0]] + points[simplices[:, 1]])
    query = np.concatenate([query, points[:20], edge_midpoints])

    reference, numba_outputs = _reference_and_numba(points, query)

    _assert_parity(reference, numba_outputs)
    if query_extent > 1.0:
        assert (reference[3] == 0).any()  # some queries fall outside the hull
    assert (reference[1] == 1).any()  # on-vertex / outside fallback rows
    assert (reference[1] == 3).any()


def test__cocircular_grid__on_vertex_on_edge_and_collinear_queries():
    grid = np.stack(np.meshgrid(np.arange(6.0), np.arange(6.0)), axis=-1)
    points = grid.reshape(-1, 2)
    simplices, _, _ = scipy_delaunay_tri_only(points)

    query = [points[7], points[14], [2.5, 2.5], [-1.0, 2.0]]
    for triangle in simplices:
        if (triangle < 0).any():
            continue
        for i in range(3):
            a, b = points[triangle[i]], points[triangle[(i + 1) % 3]]
            for t in (-0.3, 0.25, 0.5, 1.3):
                query.append(a + t * (b - a))
    query = np.asarray(query)

    reference, numba_outputs = _reference_and_numba(points, query)

    _assert_parity(reference, numba_outputs)
    assert reference[1][0] == 1 and reference[0][0, 0] == 7


def test__thin_triangle__degenerate_watson_circle_matches_reference():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.01],
            [0.5, -100.0],
            [0.5, 100.0],
            [-50.0, 0.0],
            [50.0, 0.0],
        ]
    )
    query = np.array(
        [
            [0.3, 1.0e-15],
            [0.3, 0.0],
            [0.0, 0.0],
            [0.3, 0.004],
            [0.7, 1.0e-15],
            [0.3, 1.0e-12],
        ]
    )

    reference, numba_outputs = _reference_and_numba(points, query)

    _assert_parity(reference, numba_outputs)
    assert reference[5].tolist() == [True, False, False, False, True, False]
    assert np.isnan(numba_outputs[2][0]).all()


@pytest.mark.parametrize(
    "max_cavity_triangles, max_neighbors",
    [(1, 3), (2, 32), (32, 4), (4, 4), (3, 40)],
)
def test__cavity_and_neighbour_overflow__match_reference(
    max_cavity_triangles, max_neighbors
):
    rng = np.random.default_rng(5)
    points = rng.uniform(-1.0, 1.0, size=(200, 2))
    query = np.concatenate([rng.uniform(-1.1, 1.1, size=(600, 2)), points[:10]])

    reference, numba_outputs = _reference_and_numba(
        points,
        query,
        max_cavity_triangles=max_cavity_triangles,
        max_neighbors=max_neighbors,
    )

    _assert_parity(reference, numba_outputs)
    assert reference[4].any()


def test__dispatch__numpy_branch_uses_kernel_or_reference(monkeypatch):
    rng = np.random.default_rng(7)
    points = rng.uniform(-1.0, 1.0, size=(80, 2))
    query = rng.uniform(-1.1, 1.1, size=(300, 2))
    kwargs = _tables(points, query)
    del kwargs["circumcircles"]

    reference, numba_outputs = _reference_and_numba(points, query)

    dispatched = sibson.sibson_mappings_weights_from_tables(xp=np, **kwargs)
    for expected, actual in zip(numba_outputs, dispatched):
        assert np.array_equal(expected, actual, equal_nan=True)

    monkeypatch.setattr(sibson, "_sibson_use_numba", lambda: False)
    dispatched = sibson.sibson_mappings_weights_from_tables(xp=np, **kwargs)
    for expected, actual in zip(reference, dispatched):
        assert np.array_equal(expected, actual, equal_nan=True)
