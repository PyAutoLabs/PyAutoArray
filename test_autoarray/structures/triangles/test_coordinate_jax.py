"""
Tests of the JAX `CoordinateArrayTriangles` vertex table and its containment path.

`CoordinateArrayTriangles._vertices_and_indices` returns the flat, non-deduplicated ``(3N, 2)``
vertex table with an ``arange`` index map (PyAutoArray#568, autolens_profiling#297): under ``jit`` a
static-size ``jnp.unique`` produced 3N rows anyway and only added a lexicographic sort. These tests
pin that the table round-trips to the triangles exactly, that NaN padding stays NaN and is never
returned by containment, that the kept triangles match the deduplicating NumPy sibling, and that
the traced containment carries no sort.
"""

import importlib
import re

import numpy as np
import pytest


# jax is an `[optional]` extra and is absent on the NumPy-only matrix env: every test in this module
# skips there.
if importlib.util.find_spec("jax") is None:
    pytestmark = pytest.mark.skip(reason="requires jax (the [optional] extras)")

    def test__placeholder_requires_jax():  # pragma: no cover
        pass

else:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from autoarray.structures.triangles.array import MAX_CONTAINING_SIZE
    from autoarray.structures.triangles.coordinate_array import (
        CoordinateArrayTriangles,
    )
    from autoarray.structures.triangles.coordinate_array_np import (
        CoordinateArrayTrianglesNp,
    )
    from autoarray.structures.triangles.shape import Point

    LIMITS = dict(y_min=-1.0, y_max=1.0, x_min=-1.0, x_max=1.0, scale=0.5)

    def _lattice():
        return CoordinateArrayTriangles.for_limits_and_scale(**LIMITS)

    def _lattice_np():
        return CoordinateArrayTrianglesNp.for_limits_and_scale(**LIMITS)

    def _round_trip(coordinates):
        triangles = CoordinateArrayTriangles(coordinates=coordinates, side_length=0.5)
        return (
            triangles.triangles,
            triangles.vertices.reshape(-1, 3, 2),
            triangles.with_vertices(triangles.vertices).triangles,
        )

    def _containing(coordinates):
        triangles = CoordinateArrayTriangles(coordinates=coordinates, side_length=0.5)
        return triangles.with_vertices(triangles.vertices).containing_indices(
            Point(0.1, 0.2)
        )

    def _source_points(triangles_np):
        """
        Five source points on the lattice: two interior points, one on a lattice vertex, one on the
        midpoint of an edge shared by two triangles, and one centroid. The edge is the horizontal one
        (constant element 1), whose midpoint the barycentric test keeps on both sides; the midpoints
        of the slanted edges round outside both triangles on NumPy and JAX alike.
        """
        tri = np.asarray(triangles_np.triangles)
        interior = int(np.argmin(np.linalg.norm(tri.mean(axis=1), axis=1)))
        vertex = tri[interior, 0]
        edge_midpoint = 0.5 * (tri[interior, 1] + tri[interior, 2])
        centroid = tri[interior].mean(axis=0)
        return [
            (0.1, 0.2),
            (-0.37, 0.41),
            tuple(vertex),
            tuple(edge_midpoint),
            tuple(centroid),
        ]

    def _kept_rows(triangles, point):
        kept = triangles.for_indexes(
            triangles.with_vertices(triangles.vertices).containing_indices(
                Point(*point)
            )
        )
        rows = np.asarray(kept.triangles).reshape(-1, 6)
        rows = rows[np.all(np.isfinite(rows), axis=1)]
        return {tuple(np.round(row, 8)) for row in rows}


def test__vertices_round_trip_to_triangles():
    triangles = _lattice()

    assert triangles.vertices.shape == (3 * triangles.coordinates.shape[0], 2)
    assert np.array_equal(
        np.asarray(triangles.indices),
        np.arange(3 * triangles.coordinates.shape[0]).reshape(-1, 3),
    )
    assert np.array_equal(
        np.asarray(triangles.vertices.reshape(-1, 3, 2)),
        np.asarray(triangles.triangles),
    )
    assert np.array_equal(
        np.asarray(triangles.with_vertices(triangles.vertices).triangles),
        np.asarray(triangles.triangles),
    )


def test__vertices_round_trip_to_triangles__jit():
    """
    Compared inside one jitted computation: XLA may fuse the eager and jitted triangle arithmetic
    differently (the two differ by up to 1 ulp), so the exact round trip is asserted between the
    outputs of the same compiled program.
    """
    triangles, vertices, with_vertices = jax.jit(_round_trip)(_lattice().coordinates)

    assert np.array_equal(np.asarray(vertices), np.asarray(triangles))
    assert np.array_equal(np.asarray(with_vertices), np.asarray(triangles))


def test__nan_padding_stays_nan_and_is_never_contained():
    lattice = _lattice()
    point = tuple(np.asarray(lattice.triangles[0]).mean(axis=0))

    padded = lattice.for_indexes(jnp.array([0, 2, -1, -1]))

    coordinates = np.asarray(padded.coordinates)
    assert np.all(np.isfinite(coordinates[:2]))
    assert np.all(np.isnan(coordinates[2:]))

    triangles = np.asarray(padded.with_vertices(padded.vertices).triangles)
    assert np.all(np.isfinite(triangles[:2]))
    assert np.all(np.isnan(triangles[2:]))

    containing = np.asarray(
        padded.with_vertices(padded.vertices).containing_indices(Point(*point))
    )
    assert containing[0] == 0
    assert np.all(containing[1:] == -1)


@pytest.mark.parametrize("point_index", range(5))
def test__kept_triangles_match_numpy(point_index):
    triangles_np = _lattice_np()
    triangles = _lattice()
    point = _source_points(triangles_np)[point_index]

    kept_np = _kept_rows(triangles_np, point)
    kept = _kept_rows(triangles, point)

    assert 0 < len(kept_np) < MAX_CONTAINING_SIZE
    assert kept == kept_np


def test__numpy_vertices_are_deduplicated():
    triangles_np = _lattice_np()
    n = triangles_np.coordinates.shape[0]

    assert triangles_np.vertices.shape[0] < 3 * n
    assert np.unique(triangles_np.vertices, axis=0).shape[0] == (
        triangles_np.vertices.shape[0]
    )
    assert np.array_equal(
        triangles_np.vertices[triangles_np.indices], triangles_np.triangles
    )


def test_no_sort():
    coordinates = _lattice().coordinates
    compiled = jax.jit(_containing).lower(coordinates).compile().as_text()

    sorts = re.findall(r"\bsort\b", compiled, flags=re.IGNORECASE)
    assert not sorts, f"traced containment carries {len(sorts)} sort op(s)"
