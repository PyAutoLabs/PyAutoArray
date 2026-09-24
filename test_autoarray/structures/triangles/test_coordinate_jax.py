"""
Tests of the JAX `CoordinateArrayTriangles` vertex table and its containment path.

`CoordinateArrayTriangles._vertices_and_indices` returns the flat, non-deduplicated ``(3N, 2)``
vertex table with an ``arange`` index map (PyAutoArray#568, autolens_profiling#297): under ``jit`` a
static-size ``jnp.unique`` produced 3N rows anyway and only added a lexicographic sort. These tests
pin that the table round-trips to the triangles exactly, that NaN padding stays NaN and is never
returned by containment, that the kept triangles match the deduplicating NumPy sibling, and that
the traced containment carries no sort.

The static initial lattice (`for_limits_and_scale(..., static_vertices=True)`, point-source CPU
phase 3) instead carries the cached `static_vertex_table` of geometrically unique vertices: the
tests below pin its size for the PointSolver lattice (11 859 of 69 849 slots), that it gathers back
to the triangles within a few ulp, that containment matches the flat table and the NumPy sibling,
that derived lattices drop it, and that the cache is keyed on geometry and read-only.
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
        static_vertex_table,
    )
    from autoarray.structures.triangles.coordinate_array_np import (
        CoordinateArrayTrianglesNp,
    )
    from autoarray.structures.triangles.shape import Point

    LIMITS = dict(y_min=-1.0, y_max=1.0, x_min=-1.0, x_max=1.0, scale=0.5)

    # The PointSolver's default image-plane extent for a 100x100, 0.2" grid.
    SOLVER_LIMITS = dict(y_min=-9.9, y_max=9.9, x_min=-9.9, x_max=9.9, scale=0.2)

    def _lattice():
        return CoordinateArrayTriangles.for_limits_and_scale(**LIMITS)

    def _static_lattice(limits=LIMITS):
        return CoordinateArrayTriangles.for_limits_and_scale(
            **limits, static_vertices=True
        )

    def _max_ulps(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return float(np.max(np.abs(a - b) / np.spacing(np.maximum(np.abs(a), 1.0))))

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


def test__default_lattice_has_no_vertex_table():
    assert _lattice().vertex_table is None


def test__static_vertex_table__solver_lattice_counts():
    """
    The +-9.9" / 0.2" lattice has 23 283 triangles, 69 849 vertex slots, 28 665 exact-float distinct
    rows and 11 859 geometrically distinct points -- the same count the deduplicating NumPy sibling
    gives once its ulp-level duplicates are rounded together.
    """
    lattice = _static_lattice(SOLVER_LIMITS)
    vertices = np.asarray(lattice.vertices)
    indices = np.asarray(lattice.indices)

    assert lattice.coordinates.shape[0] == 23283
    assert vertices.shape == (11859, 2)
    assert indices.shape == (23283, 3)
    assert indices.min() == 0 and indices.max() == 11858
    assert np.unique(np.round(vertices, 9), axis=0).shape[0] == 11859

    triangles_np = CoordinateArrayTrianglesNp.for_limits_and_scale(**SOLVER_LIMITS)
    flat_np = np.asarray(triangles_np.triangles).reshape(-1, 2)
    assert np.unique(flat_np, axis=0).shape[0] == 28665
    assert np.unique(np.round(flat_np, 9), axis=0).shape[0] == 11859


@pytest.mark.parametrize("limits", [LIMITS, SOLVER_LIMITS])
def test__static_vertex_table_round_trips_to_triangles(limits):
    static = _static_lattice(limits)
    default = CoordinateArrayTriangles.for_limits_and_scale(**limits)

    assert np.array_equal(
        np.asarray(static.coordinates), np.asarray(default.coordinates)
    )

    gathered = np.asarray(static.vertices)[np.asarray(static.indices)]
    assert gathered.shape == np.asarray(default.triangles).shape
    assert _max_ulps(gathered, default.triangles) <= 4

    via_with_vertices = np.asarray(static.with_vertices(static.vertices).triangles)
    assert np.array_equal(via_with_vertices, gathered)


@pytest.mark.parametrize("point_index", range(5))
def test__static_vertices_kept_triangles_match_numpy(point_index):
    triangles_np = _lattice_np()
    point = _source_points(triangles_np)[point_index]

    kept_np = _kept_rows(triangles_np, point)
    kept = _kept_rows(_static_lattice(), point)

    assert 0 < len(kept_np) < MAX_CONTAINING_SIZE
    assert kept == kept_np


@pytest.mark.parametrize("point_index", range(5))
def test__static_vertices_containing_indices_match_default(point_index):
    point = Point(*_source_points(_lattice_np())[point_index])
    static = _static_lattice()
    default = _lattice()

    static_indices = np.asarray(
        static.with_vertices(static.vertices).containing_indices(point)
    )
    default_indices = np.asarray(
        default.with_vertices(default.vertices).containing_indices(point)
    )

    assert set(static_indices[static_indices >= 0]) == set(
        default_indices[default_indices >= 0]
    )


def test__static_vertices_containing_indices__jit_matches_eager():
    static = _static_lattice(SOLVER_LIMITS)
    point = Point(0.13, -0.27)

    def containing():
        return static.with_vertices(static.vertices).containing_indices(point)

    assert np.array_equal(np.asarray(jax.jit(containing)()), np.asarray(containing()))


def test__derived_lattices_drop_the_vertex_table():
    static = _static_lattice()
    n = static.coordinates.shape[0]

    for derived in (
        static.for_indexes(jnp.arange(4)),
        static.up_sample(),
        static.neighborhood(),
    ):
        assert derived.vertex_table is None
        assert derived.vertices.shape == (3 * derived.coordinates.shape[0], 2)

    assert static.vertices.shape[0] < 3 * n


def test__static_vertex_table_is_cached_per_geometry():
    first = static_vertex_table(-1.0, 1.0, -1.0, 1.0, 0.5)

    assert static_vertex_table(-1.0, 1.0, -1.0, 1.0, 0.5) is first
    assert _static_lattice().vertex_table is first

    finer = static_vertex_table(-1.0, 1.0, -1.0, 1.0, 0.25)
    assert finer is not first
    assert finer[0].shape[0] > first[0].shape[0]


def test__static_vertex_table_is_read_only():
    vertices, indices = static_vertex_table(-1.0, 1.0, -1.0, 1.0, 0.5)

    assert not vertices.flags.writeable
    assert not indices.flags.writeable
    with pytest.raises(ValueError):
        vertices[0, 0] = 1.0
    with pytest.raises(ValueError):
        indices[0, 0] = 1


def test_no_sort__static_vertices():
    static = _static_lattice(SOLVER_LIMITS)

    def containing():
        return static.with_vertices(static.vertices).containing_indices(Point(0.1, 0.2))

    compiled = jax.jit(containing).lower().compile().as_text()

    sorts = re.findall(r"\bsort\b", compiled, flags=re.IGNORECASE)
    assert not sorts, f"traced containment carries {len(sorts)} sort op(s)"
