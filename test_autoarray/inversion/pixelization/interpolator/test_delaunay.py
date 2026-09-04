import importlib.util

import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.mesh.interpolator.delaunay import (
    barycentric_dual_area_from,
    jax_delaunay,
    scipy_delaunay,
)
from autoarray.inversion.mesh.mesh_geometry.delaunay import voronoi_areas_numpy


def test__scipy_delaunay__simplices(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7,
    )

    assert (mesh_grid.delaunay.simplices[0, :] == np.array([3, 4, 0])).all()
    assert (mesh_grid.delaunay.simplices[1, :] == np.array([3, 5, 4])).all()
    assert (mesh_grid.delaunay.simplices[-1, :] == np.array([-1, -1, -1])).all()


def test__scipy_delaunay__split(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_grid = aa.InterpolatorDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7,
    )

    assert mesh_grid.delaunay.split_points[0, :] == pytest.approx(
        [0.45059473, 0.1], 1.0e-4
    )
    assert mesh_grid.delaunay.split_points[1, :] == pytest.approx(
        [-0.25059473, 0.1], 1.0e-4
    )
    assert mesh_grid.delaunay.split_points[-1, :] == pytest.approx(
        [2.1, 0.39142161], 1.0e-4
    )
    assert mesh_grid.delaunay.splitted_mappings[0, :] == pytest.approx(
        [2, 1, 0], 1.0e-4
    )
    assert mesh_grid.delaunay.splitted_mappings[1, :] == pytest.approx(
        [0, -1, -1], 1.0e-4
    )
    assert mesh_grid.delaunay.splitted_mappings[-1, :] == pytest.approx(
        [5, 1, 2], 1.0e-4
    )


# ----------------------------------------------------------------------------
# Barycentric dual areas (euclid-dr1-prep phase 8 audit).
#
# Two different "areas" exist for a Delaunay mesh on two different code paths:
#
#   * `barycentric_dual_area_from` (here) -- sum of triangle_area / 3 over the
#     triangles touching each vertex. These tile the convex hull exactly and
#     integrate the piecewise-linear interpolant exactly.
#   * `MeshGeometryDelaunay.areas_for_magnification` -- since PyAutoArray#524
#     these ARE the dual areas (taken straight from the interpolator, which
#     computes them in-graph). They used to be scipy Voronoi cell areas with
#     only the *unbounded* cells zeroed, which do NOT tile the hull and do NOT
#     integrate the interpolant; that quantity is still available, unchanged,
#     as `voronoi_areas` / `voronoi_areas_numpy`.
#
# The tests below pin both facts, including the size of the divergence between
# the dual areas and the Voronoi areas.
# ----------------------------------------------------------------------------

# jax is an `[optional]` extra and is absent on the NumPy-only matrix env, so
# the in-graph parity test skips rather than fails there (same convention as
# test_knn_barycentric.py).
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)


def test__barycentric_dual_area__single_triangle():

    points = np.array([[0.0, 0.0], [0.0, 4.0], [3.0, 0.0]])
    simplices = np.array([[0, 1, 2]])

    area = 0.5 * 3.0 * 4.0

    dual = barycentric_dual_area_from(points, simplices, xp=np)

    assert dual == pytest.approx(np.full(3, area / 3.0), 1.0e-10)
    assert dual.sum() == pytest.approx(area, 1.0e-10)


def test__barycentric_dual_area__sums_to_convex_hull_area():
    """
    The dual areas partition the convex hull exactly, so they sum to the hull
    area. The Voronoi areas -- which `areas_for_magnification` returned before
    PyAutoArray#524 and which `voronoi_areas_numpy` still returns -- do not: on
    this configuration they overshoot the hull by ~29%, because the bounded
    boundary cells extend well outside the hull and are kept.

    Since #524 a `MeshGeometryDelaunay` built on the same points returns the
    dual areas from `areas_for_magnification`, which is also asserted here.
    """
    import scipy.spatial

    points = np.random.default_rng(1).random((40, 2))

    simplices = scipy.spatial.Delaunay(points).simplices

    dual = barycentric_dual_area_from(points, simplices, xp=np)

    hull_area = scipy.spatial.ConvexHull(points).volume

    assert dual.sum() == pytest.approx(hull_area, rel=1.0e-10)

    voronoi = voronoi_areas_numpy(points)
    voronoi = np.where(voronoi == -1.0, 0.0, voronoi)

    ratio = voronoi.sum() / hull_area

    assert ratio == pytest.approx(1.2868, 1.0e-3), (
        f"Voronoi areas (unbounded cells zeroed) sum to {voronoi.sum()} against a "
        f"convex-hull area of {hull_area} (ratio {ratio}); the two area "
        f"definitions are not interchangeable."
    )
    assert voronoi.sum() != pytest.approx(hull_area, rel=1.0e-2)

    # ... and the mesh geometry now hands out the dual areas, not those Voronoi
    # areas. `mesh_grid_xy` may swap the (y,x) columns to (x,y); dual areas are
    # invariant under that swap, so comparing against the dual areas of the same
    # `points` array is exact.
    mesh_geometry = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=40),
        mesh_grid=aa.Grid2DIrregular(points),
        data_grid=aa.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0),
    )

    assert mesh_geometry.areas_for_magnification == pytest.approx(dual, rel=1.0e-10)


def test__linear_field_integral__dual_areas_exact():
    """
    For a field that is linear over the mesh, the piecewise-linear interpolant
    is the field itself, so its integral over the hull is exactly
    `sum(f_i * dual_area_i)`. The Voronoi areas get the same integral wrong by
    tens of percent.
    """
    import scipy.spatial

    points = np.random.default_rng(1).random((40, 2))

    simplices = scipy.spatial.Delaunay(points).simplices

    f = 0.3 + 0.7 * points[:, 0] - 0.2 * points[:, 1]

    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]

    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0]
    )

    tri_area = 0.5 * np.abs(cross)

    # exact: on each triangle a linear field integrates to (mean of its three
    # vertex values) x (triangle area)
    exact = (f[simplices].mean(axis=1) * tri_area).sum()

    dual = barycentric_dual_area_from(points, simplices, xp=np)

    assert (f * dual).sum() == pytest.approx(exact, rel=1.0e-10)

    voronoi = voronoi_areas_numpy(points)
    voronoi = np.where(voronoi == -1.0, 0.0, voronoi)

    voronoi_integral = (f * voronoi).sum()

    assert (
        abs(voronoi_integral - exact) / exact > 0.01
    ), f"Voronoi-weighted integral {voronoi_integral} vs exact {exact}"


@requires_jax
def test__barycentric_dual_area__numpy_matches_jax_in_graph():
    """
    `jax_delaunay` carries its own in-graph copy of the dual-area computation
    (a masked scatter-add over the padded simplices) rather than calling
    `barycentric_dual_area_from`. The two must agree.

    Since PyAutoArray#524 both paths return the dual areas as the sixth element
    of their tuple (the mesh geometry consumes them as the magnification
    quadrature weights), so they are compared directly. The split points --
    which carry the dual areas as `areas_factor * sqrt(areas)` -- are still
    compared too, since they exercise the same values through the real library
    path.
    """
    import jax.numpy as jnp

    rng = np.random.default_rng(3)

    points = rng.random((25, 2))
    query_points = rng.random((30, 2))

    _, simplices_np, _, split_np, _, areas_np = scipy_delaunay(
        points, query_points, 0.5
    )
    _, simplices_jx, _, split_jx, _, areas_jx = jax_delaunay(
        jnp.asarray(points), jnp.asarray(query_points), 0.5
    )

    assert np.array_equal(np.asarray(simplices_np), np.asarray(simplices_jx))
    assert np.asarray(split_jx) == pytest.approx(np.asarray(split_np), abs=1.0e-10)
    assert np.asarray(areas_jx) == pytest.approx(np.asarray(areas_np), abs=1.0e-10)
