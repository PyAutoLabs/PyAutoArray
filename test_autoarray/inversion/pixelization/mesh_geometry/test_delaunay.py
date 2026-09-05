import numpy as np
import pytest

import autoarray as aa


def test__neighbors(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh_geometry = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7,
    )

    neighbors = mesh_geometry.neighbors

    expected = np.array(
        [
            [1, 2, 3, 4],
            [0, 2, 3, 5],
            [0, 1, 5, -1],
            [0, 1, 4, 5],
            [0, 3, 5, -1],
            [1, 2, 3, 4],
        ]
    )

    assert all(
        set(neighbors[i]) - {-1} == set(expected[i]) - {-1}
        for i in range(neighbors.shape[0])
    )


def test__voronoi_areas_via_delaunay_from(grid_2d_sub_1_7x7):

    mesh_grid = aa.Grid2DIrregular(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]]
    )

    mesh = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7.over_sampled,
    )

    voronoi_areas = mesh.voronoi_areas

    assert voronoi_areas[1] == pytest.approx(1.39137102, 1.0e-4)
    assert voronoi_areas[3] == pytest.approx(29.836324, 1.0e-4)
    assert voronoi_areas[4] == pytest.approx(-1.0, 1.0e-4)


def test__areas_for_magnification__uniform_lattice(grid_2d_sub_1_7x7):
    """
    Known-answer test on an n x n unit-spacing lattice, FLIPPED by PyAutoArray#524.

    `areas_for_magnification` now returns the *barycentric dual areas* (the sum of
    triangle_area / 3 over the triangles touching each vertex), which tile the convex
    hull of the mesh exactly. The total is therefore the hull area (n - 1) ** 2, and
    every boundary point carries a real, non-zero area -- where the old Voronoi-based
    definition zeroed the whole hull (every hull point has an unbounded Voronoi cell)
    and summed to (n - 2) ** 2.

    Individual cell values are deliberately not pinned: on a perfectly regular lattice
    the dual area of a vertex depends on which diagonal qhull picks for each square
    (interior values alternate 2/3 and 4/3, corners are 1/6 or 1/3), so only the
    diagonal-independent facts are asserted here. The elementwise identity against
    `barycentric_dual_area_from` is pinned by
    `test__areas_for_magnification__equals_barycentric_dual_area` below.

    (scipy's `Qbb Qc Qx Qm Q12 Pp` options handle the perfectly regular lattice
    without degenerate output, so no jitter of the interior points is needed.)
    """
    n = 5

    y, x = np.meshgrid(
        np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij"
    )
    points = np.stack([y.ravel(), x.ravel()], axis=1)

    mesh = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=n * n),
        mesh_grid=aa.Grid2DIrregular(points),
        data_grid=grid_2d_sub_1_7x7.over_sampled,
    )

    areas = mesh.areas_for_magnification.reshape(n, n)

    # the dual areas tile the hull exactly
    assert areas.sum() == pytest.approx(float((n - 1) ** 2), 1.0e-8)

    # ... which is NOT the old, Voronoi-based total
    assert areas.sum() != pytest.approx(float((n - 2) ** 2), 1.0e-2)

    # every cell, boundary included, now carries a real area
    assert (areas > 0.0).all()

    # every dual area is a sum of (unit-square-half) / 3 terms, so it is bounded by
    # the 1 (corner) to 8 (interior) triangles a lattice vertex can touch
    assert areas.min() >= 0.5 / 3.0 - 1.0e-12
    assert areas.max() <= 8.0 * 0.5 / 3.0 + 1.0e-12

    # the hull boundary is exactly the ring the old definition threw away
    boundary = areas.sum() - areas[1:-1, 1:-1].sum()
    assert boundary > 0.0


def test__areas_for_magnification__equals_barycentric_dual_area(grid_2d_sub_1_7x7):
    """
    FLIPPED by PyAutoArray#524 (was
    `test__areas_for_magnification__bounded_boundary_cells_are_kept`, which pinned the
    Voronoi semantics the euclid-dr1-prep phase 8 audit flagged as the bias source and
    which a later fix was told to flip deliberately).

    `areas_for_magnification` is now the barycentric dual area of every vertex, so on
    this 6-point configuration it equals `barycentric_dual_area_from` elementwise and
    sums to the convex-hull area.

    In particular index 3 is no longer the ~29.8 bounded-but-huge Voronoi cell, and
    index 4 -- whose Voronoi region is unbounded, and which the old definition zeroed
    outright -- now carries a real area.
    """
    import scipy.spatial

    from autoarray.inversion.mesh.interpolator.delaunay import (
        barycentric_dual_area_from,
    )

    mesh_grid = aa.Grid2DIrregular(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]]
    )

    mesh = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7.over_sampled,
    )

    areas = mesh.areas_for_magnification

    mesh_grid_xy = np.asarray(mesh.mesh_grid_xy)

    expected = barycentric_dual_area_from(
        mesh_grid_xy,
        scipy.spatial.Delaunay(mesh_grid_xy).simplices,
        xp=np,
    )

    assert areas == pytest.approx(expected, rel=1.0e-10)

    hull_area = scipy.spatial.ConvexHull(mesh_grid_xy).volume

    assert areas.sum() == pytest.approx(hull_area, rel=1.0e-10)

    # the two indexes the old, Voronoi-based definition got wrong
    assert areas[3] != pytest.approx(29.836324, 1.0e-2)
    assert areas[4] > 0.0


def test__areas_for_magnification__repeat_calls_agree(grid_2d_sub_1_7x7):
    """
    `areas_for_magnification` mutates the array it gets from `voronoi_areas`
    (`areas[areas == -1] = 0.0`). `voronoi_areas` is an uncached property that
    recomputes each call, so the `-1` sentinel must never be written back into
    any state shared between calls.
    """
    mesh_grid = aa.Grid2DIrregular(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]]
    )

    mesh = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7.over_sampled,
    )

    first = mesh.areas_for_magnification
    second = mesh.areas_for_magnification

    assert first == pytest.approx(second, 1.0e-8)

    # the sentinel is still there afterwards -- nothing was written back
    assert mesh.voronoi_areas[4] == pytest.approx(-1.0, 1.0e-8)
