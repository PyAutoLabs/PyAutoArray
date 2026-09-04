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
    Known-answer test: on an n x n unit-spacing lattice every *interior* Voronoi
    cell is the unit square around its point (area exactly 1.0), and every point
    on the convex hull has an unbounded Voronoi region, which
    `areas_for_magnification` zeroes.

    The total is therefore (n - 2) ** 2, not n ** 2 -- the summed Delaunay
    "source-plane area" is strictly smaller than the region the mesh covers.

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

    interior = areas[1:-1, 1:-1]
    assert interior == pytest.approx(np.ones((n - 2, n - 2)), 1.0e-8)

    assert areas[0, :] == pytest.approx(np.zeros(n), 1.0e-8)
    assert areas[-1, :] == pytest.approx(np.zeros(n), 1.0e-8)
    assert areas[:, 0] == pytest.approx(np.zeros(n), 1.0e-8)
    assert areas[:, -1] == pytest.approx(np.zeros(n), 1.0e-8)

    assert areas.sum() == pytest.approx(float((n - 2) ** 2), 1.0e-8)


def test__areas_for_magnification__bounded_boundary_cells_are_kept(grid_2d_sub_1_7x7):
    """
    Pins the CURRENT semantics of `areas_for_magnification`: only cells whose
    Voronoi region is *unbounded* (the `-1` sentinel) are zeroed. A cell that is
    bounded but sits at the edge of the mesh is kept at full size, even when it
    is an order of magnitude larger than the interior cells -- here index 3 keeps
    an area of ~29.8 against index 1's ~1.4.

    This is the bias candidate flagged by the euclid-dr1-prep phase 8 audit: a
    magnification denominator built from these areas is inflated by the huge
    bounded boundary cells. A later fix should flip this test *deliberately*,
    not silently.
    """
    mesh_grid = aa.Grid2DIrregular(
        [[0.0, 0.0], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]]
    )

    mesh = aa.MeshGeometryDelaunay(
        mesh=aa.mesh.Delaunay(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7.over_sampled,
    )

    areas = mesh.areas_for_magnification

    assert areas[1] == pytest.approx(1.39137102, 1.0e-4)
    # bounded, huge, and kept:
    assert areas[3] == pytest.approx(29.836324, 1.0e-4)
    # unbounded (-1 in `voronoi_areas`), and therefore zeroed:
    assert areas[4] == pytest.approx(0.0, 1.0e-8)


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
