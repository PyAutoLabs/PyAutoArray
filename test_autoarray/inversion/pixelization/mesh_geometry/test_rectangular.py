import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
    overlay_grid_from,
)


from autoarray.inversion.mesh.mesh_geometry.rectangular import (
    rectangular_neighbors_from,
)


def test__rectangular_neighbors_from():
    # I0I1I2I
    # I3I4I5I
    # I6I7I8I

    (neighbors, neighbors_sizes) = rectangular_neighbors_from(shape_native=(3, 3))

    # TODO : Use pytest.parameterize

    assert (neighbors[0] == [1, 3, -1, -1]).all()
    assert (neighbors[1] == [0, 2, 4, -1]).all()
    assert (neighbors[2] == [1, 5, -1, -1]).all()
    assert (neighbors[3] == [0, 4, 6, -1]).all()
    assert (neighbors[4] == [1, 3, 5, 7]).all()
    assert (neighbors[5] == [2, 4, 8, -1]).all()
    assert (neighbors[6] == [3, 7, -1, -1]).all()
    assert (neighbors[7] == [4, 6, 8, -1]).all()
    assert (neighbors[8] == [5, 7, -1, -1]).all()

    assert (neighbors_sizes == np.array([2, 3, 2, 3, 4, 3, 2, 3, 2])).all()

    # I0I1I 2I 3I
    # I4I5I 6I 7I
    # I8I9I10I11I

    (neighbors, neighbors_sizes) = rectangular_neighbors_from(shape_native=(3, 4))

    assert (neighbors[0] == [1, 4, -1, -1]).all()
    assert (neighbors[1] == [0, 2, 5, -1]).all()
    assert (neighbors[2] == [1, 3, 6, -1]).all()
    assert (neighbors[3] == [2, 7, -1, -1]).all()
    assert (neighbors[4] == [0, 5, 8, -1]).all()
    assert (neighbors[5] == [1, 4, 6, 9]).all()
    assert (neighbors[6] == [2, 5, 7, 10]).all()
    assert (neighbors[7] == [3, 6, 11, -1]).all()
    assert (neighbors[8] == [4, 9, -1, -1]).all()
    assert (neighbors[9] == [5, 8, 10, -1]).all()
    assert (neighbors[10] == [6, 9, 11, -1]).all()
    assert (neighbors[11] == [7, 10, -1, -1]).all()

    assert (neighbors_sizes == np.array([2, 3, 3, 2, 3, 4, 4, 3, 2, 3, 3, 2])).all()

    # I0I 1I 2I
    # I3I 4I 5I
    # I6I 7I 8I
    # I9I10I11I

    (neighbors, neighbors_sizes) = rectangular_neighbors_from(shape_native=(4, 3))

    assert (neighbors[0] == [1, 3, -1, -1]).all()
    assert (neighbors[1] == [0, 2, 4, -1]).all()
    assert (neighbors[2] == [1, 5, -1, -1]).all()
    assert (neighbors[3] == [0, 4, 6, -1]).all()
    assert (neighbors[4] == [1, 3, 5, 7]).all()
    assert (neighbors[5] == [2, 4, 8, -1]).all()
    assert (neighbors[6] == [3, 7, 9, -1]).all()
    assert (neighbors[7] == [4, 6, 8, 10]).all()
    assert (neighbors[8] == [5, 7, 11, -1]).all()
    assert (neighbors[9] == [6, 10, -1, -1]).all()
    assert (neighbors[10] == [7, 9, 11, -1]).all()
    assert (neighbors[11] == [8, 10, -1, -1]).all()

    assert (neighbors_sizes == np.array([2, 3, 2, 3, 4, 3, 3, 4, 3, 2, 3, 2])).all()

    # I0 I 1I 2I 3I
    # I4 I 5I 6I 7I
    # I8 I 9I10I11I
    # I12I13I14I15I

    (neighbors, neighbors_sizes) = rectangular_neighbors_from(shape_native=(4, 4))

    assert (neighbors[0] == [1, 4, -1, -1]).all()
    assert (neighbors[1] == [0, 2, 5, -1]).all()
    assert (neighbors[2] == [1, 3, 6, -1]).all()
    assert (neighbors[3] == [2, 7, -1, -1]).all()
    assert (neighbors[4] == [0, 5, 8, -1]).all()
    assert (neighbors[5] == [1, 4, 6, 9]).all()
    assert (neighbors[6] == [2, 5, 7, 10]).all()
    assert (neighbors[7] == [3, 6, 11, -1]).all()
    assert (neighbors[8] == [4, 9, 12, -1]).all()
    assert (neighbors[9] == [5, 8, 10, 13]).all()
    assert (neighbors[10] == [6, 9, 11, 14]).all()
    assert (neighbors[11] == [7, 10, 15, -1]).all()
    assert (neighbors[12] == [8, 13, -1, -1]).all()
    assert (neighbors[13] == [9, 12, 14, -1]).all()
    assert (neighbors[14] == [10, 13, 15, -1]).all()
    assert (neighbors[15] == [11, 14, -1, -1]).all()

    assert (
        neighbors_sizes == np.array([2, 3, 3, 2, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2])
    ).all()


def test__neighbors__compare_to_mesh_util():
    # I0 I 1I 2I 3I
    # I4 I 5I 6I 7I
    # I8 I 9I10I11I
    # I12I13I14I15I

    mesh = aa.mesh.RectangularUniform(shape=(7, 5))

    mesh_grid = overlay_grid_from(
        shape_native=mesh.shape, grid=aa.Grid2DIrregular(np.zeros((2, 2))), buffer=1e-8
    )

    mesh_geometry = aa.MeshGeometryRectangular(
        mesh=mesh, mesh_grid=mesh_grid, data_grid=None
    )

    (neighbors_util, neighbors_sizes_util) = rectangular_neighbors_from(
        shape_native=(7, 5)
    )

    assert (mesh_geometry.neighbors == neighbors_util).all()
    assert (mesh_geometry.neighbors.sizes == neighbors_sizes_util).all()


def test__areas_transformed(mask_2d_7x7):

    grid = aa.Grid2D.no_mask(
        values=[
            [-1.5, -1.5],
            [-1.5, 0.0],
            [-1.5, 1.5],
            [0.0, -1.5],
            [0.0, 0.0],
            [0.0, 1.5],
            [1.5, -1.5],
            [1.5, 0.0],
            [1.5, 1.5],
        ],
        pixel_scales=1.5,
        shape_native=(3, 3),
        over_sample_size=1,
    )

    mesh_grid = overlay_grid_from(shape_native=(3, 3), grid=grid, buffer=1e-8)

    mesh = aa.mesh.RectangularAdaptDensity(shape=(3, 3))

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=mesh_grid,
    )

    areas = interpolator.mesh_geometry.areas_transformed

    assert np.all(np.isfinite(areas))
    assert np.all(areas > 0.0)

    # The unit square maps exactly onto the data bounding box (3.0 x 3.0), so
    # the per-axis edge differences telescope and the areas sum to its area.
    assert areas.sum() == pytest.approx(9.0, rel=1e-8)

    # The input grid is 4-fold symmetric, so the transformed areas must be:
    # equal corners, equal edge-midpoints, single centre value.
    assert areas[0] == pytest.approx(areas[2], rel=1e-8)
    assert areas[0] == pytest.approx(areas[6], rel=1e-8)
    assert areas[0] == pytest.approx(areas[8], rel=1e-8)
    assert areas[1] == pytest.approx(areas[3], rel=1e-8)
    assert areas[1] == pytest.approx(areas[5], rel=1e-8)
    assert areas[1] == pytest.approx(areas[7], rel=1e-8)


def test__edges_transformed(mask_2d_7x7):

    grid = aa.Grid2D.no_mask(
        values=[
            [-1.5, -1.5],
            [-1.5, 0.0],
            [-1.5, 1.5],
            [0.0, -1.5],
            [0.0, 0.0],
            [0.0, 1.5],
            [1.5, -1.5],
            [1.5, 0.0],
            [1.5, 1.5],
        ],
        pixel_scales=1.5,
        shape_native=(3, 3),
        over_sample_size=1,
    )

    mesh_grid = overlay_grid_from(shape_native=(3, 3), grid=grid, buffer=1e-8)

    mesh = aa.mesh.RectangularAdaptDensity(shape=(3, 3))

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=mesh_grid,
    )

    mapper = aa.Mapper(interpolator=interpolator)

    assert interpolator.mesh_geometry.edges_transformed[3] == pytest.approx(
        np.array(
            [-1.5, 1.5],  # left
        ),
        abs=1e-8,
    )


def test__edges_transformed__aligned_with_interpolation_node_convention():
    """
    Regression test for issue #372: the pcolormesh plotting path draws value
    (row r, col c) inside the cell bounded by edges_transformed — that cell
    must be centred on the interpolation mapper's node for that value, not on
    a uniform [0, 1] partition (which shifted plots by ~1.5 mesh pixels).

    A delta function scattered through the mapper must land in plotted cells
    whose weighted centroid matches the input point to sub-cell precision.
    """
    from autoarray.inversion.mesh.interpolator.rectangular import (
        adaptive_rectangular_mappings_weights_via_interpolation_from,
        adaptive_rectangular_transformed_grid_from,
    )

    n = 10
    rng = np.random.default_rng(0)
    data_grid = rng.uniform(-1.0, 1.0, (5000, 2))
    test_point = np.array([[0.3, -0.2]])

    flat_indices, weights = (
        adaptive_rectangular_mappings_weights_via_interpolation_from(
            source_grid_size=n,
            data_grid=data_grid,
            data_grid_over_sampled=test_point,
        )
    )

    # The node-midpoint unit-space edges (what edges_transformed now builds),
    # pushed through the same CDF transform.
    rows = np.arange(n + 1)
    edges_y = (n - rows - 0.5) / (n - 3)
    edges_x = (rows - 1.5) / (n - 3)
    edges = np.stack([edges_y, edges_x]).T
    edges_t = adaptive_rectangular_transformed_grid_from(
        data_grid, edges, mesh_pixels=n
    )
    y_edges, x_edges = edges_t.T

    centroid_y = 0.0
    centroid_x = 0.0
    for flat, weight in zip(flat_indices[0], weights[0]):
        r, c = flat // n, flat % n
        centroid_y += weight * 0.5 * (y_edges[r] + y_edges[r + 1])
        centroid_x += weight * 0.5 * (x_edges[c] + x_edges[c + 1])

    # Half a mesh cell in these units is ~0.15; the pre-fix uniform edges
    # missed by ~0.4 in y.
    assert centroid_y == pytest.approx(test_point[0, 0], abs=0.1)
    assert centroid_x == pytest.approx(test_point[0, 1], abs=0.1)
