import numpy as np

import autoarray as aa
from autoarray.inversion.mesh.interpolator.delaunay import (
    pixel_weights_delaunay_from,
    scipy_delaunay,
)
from autoarray.inversion.mesh.interpolator.sibson import (
    InterpolatorDelaunayNN,
    scipy_delaunay_nn,
)


def test__mesh_is_public_and_selects_natural_neighbor_interpolator():
    mesh = aa.mesh.DelaunayNN(
        pixels=100,
        zeroed_pixels=8,
        areas_factor=0.4,
    )

    assert mesh.pixels == 108
    np.testing.assert_array_equal(mesh.zeroed_pixels, np.arange(100, 108))
    assert mesh.areas_factor == 0.4
    assert mesh.max_cavity_triangles == 32
    assert mesh.max_neighbors == 32
    assert mesh.query_chunk == 256
    assert mesh.interpolator_cls is InterpolatorDelaunayNN
    assert aa.InterpolatorDelaunayNN is InterpolatorDelaunayNN


def test__interpolator__partition_of_unity_linear_precision_and_split_padding(
    grid_2d_sub_1_7x7,
):
    mesh_grid = aa.Grid2DIrregular(
        values=[
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-0.25, 0.1],
            [0.35, -0.2],
        ]
    )
    interpolator = aa.InterpolatorDelaunayNN(
        mesh=aa.mesh.DelaunayNN(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=grid_2d_sub_1_7x7,
    )

    mappings, sizes, weights = interpolator._mappings_sizes_weights
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1.0e-12)
    assert mappings.shape == (grid_2d_sub_1_7x7.over_sampled.shape[0], 32)
    assert sizes.max() <= 6
    assert not interpolator.delaunay.overflow.any()
    assert not interpolator.delaunay.degenerate.any()

    inside = interpolator.delaunay.cavity_sizes > 0
    reconstructed_coordinates = np.sum(
        mesh_grid.array[mappings.clip(min=0)] * weights[..., None], axis=1
    )
    np.testing.assert_allclose(
        reconstructed_coordinates[inside],
        grid_2d_sub_1_7x7.over_sampled.array[inside],
        atol=1.0e-11,
    )

    split_mappings, split_sizes, split_weights = (
        interpolator._mappings_sizes_weights_split
    )
    assert split_mappings.shape == (4 * mesh_grid.shape[0], 33)
    assert split_weights.shape == split_mappings.shape
    np.testing.assert_array_equal(split_mappings[:, -1], -1)
    np.testing.assert_array_equal(split_weights[:, -1], 0.0)
    np.testing.assert_array_equal(split_sizes, interpolator.delaunay.splitted_sizes)
    np.testing.assert_allclose(split_weights[:, :-1].sum(axis=1), 1.0, atol=1.0e-12)
    assert not interpolator.delaunay.split_overflow.any()
    assert not interpolator.delaunay.split_degenerate.any()


def test__smooth_source_mapping_is_numerically_close_to_delaunay():
    rng = np.random.default_rng(4)
    axis = np.linspace(-1.0, 1.0, 12)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([y.ravel(), x.ravel()], axis=1)
    points += rng.normal(0.0, 0.025, size=points.shape)

    query_axis = np.linspace(-0.9, 0.9, 45)
    query_y, query_x = np.meshgrid(query_axis, query_axis, indexing="ij")
    query = np.stack([query_y.ravel(), query_x.ravel()], axis=1)

    _, _, delaunay_mappings, _, _ = scipy_delaunay(points, query, areas_factor=0.5)
    delaunay_weights = pixel_weights_delaunay_from(
        data_grid=query,
        mesh_grid=points,
        pix_indexes_for_sub_slim_index=delaunay_mappings,
    )
    delaunay_nn = scipy_delaunay_nn(points, query, areas_factor=0.5)
    nn_mappings, nn_weights = delaunay_nn[2], delaunay_nn[4]

    source = np.exp(-((points[:, 0] - 0.12) ** 2 + (points[:, 1] + 0.08) ** 2) / 0.22)
    delaunay_image = np.sum(
        source[delaunay_mappings.clip(min=0)] * delaunay_weights, axis=1
    )
    nn_image = np.sum(source[nn_mappings.clip(min=0)] * nn_weights, axis=1)

    relative_l2 = np.linalg.norm(nn_image - delaunay_image) / np.linalg.norm(
        delaunay_image
    )
    correlation = np.corrcoef(nn_image, delaunay_image)[0, 1]

    assert relative_l2 < 0.02
    assert correlation > 0.999
    assert not delaunay_nn[10].any()
    assert not delaunay_nn[11].any()
    assert not delaunay_nn[13].any()
    assert not delaunay_nn[14].any()
