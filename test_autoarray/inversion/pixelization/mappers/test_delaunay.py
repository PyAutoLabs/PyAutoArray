import numpy as np
import scipy.spatial

import autoarray as aa

from autoarray.inversion.mesh.interpolator.delaunay import (
    pix_indexes_for_sub_slim_index_delaunay_from,
)


from autoarray.inversion.mesh.interpolator.delaunay import (
    pixel_weights_delaunay_from,
)


def test__pixel_weights_delaunay_from():
    data_grid = np.array([[0.1, 0.1], [1.0, 1.0]])

    mesh_grid = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])

    pix_indexes_for_sub_slim_index = np.array([[0, 1, 2], [2, -1, -1]])

    pixel_weights = pixel_weights_delaunay_from(
        data_grid=data_grid,
        mesh_grid=mesh_grid,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    )

    assert (pixel_weights == np.array([[0.25, 0.5, 0.25], [1.0, 0.0, 0.0]])).all()


def test__pix_indexes_for_sub_slim_index__matches_util(grid_2d_sub_1_7x7):
    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh = aa.mesh.Delaunay(pixels=6)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid_2d_sub_1_7x7,
        source_plane_mesh_grid=mesh_grid,
    )

    mapper = aa.Mapper(interpolator=interpolator)

    delaunay = scipy.spatial.Delaunay(mapper.interpolator.mesh_grid_xy)

    simplex_index_for_sub_slim_index = delaunay.find_simplex(
        mapper.source_plane_data_grid
    )

    delaunay = mapper.interpolator.delaunay

    pix_indexes_for_simplex_index = delaunay.simplices

    pix_indexes_for_sub_slim_index_util = pix_indexes_for_sub_slim_index_delaunay_from(
        data_grid=mapper.source_plane_data_grid.array,
        simplex_index_for_sub_slim_index=simplex_index_for_sub_slim_index,
        pix_indexes_for_simplex_index=pix_indexes_for_simplex_index,
        delaunay_points=delaunay.points,
    )
    sizes = (
        np.sum(pix_indexes_for_sub_slim_index_util >= 0, axis=1)
        .astype(np.int32)
        .astype("int")
    )

    assert (
        mapper.pix_indexes_for_sub_slim_index == pix_indexes_for_sub_slim_index_util
    ).all()
    assert (mapper.pix_sizes_for_sub_slim_index == sizes).all()

    assert (
        mapper.pix_indexes_for_sub_slim_index
        == np.array(
            [
                [0, -1, -1],
                [1, -1, -1],
                [1, 5, 3],
                [0, -1, -1],
                [0, -1, -1],
                [3, -1, -1],
                [0, -1, -1],
                [0, -1, -1],
                [3, -1, -1],
            ]
        )
    ).all()

    assert (
        mapper.pix_sizes_for_sub_slim_index == np.array([1, 1, 3, 1, 1, 1, 1, 1, 1])
    ).all()
