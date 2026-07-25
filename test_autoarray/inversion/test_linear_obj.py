import autoarray as aa
import numpy as np


def test__data_to_pix_unique_from():
    image_pixels = 2
    sub_size = np.array([2, 2])

    pix_index_for_sub_slim_index = np.array(
        [[0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1], [0, -1]]
    ).astype("int")
    pix_sizes_for_sub_slim_index = np.array([1, 1, 1, 1, 1, 1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array(
        [
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
        ]
    )

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper_numba.data_slim_to_pixelization_unique_from(
        data_pixels=image_pixels,
        pix_indexes_for_sub_slim_index=pix_index_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_sizes_for_sub_slim_index,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=2,
        sub_size=sub_size,
    )

    assert (data_to_pix_unique[0, :] == np.array([0, -1, -1, -1])).all()
    assert (data_to_pix_unique[1, :] == np.array([0, -1, -1, -1])).all()
    assert (data_weights[0, :] == np.array([1.0, 0.0, 0.0, 0.0])).all()
    assert (data_weights[1, :] == np.array([1.0, 0.0, 0.0, 0.0])).all()
    assert (pix_lengths == np.array([1, 1])).all()

    mask = aa.Mask2D.all_false(shape_native=(1, 2), pixel_scales=0.1)

    grid = aa.Grid2D.uniform(shape_native=(1, 2), pixel_scales=0.1, over_sample_size=2)

    linear_obj = aa.AbstractLinearObjFuncList(grid=grid, regularization=None)

    assert (linear_obj.unique_mappings.data_to_pix_unique == data_to_pix_unique).all()
    assert (linear_obj.unique_mappings.data_weights == data_weights).all()
    assert (linear_obj.unique_mappings.pix_lengths == pix_lengths).all()


def test__neighbors():
    class FuncList(aa.AbstractLinearObjFuncList):
        @property
        def params(self):
            return 4

    linear_obj = FuncList(grid=None, regularization=None)

    neighbors = linear_obj.neighbors

    assert (neighbors[0] == [1, -1]).all()
    assert (neighbors[1] == [0, 2]).all()
    assert (neighbors[2] == [1, 3]).all()
    assert (neighbors[3] == [2, -1]).all()

    assert neighbors.sizes[0] == 1
    assert neighbors.sizes[1] == 2
    assert neighbors.sizes[2] == 2
    assert neighbors.sizes[3] == 1
