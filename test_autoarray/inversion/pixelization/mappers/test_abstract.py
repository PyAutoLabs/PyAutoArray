import numpy as np
import pytest

import autoarray as aa


def test__pix_indexes_for_slim_indexes__different_types_of_lists_input():
    mapper = aa.m.MockMapper(
        interpolator=aa.m.MockInterpolator(
            mappings=np.array([[0], [0], [0], [0], [0], [0], [0], [0]]),
            sizes=np.ones(8, dtype="int"),
            weights=np.ones(9),
        ),
        parameters=9,
    )

    pixe_indexes_for_slim_indexes = mapper.slim_indexes_for_pix_indexes(
        pix_indexes=[0, 1]
    )

    assert pixe_indexes_for_slim_indexes == [0, 1, 2, 3, 4, 5, 6, 7]

    mapper = aa.m.MockMapper(
        interpolator=aa.m.MockInterpolator(
            mappings=np.array([[0], [0], [0], [0], [3], [4], [4], [7]]),
            sizes=np.ones(8, dtype="int"),
            weights=np.ones(8),
        ),
        parameters=9,
    )

    pixe_indexes_for_slim_indexes = mapper.slim_indexes_for_pix_indexes(
        pix_indexes=[[0], [4]]
    )

    assert pixe_indexes_for_slim_indexes == [[0, 1, 2, 3], [5, 6]]


def test__sub_slim_indexes_for_pix_index():
    mapper = aa.m.MockMapper(
        interpolator=aa.m.MockInterpolator(
            mappings=np.array(
                [[0, 4], [1, 4], [2, 4], [0, 4], [1, 4], [3, 4], [0, 4], [3, 4]]
            ).astype("int"),
            sizes=np.ones(8).astype("int"),
            weights=np.array(
                [
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.4, 0.6],
                    [0.5, 0.5],
                    [0.6, 0.4],
                    [0.7, 0.3],
                    [0.8, 0.2],
                ]
            ),
        ),
        parameters=5,
    )

    assert mapper.sub_slim_indexes_for_pix_index == [
        [0, 3, 6],
        [1, 4],
        [2],
        [5, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]


def test__data_weight_total_for_pix_from():
    mapper = aa.m.MockMapper(
        interpolator=aa.m.MockInterpolator(
            mappings=np.array(
                [[0, 4], [1, 4], [2, 4], [0, 4], [1, 4], [3, 4], [0, 4], [3, 4]]
            ).astype("int"),
            sizes=np.ones(8).astype("int"),
            weights=np.array(
                [
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.4, 0.6],
                    [0.5, 0.5],
                    [0.6, 0.4],
                    [0.7, 0.3],
                    [0.8, 0.2],
                ]
            ),
        ),
        parameters=5,
    )

    data_weight_total_for_pix = mapper.data_weight_total_for_pix_from()

    assert data_weight_total_for_pix == pytest.approx([1.2, 0.7, 0.3, 1.4, 4.4], 1.0e-4)


def test__adaptive_pixel_signals_from___matches_util(grid_2d_7x7, image_7x7):
    pixels = 6
    signal_scale = 2.0
    interpolator = aa.m.MockInterpolator(
        mappings=np.array([[1], [1], [4], [0], [0], [3], [0], [0], [3]]),
        sizes=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
        weights=np.ones(9),
    )
    pix_weights_for_sub_slim_index = np.ones((9, 1), dtype="int")

    over_sampler = aa.OverSampler(mask=grid_2d_7x7.mask, sub_size=1)

    mapper = aa.m.MockMapper(
        source_plane_data_grid=grid_2d_7x7,
        over_sampler=over_sampler,
        interpolator=interpolator,
        adapt_data=image_7x7,
        parameters=pixels,
    )

    pixel_signals = mapper.pixel_signals_from(signal_scale=2.0)

    pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=pixels,
        pixel_weights=pix_weights_for_sub_slim_index,
        signal_scale=signal_scale,
        pix_indexes_for_sub_slim_index=interpolator.mappings,
        pix_size_for_sub_slim_index=interpolator.sizes,
        slim_index_for_sub_slim_index=over_sampler.slim_for_sub_slim,
        adapt_data=np.array(image_7x7),
    )

    assert (pixel_signals == pixel_signals_util).all()


def test__mapped_to_source_from(grid_2d_7x7):
    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh = aa.mesh.Delaunay(pixels=6)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid_2d_7x7,
        source_plane_mesh_grid=mesh_grid,
    )

    mapper = aa.Mapper(interpolator=interpolator)

    array_slim = aa.Array2D.no_mask(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        shape_native=(3, 3),
        pixel_scales=1.0,
    )

    mapped_to_source_util = aa.util.mapper.mapped_to_source_via_mapping_matrix_from(
        mapping_matrix=mapper.mapping_matrix,
        array_slim=array_slim,
    )

    mapped_to_source_mapper = mapper.mapped_to_source_from(array=array_slim)

    assert np.allclose(mapped_to_source_util, mapped_to_source_mapper)
