import autoarray as aa
import numpy as np
import pytest


def test__psf_weighted_noise_imaging_from():
    noise_map = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    native_index_for_slim_index = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    psf_weighted_noise = aa.util.inversion_imaging_numba.psf_precision_operator_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert psf_weighted_noise == pytest.approx(
        np.array(
            [
                [2.5, 1.625, 0.5, 0.375],
                [1.625, 1.3125, 0.125, 0.0625],
                [0.5, 0.125, 0.5, 0.375],
                [0.375, 0.0625, 0.375, 0.3125],
            ]
        ),
        1.0e-4,
    )


def test__psf_weighted_data_from():

    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    data = aa.Array2D(
        values=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask,
    )

    noise_map = aa.Array2D(
        values=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask,
    )

    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 0.0]])

    native_index_for_slim_index = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    weight_map = data / (noise_map**2)
    weight_map = aa.Array2D(values=weight_map, mask=mask)

    psf_weighted_data = aa.util.inversion_imaging.psf_weighted_data_from(
        weight_map_native=weight_map.native.array,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert (psf_weighted_data == np.array([5.0, 5.0, 1.5, 1.5])).all()


def test__psf_precision_operator_sparse_from():
    noise_map = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 0.0],
            [0.0, 2.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    kernel = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]])

    native_index_for_slim_index = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

    (
        psf_weighted_noise_preload,
        psf_weighted_noise_indexes,
        psf_weighted_noise_lengths,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert psf_weighted_noise_preload == pytest.approx(
        np.array(
            [1.25, 1.625, 0.5, 0.375, 0.65625, 0.125, 0.0625, 0.25, 0.375, 0.15625]
        ),
        1.0e-4,
    )
    assert psf_weighted_noise_indexes == pytest.approx(
        np.array([0, 1, 2, 3, 1, 2, 3, 2, 3, 3]), 1.0e-4
    )

    assert psf_weighted_noise_lengths == pytest.approx(np.array([4, 3, 2, 1]), 1.0e-4)


def test__data_vector_via_blurred_mapping_matrix_from():
    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    image = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    data_vector = aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix, image=image, noise_map=noise_map
    )

    assert (data_vector == np.array([2.0, 3.0, 1.0])).all()

    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    image = np.array([3.0, 1.0, 1.0, 10.0, 1.0, 1.0])
    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    data_vector = aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix, image=image, noise_map=noise_map
    )

    assert (data_vector == np.array([4.0, 14.0, 10.0])).all()

    blurred_mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    image = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector = aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix, image=image, noise_map=noise_map
    )

    assert (data_vector == np.array([2.0, 3.0, 1.0])).all()
