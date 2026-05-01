import autoarray as aa
import numpy as np
import pytest


def test__curvature_matrix_diag_via_psf_weighted_noise_from():
    psf_weighted_noise = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 2.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )

    mapping_matrix = np.array(
        [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )

    curvature_matrix = (
        aa.util.inversion.curvature_matrix_diag_via_psf_weighted_noise_from(
            psf_weighted_noise=psf_weighted_noise, mapping_matrix=mapping_matrix
        )
    )

    assert (
        curvature_matrix
        == np.array([[6.0, 8.0, 0.0], [8.0, 8.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()


def test__curvature_matrix_via_mapping_matrix_from():
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

    noise_map = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
    )

    assert (
        curvature_matrix
        == np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])
    ).all()

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

    noise_map = np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, noise_map=noise_map
    )

    assert (
        curvature_matrix
        == np.array([[1.25, 0.25, 0.0], [0.25, 2.25, 1.0], [0.0, 1.0, 1.0]])
    ).all()


def test__reconstruction_positive_negative_from():
    data_vector = np.array([1.0, 1.0, 2.0])

    curvature_reg_matrix = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])

    reconstruction = aa.util.inversion.reconstruction_positive_negative_from(
        data_vector=data_vector,
        curvature_reg_matrix=curvature_reg_matrix,
    )

    assert reconstruction == pytest.approx(np.array([1.0, -1.0, 3.0]), 1.0e-4)


def test__mapped_reconstructed_data_via_mapping_matrix_from():
    mapping_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_operated_data = (
        aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )
    )

    assert (mapped_reconstructed_operated_data == np.array([1.0, 1.0, 2.0])).all()

    mapping_matrix = np.array([[0.25, 0.50, 0.25], [0.0, 1.0, 0.0], [0.0, 0.25, 0.75]])

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_operated_data = (
        aa.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix, reconstruction=reconstruction
        )
    )

    assert (mapped_reconstructed_operated_data == np.array([1.25, 1.0, 1.75])).all()


def test__mapped_reconstructed_data_via_image_to_pix_unique_from():
    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pix_indexes_for_sub_slim_index_sizes = np.array([1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array([[1.0], [1.0], [1.0]])

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper_numba.data_slim_to_pixelization_unique_from(
        data_pixels=3,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=np.array([1, 1, 1]),
    )

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_operated_data = aa.util.inversion_imaging_numba.mapped_reconstructed_data_via_image_to_pix_unique_from(
        data_to_pix_unique=data_to_pix_unique.astype("int"),
        data_weights=data_weights,
        pix_lengths=pix_lengths.astype("int"),
        reconstruction=reconstruction,
    )

    assert (mapped_reconstructed_operated_data == np.array([1.0, 1.0, 2.0])).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0], [1], [1], [2], [1], [1], [1], [1], [1], [2], [2], [2]]
    )
    pix_indexes_for_sub_slim_index_sizes = np.ones(shape=(12,)).astype("int")
    pix_weights_for_sub_slim_index = np.ones(shape=(12, 1))

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper_numba.data_slim_to_pixelization_unique_from(
        data_pixels=3,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_indexes_for_sub_slim_index_sizes,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=np.array([2, 2, 2]),
    )

    reconstruction = np.array([1.0, 1.0, 2.0])

    mapped_reconstructed_operated_data = aa.util.inversion_imaging_numba.mapped_reconstructed_data_via_image_to_pix_unique_from(
        data_to_pix_unique=data_to_pix_unique.astype("int"),
        data_weights=data_weights,
        pix_lengths=pix_lengths.astype("int"),
        reconstruction=reconstruction,
    )

    assert (mapped_reconstructed_operated_data == np.array([1.25, 1.0, 1.75])).all()


def test__preconditioner_matrix_via_mapping_matrix_from():
    mapping_matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=1.0,
            regularization_matrix=np.zeros((3, 3)),
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    ).all()

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=np.zeros((3, 3)),
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    ).all()

    regularization_matrix = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    preconditioner_matrix = (
        aa.util.inversion.preconditioner_matrix_via_mapping_matrix_from(
            mapping_matrix=mapping_matrix,
            preconditioner_noise_normalization=2.0,
            regularization_matrix=regularization_matrix,
        )
    )

    assert (
        preconditioner_matrix
        == np.array([[5.0, 2.0, 3.0], [4.0, 9.0, 6.0], [7.0, 8.0, 13.0]])
    ).all()
