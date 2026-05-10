import autoarray as aa

import numpy as np
import pytest


def test__dft__visibilities_from__image_with_mixed_values__first_three_visibilities_match_expected(
    visibilities_7, uv_wavelengths_7x2, mask_2d_7x7
):

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    image = aa.Array2D(
        values=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask_2d_7x7,
    )

    visibilities = transformer.visibilities_from(image=image)

    assert visibilities[0:3] == pytest.approx(
        np.array(
            [
                -0.06434514 - 0.61763293j,
                1.71143349 - 1.184022j,
                0.90200541 + 0.03726693j,
            ]
        ),
        1.0e-4,
    )


def test__dft__image_from__visibilities_7__first_three_image_pixels_match_expected(
    visibilities_7, uv_wavelengths_7x2, mask_2d_7x7
):

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    image = transformer.image_from(visibilities=visibilities_7)

    assert image[0:3] == pytest.approx([-1.49022481, -0.22395855, -0.45588535], 1.0e-4)


def test__nufft__visibilities_from__all_ones_image__first_visibility_matches_expected():

    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
    real_space_mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)

    image = aa.Array2D.ones(
        shape_native=(5, 5),
        pixel_scales=0.005,
    )

    transformer_nufft = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    )

    visibilities_nufft = transformer_nufft.visibilities_from(image=image.native)

    # nufftax-backed forward NUFFT: matches the analytic DFT to machine precision.
    # For an all-ones image the visibility at any uv is N_y * N_x = 25.
    assert visibilities_nufft[0] == pytest.approx(25.0 + 0.0j, 1.0e-7)


def test__nufft_pynufft__visibilities_from__all_ones_image__first_visibility_matches_expected():

    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
    real_space_mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)

    image = aa.Array2D.ones(
        shape_native=(5, 5),
        pixel_scales=0.005,
    )

    transformer_nufft = aa.TransformerNUFFTPyNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    )

    visibilities_nufft = transformer_nufft.visibilities_from(image=image.native)

    # Legacy pynufft has a small gridding-kernel error at N=5; expected value
    # encodes that error and is retained for backwards compatibility.
    assert visibilities_nufft[0] == pytest.approx(25.02317617953263 + 0.0j, 1.0e-7)


def test__nufft__image_from__visibilities_7__first_three_image_pixels_match_expected(
    visibilities_7, uv_wavelengths_7x2, mask_2d_7x7
):

    transformer = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    image = transformer.image_from(visibilities=visibilities_7)

    # nufftax adjoint matches `TransformerDFT.image_from` exactly (no kernel
    # deconvolution applied; this is the mathematical adjoint of the forward).
    assert image[0:3] == pytest.approx([-1.49022481, -0.22395855, -0.45588535], 1.0e-4)


def test__nufft_pynufft__image_from__visibilities_7__first_three_image_pixels_match_expected(
    visibilities_7, uv_wavelengths_7x2, mask_2d_7x7
):

    transformer = aa.TransformerNUFFTPyNUFFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    image = transformer.image_from(visibilities=visibilities_7)

    # Legacy pynufft adjoint includes internal kernel deconvolution and IFFT
    # normalisation; expected values reflect that behaviour.
    assert image[0:3] == pytest.approx([0.00726546, 0.01149121, 0.01421022], 1.0e-4)


def test__nufft__transform_mapping_matrix__ones_mapping_matrix__first_element_matches_expected():
    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

    mapping_matrix = np.ones(shape=(25, 3))

    real_space_mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)

    transformer_nufft = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    )

    transformed_mapping_matrix_nufft = transformer_nufft.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    # nufftax-backed forward over a mapping matrix column reduces to the
    # all-ones forward NUFFT case; equals N_y * N_x = 25 exactly.
    assert transformed_mapping_matrix_nufft[0, 0] == pytest.approx(25.0 + 0.0j, 1.0e-4)


def test__nufft_pynufft__transform_mapping_matrix__ones_mapping_matrix__first_element_matches_expected():
    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

    mapping_matrix = np.ones(shape=(25, 3))

    real_space_mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)

    transformer_nufft = aa.TransformerNUFFTPyNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    )

    transformed_mapping_matrix_nufft = transformer_nufft.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    assert transformed_mapping_matrix_nufft[0, 0] == pytest.approx(
        25.02317 + 0.0j, 1.0e-4
    )
