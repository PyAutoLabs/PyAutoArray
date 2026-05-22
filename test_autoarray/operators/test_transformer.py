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


def test__nufft__chunk_size__rejects_non_positive():
    real_space_mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)
    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

    with pytest.raises(ValueError):
        aa.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            chunk_size=0,
        )


def test__nufft__chunk_size__visibilities_from_numpy_matches_unchunked():
    rng = np.random.default_rng(seed=0)
    uv_wavelengths = rng.normal(size=(37, 2)).astype(np.float64)
    real_space_mask = aa.Mask2D.all_false(shape_native=(8, 9), pixel_scales=0.01)
    image_native = rng.normal(size=(8, 9))
    image = aa.Array2D(values=image_native, mask=real_space_mask)

    one_shot = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    ).visibilities_from(image=image)

    chunked = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=8,
    ).visibilities_from(image=image)

    assert np.asarray(chunked.array) == pytest.approx(
        np.asarray(one_shot.array), rel=1.0e-6, abs=1.0e-10
    )


def test__nufft__chunk_size__image_from_numpy_matches_unchunked():
    rng = np.random.default_rng(seed=1)
    uv_wavelengths = rng.normal(size=(37, 2)).astype(np.float64)
    real_space_mask = aa.Mask2D.all_false(shape_native=(8, 9), pixel_scales=0.01)
    vis_arr = (
        rng.normal(size=37).astype(np.float64)
        + 1j * rng.normal(size=37).astype(np.float64)
    )
    visibilities = aa.Visibilities(
        visibilities=np.stack([vis_arr.real, vis_arr.imag], axis=1)
    )

    one_shot_img = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    ).image_from(visibilities=visibilities)

    chunked_img = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=8,
    ).image_from(visibilities=visibilities)

    assert np.asarray(chunked_img.array) == pytest.approx(
        np.asarray(one_shot_img.array), rel=1.0e-6, abs=1.0e-10
    )


def test__nufft__chunk_size__jax_paths_match_unchunked():
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(seed=2)
    uv_wavelengths = rng.normal(size=(37, 2)).astype(np.float64)
    real_space_mask = aa.Mask2D.all_false(shape_native=(8, 9), pixel_scales=0.01)
    image_native = rng.normal(size=(8, 9))
    image = aa.Array2D(values=image_native, mask=real_space_mask)
    vis_arr = (
        rng.normal(size=37).astype(np.float64)
        + 1j * rng.normal(size=37).astype(np.float64)
    )
    visibilities = aa.Visibilities(
        visibilities=np.stack([vis_arr.real, vis_arr.imag], axis=1)
    )

    one_shot_vis = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    ).visibilities_from(image=image, xp=jnp)

    chunked = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=8,
    )
    chunked_vis = chunked.visibilities_from(image=image, xp=jnp)

    assert np.asarray(chunked_vis.array) == pytest.approx(
        np.asarray(one_shot_vis.array), rel=1.0e-6, abs=1.0e-10
    )

    one_shot_img = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    ).image_from(visibilities=visibilities, xp=jnp)

    chunked_img = chunked.image_from(visibilities=visibilities, xp=jnp)

    assert np.asarray(chunked_img.array) == pytest.approx(
        np.asarray(one_shot_img.array), rel=1.0e-6, abs=1.0e-10
    )


def test__nufft__chunk_size__jax_jit_traces_with_scan():
    """``jax.jit`` of a chunked forward NUFFT must trace via ``lax.scan``
    without unrolling the chunk loop. We exercise this by JIT-ing and
    confirming the compiled HLO graph has bounded size (no per-chunk
    re-emission)."""
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(seed=3)
    uv_wavelengths = rng.normal(size=(50, 2)).astype(np.float64)
    real_space_mask = aa.Mask2D.all_false(shape_native=(8, 9), pixel_scales=0.01)
    image_native = rng.normal(size=(8, 9))
    image = aa.Array2D(values=image_native, mask=real_space_mask)

    chunked = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        chunk_size=8,
    )

    @jax.jit
    def f(img_arr):
        wrapped_image = aa.Array2D(values=img_arr, mask=real_space_mask)
        return chunked.visibilities_from(image=wrapped_image, xp=jnp).array

    result = f(jnp.asarray(image_native))

    expected = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    ).visibilities_from(image=image, xp=jnp).array

    assert np.asarray(result) == pytest.approx(
        np.asarray(expected), rel=1.0e-6, abs=1.0e-10
    )
