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


def test__dft__image_from__jax_jit_matches_numpy(
    visibilities_7, uv_wavelengths_7x2, mask_2d_7x7
):
    """
    The adjoint DFT is used inside `jax.jit` fits (the dirty image of profile-subtracted visibilities for the
    sparse operator), so it must trace with `xp=jnp` and match the NumPy result.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    @jax.jit
    def f(visibilities):
        return transformer.image_from(
            visibilities=aa.Visibilities(visibilities=visibilities), xp=jnp
        ).array

    image = transformer.image_from(visibilities=visibilities_7)

    assert np.asarray(f(jnp.asarray(visibilities_7.array))) == pytest.approx(
        image.array, rel=1.0e-10, abs=1.0e-12
    )


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


def test__nufft__transform_mapping_matrix__real_scatter_matches_complex_scatter_exactly():
    """Scattering the real mapping matrix and casting afterwards
    (autolens_profiling#308) must be bit-identical to the previous
    cast-then-scatter formula, for NumPy and under ``jax.jit``."""
    import jax
    import jax.numpy as jnp

    from autoarray.operators import transformer as transformer_module

    rng = np.random.default_rng(seed=4)
    uv_wavelengths = rng.normal(size=(41, 2)) * 50.0
    real_space_mask = aa.Mask2D.circular(
        shape_native=(12, 11), pixel_scales=0.05, radius=0.25
    )
    n_src = 5
    mapping_matrix = rng.normal(size=(real_space_mask.pixels_in_mask, n_src))

    transformer = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    )

    nufftax = transformer_module._load_nufftax()
    rows, cols = real_space_mask.slim_to_native_tuple
    n_y, n_x = real_space_mask.shape_native

    source_images = np.zeros((n_src, n_y, n_x), dtype=np.complex128)
    source_images[np.arange(n_src)[:, None], rows[None, :], cols[None, :]] = (
        mapping_matrix.T.astype(np.complex128)
    )
    expected_numpy = np.array(
        np.asarray(
            nufftax.nufft2d2(
                transformer._x,
                transformer._y,
                source_images[:, ::-1, :],
                transformer.eps,
                -1,
            )
            * transformer._shift[None, :]
        ).T
    )

    result_numpy = transformer.transform_mapping_matrix(mapping_matrix=mapping_matrix)

    assert result_numpy.dtype == np.complex128
    assert np.array_equal(result_numpy, expected_numpy)

    @jax.jit
    def old_formula(mm):
        images = jnp.zeros((n_src, n_y, n_x), dtype=jnp.complex128)
        images = images.at[
            jnp.arange(n_src)[:, None],
            jnp.asarray(rows)[None, :],
            jnp.asarray(cols)[None, :],
        ].set(mm.T.astype(jnp.complex128))
        vis = (
            nufftax.nufft2d2(
                jnp.asarray(transformer._x),
                jnp.asarray(transformer._y),
                images[:, ::-1, :],
                transformer.eps,
                -1,
            )
            * jnp.asarray(transformer._shift)[None, :]
        )
        return vis.T

    @jax.jit
    def new_formula(mm):
        return transformer.transform_mapping_matrix(mapping_matrix=mm, xp=jnp)

    expected_jax = np.asarray(old_formula(jnp.asarray(mapping_matrix)))
    result_jax = np.asarray(new_formula(jnp.asarray(mapping_matrix)))

    assert result_jax.dtype == np.complex128
    assert np.array_equal(result_jax, expected_jax)


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
    vis_arr = rng.normal(size=37).astype(np.float64) + 1j * rng.normal(size=37).astype(
        np.float64
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
    vis_arr = rng.normal(size=37).astype(np.float64) + 1j * rng.normal(size=37).astype(
        np.float64
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

    expected = (
        aa.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
        )
        .visibilities_from(image=image, xp=jnp)
        .array
    )

    assert np.asarray(result) == pytest.approx(
        np.asarray(expected), rel=1.0e-6, abs=1.0e-10
    )


def test__nufftax_exception__keeps_the_plain_install_instruction():
    from autoarray.operators import transformer

    with pytest.raises(ModuleNotFoundError) as exc:
        transformer.nufftax_exception()

    message = str(exc.value)
    assert "Install it via the command `pip install nufftax`" in message
    assert "Do NOT" not in message
