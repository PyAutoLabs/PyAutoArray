import autoarray as aa
import numpy as np
import pytest


def test__data_vector_via_transformed_mapping_matrix_from():
    mapping_matrix = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    data_real = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map_real = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector_real_via_blurred = (
        aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=mapping_matrix,
            image=data_real,
            noise_map=noise_map_real,
        )
    )

    data_imag = np.array([4.0, 1.0, 1.0, 16.0, 1.0, 1.0])
    noise_map_imag = np.array([2.0, 1.0, 1.0, 4.0, 1.0, 1.0])

    data_vector_imag_via_blurred = (
        aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
            blurred_mapping_matrix=mapping_matrix,
            image=data_imag,
            noise_map=noise_map_imag,
        )
    )

    data_vector_complex_via_blurred = (
        data_vector_real_via_blurred + data_vector_imag_via_blurred
    )

    transformed_mapping_matrix = np.array(
        [
            [1.0 + 1.0j, 1.0 + 1.0j, 0.0 + 0.0j],
            [1.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 1.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 1.0j, 1.0 + 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ]
    )

    data = np.array(
        [4.0 + 4.0j, 1.0 + 1.0j, 1.0 + 1.0j, 16.0 + 16.0j, 1.0 + 1.0j, 1.0 + 1.0j]
    )
    noise_map = np.array(
        [2.0 + 2.0j, 1.0 + 1.0j, 1.0 + 1.0j, 4.0 + 4.0j, 1.0 + 1.0j, 1.0 + 1.0j]
    )

    data_vector_via_transformed = aa.util.inversion_interferometer.data_vector_via_transformed_mapping_matrix_from(
        transformed_mapping_matrix=transformed_mapping_matrix,
        visibilities=data,
        noise_map=noise_map,
    )

    assert (data_vector_complex_via_blurred == data_vector_via_transformed).all()


def _dataset_from(mask, n_visibilities, seed):
    """
    Returns a small `TransformerDFT` interferometer dataset on the input mask, with `n_visibilities`
    seeded random visibilities and unit noise, alongside the random generator used to build it.
    """
    rng = np.random.default_rng(seed=seed)

    dataset = aa.Interferometer(
        data=aa.Visibilities(
            visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
        ),
        noise_map=aa.VisibilitiesNoiseMap(
            visibilities=np.ones((n_visibilities, 2), dtype=np.float64)
        ),
        uv_wavelengths=rng.normal(size=(n_visibilities, 2)).astype(np.float64),
        real_space_mask=mask,
        transformer_class=aa.TransformerDFT,
    )

    return dataset, rng


def _mask_7x7():
    return aa.Mask2D(
        mask=[
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        pixel_scales=2.0,
    )


def _sparse_operator_and_mask():
    """
    Returns a real `InterferometerSparseOperator` (and the mask it is defined on) built from a
    small 7x7 `TransformerDFT` interferometer dataset.
    """
    mask = _mask_7x7()

    dataset, rng = _dataset_from(mask=mask, n_visibilities=5, seed=3)

    return dataset.apply_sparse_operator(use_jax=False).sparse_operator, mask, rng


def _operator_dense(operator):
    """
    Returns the dense [M, M] matrix of the `W~` operator, computed by applying it to the identity
    on the extent grid one one-hot column at a time.
    """
    return np.array(operator.apply_operator(np.eye(operator.M)))


def _dense_from_triplets(rows, cols, vals, M, S):
    matrix = np.zeros((M, S))

    for row, col, val in zip(rows, cols, vals):
        matrix[row, col] += val

    return matrix


def test__interferometer_sparse_operator__curvature_matrix_off_diag_from():
    pytest.importorskip("jax")

    operator, mask, rng = _sparse_operator_and_mask()

    M = operator.M

    rows_0 = np.array([0, 1, 4, 4])
    cols_0 = np.array([0, 1, 0, 1])
    vals_0 = np.array([1.0, 2.0, 0.5, 0.25])

    rows_1 = np.array([1, 3, 4, 7])
    cols_1 = np.array([0, 1, 2, 2])
    vals_1 = np.array([0.75, 1.5, 3.0, 0.125])

    off_diag = np.array(
        operator.curvature_matrix_off_diag_from(
            rows0=rows_0,
            cols0=cols_0,
            vals0=vals_0,
            rows1=rows_1,
            cols1=cols_1,
            vals1=vals_1,
            S0=2,
            S1=3,
        )
    )

    matrix_0 = _dense_from_triplets(rows_0, cols_0, vals_0, M=M, S=2)
    matrix_1 = _dense_from_triplets(rows_1, cols_1, vals_1, M=M, S=3)

    off_diag_dense = matrix_0.T @ _operator_dense(operator) @ matrix_1

    assert off_diag.shape == (2, 3)
    assert off_diag == pytest.approx(off_diag_dense, 1.0e-8)


def test__interferometer_sparse_operator__curvature_matrix_off_diag_func_list_from():
    pytest.importorskip("jax")

    operator, mask, rng = _sparse_operator_and_mask()

    M = operator.M
    extent_index_for_masked_pixel = np.array(mask.extent_index_for_masked_pixel)

    rows = np.array([0, 1, 4, 4, 7])
    cols = np.array([0, 1, 0, 1, 1])
    vals = np.array([1.0, 2.0, 0.5, 0.25, 3.0])

    curvature_weights = rng.normal(size=(mask.pixels_in_mask, 3))

    off_diag = np.array(
        operator.curvature_matrix_off_diag_func_list_from(
            curvature_weights=curvature_weights,
            extent_index_for_masked_pixel=extent_index_for_masked_pixel,
            rows=rows,
            cols=cols,
            vals=vals,
            S=2,
        )
    )

    mapping_matrix = _dense_from_triplets(rows, cols, vals, M=M, S=2)

    # The linear function columns are scattered from the slim masked grid onto the extent grid,
    # with no noise weighting applied (the inverse variance lives inside `W~`).
    func_matrix = np.zeros((M, 3))
    func_matrix[extent_index_for_masked_pixel, :] = curvature_weights

    off_diag_dense = mapping_matrix.T @ _operator_dense(operator) @ func_matrix

    assert off_diag.shape == (2, 3)
    assert off_diag == pytest.approx(off_diag_dense, 1.0e-8)


def test__interferometer_sparse_operator__curvature_matrix_func_list_from():
    pytest.importorskip("jax")

    operator, mask, rng = _sparse_operator_and_mask()

    M = operator.M
    extent_index_for_masked_pixel = np.array(mask.extent_index_for_masked_pixel)

    curvature_weights_0 = rng.normal(size=(mask.pixels_in_mask, 2))
    curvature_weights_1 = rng.normal(size=(mask.pixels_in_mask, 3))

    curvature_matrix = np.array(
        operator.curvature_matrix_func_list_from(
            curvature_weights_0=curvature_weights_0,
            curvature_weights_1=curvature_weights_1,
            extent_index_for_masked_pixel=extent_index_for_masked_pixel,
        )
    )

    func_matrix_0 = np.zeros((M, 2))
    func_matrix_0[extent_index_for_masked_pixel, :] = curvature_weights_0

    func_matrix_1 = np.zeros((M, 3))
    func_matrix_1[extent_index_for_masked_pixel, :] = curvature_weights_1

    curvature_matrix_dense = func_matrix_0.T @ _operator_dense(operator) @ func_matrix_1

    assert curvature_matrix.shape == (2, 3)
    assert curvature_matrix == pytest.approx(curvature_matrix_dense, 1.0e-8)


def test__interferometer_sparse_operator__operated_matrix_slim_from():
    pytest.importorskip("jax")

    operator, mask, rng = _sparse_operator_and_mask()

    M = operator.M
    extent_index_for_masked_pixel = np.array(mask.extent_index_for_masked_pixel)

    matrix_slim = rng.normal(size=(mask.pixels_in_mask, 2))

    operated = np.array(
        operator.operated_matrix_slim_from(
            matrix_slim=matrix_slim,
            extent_index_for_masked_pixel=extent_index_for_masked_pixel,
        )
    )

    matrix_extent = np.zeros((M, 2))
    matrix_extent[extent_index_for_masked_pixel, :] = matrix_slim

    operated_dense = (_operator_dense(operator) @ matrix_extent)[
        extent_index_for_masked_pixel, :
    ]

    assert operated.shape == (mask.pixels_in_mask, 2)
    assert operated == pytest.approx(operated_dense, 1.0e-8)


def _apply_operator_via_complex_fft2(preload, operator):
    """
    Returns `W~ @ I` computed with the complex `fft2` / `ifft2` pair, written out in NumPy so that
    it is an independent reference for the `rfft2` / `irfft2` implementation in
    `InterferometerSparseOperator.apply_operator`.
    """
    y_shape, x_shape = operator.y_shape, operator.x_shape
    M = operator.M

    Fbatch_flat = np.eye(M)
    B = Fbatch_flat.shape[1]

    F_img = Fbatch_flat.T.reshape((B, y_shape, x_shape))
    F_pad = np.pad(F_img, ((0, 0), (0, y_shape), (0, x_shape)))

    Khat = np.fft.fft2(preload)
    Ghat = np.fft.fft2(F_pad) * Khat[None, :, :]
    G_pad = np.fft.ifft2(Ghat)
    G = np.real(G_pad[:, :y_shape, :x_shape])

    return G.reshape((B, M)).T


def test__interferometer_sparse_operator__apply_operator__rfft2_matches_complex_fft2_reference():
    pytest.importorskip("jax")

    cases = [
        (_mask_7x7(), 5, 3),
        (
            aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=1.0, radius=4.0),
            64,
            11,
        ),
    ]

    for mask, n_visibilities, seed in cases:
        dataset, _ = _dataset_from(mask=mask, n_visibilities=n_visibilities, seed=seed)

        preload = dataset.psf_precision_operator_from(use_jax=False)
        operator = dataset.apply_sparse_operator(
            nufft_precision_operator=preload
        ).sparse_operator

        # The preload and the batch are both real, so `rfft2` stores only the non-redundant half
        # of the spectrum: (2y, x + 1) rather than (2y, 2x).
        assert operator.Khat.shape == (2 * operator.y_shape, operator.x_shape + 1)

        operated = np.array(operator.apply_operator(np.eye(operator.M)))
        operated_via_complex_fft2 = _apply_operator_via_complex_fft2(
            preload=preload, operator=operator
        )

        # The real transform pair is exact for a real preload and a real batch, so this pin is at
        # round-off, not at an algorithmic tolerance.
        np.testing.assert_allclose(
            operated,
            operated_via_complex_fft2,
            rtol=1.0e-10,
            atol=1.0e-10 * np.abs(operated_via_complex_fft2).max(),
        )
