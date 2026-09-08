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


def _dataset_from(mask, n_visibilities, seed, uv_scale=1.0, vary_noise=False):
    """
    Returns a small `TransformerDFT` interferometer dataset on the input mask, with `n_visibilities`
    seeded random visibilities, alongside the random generator used to build it.

    `uv_scale` multiplies the `(u, v)` baselines. At the default `1.0` the seeded normal baselines
    are of order a wavelength, so `2 pi u delta_rad` is ~1e-4 and every phase in the precision
    operator is near zero — fine for a shape or plumbing test, useless as a numerical pin. A scale
    near `0.5 / delta_rad` (~1e5 for arcsecond pixels) puts the phases in `[-pi, pi]`, which is
    where the operator's entries actually vary and where a wrong sign convention is visible.

    `vary_noise` gives each visibility its own sigma, so the weights `w = 1 / sigma^2` are not all
    equal and a builder that dropped them would be caught.
    """
    rng = np.random.default_rng(seed=seed)

    if vary_noise:
        sigma = rng.uniform(0.5, 1.5, size=n_visibilities).astype(np.float64)
        noise_map = np.stack([sigma, sigma], axis=1)
    else:
        noise_map = np.ones((n_visibilities, 2), dtype=np.float64)

    dataset = aa.Interferometer(
        data=aa.Visibilities(
            visibilities=rng.normal(size=(n_visibilities, 2)).astype(np.float64)
        ),
        noise_map=aa.VisibilitiesNoiseMap(visibilities=noise_map),
        uv_wavelengths=(
            uv_scale * rng.uniform(-1.0, 1.0, size=(n_visibilities, 2))
            if uv_scale != 1.0
            else rng.normal(size=(n_visibilities, 2))
        ).astype(np.float64),
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


def _preload_inputs_from(dataset):
    """
    Returns the four arguments `Interferometer.psf_precision_operator_from` passes to every NUFFT
    precision operator builder, read off a dataset.

    The pins below compare builders, so they must be handed exactly the arrays the library hands
    them; deriving any of the four differently here would let a convention drift in between the
    implementations the pins exist to hold together.
    """
    mask = dataset.transformer.grid.mask

    return {
        "noise_map_real": np.asarray(dataset.noise_map.array.real, dtype=np.float64),
        "uv_wavelengths": np.asarray(dataset.uv_wavelengths, dtype=np.float64),
        "shape_masked_pixels_2d": mask.shape_native_masked_pixels,
        "grid_radians_2d": np.asarray(
            mask.derive_grid.all_false.in_radians.native.array, dtype=np.float64
        ),
    }


def _nufft_pin_inputs_7x7():
    """
    The shared 7x7 / K=5 fixture the rest of this module's operator tests are built from.
    """
    dataset, _ = _dataset_from(mask=_mask_7x7(), n_visibilities=5, seed=3)

    return _preload_inputs_from(dataset)


def _nufft_pin_inputs_16x16():
    """
    A seeded 16x16 circular mask with K=300, varying noise sigmas and baselines scaled so the
    phases span `[-pi, pi]`.

    The 7x7 fixture alone is a weak pin: its five normal baselines give phases of order 1e-4, so
    every entry of the operator sits within 1e-8 of the peak and a wrong sign convention would be
    almost invisible. On this fixture the entries vary over the full range and the wrong-sign
    control below is off by 18% of the peak.
    """
    dataset, _ = _dataset_from(
        mask=aa.Mask2D.circular(shape_native=(16, 16), pixel_scales=1.0, radius=6.0),
        n_visibilities=300,
        seed=7,
        uv_scale=1.0e5,
        vary_noise=True,
    )

    return _preload_inputs_from(dataset)


def _assert_matches_brute_force(operator, operator_via_np):
    """
    Asserts that a NUFFT-built precision operator matches the NumPy brute-force builder.

    The tolerance is deliberately **mixed**. A type-1 NUFFT bounds its error against `sum_k |c_k|`,
    i.e. peak-scaled, not elementwise-relative: the near-zero entries of the operator sit orders
    below its peak and carry no relative accuracy guarantee at all, so a pure `rtol` pin on them
    would be measuring fp64 round-off rather than the builder. `atol = 1e-10 * P[0, 0]` puts a
    floor under exactly those entries while every entry carrying signal stays under a full
    relative test. Never `rtol` with `atol = 0`.
    """
    np.testing.assert_allclose(
        operator,
        operator_via_np,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(operator_via_np[0, 0]),
    )


def test__nufft_precision_operator_via_nufft__matches_the_numpy_brute_force():
    pytest.importorskip("nufftax")

    for inputs in (_nufft_pin_inputs_7x7(), _nufft_pin_inputs_16x16()):
        operator_via_np = np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_via_np_from(
                **inputs
            )
        )
        operator_via_nufft = np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_via_nufft_from(
                **inputs
            )
        )

        assert operator_via_nufft.shape == operator_via_np.shape
        assert operator_via_nufft.dtype == np.float64

        _assert_matches_brute_force(operator_via_nufft, operator_via_np)


def test__nufft_precision_operator_via_nufft__nyquist_row_and_column_are_zero():
    pytest.importorskip("nufftax")

    for inputs in (_nufft_pin_inputs_7x7(), _nufft_pin_inputs_16x16()):
        y_shape, x_shape = (int(s) for s in inputs["shape_masked_pixels_2d"])

        operator = np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_via_nufft_from(
                **inputs
            )
        )

        # After `ifftshift`, index `y_shape` / `x_shape` carries the Nyquist mode, which the brute
        # force never evaluates (its quadrants span offsets -(N-1) ... N-1). The NUFFT does return
        # a value there, so the builder must zero it explicitly -- exactly, not approximately.
        assert (operator[y_shape, :] == 0.0).all()
        assert (operator[:, x_shape] == 0.0).all()

        # The neighbours are not zero, so the assertion above is testing the padding and not an
        # operator that came back empty.
        assert np.abs(operator[y_shape - 1, :]).max() > 0.0
        assert np.abs(operator[y_shape + 1, :]).max() > 0.0
        assert np.abs(operator[:, x_shape - 1]).max() > 0.0
        assert np.abs(operator[:, x_shape + 1]).max() > 0.0


def test__nufft_precision_operator_via_nufft__is_even_under_negated_offsets():
    pytest.importorskip("nufftax")

    for inputs in (_nufft_pin_inputs_7x7(), _nufft_pin_inputs_16x16()):
        y_shape, x_shape = (int(s) for s in inputs["shape_masked_pixels_2d"])

        operator = np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_via_nufft_from(
                **inputs
            )
        )

        # `P[i, j] = sum_k w_k cos(...)` is even in the offset, so `P[i, j] == P[-i, -j]`. The
        # wraparound ordering means the negated offset is a plain negative index.
        for i in range(-(y_shape - 1), y_shape):
            for j in range(-(x_shape - 1), x_shape):
                assert operator[i, j] == pytest.approx(operator[-i, -j], rel=1.0e-12)


def test__nufft_precision_operator_via_nufft__chunked_matches_one_shot():
    pytest.importorskip("nufftax")

    inputs = _nufft_pin_inputs_16x16()

    operator_via_np = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_np_from(**inputs)
    )
    one_shot = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_nufft_from(
            **inputs
        )
    )
    # K = 300, so a chunk size of 64 spreads five chunks and sums their transforms. Chunking is a
    # memory ceiling, not an approximation -- the transform is linear in the weights -- so the two
    # agree to summation order, well inside the pin the builder is held to.
    chunked = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_nufft_from(
            **inputs, chunk_size=64
        )
    )

    _assert_matches_brute_force(chunked, operator_via_np)

    np.testing.assert_allclose(
        chunked,
        one_shot,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(operator_via_np[0, 0]),
    )


def test__nufft_precision_operator_via_nufft__negated_u_fails_the_pin():
    pytest.importorskip("nufftax")

    inputs = _nufft_pin_inputs_16x16()

    operator_via_np = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_np_from(**inputs)
    )

    inputs_wrong_sign = dict(inputs)
    inputs_wrong_sign["uv_wavelengths"] = inputs["uv_wavelengths"].copy()
    inputs_wrong_sign["uv_wavelengths"][:, 0] *= -1.0

    operator_wrong_sign = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_nufft_from(
            **inputs_wrong_sign
        )
    )

    # The control: of the eight candidate mappings (axis swap x sign of x x sign of y) only two
    # agree with the brute force, and they are the same construction. Negating `u` selects one of
    # the six wrong ones, which must fail the pin loudly -- if it passed, the pin would be
    # measuring nothing about the sign convention.
    with pytest.raises(AssertionError):
        _assert_matches_brute_force(operator_wrong_sign, operator_via_np)


def test__nufft_precision_operator_from__method_routes_to_each_builder():
    pytest.importorskip("nufftax")

    inputs = _nufft_pin_inputs_16x16()

    operator_via_np = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_np_from(**inputs)
    )
    operator_via_jax = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_jax_from(**inputs)
    )
    operator_via_nufft = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_nufft_from(
            **inputs
        )
    )

    # `"numpy"` and `"jax"` are pure routing, so they return the brute-force arrays unchanged.
    np.testing.assert_array_equal(
        np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_from(
                method="numpy", **inputs
            )
        ),
        operator_via_np,
    )
    np.testing.assert_array_equal(
        np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_from(
                method="jax", **inputs
            )
        ),
        operator_via_jax,
    )

    # `use_jax=True` is kept for backwards compatibility and maps onto `method="jax"`.
    np.testing.assert_array_equal(
        np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_from(
                use_jax=True, **inputs
            )
        ),
        operator_via_jax,
    )

    # The default is the NUFFT builder, and it agrees with the brute force.
    operator_default = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_from(**inputs)
    )

    np.testing.assert_array_equal(operator_default, operator_via_nufft)
    _assert_matches_brute_force(operator_default, operator_via_np)


def test__nufft_precision_operator_from__unknown_method_raises():
    inputs = _nufft_pin_inputs_7x7()

    with pytest.raises(ValueError):
        aa.util.inversion_interferometer.nufft_precision_operator_from(
            method="type-1", **inputs
        )


def test__nufft_precision_operator_from__disable_jax_falls_back_to_the_numpy_builder(
    monkeypatch, caplog
):
    inputs = _nufft_pin_inputs_7x7()

    operator_via_np = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_np_from(**inputs)
    )

    # `PYAUTO_DISABLE_JAX=1` is the harness-level kill switch. Both the default NUFFT builder and
    # the `"jax"` brute force run on JAX, so both must fall back -- and loudly, because the NumPy
    # brute force is O(N_pix * K) where the NUFFT is O(K * nspread^2 + M log M).
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")

    for kwargs in ({}, {"method": "jax"}, {"use_jax": True}):
        caplog.clear()

        with caplog.at_level("WARNING"):
            operator = np.asarray(
                aa.util.inversion_interferometer.nufft_precision_operator_from(
                    **kwargs, **inputs
                )
            )

        np.testing.assert_array_equal(operator, operator_via_np)
        assert "PYAUTO_DISABLE_JAX" in caplog.text

    # Without the variable the default is the NUFFT builder again, so the fallback is the switch's
    # doing and not a permanent demotion.
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)

    pytest.importorskip("nufftax")

    np.testing.assert_array_equal(
        np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_from(**inputs)
        ),
        np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_via_nufft_from(
                **inputs
            )
        ),
    )


def test__nufft_precision_operator_from__nufftax_absent_falls_back_to_the_numpy_builder(
    monkeypatch, caplog
):
    inputs = _nufft_pin_inputs_7x7()

    operator_via_np = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_via_np_from(**inputs)
    )

    # `nufftax` is an optional dependency, so the default builder has to survive its absence --
    # loudly, naming the O(N_pix * K) cost the caller now pays, rather than silently.
    monkeypatch.setattr(aa.util.inversion_interferometer, "_load_nufftax", lambda: None)

    with caplog.at_level("WARNING"):
        operator = np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_from(**inputs)
        )

    np.testing.assert_array_equal(operator, operator_via_np)
    assert "nufftax" in caplog.text

    # The builder itself raises rather than falling back: only the dispatcher chooses.
    with pytest.raises(ModuleNotFoundError):
        aa.util.inversion_interferometer.nufft_precision_operator_via_nufft_from(
            **inputs
        )
