import os

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

    # `use_jax=True` only upgrades a brute-force method, so `method="numpy"` with it set is
    # the JAX brute force.
    np.testing.assert_array_equal(
        np.asarray(
            aa.util.inversion_interferometer.nufft_precision_operator_from(
                method="numpy", use_jax=True, **inputs
            )
        ),
        operator_via_jax,
    )

    # Under the default `method="nufft"` it is ignored -- the NUFFT already runs on JAX, so
    # honouring it there would demote every existing `use_jax=True` caller (the workspace
    # `apply_sparse_operator(use_jax=True)` calls) from seconds to the O(N_pix * K) brute force.
    operator_use_jax = np.asarray(
        aa.util.inversion_interferometer.nufft_precision_operator_from(
            use_jax=True, **inputs
        )
    )

    np.testing.assert_array_equal(operator_use_jax, operator_via_nufft)
    _assert_matches_brute_force(operator_use_jax, operator_via_np)

    # The control: the NUFFT array is not the JAX brute-force array, so the assertion above is
    # testing the routing and not two builders that happen to agree bitwise.
    assert not np.array_equal(operator_use_jax, operator_via_jax)

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
    # brute force is O(N_pix * K) where the NUFFT is O(K * nspread^2 + M log M). `use_jax=True`
    # falls back either way: ignored under the default, and demoted again when it upgrades
    # `method="numpy"` to the JAX brute force.
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")

    for kwargs in (
        {},
        {"method": "jax"},
        {"use_jax": True},
        {"method": "numpy", "use_jax": True},
    ):
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


def _numpy_backend_fixtures():
    """
    The two `InterferometerSparseOperator` fixtures the NumPy/JAX parity tests below run on,
    each built with `batch_size=4`.

    The 7x7 / K=5 case is the module's shared shape fixture. The 12x12 / K=64 case is the
    seeded one the `rfft2` pin uses: its source-pixel count exceeds `batch_size`, so the
    block sweep runs more than one block and finishes on a partial one -- the branch the JAX
    path needs `dynamic_update_slice` and a column mask for, and the NumPy path a plain
    Python loop.
    """
    fixtures = []

    for mask, n_visibilities, seed in (
        (_mask_7x7(), 5, 3),
        (
            aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=1.0, radius=4.0),
            64,
            11,
        ),
    ):
        dataset, rng = _dataset_from(
            mask=mask, n_visibilities=n_visibilities, seed=seed
        )

        operator = dataset.apply_sparse_operator(
            nufft_precision_operator=dataset.psf_precision_operator_from(
                method="numpy"
            ),
            batch_size=4,
        ).sparse_operator

        fixtures.append((operator, mask, rng))

    return fixtures


def _assert_numpy_matches_jax(result, result_via_jax):
    """
    Asserts a NumPy-branch result matches the JAX branch at the module's exact pin.

    Both branches evaluate the same real-FFT convolution and the same sparse triple product
    in float64, so they agree to floating-point round-off and nothing looser is warranted.
    `atol` is peak-scaled so that entries orders below the peak -- which carry no relative
    accuracy of their own -- do not turn round-off into a failure.
    """
    result_via_jax = np.asarray(result_via_jax)

    np.testing.assert_allclose(
        np.asarray(result),
        result_via_jax,
        rtol=1.0e-10,
        atol=1.0e-10 * np.abs(result_via_jax).max(),
    )


def _triplets_with_duplicates(rng, M, S, nnz):
    """
    Returns COO triplets whose row/column index ranges are small enough that
    `(row, col)` pairs repeat, so the assembly has to *sum* duplicate entries. The JAX
    branch does this with `.at[].add`, the NumPy branch with `scipy.sparse`'s COO
    duplicate summation; a NumPy branch that overwrote instead would fail the pin.
    """
    rows = rng.integers(0, min(M, 6), size=nnz)
    cols = rng.integers(0, S, size=nnz)
    vals = rng.normal(size=nnz)

    return rows, cols, vals


def test__interferometer_sparse_operator__numpy_branch_matches_jax_branch():
    """
    Every public method of `InterferometerSparseOperator` takes an `xp` and must return the
    same matrix on either backend: a CPU fit passing `xp=np` runs the NumPy/scipy bodies
    instead of the JAX ones, and that must be a change of backend only.
    """
    pytest.importorskip("jax")

    import jax.numpy as jnp

    for operator, mask, rng in _numpy_backend_fixtures():
        M = operator.M
        S = 11
        S1 = 9

        assert S > operator.batch_size

        extent_index_for_masked_pixel = np.array(mask.extent_index_for_masked_pixel)

        # apply_operator
        Fbatch = rng.normal(size=(M, 7))

        _assert_numpy_matches_jax(
            operator.apply_operator(Fbatch, xp=np),
            operator.apply_operator(jnp.asarray(Fbatch), xp=jnp),
        )

        # curvature_matrix_diag_from
        rows, cols, vals = _triplets_with_duplicates(rng, M=M, S=S, nnz=40)

        _assert_numpy_matches_jax(
            operator.curvature_matrix_diag_from(
                rows=rows, cols=cols, vals=vals, S=S, xp=np
            ),
            operator.curvature_matrix_diag_from(
                rows=rows, cols=cols, vals=vals, S=S, xp=jnp
            ),
        )

        # curvature_matrix_off_diag_from
        rows_1, cols_1, vals_1 = _triplets_with_duplicates(rng, M=M, S=S1, nnz=30)

        _assert_numpy_matches_jax(
            operator.curvature_matrix_off_diag_from(
                rows0=rows,
                cols0=cols,
                vals0=vals,
                rows1=rows_1,
                cols1=cols_1,
                vals1=vals_1,
                S0=S,
                S1=S1,
                xp=np,
            ),
            operator.curvature_matrix_off_diag_from(
                rows0=rows,
                cols0=cols,
                vals0=vals,
                rows1=rows_1,
                cols1=cols_1,
                vals1=vals_1,
                S0=S,
                S1=S1,
                xp=jnp,
            ),
        )

        # operated_matrix_slim_from
        matrix_slim = rng.normal(size=(mask.pixels_in_mask, 3))

        _assert_numpy_matches_jax(
            operator.operated_matrix_slim_from(
                matrix_slim=matrix_slim,
                extent_index_for_masked_pixel=extent_index_for_masked_pixel,
                xp=np,
            ),
            operator.operated_matrix_slim_from(
                matrix_slim=matrix_slim,
                extent_index_for_masked_pixel=extent_index_for_masked_pixel,
                xp=jnp,
            ),
        )

        # curvature_matrix_off_diag_func_list_from
        curvature_weights = rng.normal(size=(mask.pixels_in_mask, 3))

        _assert_numpy_matches_jax(
            operator.curvature_matrix_off_diag_func_list_from(
                curvature_weights=curvature_weights,
                extent_index_for_masked_pixel=extent_index_for_masked_pixel,
                rows=rows,
                cols=cols,
                vals=vals,
                S=S,
                xp=np,
            ),
            operator.curvature_matrix_off_diag_func_list_from(
                curvature_weights=curvature_weights,
                extent_index_for_masked_pixel=extent_index_for_masked_pixel,
                rows=rows,
                cols=cols,
                vals=vals,
                S=S,
                xp=jnp,
            ),
        )

        # curvature_matrix_func_list_from
        curvature_weights_0 = rng.normal(size=(mask.pixels_in_mask, 2))

        _assert_numpy_matches_jax(
            operator.curvature_matrix_func_list_from(
                curvature_weights_0=curvature_weights_0,
                curvature_weights_1=curvature_weights,
                extent_index_for_masked_pixel=extent_index_for_masked_pixel,
                xp=np,
            ),
            operator.curvature_matrix_func_list_from(
                curvature_weights_0=curvature_weights_0,
                curvature_weights_1=curvature_weights,
                extent_index_for_masked_pixel=extent_index_for_masked_pixel,
                xp=jnp,
            ),
        )


def _delaunay_triplets_from(mask, over_sample_size):
    """
    Returns the COO triplets of a real Delaunay mapper on `mask`, alongside its parameter
    count, exactly as `InversionInterferometerSparse._sparse_triplets_curvature_from`
    builds them.

    These are the triplets the production path actually hands the operator, and they carry
    two properties hand-written triplets do not: `mapper_util.sparse_triplets_from` pads
    each sub-pixel's interpolation stencil to the longest, emitting `col = -1` with weight
    `0.0` for the unused slots, and at `over_sample_size > 1` several sub-pixels of the same
    image pixel hit the same source pixel, so `(row, col)` pairs repeat.
    """
    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=over_sample_size)

    image_mesh_grid = aa.image_mesh.Overlay(shape=(4, 4)).image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )
    interpolator = aa.mesh.Delaunay(pixels=16).interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh_grid,
    )
    mapper = aa.Mapper(interpolator=interpolator)

    rows, cols, vals = aa.util.mapper.sparse_triplets_from(
        pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
        pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
        slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
        fft_index_for_masked_pixel=mask.extent_index_for_masked_pixel,
        sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
        return_rows_slim=False,
        xp=np,
    )

    return np.asarray(rows), np.asarray(cols), np.asarray(vals), mapper.params


def test__interferometer_sparse_operator__numpy_branch_matches_jax_branch__delaunay_triplets():
    """
    The same parity, on the triplets a real over-sampled Delaunay mapper produces: padded
    `col = -1` entries and repeated `(row, col)` pairs, both of which the JAX branch handles
    implicitly (an out-of-range column is dropped, a repeat is accumulated by `.at[].add`)
    and the NumPy branch must handle explicitly.
    """
    pytest.importorskip("jax")

    import jax.numpy as jnp

    mask = aa.Mask2D.circular(shape_native=(12, 12), pixel_scales=1.0, radius=4.0)

    dataset, _ = _dataset_from(mask=mask, n_visibilities=64, seed=11)

    operator = dataset.apply_sparse_operator(
        nufft_precision_operator=dataset.psf_precision_operator_from(method="numpy"),
        batch_size=4,
    ).sparse_operator

    rows, cols, vals, S = _delaunay_triplets_from(mask=mask, over_sample_size=2)

    # The fixture only tests what it contains: a mapper whose stencils happened not to be
    # padded, or whose sub-pixels happened not to collide, would leave both behaviours
    # untested and the test would still pass.
    assert (cols < 0).any()

    pairs, counts = np.unique(
        np.stack([rows[cols >= 0], cols[cols >= 0]], axis=1), axis=0, return_counts=True
    )
    assert (counts > 1).any()

    assert S > operator.batch_size

    _assert_numpy_matches_jax(
        operator.curvature_matrix_diag_from(
            rows=rows, cols=cols, vals=vals, S=S, xp=np
        ),
        operator.curvature_matrix_diag_from(
            rows=rows, cols=cols, vals=vals, S=S, xp=jnp
        ),
    )


def test__interferometer_sparse_operator__numpy_branch_runs_with_jax_unimportable():
    """
    The point of the `xp` branch: a CPU user must be able to build the operator and run every
    NumPy body in an environment where JAX is not installed at all.

    Deliberately carries no `importorskip("jax")`, so the `unittest-nojax` CI leg runs it.
    """
    import sys

    monkeypatch = pytest.MonkeyPatch()

    mask = _mask_7x7()

    dataset, rng = _dataset_from(mask=mask, n_visibilities=5, seed=3)

    try:
        # `None` in `sys.modules` is the documented way to make an import fail: `import jax`
        # then raises `ImportError` rather than finding the installed package.
        monkeypatch.setitem(sys.modules, "jax", None)
        monkeypatch.setitem(sys.modules, "jax.numpy", None)

        # The brute-force NumPy builder, because the default `"nufft"` builder runs on JAX.
        operator = dataset.apply_sparse_operator(
            nufft_precision_operator=dataset.psf_precision_operator_from(
                method="numpy"
            ),
            batch_size=4,
        ).sparse_operator

        M = operator.M
        S = 11

        extent_index_for_masked_pixel = np.array(mask.extent_index_for_masked_pixel)

        rows, cols, vals = _triplets_with_duplicates(rng, M=M, S=S, nnz=40)
        rows_1, cols_1, vals_1 = _triplets_with_duplicates(rng, M=M, S=5, nnz=20)

        curvature_weights = rng.normal(size=(mask.pixels_in_mask, 3))

        assert operator.apply_operator(rng.normal(size=(M, 3)), xp=np).shape == (M, 3)
        assert operator.curvature_matrix_diag_from(
            rows=rows, cols=cols, vals=vals, S=S, xp=np
        ).shape == (S, S)
        assert operator.curvature_matrix_off_diag_from(
            rows0=rows,
            cols0=cols,
            vals0=vals,
            rows1=rows_1,
            cols1=cols_1,
            vals1=vals_1,
            S0=S,
            S1=5,
            xp=np,
        ).shape == (S, 5)
        assert operator.operated_matrix_slim_from(
            matrix_slim=curvature_weights,
            extent_index_for_masked_pixel=extent_index_for_masked_pixel,
            xp=np,
        ).shape == (mask.pixels_in_mask, 3)
        assert operator.curvature_matrix_off_diag_func_list_from(
            curvature_weights=curvature_weights,
            extent_index_for_masked_pixel=extent_index_for_masked_pixel,
            rows=rows,
            cols=cols,
            vals=vals,
            S=S,
            xp=np,
        ).shape == (S, 3)
        assert operator.curvature_matrix_func_list_from(
            curvature_weights_0=curvature_weights,
            curvature_weights_1=curvature_weights,
            extent_index_for_masked_pixel=extent_index_for_masked_pixel,
            xp=np,
        ).shape == (3, 3)

        # Without this the test would also pass in an environment where JAX imports fine,
        # and would therefore stop testing anything the day the block above broke.
        with pytest.raises(ImportError):
            operator.Khat
    finally:
        monkeypatch.undo()


def test__interferometer_sparse_operator__numpy_branch_imports_no_jax():
    """
    Stronger than the guard above, and the actual complaint the `xp` branch answers: JAX must
    not merely be unnecessary, it must never be *imported*. Importing it costs seconds and
    allocates a backend, and the operator's JAX state (`Khat`, `col_offsets`) is lazy purely
    so that a NumPy fit never pays for it.

    Run in a subprocess because `jax` is in `sys.modules` for the rest of this session as
    soon as any other test imports it.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import os
        import sys

        import numpy as np

        import autoarray as aa

        mask = aa.Mask2D.circular(shape_native=(8, 8), pixel_scales=1.0, radius=3.0)

        rng = np.random.default_rng(seed=1)

        dataset = aa.Interferometer(
            data=aa.Visibilities(visibilities=rng.normal(size=(5, 2))),
            noise_map=aa.VisibilitiesNoiseMap(visibilities=np.ones((5, 2))),
            uv_wavelengths=rng.normal(size=(5, 2)),
            real_space_mask=mask,
            transformer_class=aa.TransformerDFT,
        )

        operator = dataset.apply_sparse_operator(
            method="numpy", batch_size=4
        ).sparse_operator

        M = operator.M

        operator.apply_operator(rng.normal(size=(M, 3)), xp=np)
        operator.curvature_matrix_diag_from(
            rows=rng.integers(0, M, 20),
            cols=rng.integers(0, 6, 20),
            vals=rng.normal(size=20),
            S=6,
            xp=np,
        )

        assert "jax" not in sys.modules, sorted(
            name for name in sys.modules if name.startswith("jax")
        )
        """
    )

    environment = dict(os.environ)
    environment["PYAUTO_DISABLE_JAX"] = "1"

    subprocess.run(
        [sys.executable, "-c", script], check=True, env=environment, timeout=600
    )
