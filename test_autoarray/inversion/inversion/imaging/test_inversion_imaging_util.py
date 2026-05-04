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


def test__psf_precision_operator_sparse_from__edge_pixels():
    # Regression test: every slim pixel sits at a corner of the 4x4 noise map,
    # so the kernel walk in psf_precision_value_from indexes off the array.
    # numba.jit() does not bounds-check, so without the explicit guard added
    # in the function those reads return uninitialized memory and produce
    # astronomically large or non-finite operator entries.
    noise_map = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0, 1.0],
            [1.0, 2.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    kernel = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 1.0]])
    native_index_for_slim_index = np.array([[0, 0], [0, 3], [3, 0], [3, 3]])

    (
        op,
        indexes,
        lengths,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    # Sanity: no inf/nan in the operator.
    assert np.isfinite(op).all()
    assert int(lengths.sum()) == op.shape[0]

    # Independent reference: a pure-numpy bounds-checked re-implementation of
    # psf_precision_value_from. The numba version with the fix applied must
    # match this byte-for-byte.
    def _reference_value(ip0_y, ip0_x, ip1_y, ip1_x):
        h, w = noise_map.shape
        kh, kw = kernel.shape
        kernel_shift_y = -(kw // 2)
        kernel_shift_x = -(kh // 2)
        ip_y_offset = ip0_y - ip1_y
        ip_x_offset = ip0_x - ip1_x
        if (
            ip_y_offset < 2 * kernel_shift_y
            or ip_y_offset > -2 * kernel_shift_y
            or ip_x_offset < 2 * kernel_shift_x
            or ip_x_offset > -2 * kernel_shift_x
        ):
            return 0.0
        total = 0.0
        for k0_y in range(kh):
            for k0_x in range(kw):
                iy = ip0_y + k0_y + kernel_shift_y
                ix = ip0_x + k0_x + kernel_shift_x
                if iy < 0 or iy >= h or ix < 0 or ix >= w:
                    continue
                v = noise_map[iy, ix]
                if v > 0.0:
                    k1_y = k0_y + ip_y_offset
                    k1_x = k0_x + ip_x_offset
                    if 0 <= k1_y < kh and 0 <= k1_x < kw:
                        total += kernel[k0_y, k0_x] * kernel[k1_y, k1_x] / v ** 2
        return total

    n_pix = native_index_for_slim_index.shape[0]
    expected = []
    expected_indexes = []
    expected_lengths = []
    for ip0 in range(n_pix):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]
        count = 0
        for ip1 in range(ip0, n_pix):
            ip1_y, ip1_x = native_index_for_slim_index[ip1]
            v = _reference_value(ip0_y, ip0_x, ip1_y, ip1_x)
            if ip0 == ip1:
                v /= 2.0
            if v > 0.0:
                expected.append(v)
                expected_indexes.append(ip1)
                count += 1
        expected_lengths.append(count)

    assert op == pytest.approx(np.array(expected), 1.0e-4)
    assert indexes == pytest.approx(np.array(expected_indexes), 1.0e-4)
    assert lengths == pytest.approx(np.array(expected_lengths), 1.0e-4)


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
