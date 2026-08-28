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


# Odd in each axis, as the kernel validation requires, but deliberately not
# all square: `kernel_shape // 2` is only orientation-safe when the y half-width
# is taken from axis 0 and the x half-width from axis 1. Values are asymmetric
# so a transposed gather cannot hide behind a symmetric kernel.
KERNELS_ODD = [
    np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 1.0], [2.0, 0.0, 1.0]]),  # 3x3 square
    np.arange(1.0, 16.0).reshape(3, 5),  # 3x5 wide
    np.arange(1.0, 16.0).reshape(5, 3),  # 5x3 tall
    np.arange(1.0, 36.0).reshape(5, 7),  # 5x7 wide
]

KERNEL_IDS = ["3x3", "3x5", "5x3", "5x7"]


@pytest.mark.parametrize("kernel", KERNELS_ODD, ids=KERNEL_IDS)
def test__psf_weighted_data_from__unmasked_pixels_on_array_edge(kernel):
    """
    Regression test for two distinct defects in the numba gather, both of which
    the zero-padded numpy implementation is the reference for.

    1. An unmasked pixel within `kernel_shape // 2` of the array edge drives the
       kernel off the weight map. numba `@jit()` does not bounds-check array
       reads, so those positions silently returned uninitialized memory (values
       of order 1e299) rather than raising, poisoning `psf_weighted_data` and
       the data vector built from it. Because the values read depend on whatever
       the allocator left next to the weight map, the corruption was heap-state
       dependent: deterministic on the first call after a cold-cache compile,
       and intermittent in forked multiprocessing workers.

    2. The y and x kernel half-widths were derived from the *transposed* kernel
       axes. That is invisible for a square kernel -- the only shape the tests
       used to cover -- but mis-centres the gather along both axes for a
       non-square one. Kernels are validated as odd per axis, never as square,
       so a 3x5 PSF reaches this path and silently returns wrong values.

    Every other test in this module masks a one-pixel border and uses a square
    kernel, so none of them exercise either path.
    """

    image = np.arange(1.0, 26.0).reshape(5, 5)
    noise_map = np.ones((5, 5))

    # Every pixel unmasked, so the border pixels push the kernel off the array.
    native_index_for_slim_index = np.array([[y, x] for y in range(5) for x in range(5)])

    psf_weighted_data = aa.util.inversion_imaging_numba.psf_weighted_data_from(
        image_native=image,
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    psf_weighted_data_numpy = aa.util.inversion_imaging.psf_weighted_data_from(
        weight_map_native=image / noise_map**2.0,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    assert np.all(np.isfinite(psf_weighted_data))
    assert psf_weighted_data == pytest.approx(psf_weighted_data_numpy, 1.0e-8)


def test__psf_weighted_data_from__kernel_axes_are_not_transposed():
    """
    A direct probe of which weight-map pixel the gather actually reads, that does
    not re-derive the implementation to do it.

    The kernel is zero everywhere except its top-left corner, so a single kernel
    tap fires per image pixel, and the weight map encodes its own coordinates as
    `10 * (y + 1) + (x + 1)`. The returned value therefore *names* the pixel that
    was gathered.

    For a (ky, kx) kernel the corner tap sits at offset `(-(ky // 2), -(kx // 2))`
    from the probe pixel. Transposing the half-widths swaps those offsets, so a
    wide kernel and its tall transpose must return different, individually
    predictable values -- which is exactly what a square kernel cannot show.
    """

    y_indexes, x_indexes = np.indices((7, 7))

    # weight[y, x] == 10 * (y + 1) + (x + 1); noise of 1 leaves image == weight.
    image = 10.0 * (y_indexes + 1.0) + (x_indexes + 1.0)
    noise_map = np.ones((7, 7))

    probe_y, probe_x = 3, 3
    native_index_for_slim_index = np.array([[probe_y, probe_x]])

    def gathered_value(kernel_shape):
        kernel = np.zeros(kernel_shape)
        kernel[0, 0] = 1.0

        return aa.util.inversion_imaging_numba.psf_weighted_data_from(
            image_native=image,
            noise_map_native=noise_map,
            kernel_native=kernel,
            native_index_for_slim_index=native_index_for_slim_index,
        )[0]

    # 3x5: y half-width 1, x half-width 2 -> reads (3 - 1, 3 - 2) == (2, 1) == 32.
    assert gathered_value((3, 5)) == pytest.approx(32.0, 1.0e-8)

    # 5x3: y half-width 2, x half-width 1 -> reads (3 - 2, 3 - 1) == (1, 2) == 23.
    assert gathered_value((5, 3)) == pytest.approx(23.0, 1.0e-8)

    # Square: both half-widths 1 -> reads (2, 2) == 33, and is blind to the swap.
    assert gathered_value((3, 3)) == pytest.approx(33.0, 1.0e-8)


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


@pytest.mark.parametrize("kernel", KERNELS_ODD, ids=KERNEL_IDS)
def test__psf_precision_operator_sparse_from__edge_pixels(kernel):
    """
    Regression test for the same two defects as the `psf_weighted_data_from`
    pair above, on the precision-operator path.

    Every slim pixel sits at a corner of the 4x4 noise map, so the kernel walk
    in `psf_precision_value_from` indexes off the array; numba.jit() does not
    bounds-check, so without the explicit guard in the function those reads
    return uninitialized memory. And the kernel half-widths were derived from
    the transposed axes, which the non-square parametrisations below exercise
    and a square kernel cannot.

    The two functions are fixed together deliberately: they must agree on kernel
    orientation, or the `psf_weighted_data` and `psf_precision_operator` paths
    would disagree with each other.
    """
    noise_map = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0, 1.0],
            [1.0, 2.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
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
    #
    # `kernel` is indexed [y, x], so the y half-width comes from its first axis
    # and the x half-width from its second. This reference used to derive them
    # the other way round -- mirroring the very bug it is meant to catch, which
    # a square kernel made invisible.
    def _reference_value(ip0_y, ip0_x, ip1_y, ip1_x):
        h, w = noise_map.shape
        kh, kw = kernel.shape
        kernel_shift_y = -(kh // 2)
        kernel_shift_x = -(kw // 2)
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
                        total += kernel[k0_y, k0_x] * kernel[k1_y, k1_x] / v**2
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


def test__psf_precision_value_from__kernel_axes_are_not_transposed():
    """
    The `psf_weighted_data_from` orientation probe's twin, on the precision path,
    so both gathers are pinned to the same kernel orientation independently.

    A single-tap kernel (non-zero only at its top-left corner) with `ip0 == ip1`
    reduces `psf_precision_value_from` to `1.0 / value_native[gathered]**2`, and
    the value map encodes its own coordinates as `10 * (y + 1) + (x + 1)`. The
    returned value therefore names the pixel that was gathered, without the test
    re-deriving the kernel walk.
    """

    y_indexes, x_indexes = np.indices((7, 7))

    value_native = 10.0 * (y_indexes + 1.0) + (x_indexes + 1.0)

    probe_y, probe_x = 3, 3

    def gathered_value(kernel_shape):
        kernel = np.zeros(kernel_shape)
        kernel[0, 0] = 1.0

        curvature_value = aa.util.inversion_imaging_numba.psf_precision_value_from(
            value_native=value_native,
            kernel_native=kernel,
            ip0_y=probe_y,
            ip0_x=probe_x,
            ip1_y=probe_y,
            ip1_x=probe_x,
        )

        # curvature_value == 1.0 / value_native[gathered] ** 2.0
        return 1.0 / np.sqrt(curvature_value)

    # 3x5: y half-width 1, x half-width 2 -> reads (3 - 1, 3 - 2) == (2, 1) == 32.
    assert gathered_value((3, 5)) == pytest.approx(32.0, 1.0e-8)

    # 5x3: y half-width 2, x half-width 1 -> reads (3 - 2, 3 - 1) == (1, 2) == 23.
    assert gathered_value((5, 3)) == pytest.approx(23.0, 1.0e-8)

    # Square: both half-widths 1 -> reads (2, 2) == 33, and is blind to the swap.
    assert gathered_value((3, 3)) == pytest.approx(33.0, 1.0e-8)


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


@pytest.mark.parametrize("kernel", KERNELS_ODD, ids=KERNEL_IDS)
def test__curvature_matrix_off_diags_via_mapper_and_blurred_curvature_weights_from__matches_dense_kernel(
    kernel,
):
    """
    The mapper x linear-func block of `F` used to expand the curvature weights onto the
    native grid, run a dense sliding-window correlation with the PSF and scatter the result,
    all inside one numba kernel. The correlation now runs as a batched FFT convolution with
    the PSF reversed along both axes (`Convolver.reversed_kernel`) and only the scatter stays
    in numba.

    This asserts the two produce the same block. The kernels are asymmetric and non-square,
    so a missing reversal (correlation computed as a convolution) or a transposed axis cannot
    pass.
    """
    mask = aa.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, False, False, False, True, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, True, False, False, False, True, True],
                [True, True, True, True, True, True, True],
            ]
        ),
        pixel_scales=1.0,
    )

    data_pixels = int(mask.pixels_in_mask)
    n_funcs = 3
    pix_pixels = 5

    rng = np.random.default_rng(505)

    curvature_weights = rng.normal(size=(data_pixels, n_funcs))

    # Every data pixel maps to one or two source pixels, with non-uniform weights.
    max_lengths = 2
    pix_lengths = rng.integers(1, max_lengths + 1, size=data_pixels).astype("int")
    data_to_pix_unique = rng.integers(
        0, pix_pixels, size=(data_pixels, max_lengths)
    ).astype("int")
    data_weights = rng.random(size=(data_pixels, max_lengths))

    off_diag_dense = aa.util.inversion_imaging_numba.curvature_matrix_off_diags_via_mapper_and_linear_func_curvature_vector_from(
        data_to_pix_unique=data_to_pix_unique,
        data_weights=data_weights,
        pix_lengths=pix_lengths,
        pix_pixels=pix_pixels,
        curvature_weights=curvature_weights,
        mask=np.array(mask),
        psf_kernel=kernel,
    )

    convolver = aa.Convolver(
        kernel=aa.Array2D.no_mask(values=kernel, pixel_scales=1.0),
    )

    blurred_curvature_weights = convolver.reversed_kernel.convolved_mapping_matrix_from(
        mapping_matrix=curvature_weights,
        mask=mask,
        xp=np,
    )

    off_diag_fft = aa.util.inversion_imaging_numba.curvature_matrix_off_diags_via_mapper_and_blurred_curvature_weights_from(
        data_to_pix_unique=data_to_pix_unique,
        data_weights=data_weights,
        pix_lengths=pix_lengths,
        pix_pixels=pix_pixels,
        blurred_curvature_weights=blurred_curvature_weights,
    )

    assert off_diag_fft == pytest.approx(off_diag_dense, rel=1.0e-6)


def test__convolver_reversed_kernel__is_the_kernel_reversed_and_convolves_as_a_correlation():
    """
    `Convolver.reversed_kernel` is the same convolver with its kernel reversed along both
    axes, so convolving with it correlates with the original kernel.
    """
    kernel = np.arange(1.0, 16.0).reshape(3, 5)

    convolver = aa.Convolver(
        kernel=aa.Array2D.no_mask(values=kernel, pixel_scales=1.0),
    )

    assert convolver.reversed_kernel.kernel.native.array == pytest.approx(
        kernel[::-1, ::-1]
    )

    # Cached, so the reversed kernel and its FFT geometry are built once.
    assert convolver.reversed_kernel is convolver.reversed_kernel


# ------------------------------------------------------------------
# The mapper x mapper block of `F` (`curvature_matrix_via_sparse_operator_from`).
#
# This is the single most expensive step of the numba CPU imaging likelihood at
# HST resolution (0.255 s of a 0.63 s evaluation, PyAutoArray#507 step 0), and
# until these tests it had no direct unit test at all -- only the end-to-end
# inversion fixtures, which cannot distinguish a restructured kernel from a
# subtly wrong one.
#
# The reference the kernel is pinned against is the definition it is an
# optimisation of:
#
#     F = M.T @ W @ M
#
# with `W` the *dense* [image_pixels, image_pixels] precision operator from
# `psf_precision_operator_from` and `M` the dense [image_pixels, pix_pixels]
# mapping matrix that `(data_to_pix_unique, data_weights, pix_lengths)` encodes.
# Nothing in the reference re-derives the kernel: it does not know about the
# sparse upper-triangle storage, the halved diagonal or the `A + A.T` fold.
#
# The kernels are asymmetric and non-square (`KERNELS_ODD`), so a transposed
# axis in the operator build cannot pass.
# ------------------------------------------------------------------


def _mapping_matrix_from(data_to_pix_unique, data_weights, pix_lengths, pix_pixels):
    """The dense mapping matrix `M` that the unique-mappings triple encodes."""
    mapping_matrix = np.zeros((pix_lengths.shape[0], pix_pixels))

    for data_index in range(pix_lengths.shape[0]):
        for pix_index in range(pix_lengths[data_index]):
            mapping_matrix[
                data_index, data_to_pix_unique[data_index, pix_index]
            ] += data_weights[data_index, pix_index]

    return mapping_matrix


def _sparse_operator_mappings(seed=507, shape=(6, 6), pix_pixels=7, max_lengths=3):
    """A small fully-unmasked dataset plus random non-uniform unique mappings."""
    rng = np.random.default_rng(seed)

    noise_map = 0.5 + rng.random(shape) * 2.0
    native_index_for_slim_index = np.array(
        [[y, x] for y in range(shape[0]) for x in range(shape[1])]
    )

    data_pixels = native_index_for_slim_index.shape[0]

    pix_lengths = rng.integers(1, max_lengths + 1, size=data_pixels).astype("int")
    data_to_pix_unique = rng.integers(
        0, pix_pixels, size=(data_pixels, max_lengths)
    ).astype("int")
    data_weights = rng.random(size=(data_pixels, max_lengths))

    return (
        noise_map,
        native_index_for_slim_index,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    )


@pytest.mark.parametrize("kernel", KERNELS_ODD, ids=KERNEL_IDS)
def test__curvature_matrix_via_sparse_operator_from__matches_dense_psf_precision_operator(
    kernel,
):
    pix_pixels = 7

    (
        noise_map,
        native_index_for_slim_index,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = _sparse_operator_mappings(pix_pixels=pix_pixels)

    (
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    curvature_matrix = (
        aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_from(
            psf_precision_operator=psf_precision_operator,
            psf_precision_indexes=psf_precision_indexes.astype("int"),
            psf_precision_lengths=psf_precision_lengths.astype("int"),
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
            pix_pixels=pix_pixels,
        )
    )

    psf_precision_operator_dense = (
        aa.util.inversion_imaging_numba.psf_precision_operator_from(
            noise_map_native=noise_map,
            kernel_native=kernel,
            native_index_for_slim_index=native_index_for_slim_index,
        )
    )

    mapping_matrix = _mapping_matrix_from(
        data_to_pix_unique=data_to_pix_unique,
        data_weights=data_weights,
        pix_lengths=pix_lengths,
        pix_pixels=pix_pixels,
    )

    curvature_matrix_dense = np.dot(
        mapping_matrix.T, np.dot(psf_precision_operator_dense, mapping_matrix)
    )

    assert curvature_matrix == pytest.approx(curvature_matrix_dense, rel=1.0e-10)


def test__curvature_matrix_via_sparse_operator_from__is_symmetric_and_the_diagonal_is_not_double_counted():
    """
    Two contracts the kernel's `A + A.T` fold depends on, asserted separately from
    the dense oracle above so a future restructuring cannot satisfy one by breaking
    the other.

    1. The returned matrix is *exactly* symmetric -- not to a tolerance. The
       inversion's `curvature_matrix` runs no global symmetrizing pass over the
       assembled F, so this block must come out symmetric on its own.

    2. `psf_precision_operator_sparse_from` halves the `ip0 == ip1` entries because
       the fold doubles the diagonal of the accumulated matrix. A single data pixel
       mapping to a single source pixel isolates exactly that pair: `F[0, 0]` must
       be `w**2 * W[0, 0]`, and would come out twice that if either half of the
       contract were dropped.
    """
    kernel = np.arange(1.0, 16.0).reshape(3, 5)

    # -- 1. exact symmetry ------------------------------------------------
    pix_pixels = 7

    (
        noise_map,
        native_index_for_slim_index,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = _sparse_operator_mappings(pix_pixels=pix_pixels)

    (
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    curvature_matrix = (
        aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_from(
            psf_precision_operator=psf_precision_operator,
            psf_precision_indexes=psf_precision_indexes.astype("int"),
            psf_precision_lengths=psf_precision_lengths.astype("int"),
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
            pix_pixels=pix_pixels,
        )
    )

    assert np.array_equal(curvature_matrix, curvature_matrix.T)

    # -- 2. the halved diagonal --------------------------------------------
    noise_map_1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    native_index_for_slim_index_1 = np.array([[1, 1]])

    weight = 0.75

    (
        psf_precision_operator_1,
        psf_precision_indexes_1,
        psf_precision_lengths_1,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=noise_map_1,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index_1,
    )

    curvature_matrix_1 = (
        aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_from(
            psf_precision_operator=psf_precision_operator_1,
            psf_precision_indexes=psf_precision_indexes_1.astype("int"),
            psf_precision_lengths=psf_precision_lengths_1.astype("int"),
            data_to_pix_unique=np.array([[0]]),
            data_weights=np.array([[weight]]),
            pix_lengths=np.array([1]),
            pix_pixels=1,
        )
    )

    psf_precision_operator_dense_1 = (
        aa.util.inversion_imaging_numba.psf_precision_operator_from(
            noise_map_native=noise_map_1,
            kernel_native=kernel,
            native_index_for_slim_index=native_index_for_slim_index_1,
        )
    )

    assert curvature_matrix_1 == pytest.approx(
        np.array([[weight**2.0 * psf_precision_operator_dense_1[0, 0]]]), rel=1.0e-12
    )


@pytest.mark.parametrize("kernel", KERNELS_ODD, ids=KERNEL_IDS)
def test__curvature_matrix_via_sparse_operator_from__matches_the_reference_kernel_bit_identically(
    kernel,
):
    """
    `curvature_matrix_via_sparse_operator_reference_from` is the unrestructured
    quadruple loop the production kernel is an optimisation of.

    `curvature_matrix_via_sparse_operator_direct_from` is the branch of the public
    `curvature_matrix_via_sparse_operator_from` taken above
    `CURVATURE_TWO_STAGE_MAX_PIX_PIXELS`.

    The PyAutoArray#507 step-1 restructuring hoists `data_1`'s `(pix, weight)` pairs
    into 1-D row views (they were re-gathered u0 times per stored pair from
    wide-stride 2-D arrays) and takes `curvature_matrix[pix_0]` as a row view. It
    leaves the accumulated expression operand-for-operand unchanged, so the result
    must be *bit-identical*, not merely close: floating-point addition is not
    associative, and `np.array_equal` is what catches a reassociation that a
    tolerance would wave through.
    """
    pix_pixels = 7

    (
        noise_map,
        native_index_for_slim_index,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = _sparse_operator_mappings(pix_pixels=pix_pixels)

    (
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    kwargs = dict(
        psf_precision_operator=psf_precision_operator,
        psf_precision_indexes=psf_precision_indexes.astype("int"),
        psf_precision_lengths=psf_precision_lengths.astype("int"),
        data_to_pix_unique=data_to_pix_unique,
        data_weights=data_weights,
        pix_lengths=pix_lengths,
        pix_pixels=pix_pixels,
    )

    curvature_matrix = aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_direct_from(
        **kwargs
    )

    curvature_matrix_reference = aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_reference_from(
        **kwargs
    )

    assert np.array_equal(curvature_matrix, curvature_matrix_reference)


# ------------------------------------------------------------------
# The two-stage mapper x mapper kernel (PyAutoArray#507 step 2).
#
# `curvature_matrix_via_sparse_operator_two_stage_from` sums exactly the same
# products as the direct loop but in a different order (a dense source-space
# accumulator per data pixel, then one contiguous AXPY per mapping), so it agrees
# to floating-point reassociation rather than bit-identically. rtol=1e-6 is the
# tolerance the likelihood itself is pinned at; the observed difference on the
# production HST geometries is ~4e-13.
#
# Control runs recorded against these tests, on the correct kernel, in the
# commit message.
# ------------------------------------------------------------------


@pytest.mark.parametrize("kernel", KERNELS_ODD, ids=KERNEL_IDS)
@pytest.mark.parametrize("pix_pixels", [1, 7, 64])
def test__curvature_matrix_via_sparse_operator_two_stage_from__matches_reference_kernel(
    kernel, pix_pixels
):
    """
    `pix_pixels = 1` is the degenerate source space (every data pixel maps to the
    same pixel, so the accumulator is a scalar and the fold is the diagonal alone);
    7 is smaller than the per-data-pixel mapping footprint; 64 is larger than it,
    so the accumulator is sparse within itself. All three sit below
    `CURVATURE_TWO_STAGE_MAX_PIX_PIXELS`, which the dispatcher test below covers
    from both sides.
    """
    (
        noise_map,
        native_index_for_slim_index,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = _sparse_operator_mappings(pix_pixels=pix_pixels)

    (
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    kwargs = dict(
        psf_precision_operator=psf_precision_operator,
        psf_precision_indexes=psf_precision_indexes.astype("int"),
        psf_precision_lengths=psf_precision_lengths.astype("int"),
        data_to_pix_unique=data_to_pix_unique,
        data_weights=data_weights,
        pix_lengths=pix_lengths,
        pix_pixels=pix_pixels,
    )

    curvature_matrix_two_stage = aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_two_stage_from(
        **kwargs
    )

    curvature_matrix_reference = aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_reference_from(
        **kwargs
    )

    assert curvature_matrix_two_stage == pytest.approx(
        curvature_matrix_reference, rel=1.0e-6
    )

    # The fold contract must survive the reformulation: exactly symmetric, and the
    # diagonal not double counted (both are what `A + A.T` on the halved-diagonal
    # accumulation buys).
    assert np.array_equal(curvature_matrix_two_stage, curvature_matrix_two_stage.T)


@pytest.mark.parametrize("kernel", KERNELS_ODD, ids=KERNEL_IDS)
def test__curvature_matrix_via_sparse_operator_two_stage_from__matches_dense_psf_precision_operator(
    kernel,
):
    """
    The two-stage kernel against the same dense `M.T @ W @ M` oracle the direct
    kernel is pinned to, so it is anchored to the definition and not only to the
    other implementation.
    """
    pix_pixels = 7

    (
        noise_map,
        native_index_for_slim_index,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = _sparse_operator_mappings(pix_pixels=pix_pixels)

    (
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=noise_map,
        kernel_native=kernel,
        native_index_for_slim_index=native_index_for_slim_index,
    )

    curvature_matrix = aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_two_stage_from(
        psf_precision_operator=psf_precision_operator,
        psf_precision_indexes=psf_precision_indexes.astype("int"),
        psf_precision_lengths=psf_precision_lengths.astype("int"),
        data_to_pix_unique=data_to_pix_unique,
        data_weights=data_weights,
        pix_lengths=pix_lengths,
        pix_pixels=pix_pixels,
    )

    psf_precision_operator_dense = (
        aa.util.inversion_imaging_numba.psf_precision_operator_from(
            noise_map_native=noise_map,
            kernel_native=kernel,
            native_index_for_slim_index=native_index_for_slim_index,
        )
    )

    mapping_matrix = _mapping_matrix_from(
        data_to_pix_unique=data_to_pix_unique,
        data_weights=data_weights,
        pix_lengths=pix_lengths,
        pix_pixels=pix_pixels,
    )

    curvature_matrix_dense = np.dot(
        mapping_matrix.T, np.dot(psf_precision_operator_dense, mapping_matrix)
    )

    assert curvature_matrix == pytest.approx(curvature_matrix_dense, rel=1.0e-6)


def test__curvature_matrix_via_sparse_operator_from__dispatches_on_the_two_stage_threshold():
    """
    Both branches of the public entry point, exercised from either side of
    `CURVATURE_TWO_STAGE_MAX_PIX_PIXELS` without building a 4096-pixel source
    space: the threshold is a per-call argument defaulting to the module constant,
    precisely so that this test does not have to allocate a 134 MB matrix to reach
    the other branch.

    Each branch is asserted *bit-identical* to the implementation it is supposed to
    have called -- so a dispatcher that quietly ran the wrong one, or ran neither
    faithfully, cannot pass, whereas an rtol comparison between the two branches
    would (they agree to ~1e-13).
    """
    pix_pixels = 7

    (
        noise_map,
        native_index_for_slim_index,
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = _sparse_operator_mappings(pix_pixels=pix_pixels)

    (
        psf_precision_operator,
        psf_precision_indexes,
        psf_precision_lengths,
    ) = aa.util.inversion_imaging_numba.psf_precision_operator_sparse_from(
        noise_map_native=np.arange(1.0, 37.0).reshape(6, 6),
        kernel_native=np.arange(1.0, 16.0).reshape(3, 5),
        native_index_for_slim_index=native_index_for_slim_index,
    )

    kwargs = dict(
        psf_precision_operator=psf_precision_operator,
        psf_precision_indexes=psf_precision_indexes.astype("int"),
        psf_precision_lengths=psf_precision_lengths.astype("int"),
        data_to_pix_unique=data_to_pix_unique,
        data_weights=data_weights,
        pix_lengths=pix_lengths,
        pix_pixels=pix_pixels,
    )

    # Default threshold (4096) -> pix_pixels of 7 is below it -> two-stage.
    assert np.array_equal(
        aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_from(
            **kwargs
        ),
        aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_two_stage_from(
            **kwargs
        ),
    )

    # Threshold pushed below pix_pixels -> the direct loop.
    assert np.array_equal(
        aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_from(
            two_stage_max_pix_pixels=pix_pixels - 1, **kwargs
        ),
        aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_direct_from(
            **kwargs
        ),
    )

    # ... and the two branches are not the same array, so the assertions above are
    # not both trivially satisfied by a single implementation.
    assert not np.array_equal(
        aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_two_stage_from(
            **kwargs
        ),
        aa.util.inversion_imaging_numba.curvature_matrix_via_sparse_operator_direct_from(
            **kwargs
        ),
    )
