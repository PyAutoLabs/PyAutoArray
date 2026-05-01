from typing import List, Optional, Tuple

from autoarray import numba_util

import numpy as np


@numba_util.jit()
def psf_weighted_data_from(
    image_native: np.ndarray,
    noise_map_native: np.ndarray,
    kernel_native: np.ndarray,
    native_index_for_slim_index,
) -> np.ndarray:
    """
    The sparse linear algebra uses a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF convolution of
    every pair of image pixels given the noise map. This can be used to efficiently compute the curvature matrix via
    the mappings between image and source pixels, in a way that omits having to perform the PSF convolution on every
    individual source pixel. This provides a significant speed up for inversions of imaging datasets.

    When this is used to perform an inversion, the mapping matrices are not computed, meaning that they cannot be
    used to compute the data vector. This method creates the vector `psf_weighted_data` which allows for the data
    vector to be computed efficiently without the mapping matrix.

    The matrix psf_weighted_data is dimensions [image_pixels] and encodes the PSF convolution with the `weight_map`,
    where the weights are the image-pixel values divided by the noise-map values squared:

    weight = image / noise**2.0

    Parameters
    ----------
    image_native
        The two dimensional masked image of values which `psf_weighted_data` is computed from.
    noise_map_native
        The two dimensional masked noise-map of values which `psf_weighted_data` is computed from.
    kernel_native
        The two dimensional PSF kernel that `psf_weighted_data` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    """

    kernel_shift_y = -(kernel_native.shape[1] // 2)
    kernel_shift_x = -(kernel_native.shape[0] // 2)

    image_pixels = len(native_index_for_slim_index)

    psf_weighted_data = np.zeros((image_pixels,))

    weight_map_native = image_native / noise_map_native**2.0

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        value = 0.0

        for k0_y in range(kernel_native.shape[0]):
            for k0_x in range(kernel_native.shape[1]):
                weight_value = weight_map_native[
                    ip0_y + k0_y + kernel_shift_y, ip0_x + k0_x + kernel_shift_x
                ]

                if not np.isnan(weight_value):
                    value += kernel_native[k0_y, k0_x] * weight_value

        psf_weighted_data[ip0] = value

    return psf_weighted_data


@numba_util.jit()
def psf_precision_operator_from(
    noise_map_native: np.ndarray, kernel_native: np.ndarray, native_index_for_slim_index
) -> np.ndarray:
    """
    The `psf_precision_operator` matrix is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF
    convolution of every pair of image pixels given the noise map. This can be used to efficiently compute the
    curvature matrix via the mappings between image and source pixels, in a way that omits having to perform the
    PSF convolution on every individual source pixel. This provides a significant speed up for inversions of imaging
    datasets.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `psf_precision_operator_sparse_from` describes a compressed representation that overcomes this hurdles. It is
    advised `psf_precision_operator` and this method are only used for testing.

    Parameters
    ----------
    noise_map_native
        The two dimensional masked noise-map of values which psf_precision_operator is computed from.
    kernel_native
        The two dimensional PSF kernel that psf_precision_operator encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the noise map that enables efficient calculation of
        the curvature matrix.
    """
    image_pixels = len(native_index_for_slim_index)

    psf_precision_operator = np.zeros((image_pixels, image_pixels))

    for ip0 in range(psf_precision_operator.shape[0]):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        for ip1 in range(ip0, psf_precision_operator.shape[1]):
            ip1_y, ip1_x = native_index_for_slim_index[ip1]

            psf_precision_operator[ip0, ip1] += psf_precision_value_from(
                value_native=noise_map_native,
                kernel_native=kernel_native,
                ip0_y=ip0_y,
                ip0_x=ip0_x,
                ip1_y=ip1_y,
                ip1_x=ip1_x,
            )

    for ip0 in range(psf_precision_operator.shape[0]):
        for ip1 in range(ip0, psf_precision_operator.shape[1]):
            psf_precision_operator[ip1, ip0] = psf_precision_operator[ip0, ip1]

    return psf_precision_operator


@numba_util.jit()
def psf_precision_operator_sparse_from(
    noise_map_native: np.ndarray, kernel_native: np.ndarray, native_index_for_slim_index
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The matrix `psf_precision_operator` is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF
    convolution of every pair of image pixels on the noise map. This can be used to efficiently compute the
    curvature matrix via the mappings between image and source pixels, in a way that omits having to repeat the PSF
    convolution on every individual source pixel. This provides a significant speed up for inversions of imaging
    datasets.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations slow. This methods creates
    a sparse matrix that can compute the matrix `psf_precision_operator` efficiently, albeit the linear algebra calculations
    in PyAutoArray bypass this matrix entirely to go straight to the curvature matrix.

    for imaging data, psf_precision_operator is a sparse matrix, whereby non-zero entries are only contained for pairs of image pixels
    where the two pixels overlap due to the kernel size. For example, if the kernel size is (11, 11) and two image
    pixels are separated by more than 20 pixels, the kernel will never convolve flux between the two pixels. Two image
    pixels will only share a convolution if they are within `kernel_overlap_size = 2 * kernel_shape - 1` pixels within
    one another.

    Thus, a `psf_precision_operator_preload` matrix of dimensions [image_pixels, kernel_overlap_size ** 2] can be computed
    which significantly reduces the memory consumption by removing the sparsity. Because the dimensions of the second
    axes is no longer `image_pixels`, a second matrix `psf_precision_indexes` must also be computed containing the slim image
    pixel indexes of every entry of `psf_precision_operator`.

    In order for the preload to store half the number of values, owing to the symmetry of the `psf_precision_operator`
    matrix, the image pixel pairs corresponding to the same image pixel are divided by two. This ensures that when the
    curvature matrix is computed these pixels are not double-counted.

    The values stored in `psf_precision_operator_preload` represent the convolution of overlapping noise-maps given the
    PSF kernel. It is common for many values to be neglibly small. Removing these values can speed up the inversion
    and reduce memory at the expense of a numerically irrelevent change of solution.

    This matrix can then be used to compute the `curvature_matrix` in a memory efficient way that exploits the sparsity
    of the linear algebra.

    Parameters
    ----------
    noise_map_native
        The two dimensional masked noise-map of values which `psf_precision_operator` is computed from.
    signal_to_noise_map_native
        The two dimensional masked signal-to-noise-map from which the threshold discarding low S/N image pixel
        pairs is used.
    kernel_native
        The two dimensional PSF kernel that `psf_precision_operator` encodes the convolution of.
    native_index_for_slim_index
        An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

    Returns
    -------
    ndarray
        A matrix that encodes the PSF convolution values between the noise map that enables efficient calculation of
        the curvature matrix, where the dimensions are reduced to save memory.
    """

    image_pixels = len(native_index_for_slim_index)

    kernel_overlap_size = (2 * kernel_native.shape[0] - 1) * (
        2 * kernel_native.shape[1] - 1
    )

    psf_precision_operator_tmp = np.zeros((image_pixels, kernel_overlap_size))
    psf_precision_indexes_tmp = np.zeros((image_pixels, kernel_overlap_size))
    psf_precision_lengths = np.zeros(image_pixels)

    for ip0 in range(image_pixels):
        ip0_y, ip0_x = native_index_for_slim_index[ip0]

        kernel_index = 0

        for ip1 in range(ip0, psf_precision_operator_tmp.shape[0]):
            ip1_y, ip1_x = native_index_for_slim_index[ip1]

            noise_value = psf_precision_value_from(
                value_native=noise_map_native,
                kernel_native=kernel_native,
                ip0_y=ip0_y,
                ip0_x=ip0_x,
                ip1_y=ip1_y,
                ip1_x=ip1_x,
            )

            if ip0 == ip1:
                noise_value /= 2.0

            if noise_value > 0.0:
                psf_precision_operator_tmp[ip0, kernel_index] = noise_value
                psf_precision_indexes_tmp[ip0, kernel_index] = ip1
                kernel_index += 1

        psf_precision_lengths[ip0] = kernel_index

    psf_precision_total_pairs = int(np.sum(psf_precision_lengths))

    psf_precision_operator = np.zeros((psf_precision_total_pairs))
    psf_precision_indexes = np.zeros((psf_precision_total_pairs))

    index = 0

    for i in range(image_pixels):
        for data_index in range(int(psf_precision_lengths[i])):
            psf_precision_operator[index] = psf_precision_operator_tmp[i, data_index]
            psf_precision_indexes[index] = psf_precision_indexes_tmp[i, data_index]

            index += 1

    return (psf_precision_operator, psf_precision_indexes, psf_precision_lengths)


@numba_util.jit()
def psf_precision_value_from(
    value_native: np.ndarray,
    kernel_native: np.ndarray,
    ip0_y,
    ip0_x,
    ip1_y,
    ip1_x,
    renormalize=False,
) -> float:
    """
    Compute the value of an entry of the `psf_precision_operator` matrix, where this entry encodes the PSF convolution of
    the noise-map between two image pixels.

    The calculation is performed by over-laying the PSF kernel over two noise-map pixels in 2D. For all pixels where
    the two overlaid PSF kernels overlap, the following calculation is performed for every noise map value:

    `value = kernel_value_0 * kernel_value_1 * (1.0 / noise_value) ** 2.0`

    This calculation infers the fraction of flux that every PSF convolution will move between each pair of noise-map
    pixels and can therefore be used to efficiently calculate the curvature_matrix that is used in the linear algebra
    calculation of an inversion.

    The sum of all values where kernel pixels overlap is returned to give the `psf_precision_operator` value.

    Parameters
    ----------
    value_native
        A two dimensional masked array of values (e.g. a noise-map, signal to noise map) which the psf_precision_operator curvature
        values are computed from.
    kernel_native
        The two dimensional PSF kernel that psf_precision_operator encodes the convolution of.
    ip0_y
        The y index of the first image pixel in the image pixel pair.
    ip0_x
        The x index of the first image pixel in the image pixel pair.
    ip1_y
        The y index of the second image pixel in the image pixel pair.
    ip1_x
        The x index of the second image pixel in the image pixel pair.

    Returns
    -------
    float
        The psf_precision_operator value that encodes the value of PSF convolution between a pair of image pixels.

    """

    curvature_value = 0.0

    kernel_shift_y = -(kernel_native.shape[1] // 2)
    kernel_shift_x = -(kernel_native.shape[0] // 2)

    ip_y_offset = ip0_y - ip1_y
    ip_x_offset = ip0_x - ip1_x

    if (
        ip_y_offset < 2 * kernel_shift_y
        or ip_y_offset > -2 * kernel_shift_y
        or ip_x_offset < 2 * kernel_shift_x
        or ip_x_offset > -2 * kernel_shift_x
    ):
        return curvature_value

    kernel_pixels = kernel_native.shape[0] * kernel_native.shape[1]
    kernel_count = 0

    for k0_y in range(kernel_native.shape[0]):
        for k0_x in range(kernel_native.shape[1]):
            iy = ip0_y + k0_y + kernel_shift_y
            ix = ip0_x + k0_x + kernel_shift_x

            # numba @jit() does not bounds-check array reads. Without this
            # guard, a kernel position that lands off the noise-map array
            # (e.g. a mask pixel within `kernel_shift` of the array edge)
            # silently reads uninitialized memory, producing astronomical
            # or non-finite contributions that poison the entire
            # psf_precision_operator and downstream curvature matrix.
            if (
                iy < 0
                or iy >= value_native.shape[0]
                or ix < 0
                or ix >= value_native.shape[1]
            ):
                continue

            value = value_native[iy, ix]

            if value > 0.0:
                k1_y = k0_y + ip_y_offset
                k1_x = k0_x + ip_x_offset

                if (
                    k1_y >= 0
                    and k1_x >= 0
                    and k1_y < kernel_native.shape[0]
                    and k1_x < kernel_native.shape[1]
                ):
                    kernel_count += 1

                    kernel_value_0 = kernel_native[k0_y, k0_x]
                    kernel_value_1 = kernel_native[k1_y, k1_x]

                    curvature_value += (
                        kernel_value_0 * kernel_value_1 * (1.0 / value) ** 2.0
                    )

    if renormalize:
        if kernel_count > 0:
            curvature_value *= kernel_pixels / kernel_count

    return curvature_value


@numba_util.jit()
def data_vector_via_blurred_mapping_matrix_from(
    blurred_mapping_matrix: np.ndarray, image: np.ndarray, noise_map: np.ndarray
) -> np.ndarray:
    r"""
    Returns the data vector `D` from a blurred mapping matrix `f` and the 1D image `d` and 1D noise-map $\sigma$`
    (see Warren & Dye 2003).

    Parameters
    ----------
    blurred_mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.
    image
        Flattened 1D array of the observed image the inversion is fitting.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    """

    data_shape = blurred_mapping_matrix.shape

    data_vector = np.zeros(data_shape[1])

    for data_index in range(data_shape[0]):
        for pix_index in range(data_shape[1]):
            data_vector[pix_index] += (
                image[data_index]
                * blurred_mapping_matrix[data_index, pix_index]
                / (noise_map[data_index] ** 2.0)
            )

    return data_vector


@numba_util.jit()
def data_vector_via_psf_weighted_data_from(
    psf_weighted_data: np.ndarray,
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    pix_pixels: int,
) -> np.ndarray:
    r"""
    Returns the data vector `D` from the `psf_weighted_data` matrix (see `psf_weighted_data_from`), which encodes the
    the 1D image `d` and 1D noise-map values `\sigma` (see Warren & Dye 2003).

    This uses the array `data_to_pix_unique`, which describes the unique mappings of every set of image sub-pixels to
    pixelization pixels and `data_weights`, which describes how many sub-pixels uniquely map to each pixelization
    pixels (see `data_slim_to_pixelization_unique_from`).

    Parameters
    ----------
    psf_weighted_data
        A matrix that encodes the PSF convolution values between the imaging divided by the noise map**2 that enables
        efficient calculation of the data vector.
    data_to_pix_unique
        An array that maps every data pixel index (e.g. the masked image pixel indexes in 1D) to its unique set of
        pixelization pixel indexes (see `data_slim_to_pixelization_unique_from`).
    data_weights
        For every unique mapping between a set of data sub-pixels and a pixelization pixel, the weight of these mapping
        based on the number of sub-pixels that map to pixelization pixel.
    pix_lengths
        A 1D array describing how many unique pixels each data pixel maps too, which is used to iterate over
        `data_to_pix_unique` and `data_weights`.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.
    """

    data_pixels = psf_weighted_data.shape[0]

    data_vector = np.zeros(pix_pixels)

    for data_0 in range(data_pixels):
        for pix_0_index in range(pix_lengths[data_0]):
            data_0_weight = data_weights[data_0, pix_0_index]
            pix_0 = data_to_pix_unique[data_0, pix_0_index]

            data_vector[pix_0] += data_0_weight * psf_weighted_data[data_0]

    return data_vector


@numba_util.jit()
def curvature_matrix_with_added_to_diag_from(
    curvature_matrix: np.ndarray,
    value: float,
    no_regularization_index_list: Optional[List] = None,
) -> np.ndarray:
    """
    It is common for the `curvature_matrix` computed to not be positive-definite, leading for the inversion
    via `np.linalg.solve` to fail and raise a `LinAlgError`.

    In many circumstances, adding a small numerical value of `1.0e-8` to the diagonal of the `curvature_matrix`
    makes it positive definite, such that the inversion is performed without raising an error.

    This function adds this numerical value to the diagonal of the curvature matrix.

    Parameters
    ----------
    curvature_matrix
        The curvature matrix which is being constructed in order to solve a linear system of equations.
    """

    for i in no_regularization_index_list:
        curvature_matrix[i, i] += value

    return curvature_matrix


@numba_util.jit()
def curvature_matrix_mirrored_from(
    curvature_matrix: np.ndarray,
) -> np.ndarray:
    curvature_matrix_mirrored = np.zeros(
        (curvature_matrix.shape[0], curvature_matrix.shape[1])
    )

    for i in range(curvature_matrix.shape[0]):
        for j in range(curvature_matrix.shape[1]):
            if curvature_matrix[i, j] != 0:
                curvature_matrix_mirrored[i, j] = curvature_matrix[i, j]
                curvature_matrix_mirrored[j, i] = curvature_matrix[i, j]
            if curvature_matrix[j, i] != 0:
                curvature_matrix_mirrored[i, j] = curvature_matrix[j, i]
                curvature_matrix_mirrored[j, i] = curvature_matrix[j, i]

    return curvature_matrix_mirrored


@numba_util.jit()
def curvature_matrix_via_sparse_operator_from(
    psf_precision_operator: np.ndarray,
    psf_precision_indexes: np.ndarray,
    psf_precision_lengths: np.ndarray,
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    pix_pixels: int,
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) by computing it using `psf_precision_operator`
    (see `psf_precision_operator_from`) for an imaging inversion.

    To compute the curvature matrix via psf_precision_operator the following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * psf_precision_operator * mapping matrix

    This function speeds this calculation up in two ways:

    1) Instead of using `psf_precision_operator` (dimensions [image_pixels, image_pixels] it uses a compressed
     `psf_precision_operator` (dimensions [image_pixels, kernel_overlap]). The massive reduction in the size of this matrix in memory allows for much fast
    computation.

    2) It omits the `mapping_matrix` and instead uses directly the 1D vector that maps every image pixel to a source
    pixel `native_index_for_slim_index`. This exploits the sparsity in the `mapping_matrix` to directly
    compute the `curvature_matrix` (e.g. it condenses the triple matrix multiplication into a double for loop!).

    Parameters
    ----------
    psf_precision_operator
        A matrix that precomputes the values for fast computation of the curvature matrix in a memory efficient way.
    psf_precision_indexes
        The image-pixel indexes of the values stored in the w tilde preload matrix, which are used to compute
        the weights of the data values when computing the curvature matrix.
    psf_precision_lengths
        The number of image pixels in every row of `psf_precision_operator`, which is iterated over when computing the
        curvature matrix.
    data_to_pix_unique
        An array that maps every data pixel index (e.g. the masked image pixel indexes in 1D) to its unique set of
        pixelization pixel indexes (see `data_slim_to_pixelization_unique_from`).
    data_weights
        For every unique mapping between a set of data sub-pixels and a pixelization pixel, the weight of these mapping
        based on the number of sub-pixels that map to pixelization pixel.
    pix_lengths
        A 1D array describing how many unique pixels each data pixel maps too, which is used to iterate over
        `data_to_pix_unique` and `data_weights`.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    data_pixels = psf_precision_lengths.shape[0]

    curvature_matrix = np.zeros((pix_pixels, pix_pixels))

    curvature_index = 0

    for data_0 in range(data_pixels):
        for data_1_index in range(psf_precision_lengths[data_0]):
            data_1 = psf_precision_indexes[curvature_index]
            psf_precision_value = psf_precision_operator[curvature_index]

            for pix_0_index in range(pix_lengths[data_0]):
                data_0_weight = data_weights[data_0, pix_0_index]
                pix_0 = data_to_pix_unique[data_0, pix_0_index]

                for pix_1_index in range(pix_lengths[data_1]):
                    data_1_weight = data_weights[data_1, pix_1_index]
                    pix_1 = data_to_pix_unique[data_1, pix_1_index]

                    curvature_matrix[pix_0, pix_1] += (
                        data_0_weight * data_1_weight * psf_precision_value
                    )

            curvature_index += 1

    for i in range(pix_pixels):
        for j in range(i, pix_pixels):
            curvature_matrix[i, j] += curvature_matrix[j, i]

    for i in range(pix_pixels):
        for j in range(i, pix_pixels):
            curvature_matrix[j, i] = curvature_matrix[i, j]

    return curvature_matrix


@numba_util.jit()
def curvature_matrix_off_diags_via_sparse_operator_from(
    psf_precision_operator: np.ndarray,
    psf_precision_indexes: np.ndarray,
    psf_precision_lengths: np.ndarray,
    data_to_pix_unique_0: np.ndarray,
    data_weights_0: np.ndarray,
    pix_lengths_0: np.ndarray,
    pix_pixels_0: int,
    data_to_pix_unique_1: np.ndarray,
    data_weights_1: np.ndarray,
    pix_lengths_1: np.ndarray,
    pix_pixels_1: int,
) -> np.ndarray:
    """
    Returns the off diagonal terms in the curvature matrix `F` (see Warren & Dye 2003) by computing them
    using `psf_precision_operator` (see `psf_precision_operator_from`) for an imaging inversion.

    When there is more than one mapper in the inversion, its `mapping_matrix` is extended to have dimensions
    [data_pixels, sum(source_pixels_in_each_mapper)]. The curvature matrix therefore will have dimensions
    [sum(source_pixels_in_each_mapper), sum(source_pixels_in_each_mapper)].

    To compute the curvature matrix via the psf precision operator following matrix multiplication is normally performed:

    curvature_matrix = mapping_matrix.T * psf_precision_operator * mapping matrix

    When the `mapping_matrix` consists of multiple mappers from different planes, this means that shared data mappings
    between source-pixels in different mappers must be accounted for when computing the `curvature_matrix`. These
    appear as off-diagonal terms in the overall curvature matrix.

    This function evaluates these off-diagonal terms, by using the w-tilde curvature preloads and the unique
    data-to-pixelization mappings of each mapper. It behaves analogous to the
    function `curvature_matrix_via_sparse_operator_from`.

    Parameters
    ----------
    psf_precision_operator
        A matrix that precomputes the values for fast computation of the curvature matrix in a memory efficient way.
    psf_precision_indexes
        The image-pixel indexes of the values stored in the w tilde preload matrix, which are used to compute
        the weights of the data values when computing the curvature matrix.
    psf_precision_lengths
        The number of image pixels in every row of `psf_precision_operator`, which is iterated over when computing the
        curvature matrix.
    data_to_pix_unique
        An array that maps every data pixel index (e.g. the masked image pixel indexes in 1D) to its unique set of
        pixelization pixel indexes (see `data_slim_to_pixelization_unique_from`).
    data_weights
        For every unique mapping between a set of data sub-pixels and a pixelization pixel, the weight of these mapping
        based on the number of sub-pixels that map to pixelization pixel.
    pix_lengths
        A 1D array describing how many unique pixels each data pixel maps too, which is used to iterate over
        `data_to_pix_unique` and `data_weights`.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """

    data_pixels = psf_precision_lengths.shape[0]

    curvature_matrix = np.zeros((pix_pixels_0, pix_pixels_1))

    curvature_index = 0

    for data_0 in range(data_pixels):
        for data_1_index in range(psf_precision_lengths[data_0]):
            data_1 = psf_precision_indexes[curvature_index]
            psf_precision_value = psf_precision_operator[curvature_index]

            for pix_0_index in range(pix_lengths_0[data_0]):
                data_0_weight = data_weights_0[data_0, pix_0_index]
                pix_0 = data_to_pix_unique_0[data_0, pix_0_index]

                for pix_1_index in range(pix_lengths_1[data_1]):
                    data_1_weight = data_weights_1[data_1, pix_1_index]
                    pix_1 = data_to_pix_unique_1[data_1, pix_1_index]

                    curvature_matrix[pix_0, pix_1] += (
                        data_0_weight * data_1_weight * psf_precision_value
                    )

            curvature_index += 1

    return curvature_matrix


@numba_util.jit()
def curvature_matrix_off_diags_via_mapper_and_linear_func_curvature_vector_from(
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    pix_pixels: int,
    curvature_weights: np.ndarray,  # shape (n_unmasked, n_funcs)
    mask: np.ndarray,  # shape (ny, nx), bool
    psf_kernel: np.ndarray,  # shape (ky, kx)
) -> np.ndarray:
    """
    Returns the off-diagonal terms in the curvature matrix `F` (see Warren & Dye 2003)
    between a mapper object and a linear func object, using the unique mappings between
    data pixels and pixelization pixels.

    This version applies the PSF directly as a 2D convolution kernel. The curvature
    weights of the linear function object (values of the linear function divided by the
    noise-map squared) are expanded into the native 2D image grid, convolved with the PSF
    kernel, and then remapped back to the 1D slim representation.

    For each unique mapping between a data pixel and a pixelization pixel, the convolved
    curvature weights at that data pixel are multiplied by the mapping weights and
    accumulated into the off-diagonal block of the curvature matrix. This accounts for
    sub-pixel mappings between data pixels and pixelization pixels.

    Parameters
    ----------
    data_to_pix_unique
        An array that maps every data pixel index (e.g. the masked image pixel indexes in 1D)
        to its unique set of pixelization pixel indexes (see `data_slim_to_pixelization_unique_from`).
    data_weights
        For every unique mapping between a set of data sub-pixels and a pixelization pixel,
        the weight of this mapping based on the number of sub-pixels that map to the pixelization pixel.
    pix_lengths
        A 1D array describing how many unique pixels each data pixel maps to. Used to iterate over
        `data_to_pix_unique` and `data_weights`.
    pix_pixels
        The total number of pixels in the pixelization that reconstructs the data.
    curvature_weights
        The operated values of the linear function divided by the noise-map squared, with shape
        [n_unmasked_data_pixels, n_linear_func_pixels].
    mask
        A 2D boolean mask of shape (ny, nx) indicating which pixels are in the data region.
    psf_kernel
        The PSF kernel in its native 2D form, centered (odd dimensions recommended).

    Returns
    -------
    ndarray
        The off-diagonal block of the curvature matrix `F` (see Warren & Dye 2003),
        with shape [pix_pixels, n_linear_func_pixels].
    """
    data_pixels = data_weights.shape[0]
    n_funcs = curvature_weights.shape[1]
    ny, nx = mask.shape

    # Expand curvature weights into native grid
    curvature_native = np.zeros((ny, nx, n_funcs))
    unmasked_coords = np.argwhere(~mask)
    for i, (y, x) in enumerate(unmasked_coords):
        for f in range(n_funcs):
            curvature_native[y, x, f] = curvature_weights[i, f]

    # Convolve in native space
    blurred_native = convolve_with_kernel_native(curvature_native, psf_kernel)

    # Map back to slim representation
    blurred_slim = np.zeros((data_pixels, n_funcs))
    for i, (y, x) in enumerate(unmasked_coords):
        for f in range(n_funcs):
            blurred_slim[i, f] = blurred_native[y, x, f]

    # Accumulate into off_diag
    off_diag = np.zeros((pix_pixels, n_funcs))
    for data_0 in range(data_pixels):
        for pix_0_index in range(pix_lengths[data_0]):
            data_0_weight = data_weights[data_0, pix_0_index]
            pix_0 = data_to_pix_unique[data_0, pix_0_index]
            for f in range(n_funcs):
                off_diag[pix_0, f] += data_0_weight * blurred_slim[data_0, f]

    return off_diag


@numba_util.jit()
def convolve_with_kernel_native(curvature_native, psf_kernel):
    """
    Convolve each function slice of curvature_native with psf_kernel using direct sliding window.

    Parameters
    ----------
    curvature_native : ndarray (ny, nx, n_funcs)
        Curvature weights expanded to the native grid, 0 in masked regions.
    psf_kernel : ndarray (ky, kx)
        The PSF kernel.

    Returns
    -------
    blurred_native : ndarray (ny, nx, n_funcs)
        The curvature weights convolved with the PSF.
    """
    ny, nx, n_funcs = curvature_native.shape
    ky, kx = psf_kernel.shape
    cy, cx = ky // 2, kx // 2  # kernel center

    blurred_native = np.zeros_like(curvature_native)

    for f in range(n_funcs):  # parallelize over functions
        for y in range(ny):
            for x in range(nx):
                acc = 0.0
                for dy in range(ky):
                    for dx in range(kx):
                        yy = y + dy - cy
                        xx = x + dx - cx
                        if 0 <= yy < ny and 0 <= xx < nx:
                            acc += psf_kernel[dy, dx] * curvature_native[yy, xx, f]
                blurred_native[y, x, f] = acc
    return blurred_native


@numba_util.jit()
def mapped_reconstructed_data_via_image_to_pix_unique_from(
    data_to_pix_unique: np.ndarray,
    data_weights: np.ndarray,
    pix_lengths: np.ndarray,
    reconstruction: np.ndarray,
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurred mapping matrix `f` and solution vector *S*.

    Parameters
    ----------
    mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """

    data_pixels = data_to_pix_unique.shape[0]

    mapped_reconstructed_operated_data = np.zeros(data_pixels)

    for data_0 in range(data_pixels):
        for pix_0 in range(pix_lengths[data_0]):
            pix_for_data = data_to_pix_unique[data_0, pix_0]

            mapped_reconstructed_operated_data[data_0] += (
                data_weights[data_0, pix_0] * reconstruction[pix_for_data]
            )

    return mapped_reconstructed_operated_data


class SparseLinAlgImagingNumba:
    def __init__(
        self,
        psf_precision_operator_sparse: np.ndarray,
        indexes: np.ndim,
        lengths: np.ndarray,
        noise_map: np.ndarray,
        psf: np.ndarray,
        mask: np.ndarray,
    ):
        """
        Packages together all derived data quantities necessary to fit `Imaging` data using an ` Inversion` via the
        sparse linear algebra formalism.

        The sparse linear algebra formalism performs linear algebra formalism in a way that speeds up the construction of the
        simultaneous linear equations by bypassing the construction of a `mapping_matrix` and precomputing the
        blurring operations performed using the imaging's PSF.

        Parameters
        ----------
        psf_precision_operator
            A matrix which uses the imaging's noise-map and PSF to preload as much of the computation of the
            curvature matrix as possible.
        indexes
            The image-pixel indexes of the curvature preload matrix, which are used to compute the curvature matrix
            efficiently when performing an inversion.
        lengths
            The lengths of how many indexes each curvature preload contains, again used to compute the curvature
            matrix efficienctly.
        """

        self.psf_precision_operator_sparse = psf_precision_operator_sparse
        self.indexes = indexes
        self.lengths = lengths
        self.noise_map = noise_map
        self.psf = psf
        self.mask = mask

    @property
    def psf_precision_operator(self):
        """
        The matrix `psf_precision_operator` is a matrix of dimensions [image_pixels, image_pixels] that encodes the PSF
        convolution of every pair of image pixels given the noise map. This can be used to efficiently compute the
        curvature matrix via the mappings between image and source pixels, in a way that omits having to perform the
        PSF convolution on every individual source pixel. This provides a significant speed up for inversions of imaging
        datasets.

        The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
        making it impossible to store in memory and its use in linear algebra calculations extremely. The method
        `psf_precision_operator_sparse_from` describes a compressed representation that overcomes this hurdles. It is
        advised `psf_precision_operator` and this method are only used for testing.

        Parameters
        ----------
        noise_map_native
            The two dimensional masked noise-map of values which psf_precision_operator is computed from.
        kernel_native
            The two dimensional PSF kernel that psf_precision_operator encodes the convolution of.
        native_index_for_slim_index
            An array of shape [total_x_pixels*sub_size] that maps pixels from the slimmed array to the native array.

        Returns
        -------
        ndarray
            A matrix that encodes the PSF convolution values between the noise map that enables efficient calculation of
            the curvature matrix.
        """

        return psf_precision_operator_from(
            noise_map_native=self.noise_map.native.array,
            kernel_native=self.psf.kernel.native.array,
            native_index_for_slim_index=np.array(
                self.mask.derive_indexes.native_for_slim
            ).astype("int"),
        )
