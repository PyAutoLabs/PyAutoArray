import numpy as np

from autoarray import exc


def array_with_new_shape(array, new_shape):
    """Resize an input array around its centre to a new input shape.

    If a new_shape dimension is smaller than the array's current dimension, the data at the edges is trimmed and
    removed. If it is larger, the data is padded with zeros.

    If the array has even sized dimensions, the central pixel around which data is trimmed / padded is chosen as
    the top-left pixel of the central quadrant of pixels.

    Parameters
    ----------
    array
        The array which is trimmed / padded to a new 2D shape.
    new_shape
        The new 2D shape of the array.
    """

    return array.resized_from(new_shape=new_shape)


def array_eps_to_counts(array_eps, exposure_time_map):
    """
    Convert an array in units of electrons per second to counts, using an exposure time map containing the exposure
    time at every point in the array.

    The conversion from electrons per second to counts is:

    [Counts] = [EPS] * [Exposure_time]

    Parameters
    ----------
    array_eps
        The array which is converted from electrons per seconds to counts.
    exposure_time_map
        The exposure time at every data-point of the array.
    """
    return array_eps * exposure_time_map


def array_counts_to_eps(array_counts, exposure_time_map):
    """
    Convert an array in units of counts to electrons per second, using an exposure time map containing the exposure
    time at every point in the array.

    The conversion from counts to electrons per second is:

    [EPS] = [Counts] / [Exposure_time]

    Parameters
    ----------
    array_counts
        The array which is converted from counts to electrons per seconds.
    exposure_time_map
        The exposure time at every data-point of the array.
    """
    return array_counts / exposure_time_map


def array_eps_to_adus(array_eps, exposure_time_map, gain):
    """
    Convert an array in units of electrons per second to adus, using an exposure time map containing the exposure
    time at every point in the array and the instrument gain.

    The conversion from electrons per second to ADUs is:

    [ADUs] = [EPS] * [Exposure_time] / [Gain]

    Parameters
    ----------
    array_eps
        The array which is converted from electrons per seconds to adus.
    exposure_time_map
        The exposure time at every data-point of the array.
    gain
        The gain of the instrument used in the conversion to / from counts and ADUs.
    """
    return (array_eps * exposure_time_map) / gain


def array_adus_to_eps(array_adus, exposure_time_map, gain):
    """
    Convert an array in units of electrons per second to adus, using an exposure time map containing the exposure
    time at every point in the array and the instrument gain.

    The conversion from ADUs to electrons per second is:

    [EPS] = [Counts] * [Gain] / [Exposure_time]

    Parameters
    ----------
    array_adus
        The array which is converted from adus to electrons per seconds
    exposure_time_map
        The exposure time at every data-point of the array.
    gain
        The gain of the instrument used in the conversion to / from counts and ADUs.
    """
    return (gain * array_adus) / exposure_time_map


def array_counts_to_counts_per_second(array_counts, exposure_time):
    if exposure_time is None:
        raise exc.ArrayException(
            "Cannot convert a Frame2D to units counts per second without an exposure time attribute (exposure_time = None)."
        )

    return array_counts / exposure_time


def array_with_random_uniform_values_added(array, upper_limit=0.001):
    """
    Add random values drawn from a uniform distribution between zero and an input upper limit to an array.

    The current use-case of this function is adding small random values to a noise-map that is constant (e.g. all noise
    map values are the same). Constant noise-maps have been found to create "broken" inversions where the source is
    reconstructed as a set of constant values.

    Parameters
    ----------
    data
        The array that the uniform noise values are added to.
    upper_limit
        The upper limit of the uniform distribution from which the values are drawn.
    """
    return array + upper_limit * np.random.uniform(size=array.shape_slim)


def noise_map_via_data_eps_and_exposure_time_map_from(data_eps, exposure_time_map):
    """
    Estimate the noise-map value in every data-point, by converting the data to units of counts and taking the
    square root of these values.

    For datasets that may have a background noise component, this function does not return the overall noise-map if the
    data is background subtracted. In this case, the returned noise-map is the Poisson noise-map.

    This function assumes the input data is in electrons per second and returns the noise-map in electrons per second.

    Parameters
    ----------
    data_eps
        The data in electrons second used to estimate the Poisson noise in every data point.
    exposure_time_map
        The exposure time at every data-point of the data.
    """
    return data_eps.with_new_array(
        np.abs(data_eps.array * exposure_time_map.array) ** 0.5
        / exposure_time_map.array
    )


def noise_map_via_weight_map_from(weight_map):
    """
    Setup the noise-map from a weight map, which is a form of noise-map that comes via HST image-reduction and
    the software package MultiDrizzle.

    The variance in each pixel is computed as:

    Variance = 1.0 / sqrt(weight_map).

    The weight map may contain zeros, in which case the variances are converted to large values to omit them from \
    the analysis.

    Parameters
    ----------
    pixel_scales
        The (y,x) arcsecond-to-pixel units conversion factor of every pixel. If this is input as a `float`,
            it is converted to a (float, float).
    weight_map
        The weight-value of each pixel which is converted to a variance.
    """
    np.seterr(divide="ignore")
    noise_map = 1.0 / weight_map**0.5
    noise_map[noise_map > 1.0e8] = 1.0e8
    return noise_map


def noise_map_via_inverse_noise_map_from(inverse_noise_map):
    """
    Setup the noise-map from an inverse noise-map.

    The variance in each pixel is computed as:

    Variance = 1.0 / inverse_noise_map.

    Parameters
    ----------
    inverse_noise_map
        The inverse noise_map value of each pixel which is converted to a variance.
    """
    return 1.0 / inverse_noise_map


def noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from(
    data_eps, exposure_time_map, background_noise_map
):
    """
    Estimate the noise-map values in every data-point, by converting the data to units of counts, adding the
    background noise-map and taking the square root of these values.

    This function assumes the input data is in electrons per second and returns the noise-map in electrons per second.

    Parameters
    ----------
    data_eps
        The background sky subtracted data in electrons second used to estimate the Poisson noise in every data point.
    exposure_time_map
        The exposure time at every data-point of the data.
    background_noise_map
        The RMS standard deviation error in every data point due to a background component of the noise-map in units
        of electrons per second.
    """
    return (
        (
            abs(data_eps * exposure_time_map)
            + (background_noise_map * exposure_time_map) ** 2
        )
        ** 0.5
    ) / exposure_time_map


def noise_map_via_data_eps_exposure_time_map_and_background_variances_from(
    data_eps, exposure_time_map, background_variances
):
    """
    This is the same as `noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from`, however the
    input `background_variances` are used instead.

    Parameters
    ----------
    data_eps
        The background sky subtracted data in electrons second used to estimate the Poisson noise in every data point.
    exposure_time_map
        The exposure time at every data-point of the data.
    background_noise_map
        The variance in every data point due to a background component of the noise-map in units
        of electrons per second.
    """
    return (
        (abs(data_eps * exposure_time_map) + (background_variances * exposure_time_map))
        ** 0.5
    ) / exposure_time_map


def edges_from(image, no_edges):
    """
    Extract the edges of an input image and return them as a concatenated 1D ndarray.

    These edges are typically empty regions of an image that may contain the background sky, which can be used
    for estimating the background sky level or noise.

    Parameters
    ----------
    image
        The image whose edge values are used to estimate the background noise.
    no_edges
        Number of edges used to estimate the background level.
    """
    edges = []

    for edge_no in range(no_edges):
        top_edge = image.native.array[
            edge_no, edge_no : image.shape_native[1] - edge_no
        ]
        bottom_edge = image.native.array[
            image.shape_native[0] - 1 - edge_no,
            edge_no : image.shape_native[1] - edge_no,
        ]
        left_edge = image.native.array[
            edge_no + 1 : image.shape_native[0] - 1 - edge_no, edge_no
        ]
        right_edge = image.native.array[
            edge_no + 1 : image.shape_native[0] - 1 - edge_no,
            image.shape_native[1] - 1 - edge_no,
        ]

        edges = np.concatenate((edges, top_edge, bottom_edge, right_edge, left_edge))

    return edges


def background_sky_level_via_edges_from(image, no_edges):
    """
    Estimate the background sky level in an image using the data values at its edges. These edge values are extracted
    and their median is used to calculate the bakcground sky level.

    Parameters
    ----------
    image
        The image whose edge values are used to estimate the background noise.
    no_edges
        Number of edges used to estimate the background level.
    """
    edges = edges_from(image=image, no_edges=no_edges)

    return np.median(edges)


def background_noise_map_via_edges_from(image, no_edges):
    """
    Estimate the background noise level in an image using the data values at its edges. These edge values are binned
    into a histogram, with a Gaussian profile fitted to this histogram, such that its standard deviation (sigma) gives
    an estimate of the background noise.

    The background noise-map is returned on a 2D array the same dimensions as the image.

    Parameters
    ----------
    image
        The image whose edge values are used to estimate the background noise.
    no_edges
        Number of edges used to estimate the background level.
    """
    from scipy.stats import norm

    from autoarray.structures.arrays.uniform_2d import Array2D

    edges = edges_from(image=image, no_edges=no_edges)

    return Array2D.full(
        fill_value=norm.fit(edges)[1],
        shape_native=image.shape_native,
        pixel_scales=image.pixel_scales,
    )


def kernel_with_odd_dimensions_from(
    kernel, rescale_factor: float, normalize: bool = True
):
    """
    If the PSF kernel has one or two even-sized dimensions, return a PSF object where the kernel has odd-sized
    dimensions (odd-sized dimensions are required for 2D convolution).

    Kernels are rescaled using the scikit-image routine rescale, which performs rescaling via an interpolation
    routine. This may lead to loss of accuracy in the PSF kernel and it is advised that users, where possible,
    create their PSF on an odd-sized array using their data reduction pipelines that remove this approximation.

    Odd-sized kernels are often required for real space convolution operations
    (e.g. centered PSFs in imaging pipelines). If the kernel has one or two
    even-sized dimensions, they are rescaled (via interpolation) and padded
    so that both dimensions are odd.

    The kernel can also be scaled larger or smaller by changing
    ``rescale_factor``. Rescaling uses ``skimage.transform.rescale`` /
    ``resize``, which interpolate pixel values and may introduce small
    inaccuracies compared to native instrument PSFs. Where possible, users
    should generate odd-sized PSFs directly from data reduction.

    Parameters
    ----------
    rescale_factor
        The factor by which the kernel is rescaled. If this has a value of 1.0, the kernel is rescaled to the
        closest odd-sized dimensions (e.g. 20 -> 19). Higher / lower values scale to higher / lower dimensions.
    normalize
        Whether the PSF should be normalized after being rescaled.
    """

    from skimage.transform import resize, rescale

    try:
        kernel_rescaled = rescale(
            kernel,
            rescale_factor,
            anti_aliasing=False,
            mode="constant",
            channel_axis=None,
        )
    except TypeError:
        kernel_rescaled = rescale(
            kernel,
            rescale_factor,
            anti_aliasing=False,
            mode="constant",
        )

    if kernel_rescaled.shape[0] % 2 == 0 and kernel_rescaled.shape[1] % 2 == 0:
        kernel_rescaled = resize(
            kernel_rescaled,
            output_shape=(
                kernel_rescaled.shape[0] + 1,
                kernel_rescaled.shape[1] + 1,
            ),
            anti_aliasing=False,
            mode="constant",
        )

    elif kernel_rescaled.shape[0] % 2 == 0 and kernel_rescaled.shape[1] % 2 != 0:
        kernel_rescaled = resize(
            kernel_rescaled,
            output_shape=(kernel_rescaled.shape[0] + 1, kernel_rescaled.shape[1]),
            anti_aliasing=False,
            mode="constant",
        )

    elif kernel_rescaled.shape[0] % 2 != 0 and kernel_rescaled.shape[1] % 2 == 0:
        kernel_rescaled = resize(
            kernel_rescaled,
            output_shape=(kernel_rescaled.shape[0], kernel_rescaled.shape[1] + 1),
            anti_aliasing=False,
            mode="constant",
        )

    if normalize:
        kernel_rescaled = np.divide(kernel_rescaled, np.sum(kernel_rescaled))

    return kernel_rescaled


def exposure_time_map_via_exposure_time_and_background_noise_map_from(
    exposure_time, background_noise_map
):
    """
    Compute the exposure time map from the exposure time of the observation and the background noise-map.

    This function assumes the only source of noise in the background noise-map is due to a variable exposure time in
    every pixel due to effects like dithering, cosmic rays, etc.

    Parameters
    ----------
    exposure_time
        The total exposure time of the observation.
    background_noise_map
        The RMS standard deviation error in every data point due to a background component of the noise-map in units
        of electrons per second.
    """
    inverse_background_noise_map = 1.0 / background_noise_map
    relative_background_noise_map = inverse_background_noise_map / np.max(
        inverse_background_noise_map
    )
    return abs(exposure_time * relative_background_noise_map)


def setup_random_seed(seed):
    """Setup the random seed. If the input seed is -1, the code will use a random seed for every run. If it is \
    positive, that seed is used for all runs, thereby giving reproducible results.

    Parameters
    ----------
    seed
        The seed of the random number generator.
    """
    if seed == -1:
        seed = np.random.randint(
            0, int(1e9)
        )  # Use one seed, so all regions have identical column non-uniformity.
    np.random.seed(seed)


def poisson_noise_via_data_eps_from(data_eps, exposure_time_map, seed=-1):
    """
    Generate a two-dimensional poisson noise_maps-mappers from an image.

    Values are computed from a Poisson distribution using the image's input values in units of counts.

    Parameters
    ----------
    data_eps
        The 2D image, whose values in counts are used to draw Poisson noise_maps values.
    exposure_time_map : Union(ndarray, int)
        2D array of the exposure time in each pixel used to convert to / from counts and electrons per second.
    seed
        The seed of the random number generator, used for the random noise_maps maps.

    Returns
    -------
    poisson_noise_map: np.ndarray
        An array describing simulated poisson noise_maps
    """
    setup_random_seed(seed)

    image_counts = data_eps.array * exposure_time_map.array
    noisy_eps_array = (
        np.random.poisson(image_counts, data_eps.shape) / exposure_time_map.array
    )

    return data_eps.with_new_array(noisy_eps_array) - data_eps


def data_eps_with_poisson_noise_added(data_eps, exposure_time_map, seed=-1):
    """
    Generate a two-dimensional poisson noise_maps-mappers from an image.

    Values are computed from a Poisson distribution using the image's input values in units of counts.

    Parameters
    ----------
    data_eps
        The 2D image, whose values in counts are used to draw Poisson noise_maps values.
    exposure_time_map : Union(ndarray, int)
        2D array of the exposure time in each pixel used to convert to / from counts and electrons per second.
    seed
        The seed of the random number generator, used for the random noise_maps maps.

    Returns
    -------
    poisson_noise_map: np.ndarray
        An array describing simulated poisson noise_maps
    """
    return data_eps + poisson_noise_via_data_eps_from(
        data_eps=data_eps, exposure_time_map=exposure_time_map, seed=seed
    )


def gaussian_noise_via_shape_and_sigma_from(shape, sigma, seed=-1):
    """Generate a two-dimensional read noises-map, generating values from a Gaussian distribution with mean 0.0.

    Params
    ----------
    shape
        The (x,y) image_shape of the generated Gaussian noises map.
    read_noise
        Standard deviation of the 1D Gaussian that each noises value is drawn from
    seed
        The seed of the random number generator, used for the random noises maps.
    """
    if seed == -1:
        # Use one seed, so all regions have identical column non-uniformity.
        seed = np.random.randint(0, int(1e9))
    np.random.seed(seed)
    read_noise_map = np.random.normal(loc=0.0, scale=sigma, size=shape)
    return read_noise_map


def data_with_gaussian_noise_added(data, sigma, seed=-1):
    return data + gaussian_noise_via_shape_and_sigma_from(
        shape=data.shape, sigma=sigma, seed=seed
    )


def data_with_complex_gaussian_noise_added(data, sigma, seed=-1):
    gaussian_noise = gaussian_noise_via_shape_and_sigma_from(
        shape=(data.shape[0], 2), sigma=sigma, seed=seed
    )

    return data + gaussian_noise[:, 0] + 1.0j * gaussian_noise[:, 1]


def noise_map_with_signal_to_noise_limit_from(
    data, noise_map, signal_to_noise_limit, noise_limit_mask=None
):
    """
    Given data and its noise map, increase the noise-values of all data points which signal to noise is above an input
    `signal_to_noise_limit`, such that the signal to noise values do not exceed this limit.

    This may be performed for dataset data with extremely high signal-to-noise regions in the data which are poorly
    fit my the model. By downweighting their signal to noise values, the model-fit with focus on other parts of the
    data with low S/N.

    A `noise_limit_mask` can be input, such that only noise values corresponding to `False` entries are scaled to the
    capped value.

    Parameters
    ----------
    data
        The data values whose S/N is used to scale the noise-map.
    noise_map
        The noise-map whose values are increased to limit the S/N values.
    signal_to_noise_limit
        The value of signal-to-noise which noise values are increased to match, if the original value exceeds
        this S/N value.
    noise_limit_mask
        A mask where noise-map scaling is applied to all `False` entries.

    Returns
    -------
    The noise map with values scaled such that the signal-to-noise values do not exceed the limit value.
    """

    from autoarray.mask.mask_2d import Mask2D
    from autoarray.structures.arrays.uniform_1d import Array1D
    from autoarray.structures.arrays.uniform_2d import Array2D

    signal_to_noise_map = data / noise_map
    signal_to_noise_map[signal_to_noise_map < 0] = 0

    if noise_limit_mask is None:
        noise_limit_mask = np.full(fill_value=False, shape=data.shape_native)

    noise_map_limit = np.where(
        (signal_to_noise_map.native > signal_to_noise_limit)
        & (noise_limit_mask == False),
        np.abs(data.native.array) / signal_to_noise_limit,
        noise_map.native.array,
    )

    mask = Mask2D.all_false(
        shape_native=data.shape_native,
        pixel_scales=data.pixel_scales,
        origin=data.origin,
    )

    if len(noise_map.native) == 1:
        return Array1D(values=noise_map_limit, mask=mask)
    return Array2D(noise_map_limit, mask=mask)


def visibilities_noise_map_with_signal_to_noise_limit_from(
    data, noise_map, signal_to_noise_limit
):
    from autoarray.structures.visibilities import VisibilitiesNoiseMap

    # TODO : Refacotr into a util

    signal_to_noise_map_real = np.divide(np.real(data), np.real(noise_map))
    signal_to_noise_map_real[signal_to_noise_map_real < 0] = 0.0
    signal_to_noise_map_imag = np.divide(np.imag(data), np.imag(noise_map))
    signal_to_noise_map_imag[signal_to_noise_map_imag < 0] = 0.0

    signal_to_noise_map = signal_to_noise_map_real + 1j * signal_to_noise_map_imag

    noise_map_limit_real = np.where(
        np.real(signal_to_noise_map) > signal_to_noise_limit,
        np.real(data) / signal_to_noise_limit,
        np.real(noise_map),
    )

    noise_map_limit_imag = np.where(
        np.imag(signal_to_noise_map) > signal_to_noise_limit,
        np.imag(data) / signal_to_noise_limit,
        np.imag(noise_map),
    )

    return VisibilitiesNoiseMap(
        visibilities=noise_map_limit_real + 1j * noise_map_limit_imag
    )
