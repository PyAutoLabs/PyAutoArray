import numpy as np


def visibilities_from(
    image_1d: np.ndarray, grid_radians: np.ndarray, uv_wavelengths: np.ndarray, xp=np
) -> np.ndarray:
    """
    Compute complex visibilities from an input sky image using the Fourier transform,
    simulating the response of an astronomical radio interferometer.

    This function converts an image defined on a sky coordinate grid into its
    visibility-space representation, given a set of (u,v) spatial frequency
    coordinates (in wavelengths), as sampled by a radio interferometer.

    Parameters
    ----------
    image_1d
        The 1D flattened sky brightness values corresponding to each pixel in the grid.
    grid_radians
        The angular (y, x) positions of each image pixel in radians, matching image_1d.
    uv_wavelengths
        The (u, v) spatial frequencies in units of wavelengths, for each baseline
        of the interferometer.

    Returns
    -------
    visibilities
        The complex visibilities (Fourier components) corresponding to each
        (u, v) coordinate, representing the interferometer’s measurement.
    """

    # Compute the dot product for each pixel-uv pair
    phase = (
        -2.0
        * xp.pi
        * (
            xp.outer(grid_radians[:, 1], uv_wavelengths[:, 0])
            + xp.outer(grid_radians[:, 0], uv_wavelengths[:, 1])
        )
    )  # shape (n_pixels, n_vis)

    # Multiply image values with phase terms
    vis_real = image_1d[:, None] * xp.cos(phase)
    vis_imag = image_1d[:, None] * xp.sin(phase)

    # Sum over all pixels for each visibility
    visibilities = xp.sum(vis_real + 1j * vis_imag, axis=0)

    return visibilities


def image_direct_from(
    visibilities: np.ndarray, grid_radians: np.ndarray, uv_wavelengths: np.ndarray
) -> np.ndarray:
    """
    Reconstruct a real-valued sky image from complex interferometric visibilities
    using an inverse Fourier transform approximation.

    This function simulates the synthesis imaging equation of a radio interferometer
    by summing sinusoidal components across all (u, v) spatial frequencies.

    Parameters
    ----------
    visibilities
        The real and imaginary parts of the complex visibilities for each (u, v) point.

    grid_radians
        The angular (y, x) coordinates of each pixel in radians.

    uv_wavelengths
        The (u, v) spatial frequencies in units of wavelengths for each baseline.

    Returns
    -------
    image_1d
        The reconstructed real-valued image in sky coordinates.
    """
    # Compute the phase term for each (pixel, visibility) pair
    phase = (
        2.0
        * np.pi
        * (
            np.outer(grid_radians[:, 1], uv_wavelengths[:, 0])
            + np.outer(grid_radians[:, 0], uv_wavelengths[:, 1])
        )
    )

    real_part = np.dot(np.cos(phase), visibilities[:, 0])
    imag_part = np.dot(np.sin(phase), visibilities[:, 1])

    image_1d = real_part - imag_part

    return image_1d


def transformed_mapping_matrix_from(
    mapping_matrix,
    grid_radians,
    uv_wavelengths,
    xp=np,
    chunk_size: int = 256,
):
    """
    Computes the Fourier-transformed mapping matrix in chunks to avoid
    materialising large (n_image_pixels x n_visibilities) arrays.

    Parameters
    ----------
    mapping_matrix : (n_image_pixels, n_source_pixels)
    grid_radians : (n_image_pixels, 2)
    uv_wavelengths : (n_visibilities, 2)
    xp : np or jax.numpy
    chunk_size : int
        Number of visibilities per chunk.

    Returns
    -------
    transformed_matrix : (n_visibilities, n_source_pixels), complex
    """
    n_vis = uv_wavelengths.shape[0]
    n_src = mapping_matrix.shape[1]

    # Preallocate output (this is small enough to be safe)
    transformed = xp.zeros((n_vis, n_src), dtype=xp.complex128)

    y = grid_radians[:, 1]  # (n_image_pixels,)
    x = grid_radians[:, 0]

    for i0 in range(0, n_vis, chunk_size):
        i1 = min(i0 + chunk_size, n_vis)

        uv_chunk = uv_wavelengths[i0:i1]  # (chunk, 2)

        # phase: (n_image_pixels, chunk)
        phase = (
            -2.0 * xp.pi * (xp.outer(y, uv_chunk[:, 0]) + xp.outer(x, uv_chunk[:, 1]))
        )

        # Compute Fourier response for this chunk
        fourier = xp.cos(phase) + 1j * xp.sin(phase)  # (n_img, chunk)

        # Accumulate: (chunk, n_src)
        vis_chunk = fourier.T @ mapping_matrix

        # Write back
        if xp.__name__.startswith("jax"):
            transformed = transformed.at[i0:i1, :].set(vis_chunk)
        else:
            transformed[i0:i1, :] = vis_chunk

    return transformed
