import numpy as np
import warnings
from scipy.ndimage import binary_dilation
from typing import Tuple

from autoarray import exc


def native_index_for_slim_index_2d_from(mask_2d: np.ndarray, xp=np) -> np.ndarray:
    """
    Returns an array of shape [total_unmasked_pixels] that maps every unmasked pixel to its
    corresponding native 2D pixel using its (y,x) pixel indexes.

    For example, for the following ``Mask2D``:

    ::
        [[True,  True,  True, True]
         [True, False, False, True],
         [True, False,  True, True],
         [True,  True,  True, True]]

    This has three unmasked (``False`` values) which have the ``slim`` indexes:

    ::
        [0, 1, 2]

    The array ``native_index_for_slim_index_2d`` is therefore:

    ::
        [[1,1], [1,2], [2,1]]

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked.

    Returns
    -------
    ndarray
        An array that maps pixels from a slimmed array of shape [total_unmasked_pixels] to its native array
        of shape [total_pixels, total_pixels].

    Examples
    --------
    mask_2d = np.array([[True, True, True],
                     [True, False, True]
                     [True, True, True]])

    native_index_for_slim_index_2d = native_index_for_slim_index_2d_from(mask_2d=mask_2d)
    """
    rows, cols = xp.where(~mask_2d.astype(bool))
    return xp.stack([rows, cols], axis=1)


def mask_2d_centres_from(
    shape_native: Tuple[int, int],
    pixel_scales: Tuple[float, float],
    centre: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Compute the (y, x) scaled central coordinates of a mask given its shape, pixel-scales, and centre.

    The coordinate system is defined such that the positive y-axis is up and the positive x-axis is right.

    Parameters
    ----------
    shape_native
        The shape of the 2D array in pixels.
    pixel_scales
        The conversion factors from pixels to scaled units.
    centre
        The central coordinate of the mask in scaled units.

    Returns
    -------
    The (y, x) scaled central coordinates of the input array.

    Examples
    --------
    >>> centres_scaled = mask_2d_centres_from(shape_native=(5, 5), pixel_scales=(0.5, 0.5), centre=(0.0, 0.0))
    >>> print(centres_scaled)
    (0.0, 0.0)
    """

    # Calculate scaled y-coordinate by centering and adjusting for pixel scale
    y_scaled = 0.5 * (shape_native[0] - 1) - (centre[0] / pixel_scales[0])

    # Calculate scaled x-coordinate by centering and adjusting for pixel scale
    x_scaled = 0.5 * (shape_native[1] - 1) + (centre[1] / pixel_scales[1])

    # Return the scaled (y, x) coordinates
    return (y_scaled, x_scaled)


def mask_2d_circular_from(
    shape_native: Tuple[int, int],
    pixel_scales: Tuple[float, float],
    radius: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Create a circular mask within a 2D array.

    This generates a 2D array where all values within the specified radius are unmasked (set to `False`).

    Parameters
    ----------
    shape_native
        The shape of the mask array in pixels.
    pixel_scales
        The conversion factors from pixels to scaled units.
    radius
        The radius of the circular mask in scaled units.
    centre
        The central coordinate of the circle in scaled units.

    Returns
    -------
    The 2D mask array with the central region defined by the radius unmasked (False).

    Examples
    --------
    >>> mask = mask_2d_circular_from(shape_native=(10, 10), pixel_scales=(0.1, 0.1), radius=0.5, centre=(0.0, 0.0))
    """

    # Get scaled coordinates of the mask center
    centres_scaled = mask_2d_centres_from(shape_native, pixel_scales, centre)

    # Create a grid of y, x indices for the mask
    y, x = np.ogrid[: shape_native[0], : shape_native[1]]

    # Scale the y and x indices based on pixel scales
    y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
    x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

    # Compute squared distances from the center for each pixel
    distances_squared = x_scaled**2 + y_scaled**2

    # Return a mask with True for pixels outside the circle and False for inside
    return distances_squared >= radius**2


def mask_2d_circular_annular_from(
    shape_native: Tuple[int, int],
    pixel_scales: Tuple[float, float],
    inner_radius: float,
    outer_radius: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Create a circular annular mask within a 2D array.

    This generates a 2D array where all values within the specified inner and outer radii are unmasked (set to `False`).

    Parameters
    ----------
    shape_native
        The shape of the mask array in pixels.
    pixel_scales
        The conversion factors from pixels to scaled units.
    inner_radius
        The inner radius of the annular mask in scaled units.
    outer_radius
        The outer radius of the annular mask in scaled units.
    centre
        The central coordinate of the annulus in scaled units.

    Returns
    -------
    The 2D mask array with the region between the inner and outer radii unmasked (False).

    Examples
    --------
    >>> mask = mask_2d_circular_annular_from(
    >>>     shape_native=(10, 10), pixel_scales=(0.1, 0.1), inner_radius=0.5, outer_radius=1.5, centre=(0.0, 0.0)
    >>> )
    """

    # Get scaled coordinates of the mask center
    centres_scaled = mask_2d_centres_from(shape_native, pixel_scales, centre)

    # Create grid of y, x indices for the mask
    y, x = np.ogrid[: shape_native[0], : shape_native[1]]

    # Scale the y and x indices based on pixel scales
    y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
    x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

    # Compute squared distances from the center for each pixel
    distances_squared = x_scaled**2 + y_scaled**2

    # Return the mask where pixels are unmasked between inner and outer radii
    return ~(
        (distances_squared >= inner_radius**2) & (distances_squared <= outer_radius**2)
    )


def mask_2d_elliptical_from(
    shape_native: Tuple[int, int],
    pixel_scales: Tuple[float, float],
    major_axis_radius: float,
    axis_ratio: float,
    angle: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Create an elliptical mask within a 2D array.

    This generates a 2D array where all values within the specified ellipse are unmasked (set to `False`).

    Parameters
    ----------
    shape_native
        The shape of the mask array in pixels.
    pixel_scales
        The conversion factors from pixels to scaled units.
    major_axis_radius
        The major axis radius of the elliptical mask in scaled units.
    axis_ratio
        The axis ratio of the ellipse (minor axis / major axis).
    angle
        The rotation angle of the ellipse in degrees, counter-clockwise from the positive x-axis.
    centre
        The central coordinate of the ellipse in scaled units.

    Returns
    -------
    np.ndarray
        The 2D mask array with the elliptical region defined by the major axis radius unmasked (False).

    Examples
    --------
    >>> mask = mask_2d_elliptical_from(
    >>>     shape_native=(10, 10), pixel_scales=(0.1, 0.1), major_axis_radius=0.5, axis_ratio=0.5, angle=45.0, centre=(0.0, 0.0)
    >>> )
    """

    # Get scaled coordinates of the mask center
    centres_scaled = mask_2d_centres_from(shape_native, pixel_scales, centre)

    # Create grid of y, x indices for the mask
    y, x = np.ogrid[: shape_native[0], : shape_native[1]]

    # Scale the y and x indices based on pixel scales
    y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
    x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

    # Compute the rotated coordinates and elliptical radius
    r_scaled = np.sqrt(x_scaled**2 + y_scaled**2)
    theta_rotated = np.arctan2(y_scaled, x_scaled) + np.radians(angle)
    y_scaled_elliptical = r_scaled * np.sin(theta_rotated)
    x_scaled_elliptical = r_scaled * np.cos(theta_rotated)
    r_scaled_elliptical = np.sqrt(
        x_scaled_elliptical**2 + (y_scaled_elliptical / axis_ratio) ** 2
    )

    # Return the mask where pixels are outside the elliptical region
    return ~(r_scaled_elliptical <= major_axis_radius)


def mask_2d_elliptical_annular_from(
    shape_native: Tuple[int, int],
    pixel_scales: Tuple[float, float],
    inner_major_axis_radius: float,
    inner_axis_ratio: float,
    inner_phi: float,
    outer_major_axis_radius: float,
    outer_axis_ratio: float,
    outer_phi: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns an elliptical annular mask from an input major-axis mask radius, axis-ratio, rotational angle for
    both the inner and outer elliptical annuli and a shape and centre for the mask.

    This creates a 2D array where all values within the elliptical annuli are unmasked and therefore `False`.

    Parameters
    ----------
    shape_native: Tuple[int, int]
        The (y,x) shape of the mask in units of pixels.
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel.
    inner_major_axis_radius
        The major-axis (in scaled units) of the inner ellipse within which pixels are masked.
    inner_axis_ratio
            The axis-ratio of the inner ellipse within which pixels are masked.
    inner_phi
        The rotation angle of the inner ellipse within which pixels are masked, (counter-clockwise from the
        positive x-axis).
    outer_major_axis_radius
        The major-axis (in scaled units) of the outer ellipse within which pixels are unmasked.
    outer_axis_ratio
            The axis-ratio of the outer ellipse within which pixels are unmasked.
    outer_phi
        The rotation angle of the outer ellipse within which pixels are unmasked, (counter-clockwise from the
        positive x-axis).
    centre
        The centre of the elliptical annuli used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose elliptical annuli pixels are masked.

    Examples
    --------
    >>> mask = mask_elliptical_annuli_from(
    >>>     shape=(10, 10), pixel_scales=(0.1, 0.1),
    >>>     inner_major_axis_radius=0.5, inner_axis_ratio=0.5, inner_phi=45.0,
    >>>     outer_major_axis_radius=1.5, outer_axis_ratio=0.8, outer_phi=90.0,
    >>>     centre=(0.0, 0.0)
    >>> )
    """

    # Get scaled coordinates of the mask center
    centres_scaled = mask_2d_centres_from(shape_native, pixel_scales, centre)

    # Create grid of y, x indices for the mask
    y, x = np.ogrid[: shape_native[0], : shape_native[1]]

    # Scale the y and x indices based on pixel scales
    y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
    x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

    # Compute and rotate coordinates for inner annulus
    r_scaled_inner = np.sqrt(x_scaled**2 + y_scaled**2)
    theta_rotated_inner = np.arctan2(y_scaled, x_scaled) + np.radians(inner_phi)
    y_scaled_elliptical_inner = r_scaled_inner * np.sin(theta_rotated_inner)
    x_scaled_elliptical_inner = r_scaled_inner * np.cos(theta_rotated_inner)
    r_scaled_elliptical_inner = np.sqrt(
        x_scaled_elliptical_inner**2
        + (y_scaled_elliptical_inner / inner_axis_ratio) ** 2
    )

    # Compute and rotate coordinates for outer annulus
    r_scaled_outer = np.sqrt(x_scaled**2 + y_scaled**2)
    theta_rotated_outer = np.arctan2(y_scaled, x_scaled) + np.radians(outer_phi)
    y_scaled_elliptical_outer = r_scaled_outer * np.sin(theta_rotated_outer)
    x_scaled_elliptical_outer = r_scaled_outer * np.cos(theta_rotated_outer)
    r_scaled_elliptical_outer = np.sqrt(
        x_scaled_elliptical_outer**2
        + (y_scaled_elliptical_outer / outer_axis_ratio) ** 2
    )

    # Return the mask where pixels are outside the inner and outer elliptical annuli
    return ~(
        (r_scaled_elliptical_inner >= inner_major_axis_radius)
        & (r_scaled_elliptical_outer <= outer_major_axis_radius)
    )


def mask_2d_via_pixel_coordinates_from(
    shape_native: Tuple[int, int], pixel_coordinates: list, buffer: int = 0
) -> np.ndarray:
    """
    Returns a mask where all unmasked `False` entries are defined from an input list of 2D pixel coordinates.

    These may be buffed via an input `buffer`, whereby all entries in all 8 neighboring directions are buffed by this
    amount.

    Parameters
    ----------
    shape_native
        The (y, x) shape of the mask in units of pixels.
    pixel_coordinates
        The input list of 2D pixel coordinates where `False` entries are created.
    buffer
        All input `pixel_coordinates` are buffed with `False` entries in all 8 neighboring directions by this
        amount.

    Returns
    -------
    np.ndarray
        The 2D mask array where all entries in the input pixel coordinates are set to `False`, with optional buffering
        applied to the neighboring entries.

    Examples
    --------
    mask = mask_2d_via_pixel_coordinates_from(
        shape_native=(10, 10),
        pixel_coordinates=[[1, 2], [3, 4], [5, 6]],
        buffer=1
    )
    """
    mask_2d = np.full(
        shape=shape_native, fill_value=True
    )  # Initialize mask with all True values

    for (
        y,
        x,
    ) in (
        pixel_coordinates
    ):  # Loop over input coordinates to set corresponding mask entries to False
        mask_2d[y, x] = False

    if buffer == 0:  # If no buffer is specified, return the mask directly
        return mask_2d
    return buffed_mask_2d_from(mask_2d=mask_2d, buffer=buffer)  # Apply buf


def min_false_distance_to_edge(mask: np.ndarray) -> Tuple[int, int]:
    """
    Compute the minimum 1D distance in the y and x directions from any `False` value at the mask's extreme positions
    (leftmost, rightmost, topmost, bottommost) to its closest edge.

    Parameters
    ----------
    mask
        A 2D boolean array where `False` represents the unmasked region.

    Returns
    -------
    Tuple[int, int]
        The smallest distances of any extreme `False` value to the nearest edge in the vertical (y) and horizontal (x)
        directions.

    Examples
    --------
    >>> mask = np.array([
    ...     [ True,  True,  True,  True],
    ...     [ True, False, False,  True],
    ...     [ True, False,  True,  True],
    ...     [ True,  True,  True,  True]
    ... ])
    >>> min_false_distance_to_edge(mask)
    (1, 1)
    """
    false_indices = np.column_stack(
        np.where(mask == False)
    )  # Find all coordinates where mask is False

    if false_indices.size == 0:
        raise ValueError(
            "No False values found in the mask."
        )  # Raise error if no False values

    leftmost = false_indices[
        np.argmin(false_indices[:, 1])
    ]  # Find the leftmost False coordinate
    rightmost = false_indices[
        np.argmax(false_indices[:, 1])
    ]  # Find the rightmost False coordinate
    topmost = false_indices[
        np.argmin(false_indices[:, 0])
    ]  # Find the topmost False coordinate
    bottommost = false_indices[
        np.argmax(false_indices[:, 0])
    ]  # Find the bottommost False coordinate

    height, width = mask.shape  # Get the height and width of the mask

    # Compute distances to respective edges
    left_dist = leftmost[1]  # Distance to left edge (column index)
    right_dist = width - 1 - rightmost[1]  # Distance to right edge
    top_dist = topmost[0]  # Distance to top edge (row index)
    bottom_dist = height - 1 - bottommost[0]  # Distance to bottom edge

    # Return the minimum distance to both edges
    return min(top_dist, bottom_dist), min(left_dist, right_dist)


import warnings
from typing import Tuple

import numpy as np
from scipy.ndimage import binary_dilation


def required_shape_for_kernel(
    mask_2d: np.ndarray,
    kernel_shape_native: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Return the minimal shape the mask must be padded to so that a kernel with the given
    footprint can be applied without sampling beyond the array edge, while preserving
    parity (odd->odd, even->even) in each dimension.

    Parameters
    ----------
    mask_2d
        2D boolean array where False is unmasked and True is masked.
    kernel_shape_native
        (ky, kx) footprint of the convolution kernel.

    Returns
    -------
    required_shape
        The minimal (ny, nx) shape such that the minimum distance from any unmasked
        pixel to the array edge is at least (ky//2, kx//2), and each dimension keeps
        the same parity as the input mask.
    """
    mask_2d = np.asarray(mask_2d, dtype=bool)

    ky, kx = kernel_shape_native
    if ky <= 0 or kx <= 0:
        raise ValueError(
            f"kernel_shape_native must be positive, got {kernel_shape_native}."
        )

    pad_y, pad_x = ky // 2, kx // 2
    y_distance, x_distance = min_false_distance_to_edge(mask_2d)

    extra_y = max(0, pad_y - y_distance)
    extra_x = max(0, pad_x - x_distance)

    new_y = mask_2d.shape[0] + 2 * extra_y
    new_x = mask_2d.shape[1] + 2 * extra_x

    # Preserve parity per axis: odd->odd, even->even
    if (new_y % 2) != (mask_2d.shape[0] % 2):
        new_y += 1
    if (new_x % 2) != (mask_2d.shape[1] % 2):
        new_x += 1

    return new_y, new_x


def blurring_mask_2d_from(
    mask_2d: np.ndarray,
    kernel_shape_native: Tuple[int, int],
    allow_padding: bool = False,
) -> np.ndarray:
    """
    Return the blurring mask for a 2D mask and kernel footprint.

    Convention (as used by PyAutoLens / PyAutoArray masks):
    - False = unmasked (included)
    - True  = masked   (excluded)

    The returned blurring mask is a *mask* where the blurring-region pixels are
    unmasked (False) and all other pixels are masked (True).

    If the input mask is too small for the kernel footprint:
    - allow_padding=False (default): raises an exception.
    - allow_padding=True: pads the mask symmetrically with masked pixels (True) to the
      minimal required shape (with parity preserved) and emits a warning.

    Parameters
    ----------
    mask_2d
        2D boolean mask.
    kernel_shape_native
        (ky, kx) kernel footprint.
    allow_padding
        If False, raise if padding is required. If True, pad and warn.

    Returns
    -------
    blurring_mask
        Boolean mask of the same shape as the (possibly padded) input.
    """
    mask_2d = np.asarray(mask_2d, dtype=bool)

    required_shape = required_shape_for_kernel(mask_2d, kernel_shape_native)

    if required_shape != mask_2d.shape:
        if not allow_padding:
            raise exc.MaskException(
                "The input mask is too small for the kernel shape. "
                f"Current shape: {mask_2d.shape}, required shape: {required_shape}. "
                "Set allow_padding=True to pad automatically."
            )

        warnings.warn(
            f"Mask padded from {mask_2d.shape} to {required_shape} "
            f"(parity preserved) to support kernel footprint {kernel_shape_native}.",
            UserWarning,
        )

        dy = required_shape[0] - mask_2d.shape[0]
        dx = required_shape[1] - mask_2d.shape[1]

        pad_top = dy // 2
        pad_bottom = dy - pad_top
        pad_left = dx // 2
        pad_right = dx - pad_left

        mask_2d = np.pad(
            mask_2d,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=True,  # outside is masked
        )

    # (Optional) hard invariant: parity preserved after any padding
    if (mask_2d.shape[0] % 2) != (required_shape[0] % 2) or (mask_2d.shape[1] % 2) != (
        required_shape[1] % 2
    ):
        raise RuntimeError(
            f"Parity invariant violated: got {mask_2d.shape}, expected parity of {required_shape}."
        )

    ky, kx = kernel_shape_native
    pad_y, pad_x = ky // 2, kx // 2
    structure = np.ones((ky, kx), dtype=bool)

    # Unmasked region (True where unmasked)
    unmasked = ~mask_2d

    # Explicit padding so outside behaves as masked => outside is NOT unmasked
    unmasked_padded = np.pad(
        unmasked,
        pad_width=((pad_y, pad_y), (pad_x, pad_x)),
        mode="constant",
        constant_values=False,
    )

    # Pixels within kernel footprint of any unmasked pixel
    near_unmasked_padded = binary_dilation(unmasked_padded, structure=structure)
    near_unmasked = near_unmasked_padded[
        pad_y : pad_y + mask_2d.shape[0],
        pad_x : pad_x + mask_2d.shape[1],
    ]

    # Blurring region: masked pixels near unmasked pixels
    blurring_region = mask_2d & near_unmasked

    # Return as a mask: blurring region is unmasked (False), everything else masked (True)
    blurring_mask = np.ones_like(mask_2d, dtype=bool)
    blurring_mask[blurring_region] = False

    return blurring_mask


def mask_slim_indexes_from(
    mask_2d: np.ndarray, return_masked_indexes: bool = True
) -> np.ndarray:
    """
    Returns a 1D array listing all masked (`value=True`) or unmasked pixel indexes (`value=False`) in the mask.

    For example, for the following ``Mask2D``:

    ::
        [[True,  True,  True, True],
         [True, False, False, True],
         [True, False,  True, True],
         [True,  True,  True, True]]

    This has three unmasked (``False`` values) which have the ``slim`` indexes, their ``unmasked_slim`` is:

    ::
        [0, 1, 2]

    Parameters
    ----------
    mask_2d
        A 2D array representing the mask, where `True` indicates a masked pixel and `False` indicates an unmasked pixel.
    return_masked_indexes
        A boolean flag that determines whether to return indexes of masked (`True`) or unmasked (`False`) pixels.

    Returns
    -------
    A 1D array of indexes corresponding to either the masked or unmasked pixels in the mask.

    Examples
    --------
    >>> mask = np.array([[True, True, True, True],
    ...                  [True, False, False, True],
    ...                  [True, False, True, True],
    ...                  [True, True, True, True]])
    >>> mask_slim_indexes_from(mask, return_masked_indexes=True)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> mask_slim_indexes_from(mask, return_masked_indexes=False)
    array([10, 11])
    """
    # Flatten the mask and use np.where to get indexes of either True or False
    mask_flat = mask_2d.flatten()

    # Get the indexes where the mask is equal to return_masked_indexes (True or False)
    return np.where(mask_flat == return_masked_indexes)[0]


def edge_1d_indexes_from(mask_2d: np.ndarray) -> np.ndarray:
    """
    Returns a 1D array listing all edge pixel indexes in the mask.

    An edge pixel is defined as a pixel on the mask which is unmasked (has a `False`) value and at least one of its 8
    direct neighbors is masked (is `True`).

    For example, for the following ``Mask2D``:

    ::
        [[True,  True,  True,  True, True],
         [True, False, False, False, True],
         [True, False, False, False, True],
         [True, False, False, False, True],
         [True,  True,  True,  True, True]]

    The `edge_slim` indexes (given via ``mask_2d.derive_indexes.edge_slim``) is given by:

    ::
         [0, 1, 2, 3, 5, 6, 7, 8]

    Note that index 4 is skipped, which corresponds to the ``False`` value in the centre of the mask, because it
    does not neighbor a ``True`` value in any one of the eight neighboring directions and is therefore not at
    an edge.

    Parameters
    ----------
    mask_2d
        A 2D boolean array where `False` values indicate unmasked pixels.

    Returns
    -------
    A 1D array of indexes of all edge pixels on the mask.

    Examples
    --------
    >>> mask = np.array([
    ...     [True, True, True, True, True],
    ...     [True, False, False, False, True],
    ...     [True, False, False, False, True],
    ...     [True, False, False, False, True],
    ...     [True, True, True, True, True]
    ... ])
    >>> edge_1d_indexes_from(mask)
    array([0, 1, 2, 3, 5, 6, 7, 8])
    """
    # Pad the mask to handle edge cases without index errors
    padded_mask = np.pad(mask_2d, pad_width=1, mode="constant", constant_values=True)

    # Identify neighbors in 3x3 regions around each pixel
    neighbors = (
        padded_mask[:-2, 1:-1]
        | padded_mask[2:, 1:-1]  # Up, Down
        | padded_mask[1:-1, :-2]
        | padded_mask[1:-1, 2:]  # Left, Right
        | padded_mask[:-2, :-2]
        | padded_mask[:-2, 2:]  # Top-left, Top-right
        | padded_mask[2:, :-2]
        | padded_mask[2:, 2:]  # Bottom-left, Bottom-right
    )

    # Identify edge pixels: False values with at least one True neighbor
    edge_mask = ~mask_2d & neighbors

    # Create an index array where False entries get sequential 1D indices
    index_array = np.full(mask_2d.shape, fill_value=-1, dtype=int)
    false_indices = np.flatnonzero(~mask_2d)
    index_array[~mask_2d] = np.arange(len(false_indices))

    # Return the 1D indexes of the edge pixels
    return index_array[edge_mask]


def border_slim_indexes_from(mask_2d: np.ndarray) -> np.ndarray:
    """
    Returns a 1D array listing all border pixel indexes in the mask.

    A border pixel is an unmasked pixel (`False` value) that can reach the edge of the mask without encountering
    a masked (`True`) pixel in any of the four cardinal directions (up, down, left, right).

    The borders pixels are thus pixels which are on the exterior edge of the mask. For example, the inner ring of edge
    pixels in an annular mask are edge pixels but not borders pixels.

    For example, for the following ``Mask2D``:

    ::
        [[True,  True,  True,  True,  True,  True,  True,  True, True],
         [True, False, False, False, False, False, False, False, True],
         [True, False,  True,  True,  True,  True,  True, False, True],
         [True, False,  True, False, False, False,  True, False, True],
         [True, False,  True, False,  True, False,  True, False, True],
         [True, False,  True, False, False, False,  True, False, True],
         [True, False,  True,  True,  True,  True,  True, False, True],
         [True, False, False, False, False, False, False, False, True],
         [True,  True,  True,  True,  True,  True,  True,  True, True]]

    The `border_slim` indexes (given via ``mask_2d.derive_indexes.border_slim``) is given by:

    ::
         [0, 1, 2, 3, 5, 6, 7, 11, 12, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    The interior 8 ``False`` values are omitted, because although they are edge pixels (neighbor a ``True``) they
    are not on the extreme exterior edge.

    Parameters
    ----------
    mask_2d
        A 2D boolean array where `False` values indicate unmasked pixels.

    Returns
    -------
    A 1D array of indexes of all border pixels on the mask.

    Examples
    --------
    >>> mask = np.array([
    ...    [True,  True,  True,  True,  True,  True,  True,  True, True],
    ...    [True, False, False, False, False, False, False, False, True],
    ...    [True, False,  True,  True,  True,  True,  True, False, True],
    ...    [True, False,  True, False, False, False,  True, False, True],
    ...    [True, False,  True, False,  True, False,  True, False, True],
    ...    [True, False,  True, False, False, False,  True, False, True],
    ...    [True, False,  True,  True,  True,  True,  True, False, True],
    ...    [True, False, False, False, False, False, False, False, True],
    ...    [True,  True,  True,  True,  True,  True,  True,  True, True]
    ... ])
    >>> border_slim_indexes_from(mask)
    array([0, 1, 2, 3, 5, 6, 7, 11, 12, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    """

    # Compute cumulative sums along each direction
    up_sums = np.cumsum(mask_2d, axis=0)
    down_sums = np.cumsum(mask_2d[::-1, :], axis=0)[::-1, :]
    left_sums = np.cumsum(mask_2d, axis=1)
    right_sums = np.cumsum(mask_2d[:, ::-1], axis=1)[:, ::-1]

    # Get mask dimensions
    height, width = mask_2d.shape

    # Identify border pixels: where the full length in any direction is True
    border_mask = (
        (up_sums == np.arange(height)[:, None])
        | (down_sums == np.arange(height - 1, -1, -1)[:, None])
        | (left_sums == np.arange(width)[None, :])
        | (right_sums == np.arange(width - 1, -1, -1)[None, :])
    ) & ~mask_2d

    # Create an index array where False entries get sequential 1D indices
    index_array = np.full(mask_2d.shape, fill_value=-1, dtype=int)
    false_indices = np.flatnonzero(~mask_2d)
    index_array[~mask_2d] = np.arange(len(false_indices))

    # Return the 1D indexes of the border pixels
    return index_array[border_mask]


def buffed_mask_2d_from(mask_2d: np.ndarray, buffer: int = 1) -> np.ndarray:
    """
    Returns a buffed mask from an input mask, where all `False` entries in the mask are "buffed" (set to `False`)
    within a specified buffer range in all 8 surrounding directions.

    A "buffed" mask is created by marking all the pixels within a square of size `buffer` around each `False`
    entry as `False`. This process simulates expanding the masked region around each `False` entry by the specified
    buffer distance.

    Parameters
    ----------
    mask_2d
        A 2D boolean array where `False` values indicate unmasked pixels.
    buffer
        The number of pixels around each `False` entry that should be buffed in all 8 surrounding directions.
        This controls how far the "buffed" region extends from each `False` value.

    Returns
    -------
    A new 2D boolean array where all `False` entries in the input mask are expanded by the specified buffer
    distance, setting all pixels within the buffer range to `False`.

    Examples
    --------
    >>> mask = np.array([
    ...     [True, False, True],
    ...     [False, False, False],
    ...     [True, True, False]
    ... ])
    >>> buffed_mask_2d_from(mask, buffer=1)
    array([[False, False, False],
           [False, False, False],
           [False, False, False]])
    """
    # Initialize buffed mask as a copy of the input mask
    buffed_mask_2d = mask_2d.copy()

    # Identify the coordinates of all False entries
    false_coords = np.nonzero(~mask_2d)

    # Create grid of offsets for the neighboring pixels (buffer range)
    buffer_range = np.arange(-buffer, buffer + 1)

    # Generate all possible neighbors for each False entry
    dy, dx = np.meshgrid(buffer_range, buffer_range, indexing="ij")
    neighbors = np.stack([dy.ravel(), dx.ravel()], axis=-1)

    # Calculate all neighboring positions for all False coordinates
    all_neighbors = np.add(np.array(false_coords).T[:, np.newaxis], neighbors)

    # Clip the neighbors to stay within the bounds of the mask
    valid_neighbors = np.clip(
        all_neighbors, [0, 0], [mask_2d.shape[0] - 1, mask_2d.shape[1] - 1]
    )

    # Update the buffed mask: set all the neighbors to False
    buffed_mask_2d[valid_neighbors[:, :, 0], valid_neighbors[:, :, 1]] = False

    return buffed_mask_2d


def rescaled_mask_2d_from(mask_2d: np.ndarray, rescale_factor: float) -> np.ndarray:
    """
    Returns a rescaled mask from an input mask, where the rescaled mask is the input mask but rescaled to a larger or
    smaller size depending on the `rescale_factor`.

    For example, a `rescale_factor` of 0.5 would reduce a 10 x 10 mask to a 5x5 mask, where the `False` entries
    of the 5 x 5 mask corresponding to pixels which had at least one `False` entry in their corresponding location on the
    10 x 10 mask. A rescale factor of 2.0 would increase the 10 x 10 mask in size to a 20 x 20 mask, with `False`
    again wherever the original mask had those entries.

    The edge of the rescaled mask is automatically set to all ` True` values to prevent border issues.

    Parameters
    ----------
    mask_2d
        The mask that is increased or decreased in size via rescaling.
    rescale_factor
        The factor by which the mask is increased in size or decreased in size.

    Returns
    -------
    np.ndarray
        The rescaled mask.
    """

    from skimage.transform import rescale

    warnings.filterwarnings("ignore")

    try:
        rescaled_mask_2d = rescale(
            image=mask_2d,
            scale=rescale_factor,
            mode="edge",
            anti_aliasing=False,
            multichannel=False,
        )
    except TypeError:
        rescaled_mask_2d = rescale(
            image=mask_2d,
            scale=rescale_factor,
            mode="edge",
            anti_aliasing=False,
            #  multichannel=False,
        )

    rescaled_mask_2d[0, :] = 1
    rescaled_mask_2d[rescaled_mask_2d.shape[0] - 1, :] = 1
    rescaled_mask_2d[:, 0] = 1
    rescaled_mask_2d[:, rescaled_mask_2d.shape[1] - 1] = 1
    return np.isclose(rescaled_mask_2d, 1)
