import numpy as np
from typing import Tuple, Union

from autoarray import type as ty
from autoarray import validate


def convert_shape_native_1d(shape_native: Union[int, Tuple[int]]) -> Tuple[int]:
    """
    Convert an input `shape_native` given as a single integer scalar to a tuple `(int,)`. If
    the input is already a `(int,)` tuple it is returned unchanged.

    This enables users to input `shape_native` as a single integer value and have the type
    automatically normalised to `(int,)` which is used internally by 1D data structures.

    Any concrete integer scalar is widened — `int` and `np.integer` — not just an exact `int`.
    An `np.integer` is what indexing a shape tuple or reading a FITS header returns, so it
    reaches this function on paths a user would consider ordinary. The widened value is cast
    to a Python `int`, so the tuple this returns is always `(int,)` regardless of what went in.

    A `float` is deliberately *not* widened: `shape_native` counts pixels, so a non-integer is
    a mistake worth surfacing rather than one to normalise away. Neither is a `bool`, nor a JAX
    tracer — see :func:`autoarray.validate.is_concrete_integer`.

    Parameters
    ----------
    shape_native
        The 1D shape to convert, either as a plain integer scalar or a 1-element tuple `(int,)`.

    Returns
    -------
    Tuple[int]
        The shape as a 1-element tuple `(int,)`.
    """

    if validate.is_concrete_integer(shape_native):
        shape_native = (int(shape_native),)

    return shape_native


def convert_pixel_scales_1d(pixel_scales: ty.PixelScales) -> Tuple[float]:
    """
    Convert an input pixel scale given as a single real scalar to a tuple `(float,)`. If the
    input is already a `(float,)` tuple it is returned unchanged.

    This enables users to input the pixel scale as a single number and have the type
    automatically normalised to `(float,)` which is used internally by 1D data structures.

    Any concrete real scalar is widened — `int`, `float`, `np.integer` and `np.floating` — not
    just an exact `float`. An `int` is a natural thing to type by hand, and an `np.floating` is
    what indexing a numpy array or reading a FITS header returns, so both reach this function on
    paths a user would consider ordinary. The widened value is cast to a Python `float`, so the
    tuple this returns is always `(float,)` regardless of what went in.

    A `bool` is deliberately *not* treated as a scalar here (see
    :func:`autoarray.validate.is_concrete_scalar`), and neither is a JAX tracer — a traced value
    passes through untouched so the function stays safe inside a `jax.jit`.

    Parameters
    ----------
    pixel_scales
        The pixel scale to convert, either as a plain real scalar or a 1-element tuple
        `(float,)`.

    Returns
    -------
    Tuple[float]
        The pixel scale as a 1-element tuple `(float,)`.

    Raises
    ------
    ValueError
        If any entry is a concrete scalar which is not finite and above zero.
    """

    validate.validate_pixel_scales(pixel_scales=pixel_scales)

    if validate.is_concrete_scalar(pixel_scales):
        pixel_scales = (float(pixel_scales),)

    return pixel_scales


def central_pixel_coordinates_1d_from(
    shape_slim: Tuple[int],
) -> Tuple[float]:
    """
    Return the central pixel coordinate of a 1D data structure from its shape.

    For an odd-length array the centre is the middle element (e.g. length 3 → 1.0). For an
    even-length array the centre falls between two pixels (e.g. length 4 → 1.5).

    Parameters
    ----------
    shape_slim
        The 1D shape of the data structure as a 1-element tuple `(int,)`.

    Returns
    -------
    Tuple[float]
        The central pixel coordinate as a 1-element tuple `(float,)`.
    """

    return (float(shape_slim[0] - 1) / 2,)


def central_scaled_coordinate_1d_from(
    shape_slim: Tuple[float],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float] = (0.0, 0.0),
):
    """
    Return the central coordinate of a 1D data structure (e.g. a `Grid1D`) in scaled units.

    This is computed from the structure's shape and pixel scale, shifted by the origin.

    Parameters
    ----------
    shape_slim
        The 1D shape of the data structure whose central scaled coordinate is computed.
    pixel_scales
        The (x,) scaled units to pixel units conversion factor of the 1D data structure.
    origin
        The (x,) scaled units origin of the coordinate system. The central coordinate is
        shifted by this origin.

    Returns
    -------
    Tuple[float]
        The central coordinate of the 1D data structure in scaled units as a 1-element tuple.
    """

    central_pixel_coordinates = central_pixel_coordinates_1d_from(shape_slim=shape_slim)

    x_pixel = central_pixel_coordinates[0] - (origin[0] / pixel_scales[0])

    return (x_pixel,)


def pixel_coordinates_1d_from(
    scaled_coordinates_1d: Tuple[float],
    shape_slim: Tuple[int],
    pixel_scales: ty.PixelScales,
    origins: Tuple[float] = (0.0, 0.0),
) -> Tuple[int]:
    """
    Convert a 1D (x,) scaled coordinate to a 1D (x,) integer pixel coordinate.

    The pixel index is 0-based from the left of the 1D array. The scaled coordinate origin is
    applied before the conversion.

    Parameters
    ----------
    scaled_coordinates_1d
        The 1D (x,) coordinate in scaled units to convert.
    shape_slim
        The 1D shape of the array as a 1-element tuple `(int,)`.
    pixel_scales
        The (x,) scaled units to pixel units conversion factor.
    origins
        The (x,) scaled units origin of the coordinate system.

    Returns
    -------
    Tuple[int]
        The 1D (x,) integer pixel coordinate.
    """
    central_pixel_coordinates = central_pixel_coordinates_1d_from(shape_slim=shape_slim)

    x_pixel = int(
        (scaled_coordinates_1d[0] - origins[0]) / pixel_scales[0]
        + central_pixel_coordinates[0]
        + 0.5
    )

    return (x_pixel,)


def scaled_coordinates_1d_from(
    pixel_coordinates_1d: Tuple[float],
    shape_slim: Tuple[int],
    pixel_scales: ty.PixelScales,
    origins: Tuple[float] = (0.0, 0.0),
) -> Tuple[float]:
    """
    Convert a 1D (x,) pixel coordinate to a 1D (x,) scaled coordinate.

    The scaled coordinate origin is applied after the conversion.

    Parameters
    ----------
    pixel_coordinates_1d
        The 1D (x,) pixel coordinate to convert to scaled units.
    shape_slim
        The 1D shape of the array as a 1-element tuple `(int,)`.
    pixel_scales
        The (x,) scaled units to pixel units conversion factor.
    origins
        The (x,) scaled units origin of the coordinate system.

    Returns
    -------
    Tuple[float]
        The 1D (x,) coordinate in scaled units.
    """
    central_scaled_coordinates = central_scaled_coordinate_1d_from(
        shape_slim=shape_slim, pixel_scales=pixel_scales, origin=origins
    )

    x_pixel = pixel_scales[0] * (
        pixel_coordinates_1d[0] - central_scaled_coordinates[0]
    )

    return (x_pixel,)


def convert_pixel_scales_2d(pixel_scales: ty.PixelScales) -> Tuple[float, float]:
    """
    Convert an input pixel scale given as a single real scalar to a tuple `(float, float)`. If
    the input is already type `(float, float)` it is returned unchanged.

    This enables users to input the pixel scale as a single number and have the type
    automatically normalised to `(float, float)` which is used internally for rectangular 2D
    grids (where both axes share the same pixel scale).

    Any concrete real scalar is widened — `int`, `float`, `np.integer` and `np.floating` — not
    just an exact `float`. An `int` is a natural thing to type by hand, and an `np.floating` is
    what indexing a numpy array or reading a FITS header returns, so both reach this function on
    paths a user would consider ordinary. The widened value is cast to a Python `float`, so the
    tuple this returns is always `(float, float)` regardless of what went in.

    A `bool` is deliberately *not* treated as a scalar here (see
    :func:`autoarray.validate.is_concrete_scalar`), and neither is a JAX tracer — a traced value
    passes through untouched so the function stays safe inside a `jax.jit`.

    Parameters
    ----------
    pixel_scales
        The pixel scale to convert, either as a plain real scalar or a 2-element tuple
        `(float, float)`.

    Returns
    -------
    Tuple[float, float]
        The pixel scale as a 2-element tuple `(float, float)`.

    Raises
    ------
    ValueError
        If any entry is a concrete scalar which is not finite and above zero.

    Notes
    -----
    This is the single chokepoint every ``Mask2D`` factory and ``Grid2D.uniform``
    funnel their ``pixel_scales`` through, so validating here covers them all rather
    than repeating a guard at each construction site.
    """

    validate.validate_pixel_scales(pixel_scales=pixel_scales)

    if validate.is_concrete_scalar(pixel_scales):
        pixel_scales = (float(pixel_scales), float(pixel_scales))

    return pixel_scales


def central_pixel_coordinates_2d_from(
    shape_native: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Returns the central pixel coordinates of a 2D geometry (and therefore a 2D data structure like an ``Array2D``)
    from the shape of that data structure.

    Examples of the central pixels are as follows:

    - For a 3x3 image, the central pixel is pixel [1, 1].
    - For a 4x4 image, the central pixel is [1.5, 1.5].

    Parameters
    ----------
    shape_native
        The dimensions of the data structure, which can be in 1D, 2D or higher dimensions.

    Returns
    -------
    The central pixel coordinates of the data structure.
    """
    return (float(shape_native[0] - 1) / 2, float(shape_native[1] - 1) / 2)


def central_scaled_coordinate_2d_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """
    Returns the central scaled coordinates of a 2D geometry (and therefore a 2D data structure like an ``Array2D``)
    from the shape of that data structure.

    This is computed by using the data structure's shape and converting it to scaled units using an input
    pixel-coordinates to scaled-coordinate conversion factor `pixel_scales`.

    The origin of the scaled grid can also be input and moved from (0.0, 0.0).

    Parameters
    ----------
    shape_native
        The 2D shape of the data structure whose central scaled coordinates are computed.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D data structure.
    origin
        The (y,x) scaled units origin of the coordinate system the central scaled coordinate is computed on.

    Returns
    -------
    The central coordinates of the 2D data structure in scaled units.
    """

    central_pixel_coordinates = central_pixel_coordinates_2d_from(
        shape_native=shape_native
    )

    y_pixel = central_pixel_coordinates[0] + (origin[0] / pixel_scales[0])
    x_pixel = central_pixel_coordinates[1] - (origin[1] / pixel_scales[1])

    return (y_pixel, x_pixel)


def pixel_coordinates_2d_from(
    scaled_coordinates_2d: Tuple[float, float],
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origins: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """
    Convert a 2D (y,x) scaled coordinate to a 2D (y,x) pixel coordinate, which are returned as floats such that they
    include the decimal offset from each pixel's top-left corner relative to the input scaled coordinate.

    The conversion is performed according to a 2D geometry on a uniform grid, where the pixel coordinate origin is at
    the top left corner, such that the pixel [0,0] corresponds to the highest (most positive) y scaled coordinate
    and lowest (most negative) x scaled coordinate on the grid.

    The scaled coordinate is defined by an origin and coordinates are shifted to this origin before computing their
    1D grid pixel coordinate values.

    Parameters
    ----------
    scaled_coordinates_2d
        The 2D (y,x) coordinates in scaled units which are converted to pixel coordinates.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted to.

    Returns
    -------
    A 2D (y,x) pixel-value coordinate.

    Examples
    --------

    scaled_coordinates_2d = (1.0, 1.0)

    grid_pixels_2d_slim = pixel_coordinates_2d_from(
        scaled_coordinates_2d=scaled_coordinates_2d,
        shape=(2,2),
        pixel_scales=(0.5, 0.5),
        origin=(0.0, 0.0)
    )
    """

    central_pixel_coordinates = central_pixel_coordinates_2d_from(
        shape_native=shape_native
    )

    y_pixel = int(
        (-scaled_coordinates_2d[0] + origins[0]) / pixel_scales[0]
        + central_pixel_coordinates[0]
        + 0.5
    )
    x_pixel = int(
        (scaled_coordinates_2d[1] - origins[1]) / pixel_scales[1]
        + central_pixel_coordinates[1]
        + 0.5
    )

    return (y_pixel, x_pixel)


def scaled_coordinates_2d_from(
    pixel_coordinates_2d: Tuple[float, float],
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origins: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """
    Convert a 2D (y,x) pixel coordinate to a 2D (y,x) scaled coordinate.

    The conversion is performed on a uniform grid where the pixel coordinate origin is at the
    top-left corner, such that pixel [0,0] corresponds to the highest (most positive) y scaled
    coordinate and lowest (most negative) x scaled coordinate on the grid.

    The geometry origin is applied so that the returned scaled coordinate is correctly shifted.

    Parameters
    ----------
    pixel_coordinates_2d
        The 2D (y,x) pixel coordinates to convert to scaled units.
    shape_native
        The (y,x) shape of the original 2D array.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origins
        The (y,x) origin of the grid in scaled units.

    Returns
    -------
    Tuple[float, float]
        The 2D (y,x) coordinates in scaled units.
    """
    central_scaled_coordinates = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origins
    )

    y_pixel = pixel_scales[0] * -(
        pixel_coordinates_2d[0] - central_scaled_coordinates[0]
    )
    x_pixel = pixel_scales[1] * (
        pixel_coordinates_2d[1] - central_scaled_coordinates[1]
    )

    return (y_pixel, x_pixel)


def pixel_coordinates_wcs_2d_from(
    scaled_coordinates_2d: Tuple[float, float],
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origins: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float]:
    """
    Return FITS / WCS pixel coordinates (1-based, pixel-centre convention) as floats.

    This function returns continuous pixel coordinates suitable for Astropy WCS
    transforms (e.g. ``wcs_pix2world`` with ``origin=1``). Pixel centres lie at
    integer values; for an image of shape ``(ny, nx)`` the geometric centre is::

        ((ny + 1) / 2, (nx + 1) / 2)

    e.g. ``(100, 100) -> (50.5, 50.5)``.

    Parameters
    ----------
    scaled_coordinates_2d
        The 2D (y, x) coordinates in scaled units which are converted to WCS
        pixel coordinates.
    shape_native
        The (y, x) shape of the 2D array on which the scaled coordinates are
        defined, used to determine the geometric centre in WCS pixel units.
    pixel_scales
        The (y, x) conversion factors from scaled units to pixel units.
    origins
        The (y, x) origin in scaled units about which the coordinates are
        defined. The scaled coordinates are shifted by this origin before being
        converted to WCS pixel coordinates.

    Returns
    -------
    pixel_coordinates_wcs_2d
        A 2D (y, x) WCS pixel coordinate in the 1-based, pixel-centre
        convention, returned as floats.
    """
    ny, nx = shape_native

    # Geometric centre in WCS pixel coordinates (1-based, pixel centres at integers)
    ycen_wcs = (ny + 1) / 2.0
    xcen_wcs = (nx + 1) / 2.0

    # Continuous WCS pixel coordinates (NO int-cast, NO +0.5 binning)
    y_wcs = (-scaled_coordinates_2d[0] + origins[0]) / pixel_scales[0] + ycen_wcs
    x_wcs = (scaled_coordinates_2d[1] - origins[1]) / pixel_scales[1] + xcen_wcs

    return (y_wcs, x_wcs)


def transform_grid_2d_to_reference_frame(
    grid_2d: np.ndarray, centre: Tuple[float, float], angle: float, xp=np
) -> np.ndarray:
    """
    Transform a 2D grid of (y,x) coordinates to a new reference frame defined by a centre and angle.

    This is used to evaluate light profiles and other functions in their own reference frame. The
    transformation includes:

    1) Translate all coordinates so that `centre` becomes the origin (subtract `centre` from every
       coordinate on the grid).
    2) Rotate the translated coordinates clockwise by `angle` degrees around the new origin.

    Parameters
    ----------
    grid_2d
        The 2D grid of (y,x) coordinates of shape (N, 2) to transform.
    centre
        The (y,x) centre of the new reference frame. All coordinates are shifted to place this
        point at the origin before rotation.
    angle
        The clockwise rotation angle in degrees applied to the translated grid.
    xp
        The array module to use (default `numpy`; pass `jax.numpy` for JAX support).

    Returns
    -------
    np.ndarray
        The transformed 2D grid of (y,x) coordinates of shape (N, 2) in the new reference frame.
    """

    shifted_grid_2d = grid_2d - xp.array(centre)

    # A clockwise rotation by `angle` is the rotation matrix applied to the shifted (y, x)
    # coordinates, built from one scalar cos / sin of the angle. This is algebraically
    # identical to the polar form (r, theta - angle) -> (r sin, r cos) but costs two
    # multiply-adds per coordinate instead of a per-pixel sqrt, arctan2, sin and cos.
    cos_angle = xp.cos(xp.radians(angle))
    sin_angle = xp.sin(xp.radians(angle))

    y = shifted_grid_2d[:, 0]
    x = shifted_grid_2d[:, 1]

    return xp.vstack(
        [
            y * cos_angle - x * sin_angle,
            x * cos_angle + y * sin_angle,
        ]
    ).T


def transform_grid_2d_from_reference_frame(
    grid_2d: np.ndarray, centre: Tuple[float, float], angle: float, xp=np
) -> np.ndarray:
    """
    Transform a 2D grid of (y,x) coordinates back from a reference frame to the original frame.

    This is the inverse of `transform_grid_2d_to_reference_frame`. The transformation includes:

    1) Rotate the grid counter-clockwise by `angle` degrees around the origin.
    2) Translate all coordinates by adding `centre`, restoring the original coordinate system.

    Parameters
    ----------
    grid_2d
        The 2D grid of (y,x) coordinates of shape (N, 2) in the rotated/translated reference frame.
    centre
        The (y,x) centre that was used in the forward transform. Added back to all coordinates
        after the inverse rotation to restore the original coordinate system.
    angle
        The clockwise rotation angle in degrees that was used in the forward transform. The inverse
        (counter-clockwise) rotation is applied here.
    xp
        The array module to use (default `numpy`; pass `jax.numpy` for JAX support).

    Returns
    -------
    np.ndarray
        The back-transformed 2D grid of (y,x) coordinates of shape (N, 2) in the original frame.
    """
    cos_angle = xp.cos(xp.radians(angle))
    sin_angle = xp.sin(xp.radians(angle))

    y = xp.add(
        xp.add(
            xp.multiply(grid_2d[:, 1], sin_angle),
            xp.multiply(grid_2d[:, 0], cos_angle),
        ),
        centre[0],
    )
    x = xp.add(
        xp.add(
            xp.multiply(grid_2d[:, 1], cos_angle),
            -xp.multiply(grid_2d[:, 0], sin_angle),
        ),
        centre[1],
    )
    return xp.vstack((y, x)).T


def grid_pixels_2d_slim_from(
    grid_scaled_2d_slim: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a slimmed grid of 2d (y,x) scaled coordinates to a slimmed grid of 2d (y,x) pixel coordinate values. Pixel
    coordinates are returned as floats such that they include the decimal offset from each pixel's top-left corner
    relative to the input scaled coordinate.

    The input and output grids are both slimmed and therefore shape (total_pixels, 2).

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the grid.

    The scaled grid is defined by an origin and coordinates are shifted to this origin before computing their
    1D grid pixel coordinate values.

    Parameters
    ----------
    grid_scaled_2d_slim
        The slimmed grid of 2D (y,x) coordinates in scaled units which are converted to pixel value coordinates.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted to.

    Returns
    -------
    A slimmed grid of 2D (y,x) pixel-value coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_2d_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_2d_slim = grid_scaled_2d_slim_from(grid_scaled_2d_slim=grid_scaled_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """
    centres_scaled = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )
    centres_scaled = centres_scaled
    pixel_scales = pixel_scales
    sign = np.array([-1, 1])
    return (sign * grid_scaled_2d_slim / pixel_scales) + centres_scaled + 0.5


def grid_pixel_centres_2d_slim_from(
    grid_scaled_2d_slim: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a slimmed grid of 2D (y,x) scaled coordinates to a slimmed grid of 2D (y,x) pixel values. Pixel coordinates
    are returned as integers such that they map directly to the pixel they are contained within.

    The input and output grids are both slimmed and therefore shape (total_pixels, 2).

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the grid.

    The scaled coordinate grid is defined by the class attribute origin, and coordinates are shifted to this
    origin before computing their 1D grid pixel indexes.

    Parameters
    ----------
    grid_scaled_2d_slim
        The slimmed grid of 2D (y,x) coordinates in scaled units which is converted to pixel indexes.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted

    Returns
    -------
    A slimmed grid of 2D (y,x) pixel indexes with dimensions (total_pixels, 2).

    Examples
    --------
    grid_scaled_2d_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixels_2d_slim = grid_scaled_2d_slim_from(grid_scaled_2d_slim=grid_scaled_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    centres_scaled = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )

    centres_scaled = np.array(centres_scaled)
    pixel_scales = np.array(pixel_scales)
    sign = np.array([-1.0, 1.0])
    return ((sign * grid_scaled_2d_slim / pixel_scales) + centres_scaled + 0.5).astype(
        int
    )


def grid_pixel_indexes_2d_slim_from(
    grid_scaled_2d_slim: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a slimmed grid of 2D (y,x) scaled coordinates to a slimmed grid of pixel indexes. Pixel coordinates are
    returned as integers such that they are the pixel from the top-left of the 2D grid going rights and then downwards.

    The input and output grids are both slimmed and have shapes (total_pixels, 2) and (total_pixels,).

    For example:

    The pixel at the top-left, whose native index is [0,0], corresponds to slimmed pixel index 0.
    The fifth pixel on the top row, whose native index is [0,5], corresponds to slimmed pixel index 4.
    The first pixel on the second row, whose native index is [0,1], has slimmed pixel index 10 if a row has 10 pixels.

    The scaled coordinate grid is defined by the class attribute origin, and coordinates are shifted to this
    origin before computing their 1D grid pixel indexes.

    The input and output grids are both of shape (total_pixels, 2).

    Parameters
    ----------
    grid_scaled_2d_slim
        The slimmed grid of 2D (y,x) coordinates in scaled units which is converted to slimmed pixel indexes.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted.

    Returns
    -------
    A grid of slimmed pixel indexes with dimensions (total_pixels,).

    Examples
    --------
    grid_scaled_2d_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixel_indexes_2d_slim = grid_pixel_indexes_2d_slim_from(grid_scaled_2d_slim=grid_scaled_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_pixels_2d_slim = grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid_scaled_2d_slim,
        shape_native=shape_native,
        pixel_scales=pixel_scales,
        origin=origin,
    )

    return (
        (grid_pixels_2d_slim * np.array([shape_native[1], 1])).sum(axis=1).astype(int)
    )


def grid_scaled_2d_slim_from(
    grid_pixels_2d_slim: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a slimmed grid of 2D (y,x) pixel coordinates to a slimmed grid of 2D (y,x) scaled values.

    The input and output grids are both slimmed and therefore shape (total_pixels, 2).

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the grid.

    The scaled coordinate origin is defined by the class attribute origin, and coordinates are shifted to this
    origin after computing their values from the 1D grid pixel indexes.

    Parameters
    ----------
    grid_pixels_2d_slim
        The slimmed grid of (y,x) coordinates in pixel values which is converted to scaled coordinates.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted.

    Returns
    -------
    A slimmed grid of 2d scaled coordinates with dimensions (total_pixels, 2).

    Examples
    --------
    grid_pixels_2d_slim = np.array([[0,0], [0,1], [1,0], [1,1])
    grid_pixels_2d_slim = grid_scaled_2d_slim_from(grid_pixels_2d_slim=grid_pixels_2d_slim, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """
    centres_scaled = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )

    centres_scaled = np.array(centres_scaled)
    pixel_scales = np.array(pixel_scales)
    sign = np.array([-1, 1])
    return (grid_pixels_2d_slim - centres_scaled - 0.5) * pixel_scales * sign


def grid_pixel_centres_2d_from(
    grid_scaled_2d: np.ndarray,
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Convert a native grid of 2D (y,x) scaled coordinates to a native grid of 2D (y,x) pixel values. Pixel coordinates
    are returned as integers such that they map directly to the pixel they are contained within.

    The input and output grids are both native resolution and therefore have shape (y_pixels, x_pixels, 2).

    The pixel coordinate origin is at the top left corner of the grid, such that the pixel [0,0] corresponds to
    the highest (most positive) y scaled coordinate and lowest (most negative) x scaled coordinate on the grid.

    The scaled coordinate grid is defined by the class attribute origin, and coordinates are shifted to this
    origin before computing their 1D grid pixel indexes.

    Parameters
    ----------
    grid_scaled_2d
        The native grid of 2D (y,x) coordinates in scaled units which is converted to pixel indexes.
    shape_native
        The (y,x) shape of the original 2D array the scaled coordinates were computed on.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the original 2D array.
    origin
        The (y,x) origin of the grid, which the scaled grid is shifted

    Returns
    -------
    A native grid of 2D (y,x) pixel indexes with dimensions (y_pixels, x_pixels, 2).

    Examples
    --------
    grid_scaled_2d = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    grid_pixel_centres_2d = grid_pixel_centres_2d_from(grid_scaled_2d=grid_scaled_2d, shape=(2,2),
                                                           pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    centres_scaled = central_scaled_coordinate_2d_from(
        shape_native=shape_native, pixel_scales=pixel_scales, origin=origin
    )

    centres_scaled = np.array(centres_scaled)
    pixel_scales = np.array(pixel_scales)
    sign = np.array([-1.0, 1.0])
    return ((sign * grid_scaled_2d / pixel_scales) + centres_scaled + 0.5).astype(int)


def extent_symmetric_from(
    extent: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Given an input extent of the form (x_min, x_max, y_min, y_max), this function returns an extent which is
    symmetric about the origin.

    For example, if the separation from x_min to x_max is 2.0 and the separation from y_min to y_max is 1.0, the
    returned extent expands the y-axis to also have a separation of 2.0.

    Parameters
    ----------
    extent
        The extent which is to be made symmetric about the origin.

    Returns
    -------
    The new extent which is symmetric about the origin.
    """

    y_min = extent[2]
    y_max = extent[3]
    x_min = extent[0]
    x_max = extent[1]

    y_sep = y_max - y_min
    x_sep = x_max - x_min

    if y_sep > x_sep:
        x_min -= (y_sep - x_sep) / 2
        x_max += (y_sep - x_sep) / 2
    elif x_sep > y_sep:
        y_min -= (x_sep - y_sep) / 2
        y_max += (x_sep - y_sep) / 2

    return (x_min, x_max, y_min, y_max)
