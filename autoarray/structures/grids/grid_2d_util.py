from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.structures.arrays import array_2d_util
from autoarray.geometry import geometry_util
from autoarray import type as ty


def convert_grid(grid: Union[np.ndarray, List]) -> np.ndarray:
    """
    Convert the input grid to a NumPy ``ndarray`` if it is a list, or unwrap it if it is an autoarray object.

    Parameters
    ----------
    grid
        The grid which may be a list, ndarray, or autoarray object.
    """
    try:
        grid = grid.array
    except AttributeError:
        pass

    if isinstance(grid, list):
        grid = np.asarray(grid)

    return grid


def check_grid_slim(grid, shape_native):
    """
    Validate that a grid is not in slim format without an accompanying ``shape_native``, and that ``shape_native``
    is a valid 2-tuple of ints.

    Raises ``GridException`` if the grid is 2D (slim) but no ``shape_native`` was provided, or if
    ``shape_native`` is not a (int, int) tuple.

    Parameters
    ----------
    grid
        The input grid to validate.
    shape_native
        The expected native shape of the grid, required when the input is in slim format.
    """
    if shape_native is None:
        raise exc.GridException(
            f"""
            The input grid is not in its native shape (an ndarray / list of shape [total_y_pixels, total_x_pixels, 2])
            and the shape_native parameter has not been input the Grid2D function.

            Either change the input array to be its native shape or input its shape_native input the function.

            The shape of the input array is {grid.shape}
            """
        )

    if shape_native and len(shape_native) != 2:
        raise exc.GridException(
            """
            The input shape_native parameter is not a tuple of type (int, int).
            """
        )


def check_grid_2d(grid_2d: np.ndarray):
    """
    Validate that a grid has a final dimension of 2 (for (y,x) coordinate pairs) and is either 2D or 3D.

    Raises ``GridException`` if either condition is violated.

    Parameters
    ----------
    grid_2d
        The grid ndarray to validate.
    """
    if grid_2d.shape[-1] != 2:
        raise exc.GridException(
            "The final dimension of the input grid is not equal to 2 (e.g. the (y,x) coordinates)"
        )

    if 2 < len(grid_2d.shape) > 3:
        raise exc.GridException("The dimensions of the input grid array is not 2 or 3")


def check_grid_2d_and_mask_2d(grid_2d: np.ndarray, mask_2d: Mask2D):
    """
    Validate that a grid and mask are compatible, i.e. that the grid's shape is consistent with the number of
    unmasked pixels (for slim format) or the mask's native shape (for native format).

    Raises ``GridException`` if the shapes are inconsistent.

    Parameters
    ----------
    grid_2d
        The grid ndarray to validate, either slim [total_unmasked_pixels, 2] or native [y, x, 2].
    mask_2d
        The mask to compare against.
    """
    if len(grid_2d.shape) == 2:
        if grid_2d.shape[0] != mask_2d.pixels_in_mask:
            raise exc.GridException(
                f"""
                The input 2D grid does not have the same number of values as pixels in
                the mask.

                The shape of the input grid_2d is {grid_2d.shape}.
                The mask shape_native is {mask_2d.shape_native}.
                The mask number of pixels is {mask_2d.pixels_in_mask}. 
                """
            )

    elif len(grid_2d.shape) == 3:
        if (grid_2d.shape[0], grid_2d.shape[1]) != mask_2d.shape_native:
            raise exc.GridException(
                f"""
                The input 2D grid is not the same dimensions as the mask
                (e.g. the mask 2D shape.)

                The shape of the input grid_2d is {grid_2d.shape}.
                The mask shape_native is {mask_2d.shape_native}.
                """
            )


def convert_grid_2d(
    grid_2d: Union[np.ndarray, List], mask_2d: Mask2D, store_native: bool = False, xp=np
) -> np.ndarray:
    """
    The `manual` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D.

    This function performs the following and checks and conversions on the input:

    1: If the input is a list, convert it to an ndarray.
    2: Check that the number of coordinates in the grid is identical to that of the mask.
    3) Map the input ndarray to its `slim` representation.

    For a Grid2D, `slim` refers to a 2D NumPy array of shape [total_coordinates, 2] and `native` a 3D NumPy array of
    shape [total_y_coordinates, total_x_coordinates, 2]

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to a ndarray if it is a list.
    mask_2d
        The mask of the output Array2D.
    store_native
        If True, the ndarray is stored in its native format [total_y_pixels, total_x_pixels, 2]. This avoids
        mapping large data arrays to and from the slim / native formats, which can be a computational bottleneck.
    """

    grid_2d = convert_grid(grid=grid_2d)

    check_grid_2d_and_mask_2d(grid_2d=grid_2d, mask_2d=mask_2d)

    is_native = len(grid_2d.shape) == 3

    if is_native:
        grid_2d = grid_2d * (~mask_2d)[..., None]

    if is_native == store_native:
        return grid_2d
    elif not store_native:
        return grid_2d_slim_from(grid_2d_native=grid_2d, mask=mask_2d, xp=xp)
    return grid_2d_native_from(grid_2d_slim=grid_2d, mask_2d=mask_2d, xp=xp)  #


def convert_grid_2d_to_slim(
    grid_2d: Union[np.ndarray, List], mask_2d: Mask2D, xp=np
) -> np.ndarray:
    """
    The ``manual`` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D.

    This function checks the dimensions of the input ``grid_2d`` and maps it to its ``slim`` representation.

    For a Grid2D, ``slim`` refers to a 2D NumPy array of shape [total_coordinates, 2].

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to its slim representation.
    mask_2d
        The mask of the output Grid2D.
    """
    if len(grid_2d.shape) == 2:
        return grid_2d
    return grid_2d_slim_from(grid_2d_native=grid_2d, mask=mask_2d, xp=xp)


def convert_grid_2d_to_native(
    grid_2d: Union[np.ndarray, List],
    mask_2d: Mask2D,
) -> np.ndarray:
    """
    The ``manual`` classmethods in the Grid2D object take as input a list or ndarray which is returned as a Grid2D.

    This function checks the dimensions of the input ``grid_2d`` and maps it to its ``native`` representation.

    For a Grid2D, ``native`` refers to a 3D NumPy array of shape [total_y_coordinates, total_x_coordinates, 2].

    Parameters
    ----------
    grid_2d
        The input (y,x) grid of coordinates which is converted to its native representation.
    mask_2d
        The mask of the output Grid2D.
    """
    if len(grid_2d.shape) == 3:
        return grid_2d
    return grid_2d_native_from(
        grid_2d_slim=grid_2d,
        mask_2d=mask_2d,
    )


def grid_2d_centre_from(grid_2d_slim: np.ndarray) -> Tuple[float, float]:
    """
    Returns the centre of a grid from a 1D grid.

    Parameters
    ----------
    grid_2d_slim
        The 1D grid of values which are mapped to a 2D array.

    Returns
    -------
    (float, float)
        The (y,x) central coordinates of the grid.
    """
    centre_y = (np.max(grid_2d_slim[:, 0]) + np.min(grid_2d_slim[:, 0])) / 2.0
    centre_x = (np.max(grid_2d_slim[:, 1]) + np.min(grid_2d_slim[:, 1])) / 2.0
    return centre_y, centre_x


def grid_2d_slim_via_mask_from(
    mask_2d: np.ndarray,
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
    xp=np,
) -> np.ndarray:
    """
    For a grid, every unmasked pixel is on a 2D mask with shape (total_y_pixels, total_x_pixels). This routine
    computes the (y,x) scaled coordinates a the centre of every pixel defined by this 2D mask array.

    The grid is returned on an array of shape (total_unmasked_pixels, 2). y coordinates are stored in the 0 index of
    the second dimension, x coordinates in the 1 index. Masked coordinates are therefore removed and not included in
    the slimmed grid.

    Grid2D are defined from the top-left corner, where the first unmasked pixel corresponds to index 0.

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        grid.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    origin
        The (y,x) origin of the 2D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A slimmed grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The grid array has dimensions (total_unmasked_pixels, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_slim = grid_2d_slim_via_mask_from(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    centres_scaled = geometry_util.central_scaled_coordinate_2d_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, origin=origin
    )

    centres_scaled = xp.asarray(centres_scaled)
    pixel_scales = xp.asarray(pixel_scales)
    sign = xp.array([-1.0, 1.0])

    rows, cols = xp.where(~mask_2d.astype(bool))
    indices = xp.stack([rows, cols], axis=1)  # shape (N_unmasked, 2)

    # (indices - centre) -> pixel offsets; apply sign and scale to get physical coords
    return (indices - centres_scaled) * sign * pixel_scales


def grid_2d_via_mask_from(
    mask_2d: np.ndarray,
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
    xp=np,
) -> np.ndarray:
    """
    For a grid, every unmasked pixel is on a 2D mask with shape (total_y_pixels, total_x_pixels). This routine computes
    the (y,x) scaled coordinates at the centre of every pixel defined by this 2D mask array.

    The grid is returned in its native dimensions with shape (total_y_pixels, total_x_pixels). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index. Masked pixels are given
    values (0.0, 0.0).

    Grids are defined from the top-left corner, where the first unmasked pixel corresponds to index 0.

    Parameters
    ----------
    mask_2d
        A 2D array of bools, where `False` values are unmasked and therefore included as part of the calculated
        grid.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    origin
        The (y,x) origin of the 2D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The grid array has dimensions (total_y_pixels, total_x_pixels).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_2d = grid_2d_via_mask_from(mask=mask, pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """

    grid_2d_slim = grid_2d_slim_via_mask_from(
        mask_2d=mask_2d, pixel_scales=pixel_scales, origin=origin, xp=xp
    )

    return grid_2d_native_from(grid_2d_slim=grid_2d_slim, mask_2d=mask_2d, xp=xp)


def grid_2d_slim_via_shape_native_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
    xp=np,
) -> np.ndarray:
    """
    For a grid, every unmasked pixel is in a 2D mask with shape (total_y_pixels, total_x_pixels). This routine computes
    the (y,x) scaled coordinates at the centre of every pixel defined by this 2D mask array.

    The grid is returned in its slimmed dimensions with shape (total_pixels, 2). y coordinates are
    stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Grid2D are defined from the top-left corner, where the first pixel corresponds to index [0,0].

    Parameters
    ----------
    shape_native
        The (y,x) shape of the 2D array the grid of coordinates is computed for.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    origin
        The (y,x) origin of the 2D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The grid is slimmed and has dimensions (total_unmasked_pixels, 2).

    Examples
    --------
    mask = np.array([[True, False, True],
                     [False, False, False]
                     [True, False, True]])
    grid_2d_slim = grid_2d_slim_via_shape_native_from(shape_native=(3,3), pixel_scales=(0.5, 0.5), origin=(0.0, 0.0))
    """
    return grid_2d_slim_via_mask_from(
        mask_2d=xp.full(fill_value=False, shape=shape_native),
        pixel_scales=pixel_scales,
        origin=origin,
        xp=xp,
    )


def grid_2d_via_shape_native_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    For a grid, every unmasked pixel is in a 2D mask with shape (total_y_pixels, total_x_pixels). This routine computes
    the (y,x) scaled coordinates at the centre of every pixel defined by this 2D mask array.

    The grid is returned in its native dimensions with shape (total_y_pixels, total_x_pixels).
    y coordinates are stored in the 0 index of the second dimension, x coordinates in the 1 index.

    Grids are defined from the top-left corner, where the first pixel corresponds to index [0,0].

    Parameters
    ----------
    shape_native
        The (y,x) shape of the 2D array the grid of coordinates is computed for.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    origin
        The (y,x) origin of the 2D array, which the grid is shifted around.

    Returns
    -------
    ndarray
        A grid of (y,x) scaled coordinates at the centre of every pixel unmasked pixel on the 2D mask
        array. The grid array has dimensions (total_y_pixels, total_x_pixels).

    Examples
    --------
    grid_2d = grid_2d_via_shape_native_from(shape_native=(3, 3), pixel_scales=(1.0, 1.0), origin=(0.0, 0.0))
    """
    return grid_2d_via_mask_from(
        mask_2d=np.full(fill_value=False, shape=shape_native),
        pixel_scales=pixel_scales,
        origin=origin,
    )


def _radial_projected_shape_slim_from(
    extent: np.ndarray,
    centre: Tuple[float, float],
    pixel_scales: ty.PixelScales,
) -> int:
    """
    The function `grid_scaled_2d_slim_radial_projected_from()` determines a projected radial grid of points from a 2D
    region of coordinates defined by an extent [xmin, xmax, ymin, ymax] and with a (y,x) centre.

    To do this, the function first performs these 3 steps:

    1) Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance of
    the 4 paths from the (y,x) centre to the edge of the region (e.g. following the positive / negative y and x axes).

    2) Use the pixel-scale corresponding to the direction chosen (e.g. if the positive x-axis was the longest, the
    pixel_scale in the x dimension is used).

    3) Determine the number of pixels between the centre and the edge of the region using the longest path between the
    two chosen above.

    A schematic is shown below:

    -------------------
    |                 |
    |<- - -  - ->x    | x = centre
    |                 | <-> = longest radial path from centre to extent edge
    |                 |
    -------------------

    Using the centre x above, this function finds the longest radial path to the edge of the extent window.

    This function returns the integer number of pixels given by this radial grid, which is then used to create
    the radial grid.

    Parameters
    ----------
    extent
        The extent of the grid the radii grid is computed using, with format [xmin, xmax, ymin, ymax]
    centre : (float, float)
        The (y,x) central coordinate which the radial grid is traced outwards from.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.

    Returns
    -------
    int
        The 1D integer shape of a radial set of points sampling the longest distance from the centre to the edge of the
        extent in along the positive x-axis.
    """

    distance_to_positive_x = extent[1] - centre[1]
    distance_to_positive_y = extent[3] - centre[0]
    distance_to_negative_x = centre[1] - extent[0]
    distance_to_negative_y = centre[0] - extent[2]

    scaled_distance = max(
        [
            distance_to_positive_x,
            distance_to_positive_y,
            distance_to_negative_x,
            distance_to_negative_y,
        ]
    )

    if (scaled_distance == distance_to_positive_y) or (
        scaled_distance == distance_to_negative_y
    ):
        pixel_scale = pixel_scales[0]
    else:
        pixel_scale = pixel_scales[1]

    return int((scaled_distance / pixel_scale)) + 1


def grid_scaled_2d_slim_radial_projected_from(
    extent: np.ndarray,
    centre: Tuple[float, float],
    pixel_scales: ty.PixelScales,
    shape_slim: Optional[int] = 0,
) -> np.ndarray:
    """
    Determine a projected radial grid of points from a 2D region of coordinates defined by an
    extent [xmin, xmax, ymin, ymax] and with a (y,x) centre.

    This functions operates as follows:

    1) Given the region defined by the extent [xmin, xmax, ymin, ymax], the algorithm finds the longest 1D distance of
    the 4 paths from the (y,x) centre to the edge of the region (e.g. following the positive / negative y and x axes).

    2) Use the pixel-scale corresponding to the direction chosen (e.g. if the positive x-axis was the longest, the
    pixel_scale in the x dimension is used).

    3) Determine the number of pixels between the centre and the edge of the region using the longest path between the
    two chosen above.

    4) Create a (y,x) grid of radial points where all points are at the centre's y value = 0.0 and the x values iterate
    from the centre in increasing steps of the pixel-scale.

    5) Rotate these radial coordinates by the input `angle` clockwise.

    A schematric is shown below:

    -------------------
    |                 |
    |<- - -  - ->x    | x = centre
    |                 | <-> = longest radial path from centre to extent edge
    |                 |
    -------------------

    Using the centre x above, this function finds the longest radial path to the edge of the extent window.

    The returned `grid_radii` represents a radial set of points that in 1D sample the 2D grid outwards from its centre.
    This grid stores the radial coordinates as (y,x) values (where all y values are the same) as opposed to a 1D data
    structure so that it can be used in functions which require that a 2D grid structure is input.

    Parameters
    ----------
    extent
        The extent of the grid the radii grid is computed using, with format [xmin, xmax, ymin, ymax]
    centre : (float, float)
        The (y,x) central coordinate which the radial grid is traced outwards from.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D mask array.
    shape_slim
        Manually choose the shape of the 1D projected grid that is returned. If 0, the border based on the 2D grid is
        used.

    Returns
    -------
    ndarray
        A radial set of points sampling the longest distance from the centre to the edge of the extent in along the
        positive x-axis.
    """

    distance_to_positive_x = extent[1] - centre[1]
    distance_to_positive_y = extent[3] - centre[0]
    distance_to_negative_x = centre[1] - extent[0]
    distance_to_negative_y = centre[0] - extent[2]

    scaled_distance = max(
        [
            distance_to_positive_x,
            distance_to_positive_y,
            distance_to_negative_x,
            distance_to_negative_y,
        ]
    )

    if (scaled_distance == distance_to_positive_y) or (
        scaled_distance == distance_to_negative_y
    ):
        pixel_scale = pixel_scales[0]
    else:
        pixel_scale = pixel_scales[1]

    if shape_slim == 0:
        shape_slim = int((scaled_distance / pixel_scale)) + 1

    grid_scaled_2d_slim_radii = np.zeros((shape_slim, 2))

    grid_scaled_2d_slim_radii[:, 0] += centre[0]

    radii = centre[1]

    # Create an array of radii values spaced by pixel_scale
    radii_array = radii + pixel_scale * np.arange(shape_slim)

    # Assign all values at once to the second column (index 1)
    grid_scaled_2d_slim_radii[:, 1] = radii_array

    return grid_scaled_2d_slim_radii + 1e-6


def grid_2d_slim_from(
    grid_2d_native: np.ndarray, mask: np.ndarray, xp=np
) -> np.ndarray:
    """
    For a native 2D grid and mask of shape [total_y_pixels, total_x_pixels, 2], map the values of all unmasked
    pixels to a slimmed grid of shape [total_unmasked_pixels, 2].

    The pixel coordinate origin is at the top left corner of the native grid and goes right-wards and downwards, such
    that for an grid of shape (3,3) where all pixels are unmasked:

    - pixel [0,0] of the 2D grid will correspond to index 0 of the 1D grid.
    - pixel [0,1] of the 2D grid will correspond to index 1 of the 1D grid.
    - pixel [1,0] of the 2D grid will correspond to index 4 of the 1D grid.

    Parameters
    ----------
    grid_2d_native : ndarray
        The native grid of (y,x) values which are mapped to the slimmed grid.
    mask
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.

    Returns
    -------
    ndarray
        A 1D grid of values mapped from the 2D grid with dimensions (total_unmasked_pixels).
    """

    grid_1d_slim_y = array_2d_util.array_2d_slim_from(
        array_2d_native=grid_2d_native[:, :, 0],
        mask_2d=mask,
    )

    grid_1d_slim_x = array_2d_util.array_2d_slim_from(
        array_2d_native=grid_2d_native[:, :, 1],
        mask_2d=mask,
    )
    return xp.stack((grid_1d_slim_y, grid_1d_slim_x), axis=-1)


def grid_2d_native_from(
    grid_2d_slim: np.ndarray, mask_2d: np.ndarray, xp=np
) -> np.ndarray:
    """
    For a slimmed 2D grid of shape [total_unmasked_pixels, 2], that was computed by extracting the unmasked values
    from a native 2D grid of shape [total_y_pixels, total_x_pixels, 2], map the slimmed grid's coordinates back to the
    native 2D grid where masked values are set to zero.

    This uses a 1D array 'slim_to_native' where each index gives the 2D pixel indexes of the grid's native unmasked
    pixels, for example:

    - If slim_to_native[0] = [0,0], the first value of the 1D array maps to the pixels [0,0,:] of the native 2D grid.
    - If slim_to_native[1] = [0,1], the second value of the 1D array maps to the pixels [0,1,:] of the native 2D grid.
    - If slim_to_native[4] = [1,1], the fifth value of the 1D array maps to the pixels [1,1,:] of the native 2D grid.

    Parameters
    ----------
    grid_2d_slim
        The (y,x) values of the slimmed 2D grid which are mapped to the native 2D grid.
    mask_2d
        A 2D array of bools, where `False` values mean unmasked and are included in the mapping.

    Returns
    -------
    ndarray
        A NumPy array of shape [total_y_pixels, total_x_pixels, 2] corresponding to the (y,x) values of the native 2D
        mapped from the slimmed grid.
    """

    grid_2d_native_y = array_2d_util.array_2d_native_from(
        array_2d_slim=grid_2d_slim[:, 0], mask_2d=mask_2d, xp=xp
    )

    grid_2d_native_x = array_2d_util.array_2d_native_from(
        array_2d_slim=grid_2d_slim[:, 1], mask_2d=mask_2d, xp=xp
    )

    return xp.stack((grid_2d_native_y, grid_2d_native_x), axis=-1)


def grid_2d_of_points_within_radius(
    radius: float, centre: Tuple[float, float], grid_2d: np.ndarray
):
    y_inside = []
    x_inside = []

    for i in range(len(grid_2d[:, 0])):
        if (grid_2d[i, 0] - centre[0]) ** 2 + (
            grid_2d[i, 1] - centre[1]
        ) ** 2 > radius**2:
            y_inside.append(grid_2d[i, 0])
            x_inside.append(grid_2d[i, 1])

    return np.asarray(y_inside, x_inside)


def grid_pixels_in_mask_pixels_from(
    grid, shape_native, pixel_scales, origin
) -> np.ndarray:
    """
    Returns an array containing the number of pixels of one grid in every pixel of another masked grid.

    For example, image-mesh adaption may be performed on a 3.0" circular mask of data. The high weight pixels
    may have 3 or more mesh pixels per image pixel, whereas low weight regions may have zero pixels. The array
    returned by this function gives the integer number of pixels in each data pixel.

    Parameters
    ----------
    grid_pixel_centres
        The 2D integer index of every image pixel that each image-mesh pixel falls within.
    shape_native
        The 2D shape of the data's mask, which the number of image-mesh pixels that fall within eac pixel is counted.

    Returns
    -------
    An array containing the integer number of image-mesh pixels that fall without each of the data's mask.
    """
    grid_pixel_centres = geometry_util.grid_pixel_centres_2d_slim_from(
        grid_scaled_2d_slim=grid,
        shape_native=shape_native,
        pixel_scales=pixel_scales,
        origin=origin,
    ).astype("int")

    mesh_pixels_per_image_pixel = np.zeros(shape=shape_native)

    # Assuming grid_pixel_centres is a 2D array where each row contains (y, x) indices.
    y_indices = grid_pixel_centres[:, 0]
    x_indices = grid_pixel_centres[:, 1]

    # Use np.add.at to increment the specific indices in a safe and efficient manner
    np.add.at(mesh_pixels_per_image_pixel, (y_indices, x_indices), 1)

    return mesh_pixels_per_image_pixel


def grid_2d_slim_via_shape_native_not_mask_from(
    shape_native: Tuple[int, int],
    pixel_scales: Tuple[float, float],
    origin: Tuple[float, float] = (0.0, 0.0),
    xp=np,
) -> np.ndarray:
    """
    Build the slim (flattened) grid of all (y, x) pixel centres for a rectangular grid
    of shape `shape_native`, scaled by `pixel_scales` and shifted by `origin`.

    This is equivalent to taking an unmasked mask of shape `shape_native` and calling
    grid_2d_slim_via_mask_from on it.

    Parameters
    ----------
    shape_native
        A pair (Ny, Nx) giving the number of pixels in y and x.
    pixel_scales
        A pair (sy, sx) giving the physical size of each pixel in y and x.
    origin
        A 2-tuple (y0, x0) around which the grid is centred.

    Returns
    -------
    grid_slim : ndarray, shape (Ny*Nx, 2)
        Each row is the (y, x) coordinate of one pixel centre, in row-major order,
        shifted so that `origin` ↔ physical pixel-centre average, and scaled by
        `pixel_scales`, with y increasing “up” and x increasing “right”.
    """
    Ny, Nx = shape_native
    sy, sx = pixel_scales
    y0, x0 = origin

    # compute the integer pixel‐centre coordinates in array index space
    # row indices 0..Ny-1, col indices 0..Nx-1
    arange = xp.arange
    meshy, meshx = xp.meshgrid(arange(Ny), arange(Nx), indexing="ij")
    coords = xp.stack([meshy, meshx], axis=-1).reshape(-1, 2)

    # convert to physical coordinates: subtract array‐centre, flip y, scale, then add origin
    # array‐centre in index space is at ((Ny-1)/2, (Nx-1)/2)
    cy, cx = (Ny - 1) / 2.0, (Nx - 1) / 2.0
    # row index i → physical y =  (cy - i) * sy + y0
    # col index j → physical x =  (j  - cx) * sx + x0
    idx_y = coords[:, 0]
    idx_x = coords[:, 1]

    phys_y = (cy - idx_y) * sy + y0
    phys_x = (idx_x - cx) * sx + x0

    return xp.stack([phys_y, phys_x], axis=1)
