import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def create_grid_hb_order(length, mask_radius):
    """
    This function will create a grid in the Hilbert space-filling curve order.

    length: the size of the square grid.
    mask_radius: the circular mask radius. This code only works with a circular mask.
    """

    from autoarray.util.gilbert_2d import gilbert2d

    xy_generator = gilbert2d(length, length)

    x1d_hb = np.zeros(length * length)
    y1d_hb = np.zeros(length * length)

    count = 0

    for x, y in xy_generator:
        x1d_hb[count] = x
        y1d_hb[count] = y
        count += 1

    x1d_hb /= length
    y1d_hb /= length

    x1d_hb -= 0.5
    y1d_hb -= 0.5

    x1d_hb *= 2.0 * mask_radius
    y1d_hb *= 2.0 * mask_radius

    return x1d_hb, y1d_hb


def create_img_and_grid_hb_order(img_2d, mask, mask_radius, pixel_scales, length_hb):
    """
    This code will create a grid in Hilbert space-filling curve order and an interpolated hyper
    image associated to that grid.
    """

    from scipy.interpolate import griddata
    from autoarray.structures.grids.uniform_2d import Grid2D

    shape_nnn = np.shape(mask)[0]

    grid = Grid2D.uniform(
        shape_native=(shape_nnn, shape_nnn),
        pixel_scales=pixel_scales,
    )

    x1d_hb, y1d_hb = create_grid_hb_order(length=length_hb, mask_radius=mask_radius)

    grid_hb = np.stack((y1d_hb, x1d_hb), axis=-1)
    grid_hb_radius = np.sqrt(grid_hb[:, 0] ** 2.0 + grid_hb[:, 1] ** 2.0)
    new_grid = grid_hb[grid_hb_radius <= mask_radius]

    new_img = griddata(
        points=grid, values=img_2d.ravel(), xi=new_grid, fill_value=0.0, method="linear"
    )

    return new_img, new_grid


def inverse_transform_sampling_interpolated(probabilities, n_samples, gridx, gridy):
    """
    Given a 1d cumulative probability function, this code will generate points following the
    probability distribution.

    probabilities: 1D normalized cumulative probablity curve.
    n_samples: the number of points to draw.
    """
    from scipy.interpolate import interp1d

    cdf = np.cumsum(probabilities)
    npixels = len(probabilities)
    id_range = np.arange(0, npixels)
    cdf[0] = 0.0
    intp_func = interp1d(cdf, id_range, kind="linear")
    intp_func_x = interp1d(id_range, gridx, kind="linear")
    intp_func_y = interp1d(id_range, gridy, kind="linear")
    linear_points = np.linspace(0, 0.99999999, n_samples)
    output_ids = intp_func(linear_points)
    output_x = intp_func_x(output_ids)
    output_y = intp_func_y(output_ids)

    return output_ids, output_x, output_y
