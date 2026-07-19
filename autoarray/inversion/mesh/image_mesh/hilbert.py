from __future__ import annotations
import numpy as np
import os

from typing import Optional

from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.mask.mask_2d import Mask2D
from autoarray.inversion.mesh.image_mesh.abstract_weighted import (
    AbstractImageMeshWeighted,
)
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray import exc
from autonerves.test_mode import skip_checks


def gilbert2d(width, height):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
    of size (width x height).
    """

    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):
    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay))  # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by))  # unit orthogonal direction

    if h == 1:
        # trivial row fill
        for i in range(0, w):
            yield (x, y)
            (x, y) = (x + dax, y + day)
        return

    if w == 1:
        # trivial column fill
        for i in range(0, h):
            yield (x, y)
            (x, y) = (x + dbx, y + dby)
        return

    (ax2, ay2) = (ax // 2, ay // 2)
    (bx2, by2) = (bx // 2, by // 2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        # long case: split in two parts only
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from generate2d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
        )


def grid_hilbert_order_from(length, mask_radius):
    """
    This function will create a grid in the Hilbert space-filling curve order.

    length: the size of the square grid.
    mask_radius: the circular mask radius. This code only works with a circular mask.
    """

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


def image_and_grid_from(
    image, mask, mask_radius, pixel_scales, hilbert_length, mask_centre=(0.0, 0.0)
):
    """
    This code will create a grid in Hilbert space-filling curve order and an interpolated hyper
    image associated to that grid.

    The Hilbert curve is generated around the origin and filtered to a disc of radius ``mask_radius``, then translated
    by ``mask_centre`` so the sampling lands inside the mask's circle for offset-centre masks. For masks centred on
    the origin the translation is a no-op and behaviour is unchanged.
    """

    from scipy.interpolate import griddata

    # For multi wavelength fits the input image may be a different resolution than the mask.

    try:
        shape_nnn = np.shape(image.native)[0]
    except AttributeError:
        shape_nnn = np.shape(mask)[0]

    grid = Grid2D.uniform(
        shape_native=(shape_nnn, shape_nnn),
        pixel_scales=pixel_scales,
    )

    x1d_hb, y1d_hb = grid_hilbert_order_from(
        length=hilbert_length, mask_radius=mask_radius
    )

    grid_hb = np.stack((y1d_hb, x1d_hb), axis=-1)
    grid_hb_radius = np.sqrt(grid_hb[:, 0] ** 2.0 + grid_hb[:, 1] ** 2.0)
    new_grid = grid_hb[grid_hb_radius <= mask_radius]
    new_grid = new_grid + np.asarray(mask_centre)

    new_img = griddata(
        points=grid,
        values=image.native.ravel(),
        xi=new_grid,
        fill_value=0.0,
        method="linear",
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
    linear_points = np.linspace(0, 0.99999999, int(n_samples))
    output_ids = intp_func(linear_points)
    output_x = intp_func_x(output_ids)
    output_y = intp_func_y(output_ids)

    return output_ids, output_x, output_y


class Hilbert(AbstractImageMeshWeighted):
    def __init__(
        self,
        pixels=10.0,
        weight_floor=0.0,
        weight_power=0.0,
    ):
        """
        Computes an image-mesh by computing the Hilbert curve of the adapt data and drawing points from it.

        This requires an adapt-image, which is the image that the Hilbert curve algorithm adapts to in order to compute
        the image mesh. This could simply be the image itself, or a model fit to the image which removes certain
        features or noise.

        For example, using the adapt image, the image mesh is computed as follows:

        1) Convert the adapt image to a weight map, which is a 2D array of weight values.

        2) Run the Hilbert algorithm on the weight map, such that the image mesh pixels cluster around the weight map
        values with higher values.

        Parameters
        ----------
        pixels
            The total number of pixels in the image mesh and drawn from the Hilbert curve.
        weight_floor
            The minimum weight value in the weight map, which allows more pixels to be drawn from the lower weight
            regions of the adapt image.
        weight_power
            The power the weight values are raised too, which allows more pixels to be drawn from the higher weight
            regions of the adapt image.

        image_mesh_min_mesh_pixels_per_pixel
            If not None, the image-mesh must place this many mesh pixels per image pixels in the N highest weighted
            regions of the adapt data, or an `InversionException` is raised. This can be used to force the image-mesh
            to cluster large numbers of source pixels to the adapt-datas brightest regions.
        image_mesh_min_mesh_number
            The value N given above in the docstring for `image_mesh_min_mesh_pixels_per_pixel`, indicating how many
            image pixels are checked for having a threshold number of mesh pixels.
        image_mesh_adapt_background_percent_threshold
            If not None, the image-mesh must place this percentage of mesh-pixels in the background regions of the
            `adapt_data`, where the background is the `image_mesh_adapt_background_percent_check` masked data pixels
            with the lowest values.
        image_mesh_adapt_background_percent_check
            The percentage of masked data pixels which are checked for the background criteria.
        """

        super().__init__(
            pixels=pixels,
            weight_floor=weight_floor,
            weight_power=weight_power,
        )

    def image_plane_mesh_grid_from(
        self,
        mask: Mask2D,
        adapt_data: Optional[np.ndarray],
    ) -> Grid2DIrregular:
        """
        Returns an image mesh by running the Hilbert curve on the weight map.

        See the `__init__` docstring for a full description of how this is performed.

        Parameters
        ----------
        grid
            The grid of (y,x) coordinates of the image data the pixelization fits, which the Hilbert curve adapts to.
        adapt_data
            The weights defining the regions of the image the Hilbert curve adapts to.

        Returns
        -------

        """
        if not mask.is_circular:
            raise exc.PixelizationException(
                """
                Hilbert image-mesh has been called but the input grid does not use a circular mask.
                
                Ensure that analysis is using a circular mask via the Mask2D.circular classmethod.
                """
            )

        adapt_data_hb, grid_hb = image_and_grid_from(
            image=adapt_data,
            mask=mask,
            mask_radius=mask.circular_radius,
            pixel_scales=mask.pixel_scales,
            hilbert_length=193,
            mask_centre=mask.mask_centre,
        )

        weight_map = self.weight_map_from(adapt_data=adapt_data_hb)

        weight_map /= np.sum(weight_map)

        (
            drawn_id,
            drawn_x,
            drawn_y,
        ) = inverse_transform_sampling_interpolated(
            probabilities=weight_map,
            n_samples=self.pixels,
            gridx=grid_hb[:, 1],
            gridy=grid_hb[:, 0],
        )

        return Grid2DIrregular(values=np.stack((drawn_y, drawn_x), axis=-1))

    def check_mesh_pixels_per_image_pixels(
        self,
        mask: Mask2D,
        mesh_grid: Grid2DIrregular,
        image_mesh_min_mesh_pixels_per_pixel=None,
        image_mesh_min_mesh_number: int = 5,
        image_mesh_adapt_background_percent_threshold: float = None,
        image_mesh_adapt_background_percent_check: float = 0.8,
    ):
        """
        Checks the number of mesh pixels in every image pixel and raises an `InversionException` if there are fewer
        mesh pixels inside a certain number of image-pixels than the input settings.

        This allows a user to force a model-fit to use image-mesh's which cluster a large number of mesh pixels to
        the brightest regions of the image data (E.g. the highst weighted regions).

        The check works as follows:

        1) Compute the 2D array of the number of mesh pixels in every masked data image pixel.
        2) Find the number of mesh pixels in the N data pixels with the larger number of mesh pixels, where N is
           given by `image_mesh_min_mesh_number`. For example, if `image_mesh_min_mesh_number=5` then
           the number of mesh pixels in the 5 data pixels with the most data pixels is computed.
        3) Compare the lowest value above to the value `image_mesh_min_mesh_pixels_per_pixel`. If the value is
           below this value, raise an `InversionException`.

        Therefore, by settings `image_mesh_min_mesh_pixels_per_pixel` to a value above 1 the code is forced
        to adapt the image mesh enough to put many mesh pixels in the brightest image pixels.

        Parameters
        ----------
        mask
            The mask of the dataset being analysed, which the pixelization grid maps too. The number of
            mesh pixels mapped inside each of this mask's image-pixels is returned.
        mesh_grid
            The image mesh-grid computed by the class which adapts to the data's mask. The number of image mesh pixels
            that fall within each of the data's mask pixels is returned.
        settings
            The inversion settings, which have the criteria dictating if the image-mesh has clustered enough or if
            an exception is raised.
        """

        if skip_checks():
            return

        if image_mesh_min_mesh_pixels_per_pixel is not None:
            mesh_pixels_per_image_pixels = self.mesh_pixels_per_image_pixels_from(
                mask=mask, mesh_grid=mesh_grid
            )

            indices_of_highest_values = np.argsort(mesh_pixels_per_image_pixels)[
                -image_mesh_min_mesh_number:
            ]
            lowest_mesh_pixels = np.min(
                mesh_pixels_per_image_pixels[indices_of_highest_values]
            )

            if lowest_mesh_pixels < image_mesh_min_mesh_pixels_per_pixel:
                raise exc.InversionException()

        return mesh_grid

    def check_adapt_background_pixels(
        self,
        mask: Mask2D,
        mesh_grid: Grid2DIrregular,
        adapt_data: Optional[np.ndarray],
        image_mesh_min_mesh_pixels_per_pixel=None,
        image_mesh_min_mesh_number: int = 5,
        image_mesh_adapt_background_percent_threshold: float = None,
        image_mesh_adapt_background_percent_check: float = 0.8,
    ):
        """
        Checks the number of mesh pixels in the background of the image-mesh and raises an `InversionException` if
        there are fewer mesh pixels in the background than the input settings.

        This allows a user to force a model-fit to use image-mesh's which cluster a minimum number of mesh pixels to
        the faintest regions of the image data (E.g. the lowest weighted regions). This prevents too few image-mesh
        pixels being allocated to the background of the data.

        The check works as follows:

        1) Find all pixels in the background of the `adapt_data`, which are N pixels with the lowest values, where N is
           a percentage given by `image_mesh_adapt_background_percent_check`. If N is 50%, then the half of
            pixels in `adapt_data` with the lowest values will be checked.
        2) Sum the total number of mesh pixels in these background pixels, thereby estimating the number of mesh pixels
            assigned to background pixels.
        3) Compare this value to the total number of mesh pixels multiplied
           by `image_mesh_adapt_background_percent_threshold` and raise an `InversionException` if the number
           of mesh pixels is below this value, meaning the background did not have sufficient mesh pixels in it.

        Therefore, by setting `image_mesh_adapt_background_percent_threshold` the code is forced
        to adapt the image mesh in a way that places many mesh pixels in the background regions.

        Parameters
        ----------
        mask
            The mask of the dataset being analysed, which the pixelization grid maps too. The number of
            mesh pixels mapped inside each of this mask's image-pixels is returned.
        mesh_grid
            The image mesh-grid computed by the class which adapts to the data's mask. The number of image mesh pixels
            that fall within each of the data's mask pixels is returned.
        adapt_data
            A image which represents one or more components in the masked 2D data in the image-plane.
        settings
            The inversion settings, which have the criteria dictating if the image-mesh has clustered enough or if
            an exception is raised.
        """

        if skip_checks():
            return

        if image_mesh_adapt_background_percent_threshold is not None:
            pixels = mesh_grid.shape[0]

            pixels_in_background = int(
                mask.shape_slim * image_mesh_adapt_background_percent_check
            )

            indices_of_lowest_values = np.argsort(adapt_data)[:pixels_in_background]
            mask_background = np.zeros_like(adapt_data, dtype=bool)
            mask_background[indices_of_lowest_values] = True

            mesh_pixels_per_image_pixels = self.mesh_pixels_per_image_pixels_from(
                mask=mask, mesh_grid=mesh_grid
            )

            mesh_pixels_in_background = sum(
                mesh_pixels_per_image_pixels[mask_background]
            )

            if mesh_pixels_in_background < (
                pixels * image_mesh_adapt_background_percent_threshold
            ):
                raise exc.InversionException()
