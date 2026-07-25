import numpy as np

from autoarray.structures import decorators
from autoarray.operators.over_sampling.decorator import over_sample


### Grids ###


def radial_grid_from(grid, *args, **kwargs):
    return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))


def ndarray_1d_from(profile, grid, *args, **kwargs):
    sersic_constant = (
        (2 * 2.0)
        - (1.0 / 3.0)
        + (4.0 / (405.0 * 2.0))
        + (46.0 / (25515.0 * 2.0**2))
        + (131.0 / (1148175.0 * 2.0**3))
        - (2194697.0 / (30690717750.0 * 2.0**4))
    )

    grid_radii = radial_grid_from(grid=grid)

    return np.exp(
        np.multiply(
            -sersic_constant,
            np.add(np.power(np.divide(grid_radii, 0.2), 1.0 / 2.0), -1),
        )
    )


def ndarray_1d_zeros_from(profile, grid, *args, **kwargs):
    return np.zeros(shape=(5, 5))


def ndarray_1d_list_from(profile, grid, *args, **kwargs):
    return [ndarray_1d_from(profile, grid)]


def angle_to_profile_grid_from(grid_angles, *args, **kwargs):
    """The angle between each (y,x) coordinate on the grid and the profile, in radians.

    Parameters
    ----------
    grid_angles
        The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
    """
    return np.cos(grid_angles), np.sin(grid_angles)


def _cartesian_grid_via_radial_from(grid, radius, *args, **kwargs):
    """
    Convert a grid of (y,x) coordinates with their specified circular radii to their original (y,x) Cartesian
    coordinates.

    Parameters
    ----------
    grid
        The (y, x) coordinates in the reference frame of the profile.
    radius
        The circular radius of each coordinate from the profile center.
    """
    grid_angles = np.arctan2(grid[:, 0], grid[:, 1])
    cos_theta, sin_theta = angle_to_profile_grid_from(grid_angles=grid_angles)
    return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)


def ndarray_2d_from(profile, grid, *args, **kwargs):
    return _cartesian_grid_via_radial_from(
        grid=grid, radius=np.full(grid.shape[0], 2.0)
    )


def ndarray_2d_list_from(profile, grid, *args, **kwargs):
    return [ndarray_2d_from(profile, grid)]


def ndarray_2d_yx_from(profile, grid, *args, **kwargs):
    return 2.0 * grid


class MockTransformProfile:
    def __init__(self, centre=(1.0, 2.0)):
        self.centre = centre

    def transformed_to_reference_frame_grid_from(self, grid, xp=None, **kwargs):
        return grid - np.array(self.centre)

    def rotated_grid_from_reference_frame_from(self, grid, xp=None, **kwargs):
        return -1.0 * grid

    @decorators.transform
    def scalar_from(self, grid, xp=None, *args, **kwargs):
        return np.sum(grid, axis=1)

    @decorators.transform(rotate_back=True)
    def vector_from(self, grid, xp=None, *args, **kwargs):
        return 2.0 * grid


class MockGrid1DLikeObj:
    def __init__(self, centre=(0.0, 0.0), angle=0.0):
        self.centre = centre
        self.angle = angle


class MockGrid2DLikeObj:
    def __init__(self):
        self.centre = (0.0, 0.0)

        pass

    @over_sample
    @decorators.to_array
    def ndarray_1d_from(self, grid, *args, **kwargs):
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a 1D ndarray
        of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return np.ones(shape=grid.shape[0])

    @decorators.to_array
    def ndarray_1d_no_oversample_from(self, grid, *args, **kwargs):
        return np.ones(shape=grid.shape[0])

    @decorators.to_grid
    def ndarray_2d_from(self, grid, *args, **kwargs):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return np.multiply(2.0, grid.array)

    @decorators.to_grid
    def ndarray_2d_raw_from(self, grid, *args, **kwargs):
        return np.multiply(2.0, grid)

    @decorators.to_vector_yx
    def ndarray_yx_2d_from(self, grid, *args, **kwargs):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D ndarray
        of shape [total_masked_grid_pixels] which represents a vector field.

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return 2.0 * grid

    @decorators.to_array
    def ndarray_1d_list_from(self, grid, *args, **kwargs):
        """
        Mock function mimicking the behaviour of a class function which given an input 1D grid, returns a list of 1D
        ndarrays of shape [total_masked_grid_pixels].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [np.ones(shape=grid.shape[0]), 2.0 * np.ones(shape=grid.shape[0])]

    @decorators.to_grid
    def ndarray_2d_list_from(self, grid, *args, **kwargs):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a 2D list of
        ndarrays of shape [total_masked_grid_pixels, 2].

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [np.multiply(1.0, grid.array), np.multiply(2.0, grid.array)]

    @decorators.to_vector_yx
    def ndarray_yx_2d_list_from(self, grid, *args, **kwargs):
        """
        Mock function mimicking the behaviour of a class function which given an input grid, returns a list of 2D
        ndarrays of shape [total_masked_grid_pixels] which represents a vector field.

        Such functions are common in **PyAutoGalaxy** for light and mass profile objects.
        """
        return [np.multiply(1.0, grid.array), np.multiply(2.0, grid.array)]
