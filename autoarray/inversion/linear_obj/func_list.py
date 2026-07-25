import numpy as np
from typing import Optional

from autonerves import cached_property

from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.linear_obj.neighbors import Neighbors
from autoarray.inversion.linear_obj.unique_mappings import UniqueMappings
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.settings import Settings
from autoarray.type import Grid1D2DLike


class AbstractLinearObjFuncList(LinearObj):
    def __init__(
        self,
        grid: Grid1D2DLike,
        regularization: Optional[AbstractRegularization],
        settings=Settings(),
        xp=np,
    ):
        """
        A linear object which reconstructs a dataset based on mapping between the data points of that dataset and
        the parameters of the linear object.

        This linear object's parameters are one or more analytic functions, the solution of which are solved for
        linearly via an inversion.

        By overwriting the `mapping_matrix` function with a method that fills in its value with the solution of the
        analytic function, this is then passed through the `inversion` package to perform the linear inversion. The
        API is identical to `Mapper` objects such that linear functions can easily be combined with mappers.

        For example, in `PyAutoGalaxy` and `PyAutoLens` the light of galaxies is represented using `LightProfile`
        objects, which describe the surface brightness of a galaxy as a function. This function can either be assigned
        an overall intensity (e.g. the normalization) which describes how bright it is. Using the `LinearObjFuncList`
        the intensity can be solved for linearly instead.

        Parameters
        ----------
        grid
            The grid of data points representing the data that is fitted and therefore where the analytic function
            is evaluated.
        regularization
            The regularization scheme which may be applied to this linear object in order to smooth its solution.
        """

        super().__init__(regularization=regularization, xp=xp)

        self.grid = grid
        self.settings = settings

    @cached_property
    def neighbors(self) -> Neighbors:
        """
        An object describing how the different parameters in the linear object neighbor one another, which is used
        to apply smoothing to neighboring parameters via regularization.

        For a `AbstractLinearObjFuncList` this object may describe how certain analytic functions reconstruct nearby
        components next to one another, which should therefore be regularized with one another.

        Returns
        -------
        An object describing how the parameters of the linear object neighbor one another.
        """
        neighbors_sizes = 2.0 * np.ones(shape=(self.params))

        neighbors_sizes[0] -= 1
        neighbors_sizes[-1] -= 1

        neighbors = -1 * np.ones(shape=(self.params, 2))

        for pixel_index in range(self.params):
            neighbors[pixel_index, 0] = pixel_index - 1
            neighbors[pixel_index, 1] = pixel_index + 1

        neighbors[0, 0] = 1
        neighbors[0, 1] = -1
        neighbors[-1, 1] = -1

        return Neighbors(
            arr=neighbors.astype("int"), sizes=neighbors_sizes.astype("int")
        )

    @cached_property
    def unique_mappings(self) -> UniqueMappings:
        """
        Returns the unique mappings of every unmasked data pixel's (e.g. `grid_slim`) sub-pixels (e.g. `grid_sub_slim`)
        to their corresponding pixelization pixels (e.g. `mesh_grid`).

        To perform an `Inversion` efficiently the linear algebra can bypass the calculation of a `mapping_matrix` and
        instead use the w-tilde formalism, which requires these unique mappings for efficient computation. For
        convenience, these mappings and associated metadata are packaged into the class `UniqueMappings`.

        For a `LinearObjFuncList` every data pixel's group of sub-pixels maps directly to the linear function.
        """
        sub_size = np.max(self.grid.over_sample_size)

        # TODO : This shape slim is prob unreliable and needs to be divided by sub_size**2

        shape_slim = self.grid.mask.shape_slim

        data_to_pix_unique = -1.0 * np.ones(shape=(shape_slim, sub_size**2)).astype(
            "int"
        )
        data_weights = np.zeros(shape=(shape_slim, sub_size**2))
        pix_lengths = np.ones(shape=shape_slim).astype("int")

        data_to_pix_unique[:, 0] = 0
        data_weights[:, 0] = 1.0

        return UniqueMappings(
            data_to_pix_unique=data_to_pix_unique,
            data_weights=data_weights,
            pix_lengths=pix_lengths,
        )
