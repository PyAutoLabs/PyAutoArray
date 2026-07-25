import numpy as np
from typing import Optional
import sys
import warnings

from autoarray.mask.mask_2d import Mask2D

from autoarray.inversion.mesh.image_mesh.abstract_weighted import (
    AbstractImageMeshWeighted,
)
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.settings import Settings

from autoarray import exc


class KMeans(AbstractImageMeshWeighted):
    def __init__(
        self,
        pixels=10.0,
        weight_floor=0.0,
        weight_power=0.0,
    ):
        """
        Computes an image-mesh by running a weighted KMeans clustering algorithm.

        This requires an adapt-image, which is the image that the KMeans algorithm adapts to in order to compute the
        image mesh. This could simply be the image itself, or a model fit to the image which removes certain
        features or noise.

        For example, using the adapt image, the image mesh is computed as follows:

        1) Convert the adapt image to a weight map, which is a 2D array of weight values.

        2) Run the KMeans algorithm on the weight map, such that the image mesh pixels cluster around the weight map
        values with higher values.

        Parameters
        ----------
        total_pixels
            The total number of pixels in the image mesh and input into the KMeans algortihm.
        weight_power
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
        Returns an image mesh by running a KMeans clustering algorithm on the weight map.

        See the `__init__` docstring for a full description of how this is performed.

        Parameters
        ----------
        grid
            The grid of (y,x) coordinates of the image data the pixelization fits, which the KMeans algorithm
            adapts to.
        adapt_data
            The weights defining the regions of the image the KMeans algorithm adapts to.

        Returns
        -------

        """

        if self.pixels > mask.shape_slim:
            print(
                """
                The number of pixels passed to the KMeans object exceeds the number of image-pixels in the mask of
                the data being fitted. This is not allowed by the KMenas algorithm and will cause an error.
                
                To fix this, you should reduce the number of pixels in the KMeans object to be less than the number
                of image-pixels in the mask of the data being fitted.
                
                If you are performing model-fitting, you should update the priors to have an upper limit which does
                not exceed the number of image-pixels in the mask of the data being fitted.
                
                For adaptive fitting, the KMeans object has been superseeded by the Hilbert object, which does not
                have this limitation and performs better in general. You should therefore consider using the Hilbert
                object instead.
                """
            )

            sys.exit()

        warnings.filterwarnings("ignore")

        weight_map = self.weight_map_from(adapt_data=adapt_data)

        from sklearn.cluster import KMeans as ScipyKMeans

        kmeans = ScipyKMeans(
            n_clusters=int(self.pixels),
            random_state=1,
            n_init=1,
            max_iter=5,
        )

        grid = mask.derive_grid.unmasked.array

        try:
            kmeans = kmeans.fit(X=grid, sample_weight=weight_map)
        except ValueError or OverflowError:
            raise exc.InversionException()

        return Grid2DIrregular(
            values=kmeans.cluster_centers_,
        )
