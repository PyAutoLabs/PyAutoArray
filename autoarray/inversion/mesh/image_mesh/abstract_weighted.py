import numpy as np

from autoarray.inversion.mesh.image_mesh.abstract import AbstractImageMesh


class AbstractImageMeshWeighted(AbstractImageMesh):
    def __init__(
        self,
        pixels=10.0,
        weight_floor=0.0,
        weight_power=0.0,
    ):
        """
        An abstract image mesh, which is used by pixelizations to determine the (y,x) mesh coordinates from an adapt
        image.

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
        """

        super().__init__()

        self.pixels = pixels
        self.weight_floor = weight_floor
        self.weight_power = weight_power

    @property
    def uses_adapt_images(self) -> bool:
        return True

    def weight_map_from(self, adapt_data: np.ndarray):
        """
        Returns the weight-map used by the image-mesh to compute the mesh pixel centres.

        This is computed from an input adapt data, which is an image representing the data which the KMeans
        clustering algorithm is applied too. This could be the image data itself, or a model fit which
        only has certain features.

        The ``weight_floor`` and ``weight_power`` attributes of the class are used to scale the weight map, which
        gives the model flexibility in how it adapts the pixelization to the image data.

        Parameters
        ----------
        adapt_data
            A image which represents one or more components in the masked 2D data in the image-plane.

        Returns
        -------
        The weight map which is used to adapt the Delaunay pixels in the image-plane to components in the data.
        """

        max_value = np.max(adapt_data)

        if max_value <= 0.0:
            # A blank adapt image (all-zero, or with no positive signal) carries
            # no structure to adapt the mesh to. Normalising by ``np.max`` would
            # divide by zero (or a non-positive peak), producing NaN/negative
            # weights that propagate into NaN mesh coordinates and crash the
            # downstream Delaunay/Voronoi triangulation ("Points cannot contain
            # NaN"). Fall back to a uniform weight map so the mesh degrades to
            # uniform sampling instead of failing.
            return np.ones_like(adapt_data, dtype=float)

        weight_map = np.abs(adapt_data) / max_value
        weight_map = weight_map**self.weight_power

        weight_map[weight_map < self.weight_floor] = self.weight_floor

        return weight_map
