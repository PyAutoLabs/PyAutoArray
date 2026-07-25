import logging
import numpy as np
from typing import List, Tuple, Union

from autoarray.structures.vectors.abstract import AbstractVectorYX2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.arrays.irregular import ArrayIrregular

from autoarray import exc

logging.basicConfig()
logger = logging.getLogger(__name__)


class VectorYX2DIrregular(AbstractVectorYX2D):
    def __init__(
        self,
        values: Union[
            np.ndarray, List[np.ndarray], List[List], List[Tuple[float, float]]
        ],
        grid: Union[Grid2DIrregular, List],
    ):
        """
        A collection of (y,x) vectors which are located on a irregular 2D grid of (y,x) coordinates.

        The (y,x) vectors are stored as a 2D NumPy array of shape [total_vectors, 2]. This array can be mapped to a
        list of tuples structure.

        Calculations should use the NumPy array structure wherever possible for efficient calculations.

        The vectors input to this function can have any of the following forms (they will be converted to the 1D NumPy
        array structure and can be converted back using the object's properties):

        [[vector_0_y, vector_0_x], [vector_1_y, vector_1_x]]
        [(vector_0_y, vector_0_x), (vector_1_y, vector_1_x)]

        If your vector field lies on a 2D uniform grid of data the `VectorField` data structure should be used.

        Parameters
        ----------
        values
            The 2D (y,x) vectors on an irregular grid that represent the vector-field.
        grid
            The irregular grid of (y,x) coordinates where each vector is located.
        """
        if type(values) is list:
            values = np.asarray(values)

        self.grid = Grid2DIrregular(values=grid)

        super().__init__(values)

    def __array_finalize__(self, obj):
        if hasattr(obj, "grid"):
            self.grid = obj.grid

    @property
    def values(self) -> np.ndarray:
        """
        The raw underlying ndarray of (y, x) vector components, with shape [total_vectors, 2].
        """
        return self.array

    @property
    def slim(self) -> np.ndarray:
        """
        The vector-field in its 1D representation, an ndarray of shape [total_vectors, 2].
        """
        return self

    @property
    def native(self) -> np.ndarray:
        return self

    @property
    def in_list(self) -> List[Tuple]:
        """
        The vector-field in its list representation, as list of (y,x) vector tuples in a structure
        [(vector_0_y, vector_0_x), ...].
        """
        return [tuple(vector) for vector in self.slim]

    @property
    def magnitudes(self) -> ArrayIrregular:
        """
        Returns the magnitude of every vector which are computed as sqrt(y**2 + x**2).
        """
        return ArrayIrregular(values=np.sqrt(self[:, 0] ** 2.0 + self[:, 1] ** 2.0))

    @property
    def average_magnitude(self) -> float:
        """
        The average magnitude of the vector field, where averaging is performed on the (vector_y, vector_x) components.
        """
        return np.sqrt(np.mean(self[:, 0]) ** 2 + np.mean(self[:, 1]) ** 2)

    @property
    def average_phi(self) -> float:
        """
        The average angle of the vector field, where averaging is performed on the (vector_y, vector_x) components.
        """
        return (
            0.5 * np.arctan2(np.mean(self[:, 0]), np.mean(self[:, 1])) * (180 / np.pi)
        )

    def vectors_within_radius(
        self, radius: float, centre: Tuple[float, float] = (0.0, 0.0)
    ) -> "VectorYX2DIrregular":
        """
        Returns a new `VectorYX2DIrregular` object which has had all vectors outside of a circle of input radius
        around an  input (y,x) centre removed.

        Parameters
        ----------
        radius
            The radius of the circle outside of which vectors are removed.
        centre
            The centre of the circle outside of which vectors are removed.

        Returns
        -------
        VectorYX2DIrregular
            The vector field where all vectors outside of the input radius are removed.

        """
        squared_distances = self.grid.distances_to_coordinate_from(coordinate=centre)
        mask = squared_distances < radius

        if np.all(mask == False):
            raise exc.VectorYXException(
                "The input radius removed all vectors / points on the grid."
            )

        return VectorYX2DIrregular(
            values=np.array(self.array)[mask], grid=Grid2DIrregular(self.grid[mask])
        )

    def vectors_within_annulus(
        self,
        inner_radius: float,
        outer_radius: float,
        centre: Tuple[float, float] = (0.0, 0.0),
    ) -> "VectorYX2DIrregular":
        """
        Returns a new `VectorFieldIrregular` object which has had all vectors outside of a circle of input radius
        around an  input (y,x) centre removed.

        Parameters
        ----------
        radius
            The radius of the circle outside of which vectors are removed.
        centre
            The centre of the circle outside of which vectors are removed.

        Returns
        -------
        VectorFieldIrregular
            The vector field where all vectors outside of the input radius are removed.

        """
        squared_distances = self.grid.distances_to_coordinate_from(coordinate=centre)
        mask = (inner_radius < squared_distances) & (squared_distances < outer_radius)

        if np.all(mask == False):
            raise exc.VectorYXException(
                "The input radius removed all vectors / points on the grid."
            )

        return VectorYX2DIrregular(
            values=self[mask], grid=Grid2DIrregular(self.grid[mask])
        )
