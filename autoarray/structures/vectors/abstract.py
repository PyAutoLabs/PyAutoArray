import logging
import numpy as np

from autoarray.structures.abstract_structure import Structure

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractVectorYX2D(Structure):
    def __array_finalize__(self, obj):
        if hasattr(obj, "mask"):
            self.mask = obj.mask

        if hasattr(obj, "grid"):
            self.grid = obj.grid

    @property
    def average_magnitude(self) -> float:
        """
        The average magnitude of the vector field, where averaging is performed on the (vector_y, vector_x) components.
        """
        return np.sqrt(np.mean(self.slim[:, 0]) ** 2 + np.mean(self.slim[:, 1]) ** 2)

    @property
    def average_phi(self) -> float:
        """
        The average angle of the vector field, where averaging is performed on the (vector_y, vector_x) components.
        """
        return (
            0.5
            * np.arctan2(np.mean(self.slim[:, 0]), np.mean(self.slim[:, 1]))
            * (180 / np.pi)
        )
