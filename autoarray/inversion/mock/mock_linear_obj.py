import numpy as np

from autoarray.inversion.linear_obj.linear_obj import LinearObj


class MockLinearObj(LinearObj):
    def __init__(
        self,
        parameters=None,
        grid=None,
        mapping_matrix=None,
        operated_mapping_matrix_override=None,
        regularization=None,
    ):
        super().__init__(regularization=regularization)

        self.grid = grid
        self._parameters = parameters
        self._mapping_matrix = mapping_matrix
        self._operated_mapping_matrix_override = operated_mapping_matrix_override

    @property
    def params(self) -> int:
        return self._parameters

    @property
    def mapping_matrix(self) -> np.ndarray:
        return self._mapping_matrix

    @property
    def operated_mapping_matrix_override(self) -> np.ndarray:
        return self._operated_mapping_matrix_override
