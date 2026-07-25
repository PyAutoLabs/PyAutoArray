import numpy as np
from typing import List

from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.settings import Settings


class MockInversion(AbstractInversion):
    def __init__(
        self,
        data=None,
        noise_map=None,
        linear_obj_list=None,
        operated_mapping_matrix=None,
        data_vector=None,
        curvature_matrix=None,
        data_vector_mapper=None,
        curvature_matrix_mapper_diag=None,
        mapper_operated_mapping_matrix_dict=None,
        regularization_matrix=None,
        curvature_reg_matrix=None,
        reconstruction: np.ndarray = None,
        reconstruction_dict: List[np.ndarray] = None,
        mapped_reconstructed_data_dict=None,
        mapped_reconstructed_operated_data_dict=None,
        reconstruction_noise_map: np.ndarray = None,
        reconstruction_noise_map_dict: List[np.ndarray] = None,
        regularization_term=None,
        log_det_curvature_reg_matrix_term=None,
        log_det_regularization_matrix_term=None,
        fast_chi_squared: float = None,
        settings: Settings = None,
    ):
        dataset = DatasetInterface(
            data=data,
            noise_map=noise_map,
        )

        settings = settings or Settings()

        super().__init__(
            dataset=dataset,
            linear_obj_list=linear_obj_list or [],
            settings=settings,
        )

        self._operated_mapping_matrix = operated_mapping_matrix
        self._data_vector = data_vector
        self._regularization_matrix = regularization_matrix
        self._curvature_matrix = curvature_matrix
        self.__data_vector_mapper = data_vector_mapper
        self.__curvature_matrix_mapper_diag = curvature_matrix_mapper_diag
        self._mapper_operated_mapping_matrix_dict = mapper_operated_mapping_matrix_dict
        self._curvature_reg_matrix = curvature_reg_matrix
        self._reconstruction = reconstruction
        self._reconstruction_dict = reconstruction_dict

        self._mapped_reconstructed_data_dict = mapped_reconstructed_data_dict
        self._mapped_reconstructed_operated_data_dict = (
            mapped_reconstructed_operated_data_dict
        )

        self._reconstruction_noise_map = reconstruction_noise_map
        self._reconstruction_noise_map_dict = reconstruction_noise_map_dict

        self._regularization_term = regularization_term
        self._log_det_curvature_reg_matrix_term = log_det_curvature_reg_matrix_term
        self._log_det_regularization_matrix_term = log_det_regularization_matrix_term
        self._fast_chi_squared = fast_chi_squared

    @property
    def operated_mapping_matrix(self) -> np.ndarray:
        if self._operated_mapping_matrix is None:
            return super().operated_mapping_matrix

        return self._operated_mapping_matrix

    @property
    def data_vector(self) -> np.ndarray:
        if self._data_vector is None:
            return super().data_vector
        return self._data_vector

    @property
    def regularization_matrix(self):
        if self._regularization_matrix is None:
            return super().regularization_matrix

        return self._regularization_matrix

    @property
    def curvature_matrix(self):
        if self._curvature_matrix is None:
            return super().curvature_matrix

        return self._curvature_matrix

    @property
    def _data_vector_mapper(self):
        if self.__data_vector_mapper is None:
            return super()._data_vector_mapper

        return self.__data_vector_mapper

    @property
    def _curvature_matrix_mapper_diag(self):
        if self.__curvature_matrix_mapper_diag is None:
            return super()._curvature_matrix_mapper_diag

        return self.__curvature_matrix_mapper_diag

    @property
    def mapper_operated_mapping_matrix_dict(self):
        if self._mapper_operated_mapping_matrix_dict is None:
            return super().mapper_operated_mapping_matrix_dict

        return self._mapper_operated_mapping_matrix_dict

    @property
    def curvature_reg_matrix(self):
        return self._curvature_reg_matrix

    @property
    def reconstruction(self):
        if self._reconstruction is None:
            return super().reconstruction
        return self._reconstruction

    @property
    def reconstruction_dict(self):
        if self._reconstruction_dict is None:
            return super().reconstruction_dict
        return self._reconstruction_dict

    @property
    def mapped_reconstructed_data_dict(self):
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane and
        reconstruct the image data.

        This uses the unique mappings of every source pixel to image pixels, which is a quantity that is already
        computed when using the w-tilde formalism.

        Returns
        -------
        Array2D
            The reconstructed image data which the inversion fits.
        """

        if self._mapped_reconstructed_data_dict is None:
            return super().mapped_reconstructed_data_dict

        return self._mapped_reconstructed_data_dict

    @property
    def mapped_reconstructed_operated_data_dict(self):
        """
        Using the reconstructed source pixel fluxes we map each source pixel flux back to the image plane and
        reconstruct the image data.

        This uses the unique mappings of every source pixel to image pixels, which is a quantity that is already
        computed when using the w-tilde formalism.

        Returns
        -------
        Array2D
            The reconstructed image data which the inversion fits.
        """

        if self._mapped_reconstructed_operated_data_dict is None:
            return super().mapped_reconstructed_operated_data_dict

        return self._mapped_reconstructed_operated_data_dict

    @property
    def reconstruction_noise_map(self):
        if self._reconstruction_noise_map is None:
            return super().reconstruction_noise_map
        return self._reconstruction_noise_map

    @property
    def reconstruction_noise_map_dict(self):
        if self._reconstruction_noise_map_dict is None:
            return super().reconstruction_noise_map_dict
        return self._reconstruction_noise_map_dict

    @property
    def regularization_term(self):
        if self._regularization_term is None:
            return super().regularization_term

        return self._regularization_term

    @property
    def log_det_curvature_reg_matrix_term(self):
        if self._log_det_curvature_reg_matrix_term is None:
            return super().log_det_curvature_reg_matrix_term

        return self._log_det_curvature_reg_matrix_term

    @property
    def log_det_regularization_matrix_term(self):
        if self._log_det_regularization_matrix_term is None:
            return super().log_det_regularization_matrix_term

        return self._log_det_regularization_matrix_term

    @property
    def fast_chi_squared(self) -> float:
        if self._fast_chi_squared is None:
            return super().fast_chi_squared

        return self._fast_chi_squared
