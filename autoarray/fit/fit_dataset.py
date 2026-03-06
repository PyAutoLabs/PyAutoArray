import warnings
from abc import ABC
from abc import abstractmethod
from typing import Dict, Optional

import numpy as np

from autoarray.dataset.grids import GridsInterface
from autoarray.dataset.dataset_model import DatasetModel
from autoarray.fit import fit_util
from autoarray.inversion.inversion.abstract import AbstractInversion
from autoarray.mask.mask_2d import Mask2D

from autoarray import type as ty


class AbstractFit(ABC):
    @property
    @abstractmethod
    def data(self) -> ty.DataLike:
        """
        Overwrite this method to returns the data of the dataset.
        """

    @property
    @abstractmethod
    def noise_map(self) -> ty.DataLike:
        """
        Overwrite this method to returns the noise-map of the dataset.
        """

    @property
    @abstractmethod
    def model_data(self) -> ty.DataLike:
        """
        Overwrite this method so it returns the model-data which is fitted to the input data.
        """

    @property
    def signal_to_noise_map(self) -> ty.DataLike:
        """
        The signal-to-noise_map of the dataset and noise-map which are fitted.
        """
        warnings.filterwarnings("ignore")

        signal_to_noise_map = self.data / self.noise_map
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @property
    def residual_map(self) -> ty.DataLike:
        """
        Returns the residual-map between the masked dataset and model data, where:

        Residuals = (Data - Model_Data).
        """
        return fit_util.residual_map_from(data=self.data, model_data=self.model_data)

    @property
    def normalized_residual_map(self) -> ty.DataLike:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        return fit_util.normalized_residual_map_from(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def chi_squared_map(self) -> ty.DataLike:
        """
        Returns the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """
        return fit_util.chi_squared_map_from(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return fit_util.chi_squared_from(
            chi_squared_map=self.chi_squared_map.array, xp=self._xp
        )

    @property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """
        return fit_util.noise_normalization_from(
            noise_map=self.noise_map.array, xp=self._xp
        )

    @property
    def log_likelihood(self) -> float:
        """
        Returns the log likelihood of each model data point's fit to the dataset, where:

        Log Likelihood = -0.5*[Chi_Squared_Term + Noise_Term] (see functions above for these definitions)
        """
        return fit_util.log_likelihood_from(
            chi_squared=self.chi_squared, noise_normalization=self.noise_normalization
        )


class FitDataset(AbstractFit):
    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        dataset,
        use_mask_in_fit: bool = False,
        dataset_model: DatasetModel = None,
        xp=np,
    ):
        """Class to fit a masked dataset where the dataset's data structures are any dimension.

        Parameters
        ----------
        dataset
            The masked dataset (data, mask, noise-map, etc.) that is fitted.
        use_mask_in_fit
            If `True`, masked data points are omitted from the fit. If `False` they are not (in most use cases the
            `dataset` will have been processed to remove masked points, for example the `slim` representation).
        dataset_model
            Attributes which allow for parts of a dataset to be treated as a model (e.g. the background sky level).

        Attributes
        -----------
        residual_map
            The residual-map of the fit (data - model_data).
        chi_squared_map
            The chi-squared-map of the fit ((data - model_data) / noise_maps ) **2.0
        chi_squared
            The overall chi-squared of the model's fit to the dataset, summed over every data point.
        reduced_chi_squared
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of data points), summed over \
            every data point.
        noise_normalization
            The overall normalization term of the noise_map, summed over every data point.
        log_likelihood
            The overall log likelihood of the model's fit to the dataset, summed over evey data point.
        """
        self.dataset = dataset
        self.use_mask_in_fit = use_mask_in_fit
        self.dataset_model = dataset_model or DatasetModel()

        # Ensures preloaded in memory so repeated calls are fast

        self.dataset.grids.lp
        self.dataset.grids.pixelization
        self.dataset.grids.blurring
        self.dataset.grids.border_relocator

        self.use_jax = xp is not np

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    @property
    def mask(self) -> Mask2D:
        return self.dataset.mask

    @property
    def grids(self) -> GridsInterface:

        """
        Qiuhan Updated
        """

        def subtracted_and_rotated_from(grid, offset, angle):
            if grid is None:
                return None

            return grid.subtracted_and_rotated_from(offset=offset, angle=angle, xp=self._xp)

        lp = subtracted_and_rotated_from(
            grid=self.dataset.grids.lp, offset=self.dataset_model.grid_offset, angle=self.dataset_model.grid_rotation_angle
        )
        pixelization = subtracted_and_rotated_from(
            grid=self.dataset.grids.pixelization, offset=self.dataset_model.grid_offset, angle=self.dataset_model.grid_rotation_angle 
        )
        blurring = subtracted_and_rotated_from(
            grid=self.dataset.grids.blurring, offset=self.dataset_model.grid_offset, angle=self.dataset_model.grid_rotation_angle
        )

        return GridsInterface(
            lp=lp,
            pixelization=pixelization,
            blurring=blurring,
            border_relocator=self.dataset.grids.border_relocator,
        )

    @property
    def data(self) -> ty.DataLike:
        return self.dataset.data

    @property
    def noise_map(self) -> ty.DataLike:
        return self.dataset.noise_map

    @property
    def residual_map(self) -> ty.DataLike:
        """
        Returns the residual-map between the masked dataset and model data, where:

        Residuals = (Data - Model_Data).
        """

        if self.use_mask_in_fit:
            return fit_util.residual_map_with_mask_from(
                data=self.data, model_data=self.model_data, mask=self.mask, xp=self._xp
            )
        return super().residual_map

    @property
    def normalized_residual_map(self) -> ty.DataLike:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        if self.use_mask_in_fit:
            return fit_util.normalized_residual_map_with_mask_from(
                residual_map=self.residual_map,
                noise_map=self.noise_map,
                mask=self.mask,
                xp=self._xp,
            )
        return super().normalized_residual_map

    @property
    def chi_squared_map(self) -> ty.DataLike:
        """
        Returns the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """
        if self.use_mask_in_fit:
            return fit_util.chi_squared_map_with_mask_from(
                residual_map=self.residual_map,
                noise_map=self.noise_map,
                mask=self.mask,
                xp=self._xp,
            )
        return super().chi_squared_map

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.

        If the dataset includes a noise covariance matrix, this is used instead to account for covariance in the
        goodness-of-fit.
        """

        if self.dataset.noise_covariance_matrix is not None:
            return fit_util.chi_squared_with_noise_covariance_from(
                residual_map=self.residual_map,
                noise_covariance_matrix_inv=self.dataset.noise_covariance_matrix_inv,
            )

        if self.use_mask_in_fit:
            return fit_util.chi_squared_with_mask_from(
                chi_squared_map=self.chi_squared_map, mask=self.mask, xp=self._xp
            )
        return super().chi_squared

    @property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """
        if self.use_mask_in_fit:
            return fit_util.noise_normalization_with_mask_from(
                noise_map=self.noise_map, mask=self.mask, xp=self._xp
            )
        return super().noise_normalization

    @property
    def log_likelihood_with_regularization(self) -> float:
        """
        Returns the log likelihood of an inversion's fit to the dataset, including a regularization term which \
        comes from an inversion:

        Log Likelihood = -0.5*[Chi_Squared_Term + Regularization_Term + Noise_Term] (see functions above for these definitions)
        """
        if self.inversion is not None:
            return fit_util.log_likelihood_with_regularization_from(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def log_evidence(self) -> float:
        """
        Returns the log evidence of the inversion's fit to a dataset, where the log evidence includes a number of terms
        which quantify the complexity of an inversion's reconstruction (see the `Inversion` module):

        Log Evidence = -0.5*[Chi_Squared_Term + Regularization_Term + Log(Covariance_Regularization_Term) -
                           Log(Regularization_Matrix_Term) + Noise_Term]

        Parameters
        ----------
        chi_squared
            The chi-squared term of the inversion's fit to the data.
        regularization_term
            The regularization term of the inversion, which is the sum of the difference between reconstructed \
            flux of every pixel multiplied by the regularization coefficient.
        log_curvature_regularization_term
            The log of the determinant of the sum of the curvature and regularization matrices.
        log_regularization_term
            The log of the determinant o the regularization matrix.
        noise_normalization
            The normalization noise_map-term for the data's noise-map.
        """
        if self.inversion is not None:
            return fit_util.log_evidence_from(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                log_curvature_regularization_term=self.inversion.log_det_curvature_reg_matrix_term,
                log_regularization_term=self.inversion.log_det_regularization_matrix_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def figure_of_merit(self) -> float:
        if self.inversion is not None:
            return self.log_evidence

        try:
            return self.log_likelihood.array
        except AttributeError:
            return self.log_likelihood

    @property
    def residual_flux_fraction_map(self) -> ty.DataLike:
        """
        Returns the residual flux fraction map, which shows the fraction of signal in each pixel that is not fitted
        by the model, therefore where:

        Residual_Flux_Fraction = ((Residuals) / (Data)) = ((Data - Model))/(Data)

        This quantity is not used for computing the log likelihood, but is available for plotting and inspection.

        It does not use the noise-map in its calculation, and therefore the residual flux fraction should only be
        reliably interpreted in high signal-to-noise regions of a dataset.
        """
        if self.use_mask_in_fit:
            return fit_util.chi_squared_map_with_mask_from(
                residual_map=self.residual_map, noise_map=self.noise_map, mask=self.mask
            )
        return super().chi_squared_map

    @property
    def inversion(self) -> Optional[AbstractInversion]:
        """
        Overwrite this method so it returns the inversion used to fit the dataset.
        """
        return None

    @property
    def reduced_chi_squared(self) -> float:
        return self.chi_squared / int(np.size(self.mask) - np.sum(self.mask))
