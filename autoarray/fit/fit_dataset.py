import functools
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

    @functools.cached_property
    def signal_to_noise_map(self) -> ty.DataLike:
        """
        The signal-to-noise_map of the dataset and noise-map which are fitted.
        """
        warnings.filterwarnings("ignore")

        signal_to_noise_map = self.data / self.noise_map
        signal_to_noise_map[signal_to_noise_map < 0] = 0
        return signal_to_noise_map

    @functools.cached_property
    def residual_map(self) -> ty.DataLike:
        """
        Returns the residual-map between the masked dataset and model data, where:

        Residuals = (Data - Model_Data).
        """
        return fit_util.residual_map_from(data=self.data, model_data=self.model_data)

    @functools.cached_property
    def normalized_residual_map(self) -> ty.DataLike:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        return fit_util.normalized_residual_map_from(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @functools.cached_property
    def chi_squared_map(self) -> ty.DataLike:
        """
        Returns the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """
        return fit_util.chi_squared_map_from(
            residual_map=self.residual_map, noise_map=self.noise_map
        )

    @functools.cached_property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return fit_util.chi_squared_from(
            chi_squared_map=self.chi_squared_map.array, xp=self._xp
        )

    @functools.cached_property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """
        return fit_util.noise_normalization_from(
            noise_map=self.noise_map.array, xp=self._xp
        )

    @functools.cached_property
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
            The overall log likelihood of the model's fit to the dataset, summed over every data point.
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
        """
        Returns the array module in use: `numpy` if JAX is disabled or `jax.numpy` if JAX is enabled.

        This is controlled by the `use_jax` flag set during initialisation and is the single point of control
        for switching between NumPy and JAX computation paths throughout the fit.
        """
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    @property
    def mask(self) -> Mask2D:
        """
        The 2D mask of the dataset being fitted, where `False` entries are unmasked and included in the fit
        and `True` entries are masked and excluded.
        """
        return self.dataset.mask

    @functools.cached_property
    def grids(self) -> GridsInterface:
        """
        The grids of (y,x) coordinates associated with the dataset, adjusted by any `grid_offset` and
        `grid_rotation_angle` specified in the `dataset_model`. Each grid (`lp`, `pixelization`, `blurring`)
        has the offset subtracted from it and is then rotated counter-clockwise by `grid_rotation_angle`
        about the offset point before being returned.
        """

        offset = self.dataset_model.grid_offset
        angle = self.dataset_model.grid_rotation_angle

        def shift_and_rotate(grid):
            if grid is None:
                return None
            return grid.subtracted_and_rotated_from(offset=offset, angle=angle, xp=self._xp)

        lp = shift_and_rotate(self.dataset.grids.lp)
        pixelization = shift_and_rotate(self.dataset.grids.pixelization)
        blurring = shift_and_rotate(self.dataset.grids.blurring)

        return GridsInterface(
            lp=lp,
            pixelization=pixelization,
            blurring=blurring,
            border_relocator=self.dataset.grids.border_relocator,
        )

    @property
    def data(self) -> ty.DataLike:
        """
        The data of the dataset being fitted.
        """
        return self.dataset.data

    @property
    def noise_map(self) -> ty.DataLike:
        """
        The noise-map of the dataset being fitted, representing the RMS noise in each pixel.
        """
        return self.dataset.noise_map

    @functools.cached_property
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

    @functools.cached_property
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

    @functools.cached_property
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

    @functools.cached_property
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

    @functools.cached_property
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

    @functools.cached_property
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

    @functools.cached_property
    def log_evidence(self) -> float:
        """
        Returns the log Bayesian evidence of the inversion's fit to a dataset, which extends the log likelihood by
        including penalty terms that quantify the complexity of the inversion's reconstruction:

        Log Evidence = -0.5 * [χ² + s^T H s + ln(det(F + H)) - ln(det(H)) + Σ ln(2π σ²)]

        where:
        - χ² is the chi-squared goodness-of-fit term
        - s^T H s is the regularization term (smoothness penalty on the reconstructed source pixels)
        - ln(det(F + H)) penalizes overly complex reconstructions (log determinant of the curvature + regularization matrix)
        - ln(det(H)) normalizes the regularization matrix complexity (log determinant of the regularization matrix)
        - Σ ln(2π σ²) is the noise normalization term

        This is described in Warren & Dye 2003 (https://arxiv.org/pdf/astro-ph/0302587.pdf) and
        Nightingale & Dye 2015 (https://arxiv.org/abs/1708.07377).

        Returns `None` if no inversion is present, in which case `log_likelihood` is used as the figure of merit.
        """
        if self.inversion is not None:
            return fit_util.log_evidence_from(
                chi_squared=self.chi_squared,
                regularization_term=self.inversion.regularization_term,
                log_curvature_regularization_term=self.inversion.log_det_curvature_reg_matrix_term,
                log_regularization_term=self.inversion.log_det_regularization_matrix_term,
                noise_normalization=self.noise_normalization,
            )

    @functools.cached_property
    def figure_of_merit(self) -> float:
        """
        The overall goodness-of-fit of the model to the dataset.

        If the fit uses an inversion, this is the `log_evidence`; otherwise it is the `log_likelihood`.
        """
        if self.inversion is not None:
            return self.log_evidence

        try:
            return self.log_likelihood.array
        except AttributeError:
            return self.log_likelihood

    @functools.cached_property
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

    @functools.cached_property
    def reduced_chi_squared(self) -> float:
        """
        The reduced chi-squared of the model's fit to the dataset, defined as:

        Reduced_Chi_Squared = Chi_Squared / N_unmasked

        where `N_unmasked` is the number of unmasked (i.e. `False`) pixels in the mask.
        """
        return self.chi_squared / int(np.size(self.mask) - np.sum(self.mask))
