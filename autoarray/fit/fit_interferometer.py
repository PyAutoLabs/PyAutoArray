import numpy as np

from autoarray.dataset.interferometer.dataset import Interferometer

from autoarray.dataset.dataset_model import DatasetModel
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.visibilities import Visibilities
from autoarray.fit.fit_dataset import FitDataset

from autoarray.fit import fit_util
from autoarray import type as ty


class FitInterferometer(FitDataset):
    def __init__(
        self,
        dataset: Interferometer,
        dataset_model: DatasetModel = None,
        use_mask_in_fit: bool = False,
        xp=np,
    ):
        """
        Class to fit a masked interferometer dataset.

        Parameters
        ----------
        dataset
            The interferometer dataset that is fitted, containing the observed visibilities and noise-map.
        dataset_model
            Attributes which allow for parts of a dataset to be treated as a model (e.g. the background sky level).
        use_mask_in_fit
            If `True`, masked data points are omitted from the fit. If `False` they are not (in most use cases the
            `dataset` will have been processed to remove masked points, for example the `slim` representation).

        Attributes
        -----------
        residual_map
            The residual-map of the fit (data - model_data).
        chi_squared_map
            The chi-squared-map of the fit ((data - model_data) / noise_map ) **2.0
        chi_squared
            The overall chi-squared of the model's fit to the dataset, summed over every data point.
        reduced_chi_squared
            The reduced chi-squared of the model's fit to simulate (chi_squared / number of data points), summed over
            every data point.
        noise_normalization
            The overall normalization term of the noise_map, summed over every data point.
        log_likelihood
            The overall log likelihood of the model's fit to the dataset, summed over every data point.
        """

        super().__init__(
            dataset=dataset,
            dataset_model=dataset_model,
            use_mask_in_fit=use_mask_in_fit,
            xp=xp,
        )

    @property
    def mask(self) -> np.ndarray:
        """
        The mask of the interferometer fit, returned as an all-`False` array matching the shape of the visibility data.

        Interferometer data is not spatially masked in the same way as imaging data — all visibility measurements
        are included in the fit — so this always returns an unmasked array.
        """
        return np.full(shape=self.data.shape, fill_value=False)

    @property
    def transformer(self) -> ty.Transformer:
        """
        The Fourier transformer used to map between image space and visibility (uv-plane) space.

        This is taken directly from the interferometer dataset and is used internally to compute the
        `dirty_*` image-space representations of the fit quantities.
        """
        return self.dataset.transformer

    @property
    def normalized_residual_map(self) -> np.ndarray:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        return fit_util.normalized_residual_map_complex_from(
            residual_map=self.residual_map,
            noise_map=self.noise_map,
        )

    @property
    def chi_squared_map(self) -> np.ndarray:
        """
        Returns the chi-squared-map between the residual-map and noise-map, where:

        Chi_Squared = ((Residuals) / (Noise)) ** 2.0 = ((Data - Model)**2.0)/(Variances)
        """
        return fit_util.chi_squared_map_complex_from(
            residual_map=self.residual_map,
            noise_map=self.noise_map,
        )

    @property
    def signal_to_noise_map(self) -> np.ndarray:
        """
        The signal-to-noise_map of the dataset and noise-map which are fitted."""
        signal_to_noise_map_real = self.data.real / self.noise_map.real

        signal_to_noise_map_real[signal_to_noise_map_real < 0] = 0.0
        signal_to_noise_map_imag = self.data.imag / self.noise_map.imag

        signal_to_noise_map_imag[signal_to_noise_map_imag < 0] = 0.0

        return signal_to_noise_map_real + 1.0j * signal_to_noise_map_imag

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.
        """
        return fit_util.chi_squared_complex_from(
            chi_squared_map=self.chi_squared_map.array,
        )

    @property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))
        """
        return fit_util.noise_normalization_complex_from(
            noise_map=self.noise_map.array,
        )

    @property
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

        For interferometer fits the chi-squared uses `inversion.fast_chi_squared`, which avoids computing the full
        residual visibilities by evaluating χ² directly from the reconstruction and inversion matrices.

        Returns `None` if no inversion is present, in which case `log_likelihood` is used as the figure of merit.
        """
        if self.inversion is not None:
            return fit_util.log_evidence_from(
                chi_squared=self.inversion.fast_chi_squared,
                regularization_term=self.inversion.regularization_term,
                log_curvature_regularization_term=self.inversion.log_det_curvature_reg_matrix_term,
                log_regularization_term=self.inversion.log_det_regularization_matrix_term,
                noise_normalization=self.noise_normalization,
            )

    @property
    def dirty_image(self) -> Array2D:
        """
        The dirty image of the observed visibility data, computed by applying the inverse Fourier transform to the
        data visibilities. This is the image-space representation of the observed data before any deconvolution.
        """
        return self.transformer.image_from(visibilities=self.data)

    @property
    def dirty_noise_map(self) -> Array2D:
        """
        The dirty noise-map, computed by applying the inverse Fourier transform to the noise-map visibilities.
        This gives an image-space representation of the noise level across the field of view.
        """
        return self.transformer.image_from(visibilities=self.noise_map)

    @property
    def dirty_signal_to_noise_map(self) -> Array2D:
        """
        The dirty signal-to-noise map, computed by applying the inverse Fourier transform to the signal-to-noise
        visibilities. This gives an image-space representation of the signal-to-noise ratio across the field of view.
        """
        return self.transformer.image_from(visibilities=self.signal_to_noise_map)

    @property
    def dirty_model_image(self) -> Array2D:
        """
        The dirty model image, computed by applying the inverse Fourier transform to the model data visibilities.
        This is the image-space representation of the model before any deconvolution.
        """
        return self.transformer.image_from(visibilities=self.model_data)

    @property
    def dirty_residual_map(self) -> Array2D:
        """
        The dirty residual map, computed by applying the inverse Fourier transform to the residual-map visibilities
        (data - model_data). This is the image-space representation of the residuals.
        """
        return self.transformer.image_from(visibilities=self.residual_map)

    @property
    def dirty_normalized_residual_map(self) -> Array2D:
        """
        The dirty normalized residual map, computed by applying the inverse Fourier transform to the
        normalized residual-map visibilities ((data - model_data) / noise_map).
        """
        return self.transformer.image_from(visibilities=self.normalized_residual_map)

    @property
    def dirty_chi_squared_map(self) -> Array2D:
        """
        The dirty chi-squared map, computed by applying the inverse Fourier transform to the chi-squared-map
        visibilities (((data - model_data) / noise_map) ** 2.0).
        """
        return self.transformer.image_from(visibilities=self.chi_squared_map)
