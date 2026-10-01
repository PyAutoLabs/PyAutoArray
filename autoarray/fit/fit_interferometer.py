import functools
import numpy as np
from typing import Optional

from autoarray.dataset.interferometer.dataset import Interferometer

from autoarray.dataset.dataset_model import DatasetModel
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.visibilities import Visibilities
from autoarray.fit.fit_dataset import FitDataset

from autoarray.fit import fit_util
from autoarray import exc
from autoarray import type as ty


def dirty_model_image_natural_from(dataset, image) -> Array2D:
    """
    Returns the naturally weighted, normalised dirty image of the model visibilities of a real-space
    `image`, `Re(F^H W F m) / sum(w)`, computed without a transformer or any visibility-sized array.

    `W~ = Re(F^H W F)` is the operator cached on the dataset's `sparse_operator` (the one the sparse
    inversion uses for its curvature matrix), so `W~ m / sum(w)` is one FFT convolution on the real-space
    grid. It is the model counterpart of `Interferometer.dirty_image_natural` (the natural dirty image of
    the data, `Re(F^H W d) / sum(w)`): their difference is the natural dirty residual map. It is available
    on both dataset types -- an array-free one built by `from_stream` / `from_sparse_terms` and an in-memory
    one after `apply_sparse_operator()` -- and is how the visualizers draw a model on an array-free dataset.

    Parameters
    ----------
    dataset
        The `Interferometer` dataset, which must carry a `sparse_operator`.
    image
        The model image `m` on the slim masked real-space grid of the dataset's `real_space_mask`.
    """
    sparse_operator = getattr(dataset, "sparse_operator", None)

    if sparse_operator is None:
        raise exc.DatasetException(
            "The natural dirty model image `W~ m / sum(w)` needs the dataset's `sparse_operator`; call "
            "`apply_sparse_operator()` on the dataset (an array-free dataset built by from_stream / "
            "from_sparse_terms always carries one)."
        )

    sparse_terms = getattr(dataset, "sparse_terms", None)

    if sparse_terms is not None:
        sum_weights = float(sparse_terms.sum_weights)
    else:
        sum_weights = float(np.sum(dataset.noise_map.array.real**-2.0))

    image = np.asarray(getattr(image, "array", image), dtype=np.float64)

    operated_image = sparse_operator.operated_matrix_slim_from(
        matrix_slim=image[:, None],
        extent_index_for_masked_pixel=dataset.real_space_mask.extent_index_for_masked_pixel,
        xp=np,
    )[:, 0]

    return Array2D(
        values=np.asarray(operated_image) / sum_weights, mask=dataset.real_space_mask
    )


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

    def _require(self, quantity: str, *names: str):
        """
        Raise a typed `exc.DatasetException` when this fit's dataset is array-free (an
        `Interferometer` built by `from_stream` / `from_sparse_terms`) and one of the named
        inputs (`data`, `noise_map`, `transformer`) the requested `quantity` needs is `None`.

        The `log_evidence` / `figure_of_merit` of a sparse inversion fit never reaches here:
        it reads `inversion.fast_chi_squared` and the sparse operator's cached
        `noise_normalization`, neither of which touches the visibility arrays.
        """
        missing = [name for name in names if getattr(self, name) is None]

        if missing:
            raise exc.DatasetException(
                f"This FitInterferometer's dataset is array-free (built by from_stream / "
                f"from_sparse_terms) and has no {' / '.join(missing)}; `{quantity}` is "
                f"unavailable. Only the `log_evidence` / `figure_of_merit` of a sparse "
                f"inversion can be computed; use the in-memory constructor if you need it."
            )

    @property
    def mask(self) -> np.ndarray:
        """
        The mask of the interferometer fit, returned as an all-`False` array matching the shape of the visibility data.

        Interferometer data is not spatially masked in the same way as imaging data — all visibility measurements
        are included in the fit — so this always returns an unmasked array.
        """
        self._require("mask", "data")
        return np.full(shape=self.data.shape, fill_value=False)

    @property
    def transformer(self) -> ty.Transformer:
        """
        The Fourier transformer used to map between image space and visibility (uv-plane) space.

        This is taken directly from the interferometer dataset and is used internally to compute the
        `dirty_*` image-space representations of the fit quantities.
        """
        transformer = self.dataset.transformer

        if transformer is None:
            raise exc.DatasetException(
                "This FitInterferometer's dataset is array-free (built by from_stream / "
                "from_sparse_terms) and has no transformer; `transformer` is unavailable. "
                "Use the in-memory constructor if you need it."
            )

        return transformer

    @functools.cached_property
    def residual_map(self):
        """
        Returns the residual-map between the visibility data and model data (data - model_data).

        Raises an `exc.DatasetException` on an array-free dataset, which has no data.
        """
        self._require("residual_map", "data")
        return super().residual_map

    @property
    def normalized_residual_map(self) -> np.ndarray:
        """
        Returns the normalized residual-map between the masked dataset and model data, where:

        Normalized_Residual = (Data - Model_Data) / Noise
        """
        self._require("normalized_residual_map", "data", "noise_map")
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
        self._require("chi_squared_map", "data", "noise_map")
        return fit_util.chi_squared_map_complex_from(
            residual_map=self.residual_map,
            noise_map=self.noise_map,
        )

    @property
    def signal_to_noise_map(self) -> np.ndarray:
        """
        The signal-to-noise_map of the dataset and noise-map which are fitted."""
        self._require("signal_to_noise_map", "data", "noise_map")
        signal_to_noise_map_real = self.data.real / self.noise_map.real

        signal_to_noise_map_real[signal_to_noise_map_real < 0] = 0.0
        signal_to_noise_map_imag = self.data.imag / self.noise_map.imag

        signal_to_noise_map_imag[signal_to_noise_map_imag < 0] = 0.0

        return signal_to_noise_map_real + 1.0j * signal_to_noise_map_imag

    @property
    def sparse_chi_squared(self) -> Optional[float]:
        """
        The chi-squared of this fit computed from the dataset's `sparse_operator` without any visibility-sized
        array, used by `chi_squared` when the dataset is array-free (built by `from_stream` /
        `from_sparse_terms`, so `data` is `None`).

        `None` by default: a bare `FitInterferometer` has no real-space model image to evaluate it from.
        Subclasses whose model visibilities are the Fourier transform `p = F i_p` of a real-space image `i_p`
        (e.g. the light-profile fits of PyAutoGalaxy and PyAutoLens without an inversion) override it with
        `data_term - 2 i_p^T d~ + i_p^T W~ i_p`
        (`inversion_interferometer_util.sparse_profile_terms_from`), so `log_likelihood` and
        `figure_of_merit` work array-free. The residual and chi-squared *maps* still need the visibilities and
        raise on such a dataset.
        """
        return None

    @property
    def chi_squared(self) -> float:
        """
        Returns the chi-squared terms of the model data's fit to an dataset, by summing the chi-squared-map.

        On an array-free dataset (no `data`) this is `sparse_chi_squared` when a subclass provides it, and
        otherwise raises an `exc.DatasetException`.
        """
        if self.data is None:
            sparse_chi_squared = self.sparse_chi_squared

            if sparse_chi_squared is not None:
                return sparse_chi_squared

        self._require("chi_squared", "data", "noise_map")
        return fit_util.chi_squared_complex_from(
            chi_squared_map=self.chi_squared_map.array,
        )

    @property
    def noise_normalization(self) -> float:
        """
        Returns the noise-map normalization term of the noise-map, summing the noise_map value in every pixel as:

        [Noise_Term] = sum(log(2*pi*[Noise]**2.0))

        When the dataset carries a sparse operator with a precomputed `noise_normalization`
        (set by `Interferometer.apply_sparse_operator` and
        `apply_sparse_operator_from_chunks`), and the noise-map fitted is the dataset's own
        (not one a subclass has scaled or replaced), that scalar is returned instead of
        reducing over every visibility's sigma on each likelihood call. It is computed with
        the same expression, so the value is identical.
        """
        sparse_operator = getattr(self.dataset, "sparse_operator", None)
        noise_normalization = getattr(sparse_operator, "noise_normalization", None)

        if noise_normalization is not None and self.noise_map is self.dataset.noise_map:
            return noise_normalization

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
