import numpy as np

from autoarray.dataset.imaging.dataset import Imaging
from autoarray.dataset.dataset_model import DatasetModel
from autoarray.fit.fit_dataset import FitDataset
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray import type as ty


class FitImaging(FitDataset):
    def __init__(
        self,
        dataset: Imaging,
        use_mask_in_fit: bool = False,
        dataset_model: DatasetModel = None,
        xp=np,
    ):
        """
        Class to fit a masked imaging dataset.

        Parameters
        ----------
        dataset
            The masked dataset that is fitted.
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

        super().__init__(
            dataset=dataset,
            use_mask_in_fit=use_mask_in_fit,
            dataset_model=dataset_model,
            xp=xp,
        )

    @property
    def data(self) -> ty.DataLike:
        """
        The imaging data being fitted, with any background sky level subtracted.

        The background sky is taken from `dataset_model.background_sky_level`, which defaults to 0.0 if not
        set, meaning this property is equivalent to `dataset.data` in the common case.
        """
        return self.dataset.data - self.dataset_model.background_sky_level

    @property
    def blurred_image(self) -> Array2D:
        """
        The PSF-convolved (blurred) model image of the fit, as a 2D `Array2D`.

        This is the model image after it has been convolved with the dataset's PSF. It must be implemented by
        child classes (e.g. a tracer fit) that produce a blurred model image as part of their fitting procedure.
        """
        raise NotImplementedError
