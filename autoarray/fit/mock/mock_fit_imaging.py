from typing import Optional

from autoarray.dataset.mock.mock_dataset import MockDataset
from autoarray.dataset.dataset_model import DatasetModel
from autoarray.fit.fit_imaging import FitImaging


class MockFitImaging(FitImaging):
    def __init__(
        self,
        dataset: Optional[MockDataset] = None,
        dataset_model: Optional[DatasetModel] = None,
        use_mask_in_fit: bool = False,
        noise_map=None,
        model_data=None,
        inversion=None,
        blurred_image=None,
    ):
        super().__init__(
            dataset=dataset or MockDataset(),
            dataset_model=dataset_model,
            use_mask_in_fit=use_mask_in_fit,
        )

        self._noise_map = noise_map
        self._model_data = model_data
        self._inversion = inversion
        self._blurred_image = blurred_image

    @property
    def model_data(self):
        return self._model_data

    @property
    def noise_map(self):
        return self._noise_map if self._noise_map is not None else super().noise_map

    @property
    def inversion(self):
        return self._inversion if self._inversion is not None else super().inversion

    @property
    def blurred_image(self):
        return self._blurred_image if self._blurred_image is not None else None
