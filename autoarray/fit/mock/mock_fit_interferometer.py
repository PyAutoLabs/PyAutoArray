from typing import Optional

from autoarray.dataset.mock.mock_dataset import MockDataset
from autoarray.fit.fit_interferometer import FitInterferometer


class MockFitInterferometer(FitInterferometer):
    def __init__(
        self,
        dataset: Optional[MockDataset] = None,
        use_mask_in_fit: bool = False,
        model_data=None,
        inversion=None,
        noise_map=None,
    ):
        super().__init__(
            dataset=dataset or MockDataset(), use_mask_in_fit=use_mask_in_fit
        )

        self._model_data = model_data
        self._inversion = inversion
        self._noise_map = noise_map

    @property
    def model_data(self):
        return self._model_data

    @property
    def noise_map(self):
        return self._noise_map if self._noise_map is not None else super().noise_map

    @property
    def inversion(self):
        return self._inversion if self._inversion is not None else super().inversion
