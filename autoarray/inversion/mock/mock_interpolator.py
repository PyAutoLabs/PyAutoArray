from autoarray.inversion.mesh.interpolator.abstract import AbstractInterpolator


class MockInterpolator(AbstractInterpolator):

    def __init__(self, mappings=None, sizes=None, weights=None):

        self._mappings = mappings
        self._sizes = sizes
        self._weights = weights

    @property
    def mappings(self):
        if self._mappings is not None:
            return self._mappings
        return super().mappings

    @property
    def sizes(self):
        if self._sizes is not None:
            return self._sizes
        return super().sizes

    @property
    def weights(self):
        if self._weights is not None:
            return self._weights
        return super().weights
