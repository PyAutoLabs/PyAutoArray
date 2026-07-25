import autoarray as aa


class MockClass:
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value
