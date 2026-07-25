from pathlib import Path

import numpy as np
import autoarray as aa

test_values_dir = Path(__file__).resolve().parent / "files"


def test__input_as_list__convert_correctly():
    values = aa.ArrayIrregular(values=[1.0, -1.0])

    assert type(values) == aa.ArrayIrregular
    assert (values == np.array([1.0, -1.0])).all()
    assert values.in_list == [1.0, -1.0]
