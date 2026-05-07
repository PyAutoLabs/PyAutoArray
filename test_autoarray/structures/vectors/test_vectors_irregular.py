from pathlib import Path

from matplotlib.patches import Ellipse
import pytest
import numpy as np

import autoarray as aa
from autoarray import exc


def test__constructor():
    # Input tuples

    vectors = aa.VectorYX2DIrregular(
        values=[(1.0, -1.0), (1.0, 1.0)], grid=[[1.0, -1.0], [1.0, 1.0]]
    )

    assert type(vectors) == aa.VectorYX2DIrregular
    assert (vectors == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert vectors.in_list == [(1.0, -1.0), (1.0, 1.0)]

    # Input np array

    vectors = aa.VectorYX2DIrregular(
        values=[np.array([1.0, -1.0]), np.array([1.0, 1.0])],
        grid=[[1.0, -1.0], [1.0, 1.0]],
    )

    assert type(vectors) == aa.VectorYX2DIrregular
    assert (vectors == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert vectors.in_list == [(1.0, -1.0), (1.0, 1.0)]

    # Input list

    vectors = aa.VectorYX2DIrregular(
        values=[[1.0, -1.0], [1.0, 1.0]], grid=[[1.0, -1.0], [1.0, 1.0]]
    )

    assert type(vectors) == aa.VectorYX2DIrregular
    assert (vectors == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert vectors.in_list == [(1.0, -1.0), (1.0, 1.0)]


def test__constructor__grid_conversions():
    # Input tuples

    vectors = aa.VectorYX2DIrregular(
        values=[(1.0, -1.0), (1.0, 1.0)],
        grid=aa.Grid2DIrregular([[1.0, -1.0], [1.0, 1.0]]),
    )

    assert type(vectors.grid) == aa.Grid2DIrregular
    assert (vectors.grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()

    # Input np array

    vectors = aa.VectorYX2DIrregular(
        values=[(1.0, -1.0), (1.0, 1.0)], grid=[[1.0, -1.0], [1.0, 1.0]]
    )

    assert type(vectors.grid) == aa.Grid2DIrregular
    assert (vectors.grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()

    # Input list

    vectors = aa.VectorYX2DIrregular(
        values=[(1.0, -1.0), (1.0, 1.0)], grid=[(1.0, -1.0), (1.0, 1.0)]
    )

    assert type(vectors.grid) == aa.Grid2DIrregular
    assert (vectors.grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()


def test__vectors_within_radius():
    vectors = aa.VectorYX2DIrregular(
        values=[(1.0, 1.0), (2.0, 2.0)], grid=[[0.0, 1.0], [0.0, 2.0]]
    )

    vectors_masked = vectors.vectors_within_radius(radius=3.0, centre=(0.0, 0.0))

    assert type(vectors_masked) == aa.VectorYX2DIrregular
    assert vectors_masked.in_list == [(1.0, 1.0), (2.0, 2.0)]
    assert vectors_masked.grid.in_list == [(0.0, 1.0), (0.0, 2.0)]

    vectors_masked = vectors.vectors_within_radius(radius=1.5, centre=(0.0, 0.0))

    assert type(vectors_masked) == aa.VectorYX2DIrregular
    assert vectors_masked.in_list == [(1.0, 1.0)]
    assert vectors_masked.grid.in_list == [(0.0, 1.0)]

    vectors_masked = vectors.vectors_within_radius(radius=0.5, centre=(0.0, 2.0))

    assert type(vectors_masked) == aa.VectorYX2DIrregular
    assert vectors_masked.in_list == [(2.0, 2.0)]
    assert vectors_masked.grid.in_list == [(0.0, 2.0)]

    with pytest.raises(exc.VectorYXException):
        vectors.vectors_within_radius(radius=0.0, centre=(0.0, 0.0))
