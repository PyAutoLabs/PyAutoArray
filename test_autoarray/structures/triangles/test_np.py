import numpy as np
import pytest

from autoarray.structures.triangles.shape import Point
from autoarray.structures.triangles.array_np import ArrayTrianglesNp


@pytest.fixture
def triangles():
    return ArrayTrianglesNp(
        indices=np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ]
        ),
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ),
        max_containing_size=5,
    )


@pytest.mark.parametrize(
    "offset",
    [-1, 0, 1],
)
def test_simple_neighborhood(offset, compare_with_nans):
    triangles = ArrayTrianglesNp(
        indices=np.array(
            [
                [0, 1, 2],
            ]
        ),
        vertices=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        + offset,
    )

    assert compare_with_nans(
        triangles.neighborhood().triangles,
        (
            np.array(
                [
                    [[-1.0, 1.0], [0.0, 0.0], [0.0, 1.0]],
                    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
                    [[0.0, 0.0], [1.0, -1.0], [1.0, 0.0]],
                    [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
                ]
            )
            + offset
        ),
    )


def test_neighborhood(triangles, compare_with_nans):
    neighborhood = triangles.neighborhood()

    assert compare_with_nans(
        neighborhood.vertices,
        np.array(
            [
                [-1.0, 1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ]
        ),
    )

    assert (
        neighborhood.indices
        == np.array(
            [
                [0, 1, 2],
                [1, 2, 5],
                [1, 4, 5],
                [2, 3, 6],
                [2, 5, 6],
                [5, 6, 7],
            ]
        )
    ).all()


def test_means(triangles):
    means = triangles.means
    assert means == pytest.approx(
        np.array(
            [
                [0.33333333, 0.33333333],
                [0.66666667, 0.66666667],
            ]
        )
    )
