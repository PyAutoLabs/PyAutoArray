import numpy as np
import pytest

import autoarray as aa


def test__no_mask():
    vectors = aa.VectorYX2D.no_mask(
        values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        pixel_scales=1.0,
    )

    assert type(vectors) == aa.VectorYX2D
    assert (vectors == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
    assert (
        vectors.native == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ).all()
    assert (
        vectors.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    ).all()
    assert (
        vectors.grid.native
        == np.array([[[0.5, -0.5], [0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]])
    ).all()
    assert vectors.pixel_scales == (1.0, 1.0)
    assert vectors.origin == (0.0, 0.0)

    vectors = aa.VectorYX2D.no_mask(
        values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        shape_native=(4, 2),
        pixel_scales=0.5,
        origin=(0.0, 1.0),
    )

    assert type(vectors) == aa.VectorYX2D
    assert (vectors == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])).all()
    assert (
        vectors.native == np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    ).all()
    assert (
        vectors.slim == np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    ).all()
    assert (
        vectors.grid.native
        == np.array([[[0.25, 0.75], [0.25, 1.25]], [[-0.25, 0.75], [-0.25, 1.25]]])
    ).all()
    assert vectors.pixel_scales == (0.5, 0.5)
    assert vectors.origin == (0.0, 1.0)


def test__from_mask():
    mask = aa.Mask2D.all_false(shape_native=(2, 2), pixel_scales=1.0)

    vectors = aa.VectorYX2D.from_mask(
        values=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], mask=mask
    )

    assert type(vectors) == aa.VectorYX2D
    assert (
        vectors.native == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
    ).all()
    assert (
        vectors.slim == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    ).all()
    assert (
        vectors.grid.native
        == np.array([[[0.5, -0.5], [0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]])
    ).all()
    assert (
        vectors.grid.slim
        == np.array([[0.5, -0.5], [0.5, 0.5], [-0.5, -0.5], [-0.5, 0.5]])
    ).all()
    assert vectors.pixel_scales == (1.0, 1.0)
    assert vectors.origin == (0.0, 0.0)

    mask = aa.Mask2D(
        mask=[[False, False], [True, False]], pixel_scales=1.0, origin=(0.0, 1.0)
    )

    vectors = aa.VectorYX2D.from_mask(
        values=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], mask=mask
    )

    assert type(vectors) == aa.VectorYX2D
    assert (
        vectors.native == np.array([[[1.0, 1.0], [2.0, 2.0]], [[0.0, 0.0], [4.0, 4.0]]])
    ).all()
    assert (vectors.slim == np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])).all()
    assert (
        vectors.grid.native
        == np.array([[[0.5, 0.5], [0.5, 1.5]], [[0.0, 0.0], [-0.5, 1.5]]])
    ).all()
    assert (vectors.grid.slim == np.array([[0.5, 0.5], [0.5, 1.5], [-0.5, 1.5]])).all()
    assert vectors.pixel_scales == (1.0, 1.0)
    assert vectors.origin == (0.0, 1.0)

    mask = aa.Mask2D(
        mask=[[False, False], [False, False], [True, True], [True, True]],
        pixel_scales=2.0,
    )

    vectors = aa.VectorYX2D.from_mask(
        values=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], mask=mask
    )

    assert type(vectors) == aa.VectorYX2D
    assert (vectors == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])).all()
    assert (
        vectors.native
        == np.array(
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[3.0, 3.0], [4.0, 4.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()
    assert (
        vectors.slim == np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    ).all()
    assert vectors.pixel_scales == (2.0, 2.0)
    assert vectors.origin == (0.0, 0.0)


def test__ones():
    vectors = aa.VectorYX2D.ones(shape_native=(2, 2), pixel_scales=1.0)

    assert type(vectors) == aa.VectorYX2D
    assert (vectors == np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])).all()
    assert (
        vectors.native == np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]])
    ).all()
    assert (
        vectors.slim == np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    ).all()
    assert (
        vectors.grid.native
        == np.array([[[0.5, -0.5], [0.5, 0.5]], [[-0.5, -0.5], [-0.5, 0.5]]])
    ).all()
    assert vectors.pixel_scales == (1.0, 1.0)
    assert vectors.origin == (0.0, 0.0)


def test__zeros():
    vectors = aa.VectorYX2D.zeros(
        shape_native=(2, 2), pixel_scales=0.5, origin=(0.0, 1.0)
    )

    assert type(vectors) == aa.VectorYX2D
    assert (vectors == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])).all()
    assert (
        vectors.native == np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    ).all()
    assert (
        vectors.slim == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ).all()
    assert (
        vectors.grid.native
        == np.array([[[0.25, 0.75], [0.25, 1.25]], [[-0.25, 0.75], [-0.25, 1.25]]])
    ).all()
    assert vectors.pixel_scales == (0.5, 0.5)
    assert vectors.origin == (0.0, 1.0)


def test__y_x():
    vectors = aa.VectorYX2D.no_mask(
        values=[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        pixel_scales=1.0,
    )

    assert isinstance(vectors.y, aa.Array2D)
    assert (vectors.y.native == np.array([[1.0, 3.0], [5.0, 7.0]])).all()

    assert isinstance(vectors.x, aa.Array2D)
    assert (vectors.x.native == np.array([[2.0, 4.0], [6.0, 8.0]])).all()


def test__apply_mask():
    mask = aa.Mask2D(
        mask=[[False, False], [False, False], [True, True], [True, True]],
        pixel_scales=2.0,
    )
    vectors = aa.VectorYX2D.no_mask(
        values=[
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
            [8.0, 8.0],
        ],
        shape_native=(4, 2),
        pixel_scales=2.0,
    )
    vectors = vectors.apply_mask(mask=mask)

    assert (
        vectors.native
        == np.array(
            [
                [[1.0, 1.0], [2.0, 2.0]],
                [[3.0, 3.0], [4.0, 4.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__magnitudes():
    vectors = aa.VectorYX2D.no_mask(
        values=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
        pixel_scales=1.0,
    )

    assert isinstance(vectors.magnitudes, aa.Array2D)
    assert vectors.magnitudes.native == pytest.approx(
        np.array([[np.sqrt(2), np.sqrt(8)], [np.sqrt(18), np.sqrt(32)]]), 1.0e-4
    )
