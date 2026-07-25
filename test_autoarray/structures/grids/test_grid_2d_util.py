import autoarray as aa
import numpy as np
import pytest


def test__grid_2d_slim_via_mask_from():
    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask,
        pixel_scales=(3.0, 6.0),
    )

    assert (grid[0] == np.array([0.0, 0.0])).all()

    mask = np.array(
        [
            [True, False, True, True],
            [False, False, False, True],
            [True, False, True, False],
        ]
    )

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask,
        pixel_scales=(3.0, 3.0),
    )

    assert (
        grid
        == np.array(
            [
                [3.0, -1.5],
                [0.0, -4.5],
                [0.0, -1.5],
                [0.0, 1.5],
                [-3.0, -1.5],
                [-3.0, 4.5],
            ]
        )
    ).all()

    mask = np.array(
        [
            [True, False, True, True],
            [False, False, False, True],
            [True, False, True, False],
        ]
    )

    grid = aa.util.grid_2d.grid_2d_slim_via_mask_from(
        mask_2d=mask, pixel_scales=(3.0, 3.0), origin=(1.0, 2.0)
    )

    assert grid == pytest.approx(
        np.array(
            [
                [4.0, 0.5],
                [1.0, -2.5],
                [1.0, 0.5],
                [1.0, 3.5],
                [-2.0, 0.5],
                [-2.0, 6.5],
            ]
        ),
        1e-4,
    )

    mask = np.array([[True, True, False], [False, False, False], [True, True, False]])


def test__grid_2d_via_mask_from():
    mask = np.array([[False, True, True], [True, True, False], [True, True, True]])

    grid_2d = aa.util.grid_2d.grid_2d_via_mask_from(
        mask_2d=mask,
        pixel_scales=(3.0, 6.0),
    )

    assert (
        grid_2d
        == np.array(
            [
                [[3.0, -6.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 6.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__grid_2d_slim_via_shape_native_from():
    grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(2, 3),
        pixel_scales=(1.0, 1.0),
    )

    assert (
        grid_2d
        == np.array(
            [
                [0.5, -1.0],
                [0.5, 0.0],
                [0.5, 1.0],
                [-0.5, -1.0],
                [-0.5, 0.0],
                [-0.5, 1.0],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(3, 2),
        pixel_scales=(1.0, 1.0),
    )

    assert (
        grid_2d
        == np.array(
            [
                [1.0, -0.5],
                [1.0, 0.5],
                [0.0, -0.5],
                [0.0, 0.5],
                [-1.0, -0.5],
                [-1.0, 0.5],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_from(
        shape_native=(3, 2), pixel_scales=(1.0, 1.0), origin=(3.0, -2.0)
    )

    assert (
        grid_2d
        == np.array(
            [
                [4.0, -2.5],
                [4.0, -1.5],
                [3.0, -2.5],
                [3.0, -1.5],
                [2.0, -2.5],
                [2.0, -1.5],
            ]
        )
    ).all()


def test__grid_2d_slim_via_shape_native_not_mask_from():
    grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_not_mask_from(
        shape_native=(2, 3),
        pixel_scales=(1.0, 1.0),
    )

    assert (
        grid_2d
        == np.array(
            [
                [0.5, -1.0],
                [0.5, 0.0],
                [0.5, 1.0],
                [-0.5, -1.0],
                [-0.5, 0.0],
                [-0.5, 1.0],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_not_mask_from(
        shape_native=(3, 2),
        pixel_scales=(1.0, 1.0),
    )

    assert (
        grid_2d
        == np.array(
            [
                [1.0, -0.5],
                [1.0, 0.5],
                [0.0, -0.5],
                [0.0, 0.5],
                [-1.0, -0.5],
                [-1.0, 0.5],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_slim_via_shape_native_not_mask_from(
        shape_native=(3, 2), pixel_scales=(1.0, 1.0), origin=(3.0, -2.0)
    )

    assert (
        grid_2d
        == np.array(
            [
                [4.0, -2.5],
                [4.0, -1.5],
                [3.0, -2.5],
                [3.0, -1.5],
                [2.0, -2.5],
                [2.0, -1.5],
            ]
        )
    ).all()


def test__grid_2d_via_shape_native_from():
    grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(2, 3),
        pixel_scales=(1.0, 1.0),
    )

    assert (
        grid_2d
        == np.array(
            [
                [[0.5, -1.0], [0.5, 0.0], [0.5, 1.0]],
                [[-0.5, -1.0], [-0.5, 0.0], [-0.5, 1.0]],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(3, 2),
        pixel_scales=(1.0, 1.0),
    )

    assert (
        grid_2d
        == np.array(
            [
                [[1.0, -0.5], [1.0, 0.5]],
                [[0.0, -0.5], [0.0, 0.5]],
                [[-1.0, -0.5], [-1.0, 0.5]],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(3, 2), pixel_scales=(1.0, 1.0), origin=(3.0, -2.0)
    )

    assert (
        grid_2d
        == np.array(
            [
                [[4.0, -2.5], [4.0, -1.5]],
                [[3.0, -2.5], [3.0, -1.5]],
                [[2.0, -2.5], [2.0, -1.5]],
            ]
        )
    ).all()


def test__grid_2d_via_shape_native_from():
    grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(2, 3),
        pixel_scales=(1.0, 1.0),
    )

    assert (
        grid_2d
        == np.array(
            [
                [[0.5, -1.0], [0.5, 0.0], [0.5, 1.0]],
                [[-0.5, -1.0], [-0.5, 0.0], [-0.5, 1.0]],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(3, 2),
        pixel_scales=(1.0, 1.0),
    )

    assert (
        grid_2d
        == np.array(
            [
                [[1.0, -0.5], [1.0, 0.5]],
                [[0.0, -0.5], [0.0, 0.5]],
                [[-1.0, -0.5], [-1.0, 0.5]],
            ]
        )
    ).all()

    grid_2d = aa.util.grid_2d.grid_2d_via_shape_native_from(
        shape_native=(3, 2), pixel_scales=(1.0, 1.0), origin=(3.0, -2.0)
    )

    assert (
        grid_2d
        == np.array(
            [
                [[4.0, -2.5], [4.0, -1.5]],
                [[3.0, -2.5], [3.0, -1.5]],
                [[2.0, -2.5], [2.0, -1.5]],
            ]
        )
    ).all()


def test__radial_projected_shape_slim_from():
    shape_slim = aa.util.grid_2d._radial_projected_shape_slim_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
    )

    assert shape_slim == 2

    shape_slim = aa.util.grid_2d._radial_projected_shape_slim_from(
        extent=np.array([-1.0, 3.0, -1.0, 1.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
    )

    assert shape_slim == 4

    shape_slim = aa.util.grid_2d._radial_projected_shape_slim_from(
        extent=np.array([-1.0, 3.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(1.0, 1.0),
    )

    assert shape_slim == 3

    shape_slim = aa.util.grid_2d._radial_projected_shape_slim_from(
        extent=np.array([-2.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(1.0, 1.0),
    )

    assert shape_slim == 4


def test__grid_scaled_2d_slim_radial_projected_from():
    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
    )

    assert grid_radii == pytest.approx(np.array([[0.0, 0.0], [0.0, 1.0]]), abs=1.0e-4)

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 3.0, -1.0, 1.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
    )

    assert grid_radii == pytest.approx(
        np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]), abs=1.0e-4
    )

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 3.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(1.0, 1.0),
    )

    assert grid_radii == pytest.approx(
        np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]), abs=1.0e-4
    )

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-2.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(1.0, 1.0),
    )

    assert grid_radii == pytest.approx(
        np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]]), abs=1.0e-4
    )

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(0.0, 1.0),
        pixel_scales=(0.1, 0.5),
    )

    assert grid_radii == pytest.approx(
        np.array([[0.0, 1.0], [0.0, 1.5], [0.0, 2.0], [0.0, 2.5], [0.0, 3.0]]),
        abs=1.0e-4,
    )

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([5.0, 8.0, 99.9, 100.1]),
        centre=(100.0, 7.0),
        pixel_scales=(10.0, 0.25),
    )

    assert grid_radii == pytest.approx(
        np.array(
            [
                [100.0, 7.0],
                [100.0, 7.25],
                [100.0, 7.5],
                [100.0, 7.75],
                [100.0, 8.0],
                [100.0, 8.25],
                [100.0, 8.5],
                [100.0, 8.75],
                [100.0, 9.0],
            ]
        ),
        abs=1.0e-4,
    )

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 3.0]),
        centre=(0.0, 0.0),
        pixel_scales=(1.0, 1.0),
    )

    assert grid_radii == pytest.approx(
        np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]), abs=1.0e-4
    )

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -2.0, 1.0]),
        centre=(1.0, 0.0),
        pixel_scales=(1.0, 1.0),
    )

    assert grid_radii == pytest.approx(
        np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]), abs=1.0e-4
    )

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([-1.0, 1.0, -1.0, 1.0]),
        centre=(1.0, 0.0),
        pixel_scales=(0.5, 0.1),
    )

    assert grid_radii == pytest.approx(
        np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [1.0, 1.5], [1.0, 2.0]]),
        abs=1.0e-4,
    )

    grid_radii = aa.util.grid_2d.grid_scaled_2d_slim_radial_projected_from(
        extent=np.array([99.9, 100.1, -1.0, 3.0]),
        centre=(-1.0, 100.0),
        pixel_scales=(1.5, 10.0),
    )

    assert grid_radii == pytest.approx(
        np.array([[-1.0, 100.0], [-1.0, 101.5], [-1.0, 103.0]]), abs=1.0e-4
    )


def test__grid_2d_slim_from():
    grid_2d = np.array(
        [
            [[1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6]],
            [[7, 7], [8, 8], [9, 9]],
        ]
    )

    mask = np.array([[True, True, True], [True, False, True], [True, True, True]])

    grid_slim = aa.util.grid_2d.grid_2d_slim_from(
        grid_2d_native=grid_2d,
        mask=mask,
    )

    assert (grid_slim == np.array([[5, 5]])).all()

    grid_2d = np.array(
        [
            [[1, 1], [2, 2], [3, 3], [4, 4]],
            [[5, 5], [6, 6], [7, 7], [8, 8]],
            [[9, 9], [10, 10], [11, 11], [12, 12]],
        ]
    )

    mask = np.array(
        [
            [True, False, True, True],
            [False, False, False, True],
            [True, False, True, False],
        ]
    )

    grid_slim = aa.util.grid_2d.grid_2d_slim_from(
        grid_2d_native=grid_2d,
        mask=mask,
    )

    assert (
        grid_slim == np.array([[2, 2], [5, 5], [6, 6], [7, 7], [10, 10], [12, 12]])
    ).all()

    grid_2d = np.array(
        [
            [[1, 1], [2, 2], [3, 3]],
            [[4, 4], [5, 5], [6, 6]],
            [[7, 7], [8, 8], [9, 9]],
            [[10, 10], [11, 11], [12, 12]],
        ]
    )

    mask = np.array(
        [
            [True, False, True],
            [False, False, False],
            [True, False, True],
            [True, True, True],
        ]
    )

    grid_slim = aa.util.grid_2d.grid_2d_slim_from(
        grid_2d_native=grid_2d,
        mask=mask,
    )

    assert (grid_slim == np.array([[2, 2], [4, 4], [5, 5], [6, 6], [8, 8]])).all()


def test__grid_2d_native_from():
    grid_slim = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

    mask = np.full(fill_value=False, shape=(2, 2))

    grid_2d = aa.util.grid_2d.grid_2d_native_from(
        grid_2d_slim=grid_slim,
        mask_2d=mask,
    )

    assert (
        grid_2d == np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])
    ).all()

    grid_slim = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [-3.0, -3.0],
        ]
    )

    mask = np.array(
        [
            [False, False, True, True],
            [False, True, True, True],
            [False, False, True, False],
        ]
    )

    grid_2d = aa.util.grid_2d.grid_2d_native_from(
        grid_2d_slim=grid_slim,
        mask_2d=mask,
    )

    assert (
        grid_2d
        == np.array(
            [
                [[1.0, 1.0], [2.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
                [[3.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[-1.0, -1.0], [-2.0, -2.0], [0.0, 0.0], [-3.0, -3.0]],
            ]
        )
    ).all()
