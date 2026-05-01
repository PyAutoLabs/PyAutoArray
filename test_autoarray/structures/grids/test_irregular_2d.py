import os
from os import path
import shutil
import numpy as np

import autoarray as aa


def test__constructor():
    # Input tuple

    grid = aa.Grid2DIrregular(values=[(1.0, -1.0)])

    assert type(grid) == aa.Grid2DIrregular
    assert (grid == np.array([[1.0, -1.0]])).all()
    assert grid.in_list == [(1.0, -1.0)]

    # Input tuples

    grid = aa.Grid2DIrregular(values=[(1.0, -1.0), (1.0, 1.0)])

    assert type(grid) == aa.Grid2DIrregular
    assert (grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

    # Input np array

    grid = aa.Grid2DIrregular(values=[np.array([1.0, -1.0]), np.array([1.0, 1.0])])

    assert type(grid) == aa.Grid2DIrregular
    assert (grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

    # Input list

    grid = aa.Grid2DIrregular(values=[[1.0, -1.0], [1.0, 1.0]])

    assert type(grid) == aa.Grid2DIrregular
    assert (grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
    assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]


def test__from_yx_1d():
    grid = aa.Grid2DIrregular.from_yx_1d(y=[1.0, 1.0], x=[-1.0, 1.0])

    assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

    grid = aa.Grid2DIrregular.from_yx_1d(
        y=[1.0, 1.0, 2.0, 4.0], x=[-1.0, 1.0, 3.0, 5.0]
    )

    assert grid.in_list == [(1.0, -1.0), (1.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


def test__from_pixels_and_mask():
    mask = aa.Mask2D(
        mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
    )

    grid = aa.Grid2DIrregular(values=[(1.0, -1.0), (1.0, 1.0)])
    grid = aa.Grid2DIrregular.from_pixels_and_mask(pixels=[(0, 0), (0, 1)], mask=mask)

    assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]


def test__grid_2d_via_deflection_grid_from():
    grid = aa.Grid2DIrregular(values=[(1.0, 1.0), (2.0, 2.0)])

    grid = grid.grid_2d_via_deflection_grid_from(
        deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
    )

    assert type(grid) == aa.Grid2DIrregular
    assert grid.in_list == [(0.0, 1.0), (1.0, 1.0)]


def test__grid_2d_via_deflection_grid_from__propagates_xp():
    # numpy-backed receiver -> numpy-backed result.
    # The JAX half of this assertion lives in
    # autolens_workspace_test/scripts/jax_assertions/grid_irregular.py.
    grid_np = aa.Grid2DIrregular(values=[(1.0, 1.0), (2.0, 2.0)])
    result_np = grid_np.grid_2d_via_deflection_grid_from(
        deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
    )
    assert result_np._xp is np


def test__furthest_distances_to_other_coordinates():
    grid = aa.Grid2DIrregular(values=[(0.0, 0.0), (0.0, 1.0)])

    assert grid.furthest_distances_to_other_coordinates.in_list == [1.0, 1.0]

    grid = aa.Grid2DIrregular(values=[(2.0, 4.0), (3.0, 6.0)])

    assert grid.furthest_distances_to_other_coordinates.in_list == [
        np.sqrt(5),
        np.sqrt(5),
    ]

    grid = aa.Grid2DIrregular(values=[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)])

    assert grid.furthest_distances_to_other_coordinates.in_list == [3.0, 2.0, 3.0]


def test__grid_of_closest_from():
    grid = aa.Grid2DIrregular(values=[(0.0, 0.0), (0.0, 1.0)])

    grid_of_closest = grid.grid_of_closest_from(
        grid_pair=aa.Grid2DIrregular(np.array([[0.0, 0.1]]))
    )

    assert (grid_of_closest == np.array([[0.0, 0.0]])).all()

    grid_of_closest = grid.grid_of_closest_from(
        grid_pair=aa.Grid2DIrregular(np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]]))
    )

    assert (grid_of_closest == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])).all()

    grid_of_closest = grid.grid_of_closest_from(
        grid_pair=aa.Grid2DIrregular(
            np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.9], [0.0, -0.1]])
        )
    )

    assert (
        grid_of_closest == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    ).all()
