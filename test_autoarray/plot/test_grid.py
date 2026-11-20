"""
``plot_grid(indexes=...)`` highlights groups of grid points. Indexes outside the
grid are dropped rather than raising ``IndexError`` (PyAutoArray#705): tutorials
pick indexes for a full-size grid, which ``PYAUTO_SMALL_DATASETS`` can cap.
"""

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import autoarray as aa
import autoarray.plot as aplt


def _grid():
    return aa.Grid2D.uniform(shape_native=(4, 4), pixel_scales=1.0)


def _highlight_offsets(ax):
    # The first collection is the base scatter of every grid point; the rest are
    # one scatter per highlighted index group.
    return [np.asarray(c.get_offsets()) for c in ax.collections[1:]]


def test__plot_grid__indexes_in_range__each_group_highlighted():
    grid = _grid()
    fig, ax = plt.subplots()

    aplt.plot_grid(grid=grid, indexes=[[0, 1], [5]], ax=ax)

    offsets = _highlight_offsets(ax)
    assert len(offsets) == 2
    values = np.asarray(grid.array)
    assert np.allclose(offsets[0], values[[0, 1]][:, ::-1])
    assert np.allclose(offsets[1], values[[5]][:, ::-1])
    plt.close(fig)


def test__plot_grid__indexes_partly_out_of_range__out_of_range_dropped(caplog):
    grid = _grid()
    fig, ax = plt.subplots()

    with caplog.at_level(logging.INFO, logger="autoarray.plot.grid"):
        aplt.plot_grid(grid=grid, indexes=[[2, 5118], [7099]], ax=ax)

    offsets = _highlight_offsets(ax)
    assert len(offsets) == 1
    assert np.allclose(offsets[0], np.asarray(grid.array)[[2]][:, ::-1])
    assert "dropped 2 highlighted index(es) outside the 16-point grid" in caplog.text
    plt.close(fig)


def test__plot_grid__indexes_all_out_of_range__no_highlight_and_no_error():
    grid = _grid()
    fig, ax = plt.subplots()

    aplt.plot_grid(
        grid=grid,
        indexes=[[5118, 5119], [3078]],
        ax=ax,
        extent=(-5.5, 5.5, -5.5, 5.5),
    )

    assert _highlight_offsets(ax) == []
    plt.close(fig)


def test__plot_grid__boolean_mask_group__highlights_masked_points():
    grid = _grid()
    fig, ax = plt.subplots()
    mask = np.zeros(16, dtype=bool)
    mask[[7, 9]] = True

    aplt.plot_grid(grid=grid, indexes=[mask], ax=ax)

    offsets = _highlight_offsets(ax)
    assert len(offsets) == 1
    assert np.allclose(offsets[0], np.asarray(grid.array)[[7, 9]][:, ::-1])
    plt.close(fig)
