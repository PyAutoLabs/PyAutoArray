"""
Overlays drawn by ``plot_array`` must stay registered to the raster whatever the
``visualize/general.yaml -> general -> imshow_origin`` config is (PyAutoArray#565).

The checks are origin-agnostic: they read the drawn overlay coordinates back off
the axes and map them through the rendered image's own extent and origin to the
pixel they land on, which must be a bright pixel of an off-centre block.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from autonerves import conf

import autoarray as aa
import autoarray.plot as aplt


def _bright_block_array():
    values = np.zeros((50, 50))
    values[10:14, 23:27] = 1.0
    return aa.Array2D.no_mask(values=values, pixel_scales=0.1)


def _native_grid(array):
    return np.asarray(array.mask.derive_grid.all_false.native)


def _pixel_value_under(ax, draw_x, draw_y):
    image = ax.images[0]
    data = np.asarray(image.get_array())
    xmin, xmax, ymin, ymax = image.get_extent()
    ny, nx = data.shape[:2]

    col = int((draw_x - xmin) / (xmax - xmin) * nx)
    frac = (draw_y - ymin) / (ymax - ymin)
    row = int((1.0 - frac) * ny) if image.origin == "upper" else int(frac * ny)

    return data[row, col]


@pytest.fixture(name="imshow_origin")
def make_imshow_origin(request):
    general = conf.instance["visualize"]["general"]["general"]
    original = general["imshow_origin"]
    try:
        general["imshow_origin"] = request.param
        yield request.param
    finally:
        general["imshow_origin"] = original


@pytest.mark.parametrize("imshow_origin", ["upper", "lower"], indirect=True)
def test__plot_array__positions_marker_lands_on_block(imshow_origin):
    array = _bright_block_array()
    grid = _native_grid(array)
    y, x = float(grid[12, 25, 0]), float(grid[12, 25, 1])

    fig, ax = plt.subplots()
    try:
        aplt.plot_array(array=array, positions=aa.Grid2DIrregular([(y, x)]), ax=ax)

        assert ax.images[0].origin == imshow_origin

        # The positions scatter is the only s=20, zorder=5 collection (the auto
        # mask-edge scatter is s=1).
        positions = [
            c
            for c in ax.collections
            if c.get_zorder() == 5 and np.allclose(c.get_sizes(), 20)
        ]
        assert len(positions) == 1
        draw_x, draw_y = positions[0].get_offsets()[0]

        assert _pixel_value_under(ax, float(draw_x), float(draw_y)) == 1.0
    finally:
        plt.close(fig)


@pytest.mark.parametrize("imshow_origin", ["upper", "lower"], indirect=True)
def test__plot_array__lines_vertices_land_on_block(imshow_origin):
    array = _bright_block_array()
    grid = _native_grid(array)

    # An asymmetric polyline whose every vertex sits on a bright pixel.
    pixels = [(10, 23), (13, 24), (11, 26)]
    line = np.array([[grid[r, c, 0], grid[r, c, 1]] for r, c in pixels])

    fig, ax = plt.subplots()
    try:
        aplt.plot_array(array=array, lines=[line], ax=ax)

        assert ax.images[0].origin == imshow_origin

        xy = ax.lines[-1].get_xydata()
        assert xy.shape == (3, 2)

        for draw_x, draw_y in xy:
            assert _pixel_value_under(ax, float(draw_x), float(draw_y)) == 1.0
    finally:
        plt.close(fig)


@pytest.mark.parametrize("imshow_origin", ["upper", "lower"], indirect=True)
def test__plot_inversion_reconstruction__uniform_mesh_grid_lands_on_pixel(
    imshow_origin, rectangular_mapper_7x7_3x3
):
    mapper = rectangular_mapper_7x7_3x3

    # One off-centre bright source pixel (row 0, column 1 of the 3x3 mesh).
    pixel_values = np.zeros(9)
    pixel_values[1] = 1.0

    mesh = aa.Array2D.no_mask(
        values=np.zeros(mapper.mesh_geometry.shape),
        pixel_scales=mapper.mesh_geometry.pixel_scales,
        origin=mapper.mesh_geometry.origin,
    )
    y, x = _native_grid(mesh)[0, 1]

    fig, ax = plt.subplots()
    try:
        aplt.plot_inversion_reconstruction(
            pixel_values=pixel_values,
            mapper=mapper,
            grid=np.array([[y, x]]),
            zoom_to_brightest=False,
            ax=ax,
        )

        assert ax.images[0].origin == imshow_origin

        draw_x, draw_y = ax.collections[-1].get_offsets()[0]

        assert _pixel_value_under(ax, float(draw_x), float(draw_y)) == 1.0
    finally:
        plt.close(fig)
