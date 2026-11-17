"""
The interpolated source-reconstruction view (PyAutoArray#700): an irregular-mesh
reconstruction evaluated on a uniform grid with the mapper's own interpolant,
i.e. the reconstruction the likelihood uses between mesh vertices.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm

import autoarray as aa
import autoarray.plot  # noqa: F401  (import order: autoarray.plot before autoarray.inversion.plot)
from autoarray import fixtures
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.inversion.plot import inversion_plots
from autoarray.plot.inversion import (
    interpolated_reconstruction_from,
    mapper_has_interpolated_view,
    plot_inversion_reconstruction,
)


def _mesh_grid():
    """A scattered (y, x) mesh with interior vertices and a clean convex hull."""
    rng = np.random.default_rng(1)
    interior = rng.uniform(-0.8, 0.8, size=(30, 2))
    corners = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    return aa.Grid2DIrregular(values=np.vstack([corners, interior]))


def _mapper_for(mesh, mesh_grid=None):
    mesh_grid = mesh_grid if mesh_grid is not None else _mesh_grid()
    data_grid = aa.Grid2D.uniform(
        shape_native=(10, 10), pixel_scales=0.2, over_sample_size=1
    )
    interpolator = mesh.interpolator_from(
        source_plane_data_grid=data_grid,
        source_plane_mesh_grid=mesh_grid,
    )
    return aa.Mapper(
        interpolator=interpolator,
        regularization=fixtures.make_regularization_constant(),
    )


# The kNN interpolators are built on JAX (`knn.py` imports it), so their cases
# skip on the no-JAX CI leg, as the repository's other JAX-dependent tests do.
try:
    import jax  # noqa: F401

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

_requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="kNN interpolators need jax")

MESHES = [
    aa.mesh.Delaunay(pixels=34),
    pytest.param(
        aa.mesh.KNearestNeighbor(pixels=34, split_neighbor_division=1),
        marks=_requires_jax,
        id="KNearestNeighbor",
    ),
    pytest.param(
        aa.mesh.KNNBarycentric(pixels=34, split_neighbor_division=1),
        marks=_requires_jax,
        id="KNNBarycentric",
    ),
    aa.mesh.DelaunayNN(pixels=34),
]


@pytest.mark.parametrize(
    "mesh", MESHES, ids=lambda m: type(m).__name__ if not isinstance(m, str) else m
)
def test__interpolated_view__equals_mapping_matrix_times_reconstruction(mesh):
    mapper = _mapper_for(mesh)
    rng = np.random.default_rng(2)
    values = rng.uniform(0.5, 2.0, size=mapper.pixels)

    extent = (-1.2, 1.2, -1.0, 1.4)
    shape_native = (12, 9)

    image = interpolated_reconstruction_from(
        mapper=mapper, pixel_values=values, extent=extent, shape_native=shape_native
    )

    assert image.shape == shape_native

    # The likelihood's route: a mapper of the same interpolator type whose data
    # grid is the plot grid, mapping_matrix @ reconstruction.
    plot_grid = aa.Grid2D.uniform(
        shape_native=shape_native,
        pixel_scales=(2.4 / 12, 2.4 / 9),
        origin=(0.2, 0.0),
        over_sample_size=1,
    )
    plot_mapper = aa.Mapper(
        interpolator=type(mapper.interpolator)(
            mesh=mapper.interpolator.mesh,
            mesh_grid=mapper.interpolator.mesh_grid,
            data_grid=plot_grid,
        ),
        regularization=None,
    )
    expected = (np.asarray(plot_mapper.mapping_matrix) @ values).reshape(
        shape_native
    )

    inside = ~np.isnan(image)
    assert inside.any() and (~inside).any()  # the extent overhangs the mesh
    assert image[inside] == pytest.approx(expected[inside], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize(
    "mesh", [aa.mesh.Delaunay(pixels=34), aa.mesh.DelaunayNN(pixels=34)]
)
def test__interpolated_view__reproduces_linear_source_exactly(mesh):
    """
    The reporter's check in its simplest form: a source sampled exactly at the
    vertices is recovered between them by the barycentric (and Sibson)
    interpolant, which is exact for a linear source -- unlike flat triangles.
    """
    mapper = _mapper_for(mesh)
    mesh_grid = np.asarray(mapper.interpolator.mesh_grid.array)

    def source(y, x):
        return 3.0 + 2.0 * y - 1.5 * x

    values = source(mesh_grid[:, 0], mesh_grid[:, 1])

    extent = (-1.0, 1.0, -1.0, 1.0)
    shape_native = (40, 40)
    image = interpolated_reconstruction_from(
        mapper=mapper, pixel_values=values, extent=extent, shape_native=shape_native
    )

    grid = aa.Grid2D.uniform(
        shape_native=shape_native, pixel_scales=0.05, over_sample_size=1
    ).native.array
    truth = source(grid[..., 0], grid[..., 1])

    assert not np.isnan(image).any()  # extent == hull: every pixel centre inside
    assert image == pytest.approx(truth, abs=1e-9)
    # Row 0 is the top of the image (largest y).
    assert image[0].mean() > image[-1].mean()


def test__interpolated_view__log10__no_values_below_norm_floor():
    mapper = _mapper_for(aa.mesh.Delaunay(pixels=34))
    values = np.full(mapper.pixels, 1.0e-8)
    values[5] = 10.0

    fig, ax = plt.subplots()
    plot_inversion_reconstruction(
        pixel_values=values,
        mapper=mapper,
        ax=ax,
        use_log10=True,
        zoom_to_brightest=False,
        interpolated=True,
        interpolated_shape_native=(30, 30),
    )

    image = ax.get_images()[0]
    assert isinstance(image.norm, LogNorm)
    data = np.ma.filled(image.get_array().astype(float), np.nan)
    finite = data[~np.isnan(data)]
    assert finite.size > 0
    assert finite.min() >= image.norm.vmin
    plt.close(fig)


def test__interpolated_flag__draws_raster_not_triangles():
    mapper = fixtures.make_delaunay_mapper_9_3x3()
    values = np.arange(1.0, 10.0)

    fig, (ax_flat, ax_interp) = plt.subplots(1, 2)
    plot_inversion_reconstruction(pixel_values=values, mapper=mapper, ax=ax_flat)
    plot_inversion_reconstruction(
        pixel_values=values,
        mapper=mapper,
        ax=ax_interp,
        interpolated=True,
        interpolated_shape_native=(16, 16),
    )

    assert len(ax_flat.get_images()) == 0 and len(ax_flat.collections) > 0
    assert len(ax_interp.get_images()) == 1
    plt.close(fig)


def test__rectangular_mapper__has_no_interpolated_view_and_flag_is_ignored():
    mapper = fixtures.make_rectangular_mapper_7x7_3x3()
    assert not mapper_has_interpolated_view(mapper)
    assert mapper_has_interpolated_view(fixtures.make_delaunay_mapper_9_3x3())
    assert mapper_has_interpolated_view(fixtures.make_knn_mapper_9_3x3())

    fig, ax = plt.subplots()
    plot_inversion_reconstruction(
        pixel_values=np.ones(9), mapper=mapper, ax=ax, interpolated=True
    )
    plt.close(fig)


def _subplot_titles(inversion, monkeypatch):
    captured = {}

    def _capture(fig, *args, **kwargs):
        captured["titles"] = [
            ax.get_title() for ax in fig.axes if ax.get_visible() and ax.get_title()
        ]
        plt.close(fig)

    monkeypatch.setattr(inversion_plots, "subplot_save", _capture)
    inversion_plots.subplot_of_mapper(inversion=inversion, mapper_index=0)
    return captured["titles"]


def test__subplot_of_mapper__delaunay__panel_11_is_interpolated(monkeypatch):
    titles = _subplot_titles(fixtures.make_delaunay_inversion_9_3x3(), monkeypatch)
    assert "Source Reconstruction (Interpolated)" in titles


def test__subplot_of_mapper__rectangular__no_interpolated_panel(monkeypatch):
    titles = _subplot_titles(fixtures.make_rectangular_inversion_7x7_3x3(), monkeypatch)
    assert "Source Reconstruction (Interpolated)" not in titles
    assert "Source Plane (Zoom)" in titles
