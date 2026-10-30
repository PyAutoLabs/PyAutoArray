"""
The source-plane positions a rectangular mapper reports for its pixels
(`mapper.source_plane_mesh_grid` = `mesh_geometry.node_centres`) must be where its
interpolant takes each reconstruction value (PyAutoArray#667).

For the adaptive meshes the interpolant lives on a regular lattice in CDF space, so
the pixel centres are that lattice mapped back through the inverse CDF. The uniform
overlay `mesh_grid` the interpolator is built from is the lattice before that mapping:
using it as the pixel positions put the bright, densely packed central nodes on cells
several times too large, so anything interpolating the reconstruction from those
positions (the reconstruction CSV, `max_pixel_centre`, kernel regularization
distances, the workspace's interpolated source) was wrong.
"""

import numpy as np
import pytest
from scipy.interpolate import griddata

import autoarray as aa
from autoarray.inversion.mesh.interpolator.rectangular import (
    KERNEL_CDF_DEFAULT_BANDWIDTH,
    KERNEL_CDF_DEFAULT_KNOTS,
    adaptive_rectangular_mappings_weights_via_interpolation_from,
)

SHAPE = 24
PIXEL_SCALE = 0.2
SUB_SIZE = 2
MESH_SHAPE = (12, 12)

GAUSS_CENTRE = (0.2, -0.1)
GAUSS_SIGMA = 0.5

RECTANGULAR_ADAPTIVE = [
    "RectangularBilinearAdaptDensity",
    "RectangularBilinearAdaptImage",
    "RectangularRTUAdaptDensity",
    "RectangularRTUAdaptImage",
]


def _gaussian(grid):
    return np.exp(
        -((grid[:, 0] - GAUSS_CENTRE[0]) ** 2 + (grid[:, 1] - GAUSS_CENTRE[1]) ** 2)
        / (2.0 * GAUSS_SIGMA**2)
    )


def _distort(points):
    """
    A smooth, lensing-like distortion of the uniform grid, so no two points share a
    y or x coordinate. On an undistorted grid the coordinates are tied in whole rows
    and columns, the rank CDF is a staircase and its inverse is ambiguous on the
    steps (the documented worst case of the rank transform); a traced source-plane
    grid has no such ties.
    """
    y, x = points[:, 0], points[:, 1]
    return np.stack(
        [y + 0.15 * np.sin(1.3 * x) + 0.05 * y**2, x + 0.15 * np.sin(1.1 * y)],
        axis=1,
    )


def _mapper_from(name):
    uniform = aa.Grid2D.uniform(
        shape_native=(SHAPE, SHAPE),
        pixel_scales=PIXEL_SCALE,
        over_sample_size=SUB_SIZE,
    )
    grid = aa.Grid2D(
        values=_distort(np.asarray(uniform.array)),
        mask=uniform.mask,
        over_sample_size=SUB_SIZE,
        over_sampled=aa.Grid2DIrregular(
            _distort(np.asarray(uniform.over_sampled.array))
        ),
    )
    adapt_data = aa.Array2D(values=_gaussian(grid.array) + 0.01, mask=grid.mask)

    interpolator = getattr(aa.mesh, name)(shape=MESH_SHAPE).interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=None,
        adapt_data=adapt_data,
    )
    return grid, aa.Mapper(interpolator=interpolator)


def _interior():
    n_y, n_x = MESH_SHAPE
    interior = np.ones(MESH_SHAPE, dtype=bool)
    interior[:2, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    return interior.ravel()


@pytest.mark.parametrize("name", RECTANGULAR_ADAPTIVE)
def test__each_node_takes_its_full_weight_at_its_centre(name):
    _, mapper = _mapper_from(name)

    geometry = mapper.mesh_geometry
    centres = np.asarray(mapper.source_plane_mesh_grid.array)

    np.testing.assert_allclose(centres, np.asarray(geometry.node_centres))

    mappings, weights = adaptive_rectangular_mappings_weights_via_interpolation_from(
        source_grid_size=MESH_SHAPE[0],
        data_grid=geometry.data_grid.array,
        data_grid_over_sampled=centres,
        mesh_weight_map=geometry.mesh_weight_map,
        bandwidth=(
            geometry.kernel_bandwidth
            if geometry.kernel_bandwidth is not None
            else KERNEL_CDF_DEFAULT_BANDWIDTH
        ),
        n_knots=(
            geometry.kernel_knots
            if geometry.kernel_knots is not None
            else KERNEL_CDF_DEFAULT_KNOTS
        ),
        transform=geometry.transform,
    )

    pixels = np.arange(mapper.params)
    self_weight = np.array([np.sum(weights[k][mappings[k] == k]) for k in pixels])

    # The centres invert the mapper's forward CDF numerically, so a node's own weight
    # at its centre is 1 up to that inversion: the kernel CDF's knot table (measured
    # min 0.994 here) and the rank CDF's piecewise-linear interpolation between this
    # coarse grid's 576 points (min 0.969).
    tolerance = 0.04 if geometry.transform == "rank" else 0.01

    assert self_weight[_interior()] == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize("name", RECTANGULAR_ADAPTIVE)
def test__centres_are_a_row_major_lattice_inside_the_data(name):
    grid, mapper = _mapper_from(name)

    centres = np.asarray(mapper.source_plane_mesh_grid.array).reshape(
        MESH_SHAPE[0], MESH_SHAPE[1], 2
    )

    # y is constant along a row and decreases down the rows; x is constant down a
    # column and increases along it
    assert np.all(np.ptp(centres[:, :, 0], axis=1) == 0.0)
    assert np.all(np.ptp(centres[:, :, 1], axis=0) == 0.0)
    assert np.all(np.diff(centres[:, 0, 0]) <= 0.0)
    assert np.all(np.diff(centres[0, :, 1]) >= 0.0)

    # guard nodes are clamped to the data span, never outside it
    data = np.asarray(grid.array)
    assert np.all(centres[:, :, 0] >= data[:, 0].min() - 1.0e-12)
    assert np.all(centres[:, :, 0] <= data[:, 0].max() + 1.0e-12)
    assert np.all(centres[:, :, 1] >= data[:, 1].min() - 1.0e-12)
    assert np.all(centres[:, :, 1] <= data[:, 1].max() + 1.0e-12)


def _reconstruction_from(grid, mapper):
    """A noise-free least-squares reconstruction of the Gaussian through the mapper."""
    mapping_matrix = np.asarray(mapper.mapping_matrix_over_sampled)
    data = _gaussian(np.asarray(grid.over_sampled.array))
    curvature = mapping_matrix.T @ mapping_matrix
    ridge = 1.0e-8 * np.mean(np.diag(curvature)) * np.eye(curvature.shape[0])
    return np.linalg.solve(curvature + ridge, mapping_matrix.T @ data)


def _interpolated_integral_from(points, reconstruction):
    fine = aa.Grid2D.uniform(shape_native=(600, 600), pixel_scales=0.01)
    interpolated = griddata(
        points=points, values=reconstruction, xi=np.asarray(fine.array)
    )
    return np.nansum(interpolated) * 0.01**2


@pytest.mark.parametrize("name", RECTANGULAR_ADAPTIVE)
def test__interpolating_the_reconstruction_from_the_centres(name):
    """
    Linearly interpolating a reconstruction from the reported pixel positions must
    put its flux in the right place. From the uniform overlay the integral was off by
    a factor 0.59 to 3.7 on this setup.

    On the density-adaptive meshes the integral then matches the analytic Gaussian to
    ~2%. The image-adaptive meshes pack their nodes tightly onto the source, where
    linear interpolation between centres in the source plane is a different
    interpolant from the mapper's bilinear one in CDF space, so their integral is
    only much closer than the overlay's (measured 1.30-1.64 against 2.9-3.7): the
    mesh quadrature `areas_for_magnification` remains the way to integrate.
    """
    grid, mapper = _mapper_from(name)

    reconstruction = _reconstruction_from(grid, mapper)
    analytic = 2.0 * np.pi * GAUSS_SIGMA**2

    centres = _interpolated_integral_from(
        np.asarray(mapper.source_plane_mesh_grid.array), reconstruction
    )
    overlay = _interpolated_integral_from(
        np.asarray(mapper.interpolator.mesh_grid.array), reconstruction
    )

    error_centres = abs(np.log(centres / analytic))
    error_overlay = abs(np.log(overlay / analytic))

    assert error_centres < 0.5 * error_overlay

    if "AdaptDensity" in name:
        assert centres / analytic == pytest.approx(1.0, abs=0.03)


def test__rectangular_uniform__centres_are_the_overlay_grid():
    _, mapper = _mapper_from("RectangularUniform")

    np.testing.assert_array_equal(
        np.asarray(mapper.source_plane_mesh_grid.array),
        np.asarray(mapper.interpolator.mesh_grid.array),
    )
