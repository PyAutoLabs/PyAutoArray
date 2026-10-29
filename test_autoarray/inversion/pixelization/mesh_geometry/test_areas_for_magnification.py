"""
Per-mesh correctness of `mesh_geometry.areas_for_magnification` (PyAutoArray#659).

Every pixelized magnification (PyAutoLens `analysis/latent.py`, the workspace
`source_science.py` scripts) divides the lensed image-plane flux by
`sum(reconstruction * areas_for_magnification)`, so the areas must be the
quadrature weights of the reconstructed source surface.

The tests use an identity lens: the source-plane data grid is a uniform image grid,
so every data sub-pixel covers a known source-plane area. For any reconstruction
`s` the mapper's own interpolant then integrates to

    Q_k = sum_j mapping_matrix_over_sampled[j, k] * sub_pixel_area
    integral of the reconstructed source = sum_k s_k * Q_k

and `sum_k s_k * A_k / sum_k s_k * Q_k` is the identity-lens magnification, which
must be 1. The checks are:

(a) the areas sum to the analytic area of the domain the mesh covers — the
    over-sampled data grid's bounding box for the rectangular meshes (the adaptive
    meshes integrate their basis functions over it; `RectangularUniform`'s overlay
    grid spans it), and the convex hull of the vertices for the Delaunay family
    (barycentric dual areas tile it);
(b) for a smooth Gaussian source reconstructed on the mesh, the identity-lens
    magnification is 1 and the integrated reconstruction matches the analytic
    integral, within a per-mesh tolerance;
(c) the guard / boundary-cell semantics are pinned.

Tolerances for meshes whose areas are an approximation rather than the exact
quadrature of their interpolant are set at the measured error (see
`MAGNIFICATION_TOLERANCE`) and documented there; they are not loosened to make a
test pass.
"""

import numpy as np
import pytest
import scipy.spatial

import autoarray as aa

SHAPE = 24
PIXEL_SCALE = 0.2  # image box is [-2.4, 2.4]^2
SUB_SIZE = 2
MESH_SHAPE = (12, 12)

GAUSS_CENTRE = (0.2, -0.1)
GAUSS_SIGMA = 0.5


# Absolute tolerance on |mu_identity - 1| (the mapper's exact quadrature of the
# reconstruction over its area-weighted sum) for every mesh, at this module's 24x24
# data grid and 12x12-equivalent mesh, a deliberately coarse mesh (Gaussian sigma ~
# 1.25 mesh spacings) on which approximation errors are largest. Values were measured
# with this exact configuration (PyAutoArray#659) and set ~1.5-2x above the
# measurement, quoted in brackets.
#
# Tight:
#   - Delaunay: barycentric dual areas are the exact quadrature of the linear
#     interpolant (PyAutoArray#524) [-1.1e-4].
#   - RectangularUniform: uniform cells centred on the overlay nodes. They tile the
#     overlay box exactly (the area SUM is exact), but per node they only
#     approximate the quadrature of the clipped + extrapolated bilinear interpolant
#     near the boundary ring (measured Q/A ~1.74 corner, ~1.32 edge, ~0.81 second
#     ring, ~1.0 interior); negligible for a source away from the mesh edge, hence
#     [-1.2e-3]. Before PyAutoArray#659 its areas came from the adaptive meshes'
#     kernel-CDF guard-ring geometry, which does not describe the uniform mapper at
#     all, and this error was +55%.
#
#   - The adaptive rectangular meshes: the exact per-axis integral of each node's
#     bilinear hat through the mapper's forward CDF
#     (`adaptive_rectangular_hat_areas_from`). The residual is this coarse grid's
#     sub-pixel quadrature `Q`, not the areas [BilinearAdaptDensity -3.1e-4,
#     BilinearAdaptImage -4.2e-3, RTUAdaptDensity +2.3e-3, RTUAdaptImage -5.3e-3].
#     The node-midpoint cells they replaced (PyAutoArray#659) were off by
#     [+3.1e-2, +2.4e-1, +2.5e-3, +8.0e-2]: the inverse CDF of a cell's edges is
#     not the integral of the node's hat once the CDF bends across the cell (the
#     rank CDF's staircase on tied coordinates; a steep adapt-image CDF).
#
#   - KNearestNeighbor / KNNBarycentric: the integral of each vertex's kNN basis
#     function over the convex hull, by midpoint quadrature of the interpolator's
#     own weights (`MeshGeometryKNearestNeighbor`, PyAutoArray#665)
#     [+1.3e-5, +1.8e-3; KNNBarycentric's clipped weights are discontinuous, so
#     the coarse sub-pixel `Q` is the limit there]. The Delaunay dual areas they
#     used before measured [+1.6e-2, -4.4e-2]. The quadrature integrates whatever
#     neighbours the interpolator selects, so it stays exact with or without the
#     open partial-block neighbour bug PyAutoArray#609.
#
# Documented approximation:
#   - DelaunayNN inherits the Delaunay dual areas, exact only for the linear
#     interpolant; for its Sibson interpolant they are an approximation [-4.3e-3].
MAGNIFICATION_TOLERANCE = {
    "RectangularUniform": 2.5e-3,
    "RectangularBilinearAdaptDensity": 1.0e-3,
    "RectangularBilinearAdaptImage": 8.0e-3,
    "RectangularRTUAdaptDensity": 5.0e-3,
    "RectangularRTUAdaptImage": 1.0e-2,
    "Delaunay": 3.0e-4,
    "DelaunayNN": 1.0e-2,
    "KNearestNeighbor": 1.0e-4,
    "KNNBarycentric": 4.0e-3,
}

RECTANGULAR_ADAPTIVE = [
    "RectangularBilinearAdaptDensity",
    "RectangularBilinearAdaptImage",
    "RectangularRTUAdaptDensity",
    "RectangularRTUAdaptImage",
]
DELAUNAY_FAMILY = ["Delaunay", "DelaunayNN", "KNearestNeighbor", "KNNBarycentric"]


def _gaussian(grid):
    return np.exp(
        -((grid[:, 0] - GAUSS_CENTRE[0]) ** 2 + (grid[:, 1] - GAUSS_CENTRE[1]) ** 2)
        / (2.0 * GAUSS_SIGMA**2)
    )


def _vertices():
    """
    A jittered 12x12 lattice spanning the image box exactly (the outer ring is not
    jittered), so the Delaunay hull is the whole image box.
    """
    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 1.0, MESH_SHAPE[0])
    yy, xx = np.meshgrid(t, t, indexing="ij")
    nodes = np.stack([yy.ravel(), xx.ravel()], axis=1)
    interior = ((nodes > 0.0) & (nodes < 1.0)).all(axis=1)
    jitter = rng.uniform(-0.3, 0.3, nodes.shape) / (MESH_SHAPE[0] - 1)
    nodes = nodes + jitter * interior[:, None]
    half = 0.5 * SHAPE * PIXEL_SCALE
    return -half + 2.0 * half * nodes


def _mesh_from(name):
    if name.startswith("Rectangular"):
        return getattr(aa.mesh, name)(shape=MESH_SHAPE)
    return getattr(aa.mesh, name)(pixels=MESH_SHAPE[0] * MESH_SHAPE[1])


_CACHE = {}


def _setup(name):
    """
    The mapper, its areas, its exact quadrature `Q` and a Gaussian reconstruction,
    cached per mesh so the parametrized tests share one build.
    """
    if name in ("KNearestNeighbor", "KNNBarycentric"):
        # the KNN interpolators import jax (unittest-nojax leg)
        pytest.importorskip("jax")

    if name in _CACHE:
        return _CACHE[name]

    grid = aa.Grid2D.uniform(
        shape_native=(SHAPE, SHAPE),
        pixel_scales=PIXEL_SCALE,
        over_sample_size=SUB_SIZE,
    )
    adapt_data = aa.Array2D(values=_gaussian(grid.array) + 0.01, mask=grid.mask)

    interpolator = _mesh_from(name).interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(_vertices()),
        adapt_data=adapt_data,
    )
    mapper = aa.Mapper(interpolator=interpolator)

    areas = np.asarray(mapper.mesh_geometry.areas_for_magnification)

    mapping_matrix = np.asarray(mapper.mapping_matrix_over_sampled)
    sub_area = (PIXEL_SCALE / SUB_SIZE) ** 2
    quadrature = mapping_matrix.sum(axis=0) * sub_area

    # Noise-free least-squares reconstruction of the Gaussian at the data sub-pixels
    # (a vanishing ridge only fixes the unmapped guard cells at zero).
    data = _gaussian(np.asarray(grid.over_sampled.array))
    curvature = mapping_matrix.T @ mapping_matrix
    ridge = 1.0e-8 * np.mean(np.diag(curvature)) * np.eye(curvature.shape[0])
    reconstruction = np.linalg.solve(curvature + ridge, mapping_matrix.T @ data)

    _CACHE[name] = (grid, mapper, areas, quadrature, reconstruction)
    return _CACHE[name]


@pytest.mark.parametrize("name", list(MAGNIFICATION_TOLERANCE))
def test__areas_sum_to_the_covered_domain(name):
    grid, mapper, areas, _, _ = _setup(name)

    assert np.all(np.isfinite(areas))
    assert np.all(areas >= 0.0)
    assert areas.shape == (mapper.params,)

    if name in DELAUNAY_FAMILY:
        vertices = np.asarray(mapper.mesh_geometry.mesh_grid_xy)
        domain = scipy.spatial.ConvexHull(vertices).volume
        rel = 1.0e-10
    elif name == "RectangularUniform":
        # the overlay grid spans the over-sampled data plus a 1e-8 buffer
        domain = np.prod(np.ptp(np.asarray(grid.over_sampled.array), axis=0))
        rel = 1.0e-6
    else:
        # the hats are integrated over the over-sampled data's bounding box and
        # partition unity on it
        domain = np.prod(np.ptp(np.asarray(grid.over_sampled.array), axis=0))
        rel = 1.0e-10

    assert areas.sum() == pytest.approx(domain, rel=rel)


@pytest.mark.parametrize("name", list(MAGNIFICATION_TOLERANCE))
def test__identity_lens_magnification_of_a_gaussian_source(name):
    _, _, areas, quadrature, reconstruction = _setup(name)

    source_flux = np.sum(reconstruction * areas)
    image_flux = np.sum(reconstruction * quadrature)

    tol = MAGNIFICATION_TOLERANCE[name]

    assert image_flux / source_flux == pytest.approx(1.0, abs=tol)

    # the integrated reconstruction against the analytic Gaussian integral (the
    # source sits well inside the image box, so its truncation is negligible)
    analytic = 2.0 * np.pi * GAUSS_SIGMA**2
    assert source_flux / analytic == pytest.approx(1.0, abs=tol)


@pytest.mark.parametrize("name", RECTANGULAR_ADAPTIVE)
def test__rectangular_adaptive__guard_cells_have_zero_area_and_no_flux(name):
    """
    The adaptive rectangular mapper is node-based with a guard ring in unit CDF
    space: rows 0 and 1 (above U_y = 1) and columns 0 and n - 1 (outside
    [0, 1] in U_x) never receive mapping weight, and their hats are never reached
    by the transformed coordinate so their area is exactly zero. Every other cell is live and
    carries a positive area — including the bottom row (U_y = 0), which is live
    although it sits in the `zeroed_pixels` edge ring.
    """
    _, mapper, areas, quadrature, _ = _setup(name)

    n_y, n_x = MESH_SHAPE
    mapped = quadrature.reshape(n_y, n_x) > 0.0
    area = areas.reshape(n_y, n_x)

    guard = np.zeros((n_y, n_x), dtype=bool)
    guard[:2, :] = True
    guard[:, 0] = True
    guard[:, -1] = True

    np.testing.assert_array_equal(mapped, ~guard)
    np.testing.assert_array_equal(area[guard], 0.0)
    assert np.all(area[~guard] > 0.0)


def test__rectangular_uniform__every_cell_is_one_node_spacing():
    """
    `RectangularUniform` interpolates between the overlay grid's pixel centres with
    no CDF and no guard ring: every cell, edge cells included, is one node spacing
    on a side and receives mapping weight.
    """
    _, mapper, areas, quadrature, _ = _setup("RectangularUniform")

    mesh_grid = np.asarray(mapper.mesh_geometry.mesh_grid.array)
    spacing = np.ptp(mesh_grid, axis=0) / (np.array(MESH_SHAPE) - 1)

    np.testing.assert_allclose(areas, np.prod(spacing), rtol=1.0e-10)
    assert np.all(quadrature > 0.0)

    edges = np.asarray(mapper.mesh_geometry.edges_transformed)
    # y edges descend from the top of the overlay box, x edges ascend
    np.testing.assert_allclose(np.diff(edges[:, 0]), -spacing[0], rtol=1.0e-10)
    np.testing.assert_allclose(np.diff(edges[:, 1]), spacing[1], rtol=1.0e-10)
    assert edges[0, 0] == pytest.approx(mesh_grid[:, 0].max() + 0.5 * spacing[0])
    assert edges[0, 1] == pytest.approx(mesh_grid[:, 1].min() - 0.5 * spacing[1])


@pytest.mark.parametrize("name", DELAUNAY_FAMILY)
def test__delaunay_family__hull_vertices_keep_their_area_and_zeroing_does_not_change_it(
    name,
):
    """
    Every vertex, hull vertices included, carries a positive barycentric dual area
    (the old Voronoi definition zeroed unbounded hull cells). The areas are a
    property of the vertices alone: zeroing the last vertices (`zeroed_pixels`, the
    appended edge ring) fixes their reconstruction to zero but does not change any
    area.
    """
    grid, mapper, areas, _, _ = _setup(name)

    assert np.all(areas > 0.0)

    mesh = getattr(aa.mesh, name)(
        pixels=MESH_SHAPE[0] * MESH_SHAPE[1] - 10, zeroed_pixels=10
    )
    zeroed_mapper = aa.Mapper(
        interpolator=mesh.interpolator_from(
            source_plane_data_grid=grid,
            source_plane_mesh_grid=aa.Grid2DIrregular(_vertices()),
        )
    )

    np.testing.assert_allclose(
        np.asarray(zeroed_mapper.mesh_geometry.areas_for_magnification),
        areas,
        rtol=1.0e-12,
    )
