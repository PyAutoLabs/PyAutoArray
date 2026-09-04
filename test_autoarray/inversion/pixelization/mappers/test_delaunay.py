import importlib.util

import numpy as np
import pytest
import scipy.spatial

import autoarray as aa

from autoarray.inversion.mesh.interpolator.delaunay import (
    pix_indexes_for_sub_slim_index_delaunay_from,
)


from autoarray.inversion.mesh.interpolator.delaunay import (
    pixel_weights_delaunay_from,
)


def test__pixel_weights_delaunay_from__two_data_points__returns_correct_barycentric_weights():
    data_grid = np.array([[0.1, 0.1], [1.0, 1.0]])

    mesh_grid = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])

    pix_indexes_for_sub_slim_index = np.array([[0, 1, 2], [2, -1, -1]])

    pixel_weights = pixel_weights_delaunay_from(
        data_grid=data_grid,
        mesh_grid=mesh_grid,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
    )

    assert (pixel_weights == np.array([[0.25, 0.5, 0.25], [1.0, 0.0, 0.0]])).all()


def test__pix_indexes_for_sub_slim_index__delaunay_mesh__matches_util_and_expected_values(
    grid_2d_sub_1_7x7,
):
    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    mesh = aa.mesh.Delaunay(pixels=6)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid_2d_sub_1_7x7,
        source_plane_mesh_grid=mesh_grid,
    )

    mapper = aa.Mapper(interpolator=interpolator)

    delaunay = scipy.spatial.Delaunay(mapper.interpolator.mesh_grid_xy)

    simplex_index_for_sub_slim_index = delaunay.find_simplex(
        mapper.source_plane_data_grid
    )

    delaunay = mapper.interpolator.delaunay

    pix_indexes_for_simplex_index = delaunay.simplices

    pix_indexes_for_sub_slim_index_util = pix_indexes_for_sub_slim_index_delaunay_from(
        data_grid=mapper.source_plane_data_grid.array,
        simplex_index_for_sub_slim_index=simplex_index_for_sub_slim_index,
        pix_indexes_for_simplex_index=pix_indexes_for_simplex_index,
        delaunay_points=delaunay.points,
    )
    sizes = (
        np.sum(pix_indexes_for_sub_slim_index_util >= 0, axis=1)
        .astype(np.int32)
        .astype("int")
    )

    assert (
        mapper.pix_indexes_for_sub_slim_index == pix_indexes_for_sub_slim_index_util
    ).all()
    assert (mapper.pix_sizes_for_sub_slim_index == sizes).all()

    assert (
        mapper.pix_indexes_for_sub_slim_index
        == np.array(
            [
                [0, -1, -1],
                [1, -1, -1],
                [1, 5, 3],
                [0, -1, -1],
                [0, -1, -1],
                [3, -1, -1],
                [0, -1, -1],
                [0, -1, -1],
                [3, -1, -1],
            ]
        )
    ).all()

    assert (
        mapper.pix_sizes_for_sub_slim_index == np.array([1, 1, 3, 1, 1, 1, 1, 1, 1])
    ).all()


# ----------------------------------------------------------------------------
# Magnification quadrature (PyAutoArray#524).
#
# `mesh_geometry.areas_for_magnification` returns the barycentric dual areas,
# which are the exact quadrature weights of the mapper's barycentric-linear
# reconstruction. The identity test below pins that against the mapping matrix
# itself -- the mapper's own definition of how flux is spread over the mesh.
# ----------------------------------------------------------------------------

# jax is an `[optional]` extra and is absent on the NumPy-only matrix env, so
# the JAX parity test skips rather than fails there (same convention as
# test_knn_barycentric.py).
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)


def _magnification_identity_setup():
    """A Delaunay mapper whose mesh hull is pinned to the data footprint.

    The data grid's pixel centres span +/- 0.95 at a pixel scale of 0.1, so its
    footprint is exactly [-1, 1] ** 2 (area 4). The mesh's outer ring sits on
    +/- 1.0, so every data pixel centre is inside the hull (no out-of-hull
    nearest-vertex fallback) and the hull area is exactly the footprint area --
    the two integrals below therefore cover the same region.

    Interior mesh points are jittered by <= 0.3 of the lattice spacing with a
    seeded rng, so the triangulation is irregular rather than a regular lattice.
    """
    data_grid = aa.Grid2D.uniform(
        shape_native=(20, 20), pixel_scales=0.1, over_sample_size=1
    )

    n = 8
    axis = np.linspace(-1.0, 1.0, n)
    spacing = axis[1] - axis[0]

    mesh_y, mesh_x = np.meshgrid(axis, axis, indexing="ij")
    mesh_points = np.stack([mesh_y.ravel(), mesh_x.ravel()], axis=1)

    rng = np.random.default_rng(42)

    interior = (np.abs(mesh_points[:, 0]) < 1.0 - 1.0e-9) & (
        np.abs(mesh_points[:, 1]) < 1.0 - 1.0e-9
    )
    mesh_points[interior] += (
        rng.uniform(-0.3, 0.3, size=(int(interior.sum()), 2)) * spacing
    )

    reconstruction = rng.uniform(0.5, 2.0, size=mesh_points.shape[0])

    return data_grid, mesh_points, reconstruction


def test__areas_for_magnification__integrates_reconstruction_like_mapping_matrix():
    """
    The mapping matrix and the dual areas are two routes to the same integral of
    the reconstruction over the source plane:

      * `(mapping_matrix @ s).sum() * pixel_scale ** 2` -- a Riemann sum of the
        interpolant over the data pixels, using the mapper's own weights.
      * `(s * areas_for_magnification).sum()` -- the exact quadrature of the same
        piecewise-linear interpolant over the mesh hull.

    With the hull pinned to the data footprint they agree to the half-pixel
    border error (~3e-5 here). The Voronoi areas -- what
    `areas_for_magnification` returned before PyAutoArray#524 -- get the same
    integral wrong by ~27%, which is asserted too so the flipped semantics are
    documented by a number.
    """
    import scipy.spatial

    data_grid, mesh_points, reconstruction = _magnification_identity_setup()

    mesh = aa.mesh.Delaunay(pixels=mesh_points.shape[0])

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=data_grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(mesh_points),
    )

    mapper = aa.Mapper(interpolator=interpolator)

    # no data pixel took the out-of-hull nearest-vertex fallback
    assert (np.asarray(mapper.interpolator.delaunay.mappings)[:, 1] >= 0).all()

    areas = np.asarray(mapper.mesh_geometry.areas_for_magnification)

    hull_area = scipy.spatial.ConvexHull(mesh_points).volume

    assert areas.sum() == pytest.approx(hull_area, rel=1.0e-10)
    assert hull_area == pytest.approx(4.0, rel=1.0e-10)

    f_map = (np.asarray(mapper.mapping_matrix) @ reconstruction).sum() * 0.1**2
    f_dual = (reconstruction * areas).sum()

    assert f_map == pytest.approx(f_dual, rel=1.0e-4)

    voronoi = np.asarray(mapper.mesh_geometry.voronoi_areas)
    voronoi = np.where(voronoi == -1.0, 0.0, voronoi)

    f_voronoi = (reconstruction * voronoi).sum()

    assert abs(f_voronoi - f_map) / f_map > 1.0e-2, (
        f"Voronoi-weighted integral {f_voronoi} vs mapping-matrix integral "
        f"{f_map}; the two area definitions are not interchangeable."
    )


@requires_jax
def test__areas_for_magnification__jax_matches_numpy():
    """
    PyAutoLens evaluates latents inside a per-sample `jax.jit`, so
    `areas_for_magnification` must come from the interpolator's in-graph dual
    areas rather than a host-side scipy call. Building the same mesh geometry on
    both backends and comparing pins that.
    """
    import jax.numpy as jnp

    data_grid, mesh_points, _ = _magnification_identity_setup()

    mesh = aa.mesh.Delaunay(pixels=mesh_points.shape[0])

    numpy_areas = mesh.interpolator_from(
        source_plane_data_grid=data_grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(mesh_points),
    ).mesh_geometry.areas_for_magnification

    jax_areas = mesh.interpolator_from(
        source_plane_data_grid=data_grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(mesh_points),
        xp=jnp,
    ).mesh_geometry.areas_for_magnification

    assert np.asarray(jax_areas) == pytest.approx(np.asarray(numpy_areas), abs=1.0e-10)
