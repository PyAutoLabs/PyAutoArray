import importlib.util

import numpy as np
import pytest

from autoarray.inversion.mesh.interpolator.knn import (
    barycentric_weights_from_3_nearest,
)

# The kNN weight computation is jax-backed; these tests need jax installed to
# run (it ships via the `[optional]` extras). The NumPy-only Python-version
# matrix has no jax, so skip there rather than fail.
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)


def test__weights__interior_query_matches_delaunay_exactly():
    mesh = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    query = np.array([[0.25, 0.25]])
    indices = np.array([[0, 1, 2]])

    weights = barycentric_weights_from_3_nearest(
        query_points=query,
        mesh_points=mesh,
        nearest_3_indices=indices,
        xp=np,
    )

    assert weights[0] == pytest.approx([0.5, 0.25, 0.25], rel=1e-12)
    assert weights[0].sum() == pytest.approx(1.0, rel=1e-12)


def test__weights__query_on_edge_splits_50_50():
    mesh = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    query = np.array([[0.5, 0.5]])
    indices = np.array([[0, 1, 2]])

    weights = barycentric_weights_from_3_nearest(
        query_points=query,
        mesh_points=mesh,
        nearest_3_indices=indices,
        xp=np,
    )

    assert weights[0] == pytest.approx([0.0, 0.5, 0.5], abs=1e-12)


def test__weights__outside_query_clipped_and_renormalized():
    mesh = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    query = np.array([[-0.5, -0.5]])
    indices = np.array([[0, 1, 2]])

    weights = barycentric_weights_from_3_nearest(
        query_points=query,
        mesh_points=mesh,
        nearest_3_indices=indices,
        xp=np,
    )

    assert (weights[0] >= 0.0).all()
    assert weights[0].sum() == pytest.approx(1.0, rel=1e-12)
    # p0 is opposite both edges the query is "outside" of → keeps all weight
    assert weights[0, 0] == pytest.approx(1.0, rel=1e-12)
    assert weights[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert weights[0, 2] == pytest.approx(0.0, abs=1e-12)


def test__weights__degenerate_triangle_falls_back_to_nearest_neighbor():
    mesh = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    query = np.array([[1.0, 0.0]])
    # column 0 = closest by construction (set by get_interpolation_weights)
    indices = np.array([[0, 1, 2]])

    weights = barycentric_weights_from_3_nearest(
        query_points=query,
        mesh_points=mesh,
        nearest_3_indices=indices,
        xp=np,
    )

    assert np.isfinite(weights).all()
    # Degenerate → fall back to nearest-neighbor: [1, 0, 0]
    assert weights[0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-12)


def test__weights__matches_pixel_weights_delaunay_from__for_inside_triangle_grid():
    from autoarray.inversion.mesh.interpolator.delaunay import (
        pixel_weights_delaunay_from,
    )

    mesh = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rng = np.random.default_rng(seed=42)
    # 200 random barycentric-coordinate samples → guaranteed inside-triangle queries
    u = rng.uniform(0, 1, size=200)
    v = rng.uniform(0, 1, size=200)
    inside = u + v <= 1.0
    u, v = u[inside], v[inside]
    queries = np.stack([u, v], axis=1)  # (M, 2) — inside the unit-right triangle

    indices = np.tile(np.array([[0, 1, 2]]), (queries.shape[0], 1))

    bary = barycentric_weights_from_3_nearest(
        query_points=queries,
        mesh_points=mesh,
        nearest_3_indices=indices,
        xp=np,
    )

    delaunay = pixel_weights_delaunay_from(
        data_grid=queries,
        mesh_grid=mesh,
        pix_indexes_for_sub_slim_index=indices,
        xp=np,
    )

    assert bary == pytest.approx(delaunay, rel=1e-12, abs=1e-12)


def test__mesh_class__interpolator_cls_is_knn_barycentric():
    import autoarray as aa
    from autoarray.inversion.mesh.interpolator.knn import (
        InterpolatorKNNBarycentric,
    )

    mesh = aa.mesh.KNNBarycentric(pixels=10)
    assert mesh.interpolator_cls is InterpolatorKNNBarycentric


def test__mesh_class__inherits_knn_knobs():
    import autoarray as aa

    mesh = aa.mesh.KNNBarycentric(pixels=10)
    # Knobs inherited from KNearestNeighbor still control regularization spacing
    assert mesh.k_neighbors == 10
    assert mesh.radius_scale == 1.5
    assert mesh.areas_factor == 0.5
    assert mesh.split_neighbor_division == 2


@requires_jax
def test__interpolator__numpy_path_returns_writable_mappings_and_weights():
    """
    Regression guard: on the numpy path, `mappings` and `weights` must be
    writable numpy arrays — not read-only views of jax.Arrays. The
    `ConstantSplit` / `AdaptSplit` regularization code uses in-place
    assignment (e.g. `splitted_mappings[i][j+1] = ...` in `reg_split_np_from`),
    which raises `ValueError: assignment destination is read-only` if the
    array is a numpy view onto a jax.Array.
    """
    import autoarray as aa
    from autoarray.inversion.mesh.interpolator.knn import (
        InterpolatorKNNBarycentric,
    )

    mesh_grid = aa.Grid2D.no_mask(
        values=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [-0.5, -0.5]],
        shape_native=(3, 2),
        pixel_scales=1.0,
        over_sample_size=1,
    )

    class _Raw:
        def __init__(self, arr):
            self.array = arr

    queries = np.array([[0.3, 0.3], [0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])

    interp = InterpolatorKNNBarycentric(
        mesh=aa.mesh.KNNBarycentric(pixels=6),
        mesh_grid=mesh_grid,
        data_grid=_Raw(queries),
        xp=np,
    )
    mappings, _, weights = interp._mappings_sizes_weights

    assert isinstance(mappings, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert mappings.flags.writeable, "mappings must be writable (regularization mutates it)"
    assert weights.flags.writeable, "weights must be writable (regularization mutates it)"
