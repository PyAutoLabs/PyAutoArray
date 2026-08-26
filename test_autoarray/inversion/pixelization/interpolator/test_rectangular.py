"""Unit tests for the adaptive rectangular interpolator (rank and
kernel-density CDF transforms).

Pure numpy — no JAX imports here. Cross-xp / gradient certification lives in
autolens_workspace_test/scripts/jax_grad per the project's "no JAX in unit
tests" rule.
"""

import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.mesh.interpolator.rectangular import (
    KERNEL_CDF_DEFAULT_KNOTS,
    KERNEL_FORWARD_BLOCK,
    InterpolatorRectangular,
    adaptive_rectangular_areas_from,
    adaptive_rectangular_mappings_weights_via_interpolation_from,
    create_transforms,
    create_transforms_rank,
)


def _seeded_inputs(M=128, K=400, seed=0):
    rng = np.random.default_rng(seed)
    data_grid = rng.standard_normal((M, 2))
    data_grid_over = rng.standard_normal((K, 2)) * 0.8
    weights = rng.uniform(0.1, 1.0, size=M)
    weights = weights / weights.sum()
    return data_grid, data_grid_over, weights


# ---------------------------------------------------------------------------
# CDF transform properties
# ---------------------------------------------------------------------------


def test__create_transforms__strictly_monotone_in_queries():
    data_grid, _, weights = _seeded_inputs(seed=1)

    fwd, _ = create_transforms(
        data_grid, mesh_pixels=16, mesh_weight_map=weights, xp=np
    )

    # Dense interior queries per axis, strictly increasing.
    q = np.linspace(data_grid.min(axis=0), data_grid.max(axis=0), 500)
    F = fwd(q)
    assert np.all(np.diff(F, axis=0) > 0.0)


def test__create_transforms__fwd_maps_data_range_to_unit_square():
    data_grid, _, weights = _seeded_inputs(seed=3)

    fwd, _ = create_transforms(
        data_grid, mesh_pixels=16, mesh_weight_map=weights, xp=np
    )

    F = fwd(data_grid)
    assert F.min() >= 0.0
    assert F.max() <= 1.0

    # The data bounding box corners map exactly onto the unit square corners.
    lo = data_grid.min(axis=0)
    hi = data_grid.max(axis=0)
    corners = fwd(np.stack([lo, hi]))
    assert corners[0] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert corners[1] == pytest.approx([1.0, 1.0], abs=1e-12)


def test__create_transforms__roundtrip_matches_identity():
    data_grid, _, weights = _seeded_inputs(seed=1)

    fwd, rev = create_transforms(
        data_grid, mesh_pixels=16, mesh_weight_map=weights, n_knots=512, xp=np
    )

    probe = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    roundtrip = fwd(rev(probe))
    assert roundtrip == pytest.approx(probe, abs=1e-4)


def test__create_transforms__unweighted_roundtrip_default_knots():
    data_grid, _, _ = _seeded_inputs(seed=2)

    fwd, rev = create_transforms(data_grid, mesh_pixels=16, xp=np)

    probe = np.array([[0.25, 0.3], [0.5, 0.5], [0.7, 0.75]])
    roundtrip = fwd(rev(probe))
    assert roundtrip == pytest.approx(probe, abs=2e-3)


def test__create_transforms__duplicate_points_are_safe():
    """Coincident traced points must not produce NaN/inf or break
    monotonicity — an empirical CDF's 1/Δknot failure mode."""
    rng = np.random.default_rng(4)
    base = rng.standard_normal((32, 2))
    data_grid = np.concatenate([base, base[:8], base[:1].repeat(4, axis=0)])

    fwd, rev = create_transforms(data_grid, mesh_pixels=8, xp=np)

    q = np.linspace(data_grid.min(axis=0), data_grid.max(axis=0), 200)
    F = fwd(q)
    assert np.all(np.isfinite(F))
    assert np.all(np.diff(F, axis=0) > 0.0)

    probe = np.array([[0.2, 0.4], [0.6, 0.8]])
    assert np.all(np.isfinite(rev(probe)))


def test__create_transforms__large_bandwidth_tends_to_uniform():
    """As h → ∞ the kernel CDF linearises, so the transform approaches the
    plain affine map of the data bounding box onto the unit square."""
    data_grid, _, _ = _seeded_inputs(seed=5)

    fwd, _ = create_transforms(data_grid, mesh_pixels=16, bandwidth=1000.0, xp=np)

    lo = data_grid.min(axis=0)
    hi = data_grid.max(axis=0)
    q = np.linspace(lo, hi, 50)
    expected = (q - lo) / (hi - lo)
    assert fwd(q) == pytest.approx(expected, abs=1e-4)


def test__create_transforms__chunked_forward_is_block_size_invariant():
    """The forward transform blocks the query axis (KERNEL_FORWARD_BLOCK) to
    cap peak memory; results must be identical to evaluating queries one at a
    time — exercised across a query count spanning multiple blocks and not a
    multiple of the block size."""
    rng = np.random.default_rng(7)
    data_grid = rng.standard_normal((77, 2))

    fwd, _ = create_transforms(data_grid, mesh_pixels=8, xp=np)

    q = rng.standard_normal((KERNEL_FORWARD_BLOCK + 137, 2))
    batched = fwd(q)
    row_wise = np.vstack([fwd(q[i : i + 1]) for i in range(q.shape[0])])

    assert batched.shape == (KERNEL_FORWARD_BLOCK + 137, 2)
    assert batched == pytest.approx(row_wise, abs=0.0)


# ---------------------------------------------------------------------------
# Empirical rank-CDF transform (the Bilinear meshes)
# ---------------------------------------------------------------------------


def test__create_transforms_rank__unweighted_cdf_is_ranks_at_points():
    data_grid, _, _ = _seeded_inputs(seed=1)
    N = data_grid.shape[0]

    fwd, _ = create_transforms_rank(data_grid, xp=np)

    # At the sorted points themselves the empirical CDF is exactly the rank
    # values (i + 1) / (N + 1), per axis.
    sort_points = np.sort(data_grid, axis=0)
    expected = np.arange(1, N + 1) / (N + 1)
    F = fwd(sort_points)
    assert F[:, 0] == pytest.approx(expected, abs=1e-12)
    assert F[:, 1] == pytest.approx(expected, abs=1e-12)


def test__create_transforms_rank__weighted_cdf_is_cumsum_at_points():
    data_grid, _, weights = _seeded_inputs(seed=2)

    fwd, _ = create_transforms_rank(data_grid, mesh_weight_map=weights, xp=np)

    for d in range(2):
        order = np.argsort(data_grid[:, d])
        expected = np.cumsum(weights[order])
        F = fwd(np.sort(data_grid, axis=0))
        assert F[:, d] == pytest.approx(expected, abs=1e-12)


def test__create_transforms_rank__monotone_and_bounded():
    data_grid, _, weights = _seeded_inputs(seed=3)

    fwd, _ = create_transforms_rank(data_grid, mesh_weight_map=weights, xp=np)

    q = np.linspace(data_grid.min(axis=0) - 0.5, data_grid.max(axis=0) + 0.5, 500)
    F = fwd(q)
    assert np.all(np.diff(F, axis=0) >= 0.0)
    assert F.min() >= 0.0
    assert F.max() <= 1.0


def test__create_transforms_rank__roundtrip_matches_identity():
    data_grid, _, _ = _seeded_inputs(seed=4)
    N = data_grid.shape[0]

    fwd, rev = create_transforms_rank(data_grid, xp=np)

    # Interior unit-square probes (inside [1/(N+1), N/(N+1)] where the
    # piecewise-linear CDF is invertible) round-trip exactly.
    probe = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    assert 0.1 > 1.0 / (N + 1) and 0.9 < N / (N + 1)
    roundtrip = fwd(rev(probe))
    assert roundtrip == pytest.approx(probe, abs=1e-12)


def test__rank__mappings_sizes_weights__shapes_and_weight_normalization():
    data_grid, over, weights = _seeded_inputs()

    idx, w = adaptive_rectangular_mappings_weights_via_interpolation_from(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        transform="rank",
        xp=np,
    )

    assert idx.shape == (400, 4)
    assert w.shape == (400, 4)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-10)


def test__rank__areas__positive_finite__total_is_bounding_box_area():
    data_grid, _, weights = _seeded_inputs(seed=6)

    areas = adaptive_rectangular_areas_from(
        source_grid_shape=(12, 12),
        data_grid=data_grid,
        mesh_weight_map=weights,
        transform="rank",
        xp=np,
    )

    assert areas.shape == (144,)
    assert np.all(np.isfinite(areas))
    assert np.all(areas > 0.0)
    span = data_grid.max(axis=0) - data_grid.min(axis=0)
    assert areas.sum() == pytest.approx(span[0] * span[1], rel=1e-8)


def test__rank__areas__adapt_to_point_density():
    """A dense cluster of traced points must shrink the mesh pixels covering
    it relative to a sparse region — the adaptive property the rank CDF
    exists to provide."""
    rng = np.random.default_rng(11)
    cluster = rng.normal(loc=-1.0, scale=0.05, size=(400, 2))
    sparse = rng.uniform(low=-2.0, high=2.0, size=(100, 2))
    data_grid = np.concatenate([cluster, sparse])

    areas = adaptive_rectangular_areas_from(
        source_grid_shape=(10, 10),
        data_grid=data_grid,
        transform="rank",
        xp=np,
    )

    # Strong adaptivity: the cluster holds 80% of the rank mass, so the cells
    # covering it shrink by orders of magnitude relative to the sparse
    # outskirts. A uniform lattice would have every cell equal.
    assert areas.min() < areas.max() / 100.0

    # The smallest cells cover the cluster: every cell of side <~ 4 sigma
    # contains cluster points, so its area is far below the uniform cell area.
    uniform_cell = areas.sum() / areas.size
    assert areas.min() < uniform_cell / 100.0


def test__transforms_from__invalid_transform_raises():
    data_grid, over, _ = _seeded_inputs()

    with pytest.raises(ValueError):
        adaptive_rectangular_mappings_weights_via_interpolation_from(
            source_grid_size=16,
            data_grid=data_grid,
            data_grid_over_sampled=over,
            transform="spline",
            xp=np,
        )


# ---------------------------------------------------------------------------
# Mapper output shapes
# ---------------------------------------------------------------------------


def test__mappings_sizes_weights__shapes_and_weight_normalization():
    data_grid, over, weights = _seeded_inputs()

    idx, w = adaptive_rectangular_mappings_weights_via_interpolation_from(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        xp=np,
    )

    assert idx.shape == (400, 4)
    assert w.shape == (400, 4)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-10)


def _index_space_nodes(idx, n):
    """
    Recover each mapped pixel's (row, col) position in index space.

    Inverse of the module's ``flatten(ix, iy) = (n - ix) * n + iy``.
    """
    return n - idx // n, idx % n


def test__mappings_sizes_weights__reproduces_the_query_position():
    """
    Bilinear interpolation must be exact for linear functions, which means the
    four weights have to reconstruct the query itself:

        sum_i w_i * node_i == grid_over_index

    Partition of unity alone does NOT imply this — it is satisfied by any
    consistent mis-pairing of corners to weights, which is exactly how the row
    weights came to be mirrored (`ix_up` carrying `1 - t_row` instead of
    `t_row`) and stayed that way from 2025-09-23 to 2026-08-26. The mirroring
    was smooth, so gradient checks passed; only this property catches it.
    See autolens_workspace_test#279.
    """
    n = 16
    data_grid, over, weights = _seeded_inputs(seed=11)

    idx, w = adaptive_rectangular_mappings_weights_via_interpolation_from(
        source_grid_size=n,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        xp=np,
    )

    # Rebuild the index-space query the function discretises internally.
    mu, scale = data_grid.mean(axis=0), data_grid.std(axis=0)
    transform_func, _ = create_transforms(
        (data_grid - mu) / scale, mesh_pixels=n, mesh_weight_map=weights, xp=np
    )
    grid_over_index = (n - 3) * transform_func((over - mu) / scale) + 1

    node_row, node_col = _index_space_nodes(idx, n)

    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose((w * node_row).sum(axis=1), grid_over_index[:, 0], atol=1e-10)
    assert np.allclose((w * node_col).sum(axis=1), grid_over_index[:, 1], atol=1e-10)


def test__mappings_sizes_weights__cell_assignment_is_continuous_at_integers():
    """
    ``transform()`` ends in ``clip(F_q, 0.0, 1.0)``, so saturated queries land on
    EXACTLY integer ``grid_over_index`` values — systematically, not by chance.
    Bracketing those with ``ceil`` collapsed the cell (``ix_up == ix_down``), so
    a 1-ULP move off the plateau jumped a point's weight a whole mesh row. That
    made the likelihood depend on floating-point association, which is how the
    eager and jitted evaluations came to disagree by ~1.6e-3.

    Here the property is asserted directly on the interpolated value of a linear
    ramp: approaching an integer row coordinate from either side must converge
    to the value AT that coordinate.
    """
    n = 16
    data_grid, _, weights = _seeded_inputs(seed=12)
    mu, scale = data_grid.mean(axis=0), data_grid.std(axis=0)

    transform_func, inv = create_transforms(
        (data_grid - mu) / scale, mesh_pixels=n, mesh_weight_map=weights, xp=np
    )

    def interpolated_ramp(over):
        idx, w = adaptive_rectangular_mappings_weights_via_interpolation_from(
            source_grid_size=n,
            data_grid=data_grid,
            data_grid_over_sampled=over,
            mesh_weight_map=weights,
            xp=np,
        )
        node_row, node_col = _index_space_nodes(idx, n)
        # A linear function of position; bilinear interpolation reproduces it
        # exactly, so any discontinuity here is a cell-assignment jump.
        return (w * (3.0 * node_row - 2.0 * node_col)).sum(axis=1)

    # Sweep the row coordinate densely across the whole data range. In index
    # space that spans [1, n - 2], so the sweep crosses every interior integer
    # boundary; a jump at any crossing is a cell-assignment discontinuity.
    # This is deliberately placement-free — the inverse transform is a knot
    # lookup and cannot land a query on an integer precisely enough to probe
    # one boundary directly.
    lo, hi = data_grid[:, 0].min(), data_grid[:, 0].max()
    sweep = np.stack(
        [np.linspace(lo, hi, 20001), np.full(20001, np.median(data_grid[:, 1]))],
        axis=1,
    )

    values = interpolated_ramp(sweep)
    steps = np.abs(np.diff(values))

    # The ramp must actually vary, or continuity is vacuous.
    assert values.max() - values.min() > 1.0

    # A whole-row flip moves the ramp by ~3.0 (its row coefficient); a
    # continuous scheme moves by ~the sweep resolution.
    assert steps.max() < 0.05


# ---------------------------------------------------------------------------
# Areas
# ---------------------------------------------------------------------------


def test__areas__positive_finite__total_is_bounding_box_area():
    data_grid, _, weights = _seeded_inputs(seed=6)

    areas = adaptive_rectangular_areas_from(
        source_grid_shape=(12, 12),
        data_grid=data_grid,
        mesh_weight_map=weights,
        xp=np,
    )

    assert areas.shape == (144,)
    assert np.all(np.isfinite(areas))
    assert np.all(areas > 0.0)
    # The per-axis edge differences telescope across the unit square, which
    # maps exactly onto the data bounding box — so the areas sum to its area.
    span = data_grid.max(axis=0) - data_grid.min(axis=0)
    assert areas.sum() == pytest.approx(span[0] * span[1], rel=1e-8)


# ---------------------------------------------------------------------------
# Interpolator construction — exercise the class directly without the full
# BorderRelocator / Grid2D pipeline (covered by the jax_grad certification).
# ---------------------------------------------------------------------------


def test__InterpolatorRectangular__mappings_sizes_weights_via_property():
    class _StubGrid:
        def __init__(self, arr):
            self.array = arr
            self.over_sampled = self
            self._array = arr

        def __getattr__(self, item):
            return getattr(self._array, item)

    rng = np.random.default_rng(5)
    data_grid = _StubGrid(rng.standard_normal((64, 2)))

    mesh = aa.mesh.RectangularRTUAdaptDensity(shape=(6, 6), bandwidth=0.8)
    interpolator = InterpolatorRectangular(
        mesh=mesh,
        mesh_grid=_StubGrid(rng.standard_normal((36, 2))),
        data_grid=data_grid,
        mesh_weight_map=None,
        bandwidth=0.8,
        xp=np,
    )

    mappings, sizes, weights = interpolator._mappings_sizes_weights
    assert mappings.shape == (64, 4)
    assert sizes.shape == (64,)
    assert weights.shape == (64, 4)
    assert np.all(sizes == 4)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)

    geometry = interpolator.mesh_geometry
    assert geometry.kernel_bandwidth == 0.8
    assert geometry.kernel_knots == KERNEL_CDF_DEFAULT_KNOTS
    assert geometry.transform == "kernel"

    areas = geometry.areas_transformed
    assert areas.shape == (36,)
    assert np.all(np.isfinite(areas))
    assert np.all(areas > 0.0)


def test__InterpolatorRectangular__rank_transform_via_property():
    class _StubGrid:
        def __init__(self, arr):
            self.array = arr
            self.over_sampled = self
            self._array = arr

        def __getattr__(self, item):
            return getattr(self._array, item)

    rng = np.random.default_rng(5)
    data_grid = _StubGrid(rng.standard_normal((64, 2)))

    mesh = aa.mesh.RectangularBilinearAdaptDensity(shape=(6, 6))
    interpolator = InterpolatorRectangular(
        mesh=mesh,
        mesh_grid=_StubGrid(rng.standard_normal((36, 2))),
        data_grid=data_grid,
        mesh_weight_map=None,
        **mesh.interpolator_kwargs,
        xp=np,
    )

    assert interpolator.transform == "rank"

    mappings, sizes, weights = interpolator._mappings_sizes_weights
    assert mappings.shape == (64, 4)
    assert sizes.shape == (64,)
    assert weights.shape == (64, 4)
    assert np.all(sizes == 4)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)

    # The geometry routes areas / edges through the same rank transform as
    # the mapper, so the two stay consistent.
    geometry = interpolator.mesh_geometry
    assert geometry.transform == "rank"

    areas = geometry.areas_transformed
    assert areas.shape == (36,)
    assert np.all(np.isfinite(areas))
    assert np.all(areas > 0.0)


# ---------------------------------------------------------------------------
# Windowed numba fast path (numpy branch)
# ---------------------------------------------------------------------------


def _dense_reference_forward(data_grid, mesh_pixels, weights, q):
    """The pre-fast-path definition: dense O(M x N) normal-CDF sum, rescaled
    so the data bounding box maps onto the unit square, clipped to [0, 1]."""
    from scipy.special import erf

    points = data_grid
    N = points.shape[0]
    w = np.full(N, 1.0 / N) if weights is None else weights / weights.sum()
    lo, hi = points.min(axis=0), points.max(axis=0)
    h = 1.0 * (hi - lo) / mesh_pixels

    def F_raw(qq):
        t = (qq[:, None, :] - points[None, :, :]) / h[None, None, :]
        return np.sum(w[None, :, None] * (0.5 * (1.0 + erf(t / np.sqrt(2.0)))), axis=1)

    F_lo = F_raw(lo[None, :])[0]
    F_hi = F_raw(hi[None, :])[0]
    return np.clip((F_raw(q) - F_lo[None, :]) / (F_hi - F_lo)[None, :], 0.0, 1.0)


@pytest.mark.parametrize("weighted", [False, True])
def test__forward_transform__windowed_numba_matches_dense_reference(weighted):
    data_grid, data_grid_over, weights = _seeded_inputs(M=300, K=500, seed=7)

    # queries beyond the data bounding box exercise the saturated tails
    q = np.concatenate(
        [
            data_grid_over,
            data_grid.min(axis=0) - 1.0 + np.zeros((1, 2)),
            data_grid.max(axis=0) + 1.0 + np.zeros((1, 2)),
        ]
    )

    fwd, _ = create_transforms(
        data_grid, mesh_pixels=16, mesh_weight_map=weights if weighted else None, xp=np
    )

    reference = _dense_reference_forward(
        data_grid, 16, weights if weighted else None, q
    )

    np.testing.assert_allclose(fwd(q), reference, rtol=0.0, atol=1e-12)
