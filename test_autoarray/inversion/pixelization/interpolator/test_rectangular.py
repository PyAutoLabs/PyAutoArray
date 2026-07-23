"""Unit tests for the adaptive rectangular interpolator (kernel-density CDF).

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

    mesh = aa.mesh.RectangularAdaptDensity(shape=(6, 6), bandwidth=0.8)
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

    areas = geometry.areas_transformed
    assert areas.shape == (36,)
    assert np.all(np.isfinite(areas))
    assert np.all(areas > 0.0)
