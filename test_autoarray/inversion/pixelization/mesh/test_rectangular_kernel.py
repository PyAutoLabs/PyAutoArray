"""Unit tests for the kernel-density-CDF rectangular meshes.

Pure numpy — no JAX imports here. Cross-xp / gradient certification lives in
autolens_workspace_test/scripts/jax_grad per the project's "no JAX in unit
tests" rule.
"""
import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.mesh.interpolator.rectangular_kernel import (
    KERNEL_CDF_DEFAULT_BANDWIDTH,
    KERNEL_CDF_DEFAULT_KNOTS,
    InterpolatorRectangularKernel,
    adaptive_rectangular_areas_from_kernel,
    adaptive_rectangular_mappings_weights_via_interpolation_from_kernel,
    create_transforms_kernel,
)
from autoarray.inversion.mesh.interpolator.rectangular import (
    adaptive_rectangular_mappings_weights_via_interpolation_from,
)
from autoarray.inversion.mesh.mesh_geometry.rectangular import (
    adaptive_rectangular_areas_from,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test__construction__shape_and_kernel_kwargs__default():
    density = aa.mesh.RectangularKernelAdaptDensity(shape=(5, 7))
    image = aa.mesh.RectangularKernelAdaptImage(shape=(5, 7))

    assert density.shape == (5, 7)
    assert density.bandwidth == KERNEL_CDF_DEFAULT_BANDWIDTH
    assert density.n_knots == KERNEL_CDF_DEFAULT_KNOTS
    assert image.shape == (5, 7)
    assert image.bandwidth == KERNEL_CDF_DEFAULT_BANDWIDTH
    assert image.n_knots == KERNEL_CDF_DEFAULT_KNOTS
    # inherited
    assert image.weight_power == 1.0
    assert image.weight_floor == 0.0


def test__construction__kernel_kwargs__overridden():
    density = aa.mesh.RectangularKernelAdaptDensity(
        shape=(3, 3), bandwidth=0.5, n_knots=128
    )
    image = aa.mesh.RectangularKernelAdaptImage(
        shape=(3, 3), weight_power=2.0, weight_floor=0.1, bandwidth=2.0, n_knots=32
    )

    assert density.bandwidth == 0.5
    assert density.n_knots == 128
    assert image.bandwidth == 2.0
    assert image.n_knots == 32
    assert image.weight_power == 2.0
    assert image.weight_floor == 0.1


def test__construction__minimum_shape_raises_same_as_parent():
    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularKernelAdaptDensity(shape=(2, 3))
    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularKernelAdaptImage(shape=(3, 2))


# ---------------------------------------------------------------------------
# Interpolator dispatch
# ---------------------------------------------------------------------------


def test__interpolator_cls__is_InterpolatorRectangularKernel():
    density = aa.mesh.RectangularKernelAdaptDensity(shape=(3, 3))
    image = aa.mesh.RectangularKernelAdaptImage(shape=(3, 3))

    assert density.interpolator_cls is InterpolatorRectangularKernel
    assert image.interpolator_cls is InterpolatorRectangularKernel


# ---------------------------------------------------------------------------
# mesh_weight_map_from inheritance
# ---------------------------------------------------------------------------


def test__mesh_weight_map_from__density__returns_none():
    density = aa.mesh.RectangularKernelAdaptDensity(shape=(3, 3))
    assert density.mesh_weight_map_from(adapt_data=None) is None


def test__mesh_weight_map_from__image__returns_weighted_normalized():
    image = aa.mesh.RectangularKernelAdaptImage(
        shape=(3, 3), weight_power=2.0, weight_floor=0.0
    )

    class _Stub:
        def __init__(self, arr):
            self.array = arr

    adapt = _Stub(np.array([1.0, 2.0, 4.0, 0.0, 8.0]))
    w = image.mesh_weight_map_from(adapt_data=adapt)

    expected = np.array([1.0, 4.0, 16.0, 1e-24, 64.0])
    expected = expected / expected.sum()
    assert w == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# CDF transform properties
# ---------------------------------------------------------------------------


def _seeded_inputs(M=128, K=400, seed=0):
    rng = np.random.default_rng(seed)
    data_grid = rng.standard_normal((M, 2))
    data_grid_over = rng.standard_normal((K, 2)) * 0.8
    weights = rng.uniform(0.1, 1.0, size=M)
    weights = weights / weights.sum()
    return data_grid, data_grid_over, weights


def test__create_transforms_kernel__strictly_monotone_in_queries():
    data_grid, _, weights = _seeded_inputs(seed=1)

    fwd, _ = create_transforms_kernel(
        data_grid, mesh_pixels=16, mesh_weight_map=weights, xp=np
    )

    # Dense interior queries per axis, strictly increasing.
    q = np.linspace(data_grid.min(axis=0), data_grid.max(axis=0), 500)
    F = fwd(q)
    assert np.all(np.diff(F, axis=0) > 0.0)


def test__create_transforms_kernel__fwd_maps_data_range_to_unit_square():
    data_grid, _, weights = _seeded_inputs(seed=3)

    fwd, _ = create_transforms_kernel(
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


def test__create_transforms_kernel__roundtrip_matches_identity():
    data_grid, _, weights = _seeded_inputs(seed=1)

    fwd, rev = create_transforms_kernel(
        data_grid, mesh_pixels=16, mesh_weight_map=weights, n_knots=512, xp=np
    )

    probe = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    roundtrip = fwd(rev(probe))
    assert roundtrip == pytest.approx(probe, abs=1e-4)


def test__create_transforms_kernel__unweighted_roundtrip_default_knots():
    data_grid, _, _ = _seeded_inputs(seed=2)

    fwd, rev = create_transforms_kernel(data_grid, mesh_pixels=16, xp=np)

    probe = np.array([[0.25, 0.3], [0.5, 0.5], [0.7, 0.75]])
    roundtrip = fwd(rev(probe))
    assert roundtrip == pytest.approx(probe, abs=2e-3)


def test__create_transforms_kernel__duplicate_points_are_safe():
    """Coincident traced points must not produce NaN/inf or break
    monotonicity — the empirical CDF's 1/Δknot failure mode."""
    rng = np.random.default_rng(4)
    base = rng.standard_normal((32, 2))
    data_grid = np.concatenate([base, base[:8], base[:1].repeat(4, axis=0)])

    fwd, rev = create_transforms_kernel(data_grid, mesh_pixels=8, xp=np)

    q = np.linspace(data_grid.min(axis=0), data_grid.max(axis=0), 200)
    F = fwd(q)
    assert np.all(np.isfinite(F))
    assert np.all(np.diff(F, axis=0) > 0.0)

    probe = np.array([[0.2, 0.4], [0.6, 0.8]])
    assert np.all(np.isfinite(rev(probe)))


def test__create_transforms_kernel__large_bandwidth_tends_to_uniform():
    """As h → ∞ the kernel CDF linearises, so the transform approaches the
    plain affine map of the data bounding box onto the unit square."""
    data_grid, _, _ = _seeded_inputs(seed=5)

    fwd, _ = create_transforms_kernel(
        data_grid, mesh_pixels=16, bandwidth=1000.0, xp=np
    )

    lo = data_grid.min(axis=0)
    hi = data_grid.max(axis=0)
    q = np.linspace(lo, hi, 50)
    expected = (q - lo) / (hi - lo)
    assert fwd(q) == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Mapper output shape + content consistency with linear variant
# ---------------------------------------------------------------------------


def test__mappings_sizes_weights__shapes_match_linear():
    data_grid, over, weights = _seeded_inputs()

    idx_k, w_k = adaptive_rectangular_mappings_weights_via_interpolation_from_kernel(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        xp=np,
    )
    idx_l, w_l = adaptive_rectangular_mappings_weights_via_interpolation_from(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        xp=np,
    )

    assert idx_k.shape == idx_l.shape == (400, 4)
    assert w_k.shape == w_l.shape == (400, 4)
    assert np.allclose(w_k.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(w_l.sum(axis=1), 1.0, atol=1e-10)


def test__mappings__flat_indices_mostly_match_linear():
    """Kernel and linear CDF route most points to the same cell — only points
    near cell boundaries can differ. Assert at least 70% exact match."""
    data_grid, over, weights = _seeded_inputs()

    idx_k, _ = adaptive_rectangular_mappings_weights_via_interpolation_from_kernel(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        xp=np,
    )
    idx_l, _ = adaptive_rectangular_mappings_weights_via_interpolation_from(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        xp=np,
    )

    exact_match_frac = float((idx_k == idx_l).mean())
    assert exact_match_frac >= 0.70, (
        f"kernel vs linear flat_index exact-match fraction = {exact_match_frac:.3f} "
        "(expected ≥ 0.70)"
    )


# ---------------------------------------------------------------------------
# Areas
# ---------------------------------------------------------------------------


def test__areas_from_kernel__positive_finite__total_matches_linear():
    data_grid, _, weights = _seeded_inputs(seed=6)

    areas_k = adaptive_rectangular_areas_from_kernel(
        source_grid_shape=(12, 12),
        data_grid=data_grid,
        mesh_weight_map=weights,
        xp=np,
    )
    areas_l = adaptive_rectangular_areas_from(
        source_grid_shape=(12, 12),
        data_grid=data_grid,
        mesh_weight_map=weights,
        xp=np,
    )

    assert areas_k.shape == (144,)
    assert np.all(np.isfinite(areas_k))
    assert np.all(areas_k > 0.0)
    # Both variants map the unit square onto the data bounding box, so the
    # per-axis edge differences telescope to the same span: totals agree.
    assert areas_k.sum() == pytest.approx(areas_l.sum(), rel=1e-8)


# ---------------------------------------------------------------------------
# Interpolator construction — exercise the class directly without the full
# BorderRelocator / Grid2D pipeline (covered by the jax_grad certification).
# ---------------------------------------------------------------------------


def test__InterpolatorRectangularKernel__mappings_sizes_weights_via_property():
    class _StubGrid:
        def __init__(self, arr):
            self.array = arr
            self.over_sampled = self
            self._array = arr

        def __getattr__(self, item):
            return getattr(self._array, item)

    rng = np.random.default_rng(5)
    data_grid = _StubGrid(rng.standard_normal((64, 2)))

    mesh = aa.mesh.RectangularKernelAdaptDensity(shape=(6, 6), bandwidth=0.8)
    interpolator = InterpolatorRectangularKernel(
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
