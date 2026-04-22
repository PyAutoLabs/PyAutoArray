"""Unit tests for the spline-CDF rectangular meshes.

Pure numpy — no JAX imports here. Cross-xp tests live in
autolens_workspace_developer benchmarks per the project's
"no JAX in unit tests" rule.
"""
import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.mesh.interpolator.rectangular_spline import (
    SPLINE_CDF_DEFAULT_DEG,
    InterpolatorRectangularSpline,
    adaptive_rectangular_mappings_weights_via_interpolation_from_spline,
    create_transforms_spline,
)
from autoarray.inversion.mesh.interpolator.rectangular import (
    adaptive_rectangular_mappings_weights_via_interpolation_from,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test__construction__shape_and_spline_deg__default():
    density = aa.mesh.RectangularSplineAdaptDensity(shape=(5, 7))
    image = aa.mesh.RectangularSplineAdaptImage(shape=(5, 7))

    assert density.shape == (5, 7)
    assert density.spline_deg == SPLINE_CDF_DEFAULT_DEG
    assert image.shape == (5, 7)
    assert image.spline_deg == SPLINE_CDF_DEFAULT_DEG
    # inherited
    assert image.weight_power == 1.0
    assert image.weight_floor == 0.0


def test__construction__spline_deg__overridden():
    density = aa.mesh.RectangularSplineAdaptDensity(shape=(3, 3), spline_deg=9)
    image = aa.mesh.RectangularSplineAdaptImage(
        shape=(3, 3), weight_power=2.0, weight_floor=0.1, spline_deg=13
    )

    assert density.spline_deg == 9
    assert image.spline_deg == 13
    assert image.weight_power == 2.0
    assert image.weight_floor == 0.1


def test__construction__minimum_shape_raises_same_as_parent():
    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularSplineAdaptDensity(shape=(2, 3))
    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularSplineAdaptImage(shape=(3, 2))


# ---------------------------------------------------------------------------
# Interpolator dispatch
# ---------------------------------------------------------------------------


def test__interpolator_cls__is_InterpolatorRectangularSpline():
    density = aa.mesh.RectangularSplineAdaptDensity(shape=(3, 3))
    image = aa.mesh.RectangularSplineAdaptImage(shape=(3, 3))

    assert density.interpolator_cls is InterpolatorRectangularSpline
    assert image.interpolator_cls is InterpolatorRectangularSpline


# ---------------------------------------------------------------------------
# mesh_weight_map_from inheritance
# ---------------------------------------------------------------------------


def test__mesh_weight_map_from__density__returns_none():
    density = aa.mesh.RectangularSplineAdaptDensity(shape=(3, 3))
    assert density.mesh_weight_map_from(adapt_data=None) is None


def test__mesh_weight_map_from__image__returns_weighted_normalized():
    image = aa.mesh.RectangularSplineAdaptImage(
        shape=(3, 3), weight_power=2.0, weight_floor=0.0
    )

    class _Stub:
        def __init__(self, arr):
            self.array = arr

    adapt = _Stub(np.array([1.0, 2.0, 4.0, 0.0, 8.0]))
    w = image.mesh_weight_map_from(adapt_data=adapt)

    # power=2 → squared, then clipped to 1e-12, then normalized
    expected = np.array([1.0, 4.0, 16.0, 1e-24, 64.0])
    expected = expected / expected.sum()
    assert w == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Mapper output shape + content consistency with linear variant
# ---------------------------------------------------------------------------


def _seeded_inputs(M=128, K=400, seed=0):
    rng = np.random.default_rng(seed)
    data_grid = rng.standard_normal((M, 2))
    data_grid_over = rng.standard_normal((K, 2)) * 0.8
    weights = rng.uniform(0.1, 1.0, size=M)
    weights = weights / weights.sum()
    return data_grid, data_grid_over, weights


def test__mappings_sizes_weights__shapes_match_linear():
    data_grid, over, weights = _seeded_inputs()

    idx_s, w_s = adaptive_rectangular_mappings_weights_via_interpolation_from_spline(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        deg=11,
        xp=np,
    )
    idx_l, w_l = adaptive_rectangular_mappings_weights_via_interpolation_from(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        xp=np,
    )

    assert idx_s.shape == idx_l.shape == (400, 4)
    assert w_s.shape == w_l.shape == (400, 4)
    # bilinear weights sum to 1 per row for both variants
    assert np.allclose(w_s.sum(axis=1), 1.0, atol=1e-10)
    assert np.allclose(w_l.sum(axis=1), 1.0, atol=1e-10)


def test__mappings__flat_indices_mostly_match_linear():
    """Spline and linear CDF route most points to the same cell — only points
    near bracket boundaries can differ. Assert at least 70% exact match."""
    data_grid, over, weights = _seeded_inputs()

    idx_s, _ = adaptive_rectangular_mappings_weights_via_interpolation_from_spline(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        deg=11,
        xp=np,
    )
    idx_l, _ = adaptive_rectangular_mappings_weights_via_interpolation_from(
        source_grid_size=16,
        data_grid=data_grid,
        data_grid_over_sampled=over,
        mesh_weight_map=weights,
        xp=np,
    )

    exact_match_frac = float((idx_s == idx_l).mean())
    assert exact_match_frac >= 0.70, (
        f"spline vs linear flat_index exact-match fraction = {exact_match_frac:.3f} "
        "(expected ≥ 0.70)"
    )


# ---------------------------------------------------------------------------
# CDF transform round-trip
# ---------------------------------------------------------------------------


def test__create_transforms_spline__roundtrip_matches_identity():
    data_grid, _, weights = _seeded_inputs(seed=1)

    fwd, rev = create_transforms_spline(
        data_grid, mesh_weight_map=weights, deg=11, xp=np
    )

    probe = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
    roundtrip = fwd(rev(probe))
    assert roundtrip == pytest.approx(probe, abs=1e-4)


def test__create_transforms_spline__unweighted_roundtrip():
    data_grid, _, _ = _seeded_inputs(seed=2)

    fwd, rev = create_transforms_spline(
        data_grid, mesh_weight_map=None, deg=11, xp=np
    )

    probe = np.array([[0.25, 0.3], [0.5, 0.5], [0.7, 0.75]])
    roundtrip = fwd(rev(probe))
    assert roundtrip == pytest.approx(probe, abs=1e-4)


def test__create_transforms_spline__fwd_maps_to_unit_square():
    data_grid, _, weights = _seeded_inputs(seed=3)

    fwd, _ = create_transforms_spline(
        data_grid, mesh_weight_map=weights, deg=11, xp=np
    )
    y = fwd(data_grid)
    assert y.min() >= 0.0 - 1e-12
    assert y.max() <= 1.0 + 1e-12


# ---------------------------------------------------------------------------
# Interpolator construction — exercise the class directly without the full
# BorderRelocator / Grid2D pipeline (covered by the workspace benchmark).
# ---------------------------------------------------------------------------


def test__InterpolatorRectangularSpline__mappings_sizes_weights_via_property():
    from autoarray.inversion.mesh.interpolator.rectangular_spline import (
        InterpolatorRectangularSpline,
    )

    class _StubGrid:
        def __init__(self, arr):
            self.array = arr
            self.over_sampled = self
            self._array = arr  # for nested .array access if needed

        def __getattr__(self, item):
            # Forward unknown attribute access to the underlying ndarray so
            # callers can probe shape / dtype without caring about the wrapper.
            return getattr(self._array, item)

    rng = np.random.default_rng(5)
    data_grid = _StubGrid(rng.standard_normal((64, 2)))

    mesh = aa.mesh.RectangularSplineAdaptDensity(shape=(6, 6), spline_deg=7)
    interpolator = InterpolatorRectangularSpline(
        mesh=mesh,
        mesh_grid=_StubGrid(rng.standard_normal((36, 2))),
        data_grid=data_grid,
        mesh_weight_map=None,
        spline_deg=7,
        xp=np,
    )

    mappings, sizes, weights = interpolator._mappings_sizes_weights
    assert mappings.shape == (64, 4)
    assert sizes.shape == (64,)
    assert weights.shape == (64, 4)
    assert np.all(sizes == 4)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)
