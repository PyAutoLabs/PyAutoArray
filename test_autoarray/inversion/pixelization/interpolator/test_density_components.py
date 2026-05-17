import numpy as np
import pytest

from autoarray.inversion.mesh.interpolator.density_components import (
    compose_density,
    uniform_density_component,
)


def _brightness_proxy(traced_points, context, xp=np):
    return context["brightness"]


_brightness_proxy.required_context_keys = ("brightness",)


def _radial_density(traced_points, context, xp=np):
    return xp.sqrt(traced_points[:, 0] ** 2 + traced_points[:, 1] ** 2)


_radial_density.required_context_keys = ()


@pytest.fixture
def traced_points():
    return np.array(
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, -2.0]],
        dtype=np.float64,
    )


def test_uniform_component_returns_ones_then_normalises_to_one_over_n(traced_points):
    rho = compose_density(
        components=[uniform_density_component],
        weights=[1.0],
        floor=0.0,
        traced_points=traced_points,
        context={},
        xp=np,
    )

    assert rho.shape == (4,)
    np.testing.assert_allclose(rho, np.full(4, 0.25))
    np.testing.assert_allclose(rho.sum(), 1.0)


def test_two_components_linear_combination_matches_hand_calculation(traced_points):
    brightness = np.array([1.0, 2.0, 3.0, 4.0])
    weights = [0.25, 0.75]

    rho = compose_density(
        components=[uniform_density_component, _brightness_proxy],
        weights=weights,
        floor=0.0,
        traced_points=traced_points,
        context={"brightness": brightness},
        xp=np,
    )

    expected_unnormalised = 0.25 * np.ones(4) + 0.75 * brightness
    expected = expected_unnormalised / expected_unnormalised.sum()

    np.testing.assert_allclose(rho, expected)


def test_floor_applies_to_composite_not_per_component(traced_points):
    """The floor is added once to the composite — verifies the design choice
    (cdf_audit.md §6.1). A per-component floor would add ``floor * K``
    instead of ``floor``.
    """
    brightness = np.array([0.0, 0.0, 1.0, 1.0])  # half-zero so the floor matters
    floor = 0.5
    weights = [1.0, 1.0]

    rho = compose_density(
        components=[uniform_density_component, _brightness_proxy],
        weights=weights,
        floor=floor,
        traced_points=traced_points,
        context={"brightness": brightness},
        xp=np,
    )

    # Composite-floor expectation: floor + uniform + brightness.
    expected_unnormalised = floor + np.ones(4) + brightness
    expected = expected_unnormalised / expected_unnormalised.sum()
    np.testing.assert_allclose(rho, expected)

    # Per-component-floor would be 2*floor + uniform + brightness, which differs.
    wrong_unnormalised = 2 * floor + np.ones(4) + brightness
    wrong = wrong_unnormalised / wrong_unnormalised.sum()
    assert not np.allclose(rho, wrong)


def test_missing_context_key_raises_keyerror_before_component_runs(traced_points):
    with pytest.raises(KeyError, match="brightness"):
        compose_density(
            components=[_brightness_proxy],
            weights=[1.0],
            floor=0.0,
            traced_points=traced_points,
            context={},  # missing "brightness"
            xp=np,
        )


def test_mismatched_components_and_weights_raises_valueerror(traced_points):
    with pytest.raises(ValueError, match="components.*weights"):
        compose_density(
            components=[uniform_density_component, _brightness_proxy],
            weights=[1.0],  # too short
            floor=0.0,
            traced_points=traced_points,
            context={"brightness": np.ones(4)},
            xp=np,
        )


def test_output_is_strictly_positive_even_when_components_zero(traced_points):
    """Eps clipping plus the floor keep the result positive — required for
    the downstream CDF cumsum to remain strictly monotone (audit §9).
    """
    zero_brightness = np.zeros(4)
    rho = compose_density(
        components=[_brightness_proxy],
        weights=[1.0],
        floor=0.0,  # floor=0 with zero brightness exercises the eps clip
        traced_points=traced_points,
        context={"brightness": zero_brightness},
        xp=np,
    )

    assert np.all(rho > 0.0)
    assert np.all(np.isfinite(rho))
    np.testing.assert_allclose(rho.sum(), 1.0)


def test_output_is_finite_with_extreme_weight_magnitude(traced_points):
    """Large positive weights should not overflow normalisation; the
    sum-to-one renormalisation cancels overall scale.
    """
    rho = compose_density(
        components=[uniform_density_component],
        weights=[1e12],
        floor=0.0,
        traced_points=traced_points,
        context={},
        xp=np,
    )

    assert np.all(np.isfinite(rho))
    np.testing.assert_allclose(rho.sum(), 1.0)


def test_uniform_required_context_keys_is_empty():
    assert uniform_density_component.required_context_keys == ()


def test_traced_points_dependent_component_uses_coordinates():
    """Verifies a component can read ``traced_points`` and that the values
    flow through unchanged into the composite (modulo normalisation). Uses
    points off the origin so the radius signal stays above the eps clip.
    """
    points = np.array(
        [[1.0, 0.0], [0.0, 1.0], [3.0, 4.0], [1.0, 1.0]],
        dtype=np.float64,
    )

    rho = compose_density(
        components=[_radial_density],
        weights=[1.0],
        floor=0.0,
        traced_points=points,
        context={},
        xp=np,
    )

    radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    expected = radii / radii.sum()
    np.testing.assert_allclose(rho, expected)
