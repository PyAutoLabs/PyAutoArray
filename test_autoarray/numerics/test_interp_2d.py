import numpy as np
import pytest

from autoarray.numerics import interp_2d


def test__interp_2d__numpy_random_inbounds():
    rng = np.random.RandomState(0)
    x_axis = np.linspace(-1.0, 1.0, 11)
    y_axis = np.linspace(-2.0, 2.0, 21)
    values = rng.randn(11, 21)

    points = rng.uniform(low=[-0.9, -1.9], high=[0.9, 1.9], size=(50, 2))
    result = interp_2d(points, x_axis, y_axis, values, xp=np)

    assert result.shape == (50,)
    assert np.all(np.isfinite(result))


def test__interp_2d__numpy_jax_parity_inbounds():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    rng = np.random.RandomState(0)
    x_axis = np.linspace(-1.0, 1.0, 11)
    y_axis = np.linspace(-2.0, 2.0, 21)
    values = rng.randn(11, 21)
    points = rng.uniform(low=[-0.9, -1.9], high=[0.9, 1.9], size=(50, 2))

    np_result = interp_2d(points, x_axis, y_axis, values, xp=np)
    jax_result = interp_2d(
        jnp.asarray(points),
        jnp.asarray(x_axis),
        jnp.asarray(y_axis),
        jnp.asarray(values),
        xp=jnp,
    )

    np.testing.assert_allclose(np.asarray(jax_result), np_result, rtol=1e-6)


def test__interp_2d__out_of_bounds_returns_fill_value():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    x_axis = np.linspace(0.0, 1.0, 5)
    y_axis = np.linspace(0.0, 1.0, 5)
    values = np.ones((5, 5))
    fill_value = -99.0

    points = np.array([[10.0, 10.0], [-5.0, 0.5], [0.5, -5.0]])

    np_result = interp_2d(points, x_axis, y_axis, values, fill_value=fill_value, xp=np)
    jax_result = interp_2d(
        jnp.asarray(points), jnp.asarray(x_axis), jnp.asarray(y_axis),
        jnp.asarray(values), fill_value=fill_value, xp=jnp,
    )

    assert np.allclose(np_result, fill_value)
    assert np.allclose(np.asarray(jax_result), fill_value)


def test__interp_2d__single_point_query_has_shape_1():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    x_axis = np.linspace(0.0, 1.0, 5)
    y_axis = np.linspace(0.0, 1.0, 5)
    values = np.ones((5, 5))

    points = np.array([[0.5, 0.5]])

    np_result = interp_2d(points, x_axis, y_axis, values, xp=np)
    jax_result = interp_2d(
        jnp.asarray(points), jnp.asarray(x_axis), jnp.asarray(y_axis),
        jnp.asarray(values), xp=jnp,
    )

    assert np_result.shape == (1,)
    assert np.asarray(jax_result).shape == (1,)
