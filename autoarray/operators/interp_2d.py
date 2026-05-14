"""
2D regular-grid bilinear interpolation with NumPy and JAX backends.

Mirrors the 1D dispatch pattern in
``autoarray.inversion.mesh.interpolator.rectangular_spline``.

NumPy path delegates to ``scipy.interpolate.RegularGridInterpolator``
(``bounds_error=False``, configurable ``fill_value``). JAX path is built on
``jax.scipy.ndimage.map_coordinates(order=1, mode='constant')``.
"""

import numpy as np


def _interp_2d_numpy(points, x_axis, y_axis, values, fill_value=0.0):
    """
    Evaluate ``values`` (regular grid on (x_axis, y_axis)) at ``(N, 2)`` query
    points using bilinear interpolation. Out-of-bounds queries return
    ``fill_value``.

    Matches the wiring used by ``DatasetInterp`` in PyAutoGalaxy: pass the
    axes and the values array unchanged, do not transpose.
    """
    from scipy import interpolate

    interpolator = interpolate.RegularGridInterpolator(
        points=(x_axis, y_axis),
        values=values,
        bounds_error=False,
        fill_value=fill_value,
    )
    return interpolator(points)


def _interp_2d_jax(points, x_axis, y_axis, values, fill_value=0.0):
    """
    JAX-compatible counterpart to ``_interp_2d_numpy``. Uses
    ``jax.scipy.ndimage.map_coordinates(order=1, mode='constant')``.

    The query ``points`` are in world coordinates with column 0 matching
    ``x_axis`` and column 1 matching ``y_axis`` (mirroring the scipy
    interpolator's ``(x, y)`` argument order). World coords are converted to
    pixel-fractional indices into ``values`` using axis spacings derived from
    the first two axis points.
    """
    import jax.numpy as jnp
    from jax.scipy.ndimage import map_coordinates

    x_axis = jnp.asarray(x_axis)
    y_axis = jnp.asarray(y_axis)
    values = jnp.asarray(values)
    pts = jnp.asarray(points)

    # World -> pixel-fractional indices.
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]
    col_coords = (pts[:, 0] - x_axis[0]) / dx  # column 0 of points -> x-axis
    row_coords = (pts[:, 1] - y_axis[0]) / dy  # column 1 of points -> y-axis

    # map_coordinates indexes values[i, j] where i is the first-axis index
    # and j the second. The scipy call uses points=(x_axis, y_axis) so the
    # first axis of `values` is interpreted as the x-axis.
    coords = jnp.stack([col_coords, row_coords], axis=0)

    return map_coordinates(
        values, coords, order=1, mode="constant", cval=fill_value
    )


def interp_2d(points, x_axis, y_axis, values, fill_value=0.0, xp=np):
    """Dispatcher -- picks ``_interp_2d_numpy`` or ``_interp_2d_jax`` based on ``xp``."""
    if xp is np:
        return _interp_2d_numpy(points, x_axis, y_axis, values, fill_value=fill_value)
    return _interp_2d_jax(points, x_axis, y_axis, values, fill_value=fill_value)
