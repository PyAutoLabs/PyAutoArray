"""
Spline-CDF variant of the rectangular adaptive interpolator.

The adaptive rectangular mesh transforms source-plane coordinates through a
per-axis CDF so that the uniform mesh pixels adapt to the density (or weight)
of the traced points.  The "linear" variant in ``rectangular.py`` uses a
piecewise-linear empirical CDF via ``np.interp`` / ``jnp.interp``.  Its
gradient is piecewise-constant with a jump at every knot, which produces
integrator noise in HMC / NUTS samplers and a ``1 / Δknot`` spike when two
traced points crowd together.

This module replaces the linear CDF with:

1. An order-``deg`` polynomial fit to the *inverse* empirical CDF, sampled at
   Chebyshev nodes to control Runge's phenomenon.  The polynomial is the
   ``rev_transform`` (unit square → source plane).
2. A cubic-Hermite spline table over the polynomial inverted numerically to
   evaluate ``fwd_transform`` (source plane → unit square) quickly.  Because
   the polynomial is monotonic in the data range, the Hermite spline is a
   single-valued inverse.

Both the rev- and fwd-transform are C¹ continuous, and the gradient flows
through JAX autograd end-to-end.  ``InvertPolySpline`` is registered as a JAX
pytree so it can be returned from ``jax.jit``.

Port from ``z_staging/rect_adap_spline_invert_jax (1).ipynb`` (canonical RSE
implementation).  Made ``xp``-aware so the same functions run under numpy and
JAX — in numpy mode we skip the ``@jax.jit`` and pytree registration (they're
no-ops) and use ``np.digitize``; in JAX mode we use
``jnp.digitize(method='scan_unrolled')`` and the gradient-safe ``dx0`` mask.
"""
from typing import Callable, Optional, Tuple

import numpy as np

from autoconf import cached_property

from autoarray.inversion.mesh.interpolator.abstract import AbstractInterpolator
from autoarray.inversion.mesh.interpolator.rectangular import InterpolatorRectangular


SPLINE_CDF_DEFAULT_DEG: int = 11


def _enforce_strict_monotone(x, xp):
    """Return a strictly-increasing copy of ``x`` along axis 0.

    Uses a running maximum plus a tiny jitter scaled by position so that
    repeated values pick up a monotone offset on the order of float64 eps.
    Works uniformly under numpy and jax.numpy — ``jnp.cummax`` lives on
    ``jax.lax`` rather than ``jnp`` so we dispatch explicitly.
    """
    if xp is np:
        running_max = np.maximum.accumulate(x, axis=0)
    else:
        from jax import lax as _lax

        running_max = _lax.cummax(x, axis=0)
    n = x.shape[0]
    jitter = xp.arange(n, dtype=x.dtype)[:, None] * xp.finfo(x.dtype).eps
    return running_max + jitter


# ---------------------------------------------------------------------------
# Chebyshev-node resampling helper (used during polyfit setup).
# ---------------------------------------------------------------------------


def _interp1d_numpy(x, xp_, fp):
    """1-D linear interp with linear *extrapolation* outside `[xp_[0], xp_[-1]]`.

    Matches the JAX `_interp1d_jax` helper's behaviour — needed for Chebyshev
    resampling near the endpoints, where `np.interp`'s default clamping would
    produce duplicate `cx` values and break the downstream `np.gradient`
    + `polyfit` chain.
    """
    i = np.clip(np.searchsorted(xp_, x, side="right"), 1, len(xp_) - 1)
    df = fp[i] - fp[i - 1]
    dx = xp_[i] - xp_[i - 1]
    delta = x - xp_[i - 1]
    # Protect against zero-width brackets from coincident knots.
    safe = np.where(np.abs(dx) <= np.finfo(xp_.dtype).eps, 1.0, dx)
    return fp[i - 1] + (delta / safe) * df


def _interp1d_jax(x, xp_, fp):
    """Gradient-safe 1-D linear interp for JAX.

    Matches the ``interp1d`` helper in the JAX notebook — uses
    ``searchsorted(method='scan_unrolled')`` and a ``dx0`` mask so that the
    derivative at duplicate knots does not produce NaNs.
    """
    import jax
    import jax.numpy as jnp

    i = jnp.clip(
        jnp.searchsorted(xp_, x, side="right", method="scan_unrolled"),
        1,
        len(xp_) - 1,
    )
    df = fp[i] - fp[i - 1]
    dx = xp_[i] - xp_[i - 1]
    delta = x - xp_[i - 1]
    epsilon = jnp.spacing(jnp.finfo(xp_.dtype).eps)
    dx0 = jax.lax.abs(dx) <= epsilon
    return jnp.where(dx0, fp[i - 1], fp[i - 1] + (delta / jnp.where(dx0, 1, dx)) * df)


def _vmapped_interp1d_jax(cheb_nodes, t, sort_points):
    import jax

    return jax.vmap(_interp1d_jax, in_axes=(None, 1, 1), out_axes=1)(
        cheb_nodes, t, sort_points
    )


# ---------------------------------------------------------------------------
# Polynomial inverse class — stores precomputed Hermite table.
# ---------------------------------------------------------------------------


class InvertPolySpline:
    """Polynomial + Hermite-spline inverse of the empirical CDF.

    ``rev_transform(y)`` evaluates ``polyval(coefs, y)`` — a smooth,
    closed-form inverse CDF (unit square → source plane).

    ``fwd_transform(x)`` inverts the polynomial at ``x`` using a precomputed
    cubic-Hermite table keyed by ``low_res`` nodes in ``[0, 1]``.  Hermite
    bases match both the polynomial value and its derivative at each node, so
    the resulting forward transform is C¹ continuous across every bracket.

    Attributes match the JAX notebook verbatim to keep the port faithful.
    """

    def __init__(self, coefs, lower_bound, upper_bound, low_res: int = 150, xp=np):
        self._xp = xp
        self.low_res = low_res

        self.coefs = coefs
        self.dcoefs = self._v_polyder(coefs)

        self.lower_bound = xp.atleast_2d(lower_bound)
        self.upper_bound = xp.atleast_2d(upper_bound)

        y_low_res_1d = xp.linspace(0.0, 1.0, low_res)
        self.y_low_res = xp.stack([y_low_res_1d, y_low_res_1d], axis=1)
        x_low_res_raw = self._v_polyval(self.coefs, self.y_low_res)
        # Enforce strict monotonicity on x_low_res.  A deg=11 polynomial fit to
        # an empirical inverse CDF occasionally wiggles by 1e-3 near the
        # endpoints (Runge residual even with Chebyshev sampling), which breaks
        # `np.digitize` / `searchsorted` in the Hermite inverter downstream.
        # The cummax-plus-jitter guarantees ``x_low_res`` is strictly
        # increasing without meaningfully biasing the fit away from the
        # polynomial it came from.
        self.x_low_res = _enforce_strict_monotone(x_low_res_raw, xp)
        self.dy_low_res = 1.0 / self._v_polyval(self.dcoefs, self.y_low_res)
        self.delta_x = xp.diff(self.x_low_res, axis=0)

    # -- Pytree plumbing -----------------------------------------------------
    # Registered lazily so the numpy path doesn't import jax.  The first JAX
    # call to ``create_transforms_spline`` triggers registration.
    _pytree_registered = False

    @classmethod
    def _register_pytree(cls):
        if cls._pytree_registered:
            return
        from jax.tree_util import register_pytree_node

        def flatten(obj):
            children = (
                obj.coefs,
                obj.dcoefs,
                obj.y_low_res,
                obj.x_low_res,
                obj.dy_low_res,
                obj.delta_x,
                obj.lower_bound,
                obj.upper_bound,
            )
            aux = (obj.low_res,)
            return children, aux

        def unflatten(aux, children):
            obj = object.__new__(cls)
            import jax.numpy as jnp

            obj._xp = jnp
            (
                obj.coefs,
                obj.dcoefs,
                obj.y_low_res,
                obj.x_low_res,
                obj.dy_low_res,
                obj.delta_x,
                obj.lower_bound,
                obj.upper_bound,
            ) = children
            (obj.low_res,) = aux
            return obj

        register_pytree_node(cls, flatten, unflatten)
        cls._pytree_registered = True

    # -- Vectorised polynomial helpers --------------------------------------
    def _v_polyder(self, c):
        xp = self._xp
        if xp is np:
            return np.stack([np.polyder(c[:, 0]), np.polyder(c[:, 1])], axis=1)
        import jax
        import jax.numpy as jnp

        return jax.vmap(jnp.polyder, in_axes=1, out_axes=1)(c)

    def _v_polyval(self, c, x):
        xp = self._xp
        if xp is np:
            return np.stack(
                [np.polyval(c[:, 0], x[:, 0]), np.polyval(c[:, 1], x[:, 1])], axis=1
            )
        import jax
        import jax.numpy as jnp

        return jax.vmap(jnp.polyval, in_axes=(1, 1), out_axes=1)(c, x)

    # -- Transforms ---------------------------------------------------------
    def rev_transform(self, y):
        """Unit square → source plane.  Smooth polynomial evaluation."""
        return self._v_polyval(self.coefs, y)

    def fwd_transform(self, x):
        """Source plane → unit square.  Cubic-Hermite spline inverse."""
        xp = self._xp
        if xp is np:
            y = np.stack(
                [_spline_invert_numpy(self, x, 0), _spline_invert_numpy(self, x, 1)],
                axis=1,
            )
        else:
            import jax

            y = jax.vmap(_spline_invert_jax, in_axes=(1, 1), out_axes=1)(self, x)

        y = xp.where(x <= self.lower_bound, 0.0, y)
        y = xp.where(x >= self.upper_bound, 1.0, y)
        return xp.clip(y, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Hermite-spline evaluators — one per backend.
# ---------------------------------------------------------------------------


def _hermite(t, y_left, y_right, dy_left, dy_right, dx_left):
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return y_left * h00 + y_right * h01 + (dy_left * h10 + dy_right * h11) * dx_left


def _spline_invert_numpy(ip: InvertPolySpline, x, idx: int):
    k_right = np.digitize(x[:, idx], ip.x_low_res[:, idx])
    k_left = k_right - 1

    k_right = np.clip(k_right, 0, ip.x_low_res.shape[0] - 1)
    k_left = np.clip(k_left, 0, ip.x_low_res.shape[0] - 2)

    dx_left = ip.delta_x[k_left, idx]
    t = (x[:, idx] - ip.x_low_res[k_left, idx]) / dx_left
    return _hermite(
        t,
        ip.y_low_res[k_left, idx],
        ip.y_low_res[k_right, idx],
        ip.dy_low_res[k_left, idx],
        ip.dy_low_res[k_right, idx],
        dx_left,
    )


def _spline_invert_jax(ip: InvertPolySpline, x):
    """x is a 1-D column (vmap strips axis 1 off the (N, 2) source point)."""
    import jax.numpy as jnp

    k_right = jnp.digitize(x, ip.x_low_res, method="scan_unrolled")
    k_left = k_right - 1
    # JAX's default OOB index gives the correct value at the right-most edge
    # of the interpolation, so no explicit clipping is required here.

    dx_left = ip.delta_x[k_left]
    t = (x - ip.x_low_res[k_left]) / dx_left
    return _hermite(
        t,
        ip.y_low_res[k_left],
        ip.y_low_res[k_right],
        ip.dy_low_res[k_left],
        ip.dy_low_res[k_right],
        dx_left,
    )


# ---------------------------------------------------------------------------
# Core transforms factory — drop-in replacement for create_transforms.
# ---------------------------------------------------------------------------


def _chebyshev_nodes(cheb_deg: int, xp):
    return (
        (xp.cos((2 * xp.arange(cheb_deg) + 1) * xp.pi / (2 * cheb_deg))[::-1]) + 1
    ) / 2


def _build_inv_poly_numpy(traced_points, mesh_weight_map, deg):
    import warnings

    N = traced_points.shape[0]

    if mesh_weight_map is None:
        t = np.arange(1, N + 1) / (N + 1)
        t = np.stack([t, t], axis=1)
        sort_points = np.sort(traced_points, axis=0)
    else:
        sdx = np.argsort(traced_points, axis=0)
        sort_points = np.take_along_axis(traced_points, sdx, axis=0)
        t = np.stack([mesh_weight_map, mesh_weight_map], axis=1)
        t = np.take_along_axis(t, sdx, axis=0)
        t = np.cumsum(t, axis=0)

    cheb_deg = 3 * deg
    cheb_nodes = _chebyshev_nodes(cheb_deg, np)
    cy = np.stack([cheb_nodes, cheb_nodes], axis=1)
    cx = np.stack(
        [
            _interp1d_numpy(cheb_nodes, t[:, 0], sort_points[:, 0]),
            _interp1d_numpy(cheb_nodes, t[:, 1], sort_points[:, 1]),
        ],
        axis=1,
    )
    w = np.stack([np.gradient(cy[:, 0], cx[:, 0]), np.gradient(cy[:, 1], cx[:, 1])], axis=1)

    rank_warning = getattr(np, "exceptions", np).__dict__.get("RankWarning", None) or getattr(
        np, "RankWarning", UserWarning
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rank_warning)
        coefs = np.stack(
            [
                np.polyfit(cy[:, 0], cx[:, 0], deg, w=w[:, 0]),
                np.polyfit(cy[:, 1], cx[:, 1], deg, w=w[:, 1]),
            ],
            axis=1,
        )

    return InvertPolySpline(
        coefs, sort_points[0], sort_points[-1], low_res=20 * deg, xp=np
    )


def _weighted_polyfit_jax(cy_col, cx_col, w_col, deg):
    """Weighted least-squares polynomial fit via normal equations.

    Replaces ``jnp.polyfit(x, y, deg, w=w)`` which compiles slowly under a
    double-vmap (the internal SVD blows up the XLA graph size).  The normal-
    equations form is a ``(deg+1) x (deg+1)`` solve per column — trivially
    small and compiles in milliseconds.  Output uses the same highest-
    first coefficient ordering as ``jnp.polyfit``.
    """
    import jax.numpy as jnp

    # Vandermonde columns in ascending power: V[:, k] = cy**k
    powers = jnp.arange(deg + 1)
    V = cy_col[:, None] ** powers[None, :]  # (cheb_deg, deg+1)
    # Weighted normal equations: (V^T W V) c = V^T W y, with w_col sample weights.
    VtW = V.T * w_col
    A = VtW @ V
    b = VtW @ cx_col
    c_asc = jnp.linalg.solve(A, b)
    return c_asc[::-1]  # descending order matches np.polyfit / np.polyval


def _build_inv_poly_jax_impl(traced_points, mesh_weight_map, deg):
    import jax
    import jax.numpy as jnp

    N = traced_points.shape[0]

    if mesh_weight_map is None:
        t = jnp.arange(1, N + 1) / (N + 1)
        t = jnp.stack([t, t], axis=1)
        sort_points = jnp.sort(traced_points, axis=0)
    else:
        sdx = jnp.argsort(traced_points, axis=0)
        sort_points = jnp.take_along_axis(traced_points, sdx, axis=0)
        t = jnp.stack([mesh_weight_map, mesh_weight_map], axis=1)
        t = jnp.take_along_axis(t, sdx, axis=0)
        t = jnp.cumsum(t, axis=0)

    cheb_deg = 3 * deg
    cheb_nodes = _chebyshev_nodes(cheb_deg, jnp)
    cy = jnp.stack([cheb_nodes, cheb_nodes], axis=1)
    cx = _vmapped_interp1d_jax(cheb_nodes, t, sort_points)

    w = jax.vmap(jnp.gradient, in_axes=(1, 1), out_axes=1)(cy, cx)
    coefs = jax.vmap(_weighted_polyfit_jax, in_axes=(1, 1, 1, None), out_axes=1)(
        cy, cx, w, deg
    )

    return InvertPolySpline(
        coefs, sort_points[0], sort_points[-1], low_res=20 * deg, xp=jnp
    )


def _build_inv_poly_jax(traced_points, mesh_weight_map, deg):
    """Build ``InvertPolySpline`` under JAX.

    Not JIT-compiled here: the fit runs inside a vmapped ``jnp.polyfit``,
    which is an SVD with very long XLA compile times when placed inside a
    separate ``jax.jit``. The build is called once per likelihood evaluation
    and the inner operations are already lowered efficiently; wrapping it in
    another JIT costs more in compilation than it saves at steady state.
    When the full likelihood is JIT-compiled by the caller (the typical path
    for gradient samplers), the build is traced as part of that outer JIT
    and benefits from the same fusion as the rest of the pipeline.
    """
    InvertPolySpline._register_pytree()
    return _build_inv_poly_jax_impl(traced_points, mesh_weight_map, deg)


def create_transforms_spline(
    traced_points,
    mesh_weight_map=None,
    deg: int = SPLINE_CDF_DEFAULT_DEG,
    xp=np,
) -> Tuple[Callable, Callable]:
    """Build forward + inverse CDF transforms via polynomial+Hermite-spline.

    Same signature and return convention as
    ``rectangular.create_transforms``: the returned ``transform`` maps source
    plane → [0, 1], and ``inv_transform`` maps [0, 1] → source plane.
    """
    if xp is np:
        inv_poly = _build_inv_poly_numpy(traced_points, mesh_weight_map, deg)
    else:
        inv_poly = _build_inv_poly_jax(traced_points, mesh_weight_map, deg)

    return inv_poly.fwd_transform, inv_poly.rev_transform


# ---------------------------------------------------------------------------
# Adaptive rectangular helpers — spline variants of the functions in
# ``rectangular.py``.  Signatures match one-for-one.
# ---------------------------------------------------------------------------


def adaptive_rectangular_transformed_grid_from_spline(
    data_grid,
    grid,
    mesh_weight_map=None,
    deg: int = SPLINE_CDF_DEFAULT_DEG,
    xp=np,
):
    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    _, inv_transform = create_transforms_spline(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, deg=deg, xp=xp
    )

    return inv_transform(grid) * scale + mu


def adaptive_rectangular_areas_from_spline(
    source_grid_shape,
    data_grid,
    mesh_weight_map=None,
    deg: int = SPLINE_CDF_DEFAULT_DEG,
    xp=np,
):
    edges_y = xp.linspace(1, 0, source_grid_shape[0] + 1)
    edges_x = xp.linspace(0, 1, source_grid_shape[1] + 1)

    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    _, inv_transform = create_transforms_spline(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, deg=deg, xp=xp
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    pixel_edges = inv_full(xp.stack([edges_y, edges_x]).T)
    pixel_lengths = xp.diff(pixel_edges, axis=0).squeeze()

    dy = pixel_lengths[:, 0]
    dx = pixel_lengths[:, 1]
    return xp.abs(xp.outer(dy, dx).flatten())


def adaptive_rectangular_mappings_weights_via_interpolation_from_spline(
    source_grid_size: int,
    data_grid,
    data_grid_over_sampled,
    mesh_weight_map=None,
    deg: int = SPLINE_CDF_DEFAULT_DEG,
    xp=np,
):
    """Spline-CDF version of the linear helper in ``rectangular.py``.

    Steps 1–2 build the spline transforms.  Steps 3–7 (floor/ceil, flatten,
    bilinear weights) are identical to the linear path — copied here to keep
    the two variants independently auditable.
    """
    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    transform, _ = create_transforms_spline(
        source_grid_scaled, mesh_weight_map=mesh_weight_map, deg=deg, xp=xp
    )

    grid_over_sampled_scaled = (data_grid_over_sampled - mu) / scale
    grid_over_sampled_transformed = transform(grid_over_sampled_scaled)
    grid_over_index = (source_grid_size - 3) * grid_over_sampled_transformed + 1

    ix_down = xp.floor(grid_over_index[:, 0])
    ix_up = xp.ceil(grid_over_index[:, 0])
    iy_down = xp.floor(grid_over_index[:, 1])
    iy_up = xp.ceil(grid_over_index[:, 1])

    idx_tl = xp.stack([ix_up, iy_down], axis=1)
    idx_tr = xp.stack([ix_up, iy_up], axis=1)
    idx_br = xp.stack([ix_down, iy_up], axis=1)
    idx_bl = xp.stack([ix_down, iy_down], axis=1)

    def flatten(idx, n):
        return (n - idx[:, 0]) * n + idx[:, 1]

    flat_tl = flatten(idx_tl, source_grid_size)
    flat_tr = flatten(idx_tr, source_grid_size)
    flat_bl = flatten(idx_bl, source_grid_size)
    flat_br = flatten(idx_br, source_grid_size)

    flat_indices = xp.stack([flat_tl, flat_tr, flat_bl, flat_br], axis=1).astype(
        "int64"
    )

    t_row = (grid_over_index[:, 0] - ix_down) / (ix_up - ix_down + 1e-12)
    t_col = (grid_over_index[:, 1] - iy_down) / (iy_up - iy_down + 1e-12)

    w_tl = (1 - t_row) * (1 - t_col)
    w_tr = (1 - t_row) * t_col
    w_bl = t_row * (1 - t_col)
    w_br = t_row * t_col
    weights = xp.stack([w_tl, w_tr, w_bl, w_br], axis=1)

    return flat_indices, weights


# ---------------------------------------------------------------------------
# Interpolator subclass — mirrors InterpolatorRectangular but routes through
# the spline helpers.
# ---------------------------------------------------------------------------


class InterpolatorRectangularSpline(InterpolatorRectangular):
    """Spline-CDF adaptive rectangular interpolator.

    Subclasses :class:`InterpolatorRectangular` so that existing
    ``isinstance(..., InterpolatorRectangular)`` dispatch sites (e.g.
    :mod:`autoarray.plot.inversion`) treat it as an adaptive rectangular
    interpolator, and the source-plane mesh reconstruction renders through
    the same ``pcolormesh`` path.
    """

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        mesh_weight_map,
        adapt_data: Optional[np.ndarray] = None,
        spline_deg: int = SPLINE_CDF_DEFAULT_DEG,
        xp=np,
    ):
        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid=data_grid,
            mesh_weight_map=mesh_weight_map,
            adapt_data=adapt_data,
            xp=xp,
        )
        self.spline_deg = spline_deg

    @cached_property
    def mesh_geometry(self):
        from autoarray.inversion.mesh.mesh_geometry.rectangular import (
            MeshGeometryRectangular,
        )

        return MeshGeometryRectangular(
            mesh=self.mesh,
            mesh_grid=self.mesh_grid,
            data_grid=self.data_grid,
            mesh_weight_map=self.mesh_weight_map,
            spline_deg=self.spline_deg,
            xp=self._xp,
        )

    @cached_property
    def _mappings_sizes_weights(self):
        mappings, weights = (
            adaptive_rectangular_mappings_weights_via_interpolation_from_spline(
                source_grid_size=self.mesh.shape[0],
                data_grid=self.data_grid.array,
                data_grid_over_sampled=self.data_grid.over_sampled.array,
                mesh_weight_map=self.mesh_weight_map,
                deg=self.spline_deg,
                xp=self._xp,
            )
        )
        sizes = 4 * self._xp.ones(len(mappings), dtype="int")
        return mappings, sizes, weights

    @cached_property
    def _mappings_sizes_weights_split(self):
        return self._mappings_sizes_weights
