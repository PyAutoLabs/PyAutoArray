"""
Kernel-density CDF variant of the rectangular adaptive interpolator.

The adaptive rectangular mesh transforms source-plane coordinates through a
per-axis CDF so that the uniform mesh pixels adapt to the density (or weight)
of the traced points.  The "linear" variant in ``rectangular.py`` uses a
piecewise-linear empirical CDF whose knots are the traced points themselves:
whenever interp queries coincide with knots (imaging at pixelization
over-sampling 1, the interferometer sparse path) the likelihood becomes
exactly piecewise-constant in the mass/shear parameters, and every rank swap
between traced points leaves a kink even when gradients do flow.

This module replaces the empirical CDF with a smooth kernel-density CDF
(the RTU formulation of Enzi et al., arXiv:2606.30620): per axis

    F(x) = sum_i w_i * Phi((x - x_i) / h)

where ``Phi`` is the standard normal CDF, ``x_i`` the traced points, ``w_i``
uniform weights (density adaption) or the normalized adapt-image weights
(image adaption), and ``h`` a bandwidth tied to the mesh resolution.

Properties, by construction:

- strictly monotone (no jitter hack, no ``_enforce_strict_monotone``);
- C-infinity in the queries AND the traced-point positions — no ranks, no
  sorts, no ``argsort`` anywhere, so there is nothing to swap;
- duplicate-safe: coincident traced points simply stack their weights (no
  ``1 / Δknot`` terms).

The forward transform is evaluated exactly at the query points (keeping the
C-infinity guarantee) and rescaled so the data bounding box maps onto the
unit square exactly — matching the linear variant's convention, including
clamping to [0, 1] outside the data range.  The inverse — only needed at
fixed unit-square grid values (mesh pixel centres/edges) — is a linear-interp
lookup on a small fixed table of ``n_knots`` knots spanning the data range;
because the inverse queries are constants, gradients flow smoothly through
the table values.

The linear meshes in ``rectangular.py`` and the spline meshes in
``rectangular_spline.py`` are untouched; per the pattern established there,
the bilinear-weights steps are copied rather than shared so the variants stay
independently auditable.
"""
import numpy as np
from functools import partial
from typing import Optional

from autoconf import cached_property

from autoarray.inversion.mesh.interpolator.rectangular import (
    InterpolatorRectangular,
    reverse_interp,
    reverse_interp_np,
)


KERNEL_CDF_DEFAULT_BANDWIDTH: float = 1.0
KERNEL_CDF_DEFAULT_KNOTS: int = 64

# Queries per block of the forward-transform evaluation. The exact kernel sum
# broadcasts an (M, N, 2) array; blocking the query dimension caps the peak at
# (block, N, 2) — 512 queries × 15.4k points × 2 axes × 8 B ≈ 126 MB at
# production imaging scale (previously ~60 GB unblocked at M ≈ 246k). Values
# are identical to float precision: the sum over points is unchanged, blocks
# only tile the query axis.
KERNEL_FORWARD_BLOCK: int = 512

_SQRT2 = np.sqrt(2.0)


def _norm_cdf(t, xp):
    """Standard normal CDF, xp-aware (scipy erf on numpy, jax.scipy under jax)."""
    if xp is np:
        from scipy.special import erf
    else:
        from jax.scipy.special import erf

    return 0.5 * (1.0 + erf(t / _SQRT2))


def create_transforms_kernel(
    traced_points,
    mesh_pixels: int,
    mesh_weight_map=None,
    bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
    n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
    xp=np,
):
    """
    Build the per-axis kernel-density CDF transform pair.

    Drop-in for ``rectangular.create_transforms``: the returned ``transform``
    maps (scaled) source-plane coordinates into the unit square and
    ``inv_transform`` maps unit-square coordinates back to the (scaled)
    source plane.

    Parameters
    ----------
    traced_points
        The (N, 2) scaled source-plane coordinates the CDF adapts to.
    mesh_pixels
        Mesh pixels per axis; sets the default bandwidth scale
        ``h = bandwidth * data_span / mesh_pixels`` per axis.
    mesh_weight_map
        Optional (N,) weights from the adapt image (image adaption). ``None``
        gives uniform weights (density adaption).
    bandwidth
        Bandwidth in units of the mesh pixel scale (data span / mesh_pixels).
        Smaller values track the point density more sharply (approaching the
        empirical CDF and its staircase); larger values smooth the mesh
        geometry towards uniform.
    n_knots
        Size of the fixed knot table used to invert the CDF.
    xp
        The array library to use (numpy or jax.numpy).
    """
    points = traced_points
    N = points.shape[0]

    if mesh_weight_map is None:
        w = xp.full((N,), 1.0 / N)
    else:
        w = mesh_weight_map / xp.sum(mesh_weight_map)

    lo = xp.min(points, axis=0)
    hi = xp.max(points, axis=0)
    span = hi - lo
    h = bandwidth * span / mesh_pixels

    def F_raw_block(q_block):
        t = (q_block[:, None, :] - points[None, :, :]) / h[None, None, :]
        return xp.sum(w[None, :, None] * _norm_cdf(t, xp), axis=1)

    if xp.__name__.startswith("jax"):

        def F_raw(q):
            import jax

            M = q.shape[0]
            n_blocks = -(-M // KERNEL_FORWARD_BLOCK)
            pad = n_blocks * KERNEL_FORWARD_BLOCK - M
            q_padded = xp.pad(q, ((0, pad), (0, 0)))
            blocks = q_padded.reshape(n_blocks, KERNEL_FORWARD_BLOCK, 2)
            out = jax.lax.map(F_raw_block, blocks)
            return out.reshape(n_blocks * KERNEL_FORWARD_BLOCK, 2)[:M]

    else:

        def F_raw(q):
            return np.concatenate(
                [
                    F_raw_block(q[i : i + KERNEL_FORWARD_BLOCK])
                    for i in range(0, q.shape[0], KERNEL_FORWARD_BLOCK)
                ],
                axis=0,
            )

    # The unit square maps onto the data bounding box exactly, matching the
    # linear variant's convention (its empirical-CDF knots end at the extreme
    # points); the kernel tails outside [lo, hi] are absorbed by the rescale.
    u = xp.linspace(0.0, 1.0, n_knots)
    knots = lo[None, :] + u[:, None] * span[None, :]

    F_knots_raw = F_raw(knots)
    F_lo = F_knots_raw[0]
    F_hi = F_knots_raw[-1]
    denom = F_hi - F_lo

    F_knots = (F_knots_raw - F_lo[None, :]) / denom[None, :]

    def transform(q):
        F_q = (F_raw(q) - F_lo[None, :]) / denom[None, :]
        return xp.clip(F_q, 0.0, 1.0)

    if xp.__name__.startswith("jax"):
        inv_transform = partial(reverse_interp, F_knots, knots)
        return transform, inv_transform

    inv_transform = partial(reverse_interp_np, F_knots, knots)
    return transform, inv_transform


def adaptive_rectangular_transformed_grid_from_kernel(
    data_grid,
    grid,
    mesh_pixels: int,
    mesh_weight_map=None,
    bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
    n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
    xp=np,
):
    """Kernel-CDF version of ``rectangular.adaptive_rectangular_transformed_grid_from``."""
    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    _, inv_transform = create_transforms_kernel(
        source_grid_scaled,
        mesh_pixels=mesh_pixels,
        mesh_weight_map=mesh_weight_map,
        bandwidth=bandwidth,
        n_knots=n_knots,
        xp=xp,
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    return inv_full(grid)


def adaptive_rectangular_areas_from_kernel(
    source_grid_shape,
    data_grid,
    mesh_weight_map=None,
    bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
    n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
    xp=np,
):
    """Kernel-CDF version of ``mesh_geometry.rectangular.adaptive_rectangular_areas_from``."""
    edges_y = xp.linspace(1, 0, source_grid_shape[0] + 1)
    edges_x = xp.linspace(0, 1, source_grid_shape[1] + 1)

    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    _, inv_transform = create_transforms_kernel(
        source_grid_scaled,
        mesh_pixels=source_grid_shape[0],
        mesh_weight_map=mesh_weight_map,
        bandwidth=bandwidth,
        n_knots=n_knots,
        xp=xp,
    )

    def inv_full(U):
        return inv_transform(U) * scale + mu

    pixel_edges = inv_full(xp.stack([edges_y, edges_x]).T)
    pixel_lengths = xp.diff(pixel_edges, axis=0).squeeze()

    dy = pixel_lengths[:, 0]
    dx = pixel_lengths[:, 1]

    return xp.abs(xp.outer(dy, dx).flatten())


def adaptive_rectangular_mappings_weights_via_interpolation_from_kernel(
    source_grid_size: int,
    data_grid,
    data_grid_over_sampled,
    mesh_weight_map=None,
    bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
    n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
    xp=np,
):
    """Kernel-CDF version of the linear helper in ``rectangular.py``.

    Steps 1–2 build the kernel transforms.  Steps 3–7 (floor/ceil, flatten,
    bilinear weights) are identical to the linear path — copied here to keep
    the variants independently auditable.
    """
    mu = data_grid.mean(axis=0)
    scale = data_grid.std(axis=0).min()
    source_grid_scaled = (data_grid - mu) / scale

    transform, _ = create_transforms_kernel(
        source_grid_scaled,
        mesh_pixels=source_grid_size,
        mesh_weight_map=mesh_weight_map,
        bandwidth=bandwidth,
        n_knots=n_knots,
        xp=xp,
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


class InterpolatorRectangularKernel(InterpolatorRectangular):
    """Kernel-density-CDF adaptive rectangular interpolator.

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
        bandwidth: float = KERNEL_CDF_DEFAULT_BANDWIDTH,
        n_knots: int = KERNEL_CDF_DEFAULT_KNOTS,
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
        self.bandwidth = bandwidth
        self.n_knots = n_knots

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
            kernel_bandwidth=self.bandwidth,
            kernel_knots=self.n_knots,
            xp=self._xp,
        )

    @cached_property
    def _mappings_sizes_weights(self):
        mappings, weights = (
            adaptive_rectangular_mappings_weights_via_interpolation_from_kernel(
                source_grid_size=self.mesh.shape[0],
                data_grid=self.data_grid.array,
                data_grid_over_sampled=self.data_grid.over_sampled.array,
                mesh_weight_map=self.mesh_weight_map,
                bandwidth=self.bandwidth,
                n_knots=self.n_knots,
                xp=self._xp,
            )
        )

        sizes = 4 * self._xp.ones(len(mappings), dtype="int")

        return mappings, sizes, weights
