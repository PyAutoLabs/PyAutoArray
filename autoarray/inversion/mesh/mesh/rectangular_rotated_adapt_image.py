"""Brightness-weighted PCA-rotated adaptive rectangular mesh.

Solves the multi-modal "ghost-peak" failure mode of
:class:`RectangularSplineAdaptImage` — see ``files/ghost_peak_findings.md``
for the empirical confirmation and ``files/cdf_audit.md`` for the underlying
mechanism.

Strategy
--------

The separable per-axis CDF concentrates pixels at the outer product of the
y-marginal modes and x-marginal modes. For a single-mode adapt image this
is correct; for a multi-modal adapt image with K bright peaks at off-axis
positions, the outer product creates K**2 dense zones — only K of which are
real source peaks, the remaining K**2 - K being "ghost" cross-products.

This mesh rotates the source plane into the brightness-weighted PCA frame
before running the existing spline CDF. In that frame, the dominant
direction of brightness variance is axis-aligned, so the marginal modes
collapse onto each other rather than producing off-axis ghosts. The
empirical test cases (K=2 diagonal and K=3 triangular) drop from 99 % and
96 % ghost-to-real density down to 0 % and 6 % respectively.

The class is a drop-in subclass of :class:`RectangularSplineAdaptImage`;
``weight_power``, ``weight_floor`` and ``spline_deg`` inherit unchanged.
The rotation runs once per fit (not per likelihood eval); the per-likelihood
cost is identical to the parent class plus two ``(N, 2) @ (2, 2)`` matrix
multiplies.
"""
from typing import Optional, Tuple

import numpy as np

from autoarray.inversion.mesh.border_relocator import BorderRelocator
from autoarray.inversion.mesh.mesh.rectangular_adapt_density import (
    overlay_grid_from,
)
from autoarray.inversion.mesh.mesh.rectangular_spline_adapt_image import (
    RectangularSplineAdaptImage,
)
from autoarray.inversion.mesh.interpolator.rectangular_spline import (
    SPLINE_CDF_DEFAULT_DEG,
)
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D


class _GridLike:
    """Thin numpy-array wrapper exposing the duck-type surface that the
    spline interpolator + geometry reach into.

    Replicates the parts of :class:`Grid2D` that downstream code touches:
    ``.array`` and basic ndarray ops (``mean``, ``std``, arithmetic). The
    full Mask2D / over_sample machinery is not needed once coordinates have
    been rotated into the PCA frame — downstream consumers only read the
    coordinate ndarray.
    """

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    @property
    def array(self):
        return self._arr

    def mean(self, axis=None):
        return self._arr.mean(axis=axis)

    def std(self, axis=None):
        return self._arr.std(axis=axis)

    def min(self, axis=None):
        return self._arr.min(axis=axis)

    def max(self, axis=None):
        return self._arr.max(axis=axis)

    def __sub__(self, other):
        return self._arr - other

    def __rsub__(self, other):
        return other - self._arr

    def __add__(self, other):
        return self._arr + other

    def __radd__(self, other):
        return self._arr + other

    def __mul__(self, other):
        return self._arr * other

    def __rmul__(self, other):
        return self._arr * other

    def __truediv__(self, other):
        return self._arr / other

    def __getitem__(self, key):
        return self._arr[key]

    def __len__(self):
        return len(self._arr)

    @property
    def shape(self):
        return self._arr.shape


class _RotatedGridShim:
    """``data_grid`` substitute carrying the base + oversampled coordinates
    after rotation into the PCA frame.

    Exposes the minimal surface that
    :class:`InterpolatorRectangularSpline._mappings_sizes_weights`,
    :class:`MeshGeometryRectangular.areas_transformed` /
    :meth:`edges_transformed`, AND the broader mapper / inversion pipeline
    read:

    - ``.array`` and ``.over_sampled.array`` — rotated coordinates
    - ``.mask`` and ``.over_sampler`` — forwarded unchanged from the
      original Grid2D (rotation doesn't change which pixels are masked
      or the sub-sample structure)

    The ``original_grid`` is the un-rotated ``Grid2D`` that supplied the
    coordinates; we keep a reference solely to forward its mask and
    over-sampler. The coordinate fields ``self.array`` and
    ``self.over_sampled.array`` are the post-rotation versions.
    """

    def __init__(self, original_grid, base_rotated, over_sampled_rotated):
        self._base = _GridLike(base_rotated)
        self.over_sampled = _GridLike(over_sampled_rotated)
        # Mask and over-sampler are rotation-invariant — same pixels are
        # masked, same sub-sample factor — so forward them directly.
        self.mask = original_grid.mask
        self.over_sampler = original_grid.over_sampler

    @property
    def array(self):
        return self._base.array

    def mean(self, axis=None):
        return self._base.mean(axis=axis)

    def std(self, axis=None):
        return self._base.std(axis=axis)


def _pca_rotation_from(traced_points: np.ndarray, brightness: np.ndarray, xp=np):
    """Compute a brightness-weighted PCA rotation matrix and centroid.

    Returns ``(R, centroid)`` such that
    ``rotated = (traced_points - centroid) @ R.T`` puts the principal axis
    of brightness-weighted variance along the first coordinate.

    Sign convention: the first basis vector is chosen with positive
    component along the original y-axis (or x if y is zero) for determinism;
    the second is set so ``det(R) = +1`` (proper rotation, no reflection).
    """
    w = xp.clip(brightness, 1e-12, None)
    w = w / xp.sum(w)
    centroid = (w[:, None] * traced_points).sum(axis=0)
    centred = traced_points - centroid
    cov = (w[:, None, None] * centred[:, :, None] * centred[:, None, :]).sum(axis=0)

    eigvals, eigvecs = xp.linalg.eigh(cov)
    order = xp.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Deterministic sign: first eigenvector has non-negative first component;
    # tie-break on the second component if the first is zero.
    sign = xp.where(eigvecs[0, 0] != 0.0, xp.sign(eigvecs[0, 0]), xp.sign(eigvecs[1, 0]))
    sign = xp.where(sign == 0.0, 1.0, sign)
    eigvecs = eigvecs * xp.array([sign, 1.0])

    # Enforce det(R) = +1 by flipping the second column if needed.
    if xp.linalg.det(eigvecs) < 0:
        eigvecs = eigvecs * xp.array([1.0, -1.0])

    R = eigvecs.T
    return R, centroid


class RectangularRotatedAdaptImage(RectangularSplineAdaptImage):
    """PCA-rotated adaptive rectangular mesh for multi-modal adapt images.

    Constructor inherits ``shape``, ``weight_power``, ``weight_floor``,
    ``spline_deg`` from :class:`RectangularSplineAdaptImage`. The rotation
    is computed automatically at :meth:`interpolator_from` time using the
    brightness-weighted PCA of the relocated source-plane grid.

    Parameters
    ----------
    shape
        2D dimensions of the rectangular pixel grid ``(total_y_pixels,
        total_x_pixels)``.
    weight_power
        Exponent applied to the adapt-image weights (same as parent).
    weight_floor
        Minimum weight applied to ensure numerical stability (same as parent).
    spline_deg
        Polynomial degree of the inverse-CDF spline fit (same as parent).
    """

    @property
    def interpolator_cls(self):
        from autoarray.inversion.mesh.interpolator.rectangular_rotated_spline import (
            InterpolatorRectangularRotatedSpline,
        )

        return InterpolatorRectangularRotatedSpline

    def interpolator_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        border_relocator: Optional[BorderRelocator] = None,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        if adapt_data is None:
            raise ValueError(
                "RectangularRotatedAdaptImage requires an adapt_data input to "
                "compute the brightness-weighted PCA rotation. For the "
                "non-rotated single-signal case, use RectangularSplineAdaptImage."
            )

        relocated_grid = self.relocated_grid_from(
            border_relocator=border_relocator,
            source_plane_data_grid=source_plane_data_grid,
            xp=xp,
        )

        # Brightness used for both PCA weighting AND the mesh_weight_map.
        # Match the parent's recipe: clip + power. (The PCA rotation only
        # cares about relative weighting; the parent's per-signal floor and
        # renormalisation cancel under PCA.)
        b = adapt_data.array if hasattr(adapt_data, "array") else adapt_data
        brightness_for_pca = xp.clip(b, 1e-12, None) ** self.weight_power

        R, centroid = _pca_rotation_from(
            traced_points=relocated_grid.array,
            brightness=brightness_for_pca,
            xp=xp,
        )

        # Rotate the base + oversampled grids into the PCA frame.
        base_rotated = (relocated_grid.array - centroid) @ R.T
        over_sampled_rotated = (
            relocated_grid.over_sampled.array - centroid
        ) @ R.T

        rotated_grid = _RotatedGridShim(
            original_grid=relocated_grid,
            base_rotated=base_rotated,
            over_sampled_rotated=over_sampled_rotated,
        )

        # Pixel-centre overlay built in the rotated frame (aligned with the
        # PCA-aligned CDF), then mapped back to the source frame so the
        # stored mesh_grid plots correctly without extra un-rotation steps.
        mesh_grid_rotated = overlay_grid_from(
            shape_native=self.shape,
            grid=rotated_grid.over_sampled,
            xp=xp,
        )
        mesh_grid_source = mesh_grid_rotated @ R + centroid

        mesh_weight_map = self.mesh_weight_map_from(adapt_data=adapt_data, xp=xp)

        return self.interpolator_cls(
            mesh=self,
            data_grid=rotated_grid,
            mesh_grid=Grid2DIrregular(mesh_grid_source),
            mesh_weight_map=mesh_weight_map,
            adapt_data=adapt_data,
            spline_deg=self.spline_deg,
            rotation_matrix=R,
            rotation_centroid=centroid,
            xp=xp,
        )
