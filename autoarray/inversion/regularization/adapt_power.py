from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.adapt import Adapt
from autoarray.inversion.regularization.adapt import adapt_regularization_weights_from
from autoarray.inversion.regularization.adapt import (
    weighted_regularization_matrix_single_scatter_from,
)


class AdaptPower(Adapt):
    def __init__(
        self,
        inner_coefficient: float = 1.0,
        outer_coefficient: float = 1.0,
        signal_scale: float = 1.0,
        power: float = 1.0,
    ):
        """
        Regularization which uses the neighbors of the mesh (e.g. shared Delaunay vertexes) and values adapted to
        the data being fitted to smooth an inversion's solution, with the coefficient convention of ``Constant``.

        This is the corrected sibling of ``Adapt``. It reconstructs the same adaptive weighting -- high smoothing
        where there is no signal, less where there is (Nightingale, Dye and Massey 2018) -- but fixes the two ways
        in which ``Adapt`` diverges from ``Constant``:

        1. **The coefficient enters at ``lambda^2``, not ``lambda^4``.** ``Adapt`` squares its coefficients twice
           (once when interpolating them into per-pixel weights, once in the matrix builder), so its matrix scales
           as the fourth power of the coefficient. This class raises the interpolated coefficient to ``power``
           before the builder squares it, so the effective exponent is ``2 * power`` and the default ``power=1.0``
           matches ``Constant``. Under the shared ``LogUniform(1e-6, 1e6)`` prior that means the prior now spans
           ``lambda^2``, and the regularization matrix stays positive-definite to ``c ~ 1e6`` rather than
           collapsing from ``c ~ 1e4`` -- the fragility that produced the likelihood-overflow floods seen in
           free-coefficient adaptive fits.
        2. **Every mesh edge is scattered once.** ``Adapt`` adds each ordered neighbor pair to both the ``(i, j)``
           and ``(j, i)`` entries, and the neighbor list already holds each unordered edge twice, so its matrix is
           exactly ``2 x`` ``Constant``'s. This class uses
           ``weighted_regularization_matrix_single_scatter_from``, which scatters each ordered pair once with the
           symmetric edge weight ``0.5 * (w_i ** 2 + w_j ** 2)`` -- a weighted graph Laplacian, so still symmetric
           and positive semi-definite.

        Together these make ``AdaptPower(inner_coefficient=c, outer_coefficient=c)`` **exactly equal** to
        ``Constant(coefficient=c)`` for any ``c``, which is what ``Adapt``'s docstring always claimed and never
        delivered.

        A full description of regularization and this matrix can be found in the parent ``AbstractRegularization``
        class; the ``B`` matrix construction is described on ``Adapt``.

        **Migration from ``Adapt``.** The coefficient scale is squared: ``c_new = c_old ** 2``. To reproduce the
        legacy class exactly, pass ``power=2.0`` -- but note the factor-2 scatter is fixed regardless, so
        ``AdaptPower(power=2.0)`` is ``0.5 x`` ``Adapt`` with the same coefficients.

        **JAX & gradient support**: as for ``Adapt`` -- JAX-differentiable and FD-certified on the rectangular
        mesh family, but raising ``TracerArrayConversionError`` on the Delaunay mesh family, whose neighbors come
        from a direct scipy call on the traced mesh grid (use ``AdaptSplitPower`` there).

        Parameters
        ----------
        inner_coefficient
            The inner regularization coefficient which controls the degree of smoothing of the inversion
            reconstruction in the inner (high signal) regions of a mesh's reconstruction.
        outer_coefficient
            The outer regularization coefficient which controls the degree of smoothing of the inversion
            reconstruction in the outer (low signal) regions of a mesh's reconstruction.
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions
            to low signal regions.
        power
            The exponent the interpolated coefficient is raised to before the matrix builder squares it, so the
            coefficient enters the regularization matrix at the power ``2 * power``. The default ``1.0`` is the
            ``Constant`` convention; ``2.0`` is the legacy ``Adapt`` convention. This is a convention switch, not
            a model parameter -- the shipped prior config fixes it as a ``Constant`` prior so a search never
            samples it.
        """
        super().__init__(
            inner_coefficient=inner_coefficient,
            outer_coefficient=outer_coefficient,
            signal_scale=signal_scale,
        )

        self.power = power

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        These are the interpolated inner / outer coefficients raised to ``self.power`` (default ``1.0``), as
        opposed to ``Adapt``, which squares them.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        pixel_signals = linear_obj.pixel_signals_from(
            signal_scale=self.signal_scale, xp=xp
        )

        return adapt_regularization_weights_from(
            inner_coefficient=self.inner_coefficient,
            outer_coefficient=self.outer_coefficient,
            pixel_signals=pixel_signals,
            power=self.power,
        )

    def regularization_matrix_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization matrix with shape [pixels, pixels].

        Every mesh edge is scattered once (unlike ``Adapt``), so uniform weights reproduce ``Constant`` exactly.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses this matrix to perform regularization.

        Returns
        -------
        The regularization matrix.
        """
        regularization_weights = self.regularization_weights_from(
            linear_obj=linear_obj, xp=xp
        )

        return weighted_regularization_matrix_single_scatter_from(
            regularization_weights=regularization_weights,
            neighbors=linear_obj.mesh_geometry.neighbors,
            xp=xp,
        )
