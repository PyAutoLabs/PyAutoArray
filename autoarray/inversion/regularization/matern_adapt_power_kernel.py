from __future__ import annotations
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.adapt import adapt_regularization_weights_from
from autoarray.inversion.regularization.matern_adapt_kernel import MaternAdaptKernel


class MaternAdaptPowerKernel(MaternAdaptKernel):
    def __init__(
        self,
        scale: float = 1.0,
        nu: float = 0.5,
        inner_coefficient: float = 1.0,
        outer_coefficient: float = 1.0,
        signal_scale: float = 1.0,
        jitter: Optional[float] = None,
        jitter_relative: bool = False,
        power: float = 1.0,
    ):
        """
        Regularization which uses a Matern smoothing kernel with regularization weights that adapt to the
        brightness of the source being reconstructed, with the coefficient convention of ``Constant``.

        This is the corrected sibling of ``MaternAdaptKernel``. The kernel, the covariance construction and the
        jitter handling are all unchanged -- the only difference is the coefficient convention.

        ``MaternAdaptKernel`` squares its coefficients when interpolating them into per-pixel weights, and those
        weights then enter the kernel covariance as ``C_ij = K(d_ij) * w_i * w_j`` with
        ``w = 1 / regularization_weights``. The regularization matrix ``H = C^-1`` therefore scales as the fourth
        power of the coefficient. This class raises the interpolated coefficient to ``power`` instead, so the
        effective exponent is ``2 * power`` and the default ``power=1.0`` puts the coefficient into ``H`` at
        ``lambda^2`` -- the ``Constant`` convention. Under the shared ``LogUniform(1e-6, 1e6)`` prior the prior
        now spans ``lambda^2``.

        Note that ``MaternKernel`` itself scales *linearly* in its ``coefficient`` (``H = coefficient * C^-1``),
        so neither this class nor ``MaternAdaptKernel`` reduces to it; the reference convention here is
        ``Constant`` / ``AdaptPower``.

        **Migration from ``MaternAdaptKernel``.** The coefficient scale is squared: ``c_new = c_old ** 2``. To
        reproduce the legacy class exactly, pass ``power=2.0``.

        **JAX & gradient support**: as for ``MaternAdaptKernel`` (tfp ``bessel_kve`` gradients; explicit-inverse
        conditioning caveat).

        Parameters
        ----------
        scale
            The typical scale (correlation length) of the Matern regularization kernel.
        nu
            Controls the smoothness (differentiability) of the Matern kernel; ``nu=0.5`` corresponds to an
            exponential (Ornstein-Uhlenbeck) kernel, while a Gaussian covariance is obtained in the limit as
            ``nu`` approaches infinity.
        inner_coefficient
            The inner regularization coefficient which controls the degree of smoothing in the inner (high
            signal) regions of a mesh's reconstruction.
        outer_coefficient
            The outer regularization coefficient which controls the degree of smoothing in the outer (low
            signal) regions of a mesh's reconstruction.
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions
            to low signal regions.
        jitter
            The small value added to the covariance diagonal for numerical stability. ``None`` (default) uses the
            historical value 1e-8.
        jitter_relative
            If ``True`` the jitter is applied *relative* to each pixel's own variance (``C_ii *= 1 + jitter``)
            rather than as a fixed absolute ``jitter * I``. See :func:`apply_jitter`.
        power
            The exponent the interpolated coefficient is raised to, so the coefficient enters the regularization
            matrix at the power ``2 * power``. The default ``1.0`` is the ``Constant`` convention; ``2.0`` is the
            legacy ``MaternAdaptKernel`` convention. This is a convention switch, not a model parameter -- the
            shipped prior config fixes it as a ``Constant`` prior so a search never samples it.
        """
        super().__init__(
            scale=scale,
            nu=nu,
            inner_coefficient=inner_coefficient,
            outer_coefficient=outer_coefficient,
            signal_scale=signal_scale,
            jitter=jitter,
            jitter_relative=jitter_relative,
        )

        self.power = power

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        These are the interpolated inner / outer coefficients raised to ``self.power`` (default ``1.0``), as
        opposed to ``MaternAdaptKernel``, which squares them.

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
