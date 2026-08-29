from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.adapt import adapt_regularization_weights_from
from autoarray.inversion.regularization.adapt_split_zeroth import AdaptSplitZeroth


class AdaptSplitZerothPower(AdaptSplitZeroth):
    is_split_regularization = True

    def __init__(
        self,
        zeroth_coefficient: float = 1.0,
        zeroth_signal_scale: float = 1.0,
        inner_coefficient: float = 1.0,
        outer_coefficient: float = 1.0,
        signal_scale: float = 1.0,
        power: float = 1.0,
    ):
        """
        Regularization which uses zeroth order regularization, the derivatives at a cross of four points around
        each pixel centre and values adapted to the data being fitted to smooth an inversion's solution, with the
        coefficient convention of ``ConstantSplit``.

        This is the corrected sibling of ``AdaptSplitZeroth``. The split geometry, the zeroth-order leg and the
        matrix builders are all unchanged -- the only difference is the convention of the split leg's
        ``inner_coefficient`` / ``outer_coefficient``:

        ``AdaptSplitZeroth`` squares them twice (once when interpolating them into per-pixel weights, once in the
        matrix builder), so the split leg scales as the fourth power of the coefficient while ``ConstantSplit``
        scales as the second. This class raises the interpolated coefficient to ``power`` before the builder
        squares it, so the effective exponent is ``2 * power`` and the default ``power=1.0`` matches
        ``ConstantSplit``. Under the shared ``LogUniform(1e-6, 1e6)`` prior the prior now spans ``lambda^2``, and
        the regularization matrix stays positive-definite to ``c ~ 1e6`` rather than collapsing from ``c ~ 1e4``.

        The **zeroth leg is unaffected**: ``BrightnessZeroth`` already squares its ``zeroth_coefficient`` exactly
        once, so ``zeroth_coefficient`` keeps its meaning between the two classes.

        **Migration from ``AdaptSplitZeroth``.** The split leg's coefficient scale is squared:
        ``c_new = c_old ** 2``; ``zeroth_coefficient`` is unchanged. To reproduce the legacy class exactly, pass
        ``power=2.0``.

        **JAX & gradient support**: as for ``AdaptSplitZeroth`` -- FD-certified on the Delaunay mesh family (KNN
        meshes), structurally incompatible with the rectangular meshes.

        Parameters
        ----------
        zeroth_coefficient
            The regularization coefficient of the zeroth-order leg, which is squared exactly once and is
            therefore unchanged from ``AdaptSplitZeroth``.
        zeroth_signal_scale
            A factor which controls how rapidly the zeroth-order regularization varies from high signal regions
            to low signal regions.
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
            coefficient enters the split leg of the regularization matrix at the power ``2 * power``. The default
            ``1.0`` is the ``ConstantSplit`` convention; ``2.0`` is the legacy ``AdaptSplitZeroth`` convention.
            This is a convention switch, not a model parameter -- the shipped prior config fixes it as a
            ``Constant`` prior so a search never samples it.
        """
        super().__init__(
            zeroth_coefficient=zeroth_coefficient,
            zeroth_signal_scale=zeroth_signal_scale,
            inner_coefficient=inner_coefficient,
            outer_coefficient=outer_coefficient,
            signal_scale=signal_scale,
        )

        self.power = power

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        These are the interpolated inner / outer coefficients raised to ``self.power`` (default ``1.0``), as
        opposed to ``AdaptSplitZeroth``, which squares them.

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
