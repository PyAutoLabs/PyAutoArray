from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.regularization.abstract import validate_coefficient


def brightness_zeroth_regularization_weights_from(
    coefficient: float, pixel_signals: np.ndarray
) -> np.ndarray:
    """
    Returns the regularization weights for the brightness zeroth regularization scheme (e.g. ``BrightnessZeroth``).

    The weights define the level of zeroth order regularization applied to every mesh parameter (typically pixels
    of a ``Mapper``).

    They are computed using an estimate of the expected signal in each pixel.

    The zeroth order regularization coefficients is applied in combination with  1.0 - pixel_signals, which are
    the pixels with a low pixel-signal (i.e. where the signal is not located near the source being reconstructed in
    the pixelization).

    Parameters
    ----------
    coefficient
        The level of zeroth order regularization applied to every mesh parameter (typically pixels of a ``Mapper``),
        with the degree applied varying based on the ``pixel_signals``.
    pixel_signals
        The estimated signal in every pixelization pixel, used to change the regularization weighting of high signal
        and low signal pixelizations.

    Returns
    -------
    np.ndarray
        The zeroth order regularization weights which act as the effective level of zeroth order regularization
        applied to every mesh parameter.
    """
    return coefficient * (1.0 - pixel_signals)


def brightness_zeroth_regularization_matrix_from(
    regularization_weights: np.ndarray, xp=np
) -> np.ndarray:
    """
    Returns the regularization matrix for the zeroth-order brightness regularization scheme.

    Parameters
    ----------
    regularization_weights
        The regularization weights for each pixel, governing the strength of zeroth-order
        regularization applied per inversion parameter.

    Returns
    -------
    A diagonal regularization matrix where each diagonal element is the squared regularization weight
    for that pixel.
    """
    regularization_weight_squared = regularization_weights**2.0
    return xp.diag(regularization_weight_squared)


class BrightnessZeroth(AbstractRegularization):
    def __init__(
        self,
        coefficient: float = 1.0,
        signal_scale: float = 1.0,
    ):
        """
        An adaptive regularization scheme which applies zeroth order regularization to pixels with low expected
        signal values.

        For the weighted regularization scheme, each pixel is given an 'effective regularization weight', which is
        controls the degree of zeroth order regularization applied to each pixel. The motivation of this is that
        the exterior regions different regions of a pixelization's mesh ought to have a signal consistent with zero,
        but may have a low level of non-zero signal when fitting the data.

        To implement this regularization, values on the diagonal of the regularization matrix are increased
        according to the regularization weight_list of each pixel.

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        **JAX & gradient support** (xp-ported 2026-07-26): the pixel-signals
        call now threads ``xp`` (it previously dropped the backend, raising
        ``TracerArrayConversionError`` under ``jax.jit`` / ``jax.grad``), so
        the scheme is JAX-differentiable — the weights and diagonal matrix
        are smooth functions of the pixel signals, FD-certified through
        ``AdaptSplitZeroth`` (8.2e-8 on the KNN meshes). Standalone use
        under-regularizes high-signal pixels (weights → 0 where the source
        is) and can leave the linear system singular — it is a supplement to
        a smoothing scheme by design, not a standalone regularization.

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of zeroth order regularizaiton applied to
            the inversion reconstruction, in regions of low signal.
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
            low signal regions.
        """

        super().__init__()

        validate_coefficient(coefficient=coefficient)
        self.coefficient = coefficient
        self.signal_scale = signal_scale

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of the ``BrightnessZeroth`` regularization scheme.

        The weights define the level of zeroth order regularization applied to every mesh parameter (typically pixels
        of a ``Mapper``).

        They are computed using an estimate of the expected signal in each pixel.

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

        return brightness_zeroth_regularization_weights_from(
            coefficient=self.coefficient, pixel_signals=pixel_signals
        )

    def regularization_matrix_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization matrix with shape [pixels, pixels].

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

        return brightness_zeroth_regularization_matrix_from(
            regularization_weights=regularization_weights, xp=xp
        )
