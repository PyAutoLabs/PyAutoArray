from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.regularization.abstract import validate_coefficient
from autoarray.operators import derivative_util


def fourth_order_reg_matrix_via_mask_from(
    mask, pixel_scale: float = 1.0
) -> np.ndarray:
    """
    The fourth-order regularization matrix of the unmasked pixels of a
    rectangular masked 2D grid, H = H4x^T H4x + H4y^T H4y, where H4x / H4y
    are forward fourth-difference operators which degrade gracefully to
    third / second / first / zeroth order at the mask edge (see
    ``derivative_util.forward_difference_operators_from``).

    This is the fourth-order regularization scheme used by the
    gravitational-imaging (potential correction) technique of Cao et al. 2025
    (https://github.com/caoxiaoyue/lensing_potential_correction; cite via
    https://github.com/caoxiaoyue/potential_correction_paper), applied to the
    pixelized corrections of the lensing potential defined on a coarse
    rectangular mesh.

    Parameters
    ----------
    mask
        The 2D bool mask (``True`` = masked) of the rectangular grid whose
        unmasked pixels are regularized.
    pixel_scale
        The finite-difference step; regularization matrices are
        conventionally built with 1.0, the coefficient absorbing the scale.

    Returns
    -------
    The [n_unmasked, n_unmasked] regularization matrix.
    """
    return derivative_util.forward_difference_reg_matrix_from(
        mask=mask, pixel_scale=pixel_scale, max_order=4
    ).toarray()


class FourthOrderMask(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0):
        """
        Fourth-order regularization on the unmasked pixels of a rectangular
        masked 2D grid.

        Each unmasked pixel is regularized with a forward fourth-difference
        stencil along both grid directions, degrading to third / second /
        first / zeroth order where the mask edge truncates the stencil.
        Penalising the fourth derivative permits solutions with curvature
        (e.g. localised perturbations) while still suppressing high-frequency
        noise, making it a weaker prior than ``CurvatureMask``. The linear
        object regularized must be defined on a masked rectangular grid and
        expose its ``mask``.

        This is the fourth-order regularization scheme of the
        gravitational-imaging (potential correction) technique, applied to
        pixelized corrections of the lensing potential; it is ported from the
        ``potential_correction`` package of Cao et al. 2025
        (https://github.com/caoxiaoyue/lensing_potential_correction). If you
        use it in your research, please cite Cao et al. 2025; citation
        materials are provided at
        https://github.com/caoxiaoyue/potential_correction_paper.

        Parameters
        ----------
        coefficient
            The regularization coefficient which multiplies the matrix,
            setting the strength of the smoothing.
        """
        validate_coefficient(coefficient=coefficient)
        self.coefficient = coefficient

        super().__init__()

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme,
        which are equal for every parameter.

        Parameters
        ----------
        linear_obj
            The linear object which uses these weights when performing
            regularization.
        """
        return self.coefficient * xp.ones(linear_obj.params)

    def regularization_matrix_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization matrix with shape [pixels, pixels].

        Parameters
        ----------
        linear_obj
            The linear object which uses this matrix to perform
            regularization. It must expose the ``mask`` of the rectangular
            masked 2D grid its parameters are defined on.

        Returns
        -------
        The regularization matrix.
        """
        return self.coefficient * fourth_order_reg_matrix_via_mask_from(
            mask=linear_obj.mask
        )
