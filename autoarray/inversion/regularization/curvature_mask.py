from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.operators import derivative_util


def curvature_reg_matrix_via_mask_from(mask, pixel_scale: float = 1.0) -> np.ndarray:
    """
    The curvature regularization matrix of the unmasked pixels of a
    rectangular masked 2D grid, H = Hxx^T Hxx + Hyy^T Hyy, where Hxx / Hyy are
    forward second-difference operators which degrade gracefully to first /
    zeroth order at the mask edge (see
    ``derivative_util.forward_difference_operators_from``).

    This is the curvature regularization scheme used by the
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
        mask=mask, pixel_scale=pixel_scale, max_order=2
    ).toarray()


class CurvatureMask(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0):
        """
        Curvature regularization on the unmasked pixels of a rectangular
        masked 2D grid.

        Each unmasked pixel is regularized with a forward second-difference
        stencil along both grid directions, degrading to first / zeroth order
        where the mask edge truncates the stencil, penalising curvature in
        the reconstructed solution. This contrasts mapper-based schemes
        (e.g. ``Constant``), which regularize via mesh-neighbour differences:
        here the linear object is defined on a masked rectangular grid and
        must expose its ``mask``.

        This is the curvature regularization scheme of the
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
        return self.coefficient * curvature_reg_matrix_via_mask_from(
            mask=linear_obj.mask
        )
