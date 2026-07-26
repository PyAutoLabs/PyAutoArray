from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization


def zeroth_regularization_matrix_from(
    coefficient: float, pixels: int, xp=np
) -> np.ndarray:
    """
    Apply zeroth order regularization which penalizes every pixel's deviation from zero by addiing non-zero terms
    to the regularization matrix.

    A complete description of regularization and the `regularization_matrix` can be found in the `Regularization`
    class in the module `autoarray.inversion.regularization`.

    Parameters
    ----------
    pixels
        The number of pixels in the linear object which is to be regularized, being used to in the inversion.
    coefficient
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.

    Returns
    -------
    np.ndarray
        The regularization matrix computed using Regularization where the effective regularization
        coefficient of every source pixel is the same.
    """

    reg_coeff = coefficient**2.0

    # Identity matrix scaled by reg_coeff does exactly ∑_i reg_coeff * e_i e_i^T

    return xp.eye(pixels) * reg_coeff


class Zeroth(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0):
        """
        A zeroth order regularization scheme which zeroth order regularization to pixels with low expected
        signal values.

        Zeroth order regularization assumes a prior on the solution that its values should be closer to zero,
        penalizing solutions where they deviate further from zero. This is typically applied to prevent solutions
        from going to large positive and negative values that alternate.

        For this regularization scheme, there is only 1 regularization coefficient that is applied to
        all pixels by themselves (e.g. no neighboring scheme is used) For example:

        B = [1, 0]  0 -> 0
            [0, 1]  1 -> 1

        A small numerical value of 1.0e-8 is added to all elements in constant regularization matrix, to ensure that
        it is positive definite.

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        **JAX & gradient support** (2026-07 gradient sweep): neighbor-free and
        pure-xp — JAX-differentiable and FD-certified on both the rectangular
        and KNN mesh families.

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        """

        self.coefficient = coefficient

        super().__init__()

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        The regularization weights define the level of regularization applied to each parameter in the linear object
        (e.g. the ``pixels`` in a ``Mapper``).

        For standard regularization (e.g. ``Constant``) are weights are equal, however for adaptive schemes
        (e.g. ``Adapt``) they vary to adapt to the data being reconstructed.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        return self.coefficient * xp.ones(linear_obj.params)

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
        return zeroth_regularization_matrix_from(
            coefficient=self.coefficient, pixels=linear_obj.params
        )
