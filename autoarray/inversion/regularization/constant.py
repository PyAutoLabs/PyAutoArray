from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.regularization.abstract import validate_coefficient


def constant_regularization_matrix_from(
    coefficient: float, neighbors: np.ndarray, neighbors_sizes: np.ndarray, xp=np
) -> np.ndarray:
    """
    From the pixel-neighbors array, setup the regularization matrix using the instance regularization scheme.

    A complete description of regularizatin and the `regularization_matrix` can be found in the `Regularization`
    class in the module `autoarray.inversion.regularization`.

    Memory requirement: 2SP + S^2
    FLOPS: 1 + 2S + 2SP

    Parameters
    ----------
    coefficient
        The regularization coefficients which controls the degree of smoothing of the inversion reconstruction.
    neighbors : ndarray, shape (S, P), dtype=int64
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the Delaunay grid (entries of -1 correspond to no neighbor).
    neighbors_sizes : ndarray, shape (S,), dtype=int64
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Delaunay grid.

    Returns
    -------
    regularization_matrix : ndarray, shape (S, S), dtype=float64
        The regularization matrix computed using Regularization where the effective regularization
        coefficient of every source pixel is the same.
    """
    S, P = neighbors.shape
    # as the regularization matrix is S by S, S would be out of bound (any out of bound index would do)
    OUT_OF_BOUND_IDX = S
    regularization_coefficient = coefficient * coefficient

    # flatten it for feeding into the matrix as j indices
    neighbors = neighbors.flatten()
    # now create the corresponding i indices
    I_IDX = xp.repeat(xp.arange(S), P)
    # Entries of `-1` in `neighbors` (indicating no neighbor) are replaced with an out-of-bounds index.
    # This ensures that JAX can efficiently drop these entries during matrix updates.
    neighbors = xp.where(neighbors == -1, OUT_OF_BOUND_IDX, neighbors)
    diag_vals = 1e-8 + regularization_coefficient * neighbors_sizes

    if xp.__name__.startswith("jax"):
        return (
            xp.diag(diag_vals)
            .at[I_IDX, neighbors]
            .add(-regularization_coefficient, mode="drop", unique_indices=True)
        )
    else:
        mat = xp.diag(diag_vals).copy()
        valid_mask = (neighbors >= 0) & (neighbors < mat.shape[1])
        I_valid = I_IDX[valid_mask]
        neigh_valid = neighbors[valid_mask]

        # scatter-add
        xp.add.at(mat, (I_valid, neigh_valid), -regularization_coefficient)
        return mat


class Constant(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0):
        """
        Regularization which uses the neighbors of the mesh (e.g. shared Delaunay vertexes) and
        a single value to smooth an inversion's solution.

        For this regularization scheme, there is only 1 regularization coefficient that is applied to
        all neighboring pixels / parameters. This means that the matrix B only needs to regularize pixels / parameters
        in one direction (e.g. pixel 0 regularizes pixel 1, but NOT visa versa). For example:

        B = [-1, 1]  [0->1]
            [0, -1]  1 does not regularization with 0

        A small numerical value of 1.0e-8 is added to all elements in constant regularization matrix, to ensure that
        it is positive definite.

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        **JAX & gradient support** (2026-07 gradient sweep): on meshes with an
        analytic neighbor structure (the rectangular family) this scheme is
        JAX-differentiable and FD-certified. On the Delaunay mesh family
        (``Delaunay``, ``KNearestNeighbor``, ``KNNBarycentric``) the neighbors
        come from a direct ``scipy.spatial.Delaunay`` call on the traced
        source-plane mesh grid, so it raises ``TracerArrayConversionError``
        under ``jax.jit`` / ``jax.grad`` — use a split-family scheme
        (e.g. ``ConstantSplit``) there instead.

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        """

        validate_coefficient(coefficient=coefficient)
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

        return constant_regularization_matrix_from(
            coefficient=self.coefficient,
            neighbors=linear_obj.neighbors,
            neighbors_sizes=linear_obj.neighbors.sizes,
            xp=xp,
        )
