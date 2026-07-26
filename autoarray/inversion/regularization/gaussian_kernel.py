from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization


def gauss_cov_matrix_from(
    scale: float, pixel_points: np.ndarray, xp=np  # shape (N, 2)
) -> np.ndarray:
    """
    Construct the source‐pixel Gaussian covariance matrix for regularization.

    For N source‐pixels at coordinates (y_i, x_i), we define

      C_ij = exp( -||p_i - p_j||^2 / (2 scale^2) )

    plus a tiny diagonal “jitter” (1e-8) to ensure numerical stability.

    Parameters
    ----------
    scale
        The characteristic length scale of the Gaussian kernel.
    pixel_points
        Array of shape (N, 2), giving the (y, x) coordinates of each source pixel.

    Returns
    -------
    cov : np.ndarray, shape (N, N)
        The Gaussian covariance matrix.
    """
    # Ensure array:
    pts = pixel_points  # (N, 2)
    # Compute squared distances: ||p_i - p_j||^2
    diffs = pts[:, None, :] - pts[None, :, :]  # (N, N, 2)
    d2 = xp.sum(diffs**2, axis=-1)  # (N, N)

    # Gaussian kernel
    cov = xp.exp(-d2 / (2.0 * scale**2))  # (N, N)

    # Add tiny jitter on the diagonal
    N = pts.shape[0]
    cov = cov + xp.eye(N, dtype=cov.dtype) * 1e-8

    return cov


class GaussianKernel(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0, scale: float = 1.0):
        """
        Regularization which uses a Gaussian smoothing kernel to regularize the solution.

        For this regularization scheme, every pixel is regularized with every other pixel. This contrasts many other
        schemes, where regularization is based on neighboring (e.g. do the pixels share a Delaunay edge?) or computing
        derivates around the center of the pixel (where nearby pixels are regularization locally in similar ways).

        This makes the regularization matrix fully dense and therefore maybe change the run times of the solution.
        It also leads to more overall smoothing which can lead to more stable linear inversions.

        This scheme is introduced by Vernardos et al. (2022): https://arxiv.org/abs/2202.09378

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        **JAX & gradient support** (2026-07 gradient sweep): xp-threaded and
        JAX-differentiable end-to-end. Caveat: the regularization matrix is an
        explicit dense inverse of the kernel covariance, whose conditioning on
        clustered mesh vertices puts a small numerical noise floor on the
        likelihood (see ``MaternKernel`` for the measured detail).

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        scale
            The typical scale of the exponential regularization pattern.
        """
        self.coefficient = coefficient
        self.scale = scale
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
        covariance_matrix = gauss_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            xp=xp,
        )

        regularization_matrix = self.coefficient * xp.linalg.inv(covariance_matrix)

        # inv() loses exact symmetry and can introduce tiny negative eigenvalues
        # when the covariance matrix is near-singular (e.g. scale >> pixel
        # spacing). Symmetrise and add a trace-scaled diagonal jitter so the
        # downstream cholesky in log_det_regularization_matrix_term cannot
        # fail on floating-point noise.
        regularization_matrix = 0.5 * (regularization_matrix + regularization_matrix.T)
        N = regularization_matrix.shape[0]
        diag_mean = xp.mean(xp.diag(regularization_matrix))
        jitter = 1e-8 * xp.abs(diag_mean)
        regularization_matrix = regularization_matrix + xp.eye(
            N, dtype=regularization_matrix.dtype
        ) * jitter

        return regularization_matrix
