from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization


def exp_cov_matrix_from(
    scale: float,
    pixel_points: np.ndarray,  # shape (N, 2)
    xp=np,
) -> np.ndarray:  # shape (N, N)
    """
    Construct the source brightness covariance matrix using an exponential kernel:

        cov[i,j] = exp(- d_{ij} / scale)

    with a tiny jitter 1e-8 added on the diagonal for numerical stability.

    The pairwise distances use the ``||x||^2 + ||y||^2 - 2 x.y`` identity with a
    ``sqrt(d^2 + 1e-20)`` floor (mirroring ``matern_cov_matrix_from``) rather than
    ``linalg.norm`` of an (N, N, 2) difference cube: the norm's derivative is NaN at
    the zero diagonal, which would poison every JAX gradient through this kernel.

    Parameters
    ----------
    scale
        The length‐scale of the exponential kernel.
    pixel_points
        Array of shape (N, 2) giving the (y,x) coordinates of each source‐plane pixel.
    xp
        Backend (numpy or jax.numpy).

    Returns
    -------
    np.ndarray, shape (N, N)
        The exponential covariance matrix.
    """
    pts = xp.asarray(pixel_points)

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    x2 = xp.sum(pts * pts, axis=1, keepdims=True)  # (N, 1)
    dist_sq = x2 + x2.T - 2.0 * (pts @ pts.T)  # (N, N)
    dist_sq = xp.maximum(dist_sq, 0.0)  # numerical safety

    d = xp.sqrt(dist_sq + 1e-20)  # (N, N)

    # exponential kernel
    cov = xp.exp(-d / scale)

    # add a small jitter on the diagonal
    N = pts.shape[0]
    cov = cov + xp.eye(N, dtype=cov.dtype) * 1e-8

    return cov


class ExponentialKernel(AbstractRegularization):
    def __init__(self, coefficient: float = 1.0, scale: float = 1.0):
        """
        Regularization which uses an Exponential smoothing kernel to regularize the solution.

        For this regularization scheme, every pixel is regularized with every other pixel. This contrasts many other
        schemes, where regularization is based on neighboring (e.g. do the pixels share a Delaunay edge?) or computing
        derivates around the center of the pixel (where nearby pixels are regularization locally in similar ways).

        This makes the regularization matrix fully dense and therefore maybe change the run times of the solution.
        It also leads to more overall smoothing which can lead to more stable linear inversions.

        This scheme is introduced by Vernardos et al. (2022): https://arxiv.org/abs/2202.09378

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        **JAX & gradient support** (2026-07): xp-threaded and
        JAX-differentiable end-to-end — the pairwise distances use the
        NaN-safe ``sqrt(d^2 + 1e-20)`` form (see ``exp_cov_matrix_from``).
        Caveat: the regularization matrix is an explicit dense inverse of the
        kernel covariance, whose conditioning on clustered mesh vertices puts
        a small numerical noise floor on the likelihood (see ``MaternKernel``
        for the measured detail).

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
        covariance_matrix = exp_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            xp=xp,
        )

        return self.coefficient * xp.linalg.inv(covariance_matrix)
