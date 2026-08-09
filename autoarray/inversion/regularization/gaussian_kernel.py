from __future__ import annotations
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization


def gauss_cov_matrix_from(
    scale: float,
    pixel_points: np.ndarray,  # shape (N, 2)
    jitter: float = 1e-8,
    jitter_relative: bool = False,
    xp=np,
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
    from autoarray.inversion.regularization.matern_kernel import apply_jitter

    cov = apply_jitter(cov, jitter=jitter, jitter_relative=jitter_relative, xp=xp)

    return cov


class GaussianKernel(AbstractRegularization):
    def __init__(
        self,
        coefficient: float = 1.0,
        scale: float = 1.0,
        jitter: Optional[float] = None,
        jitter_relative: bool = False,
    ):
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
        jitter
            The small value added to the covariance diagonal for numerical stability.
            ``None`` (default) uses the historical value 1e-8 — behaviour is identical
            to not having this parameter (it is a fixed setting, not a free model
            parameter, hence the ``None`` default).
        jitter_relative
            If ``True`` the jitter is applied *relative* to each pixel's own variance
            (``C_ii *= 1 + jitter``) rather than as a fixed absolute ``jitter * I``.
            ``False`` (default) preserves the historical behaviour exactly. The absolute
            convention assumes ``C_ii ~ 1``, which holds for this unweighted kernel but not
            for the adaptive one; see :func:`apply_jitter` for why and when to switch.
        """
        self.coefficient = coefficient
        self.scale = scale
        self.jitter = jitter
        self.jitter_relative = jitter_relative
        super().__init__()

    @property
    def jitter_value(self) -> float:
        return 1e-8 if self.jitter is None else self.jitter

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
        from autoarray.inversion.regularization.matern_kernel import inv_via_cholesky

        covariance_matrix = gauss_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            jitter=self.jitter_value,
            jitter_relative=self.jitter_relative,
            xp=xp,
        )

        # The SPD inverse via Cholesky (as MaternKernel) — identical quantity to
        # inv(), with better accuracy and symmetry on the SPD covariance.
        regularization_matrix = self.coefficient * inv_via_cholesky(
            covariance_matrix, xp=xp
        )

        # The inverse can still lose exact symmetry and introduce tiny negative
        # eigenvalues when the covariance matrix is near-singular (e.g. scale >>
        # pixel spacing). Symmetrise and add a trace-scaled diagonal jitter so the
        # downstream cholesky in log_det_regularization_matrix_term cannot fail on
        # floating-point noise.
        regularization_matrix = 0.5 * (regularization_matrix + regularization_matrix.T)
        N = regularization_matrix.shape[0]
        diag_mean = xp.mean(xp.diag(regularization_matrix))
        h_jitter = 1e-8 * xp.abs(diag_mean)
        regularization_matrix = (
            regularization_matrix
            + xp.eye(N, dtype=regularization_matrix.dtype) * h_jitter
        )

        return regularization_matrix

    def log_det_regularization_matrix_term_from(
        self, linear_obj: LinearObj, xp=np
    ) -> float:
        """
        The analytically exact ``log det H`` from a single Cholesky of the kernel
        covariance: ``H = coefficient * C^-1``, so
        ``log det H = pixels * log(coefficient) - 2 * sum(log(diag(cholesky(C))))``.

        This is the log-determinant of the analytic ``coefficient * C^-1`` — it
        deliberately excludes the trace-scaled stabilisation jitter that
        :meth:`regularization_matrix_from` adds to the formed matrix (that jitter
        exists only to guard the factorization of the explicit inverse, which this
        shortcut avoids entirely).

        Consumed by the inversion only when ``Settings.log_det_method == "slogdet"``
        (see :meth:`AbstractRegularization.log_det_regularization_matrix_term_from`);
        the default evidence path factorizes the formed ``H`` and is unchanged.
        """
        covariance_matrix = gauss_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            jitter=self.jitter_value,
            jitter_relative=self.jitter_relative,
            xp=xp,
        )

        log_det_covariance = 2.0 * xp.sum(
            xp.log(xp.diag(xp.linalg.cholesky(covariance_matrix)))
        )

        return linear_obj.params * np.log(self.coefficient) - log_det_covariance

    def regularization_term_from(
        self, linear_obj: LinearObj, reconstruction: np.ndarray, xp=np
    ) -> float:
        """
        The regularization term ``s^T H s`` from a single Cholesky solve of the kernel
        covariance: ``H = coefficient * C^-1``, so
        ``s^T H s = coefficient * s^T C^-1 s``, with the quadratic form evaluated by
        solving ``C x = s`` rather than by forming ``C^-1``.

        As with :meth:`log_det_regularization_matrix_term_from`, this is the term of
        the analytic ``coefficient * C^-1``: it excludes both the symmetrisation and
        the trace-scaled stabilisation jitter that :meth:`regularization_matrix_from`
        applies to the formed matrix, since both exist only to guard the
        factorization of the explicit inverse that this shortcut avoids entirely.

        Consumed by the inversion only when
        ``Settings.regularization_term_method == "cho_solve"`` (see
        :meth:`AbstractRegularization.regularization_term_from`); the default
        ``"matmul"`` path contracts the formed ``H`` and is unchanged.
        """
        from autoarray.inversion.regularization.matern_kernel import (
            quadratic_form_via_cholesky,
        )

        covariance_matrix = gauss_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            jitter=self.jitter_value,
            jitter_relative=self.jitter_relative,
            xp=xp,
        )

        return self.coefficient * quadratic_form_via_cholesky(
            covariance_matrix, reconstruction, xp=xp
        )
