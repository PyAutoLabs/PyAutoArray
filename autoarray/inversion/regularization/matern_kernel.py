from __future__ import annotations
import numpy as np
import math
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization


def kv_xp(v, z, xp=np):
    """
    XP-compatible modified Bessel K_v(v, z).

    NumPy backend:
        -> scipy.special.kv

    JAX backend:
        -> tensorflow_probability.substrates.jax.math.bessel_kve * exp(-|z|)

    `jax.scipy.special` has no modified-Bessel-K (`kv`/`kve`) of arbitrary real
    order, so the JAX path relies on tfp's `bessel_kve`. Note the tfp dependency
    is the *nightly* build (`tfp-nightly`): the last stable release
    (`tensorflow-probability==0.25.0`) crashes at import under the resolved
    `jax>=0.7` stack (it references `jax.interpreters.xla.pytype_aval_mappings`,
    removed from modern JAX).
    """

    # -------------------------
    # NumPy backend
    # -------------------------
    if xp is np:
        import scipy.special as sc

        return sc.kv(v, z)

    # -------------------------
    # JAX backend
    # -------------------------
    else:
        try:
            import tensorflow_probability.substrates.jax as tfp

            return tfp.math.bessel_kve(v, z) * xp.exp(-xp.abs(z))
        except ImportError:
            raise ImportError(
                "To use the JAX backend with the Matérn kernel, install the "
                "tensorflow-probability nightly via "
                "`pip install tfp-nightly==0.26.0.dev20260713` "
                "(the last stable release, 0.25.0, is incompatible with modern JAX)."
            )


def gamma_xp(x, xp=np):
    """
    XP-compatible Gamma(x).
    """
    if xp is np:
        import scipy.special as sc

        return sc.gamma(x)
    else:
        import jax.scipy.special as jsp

        return jsp.gamma(x)


def matern_kernel(r, l: float = 1.0, v: float = 0.5, xp=np):
    """
    XP-compatible Matérn kernel.
    Works with NumPy or JAX.
    """

    # Avoid r = 0 singularity (JAX-safe)
    r = xp.maximum(xp.abs(r), 1e-8)

    z = xp.sqrt(2.0 * v) * r / l

    part1 = 2.0 ** (1.0 - v) / gamma_xp(v, xp)  # scalar constant
    part2 = z**v
    part3 = kv_xp(v, z, xp)

    return part1 * part2 * part3


def matern_cov_matrix_from(
    scale: float,
    nu: float,
    pixel_points,
    weights=None,
    jitter: float = 1e-8,
    jitter_relative: bool = False,
    xp=np,
):
    """
    Construct the regularization covariance matrix (N x N) using a Matérn kernel,
    optionally modulated by per-pixel weights.

    If `weights` is provided (shape [N]), the covariance is:
        C_ij = K(d_ij; scale, nu) * w_i * w_j
    with a small diagonal jitter added for numerical stability.

    Parameters
    ----------
    scale
        Typical correlation length of the Matérn kernel.
    nu
        Smoothness parameter of the Matérn kernel.
    pixel_points
        Array-like of shape [N, 2] with (y, x) coordinates (or any 2D coords; only distances matter).
    weights
        Optional array-like of shape [N]. If None, treated as all ones.
    jitter
        The small value added to the covariance diagonal for numerical stability.
    jitter_relative
        If ``True`` the jitter is scaled by each pixel's own variance (``C_ii *= 1 + jitter``)
        rather than added as a fixed absolute value; see :func:`apply_jitter`.
    xp
        Backend (numpy or jax.numpy).

    Returns
    -------
    covariance_matrix
        Array of shape [N, N].
    """

    # --------------------------------
    # Pairwise distances WITHOUT (N,N,2) diff
    # --------------------------------
    pts = xp.asarray(pixel_points)

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    x2 = xp.sum(pts * pts, axis=1, keepdims=True)  # (N, 1)
    dist_sq = x2 + x2.T - 2.0 * (pts @ pts.T)  # (N, N)
    dist_sq = xp.maximum(dist_sq, 0.0)  # numerical safety

    d_ij = xp.sqrt(dist_sq + 1e-20)  # (N, N)

    # --------------------------------
    # Base Matérn covariance
    # --------------------------------
    covariance_matrix = matern_kernel(d_ij, l=scale, v=nu, xp=xp)  # (N, N)

    # --------------------------------
    # Apply weights: C_ij *= w_i * w_j
    # --------------------------------
    if weights is not None:
        w = xp.asarray(weights)
        covariance_matrix = covariance_matrix * (w[:, None] * w[None, :])

    # --------------------------------
    # Add diagonal jitter (JAX-safe)
    # --------------------------------
    return apply_jitter(
        covariance_matrix, jitter=jitter, jitter_relative=jitter_relative, xp=xp
    )


def apply_jitter(
    covariance_matrix, jitter: float, jitter_relative: bool = False, xp=np
):
    """
    Add the stabilisation jitter to a kernel covariance's diagonal.

    Two conventions, selected by ``jitter_relative``:

    - ``False`` (default, historical) — an **absolute** ``jitter * I``. This is only
      meaningful when the covariance's diagonal is itself ~1, which holds for the
      unweighted kernels (``K(0) == 1``) but *not* for the weighted ones, where
      ``C_ii = w_i^2`` spans the dynamic range of the adaptive weights.
    - ``True`` — a **relative** ``jitter * diag(diag(C))``, i.e. ``C_ii *= (1 + jitter)``.
      Writing ``C = D^(1/2) R D^(1/2)`` for the correlation matrix ``R`` (unit diagonal),
      this is exactly ``D^(1/2) (R + jitter I) D^(1/2)`` — the jitter is applied to the
      *correlation* matrix, so every pixel receives the same relative protection whatever
      its own scale, and the conditioning of the correlation part is bounded exactly as in
      the absolute convention.

    The absolute convention silently breaks down on ``MaternAdaptKernel`` at wide
    adaptive-weight ranges: with ``inner_coefficient=0.1``/``outer_coefficient=100`` the
    faintest pixel's ``C_ii`` reaches ``1e-8``, at which point a fixed ``1e-8`` jitter is
    100% of that pixel's variance and its kernel structure is destroyed. Under the
    relative convention that pixel is perturbed by ``1e-8`` relative, like every other.

    Parameters
    ----------
    covariance_matrix
        The (N, N) kernel covariance, before jitter.
    jitter
        The jitter magnitude — absolute or relative per the flag below.
    jitter_relative
        If ``True`` scale the jitter by each pixel's own variance; if ``False`` (default)
        add it as a fixed absolute value.
    xp
        Backend (numpy or jax.numpy).

    Returns
    -------
    The covariance matrix with the jitter added to its diagonal.
    """
    if jitter_relative:
        return covariance_matrix + jitter * xp.diag(xp.diag(covariance_matrix))

    pixels = covariance_matrix.shape[0]

    return covariance_matrix + jitter * xp.eye(pixels, dtype=covariance_matrix.dtype)


def inv_via_cholesky(C, xp=np):
    # NumPy
    if xp is np:
        import scipy.linalg as la

        cho = la.cho_factor(C, lower=True, check_finite=False)
        I = np.eye(C.shape[0], dtype=C.dtype)
        return la.cho_solve(cho, I, check_finite=False)

    # JAX
    import jax.scipy.linalg as jla

    L = xp.linalg.cholesky(C)
    I = xp.eye(C.shape[0], dtype=C.dtype)
    return jla.cho_solve((L, True), I)


def quadratic_form_via_cholesky(C, s, xp=np):
    """
    The quadratic form ``s^T C^-1 s`` evaluated through a single Cholesky solve.

    This is the implicit counterpart of ``s @ inv_via_cholesky(C) @ s``: it never
    forms ``C^-1``, solving ``C x = s`` for the single right-hand side ``s`` instead
    of the ``N`` right-hand sides of the identity. Both accuracy and cost improve —
    the explicit inverse's round-off is amplified by ``cond(C)``, which reaches ~1e9
    on the clustered traced vertices of the kNN mesh families, and it performs ``N``
    triangular solves where one suffices.

    Parameters
    ----------
    C
        The (N, N) symmetric positive-definite kernel covariance matrix.
    s
        The (N,) vector the quadratic form is taken over (the reconstruction).
    xp
        Backend (numpy or jax.numpy).

    Returns
    -------
    The scalar ``s^T C^-1 s``.
    """
    # NumPy
    if xp is np:
        import scipy.linalg as la

        cho = la.cho_factor(C, lower=True, check_finite=False)
        return s @ la.cho_solve(cho, s, check_finite=False)

    # JAX
    import jax.scipy.linalg as jla

    L = xp.linalg.cholesky(C)
    return s @ jla.cho_solve((L, True), s)


class MaternKernel(AbstractRegularization):
    def __init__(
        self,
        coefficient: float = 1.0,
        scale: float = 1.0,
        nu: float = 0.5,
        jitter: Optional[float] = None,
        jitter_relative: bool = False,
    ):
        """
        Regularization which uses a Matern smoothing kernel to regularize the solution.

        For this regularization scheme, every pixel is regularized with every other pixel. This contrasts many other
        schemes, where regularization is based on neighboring (e.g. do the pixels share a Delaunay edge?) or computing
        derivates around the center of the pixel (where nearby pixels are regularization locally in similar ways).

        This makes the regularization matrix fully dense and therefore maybe change the run times of the solution.
        It also leads to more overall smoothing which can lead to more stable linear inversions.

        This scheme is not used by Vernardos et al. (2022): https://arxiv.org/abs/2202.09378, but it follows
        a similar approach.

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        **JAX & gradient support** (2026-07 gradient sweep): the JAX path
        evaluates the modified Bessel ``K_nu`` through
        ``tensorflow_probability.substrates.jax.math.bessel_kve`` (requires
        ``tfp-nightly``; see ``kv_xp``), which ships a registered gradient
        with respect to its argument (``nu`` is a static float), and the
        covariance inverse differentiates through the Cholesky — gradients
        flow end-to-end and are FD-certified on the rectangular mesh at
        ``nu=2.5``. Caveat: the regularization matrix is an explicit dense
        inverse, so on clustered mesh vertices (e.g. the KNN meshes' traced
        vertices, where cond(C) can reach ~1e9) it puts a ~1e-6 absolute
        numerical noise floor on the likelihood.

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of smooth of the inversion reconstruction.
        scale
            The typical scale of the exponential regularization pattern.
        nu
            Controls the derivative of the regularization pattern (`nu=0.5` is a Gaussian).
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
        self.nu = nu
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
        covariance_matrix = matern_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            nu=self.nu,
            jitter=self.jitter_value,
            jitter_relative=self.jitter_relative,
            xp=xp,
        )

        return self.coefficient * inv_via_cholesky(covariance_matrix, xp=xp)

    def log_det_regularization_matrix_term_from(
        self, linear_obj: LinearObj, xp=np
    ) -> float:
        """
        The analytically exact ``log det H`` from a single Cholesky of the kernel
        covariance: ``H = coefficient * C^-1``, so
        ``log det H = pixels * log(coefficient) - 2 * sum(log(diag(cholesky(C))))``.

        Consumed by the inversion only when ``Settings.log_det_method == "slogdet"``
        (see :meth:`AbstractRegularization.log_det_regularization_matrix_term_from`);
        the default evidence path factorizes the formed ``H`` and is unchanged.
        """
        covariance_matrix = matern_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            nu=self.nu,
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

        Consumed by the inversion only when
        ``Settings.regularization_term_method == "cho_solve"`` (see
        :meth:`AbstractRegularization.regularization_term_from`); the default
        ``"matmul"`` path contracts the formed ``H`` and is unchanged.
        """
        covariance_matrix = matern_cov_matrix_from(
            scale=self.scale,
            pixel_points=linear_obj.source_plane_mesh_grid.array,
            nu=self.nu,
            jitter=self.jitter_value,
            jitter_relative=self.jitter_relative,
            xp=xp,
        )

        return self.coefficient * quadratic_form_via_cholesky(
            covariance_matrix, reconstruction, xp=xp
        )
