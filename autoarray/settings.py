import logging
from typing import Optional

from autonerves import conf

logging.basicConfig()
logger = logging.getLogger(__name__)


class Settings:
    def __init__(
        self,
        use_mixed_precision: bool = False,
        use_positive_only_solver: Optional[bool] = None,
        use_edge_zeroed_pixels: Optional[bool] = None,
        use_border_relocator: Optional[bool] = None,
        no_regularization_add_to_curvature_diag_value: float = None,
        nnls_solver_tol: Optional[float] = None,
        nnls_max_iter: Optional[int] = None,
        log_det_method: Optional[str] = None,
    ):
        """
        The settings of an Inversion, customizing how a linear set of equations are solved for.

        An Inversion is used to reconstruct a dataset, for example the luminous emission of a galaxy.

        Parameters
        ----------
        use_mixed_precision
            If `True`, a targeted subset of the inversion's linear algebra runs in single precision (float32 /
            complex64) instead of double precision (float64 / complex128). This is intended to reduce VRAM use and
            speed up the FFT-heavy and bandwidth-bound steps on GPU and CPU; only the JAX (`xp=jnp`) paths honor
            the flag — the NumPy backend always runs in fp64.

            Paths that honor the flag:

            - PSF FFT convolution in :meth:`Convolver.convolved_image_from` (the light-profile blurring path,
              used by linear MGE bases and similar): the input image, kernel multiply and inverse FFT all run in
              complex64 / float32 end to end. This is the headline GPU win for MGE imaging pipelines.
            - PSF FFT convolution in :meth:`Convolver.convolved_mapping_matrix_from` (the pixelization mapping
              matrix path): the input cube is fp32 and the forward ``rfft2`` runs in complex64, but the kernel
              multiply intentionally upcasts back to complex128 so the inverse FFT and downstream linear
              algebra stay fp64. Pixelization meshes with K ≫ 40 source pixels accumulate enough fp32
              round-off through NNLS / log-determinant to shift ``figure_of_merit`` by O(1) units; the upcast
              preserves precision while the cheaper fp32 scatter and forward FFT are kept.
            - The mapping matrix native cube allocation in
              :func:`autoarray.inversion.mappers.mapper_util.mapping_matrix_from` — output dtype becomes fp32.
            - The internal compute dtype of the curvature matrix accumulation in
              :func:`autoarray.inversion.inversion.inversion_util.curvature_matrix_via_mapping_matrix_from` —
              the noise-weighted ``A.T @ A`` is formed in fp32 then cast to fp64 for downstream stability.

            Empirical platform notes:

            - **GPU**: full pipeline single-JIT roughly matches the fp64 baseline; vmap-batched evaluation
              (the production sampler hot path) shows 25–30% speedup on RTX 2060-class hardware.
            - **CPU**: the per-call FFT itself is ~1.6× faster in fp32, but JAX/XLA's CPU FFT lowering does
              not always re-compose well across ~40-call MGE-basis pipelines, so the single-JIT measurement
              can be neutral or slightly slower than fp64. vmap remains comparable to or slightly faster than
              fp64. The flag is most beneficial for GPU users.

            Paths that intentionally stay in fp64:

            - The NNLS reconstruction (jaxnnls / Cholesky factor + cho_solve) in
              :func:`autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from`. Active-set
              and PDIP solvers are sensitive to fp32 noise on ill-conditioned source meshes.
            - The log-determinant of the curvature regularization matrix used by ``figure_of_merit``: condition
              numbers can exceed 1e6 on fine pixelizations and fp32 silently loses 1+ digit there.
            - Light profile evaluation on the (over-)sampled grid; only the resulting mapping matrix is downcast.

            Empirical numerical impact on the MGE imaging regression (HST-shaped, 15k masked pixels, 40 linear
            Gaussians): Δlog-likelihood ≈ 1e-4 absolute at log-likelihood ≈ 27,400. Well below the natural χ²
            sampling noise floor (σ ≈ √(2N) ≈ 175). Pixelization paths with K ≫ 40 source pixels are more
            sensitive — verify on representative integration tests before turning on for production fits.

            If `False` (default), all paths run in fp64.
        use_positive_only_solver
            Whether to use a positive-only linear system solver, which requires that every reconstructed value is
            positive but is computationally much slower than the default solver (which allows for positive and
            negative values).
        use_border_relocator
            If `True`, all coordinates of all pixelization source mesh grids have pixels outside their border
            relocated to their edge.
        no_regularization_add_to_curvature_diag_value
            If a linear func object does not have a corresponding regularization, this value is added to its
            diagonal entries of the curvature regularization matrix to ensure the matrix is positive-definite.
        nnls_solver_tol
            Convergence tolerance (infinity-norm KKT residual) of the JAX positive-only (NNLS) interior-point
            solve. `None` (default) uses jaxnnls's own tolerance ``min(n * eps * 5e3, 1e-2)`` (~1.7e-9 at
            n=1500 fp64) — behaviour is identical to not having this setting. Each solver iteration is a fresh
            dense Cholesky of the (n, n) system, so looser tolerances buy real speed: ``1e-6`` saves ~15-20% of
            solve time with a log-evidence shift of order 1e-8 on production HST pixelization+MGE fits
            (measured in https://github.com/PyAutoLabs/PyAutoArray/issues/369). Only the JAX (`xp=jnp`) path
            honors this; the NumPy fnnls path is unaffected.
        nnls_max_iter
            Iteration cap of the JAX positive-only (NNLS) interior-point solve. `None` (default) uses
            jaxnnls's own hard-coded cap of 50 (production HST pixelization+MGE systems converge in ~19-21
            iterations). Under `vmap` the solve runs until the slowest lane in the batch converges, so this
            also caps the worst-case batched cost. Only the JAX (`xp=jnp`) path honors this.
        log_det_method
            Which computation is used for the two Bayesian-evidence log-determinant terms
            (``log_det_curvature_reg_matrix_term`` and ``log_det_regularization_matrix_term``). `None`
            (default) reads the packaged value ``"cholesky"``.

            - ``"cholesky"`` (default) — ``2 * sum(log(diag(cholesky(M))))``, the historical computation.
              On a non-positive-definite matrix the NumPy backend raises and the JAX backend returns NaN.
            - ``"slogdet"`` — the ``logabsdet`` of ``xp.linalg.slogdet(M)``. Where ``M`` is positive-definite
              this equals the Cholesky value exactly; where the Cholesky would NaN it returns a finite,
              differentiable value instead, so it never stalls a gradient-based search
              (autolens_workspace_developer#104). This is an **opt-in, non-default** alternative intended for
              gradient-based work and for comparison against the Cholesky evidence — it is not a replacement
              for the default and the default evidence path is unchanged. See PyAutoArray#391.

              Under ``"slogdet"``, the kernel regularization schemes (``MaternKernel``, ``GaussianKernel``,
              ``ExponentialKernel``, ``MaternAdaptKernel``) additionally compute
              ``log_det_regularization_matrix_term`` analytically from a single Cholesky of their kernel
              covariance ``C`` (``log det H = pixels * log(coeff) - log det C``) instead of factorizing the
              explicitly formed inverse — more accurate (the formed ``C^-1`` carries round-off amplified by
              ``cond(C)``, ~1e-6 absolute in the evidence on clustered traced mesh vertices) and finite at any
              regularization coefficient (``C``'s conditioning does not depend on it). See
              :meth:`AbstractRegularization.log_det_regularization_matrix_term_from`.
        """
        self.use_mixed_precision = use_mixed_precision
        self.nnls_solver_tol = nnls_solver_tol
        self.nnls_max_iter = nnls_max_iter
        self._use_positive_only_solver = use_positive_only_solver
        self._use_edge_zeroed_pixels = use_edge_zeroed_pixels
        self._use_border_relocator = use_border_relocator
        self._no_regularization_add_to_curvature_diag_value = (
            no_regularization_add_to_curvature_diag_value
        )
        self._log_det_method = log_det_method

    @property
    def use_positive_only_solver(self):
        if self._use_positive_only_solver is None:
            return conf.instance["general"]["inversion"]["use_positive_only_solver"]

        return self._use_positive_only_solver

    @property
    def use_edge_zeroed_pixels(self):
        if self._use_edge_zeroed_pixels is None:
            return conf.instance["general"]["inversion"]["use_edge_zeroed_pixels"]

        return self._use_edge_zeroed_pixels

    @property
    def use_border_relocator(self):
        if self._use_border_relocator is None:
            return conf.instance["general"]["inversion"]["use_border_relocator"]

        return self._use_border_relocator

    @property
    def no_regularization_add_to_curvature_diag_value(self):
        if self._no_regularization_add_to_curvature_diag_value is None:
            return conf.instance["general"]["inversion"][
                "no_regularization_add_to_curvature_diag_value"
            ]

        return self._no_regularization_add_to_curvature_diag_value

    @property
    def log_det_method(self):
        if self._log_det_method is None:
            return conf.instance["general"]["inversion"]["log_det_method"]

        return self._log_det_method
