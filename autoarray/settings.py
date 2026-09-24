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
        nnls_warm_start_memo: Optional[bool] = None,
        nnls_warm_start_error_tolerance: Optional[float] = None,
        log_det_method: Optional[str] = None,
        regularization_term_method: Optional[str] = None,
        interferometer_numba_nnz_per_source_max: Optional[float] = None,
        positive_only_solver: Optional[str] = None,
        certified_pass_budget: Optional[int] = None,
        certified_fallback: Optional[str] = None,
        certified_tau_rel: Optional[float] = None,
        nnls_preconditioning_no_mapper: Optional[str] = None,
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
            - The mapping matrix allocation in
              :func:`autoarray.inversion.mappers.mapper_util.mapping_matrix_from` and the native cube in
              :meth:`Convolver.mapping_matrix_native_from` — output dtype becomes fp32 on the JAX backend
              only; under ``xp=np`` both stay fp64, so a NumPy fit is a true fp64 reference for a
              mixed-precision JAX fit (PyAutoArray#552).

            Empirical platform notes:

            - **GPU**: full pipeline single-JIT roughly matches the fp64 baseline; vmap-batched evaluation
              (the production sampler hot path) shows 25–30% speedup on RTX 2060-class hardware.
            - **CPU**: the per-call FFT itself is ~1.6× faster in fp32, but JAX/XLA's CPU FFT lowering does
              not always re-compose well across ~40-call MGE-basis pipelines, so the single-JIT measurement
              can be neutral or slightly slower than fp64. vmap remains comparable to or slightly faster than
              fp64. The flag is most beneficial for GPU users.

            Paths that intentionally stay in fp64:

            - The noise weighting of the curvature matrix in
              :func:`autoarray.inversion.inversion.inversion_util.curvature_matrix_via_mapping_matrix_from`:
              ``F`` and the data vector ``D`` are both formed with fp64 ``1 / noise_map`` on every backend,
              so the linear system is weighted consistently (an earlier fp32 ``1 / noise_map`` on the JAX
              branch never produced an fp32 accumulation, because the blurred mapping matrix is already
              fp64 — it only biased ``F`` against ``D``; PyAutoArray#552).
            - The NNLS reconstruction (jaxnnls / Cholesky factor + cho_solve) in
              :func:`autoarray.inversion.inversion.inversion_util.reconstruction_positive_only_from`. Active-set
              and PDIP solvers are sensitive to fp32 noise on ill-conditioned source meshes.
            - The log-determinant of the curvature regularization matrix used by ``figure_of_merit``: condition
              numbers can exceed 1e6 on fine pixelizations and fp32 silently loses 1+ digit there.
            - Light profile evaluation on the (over-)sampled grid; only the resulting mapping matrix is downcast.

            Empirical numerical impact on the MGE imaging regression (HST-shaped, 15k masked pixels, 40 linear
            Gaussians): Δlog-likelihood ≈ 1e-4 absolute at log-likelihood ≈ 27,400. Well below the natural χ²
            sampling noise floor (σ ≈ √(2N) ≈ 175). On a 316-pixel ``RectangularBilinearAdaptImage`` +
            ``Adapt`` inversion (17×17 mesh) the whole mixed-precision effect on the JAX path is
            Δlog-likelihood ≈ 1e-4 nats against the fp64 path (PyAutoArray#552); pixelization paths with
            K ≫ 40 source pixels are more sensitive — verify on representative integration tests before
            turning on for production fits.

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
            The packaged configuration defaults to `1.0e-3`; workspaces may override it. The addition is absolute,
            so its effect depends on the scale of the curvature matrix.
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
        nnls_warm_start_memo
            Whether the NumPy / numba positive-only (NNLS) solve warm-starts its active-set iteration from the
            passive set of the *previous* likelihood evaluation, held in a small process-local memo
            (:mod:`autoarray.inversion.inversion.nnls_memo`). Successive sampler evaluations sit close together in
            parameter space and share most of their passive set, so the seeded start removes active-set iterations
            from the solve, which is ~70% of a numba CPU likelihood evaluation on production Delaunay meshes. The
            NNLS optimum is unique, so the reconstruction is unchanged. `None` (default) reads the packaged value
            (`true`); setting ``AUTOARRAY_NNLS_WARM_START=0`` is the process-wide kill-switch that disables the
            memo whatever this setting says. Only the NumPy (`xp=np`) fnnls path honors this; the JAX path
            ignores it.
        nnls_warm_start_error_tolerance
            The relative guard on a memo seed's quality. Each memo entry carries a reference: the error
            fraction (warm-start errors / solve size) of the most recent solve for that key which started
            from the dense-sign guess. A memo-seeded solve whose own error fraction exceeds
            ``tolerance * reference`` is judged worse than simply starting dense, so its entry is discarded
            and the next solve for that key restarts from the dense-sign start, refreshing the reference.
            The guard is relative because the absolute error fraction does not separate helpful from
            unhelpful seeds (the two populations overlap at 0.048-0.138 in the PyAutoArray#498 32-cell
            robustness matrix) whereas the ratio to the dense-sign start does (helpful cells top out at
            0.89, the worst seed measured reaches 1.42). `None` (default) reads the packaged value
            (``1.5``), chosen above that worst observed ratio so the guard is protective against regimes
            far outside the matrix rather than flapping inside it. The guard is **disabled** by any value
            that is not finite and positive -- ``float("inf")`` is the idiomatic choice, and ``0`` or a
            negative value disables it too rather than meaning "drop everything". Only the NumPy
            (`xp=np`) fnnls path honors this; the JAX path ignores it.
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
        regularization_term_method
            How the Bayesian-evidence regularization term ``s^T H s``
            (``AbstractInversion.regularization_term``) is computed. `None` (default) reads the
            packaged value ``"matmul"``.

            - ``"matmul"`` (default) — ``s @ (H @ s)`` against the explicitly formed regularization
              matrix, the historical computation.
            - ``"cho_solve"`` — for the kernel regularization schemes (``MaternKernel``,
              ``GaussianKernel``, ``ExponentialKernel``, ``MaternAdaptKernel``), whose
              ``H = coefficient * C^-1``, evaluate ``coefficient * s^T C^-1 s`` by solving
              ``C x = s`` through one Cholesky of the kernel covariance ``C`` instead of forming
              ``C^-1`` and contracting it. More accurate — the explicit inverse's round-off is
              amplified by ``cond(C)``, which reaches ~1e9 on the clustered traced vertices of the
              kNN mesh families — and cheaper, being one triangular solve rather than ``N``. This
              is an **opt-in, non-default** alternative; schemes with no such factorization
              (``Constant``, ``Adapt``, the split families) have no shortcut and fall back to the
              formed matrix, so mixed inversions stay correct. See
              :meth:`AbstractRegularization.regularization_term_from`.

              This is deliberately separate from ``log_det_method``: the two terms can be moved
              onto their exact factorizations independently, which is what makes it possible to
              attribute an evidence shift to one term rather than both.

              Note neither option removes the explicit inverse from the inversion as a whole —
              ``curvature_reg_matrix`` is a dense ``F + H`` feeding the dense solve for the
              reconstruction, so ``H`` is still formed there regardless.
        interferometer_numba_nnz_per_source_max
            The geometry gate above which the numba `direct_conv` interferometer curvature
            path is not used, in mean non-zeros per source column
            (``mapper.pix_sizes_for_sub_slim_index.sum() / mapper.params``). `None`
            (default) reads the packaged value (`60.0`); `0` disables the numba path. See
            the property of the same name for the measured crossovers and why the constant
            is machine-dependent.
        positive_only_solver
            Which solver the JAX (`xp=jnp`) positive-only reconstruction uses. `None` (default) reads the packaged
            value ``"pdip"``.

            - ``"pdip"`` (default) — the jaxnnls primal-dual interior-point solve, unchanged.
            - ``"certified"`` — the certified active-set solve (:mod:`autoarray.util.jax_active_set`): a
              budgeted ``lax.while_loop`` of masked Cholesky solves that stops once the iterate satisfies the
              primal and dual (KKT) conditions, with the PDIP solve as a fallback when the budget is exhausted.
              Its gradient is the exact implicit active-set derivative. Measured 1.2-2.6x faster than PDIP on
              source-only inversions returning the same constrained optimum (PyAutoArray#566).

            ``"certified"`` is applied **only** on the JAX backend to **mapper-only** inversions (no linear
            light-profile / MGE coefficients, which converge poorly under the active-set scheme); every other
            inversion, and the whole NumPy path, keeps its existing solver
            (`AbstractInversion.positive_only_solver_used` records the decision). Opt-in until the batched
            (``vmap``) policy is measured.
        certified_pass_budget
            Maximum number of restricted active-set passes of the ``"certified"`` solver. `None` (default) reads
            the packaged value (`16`). Measured passes to certification: rectangular <= 11, Delaunay <= 7; the
            loop exits at certification, so unused budget costs nothing.
        certified_fallback
            What the ``"certified"`` solver returns when it exhausts its budget uncertified. `None` (default)
            reads the packaged value ``"pdip"`` (run the PDIP solve instead, via ``lax.cond``; under ``vmap`` that
            ``cond`` executes both solvers for every lane). ``"none"`` returns the last, uncertified iterate.
        certified_tau_rel
            Relative KKT tolerance of the ``"certified"`` solver's certificate: primal violations are
            ``x < -tau_rel * max|x|``, dual violations ``g < -tau_rel * max|q|``. `None` (default) reads the
            packaged value (`1.0e-9`).
        nnls_preconditioning_no_mapper
            How the JAX positive-only PDIP solve scales an inversion **with no `Mapper`** (linear light profiles /
            MGE only). `None` (default) reads the packaged value ``"raw"``.

            - ``"raw"`` (default) -- the forward PDIP solve runs on the un-preconditioned system with a
              data-scaled KKT tolerance (``1e-2 * n * eps_pdip * max(1, max|data_vector|)``, or
              `nnls_solver_tol` if set); the gradient is the same Jacobi-space relaxed-KKT pass as ``"jacobi"``.
              On the SLaM `source_lp[1]` MGE model (2 x 20 lens + 20 source Gaussians) Jacobi scaling made 14/48
              near-truth points hit the 50-iteration cap with wrong log-likelihoods (signal-free Gaussian columns,
              whose diagonal is only `no_regularization_add_to_curvature_diag_value`, become degenerate
              coordinates on which the PDIP dual diverges); the raw solve converges on all of them in 16-19
              iterations (PyAutoArray#571).
            - ``"jacobi"`` -- the Jacobi-preconditioned solve, as for mapper inversions.

            Inversions containing a `Mapper` always use ``"jacobi"``
            (`AbstractInversion.positive_only_preconditioning_used` records the decision). The NumPy path always
            runs fnnls and ignores this.
        """
        self.use_mixed_precision = use_mixed_precision
        self.nnls_solver_tol = nnls_solver_tol
        self.nnls_max_iter = nnls_max_iter
        self._nnls_warm_start_memo = nnls_warm_start_memo
        self._nnls_warm_start_error_tolerance = nnls_warm_start_error_tolerance
        self._use_positive_only_solver = use_positive_only_solver
        self._use_edge_zeroed_pixels = use_edge_zeroed_pixels
        self._use_border_relocator = use_border_relocator
        self._no_regularization_add_to_curvature_diag_value = (
            no_regularization_add_to_curvature_diag_value
        )
        self._log_det_method = log_det_method
        self._regularization_term_method = regularization_term_method
        self._interferometer_numba_nnz_per_source_max = (
            interferometer_numba_nnz_per_source_max
        )
        self._positive_only_solver = positive_only_solver
        self._certified_pass_budget = certified_pass_budget
        self._certified_fallback = certified_fallback
        self._certified_tau_rel = certified_tau_rel
        self._nnls_preconditioning_no_mapper = nnls_preconditioning_no_mapper

        # Validate explicit values eagerly, so a typo fails at construction rather than deep inside a fit.
        if positive_only_solver is not None:
            self.positive_only_solver
        if certified_fallback is not None:
            self.certified_fallback
        if nnls_preconditioning_no_mapper is not None:
            self.nnls_preconditioning_no_mapper

    @property
    def use_positive_only_solver(self):
        if self._use_positive_only_solver is None:
            return conf.instance["general"]["inversion"]["use_positive_only_solver"]

        return self._use_positive_only_solver

    @property
    def use_edge_zeroed_pixels(self):
        """
        Whether a mesh's edge pixels are excluded from the inversion and fixed to zero.

        This is consulted **only when `use_positive_only_solver` is `True`**. Under the
        positive-negative solver the full system is solved and this setting has no effect, so the two
        are not independent switches despite reading that way in `config/general.yaml`.

        That scoping is deliberate. It is called out here because the nesting is not visible from the
        config, and has been mistaken for a bug (a setting "silently ignored") by someone reading the
        control flow in `AbstractInversion.reconstruction` without it.
        """
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
    def nnls_warm_start_memo(self) -> bool:
        """
        Whether the NumPy fnnls solve is warm-started from the previous evaluation's passive set.
        """
        if self._nnls_warm_start_memo is None:
            try:
                return conf.instance["general"]["inversion"]["nnls_warm_start_memo"]
            except KeyError:
                # A workspace `general.yaml` normally omits this key, so autoconf's
                # config-path list falls through to autoarray's packaged value (`true`).
                # This fallback fires only when the workspace config is the sole config
                # path (isolated test configs push one dir) and returns that same value,
                # so both routes resolve identically.
                return True

        return self._nnls_warm_start_memo

    @property
    def nnls_warm_start_error_tolerance(self) -> float:
        """
        How much worse than the dense-sign start a memo seed may be before its entry is dropped.

        A seeded solve whose error fraction exceeds this multiple of the entry's dense-sign reference
        fraction is discarded, so the next solve for that key restarts dense. Any value that is not
        finite and positive disables the guard.
        """
        if self._nnls_warm_start_error_tolerance is None:
            try:
                return conf.instance["general"]["inversion"][
                    "nnls_warm_start_error_tolerance"
                ]
            except KeyError:
                # A workspace `general.yaml` normally omits this key, so autoconf's
                # config-path list falls through to autoarray's packaged value (`1.5`).
                # This fallback fires only when the workspace config is the sole config
                # path (isolated test configs push one dir) and returns that same value,
                # so both routes resolve identically.
                return 1.5

        return self._nnls_warm_start_error_tolerance

    @property
    def log_det_method(self):
        if self._log_det_method is None:
            return conf.instance["general"]["inversion"]["log_det_method"]

        return self._log_det_method

    @property
    def regularization_term_method(self):
        if self._regularization_term_method is None:
            return conf.instance["general"]["inversion"]["regularization_term_method"]

        return self._regularization_term_method

    @property
    def interferometer_numba_nnz_per_source_max(self) -> float:
        """
        The geometry gate above which the numba `direct_conv` interferometer curvature
        path is not used.

        `InversionInterferometerSparseNumba` convolves each source column of the mapping
        operator over the extent rectangle, at a cost that scales with the column's
        non-zeros; the FFT route (`InversionInterferometerSparse`) costs the same whatever
        the density. The two therefore cross at a roughly fixed number of non-zeros per
        source column, `mapper.pix_sizes_for_sub_slim_index.sum() / mapper.params`, and
        the factory routes to numba only at or below this value.

        The measured crossovers are **~60 non-zeros per source column on Delaunay meshes**
        and **~77 on rectangular meshes** (autolens_profiling issue #226 verdict, section
        2), where the numba kernel is 2-7x faster than JAX-CPU well below the crossover.
        The default is the conservative of the two.

        This is a **machine-dependent constant**: it is set by the ratio of scalar AXPY
        throughput to FFT throughput on the CPU running the fit, so a machine with a very
        different cache hierarchy or FFT library will cross somewhere else. Re-measure
        before tuning it for a new machine; `0` disables the numba path entirely.
        """
        if self._interferometer_numba_nnz_per_source_max is None:
            try:
                return conf.instance["general"]["inversion"][
                    "interferometer_numba_nnz_per_source_max"
                ]
            except KeyError:
                # A workspace `general.yaml` normally omits this key, so autoconf's
                # config-path list falls through to autoarray's packaged value (`60.0`).
                # This fallback fires only when the workspace config is the sole config
                # path (isolated test configs push one dir) and returns that same value,
                # so both routes resolve identically.
                return 60.0

        return self._interferometer_numba_nnz_per_source_max

    def _inversion_config_value(self, key, default):
        # A workspace `general.yaml` normally omits the newer keys, so autoconf's config-path
        # list falls through to autoarray's packaged value. The fallback fires only when the
        # workspace config is the sole config path (isolated test configs push one dir) and
        # returns that same packaged value, so both routes resolve identically.
        try:
            return conf.instance["general"]["inversion"][key]
        except KeyError:
            return default

    @property
    def positive_only_solver(self) -> str:
        """
        Which solver the JAX positive-only reconstruction uses: ``"pdip"`` or ``"certified"``.

        See the constructor docstring; ``"certified"`` is only applied to mapper-only JAX inversions.
        """
        value = self._positive_only_solver
        if value is None:
            value = self._inversion_config_value("positive_only_solver", "pdip")

        if value not in ("pdip", "certified"):
            raise ValueError(
                f"positive_only_solver={value!r} is invalid; expected 'pdip' or 'certified'."
            )

        return value

    @property
    def certified_pass_budget(self) -> int:
        """
        Maximum number of restricted active-set passes of the ``"certified"`` solver.
        """
        if self._certified_pass_budget is None:
            return self._inversion_config_value("certified_pass_budget", 16)

        return self._certified_pass_budget

    @property
    def certified_fallback(self) -> str:
        """
        What an exhausted ``"certified"`` solve returns: ``"pdip"`` (the PDIP solve) or ``"none"``.
        """
        value = self._certified_fallback
        if value is None:
            value = self._inversion_config_value("certified_fallback", "pdip")

        if value not in ("pdip", "none"):
            raise ValueError(
                f"certified_fallback={value!r} is invalid; expected 'pdip' or 'none'."
            )

        return value

    @property
    def certified_tau_rel(self) -> float:
        """
        Relative KKT tolerance of the ``"certified"`` solver's certificate.
        """
        if self._certified_tau_rel is None:
            return self._inversion_config_value("certified_tau_rel", 1.0e-9)

        return self._certified_tau_rel

    @property
    def nnls_preconditioning_no_mapper(self) -> str:
        """
        How the JAX PDIP solve scales an inversion with no `Mapper`: ``"raw"`` or ``"jacobi"``.

        See the constructor docstring; inversions with a `Mapper` always use ``"jacobi"``.
        """
        value = self._nnls_preconditioning_no_mapper
        if value is None:
            value = self._inversion_config_value(
                "nnls_preconditioning_no_mapper", "raw"
            )

        if value not in ("raw", "jacobi"):
            raise ValueError(
                f"nnls_preconditioning_no_mapper={value!r} is invalid; expected 'raw' or 'jacobi'."
            )

        return value
