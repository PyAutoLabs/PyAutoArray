"""
Matrix-free utilities for interferometer pixelized-source inversions (NumPy / scipy only).

Scope
-----
This module is phase 1 of the matrix-free interferometer inversion epic (PyAutoArray #638,
community proposal by @HRSAstro, https://github.com/PyAutoLabs/.github/discussions/39).

The sparse (w-tilde) interferometer inversion, `InversionInterferometerSparse`, forms two dense
`(S, S)` matrices for a mesh of `S` source pixels: the curvature matrix `F = Aᵀ W~ A` and the
regularization matrix `H`. Their memory and the `O(S^3)` dense factorisation of `F + H` cap an
interferometer pixelized source at roughly `100^2` mesh pixels. Neither matrix is needed to *solve*
the linear system, only their action on a vector:

- the mapping matrix `A` (image pixels x source pixels) has a handful of non-zeros per row, so it is
  held as a `scipy.sparse` matrix `M`;
- `W~ = Re(Fᴴ N⁻¹ F)` is translation invariant on the real-space grid, so `W~ x` is a convolution
  with the `nufft_precision_operator` kernel, applied by one real FFT pair on the 2x zero-padded grid;
- the regularization matrices autoarray builds (`Constant`, `AdaptiveBrightness`, ...) are banded,
  so they are held as `scipy.sparse` matrices too.

The curvature matrix action is then `F s = Mᵀ W~ (M s)`, the data vector is `D = Mᵀ d~`, and the
reconstruction solves

    (F + λ H) s = D

by preconditioned conjugate gradients (PCG), with the Powell et al. (2021) preconditioner
`P = κ₀ MᵀM + λ H`, where `κ₀` is the kernel's zero-lag value (the diagonal of `W~`). `P` replaces
`W~` by its peak, which keeps it sparse; it is factorised once per `λ` by a sparse LU.

Phase 1 landed the operator data term and the solver as library utilities, phase 2 the natively sparse
Matérn precision (`aa.reg.MaternSparse`), phase 3 the evidence log-determinants (below) and phase 4 the
positivity constraint (below), each with parity tests against the dense sparse-path quantities. None of
it is wired into fits and there is no `Settings` flag.

Positivity (phase 4)
--------------------
`reconstruction_positive_via_interior_point_from` solves `(F + λH) s = D` subject to `s >= 0` by a
Mehrotra predictor-corrector primal-dual interior point method whose Newton systems are solved by CG
preconditioned by the Powell matrix plus the barrier diagonal, followed by an exact solve on the
detected support with primal-dual active-set corrections (Hintermüller, Ito & Kunisch 2002), so the
held pixels are exact zeros. On the test case (a `10 x 10` `MaternSparse` mesh on the data-dominated
dataset, 33-72% of pixels at zero) it reproduces the support of `scipy.optimize.nnls` on the whitened
dense system exactly and its values to ~1e-10 of the max in 8-10 interior-point steps and one polish
solve.

Remaining work (phase 5)
------------------------
- wiring as an inversion class or a `Settings` option, so fits can select the matrix-free path;
- a profiling note, dense vs matrix-free at `50^2`, `64^2`, `128^2` and `192^2` mesh pixels, ingested
  by PyAutoPulse;
- a workspace example;
- reconstruction uncertainty maps, the diagonal of `(F + λH)⁻¹`, by a stochastic diagonal estimator
  (there is no dense inverse to read it from);
- linear light profiles and multiple mappers (every function here assumes a single mapper's `M`).

The log evidence (phase 3)
--------------------------
The Bayesian evidence also needs `ln det(F + λ H)` and `ln det(λ H)`. `ln det H` is exact from a
sparse factorisation (for `MaternSparse`, `log_det_regularization_matrix_term_from`). For the curvature
term, `ln det(F + λ H) = ln det P + ln det(P⁻¹(F + λ H))`: `ln det P` is exact from the Powell
preconditioner's sparse LU (`log_det_preconditioner_from`) and the second term is estimated by
stochastic Lanczos quadrature in the `P` inner product (`log_det_curvature_reg_via_slq_from`), with
probes `v ~ N(0, P⁻¹)` drawn from a sparse factor `R` of `H = RᵀR` and a fixed seed so the estimate is
deterministic and smooth in the model parameters. `log_evidence_matrix_free_from` assembles the five
evidence terms with the conventions of `FitInterferometer.log_evidence`.

Tolerance: the estimate is only as good as the preconditioner. On the test case (a `6 x 6`
`MaternSparse` rectangular mesh, 500 visibilities on long baselines, so the dirty beam is compact and
`P⁻¹(F + H)` has eigenvalues in `~[0.6, 1.6]`) it lies within 0.5% of the dense `slogdet(F + H)` at
the defaults (10 probes, 40 Lanczos steps; measured ~0.1-0.15% for the default seed), and the error
falls as `1 / sqrt(n_probes)`. When `W~` is far from `κ₀ I` (few visibilities, fringes broader than
the mask) the eigenvalues spread over orders of magnitude and the per-probe variance grows by an
order of magnitude or more; the per-probe values returned give the Monte-Carlo error of the mean.

The regularization matrix
-------------------------
Every function here takes the regularization matrix as a `scipy.sparse` matrix. In this phase it is
any existing dense `regularization_matrix` converted with `scipy.sparse.csr_matrix(...)`. Note that
autoarray's regularization matrices already carry their coefficient (e.g. `Constant` builds
`coefficient^2 B + 1e-8 I`), so the `coefficient` argument here, the `λ` of `F + λ H`, is `1.0` to
reproduce an autoarray inversion's `curvature_reg_matrix` exactly. A natively sparse Matérn
precision (SPDE construction, Lindgren, Rue & Lindström 2011) replacing the dense `MaternKernel`
matrix is phase 2.

JAX (deferred to phase 2)
-------------------------
Everything here is NumPy / scipy and imports no JAX, so it runs in the `unittest-nojax` leg. Porting
it to JAX is an open strategy question for phase 2, because the scipy solve does not map onto `jit`
directly:

- `scipy.sparse.linalg.cg` stops on a tolerance, a data-dependent loop length; under `jit` the
  iteration count must be fixed (or run in a `lax.while_loop`, which is not reverse-differentiable),
  so a fixed iteration budget and its accuracy need choosing;
- `scipy.sparse` matrices have data-dependent non-zero counts; the JAX form needs static shapes
  (padded COO triplets with a fixed `nnz`, as `mapper_util.sparse_triplets_from` already produces);
- `scipy.sparse.linalg.splu` has no `jit` counterpart, so the Powell preconditioner needs another
  factorisation route (e.g. a host callback, a banded Cholesky, or a cheaper preconditioner).
"""

import numpy as np
import scipy.fft
import scipy.sparse
import scipy.sparse.linalg


def sparse_mapping_matrix_from(inversion, mapper) -> scipy.sparse.csr_matrix:
    """
    Returns a mapper's real-space mapping matrix as a `scipy.sparse` CSR matrix of shape
    `(image_pixels, source_pixels)`, with rows indexed on the slim (masked-pixel) grid.

    It is assembled from the sparse COO triplets `InversionInterferometerSparse` already builds for
    its data vector (`_sparse_triplets_slim_from`), so the dense `mapping_matrix` is never formed.
    Each sub-pixel contributes `pix_weight * sub_fraction` to every source pixel of its interpolation
    stencil; CSR construction *sums* duplicate `(row, column)` entries, which is what accumulates the
    sub-pixels of one image pixel (and, for a Delaunay mesh, repeated vertices) into a single entry.
    Stencil slots the triplets carry with zero weight are removed, so only true non-zeros are stored.

    If the dataset has a primary beam `P`, the triplets (and therefore this matrix) are those of the
    beamed operator `A' = P A`, which is the response the operator form wants: the curvature matrix
    is then `A'ᵀ W~ A'` and the data vector `A'ᵀ d~`. Without a primary beam the dense form of the
    returned matrix equals `mapper.mapping_matrix`.

    Parameters
    ----------
    inversion
        An `InversionInterferometerSparse` whose dataset carries the `sparse_operator` (and any
        primary beam) the mapping is built for.
    mapper
        The mapper whose mapping matrix is returned.

    Returns
    -------
    The `(image_pixels, source_pixels)` sparse mapping matrix `M`.
    """
    rows, cols, vals = inversion._sparse_triplets_slim_from(mapper=mapper)

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)

    shape = (int(inversion.mask.pixels_in_mask), int(mapper.params))

    mapping_matrix = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=shape)
    mapping_matrix.eliminate_zeros()

    return mapping_matrix


class MatrixFreeDataOperator:
    def __init__(
        self,
        nufft_precision_operator: np.ndarray,
        dirty_image: np.ndarray,
        mapping_matrix: scipy.sparse.spmatrix,
        mask,
    ):
        """
        The data term of an interferometer inversion, `F = Mᵀ W~ M` and `D = Mᵀ d~`, held as an
        operator rather than as a dense `(S, S)` curvature matrix.

        `W~ = Re(Fᴴ N⁻¹ F)` (with `F` the non-uniform Fourier transform and `N` the visibility noise
        covariance) depends only on the offset between two real-space pixels, so it is a convolution
        with the `nufft_precision_operator` kernel. The kernel lives on the `(2y, 2x)` grid of pixel
        offsets, in FFT wraparound order, where `(y, x)` is the unmasked extent of the real-space mask
        (`Mask2D.shape_native_masked_pixels`, the bounding box of the unmasked pixels). An image `x` on
        the `(y, x)` extent grid is therefore operated on as

            W~ x = IRFFT2( RFFT2(x_pad) * RFFT2(kernel) )[:y, :x]

        with `x_pad` the `(2y, 2x)` zero-padded image. This is exactly the transform
        `InterferometerSparseOperator.apply_operator` uses to assemble the dense curvature matrix;
        padding to twice the extent means the circular convolution never wraps an image pixel onto
        another, so the product is the linear convolution.

        Applying `F` to a source vector `s` costs two sparse products and one FFT pair:

            F s = Mᵀ W~ (M s)

        where `M s` (on the slim grid) is scattered onto the extent grid and `W~ (M s)` is gathered
        back to the slim grid. The data vector is `D = Mᵀ d~`, with `d~ = Re(Fᴴ N⁻¹ d)` the dirty
        image.

        Parameters
        ----------
        nufft_precision_operator
            The real `(2y, 2x)` w-tilde kernel, as stored on the dataset's `sparse_operator`
            (`SparseTerms.nufft_precision_operator`).
        dirty_image
            The slim (masked-pixel indexed) dirty image `d~`.
        mapping_matrix
            The `(image_pixels, source_pixels)` sparse mapping matrix `M` (see
            `sparse_mapping_matrix_from`), including any primary beam.
        mask
            The real-space `Mask2D` the slim grid is defined on.
        """
        nufft_precision_operator = np.asarray(
            nufft_precision_operator, dtype=np.float64
        )

        H2, W2 = nufft_precision_operator.shape

        if (H2 % 2) != 0 or (W2 % 2) != 0:
            raise ValueError(
                "nufft_precision_operator must have even shape (2y, 2x). "
                f"Got {nufft_precision_operator.shape}."
            )

        self.y_shape = H2 // 2
        self.x_shape = W2 // 2

        if tuple(mask.shape_native_masked_pixels) != (self.y_shape, self.x_shape):
            raise ValueError(
                f"The nufft_precision_operator shape {nufft_precision_operator.shape} is not twice "
                f"the unmasked extent {tuple(mask.shape_native_masked_pixels)} of the mask."
            )

        self.nufft_precision_operator = nufft_precision_operator
        self.kernel_hat = scipy.fft.rfft2(nufft_precision_operator)

        self.mask = mask
        self.extent_index_for_masked_pixel = np.asarray(
            mask.extent_index_for_masked_pixel, dtype=np.int64
        )

        self.dirty_image = np.asarray(dirty_image, dtype=np.float64)

        self.mapping_matrix = mapping_matrix.tocsr()
        self.mapping_matrix_transpose = mapping_matrix.T.tocsr()

    @classmethod
    def from_inversion(cls, inversion, mapper) -> "MatrixFreeDataOperator":
        """
        Returns the operator for one mapper of an `InversionInterferometerSparse`, reading the kernel
        from the dataset's `sparse_operator` and the dirty image exactly as the inversion's
        `data_vector` does (the dataset's `sparse_dirty_image` override if present, else the
        operator's own dirty image).

        Parameters
        ----------
        inversion
            The sparse-path inversion the operator reproduces.
        mapper
            The mapper whose data term is represented.
        """
        sparse_operator = inversion.dataset.sparse_operator

        dirty_image = getattr(inversion.dataset, "sparse_dirty_image", None)

        if dirty_image is None:
            dirty_image = sparse_operator.dirty_image

        dirty_image = getattr(dirty_image, "array", dirty_image)

        return cls(
            nufft_precision_operator=np.asarray(
                sparse_operator.nufft_precision_operator
            ),
            dirty_image=np.asarray(dirty_image),
            mapping_matrix=sparse_mapping_matrix_from(
                inversion=inversion, mapper=mapper
            ),
            mask=inversion.mask,
        )

    @property
    def params(self) -> int:
        """
        The number of source pixels `S`, the dimension of the linear system.
        """
        return int(self.mapping_matrix.shape[1])

    @property
    def kappa0(self) -> float:
        """
        The kernel's zero-lag value `κ₀`, which is every diagonal entry of `W~` (the sum of the
        inverse noise variances of the visibilities). The Powell preconditioner replaces `W~` by it.
        """
        return float(self.nufft_precision_operator[0, 0])

    def wtilde_image_from(self, image_extent: np.ndarray) -> np.ndarray:
        """
        Returns `W~ x` for an image `x` on the `(y, x)` unmasked-extent grid, by one real FFT pair
        on the `(2y, 2x)` zero-padded grid (see the class docstring).

        Both the image and the kernel are real, so the `rfft2` / `irfft2` pair is exact rather than
        an approximation of the complex transform.

        Parameters
        ----------
        image_extent
            The real `(y, x)` image on the unmasked-extent grid of the mask.

        Returns
        -------
        The `(y, x)` image `W~ x`.
        """
        y_shape, x_shape = self.y_shape, self.x_shape

        image_pad = np.zeros((2 * y_shape, 2 * x_shape), dtype=np.float64)
        image_pad[:y_shape, :x_shape] = image_extent

        image_hat = scipy.fft.rfft2(image_pad)
        image_hat *= self.kernel_hat

        return scipy.fft.irfft2(image_hat, s=(2 * y_shape, 2 * x_shape))[
            :y_shape, :x_shape
        ]

    def curvature_matvec(self, s: np.ndarray) -> np.ndarray:
        """
        Returns the curvature matrix applied to a source vector, `F s = Mᵀ W~ (M s)`, without forming
        `F`.

        `M s` is the image of `s` on the slim grid; it is scattered onto the extent grid (masked
        pixels zero), operated on by `W~`, gathered back to the slim grid and projected by `Mᵀ`.

        Parameters
        ----------
        s
            A vector of `S` source-pixel values.
        """
        image_extent = np.zeros(self.y_shape * self.x_shape, dtype=np.float64)
        image_extent[self.extent_index_for_masked_pixel] = self.mapping_matrix @ s

        wtilde_image = self.wtilde_image_from(
            image_extent.reshape(self.y_shape, self.x_shape)
        ).ravel()

        return (
            self.mapping_matrix_transpose
            @ wtilde_image[self.extent_index_for_masked_pixel]
        )

    @property
    def data_vector(self) -> np.ndarray:
        """
        The data vector `D = Mᵀ d~`, the right-hand side of the linear system.
        """
        return self.mapping_matrix_transpose @ self.dirty_image


def powell_preconditioner_from(
    operator: MatrixFreeDataOperator,
    regularization_matrix_sparse: scipy.sparse.spmatrix,
    coefficient: float,
):
    """
    Returns the Powell et al. (2021) preconditioner `P = κ₀ MᵀM + λ H` for the system
    `(F + λ H) s = D`, as a sparse matrix and its sparse LU factorisation.

    `F = Mᵀ W~ M` is dense because `W~` couples every pair of image pixels. Replacing `W~` by its
    diagonal, the zero-lag value `κ₀` times the identity, gives `κ₀ MᵀM`, which has the sparsity of the
    mesh adjacency and captures the dominant (short-baseline) part of `F`. `P` is symmetric positive
    definite whenever `H` is, so it is a valid CG preconditioner; it is factorised once per `λ`.

    The LU uses the symmetric-mode options (`MMD_AT_PLUS_A` ordering, no pivoting threshold), which
    preserve the symmetric fill-reducing ordering for an SPD matrix.

    Parameters
    ----------
    operator
        The data operator providing `M` and `κ₀`.
    regularization_matrix_sparse
        The regularization matrix `H` as a `scipy.sparse` matrix (any autoarray `regularization_matrix`
        converted with `scipy.sparse.csr_matrix`).
    coefficient
        The scalar `λ` multiplying `H`; `1.0` for an autoarray regularization matrix, which already
        carries its coefficient.

    Returns
    -------
    The preconditioner `P` as a CSC matrix, and its `scipy.sparse.linalg.SuperLU` factorisation.
    """
    preconditioner = (
        operator.kappa0 * (operator.mapping_matrix_transpose @ operator.mapping_matrix)
        + coefficient * scipy.sparse.csr_matrix(regularization_matrix_sparse)
    ).tocsc()

    factor = scipy.sparse.linalg.splu(
        preconditioner,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options=dict(SymmetricMode=True),
    )

    return preconditioner, factor


def reconstruction_via_pcg_from(
    operator: MatrixFreeDataOperator,
    regularization_matrix_sparse: scipy.sparse.spmatrix,
    coefficient: float,
    x0=None,
    rtol: float = 1.0e-8,
    maxiter: int = 5000,
):
    """
    Returns the unconstrained reconstruction solving `(F + λ H) s = D` by preconditioned conjugate
    gradients, with `F s` applied matrix-free by `operator.curvature_matvec` and the Powell
    preconditioner (`powell_preconditioner_from`) applied through its sparse LU solve.

    This is the matrix-free counterpart of the positive-negative (`use_positive_only_solver=False`)
    solve of an inversion's `curvature_reg_matrix`. No positivity is imposed (see
    `reconstruction_positive_via_interior_point_from` for the positive solve).

    Parameters
    ----------
    operator
        The data operator providing `F` and `D`.
    regularization_matrix_sparse
        The regularization matrix `H` as a `scipy.sparse` matrix.
    coefficient
        The scalar `λ` multiplying `H`; `1.0` for an autoarray regularization matrix, which already
        carries its coefficient.
    x0
        An optional starting vector (e.g. the previous reconstruction); zeros if `None`.
    rtol
        The relative residual tolerance `|r| <= rtol |D|` at which CG stops.
    maxiter
        The maximum number of CG iterations.

    Returns
    -------
    The reconstruction `s` and the `scipy.sparse.linalg.cg` info flag (`0` on convergence, the
    iteration count if `maxiter` was reached without converging).
    """
    regularization_matrix_sparse = scipy.sparse.csr_matrix(regularization_matrix_sparse)

    n = operator.params

    curvature_reg_operator = scipy.sparse.linalg.LinearOperator(
        (n, n),
        matvec=lambda v: operator.curvature_matvec(v)
        + coefficient * (regularization_matrix_sparse @ v),
        dtype=np.float64,
    )

    _, factor = powell_preconditioner_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix_sparse,
        coefficient=coefficient,
    )

    preconditioner_operator = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=factor.solve, dtype=np.float64
    )

    return scipy.sparse.linalg.cg(
        curvature_reg_operator,
        operator.data_vector,
        x0=x0,
        rtol=rtol,
        maxiter=maxiter,
        M=preconditioner_operator,
    )


def log_det_preconditioner_from(factor) -> float:
    """
    Returns the exact `ln det P` of the Powell preconditioner from its sparse LU factorisation (the
    second return value of `powell_preconditioner_from`).

    SuperLU factorises `Pr P Pc = L U` with row and column permutations `Pr`, `Pc` and a unit-diagonal
    `L`, so `|det P| = prod_i |U_ii|` (a permutation has determinant `+-1`). `P` is symmetric positive
    definite, so `det P > 0` and

        ln det P = sum_i ln |U_ii|.

    Parameters
    ----------
    factor
        The `scipy.sparse.linalg.SuperLU` factorisation of `P`.

    Returns
    -------
    The log determinant `ln det P`.
    """
    return float(np.sum(np.log(np.abs(factor.U.diagonal()))))


def log_det_curvature_reg_via_slq_from(
    operator: MatrixFreeDataOperator,
    regularization_matrix_sparse: scipy.sparse.spmatrix,
    regularization_factor_sparse: scipy.sparse.spmatrix,
    coefficient: float,
    n_probes: int = 10,
    n_steps: int = 40,
    seed: int = 20221,
):
    """
    Returns a stochastic Lanczos quadrature (SLQ, Ubaru, Chen & Saad 2017) estimate of
    `ln det(F + λ H)`, the curvature-regularization log determinant of the Bayesian evidence, without
    forming the dense `(S, S)` matrix `A = F + λ H`.

    With the Powell preconditioner `P = κ₀ MᵀM + λ H` (`powell_preconditioner_from`),

        ln det A = ln det P + ln det(P⁻¹ A),

    where `ln det P` is exact from its sparse LU (`log_det_preconditioner_from`) and only the second
    term, whose eigenvalues cluster near one because `P` captures the dominant part of `A`, is
    estimated stochastically:

    - **Probes** `v ~ N(0, P⁻¹)`. Writing `H = RᵀR` (`R` the `regularization_factor_sparse`),
      `P = BᵀB` with `B = [√κ₀ M; √λ R]`, so `v = P⁻¹ Bᵀ g = P⁻¹(√κ₀ Mᵀ g₁ + √λ Rᵀ g₂)` with `g`
      standard normal has covariance `P⁻¹ BᵀB P⁻¹ = P⁻¹`. Then `E[vᵀ P f(P⁻¹A) v] = tr f(P⁻¹A)`
      for `f = ln`, i.e. `E = ln det(P⁻¹A)`.
    - **Lanczos** on `P⁻¹A`, which is self-adjoint in the `P` inner product `<x, y>_P = xᵀ P y`.
      Each probe starts from `q₁ = v / ‖v‖_P`; the recurrence keeps every basis vector
      `P`-orthonormal with full reorthogonalisation (the Krylov space is small, `n_steps` vectors) and
      stops early once the off-diagonal falls below `1e-10` of the diagonal (an invariant subspace).
    - **Gauss quadrature** on the tridiagonal `T` with eigenpairs `(θ_k, u_k)`: the probe's value is
      `‖v‖²_P Σ_k u_k[0]² ln θ_k`.

    The random generator is `np.random.default_rng(seed)` with a fixed default seed, so the estimate
    is deterministic and, because the same probes are reused, a smooth function of `λ` (and of any
    other parameter) -- which a non-linear search over the evidence needs.

    Accuracy: on the small rectangular-mesh test cases the estimate lies within 0.5% of the dense
    `slogdet(F + H)` at the defaults (10 probes, 40 steps); the per-probe spread returned gives the
    Monte-Carlo error of the mean (`std / sqrt(n_probes)`).

    Parameters
    ----------
    operator
        The data operator providing `F`, `M` and `κ₀`.
    regularization_matrix_sparse
        The regularization matrix `H` as a `scipy.sparse` matrix.
    regularization_factor_sparse
        A sparse factor `R` of exactly that matrix, `H = RᵀR`. For `MaternSparse`, whose
        `regularization_matrix_sparse_from` returns `coefficient * R₀ᵀR₀` with
        `R₀ = regularization_factor_sparse_from(...)`, this is `sqrt(regularization.coefficient) * R₀`.
        It is checked against `H` on one random vector.
    coefficient
        The scalar `λ` multiplying `H`, as in `powell_preconditioner_from`; `1.0` for an autoarray
        regularization matrix, which already carries its coefficient.
    n_probes
        The number of Gaussian probe vectors.
    n_steps
        The maximum number of Lanczos steps per probe (capped at `S`).
    seed
        The seed of the probe generator.

    Returns
    -------
    The estimate of `ln det(F + λ H)` and the `(n_probes,)` array of per-probe estimates (each
    `ln det P` plus that probe's quadrature value), whose mean is the estimate.
    """
    regularization_matrix_sparse = scipy.sparse.csr_matrix(regularization_matrix_sparse)
    regularization_factor_sparse = scipy.sparse.csr_matrix(regularization_factor_sparse)

    n = operator.params

    rng = np.random.default_rng(seed)

    check = rng.standard_normal(n)
    h_check = regularization_matrix_sparse @ check
    rtr_check = regularization_factor_sparse.T @ (regularization_factor_sparse @ check)

    if np.linalg.norm(rtr_check - h_check) > 1.0e-8 * np.linalg.norm(h_check):
        raise ValueError(
            "regularization_factor_sparse R does not factor regularization_matrix_sparse H "
            "(H != RᵀR); include the regularization coefficient in R (e.g. "
            "sqrt(regularization.coefficient) * regularization.regularization_factor_sparse_from(mapper))."
        )

    preconditioner, factor = powell_preconditioner_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix_sparse,
        coefficient=coefficient,
    )

    log_det_preconditioner = log_det_preconditioner_from(factor)

    def curvature_reg_matvec(v):
        return operator.curvature_matvec(v) + coefficient * (
            regularization_matrix_sparse @ v
        )

    n_steps = min(int(n_steps), n)

    sqrt_kappa0 = np.sqrt(operator.kappa0)
    sqrt_coefficient = np.sqrt(coefficient)

    probe_values = np.zeros(n_probes)

    for probe in range(n_probes):
        g_image = rng.standard_normal(operator.mapping_matrix.shape[0])
        g_regularization = rng.standard_normal(regularization_factor_sparse.shape[0])

        v = factor.solve(
            sqrt_kappa0 * (operator.mapping_matrix_transpose @ g_image)
            + sqrt_coefficient * (regularization_factor_sparse.T @ g_regularization)
        )

        pv = preconditioner @ v
        norm2 = float(v @ pv)
        norm = np.sqrt(norm2)

        q = v / norm
        pq = pv / norm
        q_prev = np.zeros(n)
        beta_prev = 0.0

        basis = []
        basis_p = []
        alpha = []
        beta = []

        for step in range(n_steps):
            basis.append(q)
            basis_p.append(pq)

            aq = curvature_reg_matvec(q)
            a = float(q @ aq)

            r = factor.solve(aq) - a * q - beta_prev * q_prev

            for q_j, pq_j in zip(basis, basis_p):
                r -= float(pq_j @ r) * q_j

            pr = preconditioner @ r
            b = float(np.sqrt(max(float(r @ pr), 0.0)))

            alpha.append(a)

            if b < 1.0e-10 * abs(a) or step == n_steps - 1:
                break

            beta.append(b)

            q_prev, q, pq, beta_prev = q, r / b, pr / b, b

        k = len(alpha)

        tridiagonal = (
            np.diag(alpha) + np.diag(beta[: k - 1], 1) + np.diag(beta[: k - 1], -1)
        )

        theta, eigenvectors = np.linalg.eigh(tridiagonal)

        probe_values[probe] = norm2 * float(
            np.sum(eigenvectors[0] ** 2 * np.log(np.clip(theta, 1.0e-300, None)))
        )

    log_det_probes = log_det_preconditioner + probe_values

    return float(np.mean(log_det_probes)), log_det_probes


def log_evidence_matrix_free_from(
    operator: MatrixFreeDataOperator,
    regularization_matrix_sparse: scipy.sparse.spmatrix,
    regularization_factor_sparse: scipy.sparse.spmatrix,
    coefficient: float,
    reconstruction: np.ndarray,
    log_det_regularization: float,
    data_term: float,
    noise_normalization: float,
    n_probes: int = 10,
    n_steps: int = 40,
    seed: int = 20221,
    log_det_curvature_reg: float = None,
) -> float:
    """
    Returns the log Bayesian evidence of a matrix-free interferometer inversion, assembling the same
    five terms with the same conventions as `FitInterferometer.log_evidence` /
    `fit_util.log_evidence_from`:

        ln Z = -0.5 * [χ² + sᵀ(λH)s + ln det(F + λH) - ln det(λH) + Σ ln(2π σ²)]

    with

    - `χ² = dᵀN⁻¹d - 2 sᵀD + sᵀFs`, the dense path's `fast_chi_squared` (`data_term = dᵀN⁻¹d`,
      `D = operator.data_vector`, `F s = operator.curvature_matvec(s)`);
    - `sᵀ(λH)s` the regularization term;
    - `ln det(F + λH)` by stochastic Lanczos quadrature (`log_det_curvature_reg_via_slq_from`), the
      only approximate term (within 0.5% at the defaults on the test cases), unless
      `log_det_curvature_reg` is given;
    - `ln det(λH) = S ln λ + log_det_regularization`, exact (for `MaternSparse`,
      `log_det_regularization_matrix_term_from`);
    - the noise normalisation `Σ ln(2π σ²)`.

    With `λ = coefficient = 1.0` and an autoarray regularization matrix (which already carries its
    coefficient) every term is the dense sparse-path quantity.

    Parameters
    ----------
    operator
        The data operator providing `F` and `D`.
    regularization_matrix_sparse
        The regularization matrix `H` as a `scipy.sparse` matrix.
    regularization_factor_sparse
        A sparse factor `R` with `H = RᵀR` (see `log_det_curvature_reg_via_slq_from`).
    coefficient
        The scalar `λ` multiplying `H`; `1.0` for an autoarray regularization matrix.
    reconstruction
        The reconstruction `s` (e.g. from `reconstruction_via_pcg_from`).
    log_det_regularization
        The exact `ln det H` of `regularization_matrix_sparse` (without `λ`).
    data_term
        `dᵀN⁻¹d = Σ d_r²/σ_r² + Σ d_i²/σ_i²`, as cached on the dataset's `sparse_operator.data_term`
        (or the dataset interface's own `data_term` for profile-subtracted visibilities).
    noise_normalization
        `Σ ln(2π σ²)` over the real and imaginary visibility noise, as cached on the dataset's
        `sparse_operator.noise_normalization`.
    n_probes, n_steps, seed
        The SLQ settings (see `log_det_curvature_reg_via_slq_from`).
    log_det_curvature_reg
        An optional precomputed `ln det(F + λH)` used in place of the SLQ estimate (e.g. the dense
        value, for parity checks of the other terms).

    Returns
    -------
    The log evidence.
    """
    regularization_matrix_sparse = scipy.sparse.csr_matrix(regularization_matrix_sparse)

    s = np.asarray(reconstruction, dtype=np.float64)

    chi_squared = (
        float(data_term)
        - 2.0 * float(s @ operator.data_vector)
        + float(s @ operator.curvature_matvec(s))
    )

    regularization_term = coefficient * float(s @ (regularization_matrix_sparse @ s))

    if log_det_curvature_reg is None:
        log_det_curvature_reg, _ = log_det_curvature_reg_via_slq_from(
            operator=operator,
            regularization_matrix_sparse=regularization_matrix_sparse,
            regularization_factor_sparse=regularization_factor_sparse,
            coefficient=coefficient,
            n_probes=n_probes,
            n_steps=n_steps,
            seed=seed,
        )

    log_det_regularization_term = s.size * np.log(coefficient) + float(
        log_det_regularization
    )

    return float(
        -0.5
        * (
            chi_squared
            + regularization_term
            + log_det_curvature_reg
            - log_det_regularization_term
            + float(noise_normalization)
        )
    )


class PositiveReconstructionResult:
    def __init__(
        self,
        reconstruction: np.ndarray,
        support: np.ndarray,
        interior_point_steps: int,
        polish_iterations: int,
        gap: float,
        residual: float,
        converged: bool,
    ):
        """
        The output of `reconstruction_positive_via_interior_point_from`.

        Parameters
        ----------
        reconstruction
            The non-negative reconstruction `s`, with exact zeros on the held (active) set.
        support
            Boolean mask of the free (passive) pixels, those solved for; `reconstruction` is zero
            wherever it is `False`.
        interior_point_steps
            The number of Mehrotra predictor-corrector steps taken.
        polish_iterations
            The number of exact free-set solves in the active-set polish (one if the support detected
            from the interior point was already optimal).
        gap
            The normalised duality gap `n μ / |xᵀD|` of the final interior-point iterate.
        residual
            The normalised dual residual `|A x - D - z| / |D|` of the final interior-point iterate.
        converged
            `True` if the active-set polish ended on a support satisfying the KKT conditions (no
            negative free pixel and no held pixel with a negative gradient), `False` if it reached
            its iteration cap and the result was clipped at zero.
        """
        self.reconstruction = reconstruction
        self.support = support
        self.interior_point_steps = interior_point_steps
        self.polish_iterations = polish_iterations
        self.gap = gap
        self.residual = residual
        self.converged = converged


def _step_to_boundary(v: np.ndarray, dv: np.ndarray) -> float:
    """
    The largest step `α <= 1` keeping `v + α dv >= 0` (for `v > 0`).
    """
    decreasing = dv < 0.0

    if not np.any(decreasing):
        return 1.0

    return float(min(1.0, np.min(-v[decreasing] / dv[decreasing])))


def reconstruction_positive_via_interior_point_from(
    operator: MatrixFreeDataOperator,
    regularization_matrix_sparse: scipy.sparse.spmatrix,
    coefficient: float,
    max_steps: int = 60,
    tol: float = 1.0e-9,
    cg_rtol_min: float = 1.0e-8,
    cg_rtol_max: float = 1.0e-2,
    polish_rtol: float = 1.0e-10,
    max_polish: int = 10,
    maxiter: int = 5000,
) -> PositiveReconstructionResult:
    """
    Returns the non-negative reconstruction solving

        min_s  ½ sᵀ(F + λH)s - sᵀD    subject to  s >= 0

    (equivalently `(F + λH) s = D` on the free pixels, zero elsewhere) without forming or factorising
    the dense `(S, S)` matrix `A = F + λH`: every operation is a matrix-free product with `A` or a
    sparse factorisation of the Powell preconditioner `P = κ₀ MᵀM + λH` plus a diagonal.

    This is the matrix-free counterpart of the positive-only (`use_positive_only_solver=True`) solve
    of an inversion's `curvature_reg_matrix`. The method follows @HRSAstro's reference
    (PyAutoArray #638, phase 4) in two stages.

    **Primal-dual interior point (Mehrotra predictor-corrector).** The KKT conditions are
    `A x - D = z`, `x >= 0`, `z >= 0`, `x z = 0`. Starting from one Jacobi step,
    `x₀ = max(D / diag, 0)` shifted by `10⁻² max(x₀)` into the interior (with `diag` the diagonal of
    `P`, the Powell approximation of the diagonal of `A`), and a dual `z₀` from the positive part of
    the gradient, each step eliminates `dz` from the linearised KKT system, leaving

        (A + Z X⁻¹) dx = -r_d - z + c,       r_d = A x - D - z,

    with `c = 0` for the affine (predictor) direction and `c = (σμ - dx_aff dz_aff) / x` for the
    combined direction, centring parameter `σ = (μ_aff / μ)³` and `μ = xᵀz / n`. Each of the two
    solves is CG preconditioned by the barrier-diagonal-augmented Powell matrix `P + Z X⁻¹`, refactorised
    by a sparse LU every step (the barrier diagonal changes); the corrector is warm-started at the
    predictor. The CG tolerance is tied to the progress, `0.1 min(gap, residual + gap)`, clipped to
    `[cg_rtol_min, cg_rtol_max]`, so early steps are cheap. Steps are damped to 0.99 of the distance to
    the boundary. Iteration stops when the normalised gap `n μ / |xᵀD|` and dual residual
    `|r_d| / |D|` are both below `tol`, or after `max_steps`.

    **Exact active-set polish.** The interior point never reaches exact zeros, so its support is read
    off as the free set `{i : z_i <= diag_i x_i}` and the free pixels are solved exactly,
    `A_ff s_f = D_f`, by CG preconditioned by the sparse LU of the sub-matrix `P_ff` and warm-started
    at the interior-point iterate; the held pixels are exactly zero. The KKT conditions are then
    checked through the multiplier `g = A s - D` and the support updated by the primal-dual active-set
    rule (Hintermüller, Ito & Kunisch 2002): a pixel is held when `g_i > diag_i s_i`, which releases
    held pixels with a negative gradient and holds free pixels that came out negative. This repeats
    until the support is unchanged (normally after one or two solves), or `max_polish` solves, after
    which the solution is clipped at zero and `converged` is `False`.

    The contributor found projected Newton and a pure active-set method stall at weak priors with
    ~30% of pixels at zero; the interior point is insensitive to the size of the active set and the
    polish starts from an essentially correct support, so the result matches `scipy.optimize.nnls`
    on the whitened dense system.

    Parameters
    ----------
    operator
        The data operator providing `F` and `D`.
    regularization_matrix_sparse
        The regularization matrix `H` as a `scipy.sparse` matrix.
    coefficient
        The scalar `λ` multiplying `H`; `1.0` for an autoarray regularization matrix, which already
        carries its coefficient.
    max_steps
        The maximum number of interior-point steps.
    tol
        The normalised gap and dual residual at which the interior point stops.
    cg_rtol_min, cg_rtol_max
        The clip range of the interior-point CG tolerance.
    polish_rtol
        The CG tolerance of the exact free-set solves, relative to `|D_f|`.
    max_polish
        The maximum number of free-set solves in the active-set polish.
    maxiter
        The maximum number of iterations of every CG solve.

    Returns
    -------
    A `PositiveReconstructionResult` holding the reconstruction (exact zeros on the held set), the
    support mask, the interior-point step and polish iteration counts and the final gap and residual.
    """
    regularization_matrix_sparse = scipy.sparse.csr_matrix(regularization_matrix_sparse)

    n = operator.params
    data_vector = operator.data_vector
    data_vector_norm = float(np.linalg.norm(data_vector))

    def curvature_reg_matvec(v):
        return operator.curvature_matvec(v) + coefficient * (
            regularization_matrix_sparse @ v
        )

    preconditioner, _ = powell_preconditioner_from(
        operator=operator,
        regularization_matrix_sparse=regularization_matrix_sparse,
        coefficient=coefficient,
    )
    preconditioner = preconditioner.tocsc()
    diag = preconditioner.diagonal()

    def splu_from(matrix):
        return scipy.sparse.linalg.splu(
            matrix.tocsc(),
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options=dict(SymmetricMode=True),
        )

    # Starting point: one Jacobi step, shifted into the interior, and a positive dual.
    x = np.clip(data_vector / diag, 0.0, None)
    x = x + 1.0e-2 * max(float(np.max(x)), np.finfo(float).tiny)

    gradient = curvature_reg_matvec(x) - data_vector
    z = np.maximum(gradient, 0.0) + 1.0e-2 * max(
        float(np.max(np.abs(gradient))), np.finfo(float).tiny
    )

    interior_point_steps = 0
    gap = residual = np.inf

    for _ in range(max_steps):
        dual_residual = gradient - z
        mu = float(x @ z) / n

        gap = n * mu / abs(float(x @ data_vector))
        residual = float(np.linalg.norm(dual_residual)) / data_vector_norm

        if gap < tol and residual < tol:
            break

        interior_point_steps += 1

        barrier_diagonal = z / x

        factor = splu_from(preconditioner + scipy.sparse.diags(barrier_diagonal))

        newton_operator = scipy.sparse.linalg.LinearOperator(
            (n, n),
            matvec=lambda v: curvature_reg_matvec(v) + barrier_diagonal * v,
            dtype=np.float64,
        )
        preconditioner_operator = scipy.sparse.linalg.LinearOperator(
            (n, n), matvec=factor.solve, dtype=np.float64
        )

        rtol = float(np.clip(0.1 * min(gap, residual + gap), cg_rtol_min, cg_rtol_max))

        # Predictor (affine scaling) direction.
        dx, _ = scipy.sparse.linalg.cg(
            newton_operator,
            -dual_residual - z,
            rtol=rtol,
            maxiter=maxiter,
            M=preconditioner_operator,
        )
        dz = -z - barrier_diagonal * dx

        mu_affine = (
            float(
                (x + _step_to_boundary(x, dx) * dx)
                @ (z + _step_to_boundary(z, dz) * dz)
            )
            / n
        )
        sigma = (mu_affine / mu) ** 3

        # Combined centring + corrector direction, warm-started at the predictor.
        correction = (sigma * mu - dx * dz) / x

        dx, _ = scipy.sparse.linalg.cg(
            newton_operator,
            -dual_residual - z + correction,
            x0=dx,
            rtol=rtol,
            maxiter=maxiter,
            M=preconditioner_operator,
        )
        dz = -z + correction - barrier_diagonal * dx

        x = x + 0.99 * _step_to_boundary(x, dx) * dx
        z = z + 0.99 * _step_to_boundary(z, dz) * dz

        gradient = curvature_reg_matvec(x) - data_vector

    # Exact active-set polish on the support detected from the interior point.
    free = z <= diag * x
    warm_start = x.copy()

    reconstruction = np.zeros(n)
    polish_iterations = 0
    converged = False

    for _ in range(max_polish):
        polish_iterations += 1

        free_indexes = np.flatnonzero(free)
        reconstruction = np.zeros(n)

        if free_indexes.size > 0:

            def free_matvec(v, free_indexes=free_indexes):
                return curvature_reg_matvec(
                    np.bincount(free_indexes, weights=v, minlength=n)
                )[free_indexes]

            sub_factor = splu_from(preconditioner[free_indexes][:, free_indexes])

            free_operator = scipy.sparse.linalg.LinearOperator(
                (free_indexes.size, free_indexes.size),
                matvec=free_matvec,
                dtype=np.float64,
            )
            sub_preconditioner_operator = scipy.sparse.linalg.LinearOperator(
                (free_indexes.size, free_indexes.size),
                matvec=sub_factor.solve,
                dtype=np.float64,
            )

            reconstruction_free, _ = scipy.sparse.linalg.cg(
                free_operator,
                data_vector[free_indexes],
                x0=warm_start[free_indexes],
                rtol=polish_rtol,
                maxiter=maxiter,
                M=sub_preconditioner_operator,
            )

            reconstruction[free_indexes] = reconstruction_free

        # Primal-dual active-set update: the multiplier of the held pixels is the gradient.
        multiplier = curvature_reg_matvec(reconstruction) - data_vector
        multiplier[free] = 0.0

        free_new = ~(multiplier > diag * reconstruction)

        if np.array_equal(free_new, free):
            converged = bool(np.all(reconstruction >= 0.0))
            break

        free = free_new
        warm_start = np.clip(reconstruction, 0.0, None)

    if not converged:
        reconstruction = np.clip(reconstruction, 0.0, None)
        free = reconstruction > 0.0

    return PositiveReconstructionResult(
        reconstruction=reconstruction,
        support=free,
        interior_point_steps=interior_point_steps,
        polish_iterations=polish_iterations,
        gap=float(gap),
        residual=float(residual),
        converged=converged,
    )
