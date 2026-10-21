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

This phase lands the operator data term and the solver as library utilities, with parity tests
against the dense sparse-path quantities. It does **not** wire them into fits and adds no
`Settings` flag. Later phases (sparse Matérn precision, stochastic Lanczos log-determinant for the
evidence, interior-point positivity, profiling and workspace examples) follow as their own tasks.

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
    solve of an inversion's `curvature_reg_matrix`. No positivity is imposed (an interior-point
    positive solve is a later phase).

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
