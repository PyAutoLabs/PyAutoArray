"""
A sparse Matérn regularization for rectangular meshes, built from the SPDE construction of the Matérn
field (Lindgren, Rue & Lindström 2011).

This is phase 2 of the matrix-free interferometer inversion epic (PyAutoArray #638, community proposal by
@HRSAstro, https://github.com/PyAutoLabs/.github/discussions/39). The dense `MaternKernel` builds the
`(S, S)` Matérn covariance `C` and inverts it, so its regularization matrix `H = coefficient * C^-1` is
fully dense. For integer `alpha = nu + 1` (in 2D) the Matérn *precision* `C^-1` is instead sparse: it is
a polynomial in the discrete Laplacian. This module builds it for `nu = 1` (`alpha = 2`), giving an `H`
with the 13-point stencil of `L^2`, which the matrix-free utilities of
`inversion_interferometer_matrix_free_util` take directly as a `scipy.sparse` matrix.

The module is NumPy / scipy only and imports no JAX.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.regularization.abstract import validate_coefficient


def rectangular_laplacian_sparse_from(
    shape_native: Tuple[int, int], pixel_scales: Tuple[float, float]
) -> scipy.sparse.csr_matrix:
    """
    Returns the 5-point discrete Laplacian of a rectangular mesh with reflecting (zero-flux Neumann)
    edges, as a `scipy.sparse` CSR matrix of shape `(pixels, pixels)`.

    Mesh pixels are indexed in row-major order, which is the order of a rectangular mesh's
    `source_plane_mesh_grid` (built by `grid_2d_slim_via_shape_native_not_mask_from`): pixel
    `k = i * total_x_pixels + j` is in row `i` (row 0 is the top row, the largest y) and column `j`
    (column 0 is the left column, the smallest x). Pixel `k` therefore neighbours `k +- 1` along x (within
    its row) and `k +- total_x_pixels` along y.

    The Laplacian is the adjacency matrix minus its row sums, each axis divided by the square of its
    own pixel scale, so for an interior pixel

        (lap s)_k = (s_{k-1} - 2 s_k + s_{k+1}) / dx^2 + (s_{k-nx} - 2 s_k + s_{k+nx}) / dy^2

    and an edge pixel simply drops its missing neighbours (and the matching diagonal term), which is the
    reflecting boundary condition. The rows of the matrix sum to zero.

    Parameters
    ----------
    shape_native
        The 2D dimensions of the rectangular mesh, `(total_y_pixels, total_x_pixels)`.
    pixel_scales
        The `(y, x)` spacing between neighbouring mesh pixel centres.

    Returns
    -------
    The sparse Laplacian.
    """
    total_y_pixels, total_x_pixels = int(shape_native[0]), int(shape_native[1])
    pixel_scale_y, pixel_scale_x = float(pixel_scales[0]), float(pixel_scales[1])

    pixels = total_y_pixels * total_x_pixels

    index = np.arange(pixels).reshape(total_y_pixels, total_x_pixels)

    rows = []
    cols = []
    data = []

    for left, right, pixel_scale in (
        (index[:, :-1], index[:, 1:], pixel_scale_x),
        (index[:-1, :], index[1:, :], pixel_scale_y),
    ):
        value = 1.0 / pixel_scale**2

        rows += [left.ravel(), right.ravel()]
        cols += [right.ravel(), left.ravel()]
        data += [np.full(2 * left.size, value)]

    adjacency = scipy.sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(pixels, pixels),
    )

    return (
        adjacency - scipy.sparse.diags(np.asarray(adjacency.sum(axis=1)).ravel())
    ).tocsr()


def _splu_symmetric_from(matrix: scipy.sparse.spmatrix):
    """
    Sparse LU of a symmetric positive definite matrix with the symmetric-mode options (`MMD_AT_PLUS_A`
    ordering, no pivoting threshold), matching `powell_preconditioner_from`.
    """
    return scipy.sparse.linalg.splu(
        scipy.sparse.csc_matrix(matrix),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
        options=dict(SymmetricMode=True),
    )


class MaternSparse(AbstractRegularization):
    def __init__(
        self,
        coefficient: float = 1.0,
        scale: float = 1.0,
        nu: float = 1.0,
        weights: Optional[np.ndarray] = None,
    ):
        """
        Regularization which uses a Matérn prior on the source, expressed through its sparse precision
        matrix (the SPDE construction of Lindgren, Rue & Lindström 2011) rather than a dense covariance.

        The dense `MaternKernel` regularizes every pixel with every other pixel by building the Matérn
        covariance `C` and inverting it, so its regularization matrix is dense and costs `O(S^3)` to form.
        For integer `alpha = nu + 1` a Matérn field on a 2D grid is the solution of the stochastic PDE
        `L^(alpha / 2) f = noise` with

            L = kappa^2 I - lap,    kappa = sqrt(2 nu) / scale,

        where `lap` is the discrete Laplacian. Its precision is then `C^-1 = c L^alpha`, a sparse matrix.
        This class implements `nu = 1` (`alpha = 2`):

            H = coefficient * W^-1 c L^2 W^-1 = coefficient * R^T R,    R = sqrt(c) L W^-1,

        with:

        - `lap` the 5-point Laplacian with reflecting edges (`rectangular_laplacian_sparse_from`), each
          axis scaled by its own mesh pixel scale;
        - `c = e^T L^-2 e` with `e` the indicator of the mesh's centre pixel, so the prior variance at the
          mesh centre, `e^T (c L^2)^-1 e = (e^T L^-2 e) / c`, is 1, matching the unit zero-lag value `K(0) = 1` of the dense kernel (edge
          pixels, with fewer neighbours, have a larger variance under reflecting edges);
        - `W = diag(w)` the per-pixel kernel weights (`kernel_weights_from`), which scale the prior
          standard deviation of each pixel exactly as `C_ij = K(d_ij) w_i w_j` does for the adaptive dense
          kernels (`MaternAdaptKernel` uses `w = 1 / regularization_weights`). They are unit by default.

        `H` has the 13-point stencil of `L^2`. It is exactly symmetric positive definite (`L` is, since
        `kappa^2 > 0`), so unlike the dense kernels no diagonal jitter is added.

        **Meshes**: only rectangular meshes (`RectangularUniform`, `RectangularRTUAdaptDensity` and their
        subclasses) are supported, because the 5-point Laplacian needs the mesh's grid topology. The
        Laplacian is built on the mapper's `source_plane_mesh_grid` (the uniform overlay grid, also the
        grid the dense kernel measures distances on), whose pixels are in row-major order from the
        top-left (see `rectangular_laplacian_sparse_from`). Other meshes (e.g. `Delaunay`) raise
        `NotImplementedError`.

        **Smoothness**: only `nu = 1` is implemented. Higher integer `nu` is `c L^(nu + 1)`, a follow-up;
        non-integer `alpha` has no exact sparse precision.

        **Matrix-free use**: `regularization_matrix_sparse_from` returns `H` as a `scipy.sparse` matrix,
        to be passed to `powell_preconditioner_from` / `reconstruction_via_pcg_from` with `coefficient=1.0`
        (it already carries this class's coefficient), and `regularization_factor_sparse_from` returns
        `R`, which draws the evidence probes `R^T z`. `regularization_matrix_from` returns the dense array
        of the same `H`, so the class is a drop-in on today's dense inversion path.

        The class is NumPy / scipy only: `xp` is honoured only for the dense array returned by
        `regularization_matrix_from`, and none of its methods can be traced by JAX.

        A full description of regularization and this matrix can be found in the parent
        `AbstractRegularization` class.

        Parameters
        ----------
        coefficient
            The regularization coefficient which controls the degree of smoothing of the inversion
            reconstruction.
        scale
            The correlation length of the Matérn prior, in the units of the mesh coordinates.
        nu
            The smoothness of the Matérn prior. Only `nu = 1` is implemented.
        weights
            Optional per-pixel kernel weights `w` of shape `[pixels]` (the prior standard deviation of each
            pixel, relative to the unit centre variance). `None` gives unit weights. Subclasses coupling the
            weights to the data (e.g. adaptive brightness) override `kernel_weights_from` instead.
        """
        if nu != 1.0:
            raise NotImplementedError(
                f"MaternSparse implements only nu = 1 (alpha = nu + 1 = 2, precision c L^2); "
                f"nu = {nu} was input. Higher integer nu (precision c L^(nu + 1)) is a follow-up, and "
                f"non-integer alpha has no exact sparse precision -- use MaternKernel for those."
            )

        validate_coefficient(coefficient=coefficient)

        self.coefficient = coefficient
        self.scale = scale
        self.nu = nu
        self.weights = weights

        super().__init__()

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        As for the dense `MaternKernel`, these are the coefficient for every pixel; the per-pixel prior
        scaling of this scheme is held by `kernel_weights_from`.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        return self.coefficient * xp.ones(linear_obj.params)

    def kernel_weights_from(self, linear_obj: LinearObj) -> np.ndarray:
        """
        Returns the per-pixel kernel weights `w` entering `H = coefficient * W^-1 c L^2 W^-1`: the input
        `weights` if given, otherwise ones.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The kernel weights, shape `[pixels]`.
        """
        if self.weights is None:
            return np.ones(linear_obj.params)

        weights = np.asarray(self.weights, dtype=float)

        if weights.shape != (linear_obj.params,):
            raise ValueError(
                f"MaternSparse weights have shape {weights.shape} but the mesh has "
                f"{linear_obj.params} pixels."
            )

        return weights

    @staticmethod
    def mesh_shape_and_pixel_scales_from(
        linear_obj: LinearObj,
    ) -> Tuple[Tuple[int, int], Tuple[float, float]]:
        """
        Returns the `(total_y_pixels, total_x_pixels)` shape and the `(y, x)` spacing of pixel centres of
        a linear object's rectangular mesh, read from its `source_plane_mesh_grid`.

        Raises `NotImplementedError` if the linear object is not a mapper with a rectangular mesh.
        """
        from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
            RectangularRTUAdaptDensity,
        )

        mesh = getattr(linear_obj, "mesh", None)

        if not isinstance(mesh, RectangularRTUAdaptDensity):
            raise NotImplementedError(
                f"MaternSparse supports only rectangular meshes (RectangularUniform, "
                f"RectangularRTUAdaptDensity and their subclasses), because its sparse precision uses the "
                f"5-point Laplacian of the mesh grid; got a {type(mesh).__name__} mesh. Use MaternKernel "
                f"for other meshes (e.g. Delaunay)."
            )

        shape_native = (int(mesh.shape[0]), int(mesh.shape[1]))

        grid = np.asarray(linear_obj.source_plane_mesh_grid.array).reshape(
            shape_native[0], shape_native[1], 2
        )

        # Row-major from the top-left: y decreases down the rows, x increases along the columns.
        pixel_scales = (
            float(grid[0, 0, 0] - grid[1, 0, 0]),
            float(grid[0, 1, 1] - grid[0, 0, 1]),
        )

        return shape_native, pixel_scales

    def _laplacian_operator_and_c_from(self, linear_obj: LinearObj):
        """
        Returns `L = kappa^2 I - lap` (CSC), its sparse LU and the centre-variance normalization `c`.
        """
        shape_native, pixel_scales = self.mesh_shape_and_pixel_scales_from(
            linear_obj=linear_obj
        )

        pixels = shape_native[0] * shape_native[1]

        kappa = np.sqrt(2.0 * self.nu) / self.scale

        laplacian_operator = (
            kappa**2 * scipy.sparse.identity(pixels, format="csr")
            - rectangular_laplacian_sparse_from(
                shape_native=shape_native, pixel_scales=pixel_scales
            )
        ).tocsc()

        factor = _splu_symmetric_from(laplacian_operator)

        centre = np.zeros(pixels)
        centre[(shape_native[0] // 2) * shape_native[1] + shape_native[1] // 2] = 1.0

        l_inv_centre = factor.solve(centre)

        # The covariance (c L^2)^-1 has centre variance (e^T L^-2 e) / c, so c = e^T L^-2 e makes it one;
        # e^T L^-2 e = |L^-1 e|^2 since L is symmetric.
        c = float(l_inv_centre @ l_inv_centre)

        return laplacian_operator, factor, c

    def regularization_factor_sparse_from(
        self, linear_obj: LinearObj
    ) -> scipy.sparse.csr_matrix:
        """
        Returns the factor `R = sqrt(c) L W^-1` of the unscaled precision, `c W^-1 L^2 W^-1 = R^T R`, as a
        `scipy.sparse` CSR matrix (the regularization matrix is `coefficient * R^T R`).

        `R^T z` with `z ~ N(0, I)` is a draw with covariance `R^T R`, which is how the stochastic evidence
        log-determinant of the matrix-free path draws its probes.

        Parameters
        ----------
        linear_obj
            The linear object (a ``Mapper`` with a rectangular mesh).

        Returns
        -------
        The sparse factor `R`.
        """
        laplacian_operator, _, c = self._laplacian_operator_and_c_from(
            linear_obj=linear_obj
        )

        weights = self.kernel_weights_from(linear_obj=linear_obj)

        return (
            np.sqrt(c) * laplacian_operator @ scipy.sparse.diags(1.0 / weights)
        ).tocsr()

    def regularization_matrix_sparse_from(
        self, linear_obj: LinearObj
    ) -> scipy.sparse.csr_matrix:
        """
        Returns the regularization matrix `H = coefficient * R^T R = coefficient * W^-1 c L^2 W^-1` as a
        `scipy.sparse` CSR matrix of shape `[pixels, pixels]`.

        This is the matrix the matrix-free utilities take; pass it with `coefficient=1.0` there, since it
        already carries this class's coefficient.

        Parameters
        ----------
        linear_obj
            The linear object (a ``Mapper`` with a rectangular mesh).

        Returns
        -------
        The sparse regularization matrix.
        """
        factor = self.regularization_factor_sparse_from(linear_obj=linear_obj)

        return (self.coefficient * (factor.T @ factor)).tocsr()

    def regularization_matrix_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization matrix with shape [pixels, pixels], as the dense array of
        `regularization_matrix_sparse_from`, so this scheme is a drop-in on the dense inversion path.

        Parameters
        ----------
        linear_obj
            The linear object (a ``Mapper`` with a rectangular mesh).

        Returns
        -------
        The regularization matrix.
        """
        return xp.asarray(
            self.regularization_matrix_sparse_from(linear_obj=linear_obj).toarray()
        )

    def log_det_regularization_matrix_term_from(
        self, linear_obj: LinearObj, xp=np
    ) -> float:
        """
        The analytically exact ``log det H`` from the sparse LU of `L`:

            log det H = pixels * log(coefficient) + pixels * log(c) + 2 * sum(log|U_ii|) - 2 * sum(log w)

        since `H = coefficient * c W^-1 L^2 W^-1`, `det(L^2) = det(L)^2` and the LU's `L` factor has a
        unit diagonal (`L` is SPD, so `det L > 0`). This is the same quantity the dense `MaternKernel`
        returns (``log det H``, there ``pixels * log(coefficient) - log det C``).

        Consumed by the inversion only when ``Settings.log_det_method == "slogdet"`` (see
        :meth:`AbstractRegularization.log_det_regularization_matrix_term_from`); the default evidence path
        factorizes the formed ``H``.
        """
        _, factor, c = self._laplacian_operator_and_c_from(linear_obj=linear_obj)

        weights = self.kernel_weights_from(linear_obj=linear_obj)

        pixels = linear_obj.params

        return float(
            pixels * np.log(self.coefficient)
            + pixels * np.log(c)
            + 2.0 * np.sum(np.log(np.abs(factor.U.diagonal())))
            - 2.0 * np.sum(np.log(weights))
        )

    def regularization_term_from(
        self, linear_obj: LinearObj, reconstruction: np.ndarray, xp=np
    ) -> float:
        """
        The regularization term ``s^T H s``, evaluated as ``coefficient * |R s|^2`` with the sparse factor
        `R`, so no dense matrix is formed.

        Consumed by the inversion only when ``Settings.regularization_term_method == "cho_solve"`` (see
        :meth:`AbstractRegularization.regularization_term_from`); the default ``"matmul"`` path contracts
        the formed ``H``.
        """
        factor = self.regularization_factor_sparse_from(linear_obj=linear_obj)

        r_s = factor @ np.asarray(reconstruction, dtype=float)

        return float(self.coefficient * (r_s @ r_s))
