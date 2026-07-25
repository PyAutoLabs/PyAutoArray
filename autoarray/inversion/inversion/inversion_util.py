import numpy as np

from typing import List, Optional, Type

from autonerves import is_test_mode

from autoarray.settings import Settings

from autoarray import exc


def curvature_matrix_diag_via_psf_weighted_noise_from(
    psf_weighted_noise: np.ndarray, mapping_matrix: np.ndarray, xp=np
) -> np.ndarray:
    """
    Returns the curvature matrix `F` (see Warren & Dye 2003) from the `psf_weighted_noise`.

    The dimensions of `psf_weighted_noise` are [image_pixels, image_pixels], meaning that for datasets with many image
    pixels this matrix can take up 10's of GB of memory. The calculation of the `curvature_matrix` via this function
    will therefore be very slow, and the method `curvature_matrix_diag_via_sparse_operator_from` should be used
    instead.

    Parameters
    ----------
    psf_weighted_noise
        A matrix of dimensions [image_pixels, image_pixels] that encodes the convolution or NUFFT of every image pixel
        pair on the noise map.
    mapping_matrix
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.

    Returns
    -------
    ndarray
        The curvature matrix `F` (see Warren & Dye 2003).
    """
    return xp.dot(mapping_matrix.T, xp.dot(psf_weighted_noise, mapping_matrix))


def curvature_matrix_with_added_to_diag_from(
    curvature_matrix: np.ndarray,
    value: float,
    no_regularization_index_list: Optional[List] = None,
    xp=np,
) -> np.ndarray:
    """
    It is common for the `curvature_matrix` computed to not be positive-definite, leading for the inversion
    via `np.linalg.solve` to fail and raise a `LinAlgError`.

    In many circumstances, adding a small numerical value of `1.0e-8` to the diagonal of the `curvature_matrix`
    makes it positive definite, such that the inversion is performed without raising an error.

    This function adds this numerical value to the diagonal of the curvature matrix.

    Parameters
    ----------
    curvature_matrix
        The curvature matrix which is being constructed in order to solve a linear system of equations.
    """
    if xp.__name__.startswith("jax"):
        return curvature_matrix.at[
            no_regularization_index_list, no_regularization_index_list
        ].add(value)
    curvature_matrix[
        no_regularization_index_list, no_regularization_index_list
    ] += value
    return curvature_matrix


def curvature_matrix_mirrored_from(curvature_matrix: np.ndarray, xp=np) -> np.ndarray:

    # Copy the original matrix and its transpose
    m1 = curvature_matrix
    m2 = curvature_matrix.T

    # For each entry, prefer the non-zero value from either the matrix or its transpose
    mirrored = xp.where(m1 != 0, m1, m2)

    return mirrored


def curvature_matrix_via_mapping_matrix_from(
    mapping_matrix: "np.ndarray",
    noise_map: "np.ndarray",
    add_to_curvature_diag: bool = False,
    no_regularization_index_list: Optional[List] = None,
    settings: "Settings" = Settings(),
    xp=np,
) -> np.ndarray:
    r"""
    Returns the curvature matrix `F` from a blurred mapping matrix `f` and the 1D noise-map $\sigma$
     (see Warren & Dye 2003).

    Parameters
    ----------
    mapping_matrix
        The matrix representing the mappings (these could be blurred or transformed) between sub-grid pixels and
        pixelization pixels.
    noise_map
        Flattened 1D array of the noise-map used by the inversion during the fit.
    add_to_curvature_diag
        If `True`, adds a small numerical value to the diagonal entries of the curvature matrix at the indices
        specified by `no_regularization_index_list`, to ensure the matrix is positive-definite.
    no_regularization_index_list
        A list of parameter indices which have no regularization applied. A small value is added to their diagonal
        entries if `add_to_curvature_diag` is `True`.
    settings
        Settings controlling how the inversion is performed, for example whether mixed precision is used.
    xp
        The array module to use (`numpy` by default; pass `jax.numpy` for JAX support).
    """
    # NumPy path: keep it simple + stable
    if xp is np:
        A = mapping_matrix / noise_map[:, None]
        curvature_matrix = xp.dot(A.T, A)
    else:
        # Choose compute dtype

        compute_dtype = xp.float32 if settings.use_mixed_precision else xp.float64
        out_dtype = xp.float64  # always return float64 for downstream stability

        A = mapping_matrix
        w = (1.0 / noise_map).astype(compute_dtype)
        A = A * w[:, None]
        curvature_matrix = xp.dot(A.T, A).astype(out_dtype)

    if (
        add_to_curvature_diag
        and no_regularization_index_list
        and len(no_regularization_index_list) > 0
    ):
        curvature_matrix = curvature_matrix_with_added_to_diag_from(
            curvature_matrix=curvature_matrix,
            value=settings.no_regularization_add_to_curvature_diag_value,
            no_regularization_index_list=no_regularization_index_list,
            xp=xp,
        )

    return curvature_matrix


def mapped_reconstructed_data_via_mapping_matrix_from(
    mapping_matrix: np.ndarray, reconstruction: np.ndarray, xp=np
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the blurred mapping matrix `f` and solution vector *S*.

    Parameters
    ----------
    mapping_matrix
        The matrix representing the blurred mappings between sub-grid pixels and pixelization pixels.

    """
    return xp.dot(mapping_matrix, reconstruction)


def mapped_reconstructed_data_via_psf_weighted_noise_from(
    psf_weighted_noise: np.ndarray,
    mapping_matrix: np.ndarray,
    reconstruction: np.ndarray,
) -> np.ndarray:
    """
    Returns the reconstructed data vector from the unblurred mapping matrix `M`,
    the reconstruction vector `s`, and the PSF convolution operator `psf_weighted_noise`.

    Equivalent to:
        reconstructed = (W @ M) @ s
                      = W @ (M @ s)

    Parameters
    ----------
    psf_weighted_noise
        Array of shape [image_pixels, image_pixels], the PSF convolution operator.
    mapping_matrix
        Array of shape [image_pixels, source_pixels], unblurred mapping matrix.
    reconstruction
        Array of shape [source_pixels], solution vector.

    Returns
    -------
    ndarray
        The reconstructed data vector of shape [image_pixels].
    """
    return psf_weighted_noise @ (mapping_matrix @ reconstruction)


def reconstruction_positive_negative_from(
    data_vector: np.ndarray,
    curvature_reg_matrix: np.ndarray,
    xp=np,
):
    """
    Solve the linear system [F + reg_coeff*H] S = D -> S = [F + reg_coeff*H]^-1 D given by equation (12)
    of https://arxiv.org/pdf/astro-ph/0302587.pdf

    S is the vector of reconstructed inversion values.

    This reconstruction uses a linear algebra solver that allows for negative and positives values in the solution.
    By allowing negative values, the solver is efficient, but there are many inference problems where negative values
    are nonphysical or undesirable.

    This function checks that the solution does not give a linear algebra error (e.g. because the input matrix is
    not positive-definitive).

    It also explicitly checks solutions where all reconstructed values go to the same value, and raises an exception if
    this occurs. This solution occurs in many scenarios when it is clear not a valid solution, and therefore is checked
    for and removed.

    Parameters
    ----------
    data_vector
        The `data_vector` D which is solved for.
    curvature_reg_matrix
        The sum of the curvature and regularization matrices.
    mapper_param_range_list
        A list of lists, where each list contains the range of values in the solution vector (reconstruction) that
        correspond to values that are part of a mapper's mesh.
    force_check_reconstruction
        If `True`, the reconstruction is forced to check for solutions where all reconstructed values go to the same
        value irrespective of the configuration file value.

    Returns
    -------
    curvature_reg_matrix
        The curvature_matrix plus regularization matrix, overwriting the curvature_matrix in memory.
    """
    try:
        return xp.linalg.solve(curvature_reg_matrix, data_vector)
    except np.linalg.LinAlgError:
        if is_test_mode():
            # Test-mode fits run fabricated / under-converged models whose
            # curvature-regularization matrix can be singular; the reconstruction
            # value is discarded. Return a benign dummy so unguarded test-mode
            # paths (search chaining, preloads, result construction) do not crash.
            # Normal runs re-raise unchanged and resample via the likelihood's
            # FitException guard — real-mode numerics are untouched. (The JAX path
            # returns NaN rather than raising, so this branch is numpy-only.)
            return xp.ones_like(data_vector)
        raise


def reconstruction_positive_only_from(
    data_vector: np.ndarray,
    curvature_reg_matrix: np.ndarray,
    settings: Settings = None,
    xp=np,
):
    """
    Solve the linear system Eq.(2) (in terms of minimizing the quadratic value) of
    https://arxiv.org/pdf/astro-ph/0302587.pdf. Not finding the exact solution of Eq.(3) or Eq.(4).

    This reconstruction uses a linear algebra optimizer that allows only positives values in the solution.
    By not allowing negative values, the solver is slower than methods which allow negative values, but there are
    many inference problems where negative values are nonphysical or undesirable and removing them improves the solution.

    The non-negative optimizer we use is a modified version of fnnls (https://github.com/jvendrow/fnnls). The algorithm
    is published by:

    Bro & Jong (1997) ("A fast non‐negativity‐constrained least squares algorithm."
                Journal of Chemometrics: A Journal of the Chemometrics Society 11, no. 5 (1997): 393-401.)

    The modification we made here is that we create a function called fnnls_Cholesky which directly takes ZTZ and ZTx
    as inputs. The reason is that we realize for this specific algorithm (Bro & Jong (1997)), ZTZ and ZTx happen to
    be the curvature_reg_matrix and data_vector, respectively, already defined in PyAutoArray (verified). Besides,
    we build a Cholesky scheme that solves the lstsq problem in each iteration within the fnnls algorithm by updating
    the Cholesky factorisation.

    Please note that we are trying to find non-negative solution S that minimizes |Z * S - x|^2. We are not trying to
    find a solution that minimizes |ZTZ * S - ZTx|^2! ZTZ and ZTx are just some variables help to
    minimize |Z * S - x|^2. It is just a coincidence (or fundamentally not) that ZTZ and ZTx are the
    curvature_reg_matrix and data_vector, respectively.

    If we no longer uses fnnls (the algorithm of Bro & Jong (1997)), we need to check if the algorithm takes Z or
    ZTZ (x or ZTx) as an input. If not, we need to build Z and x in PyAutoArray.

    Parameters
    ----------
    data_vector
        The `data_vector` D happens to be the ZTx.
    curvature_reg_matrix
        The sum of the curvature and regularization matrices. Taken as ZTZ in our problem.
    settings
        Controls the settings of the inversion, for this function where the solution is checked to not be all
        the same values.\

    Returns
    -------
    Non-negative S that minimizes the Eq.(2) of https://arxiv.org/pdf/astro-ph/0302587.pdf.
    """
    if xp.__name__.startswith("jax"):

        from autonerves import conf

        from autoarray.util.jax_nnls import solve_nnls_primal

        try:
            use_jacobi = conf.instance["general"]["inversion"][
                "nnls_jacobi_preconditioning"
            ]
        except KeyError:
            # Workspaces ship their own general.yaml that shadows autoarray's;
            # default to True so gradients remain well-defined unless the user
            # explicitly disables preconditioning in the shadowing config.
            use_jacobi = True

        try:
            target_kappa = conf.instance["general"]["inversion"][
                "nnls_target_kappa"
            ]
        except KeyError:
            # Workspaces ship their own general.yaml that shadows autoarray's;
            # fall back to the same value autoarray's general.yaml declares.
            # jaxnnls's own 1e-3 default produces NaN in the relaxed-KKT
            # backward pass on ill-conditioned curvature matrices; 1e-11 is
            # finite across all MGE/rectangular/delaunay pipelines (imaging +
            # interferometer) with scale invariance verified over 5 orders of
            # magnitude in noise.
            target_kappa = 1.0e-11

        # Per-fit solver knobs from the Settings class; the defaults (None)
        # reproduce jaxnnls's own tolerance min(n * eps * 5e3, 1e-2) and its
        # hard-coded 50-iteration cap exactly.
        solver_tol = settings.nnls_solver_tol if settings is not None else None
        max_iter = settings.nnls_max_iter if settings is not None else None
        if max_iter is None:
            max_iter = 50

        if use_jacobi:
            # Ill-conditioned Q makes jaxnnls's relaxed-KKT backward pass
            # produce NaN gradients. Rescale Q so its diagonal is unit:
            # solve (D Q D) y = D q with y >= 0, recover x = D y. D is
            # diagonal positive, so non-negativity is preserved and the
            # primal solution is mathematically equivalent.
            d = xp.sqrt(xp.diag(curvature_reg_matrix))
            D = 1.0 / d
            Q_pc = (curvature_reg_matrix * D[:, None]) * D[None, :]
            q_pc = data_vector * D
            return (
                solve_nnls_primal(
                    Q_pc,
                    q_pc,
                    target_kappa=target_kappa,
                    solver_tol=solver_tol,
                    max_iter=max_iter,
                )
                * D
            )

        return solve_nnls_primal(
            curvature_reg_matrix,
            data_vector,
            target_kappa=target_kappa,
            solver_tol=solver_tol,
            max_iter=max_iter,
        )

    try:

        from autoarray.util.fnnls import fnnls_cholesky

        return fnnls_cholesky(
            curvature_reg_matrix,
            (data_vector).T,
            P_initial=np.linalg.solve(curvature_reg_matrix, data_vector) > 0,
        )

    except (RuntimeError, np.linalg.LinAlgError, ValueError) as e:
        if is_test_mode():
            # See reconstruction_positive_negative_from: benign dummy in test mode,
            # unchanged (raise InversionException) in normal operation.
            return xp.ones_like(data_vector)
        raise exc.InversionException() from e


def preconditioner_matrix_via_mapping_matrix_from(
    mapping_matrix: np.ndarray,
    regularization_matrix: np.ndarray,
    preconditioner_noise_normalization: float,
) -> np.ndarray:
    """
    Returns the preconditioner matrix `{` from a mapping matrix `f` and the sum of the inverse of the 1D noise-map
    values squared (see Powell et al. 2020).

    Parameters
    ----------
    mapping_matrix
        The matrix representing the mappings between sub-grid pixels and pixelization pixels.
    regularization_matrix
        The matrix defining how the pixelization's pixels are regularized with one another for smoothing (H).
    preconditioner_noise_normalization
        The sum of (1.0 / noise-map**2.0) every value in the noise-map.
    """

    curvature_matrix = curvature_matrix_via_mapping_matrix_from(
        mapping_matrix=mapping_matrix,
        noise_map=np.ones(shape=(mapping_matrix.shape[0])),
    )

    return (
        preconditioner_noise_normalization * curvature_matrix
    ) + regularization_matrix


def param_range_list_from(cls: Type, linear_obj_list) -> List[List[int]]:
    """
    Each linear object in the `Inversion` has N parameters, and these parameters correspond to a certain range
    of indexing values in the matrices used to perform the inversion.

    This function returns the `param_range_list` of an input type of linear object, which gives the indexing range
    of each linear object of the input type.

    For example, if an `Inversion` has:

    - A `LinearFuncList` linear object with 3 `params`.
    - A `Mapper` with 100 `params`.
    - A `Mapper` with 200 `params`.

    The corresponding matrices of this inversion (e.g. the `curvature_matrix`) have `shape=(303, 303)` where:

    - The `LinearFuncList` values are in the entries `[0:3]`.
    - The first `Mapper` values are in the entries `[3:103]`.
    - The second `Mapper` values are in the entries `[103:303]

    For this example, `param_range_list_from(cls=Mapper)` therefore returns the
    list `[[3, 103], [103, 303]]`.

    Parameters
    ----------
    cls
        The type of class that the list of their parameter range index values are returned for.

    Returns
    -------
    A list of the index range of the parameters of each linear object in the inversion of the input cls type.
    """
    index_list = []

    pixel_count = 0

    for linear_obj in linear_obj_list:
        if isinstance(linear_obj, cls):
            index_list.append([pixel_count, pixel_count + linear_obj.params])

        pixel_count += linear_obj.params

    return index_list
