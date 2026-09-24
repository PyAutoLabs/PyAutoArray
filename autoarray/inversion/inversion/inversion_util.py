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

    In many circumstances, adding a small numerical value to the diagonal of the `curvature_matrix`
    makes it positive definite, such that the inversion is performed without raising an error.

    This function adds the caller-supplied `value` to the diagonal entries selected by
    `no_regularization_index_list`. The normal inversion path reads this value from
    `Settings.no_regularization_add_to_curvature_diag_value`; the packaged configuration defaults to
    `1.0e-3`, and workspaces may override it. The addition is absolute, so its effect depends on the scale of
    the curvature matrix.

    Parameters
    ----------
    curvature_matrix
        The curvature matrix which is being constructed in order to solve a linear system of equations.
    value
        The numerical value added to each selected diagonal entry.
    no_regularization_index_list
        The indices of parameters without regularization whose diagonal entries receive `value`.
    xp
        The array module to use (`numpy` by default; pass `jax.numpy` for JAX support).

    Returns
    -------
    ndarray
        The curvature matrix with `value` added to the selected diagonal entries.
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
    # The noise weighting is fp64 on every backend, whatever
    # `settings.use_mixed_precision` says. The mapping matrix arrives fp64 from
    # every operated path (the mixed-precision FFT upcasts at the kernel
    # multiply), so the former fp32 `1 / noise_map` on the JAX branch never
    # bought an fp32 accumulation -- it only rounded the weights of F
    # inconsistently with the fp64 `1 / noise_map**2` of the data vector
    # (PyAutoArray#552).
    A = xp.asarray(mapping_matrix, dtype=xp.float64) / noise_map[:, None]
    curvature_matrix = xp.dot(A.T, A)

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


def _certified_positive_only_from(
    Q, q, settings, target_kappa, solver_tol, max_iter, stats=None
):
    """
    The JAX certified active-set positive-only solve of ``(Q, q)`` with the PDIP solve as its fallback.

    Called by `reconstruction_positive_only_from` on its (usually Jacobi-scaled) system; see
    :mod:`autoarray.util.jax_active_set` for the algorithm, budgets, gradient contract and vmap caveat.
    """
    from autoarray.util.jax_active_set import solve_certified_with_fallback
    from autoarray.util.jax_nnls import solve_nnls_primal

    settings = settings or Settings()

    def pdip_fn():
        return solve_nnls_primal(
            Q,
            q,
            target_kappa=target_kappa,
            solver_tol=solver_tol,
            max_iter=max_iter,
        )

    x, certified, passes = solve_certified_with_fallback(
        Q,
        q,
        pdip_fn=pdip_fn,
        fallback=settings.certified_fallback == "pdip",
        pass_budget=int(settings.certified_pass_budget),
        tau_rel=float(settings.certified_tau_rel),
    )

    if stats is not None:
        stats["solver"] = "certified"
        stats["certified"] = certified
        stats["passes"] = passes

    return x


def reconstruction_positive_only_from(
    data_vector: np.ndarray,
    curvature_reg_matrix: np.ndarray,
    settings: Settings = None,
    xp=np,
    fingerprint=None,
    factor: Optional[dict] = None,
    solver: str = "pdip",
    stats: Optional[dict] = None,
    preconditioning: str = "jacobi",
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
    fingerprint
        Identifies the index space this solve's passive set lives in, enabling the cross-evaluation warm-start
        memo (`Settings.nnls_warm_start_memo`) on the NumPy path. `None` disables the memo for this call.
    factor
        If a dict is passed it is cleared on entry and filled with the Cholesky factor of the solve's passive
        submatrix that `fnnls_cholesky` built anyway -- see that function's `factor` parameter for the keys, and
        `log_det_from_passive_cholesky_from` for the caller that reads them. It is handed to *both*
        `fnnls_cholesky` calls below, so a memo-seeded attempt that raises cannot leave the factor of a solve
        whose result was discarded behind for the retry's caller to read. Purely observational: the returned
        reconstruction is byte-identical whether or not it is passed.
    solver
        Which positive-only solver the JAX (`xp=jnp`) path uses: ``"pdip"`` (default, the jaxnnls primal-dual
        interior-point solve, byte-identical to before this option existed) or ``"certified"`` (the certified
        active-set solve of :mod:`autoarray.util.jax_active_set`, applied to the same Jacobi-scaled system, with
        the PDIP solve as its fallback when ``settings.certified_fallback == "pdip"`` and the pass budget
        ``settings.certified_pass_budget`` is exhausted). The caller (`AbstractInversion.reconstruction`) passes
        ``"certified"`` only for mapper-only inversions on the JAX backend -- see
        `AbstractInversion.positive_only_solver_used`. The NumPy path ignores it and always runs fnnls.
    stats
        Optional out-dict for solver observability on the JAX path. With ``solver="certified"`` it receives
        ``certified`` (whether the active-set search certified within budget; ``False`` means the fallback or an
        uncertified iterate was returned) and ``passes`` (restricted passes run) as *traced* JAX scalars, so it is
        safe under ``jax.jit`` / ``vmap`` -- read them as outputs of the traced function or through
        ``jax.debug.callback``. With ``solver="pdip"`` it receives ``converged`` (``1`` if the PDIP KKT residual
        met its tolerance within the iteration cap, ``0`` if the cap was hit and the returned reconstruction is
        the unconverged iterate) and ``iterations`` (PDIP iterations run), also as traced JAX scalars, plus
        ``preconditioning``. Every call also records ``solver``. It never changes the returned reconstruction.
    preconditioning
        How the JAX PDIP solve (``solver="pdip"``) scales the system (PyAutoArray#571). ``"jacobi"`` (default,
        byte-identical to before this option existed): the Jacobi-preconditioned solve ``(D Q D) y = D q``
        governed by the ``nnls_jacobi_preconditioning`` config key. ``"raw"``: the forward PDIP solve runs on
        the un-preconditioned ``(Q, q)`` with the data-scaled tolerance
        :func:`autoarray.util.jax_nnls.data_scaled_solver_tol` (or ``settings.nnls_solver_tol`` if set), and the
        gradient is the Jacobi-space relaxed-KKT pass as in ``"jacobi"`` -- see
        :func:`autoarray.util.jax_nnls.solve_nnls_primal_raw_forward`. Jacobi scaling makes the signal-free
        columns of linear-object-only (MGE) inversions, whose diagonal is only the
        ``no_regularization_add_to_curvature_diag_value`` floor, degenerate coordinates on which the PDIP dual
        diverges; the caller (`AbstractInversion.reconstruction`) therefore passes ``"raw"`` for inversions with
        no `Mapper` -- see `AbstractInversion.positive_only_preconditioning_used`. Ignored on the NumPy path.

    Notes
    -----
    On the NumPy path this function writes two keys into the `stats` dict it passes to `fnnls_cholesky`
    that `fnnls_cholesky` itself knows nothing about: ``seed_source`` (``"memo"`` if the solve that
    produced the returned reconstruction started from a memo seed, ``"dense"`` if it started from the
    sign of the unconstrained dense solve) and ``warm_start_fallback`` (`True` if a memo seed breached
    `Settings.nnls_warm_start_error_tolerance` and its entry was dropped, so the next solve for that key
    restarts dense). They are set after `fnnls_cholesky` returns, so a diagnostic wrapping the solver
    must read the dict it handed in *after* the evaluation, not at the point the solver returns.

    Returns
    -------
    Non-negative S that minimizes the Eq.(2) of https://arxiv.org/pdf/astro-ph/0302587.pdf.
    """
    if factor is not None:
        # Cleared before anything else, so that every route which does not reach a successful
        # `fnnls_cholesky` -- the JAX branch, a memo-seeded attempt that raises, the test-mode
        # dummy -- leaves the caller reading "no factor" rather than a factor belonging to some
        # other matrix. `fnnls_cholesky` publishes only on a successful return.
        factor.clear()

    if solver not in ("pdip", "certified"):
        raise ValueError(
            f"solver={solver!r} is not a valid positive-only solver; expected 'pdip' or 'certified'."
        )

    if preconditioning not in ("jacobi", "raw"):
        raise ValueError(
            f"preconditioning={preconditioning!r} is invalid; expected 'jacobi' or 'raw'."
        )

    if preconditioning == "raw" and solver != "pdip":
        raise ValueError(
            "preconditioning='raw' applies only to solver='pdip'; the certified solver always runs on the "
            "Jacobi-scaled system."
        )

    if xp.__name__.startswith("jax"):

        from autonerves import conf

        from autoarray.util.jax_nnls import (
            solve_nnls_primal_raw_forward,
            solve_nnls_primal_with_status,
        )

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
            target_kappa = conf.instance["general"]["inversion"]["nnls_target_kappa"]
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

        def _record_pdip(converged, iterations):
            if stats is not None:
                stats["solver"] = "pdip"
                stats["preconditioning"] = preconditioning
                stats["converged"] = converged
                stats["iterations"] = iterations

        if preconditioning == "raw":
            # Same Jacobi quantities as below: the backward pass runs on the scaled system, the forward
            # solve on the raw one (see `solve_nnls_primal_raw_forward`).
            d = xp.sqrt(xp.diag(curvature_reg_matrix))
            D = 1.0 / d
            Q_pc = (curvature_reg_matrix * D[:, None]) * D[None, :]
            q_pc = data_vector * D

            y, converged, iterations = solve_nnls_primal_raw_forward(
                Q_pc,
                q_pc,
                curvature_reg_matrix,
                data_vector,
                D,
                target_kappa=target_kappa,
                solver_tol=solver_tol,
                max_iter=max_iter,
            )
            _record_pdip(converged, iterations)
            return y * D

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

            if solver == "certified":
                return (
                    _certified_positive_only_from(
                        Q=Q_pc,
                        q=q_pc,
                        settings=settings,
                        target_kappa=target_kappa,
                        solver_tol=solver_tol,
                        max_iter=max_iter,
                        stats=stats,
                    )
                    * D
                )

            x, converged, iterations = solve_nnls_primal_with_status(
                Q_pc,
                q_pc,
                target_kappa=target_kappa,
                solver_tol=solver_tol,
                max_iter=max_iter,
            )
            _record_pdip(converged, iterations)
            return x * D

        if solver == "certified":
            return _certified_positive_only_from(
                Q=curvature_reg_matrix,
                q=data_vector,
                settings=settings,
                target_kappa=target_kappa,
                solver_tol=solver_tol,
                max_iter=max_iter,
                stats=stats,
            )

        x, converged, iterations = solve_nnls_primal_with_status(
            curvature_reg_matrix,
            data_vector,
            target_kappa=target_kappa,
            solver_tol=solver_tol,
            max_iter=max_iter,
        )
        _record_pdip(converged, iterations)
        return x

    # `solver` is deliberately ignored on the NumPy path: a NumPy port of the certified active-set scheme
    # measured 3-7 % slower than fnnls and lost to its warm-start memo (PyAutoArray#566), so fnnls stays.
    try:

        from autoarray.util.fnnls import fnnls_cholesky
        from autoarray.inversion.inversion import nnls_memo

        use_memo = (
            settings is not None
            and settings.nnls_warm_start_memo
            and nnls_memo.memo_enabled()
            and fingerprint is not None
        )

        stats = {}

        key = (
            nnls_memo.memo_key(n=data_vector.shape[0], fingerprint=fingerprint)
            if use_memo
            else None
        )

        n = data_vector.shape[0]

        entry = nnls_memo.passive_set_get(key=key, n=n) if use_memo else None

        stats["seed_source"] = "dense"
        stats["warm_start_fallback"] = False

        if entry is not None:
            try:
                reconstruction = fnnls_cholesky(
                    curvature_reg_matrix,
                    (data_vector).T,
                    P_initial=entry.passive_set,
                    stats=stats,
                    factor=factor,
                )
                stats["seed_source"] = "memo"
            except (RuntimeError, np.linalg.LinAlgError, ValueError):
                # A seed from a previous evaluation is a guess about a
                # different matrix, so it can factorise badly where the
                # dense-sign start would not. That must cost one retry, never a
                # resample: fall back to exactly the un-memoized computation
                # before the InversionException path below is reached.
                reconstruction = None
        else:
            reconstruction = None

        if reconstruction is None:
            reconstruction = fnnls_cholesky(
                curvature_reg_matrix,
                (data_vector).T,
                P_initial=np.linalg.solve(curvature_reg_matrix, data_vector) > 0,
                stats=stats,
                factor=factor,
            )
            stats["seed_source"] = "dense"

        if use_memo:
            error_fraction = stats["warm_start_errors"] / max(n, 1)

            if stats["seed_source"] == "memo":
                # A memo seed is judged against the dense-sign start it
                # replaced, not against an absolute error count: the absolute
                # fraction does not separate seeds that save iterations from
                # seeds that cost them, the ratio to the dense-sign reference
                # does. A seed that breaches the tolerance is discarded, so the
                # next solve for this key restarts dense and refreshes the
                # reference -- no stale seed can be dragged through a run in a
                # regime the reference was never measured in.
                tolerance = settings.nnls_warm_start_error_tolerance

                guard_active = (
                    tolerance is not None and np.isfinite(tolerance) and tolerance > 0.0
                )

                if (
                    guard_active
                    and error_fraction > tolerance * entry.dense_error_fraction
                ):
                    nnls_memo.memo_drop(key=key)
                    stats["warm_start_fallback"] = True
                else:
                    # The reference describes the dense-sign start, so it is
                    # carried forward unchanged; only a dense solve refreshes it.
                    nnls_memo.passive_set_put(
                        key=key,
                        passive_set=stats["passive_set"],
                        dense_error_fraction=entry.dense_error_fraction,
                    )
            else:
                nnls_memo.passive_set_put(
                    key=key,
                    passive_set=stats["passive_set"],
                    dense_error_fraction=error_fraction,
                )

        return reconstruction

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


#: The mean non-zeros per row up to which :func:`log_det_sparse_spd_from` factorizes a matrix
#: sparsely rather than returning `None` and leaving it to the dense Cholesky.
#:
#: This is a **regime switch, not a tolerance**. Two regimes of regularization matrix are
#: produced by this library and nothing in between:
#:
#: - a **geometric stencil** `H` (the neighbour schemes and the split family), whose non-zeros
#:   follow the planar graph of a mesh -- ~5-30 non-zeros per row independent of `pixels`, and
#:   low fill-in under a minimum-degree ordering. The production HST Delaunay `pixels=1500`
#:   `AdaptSplit` matrix measures 8.45 non-zeros per row and factorizes 6.1x faster sparsely
#:   than densely (36.9 ms -> 6.1 ms at 1 thread).
#: - a **kernel** `H` (`MaternKernel` and its siblings), which is `coefficient * C^-1` and so
#:   fully dense -- `pixels` non-zeros per row, where a sparse factorization is ~7x *slower*
#:   than the dense Cholesky at `pixels=1500`.
#:
#: 32 sits far above the first regime and far below the second. Random sparse patterns of this
#: density would fill in catastrophically and lose to the dense Cholesky, but no scheme in this
#: library produces one: every sparse `H` here is a mesh's adjacency.
SPARSE_LOG_DET_MAX_NNZ_PER_ROW = 32

#: The smallest `pixels` at which :func:`log_det_sparse_spd_from` factorizes a matrix sparsely
#: rather than returning `None` and leaving it to the dense Cholesky.
#:
#: A sparse factorization carries a fixed ~0.14 ms of Python and SuperLU setup which a small
#: dense Cholesky beats outright. Measured on a split-stencil pattern at 1 thread, sparse/dense
#: is 0.07x at `pixels=9`, 0.13x at 64, 0.38x at 128, **1.8x at 256**, 5.0x at 512 and 10.8x at
#: 1500 -- so 256 is the first measured size where the sparse route pays for itself. Below it
#: the term costs under 0.1 ms either way and the dense Cholesky runs, which also keeps the
#: small unit-test and NumPy/JAX parity fixtures on the historical value exactly.
SPARSE_LOG_DET_MIN_PIXELS = 256


def log_det_from_passive_cholesky_from(
    matrix: np.ndarray,
    U_buffer: np.ndarray,
    k_active: int,
    passive_set: np.ndarray,
) -> float:
    """
    Returns `log det M` of a symmetric positive-definite `matrix` M from the Cholesky factor of
    one of its principal submatrices, via the Schur complement of the remaining block.

    The positive-only reconstruction is a non-negative least squares solve, and `fnnls_cholesky`
    finishes it holding a Cholesky factor `U` of `M[P][:, P]`, where `P` is the solve's final
    *passive* set (the pixels whose reconstructed value is non-zero). On the production HST
    Delaunay `pixels=1500` `AdaptSplit` system that is 1485 of the 1500 columns. The Bayesian
    evidence then needs `log det M` of the full system, and until this function existed it
    factorised the whole `[pixels, pixels]` matrix a second time from scratch -- ~40 ms of a
    ~270 ms NumPy likelihood call, redoing 99% of the work `fnnls_cholesky` had just done.

    Partitioning M by the passive set `P` and the active set `A` (its complement),

        log det M = log det M_PP + log det(M_AA - M_AP M_PP^-1 M_PA)

    -- the block-determinant identity, where the second term is the Schur complement of the
    already-factorised block. The first term is `2 * sum(log(diag(U)))`, free. The second needs
    one triangular solve `U^T Y = M_PA` with `|A|` right-hand sides and one `|A| x |A|`
    Cholesky, because `Y^T Y == M_AP M_PP^-1 M_PA`. With `|A|` of order ten that is ~5 ms
    instead of ~40 ms, and it agrees with the dense factorisation to <= 2e-12 nats (measured
    across six instances of the production system; the fiducial fit's log evidence is
    bit-identical).

    Both blocks must be positive-definite for M to be, so a failure of either Cholesky raises
    `np.linalg.LinAlgError` -- the same error the dense route raises on the same matrix, so the
    caller's test-mode guard and `FitException` resampling apply unchanged. This function never
    silently substitutes a value for a matrix it could not factorise.

    Parameters
    ----------
    matrix
        The `[pixels, pixels]` symmetric positive-definite matrix M whose log determinant is
        computed, e.g. an inversion's `curvature_reg_matrix_reduced`.
    U_buffer
        The buffer `fnnls_cholesky` published, whose leading `k_active x k_active` upper
        triangle is a Cholesky factor `U` with `matrix[P][:, P] == U.T @ U`. Only that corner is
        read; the rest of the buffer is zero and meaningless.
    k_active
        The size of that valid leading block, i.e. `len(passive_set)`.
    passive_set
        The passive indices **in the order the factor's rows and columns are in**. They are
        neither sorted nor contiguous (the active-set solver appends and deletes), so this order
        is what pairs `U` with `matrix`, and it must be used for the off-diagonal block too.

    Returns
    -------
    The log determinant of `matrix`.
    """
    from scipy.linalg import cholesky, solve_triangular

    passive_set = np.asarray(passive_set)

    upper_passive = U_buffer[:k_active, :k_active]

    log_det = 2.0 * float(np.sum(np.log(np.diag(upper_passive))))

    active = np.setdiff1d(np.arange(matrix.shape[0]), passive_set)

    if active.size == 0:
        # Every column is passive: the factor is of the whole matrix already.
        return log_det

    # `trans=1` solves `U^T Y = M_PA` against the upper factor in place of transposing it,
    # which would hand LAPACK a non-contiguous array to copy. All `|A|` right-hand sides go
    # in one call: `cholesky_funcs._solve_upper_transposed_buffer` is the copy-free kernel the
    # solver itself uses, but it takes a single vector and overwrites it in place, so reaching
    # for it here would mean `|A|` numba calls over `|A|` writable copies to save a LAPACK
    # call on a 1485 x 15 block. It is not a clean fit at this size.
    y = solve_triangular(
        upper_passive,
        matrix[np.ix_(passive_set, active)],
        trans=1,
        lower=False,
        check_finite=False,
    )

    schur = matrix[np.ix_(active, active)] - y.T @ y

    # `cholesky` raises `np.linalg.LinAlgError` if this block is not positive-definite, which
    # is propagated deliberately: it means `matrix` is not either, and the caller must treat
    # that exactly as it treats a failure of the dense factorisation.
    return log_det + 2.0 * float(
        np.sum(np.log(np.diag(cholesky(schur, lower=False, check_finite=False))))
    )


def log_det_sparse_spd_from(matrix: np.ndarray) -> Optional[float]:
    """
    Returns `log det M` of a sparse, symmetric positive-definite `matrix` via a sparse LU
    factorization, or `None` if the matrix is too dense for that to be the faster route.

    The two Bayesian-evidence log-determinant terms historically factorized their matrices
    densely. For `log_det_regularization_matrix_term` on the split regularization family that
    is the wrong algorithm: `H` has `O(1)` non-zeros per row (8.45 on the production HST
    Delaunay `pixels=1500` `AdaptSplit` matrix), so a `O(pixels^3)` dense Cholesky spends
    almost all of its work on structural zeros -- ~37 ms of a ~300 ms NumPy likelihood call.

    The factorization is SuperLU in symmetric mode (`permc_spec="MMD_AT_PLUS_A"`,
    `diag_pivot_thresh=0.0`), which orders for the symmetric pattern and pivots on the
    diagonal, so `diag(U)` are the factorization's pivots and `log det M = sum(log(diag(U)))`
    (`L` has a unit diagonal). Because row pivoting is off, those pivots are positive if and
    only if the matrix is positive-definite, which is checked: a non-positive pivot raises
    `np.linalg.LinAlgError` exactly as `np.linalg.cholesky` does, and an exactly singular
    matrix's `RuntimeError` from SuperLU is converted to the same error. The caller's
    test-mode guard therefore applies unchanged.

    The `[pixels, pixels]` dense `matrix` still has to be scanned to find its non-zeros, and
    that scan is a meaningful share of the sparse route's cost (2.9 ms of the 6.1 ms at
    `pixels=1500`), so it is done once: the boolean mask and its per-row counts give both the
    density regime check and, directly, the CSC `data`/`indices`/`indptr` triple, with no
    second pass and no `scipy.sparse.csc_matrix(dense)` conversion (which costs 13 ms).

    The row-major scan builds the CSR triple, which is passed to `csc_matrix` as if it were
    CSC. For the symmetric matrices this is called on the two are identical; more generally it
    factorizes `M.T`, whose determinant is the same, so the returned value is correct either
    way.

    Parameters
    ----------
    matrix
        The `[pixels, pixels]` symmetric positive-definite matrix whose log determinant is
        computed, e.g. an inversion's `regularization_matrix_reduced`.

    Returns
    -------
    The log determinant of the matrix, or `None` where the dense Cholesky is the faster route
    and the caller should use it: matrices smaller than `SPARSE_LOG_DET_MIN_PIXELS`, and
    matrices with more than `SPARSE_LOG_DET_MAX_NNZ_PER_ROW` non-zeros per row on average.
    """
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    pixels = matrix.shape[0]

    if pixels < SPARSE_LOG_DET_MIN_PIXELS:
        return None

    non_zeros = matrix != 0.0
    non_zeros_per_row = np.count_nonzero(non_zeros, axis=1)

    if non_zeros_per_row.sum() > SPARSE_LOG_DET_MAX_NNZ_PER_ROW * pixels:
        return None

    flat_indices = np.flatnonzero(non_zeros)

    indptr = np.empty(pixels + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(non_zeros_per_row, out=indptr[1:])

    matrix_sparse = csc_matrix(
        (
            matrix.ravel()[flat_indices],
            (flat_indices % matrix.shape[1]).astype(np.int32),
            indptr,
        ),
        shape=matrix.shape,
    )

    try:
        lu = splu(
            matrix_sparse,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options=dict(SymmetricMode=True),
        )
    except RuntimeError as e:
        # SuperLU raises `RuntimeError("Factor is exactly singular")`; the dense Cholesky
        # raises `LinAlgError` on the same matrix and the caller guards on that.
        raise np.linalg.LinAlgError(str(e)) from e

    pivots = lu.U.diagonal()

    if not np.all(pivots > 0.0):
        raise np.linalg.LinAlgError("Matrix is not positive definite")

    return float(np.sum(np.log(pivots)))
