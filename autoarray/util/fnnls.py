import numpy as np

from typing import Optional

from autoarray.util.cholesky_funcs import (
    _cho_solve_buffer,
    cholinsertlast_inplace,
    choldeleteindexes_inplace,
)

from autoarray import exc

"""
    This file contains functions use the Bro & Jong (1997) algorithm to solve the non-negative least
        square problem. The `fnnls and fix_constraint` is orginally copied from 
        "https://github.com/jvendrow/fnnls".
    For our purpose in PyAutoArray, we create `fnnls_modefied` to take ZTZ and ZTx as inputs directly.
    Furthermore, we add two functions `fnnls_Cholesky and fix_constraint_Cholesky` to realize a scheme
        that solves the lstsq problem in the algorithm by Cholesky factorisation. For ~ 1000 free 
        parameters, we see a speed up by 2 times and should be more for more parameters.
    We have also noticed that by setting the P_initial to be `sla.solve(ZTZ, ZTx, assume_a='pos') > 0`
        will speed up our task (~ 1000 free parameters) by ~ 3 times as it significantly reduces the
        iteration time.
"""


def fnnls_cholesky(
    ZTZ,
    ZTx,
    P_initial=np.zeros(0, dtype=int),
    stats: Optional[dict] = None,
):
    """
    Similar to fnnls, but use solving the lstsq problem by updating Cholesky factorisation.

    Parameters
    ----------
    P_initial
        The warm-start passive set, either as a length-n boolean mask (what the
        production dense-sign start hands over) or as an integer index array.
    stats
        If a dict is passed it is filled on return with the solve's diagnostics:
        `outer_iterations`, `inner_iterations`, `passive_set` (the final passive
        indices, in the order they were added), `n_passive` and
        `warm_start_errors` (how many entries the warm start got wrong). Purely
        observational -- the returned solution is unaffected.
    """
    from scipy import linalg as slg

    # The buffer kernels below (_cho_solve_buffer / cholinsertlast_inplace)
    # overwrite their vector argument in place, so every slice handed to them
    # must be a writeable numpy array. A JAX ZTZ / ZTx — the sparse-operator
    # inversion path hands one over even when the fit itself runs the numba
    # CPU path — breaks that contract: indexing a JAX array yields another
    # JAX array, which numba maps to a *readonly* buffer and rejects at
    # compile time ("Cannot modify readonly array"). Coerce once at the
    # boundary — fancy indexing a numpy parent then hands the kernels fresh
    # writeable arrays, exactly as the scipy solvers this replaced tolerated
    # by copying internally.
    ZTZ = np.asarray(ZTZ)
    ZTx = np.asarray(ZTx)
    P_initial = np.asarray(P_initial)

    n = np.shape(ZTZ)[0]
    epsilon = 2.2204e-16
    tolerance = epsilon * n
    max_repetitions = 3
    no_update = 0
    loop_count = 0
    loop_count2 = 0

    # `P_initial` arrives either as a boolean mask (the production dense-sign
    # start, and the memo's re-seeded passive set once expanded) or as an
    # integer index array (the tests, and the historical call signature).
    # Normalise both to the pair the algorithm actually uses -- the mask `P`
    # and the insertion-ordered index array `P_inorder` -- once, here, rather
    # than re-deriving one from the other at each use.
    P = np.zeros(n, dtype=bool)

    if P_initial.dtype == bool:
        P[:] = P_initial
        P_inorder = np.where(P_initial)[0].astype(int)
    else:
        P[P_initial] = True
        P_inorder = P_initial.astype(int)

    P_initial_mask = P.copy()

    d = np.zeros(n)
    w = ZTx - (ZTZ) @ d
    s_chol = np.zeros(n)

    # The Cholesky factor of ZTZ[passive][:, passive] lives in the top-left
    # k_active x k_active corner of a single preallocated buffer, updated in
    # place by cholinsertlast_inplace / choldeleteindexes_inplace as the
    # active set changes. The buffer is allocated once (first factorisation)
    # instead of the factor being rebuilt with np.insert/np.delete every
    # iteration — the dominant cost of this solver at n ~ 1000. Zeroed, not
    # empty: the update/solve kernels only ever read the upper triangle, but
    # keeping the rest exactly zero costs one memset and keeps every k x k
    # view a valid dense factor for inspection and tests.
    U_buffer = np.zeros((n, n))
    k_active = 0

    if P_inorder.size != 0:
        # Factorise the warm-start passive set ONCE and keep the factor: the
        # outer loop below then only ever extends it by one column
        # (`cholinsertlast_inplace`). Previously the warm start did a dense
        # `slg.solve` here and the first outer iteration threw the result away
        # to rebuild the whole factor from scratch -- an O(k^3) factorisation
        # of ~1000 columns on every likelihood evaluation.
        U = slg.cholesky(ZTZ[P_inorder][:, P_inorder])
        k_active = U.shape[0]
        U_buffer[:k_active, :k_active] = U

        s_chol[P_inorder] = _cho_solve_buffer(U_buffer, k_active, ZTx[P_inorder])

        # A warm start whose passive set contains entries with a non-positive
        # unconstrained solution must be repaired BEFORE the outer loop: the
        # old code merely clipped, and if `P` happened to be all-True the outer
        # `while (not np.all(P))` never ran and the clipped -- wrong -- vector
        # was returned. The dense-sign start could not reach that state; a
        # memo-seeded start can.
        #
        # The repair is deliberately NOT `fix_constraint_cholesky`. That step
        # interpolates from the previous feasible iterate, and a warm start has
        # none: with `d` the clipped `s_chol`, every violator gives
        # `d[q] - s_chol[q] == 0`, so its `alpha` is 0/0 or x/0 and nan/inf
        # propagates into the whole solution. The alpha -> 0 limit of that step
        # is exactly "drop every violator and re-solve", so do that directly.
        # It terminates (`P` strictly shrinks) and cannot increase the
        # objective (the surviving subspace still contains the zero vector), so
        # the outer loop receives a feasible iterate exactly as it expects.
        while P_inorder.size and np.min(s_chol[P_inorder]) <= tolerance:
            id_delete = np.where(s_chol[P_inorder] <= tolerance)[0]

            k_active = choldeleteindexes_inplace(U_buffer, k_active, id_delete)

            P[P_inorder[id_delete]] = False
            P_inorder = np.delete(P_inorder, id_delete)

            s_chol[~P] = 0.0

            if P_inorder.size:
                s_chol[P_inorder] = _cho_solve_buffer(
                    U_buffer, k_active, ZTx[P_inorder]
                )

            loop_count2 += 1
            if loop_count2 > 10000:
                raise RuntimeError

        d = s_chol.copy()
        w = ZTx - (ZTZ) @ d

    # P_inorder is similar as P. They are both used to select solutions in the passive set.
    # P_inorder saves the `indexes` of those passive solutions.
    # P saves [True/False] for all solutions. True indicates a solution in the passive set while False
    #     indicates it's in the active set.
    # The benifit of P_inorder is that we are able to not only select out solutions in the passive set
    #     and can sort them in the order of added to the passive set. This will make updating the
    #     Cholesky factorisation simpler and thus save time.

    while (not np.all(P)) and np.max(w[~P]) > tolerance:
        # make copy of passive set to check for change at end of loop

        current_P = P.copy()
        idmax = np.argmax(w * ~P)
        P_inorder = np.append(P_inorder, int(idmax))

        if k_active == 0:
            # Cold start (or a passive set emptied by the constraint fixer):
            # there is no factor to extend, so build the 1 x 1 one.
            U = slg.cholesky(ZTZ[P_inorder][:, P_inorder])
            k_active = U.shape[0]
            U_buffer[:k_active, :k_active] = U
        else:
            k_active = cholinsertlast_inplace(
                U_buffer, k_active, ZTZ[idmax][P_inorder]
            )

        # solve the lstsq problem via the copy-free buffer cho_solve

        s_chol[P_inorder] = _cho_solve_buffer(U_buffer, k_active, ZTx[P_inorder])

        P[idmax] = True
        while np.any(P) and np.min(s_chol[P]) <= tolerance:
            s_chol, d, P, P_inorder, k_active = fix_constraint_cholesky(
                ZTx=ZTx,
                s_chol=s_chol,
                d=d,
                P=P,
                P_inorder=P_inorder,
                U_buffer=U_buffer,
                k_active=k_active,
                tolerance=tolerance,
            )

            loop_count2 += 1
            if loop_count2 > 10000:
                raise RuntimeError

        d = s_chol.copy()
        w = ZTx - (ZTZ) @ d
        loop_count += 1

        if loop_count > 10000:
            raise RuntimeError

        if np.all(current_P == P):
            no_update += 1
        else:
            no_update = 0

        if no_update >= max_repetitions:
            break

    if not np.all(np.isfinite(d)):
        # A non-finite solution is not a solution. NaN/inf never raises on its
        # own, so without this check a degenerate solve is returned to the
        # caller as if it were a valid reconstruction -- and a NaN
        # reconstruction goes on to poison the adapt image, and through it the
        # mesh vertices of the next pixelization stage, where it finally
        # surfaces as an unrelated-looking qhull "Points cannot contain NaN".
        # Fail here instead, at the producer, with the same exception type the
        # inversion machinery already handles.
        raise np.linalg.LinAlgError(
            "fnnls_cholesky produced a non-finite solution "
            f"({np.count_nonzero(~np.isfinite(d))} of {d.size} entries). The "
            f"normal-equations matrix is singular to working precision."
        )

    if stats is not None:
        stats["outer_iterations"] = loop_count
        stats["inner_iterations"] = loop_count2
        stats["passive_set"] = P_inorder.copy()
        stats["n_passive"] = int(P_inorder.size)
        stats["warm_start_errors"] = int(np.count_nonzero(P_initial_mask != P))

    return d


def fix_constraint_cholesky(ZTx, s_chol, d, P, P_inorder, U_buffer, k_active, tolerance):
    """
    Similar to fix_constraint, but solve the lstsq by Cholesky factorisation.
    If this function is called, it means some solutions in the current passive sets needed to be
        taken out and put into the active set.
    So, this function involves 3 procedure:
        1. Identifying what solutions should be taken out of the current passive set.
        2. Updating the P, P_inorder and the Cholesky factorisation (the active
           k_active x k_active corner of U_buffer, updated in place).
        3. Solving the lstsq by using the new Cholesky factorisation.
    As some solutions are taken out from the passive set, the Cholesky factorisation needs to be
            updated in place by `choldeleteindexes_inplace` from cholesky_funcs.
    """
    q = P * (s_chol <= tolerance)
    alpha = np.min(d[q] / (d[q] - s_chol[q]))

    # set d as close to s as possible while maintaining non-negativity
    d = d + alpha * (s_chol - d)

    id_delete = np.where(d[P_inorder] <= tolerance)[0]

    # update the Cholesky factorisation

    k_active = choldeleteindexes_inplace(U_buffer, k_active, id_delete)

    P_inorder = np.delete(P_inorder, id_delete)  # update the P_inorder

    P[d <= tolerance] = False  # update the P

    # solve the lstsq problem via the copy-free buffer cho_solve

    if len(P_inorder):
        # there could be a case where P_inorder is empty.
        s_chol[P_inorder] = _cho_solve_buffer(U_buffer, k_active, ZTx[P_inorder])

    s_chol[~P] = 0.0  # set solutions taken out of the passive set to be 0

    return s_chol, d, P, P_inorder, k_active
