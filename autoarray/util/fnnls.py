import numpy as np

from autoarray.util.cholesky_funcs import cholinsertlast, choldeleteindexes

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
):
    """
    Similar to fnnls, but use solving the lstsq problem by updating Cholesky factorisation.
    """
    from scipy import linalg as slg

    lstsq = lambda A, x: slg.solve(
        A,
        x,
        assume_a="pos",
        overwrite_a=True,
        overwrite_b=True,
    )

    n = np.shape(ZTZ)[0]
    epsilon = 2.2204e-16
    tolerance = epsilon * n
    max_repetitions = 3
    no_update = 0
    loop_count = 0
    loop_count2 = 0

    P = np.zeros(n, dtype=bool)
    P[P_initial] = True
    d = np.zeros(n)
    w = ZTx - (ZTZ) @ d
    s_chol = np.zeros(n)

    if P_initial.shape[0] != 0:
        P_number = np.arange(len(P), dtype="int")
        P_inorder = P_number[P_initial]
        s_chol[P] = lstsq((ZTZ)[P][:, P], (ZTx)[P])
        d = s_chol.clip(min=0)
    else:
        P_inorder = np.array([], dtype="int")

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

        if loop_count == 0:
            # We need to initialize the Cholesky factorisation, U, for the first loop.
            U = slg.cholesky(ZTZ[P_inorder][:, P_inorder])
        else:
            U = cholinsertlast(U, ZTZ[idmax][P_inorder])

        # solve the lstsq problem by cho_solve

        s_chol[P_inorder] = slg.cho_solve((U, False), ZTx[P_inorder])

        P[idmax] = True
        while np.any(P) and np.min(s_chol[P]) <= tolerance:
            s_chol, d, P, P_inorder, U = fix_constraint_cholesky(
                ZTx=ZTx,
                s_chol=s_chol,
                d=d,
                P=P,
                P_inorder=P_inorder,
                U=U,
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

    return d


def fix_constraint_cholesky(ZTx, s_chol, d, P, P_inorder, U, tolerance):
    """
    Similar to fix_constraint, but solve the lstsq by Cholesky factorisation.
    If this function is called, it means some solutions in the current passive sets needed to be
        taken out and put into the active set.
    So, this function involves 3 procedure:
        1. Identifying what solutions should be taken out of the current passive set.
        2. Updating the P, P_inorder and the Cholesky factorisation U.
        3. Solving the lstsq by using the new Cholesky factorisation U.
    As some solutions are taken out from the passive set, the Cholesky factorisation needs to be
            updated by choldeleteindexes. To realize that, we call the `choldeleteindexes` from
            cholesky_funcs.
    """
    q = P * (s_chol <= tolerance)
    alpha = np.min(d[q] / (d[q] - s_chol[q]))

    # set d as close to s as possible while maintaining non-negativity
    d = d + alpha * (s_chol - d)

    id_delete = np.where(d[P_inorder] <= tolerance)[0]

    U = choldeleteindexes(U, id_delete)  # update the Cholesky factorisation

    P_inorder = np.delete(P_inorder, id_delete)  # update the P_inorder

    P[d <= tolerance] = False  # update the P

    # solve the lstsq problem by cho_solve

    if len(P_inorder):
        from scipy import linalg as slg

        # there could be a case where P_inorder is empty.
        s_chol[P_inorder] = slg.cho_solve((U, False), ZTx[P_inorder])

    s_chol[~P] = 0.0  # set solutions taken out of the passive set to be 0

    return s_chol, d, P, P_inorder, U
