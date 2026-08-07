import numpy as np
import math
import time
from autoarray import numba_util


@numba_util.jit()
def _choldowndate(U, x):
    n = x.size
    for k in range(n - 1):
        Ukk = U[k, k]
        xk = x[k]
        r = math.sqrt(Ukk**2 - xk**2)
        c = r / Ukk
        s = xk / Ukk
        U[k, k] = r
        U[k, k + 1 :] = (U[k, (k + 1) :] - s * x[k + 1 :]) / c
        x[k + 1 :] = c * x[k + 1 :] - s * U[k, k + 1 :]

    k = n - 1
    U[k, k] = math.sqrt(U[k, k] ** 2 - x[k] ** 2)
    return U


@numba_util.jit()
def _cholupdate(U, x):
    n = x.size
    for k in range(n - 1):
        Ukk = U[k, k]
        xk = x[k]

        r = np.sqrt(Ukk**2 + xk**2)

        c = r / Ukk
        s = xk / Ukk
        U[k, k] = r

        U[k, k + 1 :] = (U[k, (k + 1) :] + s * x[k + 1 :]) / c
        x[k + 1 :] = c * x[k + 1 :] - s * U[k, k + 1 :]

    k = n - 1
    U[k, k] = np.sqrt(U[k, k] ** 2 + x[k] ** 2)

    return U


def _pivot_from_schur(schur, diagonal, index):
    """
    Turn the Schur complement of a Cholesky insertion into the new pivot.

    The pivot is `sqrt(schur)`, which is only defined for a positive-definite
    matrix. When the matrix being factorised is singular -- as it is when two
    source-plane mesh vertices (near-)coincide, giving (near-)identical columns
    in the mapping matrix -- `schur` is zero to within rounding, and which side
    of zero it lands on depends purely on floating-point summation order (and
    therefore on the BLAS thread count).

    All three of those roundings are the same degenerate matrix, so they must
    fail the same way. Taking `sqrt` directly does not do that: a tiny negative
    raises `ValueError`, but an exact zero yields a zero diagonal in `U` and a
    tiny positive yields a pivot that the following `cho_solve` amplifies --
    both of which return NaN *without raising*, so the caller cannot tell a
    degenerate solve from a good one.

    The test is `schur > 0` and nothing stricter, which is deliberate:

    * `schur < 0` already raised (`math.sqrt` of a negative), so rejecting it
      changes nothing -- `LinAlgError` subclasses `ValueError`, so every
      existing `except ValueError` still catches it.
    * `schur == 0` is the silent case. The pivot is exactly zero, so the
      following `cho_solve` divides by zero and the reconstruction is NaN with
      certainty. There is no finite result to preserve.
    * `schur > 0`, however small, still yields a positive finite pivot, so it
      is passed through untouched and returns a BITWISE IDENTICAL pivot to the
      old code.

    A relative tolerance was tried here first and rejected: it also refused
    small-but-positive pivots that were producing perfectly usable
    reconstructions, which would have changed likelihood evaluations. Anything
    that survives this check but still degenerates into NaN is caught at the
    solver boundary in `fnnls.py`, where the failure is unambiguous.
    """
    if not schur > 0.0:
        raise np.linalg.LinAlgError(
            f"Cholesky insertion at index {index} is not positive definite: "
            f"Schur complement is {schur:.6e} (diagonal {diagonal:.6e}). The "
            f"matrix is singular to working precision, which for an inversion "
            f"means (near-)degenerate mesh vertices."
        )

    return math.sqrt(schur)


def cholinsert(U, index, x):
    from scipy import linalg

    S = np.insert(np.insert(U, index, 0, axis=0), index, 0, axis=1)

    S[:index, index] = S12 = linalg.solve_triangular(
        U[:index, :index], x[:index], trans=1, lower=False, overwrite_b=True
    )

    S[index, index] = s22 = _pivot_from_schur(
        schur=x[index] - S12.dot(S12), diagonal=x[index], index=index
    )

    if index == U.shape[0]:
        return S
    else:
        S[index, index + 1 :] = S23 = (x[index + 1 :] - S12.T @ U[:index, index:]) / s22
        _choldowndate(S[index + 1 :, index + 1 :], S23)  # S33
        return S


def cholinsertlast(U, x):
    """
    Update the Cholesky matrix U by inserting a vector at the end of the matrix
    Inserting a vector to the end of U doesn't require _cholupdate, so save some time.
    It's a special case of `cholinsert` (as shown above, if index == U.shape[0])
    As in current Cholesky scheme implemented in fnnls, we only use this kind of insertion, so I
        separate it out from the `cholinsert`.
    """
    from scipy import linalg

    index = U.shape[0]

    S = np.insert(np.insert(U, index, 0, axis=0), index, 0, axis=1)

    S[:index, index] = S12 = linalg.solve_triangular(
        U[:index, :index], x[:index], trans=1, lower=False, overwrite_b=True
    )

    S[index, index] = s22 = _pivot_from_schur(
        schur=x[index] - S12.dot(S12), diagonal=x[index], index=index
    )

    return S


def choldeleteindexes(U, indexes):
    indexes = sorted(indexes, reverse=True)

    for index in indexes:
        L = np.delete(np.delete(U, index, axis=0), index, axis=1)

        # If the deleted index is at the end of matrix, then we do not need to update the U.

        if index == L.shape[0]:
            U = L
        else:
            _cholupdate(L[index:, index:], U[index, index + 1 :])
            U = L

    return U
