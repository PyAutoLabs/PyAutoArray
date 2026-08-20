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


@numba_util.jit()
def _solve_upper_transposed_buffer(Ubuf, k, b):
    """
    Solve ``U^T y = b`` for the active k x k upper factor ``Ubuf[:k, :k]``,
    overwriting ``b`` with ``y`` (row-oriented forward substitution, touching
    only contiguous row slices of the buffer's upper triangle).

    This replaces `scipy.linalg.solve_triangular(..., trans=1)` on the buffer
    view: scipy copies a non-contiguous view into a fresh array and scans it
    for non-finite values on every call, which at n ~ 1000 and ~150 calls per
    fnnls solve re-creates the very memory traffic the in-place buffer
    removes.
    """
    for i in range(k):
        yi = b[i] / Ubuf[i, i]
        b[i] = yi
        for j in range(i + 1, k):
            b[j] -= Ubuf[i, j] * yi
    return b


@numba_util.jit()
def _cho_solve_buffer(Ubuf, k, b):
    """
    Solve ``(U^T U) s = b`` for the active k x k upper factor
    ``Ubuf[:k, :k]``, overwriting ``b`` with ``s`` — LAPACK ``cho_solve`` for
    an upper factor, reading only the buffer's upper triangle and making no
    copies (see `_solve_upper_transposed_buffer` for why that matters).
    """
    b = _solve_upper_transposed_buffer(Ubuf, k, b)
    for i in range(k - 1, -1, -1):
        b[i] = (b[i] - np.dot(Ubuf[i, i + 1 : k], b[i + 1 : k])) / Ubuf[i, i]
    return b


def cholinsertlast_inplace(Ubuf, k, x):
    """
    In-place variant of `cholinsertlast` for a factor held in a preallocated
    buffer: the active k x k factor is `Ubuf[:k, :k]` and the new row/column is
    written directly into row/column `k` of the buffer.

    Same arithmetic as `cholinsertlast` (a transposed-triangular solve for
    the new column, the same `_pivot_from_schur` pivot, on the same values),
    but with a copy-free numba substitution in place of scipy's
    `solve_triangular` and without the two full O(k^2) `np.insert`
    reallocations per call — which dominate the
    positive-only inversion solve at n ~ 1000 (one such insertion per fnnls
    active-set iteration, ~150 iterations per likelihood evaluation).

    Only the upper triangle of the active region is maintained; entries below
    the diagonal are never written or read (every consumer — the solve and
    update kernels above — references the upper triangle only). ``x[:k]`` is
    overwritten by the solve.

    Returns the new active size ``k + 1``; the factor is ``Ubuf[:k+1, :k+1]``.
    """
    S12 = _solve_upper_transposed_buffer(Ubuf, k, x[:k])

    Ubuf[:k, k] = S12

    Ubuf[k, k] = _pivot_from_schur(
        schur=x[k] - S12.dot(S12), diagonal=x[k], index=k
    )

    return k + 1


@numba_util.jit()
def _choldelete_shift_buffer(Ubuf, k, index):
    """
    Shift the active k x k factor's rows/columns to close the gap left by
    deleting row+column ``index``: the top-right block moves one column left,
    the trailing block moves one step up-left along the diagonal. Pure value
    movement (bitwise), touching only the upper triangle — the numba loops
    write destinations strictly behind their sources, so no temporary is
    needed (numpy's overlapping slice assignment buffers the source instead,
    which at ~100 deletes per fnnls solve is real allocation traffic).
    """
    for i in range(index):
        for j in range(index, k - 1):
            Ubuf[i, j] = Ubuf[i, j + 1]
    for i in range(index, k - 1):
        for j in range(i, k - 1):
            Ubuf[i, j] = Ubuf[i + 1, j + 1]


def choldeleteindexes_inplace(Ubuf, k, indexes):
    """
    In-place variant of `choldeleteindexes` for a factor held in a
    preallocated buffer: remove the given positions from the active k x k
    factor `Ubuf[:k, :k]` by shifting the surviving rows/columns within the
    buffer (`_choldelete_shift_buffer`), then re-triangularize the trailing
    block with the same numba Givens kernel (`_cholupdate`) on the same
    values — no `np.delete` reallocations (two full-factor copies per
    deleted index). Only the upper triangle is maintained, as in
    `cholinsertlast_inplace`.

    Returns the new active size; the factor is ``Ubuf[:k', :k']``.
    """
    for index in sorted(indexes, reverse=True):
        # The deleted row's tail is the rank-1 update vector for the trailing
        # block — copied out before the shifts overwrite it.
        x = Ubuf[index, index + 1 : k].copy()

        _choldelete_shift_buffer(Ubuf, k, index)

        k -= 1

        # If the deleted index was at the end, the factor needs no update.

        if index < k:
            _cholupdate(Ubuf[index:k, index:k], x)

    return k
