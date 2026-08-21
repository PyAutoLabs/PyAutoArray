"""
The in-place Cholesky update path (`cholinsertlast_inplace` /
`choldeleteindexes_inplace` + the preallocated buffer in `fnnls_cholesky`)
must reproduce the out-of-place `cholinsertlast` / `choldeleteindexes`
results to the last few ulp, and the maintained factor must stay an exact
Cholesky factor of the active submatrix.

Exact bitwise agreement between the two implementations is NOT required (and
does not hold): LAPACK picks different, equally valid dtrtrs/dpotrs
invocations depending on the input's memory layout (F-contiguous vs
C-contiguous vs strided view), producing last-ulp differences — the
out-of-place implementation already mixes layouts between its own iterations
(`scipy.linalg.cholesky` returns F-order, `np.insert`/`np.delete` return
C-order). The production tolerance for this solver's output is the profiling
pins' rtol=1e-6; the cross-implementation tolerance here is far tighter.
"""

import numpy as np
import pytest
from scipy import linalg as slg
from scipy.optimize import nnls

from autoarray.util.cholesky_funcs import (
    cholinsertlast,
    cholinsertlast_inplace,
    choldeleteindexes,
    choldeleteindexes_inplace,
)
from autoarray.util.fnnls import fnnls_cholesky


def _random_spd(n, seed):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(2 * n, n))
    return Z.T @ Z + n * np.eye(n)


def _buffer_from(U, n_max):
    buffer = np.zeros((n_max, n_max))
    k = U.shape[0]
    buffer[:k, :k] = U
    return buffer, k


def _assert_factors_match(U_reference, U_inplace):
    """The two factors agree to within a few ulp (see module docstring)."""
    np.testing.assert_allclose(
        np.triu(U_inplace), np.triu(U_reference), rtol=1e-13, atol=1e-13
    )


def _assert_is_cholesky_of(U_view, A_sub):
    """The maintained upper triangle is an exact Cholesky factor of the
    active submatrix (the property every downstream cho_solve relies on)."""
    R = np.triu(U_view)
    np.testing.assert_allclose(R.T @ R, A_sub, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test__cholinsertlast_inplace__matches_out_of_place(seed):
    n = 12
    A = _random_spd(n, seed)

    k = 7
    U = slg.cholesky(A[:k, :k])
    x = A[k, : k + 1].copy()

    S = cholinsertlast(U.copy(), x.copy())

    buffer, k_active = _buffer_from(U, n)
    k_active = cholinsertlast_inplace(buffer, k_active, x.copy())

    assert k_active == k + 1
    _assert_factors_match(S, buffer[:k_active, :k_active])
    _assert_is_cholesky_of(buffer[:k_active, :k_active], A[: k + 1, : k + 1])


@pytest.mark.parametrize(
    "indexes",
    [[0], [7], [3, 5], [0, 1, 6], [2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7]],
)
def test__choldeleteindexes_inplace__matches_out_of_place(indexes):
    n = 12
    A = _random_spd(n, seed=3)

    k = 8
    U = slg.cholesky(A[:k, :k])

    L = choldeleteindexes(U.copy(), list(indexes))

    buffer, k_active = _buffer_from(U, n)
    k_active = choldeleteindexes_inplace(buffer, k_active, list(indexes))

    assert k_active == k - len(indexes)
    _assert_factors_match(L, buffer[:k_active, :k_active])

    keep = [i for i in range(k) if i not in indexes]
    _assert_is_cholesky_of(
        buffer[:k_active, :k_active], A[np.ix_(keep, keep)]
    )


def test__interleaved_inserts_and_deletes__match():
    n = 20
    A = _random_spd(n, seed=4)

    k = 4
    U = slg.cholesky(A[:k, :k])
    buffer, k_active = _buffer_from(U, n)

    # Mimic fnnls's usage: grow to the next leading size, shed some indexes,
    # grow again — comparing the two implementations after every operation.
    for op, arg in [
        ("insert", None),
        ("insert", None),
        ("delete", [1, 3]),
        ("insert", None),
        ("delete", [0]),
        ("insert", None),
        ("insert", None),
    ]:
        if op == "insert":
            k_old = U.shape[0]
            x = A[k_old, : k_old + 1].copy()
            U = cholinsertlast(U, x.copy())
            k_active = cholinsertlast_inplace(buffer, k_active, x.copy())
        else:
            U = choldeleteindexes(U, arg)
            k_active = choldeleteindexes_inplace(buffer, k_active, arg)

        assert k_active == U.shape[0]
        _assert_factors_match(U, buffer[:k_active, :k_active])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test__fnnls_cholesky__matches_scipy_nnls(seed):
    rng = np.random.default_rng(seed)
    n = 30
    Z = rng.normal(size=(50, n))
    # A mixed-sign target so a substantial subset of the solution is clamped
    # at zero and the delete path is exercised.
    x = Z @ rng.normal(size=n) + rng.normal(size=50)

    ZTZ = Z.T @ Z
    ZTx = Z.T @ x

    d = fnnls_cholesky(ZTZ, ZTx)
    d_ref, _ = nnls(Z, x)

    assert np.all(d >= 0.0)
    assert d == pytest.approx(d_ref, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test__fnnls_cholesky__warm_start_matches_cold_start(seed):
    rng = np.random.default_rng(seed)
    n = 30
    Z = rng.normal(size=(50, n))
    x = Z @ rng.normal(size=n) + rng.normal(size=50)

    ZTZ = Z.T @ Z
    ZTx = Z.T @ x

    d_cold = fnnls_cholesky(ZTZ, ZTx)

    P_initial = np.where(slg.solve(ZTZ.copy(), ZTx.copy(), assume_a="pos") > 0)[0]
    d_warm = fnnls_cholesky(ZTZ, ZTx, P_initial=P_initial)

    assert d_warm == pytest.approx(d_cold, rel=1e-8, abs=1e-10)


@pytest.mark.parametrize("seed", [0, 1])
def test__fnnls_cholesky__accepts_jax_arrays(seed):
    """
    The sparse-operator inversion path hands fnnls_cholesky JAX arrays even
    when the fit runs the numba CPU path. Indexing a JAX array yields another
    JAX array, which numba maps to a readonly buffer — before the boundary
    coercion in fnnls_cholesky this failed kernel compilation with
    "Cannot modify readonly array" (HowToLens smoke, 2026-08-20).
    """
    jnp = pytest.importorskip("jax.numpy")

    rng = np.random.default_rng(seed)
    n = 30
    Z = rng.normal(size=(50, n))
    x = Z @ rng.normal(size=n) + rng.normal(size=50)

    ZTZ = Z.T @ Z
    ZTx = Z.T @ x

    d_np = fnnls_cholesky(ZTZ, ZTx)
    d_jax = fnnls_cholesky(jnp.asarray(ZTZ), jnp.asarray(ZTx))

    assert np.all(np.asarray(d_jax) >= 0.0)
    assert np.asarray(d_jax) == pytest.approx(d_np, rel=1e-6, abs=1e-8)
