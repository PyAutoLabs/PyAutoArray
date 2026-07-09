"""
Configurable driver for the jaxnnls primal-dual interior-point NNLS solver.

jaxnnls hard-codes its convergence tolerance (``n * eps * 5e3``, capped at
1e-2) and iteration cap (``MAX_ITER = 50``) inside ``pdip.solve_nnls``, and
neither is exposed through ``solve_nnls_primal``. This module re-implements
only that ``while_loop`` driver with both knobs as arguments, reusing every
jaxnnls building block (``initialize``, ``pdip_pc_step``,
``solve_relaxed_nnls``, ``diff_nnls``) and the same custom-vjp relaxed-KKT
backward pass. With the knobs at their defaults the solve and its gradients
are identical to upstream jaxnnls.

The knobs are read from ``general.yaml -> inversion -> nnls_solver_tol /
nnls_max_iter`` by ``inversion_util.reconstruction_positive_only_from``.
Measured motivation (PyAutoArray#369, real HST pixelization+MGE systems):
each PDIP iteration is a fresh dense Cholesky of the (n, n) KKT system, so
iterations are the whole cost; ``solver_tol=1e-6`` saves ~15-20% of solve
time with a log-evidence shift of order 1e-8. Under ``vmap`` the while_loop
runs until the slowest lane converges, so ``max_iter`` also caps the
worst-case batched cost.

JAX is imported inside functions, never at module level (see
``docs/agents/jax_and_decorators.md``); this module must only be imported
on the ``xp=jnp`` path. The solver knobs are static closure parameters —
the ``lru_cache`` returns the same function object for repeated settings so
``jax.jit`` tracing caches hit (a fresh closure per call would cache-bust).
"""

from functools import lru_cache


def solve_nnls(Q, q, solver_tol=None, max_iter=50):
    """
    Solve the non-negative least squares problem with the jaxnnls PDIP
    algorithm, with configurable convergence tolerance and iteration cap.

    Mirrors ``jaxnnls.pdip.solve_nnls`` exactly at the default settings.

    Parameters
    ----------
    Q
        The (n, n) positive definite matrix (the curvature-regularization
        matrix of an inversion).
    q
        The (n,) vector (the data vector of an inversion).
    solver_tol
        Infinity-norm KKT residual below which the solve is converged.
        ``None`` (default) reproduces jaxnnls's own tolerance
        ``min(n * eps * 5e3, 1e-2)``.
    max_iter
        Maximum number of PDIP iterations (jaxnnls hard-codes 50).

    Returns
    -------
    The tuple (x, s, z, converged, pdip_iter) of the primal solution, slack
    and dual variables, convergence flag and iteration count.
    """
    import jax
    import jax.numpy as jnp
    from jaxnnls.pdip import EPSILON, initialize, pdip_pc_step

    x, s, z = initialize(Q, q)

    if solver_tol is None:
        solver_tol = jax.lax.min(Q.shape[0] * EPSILON, 1e-2)
    solver_tol = jnp.asarray(solver_tol, dtype=q.dtype)

    def converged_check(inputs):
        _, _, _, _, _, _, converged, pdip_iter = inputs
        return jnp.logical_and(pdip_iter < max_iter, converged == 0)

    init_inputs = (Q, q, x, s, z, solver_tol, 0, 0)
    outputs = jax.lax.while_loop(converged_check, pdip_pc_step, init_inputs)
    _, _, x, s, z, _, converged, pdip_iter = outputs
    return x, s, z, converged, pdip_iter


@lru_cache(maxsize=None)
def _solve_nnls_primal_with(target_kappa, solver_tol, max_iter):
    """
    Build (and cache) the differentiable primal solver for one static
    setting of the knobs. The returned function takes only (Q, q), so the
    custom-vjp backward pass returns exactly (dQ, dq).
    """
    import jax
    from jaxnnls.diff_qp import diff_nnls
    from jaxnnls.pdip_relaxed import solve_relaxed_nnls

    def primal(Q, q):
        return solve_nnls(Q, q, solver_tol=solver_tol, max_iter=max_iter)[0]

    def forward(Q, q):
        x, s, z, _, _ = solve_nnls(Q, q, solver_tol=solver_tol, max_iter=max_iter)
        # Relax the solution with vanilla Newton steps on the relaxed KKT
        # conditions; only the backward pass consumes the relaxed variables.
        xr, sr, zr, _, _ = solve_relaxed_nnls(
            Q, q, x, s, z, target_kappa=target_kappa
        )
        return x, (Q, xr, sr, zr)

    def backward(res, input_grad):
        Q, xr, sr, zr = res
        return diff_nnls(Q, xr, sr, zr, input_grad)

    primal = jax.custom_vjp(primal)
    primal.defvjp(forward, backward)
    return primal


def solve_nnls_primal(Q, q, target_kappa=1e-3, solver_tol=None, max_iter=50):
    """
    Solve the non-negative least squares problem, differentiable via the
    relaxed-KKT implicit backward pass.

    Drop-in replacement for ``jaxnnls.solve_nnls_primal`` with two extra
    knobs; at their defaults (``solver_tol=None``, ``max_iter=50``) the
    forward solve and gradients are identical to upstream.
    """
    return _solve_nnls_primal_with(target_kappa, solver_tol, max_iter)(Q, q)
