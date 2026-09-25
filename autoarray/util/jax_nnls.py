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

The knobs are exposed per-fit through the ``Settings`` class
(``nnls_solver_tol`` / ``nnls_max_iter``, defaults ``None`` = upstream
behaviour) and read by ``inversion_util.reconstruction_positive_only_from``.
Measured motivation (PyAutoArray#369, real HST pixelization+MGE systems):
each PDIP iteration is a fresh dense Cholesky of the (n, n) KKT system, so
iterations are the whole cost; ``solver_tol=1e-6`` saves ~15-20% of solve
time with a log-evidence shift of order 1e-8. Under ``vmap`` the while_loop
runs until the slowest lane converges, so ``max_iter`` also caps the
worst-case batched cost.

Convergence is observable (PyAutoArray#571): :func:`solve_nnls_primal_with_status`
returns the PDIP ``converged`` flag and iteration count next to ``x`` (the
plain :func:`solve_nnls_primal` keeps its drop-in signature). The ``"raw"``
positive-only mode (:func:`solve_nnls_primal_raw_forward`) runs the forward
solve on the un-preconditioned system with a data-scaled tolerance
(:func:`data_scaled_solver_tol`) and keeps the Jacobi-space backward pass. It
exists because Jacobi scaling of signal-free MGE columns (diagonal = the
no-regularization floor) makes the PDIP dual diverge. Its backward pass polishes
the mapped forward iterate with a few tight, warm-started PDIP iterations on the
Jacobi system before the relaxed-KKT solve, which otherwise diverges to NaN from
the loose forward tolerance (PyAutoArray#573); :func:`raw_forward_backward_status`
reports that pass's convergence.

JAX is imported inside functions, never at module level (see
``docs/agents/jax_and_decorators.md``); this module must only be imported
on the ``xp=jnp`` path. The solver knobs are static closure parameters —
the ``lru_cache`` returns the same function object for repeated settings so
``jax.jit`` tracing caches hit (a fresh closure per call would cache-bust).
"""

from functools import lru_cache


def solve_nnls(Q, q, solver_tol=None, max_iter=50, init=None):
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
    init
        Optional ``(x, s, z)`` warm start (strictly positive ``s`` and ``z``)
        replacing jaxnnls's ``initialize``. ``None`` (default) is the upstream
        cold start.

    Returns
    -------
    The tuple (x, s, z, converged, pdip_iter) of the primal solution, slack
    and dual variables, convergence flag and iteration count.
    """
    import jax
    import jax.numpy as jnp
    from jaxnnls.pdip import EPSILON, initialize, pdip_pc_step

    x, s, z = initialize(Q, q) if init is None else init

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


# The data-scaled tolerance of the "raw" (un-preconditioned) mode is this fraction of jaxnnls's
# own ``n * EPSILON`` rule, multiplied by ``max(1, max|q|)``. Measured on the SLaM MGE fixture
# (PyAutoArray#571): factor 1 stops ~1e-11 (relative objective) short of fnnls, 1e-2 reaches
# <= 4e-13 for 1-2 extra iterations (16-19 in total), 1e-3 / 1e-4 buy one more digit per iteration.
DATA_SCALED_TOL_FACTOR = 1.0e-2


def data_scaled_solver_tol(q):
    """
    The convergence tolerance of the ``"raw"`` positive-only mode:
    ``DATA_SCALED_TOL_FACTOR * n * EPSILON * max(1, max|q|)``.

    jaxnnls's own rule ``n * EPSILON`` is absolute, so on an unscaled system
    (curvature entries ~1e7) it is unreachable in floating point; scaling it by
    the data vector makes it a relative KKT tolerance. ``q`` may be traced.
    """
    import jax.numpy as jnp
    from jaxnnls.pdip import EPSILON

    return (
        DATA_SCALED_TOL_FACTOR
        * q.shape[0]
        * EPSILON
        * jnp.maximum(1.0, jnp.max(jnp.abs(q)))
    )


@lru_cache(maxsize=None)
def _solve_nnls_primal_with(target_kappa, solver_tol, max_iter):
    """
    Build (and cache) the differentiable primal solver for one static
    setting of the knobs. The returned function takes only (Q, q) and returns
    ``(x, converged, pdip_iter)``; the custom-vjp backward pass returns
    exactly (dQ, dq) from the cotangent of ``x`` (the integer status outputs
    carry no cotangent).
    """
    import jax
    from jaxnnls.diff_qp import diff_nnls
    from jaxnnls.pdip_relaxed import solve_relaxed_nnls

    def primal(Q, q):
        x, _, _, converged, pdip_iter = solve_nnls(
            Q, q, solver_tol=solver_tol, max_iter=max_iter
        )
        return x, converged, pdip_iter

    def forward(Q, q):
        x, s, z, converged, pdip_iter = solve_nnls(
            Q, q, solver_tol=solver_tol, max_iter=max_iter
        )
        # Relax the solution with vanilla Newton steps on the relaxed KKT
        # conditions; only the backward pass consumes the relaxed variables.
        xr, sr, zr, _, _ = solve_relaxed_nnls(Q, q, x, s, z, target_kappa=target_kappa)
        return (x, converged, pdip_iter), (Q, xr, sr, zr)

    def backward(res, output_grad):
        Q, xr, sr, zr = res
        return diff_nnls(Q, xr, sr, zr, output_grad[0])

    primal = jax.custom_vjp(primal)
    primal.defvjp(forward, backward)
    return primal


def solve_nnls_primal_with_status(
    Q, q, target_kappa=1e-3, solver_tol=None, max_iter=50
):
    """
    As :func:`solve_nnls_primal`, but also returns the PDIP convergence flag
    and iteration count: ``(x, converged, pdip_iter)``.

    ``x`` (value and gradient) is identical to :func:`solve_nnls_primal`;
    ``converged`` (``1`` if the KKT residual met the tolerance within
    ``max_iter``) and ``pdip_iter`` are non-differentiable integer outputs, so
    they are safe to return from ``jax.jit`` / ``vmap``-ed code.
    """
    return _solve_nnls_primal_with(target_kappa, solver_tol, max_iter)(Q, q)


def solve_nnls_primal(Q, q, target_kappa=1e-3, solver_tol=None, max_iter=50):
    """
    Solve the non-negative least squares problem, differentiable via the
    relaxed-KKT implicit backward pass.

    Drop-in replacement for ``jaxnnls.solve_nnls_primal`` with two extra
    knobs; at their defaults (``solver_tol=None``, ``max_iter=50``) the
    forward solve and gradients are identical to upstream. Use
    :func:`solve_nnls_primal_with_status` to also get the convergence flag.
    """
    return solve_nnls_primal_with_status(
        Q, q, target_kappa=target_kappa, solver_tol=solver_tol, max_iter=max_iter
    )[0]


# The backward pass of the ``"raw"`` mode first polishes the mapped raw-forward iterate with at most this many
# PDIP iterations on the Jacobi-scaled system at jaxnnls's own tight tolerance (PyAutoArray#573). Measured on the
# SLaM MGE fixture, the 48 SLaM ``source_lp[1]`` systems and the jax_grad/mge.py points: 4-6 iterations.
RAW_BACKWARD_POLISH_MAX_ITER = 10


def _raw_forward_backward_point(
    Q_pc, q_pc, Q, q, D, target_kappa, solver_tol, max_iter
):
    """
    The forward solve and the relaxed-KKT point of the ``"raw"`` mode (shared by
    the custom-vjp forward pass and :func:`raw_forward_backward_status`).

    Returns ``(y, converged, pdip_iter)`` of the raw forward solve (mapped to the
    Jacobi coordinates), the relaxed point ``(yr, sr, zr)`` the backward pass
    differentiates at, and the status ``(relaxed_converged, relaxed_iter,
    polish_converged, polish_iter)``.
    """
    import jax.numpy as jnp
    from jaxnnls.pdip_relaxed import solve_relaxed_nnls

    tol = data_scaled_solver_tol(q) if solver_tol is None else solver_tol
    x, s, z, converged, pdip_iter = solve_nnls(Q, q, solver_tol=tol, max_iter=max_iter)
    y, sy, zy = x / D, s / D, z * D

    # Polish (PyAutoArray#573): the data-scaled tolerance leaves s * z ~ 1e-10 .. 1e-9, far above
    # ``target_kappa``, so the relaxed solve below would have to push toward the boundary from z / s ~ 1e13
    # and its fixed 50-iteration while_loop overshoots to NaN. A few tight PDIP iterations on the scaled
    # system, warm-started from the mapped iterate, bring s * z down to the jaxnnls tolerance first. If the
    # polish does not converge (the scaled dual is what diverges on #571's systems from a cold start), the
    # mapped iterate is kept, i.e. the pre-polish behaviour.
    yp, sp, zp, polish_converged, polish_iter = solve_nnls(
        Q_pc, q_pc, max_iter=RAW_BACKWARD_POLISH_MAX_ITER, init=(y, sy, zy)
    )
    ok = jnp.logical_and(
        polish_converged == 1,
        jnp.all(jnp.isfinite(yp)) & jnp.all(sp > 0) & jnp.all(zp > 0),
    )
    yp, sp, zp = (jnp.where(ok, a, b) for a, b in ((yp, y), (sp, sy), (zp, zy)))

    yr, sr, zr, relaxed_converged, relaxed_iter = solve_relaxed_nnls(
        Q_pc, q_pc, yp, sp, zp, target_kappa=target_kappa
    )
    status = (relaxed_converged, relaxed_iter, ok.astype(int), polish_iter)
    return (y, converged, pdip_iter), (yr, sr, zr), status


@lru_cache(maxsize=None)
def _solve_nnls_raw_forward_with(target_kappa, solver_tol, max_iter):
    """
    Build (and cache) the ``"raw"``-mode solver (PyAutoArray#571, #573).

    The returned function takes the Jacobi-scaled system ``(Q_pc, q_pc)``
    (``Q_pc = D Q D``, ``q_pc = D q``) together with the raw system ``(Q, q)``
    and ``D``, and returns ``(y, converged, pdip_iter)`` with ``y`` the solution
    of the scaled system, so ``x = D * y``.

    - **Forward:** the PDIP solve runs on the *raw* ``(Q, q)`` with a
      data-scaled tolerance (:func:`data_scaled_solver_tol`, unless
      ``solver_tol`` is given), and its iterate is mapped to the scaled
      coordinates (``y = x / D``, slack ``s / D``, dual ``z * D``). On
      linear-object-only (MGE) systems, Jacobi scaling turns signal-free columns
      whose diagonal is only the no-regularization floor into degenerate
      coordinates that make the PDIP dual diverge; the raw solve does not.
    - **Backward:** the relaxed-KKT implicit derivative on ``Q_pc`` (as the
      Jacobi mode), started from the mapped iterate after a *polish*: at most
      :data:`RAW_BACKWARD_POLISH_MAX_ITER` PDIP iterations on ``(Q_pc, q_pc)``
      at jaxnnls's tight tolerance, warm-started from the mapped iterate
      (kept only if it converges). Without it the loose forward tolerance
      leaves complementarity ``s * z`` orders of magnitude above
      ``target_kappa`` and the relaxed solve diverges to NaN on a fraction of
      points (PyAutoArray#573); with it the relaxed solve converges in about
      one iteration. The primal ``y`` is the unpolished forward solution, so
      the forward value is unchanged. The relaxed-KKT pass on the raw,
      ill-conditioned ``Q`` produces NaN gradients, which is why Jacobi scaling
      was introduced. ``(Q, q, D)`` get zero cotangents: ``y`` depends only on
      ``(Q_pc, q_pc)``, and the caller's autodiff carries the dependence of
      those, and of ``D``, on the raw inputs.

    The backward-pass convergence is observable through
    :func:`raw_forward_backward_status`.
    """
    import jax
    import jax.numpy as jnp
    from jaxnnls.diff_qp import diff_nnls

    def primal(Q_pc, q_pc, Q, q, D):
        tol = data_scaled_solver_tol(q) if solver_tol is None else solver_tol
        x, _, _, converged, pdip_iter = solve_nnls(
            Q, q, solver_tol=tol, max_iter=max_iter
        )
        return x / D, converged, pdip_iter

    def forward(Q_pc, q_pc, Q, q, D):
        out, (yr, sr, zr), status = _raw_forward_backward_point(
            Q_pc, q_pc, Q, q, D, target_kappa, solver_tol, max_iter
        )
        return out, (Q_pc, yr, sr, zr, status[0], Q, q, D)

    def backward(res, output_grad):
        Q_pc, yr, sr, zr, _, Q, q, D = res
        dQ_pc, dq_pc = diff_nnls(Q_pc, yr, sr, zr, output_grad[0])
        return dQ_pc, dq_pc, jnp.zeros_like(Q), jnp.zeros_like(q), jnp.zeros_like(D)

    primal = jax.custom_vjp(primal)
    primal.defvjp(forward, backward)
    return primal


def raw_forward_backward_status(
    Q_pc, q_pc, Q, q, D, target_kappa=1e-3, solver_tol=None, max_iter=50
):
    """
    Diagnostic (not differentiable): the convergence of the ``"raw"`` mode's
    backward-pass preparation for one system, as the integer tuple
    ``(relaxed_converged, relaxed_iter, polish_converged, polish_iter)``.

    ``relaxed_*`` describe the relaxed-KKT solve whose point the gradient is
    taken at; ``polish_*`` the tight warm-started PDIP polish before it
    (``polish_converged == 0`` means the mapped iterate was used unpolished).
    Arguments are those of :func:`solve_nnls_primal_raw_forward`.
    """
    return _raw_forward_backward_point(
        Q_pc, q_pc, Q, q, D, target_kappa, solver_tol, max_iter
    )[2]


def solve_nnls_primal_raw_forward(
    Q_pc, q_pc, Q, q, D, target_kappa=1e-3, solver_tol=None, max_iter=50
):
    """
    The ``"raw"`` positive-only mode: forward PDIP on the raw system with a
    data-scaled tolerance, backward pass on the Jacobi-scaled system. Returns
    ``(y, converged, pdip_iter)`` with ``x = D * y``; see
    :func:`_solve_nnls_raw_forward_with`.
    """
    return _solve_nnls_raw_forward_with(target_kappa, solver_tol, max_iter)(
        Q_pc, q_pc, Q, q, D
    )
