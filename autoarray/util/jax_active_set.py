"""
Certified active-set positive-only (NNLS) solver for the JAX backend.

Solves the non-negative quadratic program

    minimise  1/2 x^T Q x - q^T x    subject to  x >= 0

for a symmetric positive-definite ``Q`` (an inversion's
``curvature_reg_matrix``, usually Jacobi-scaled to a unit diagonal) and a
vector ``q`` (its ``data_vector``). It is an alternative to the primal-dual
interior-point (PDIP) solve in :mod:`autoarray.util.jax_nnls`, selected by
``Settings.positive_only_solver = "certified"``.

Algorithm
---------
A "free-all" active-set scheme on a boolean *fixed set* ``Z`` (the indices
held at exactly zero):

1. **Pass 0** — the unconstrained solve ``x0 = Q^-1 q`` (with the permanent
   set held at zero). If it has no negative entry it is already the answer.
2. **Seed** ``Z = permanent | (x0 < 0)``.
3. **Each pass** solves the restricted system on the free set ``F = ~Z`` and
   certifies the iterate (see below). If it certifies, the loop stops; if not,
   every free index with a violating negative value is added to ``Z`` and every
   fixed (non-permanent) index with a violating negative gradient is released,
   ``Z <- (Z | primal_violations) & ~dual_violations``, and the next pass runs.

Each pass is a **masked, full-size** Cholesky solve (:func:`masked_solve`):
``Q`` with ``Z``'s rows/columns replaced by the identity and ``q`` zeroed on
``Z``, so the solve returns ``x_F = Q[F, F]^-1 q_F`` and ``x_Z = 0`` exactly.
That is costlier per pass than factorising the free block alone, but it is the
only form with a static shape — the free set changes every pass, and an index
list would retrace under ``jax.jit`` where a boolean mask does not.

Certification
-------------
An iterate ``x`` with gradient ``g = Q x - q`` is certified optimal when the
KKT conditions hold to a relative tolerance ``tau_rel``:

- **primal**: no free entry is negative beyond ``tau_x = tau_rel * max|x|``;
- **dual**: no fixed, releasable entry has a gradient below
  ``-tau_g = -tau_rel * max|q|`` (such an index would lower the objective if it
  were allowed to become positive).

Stationarity on the free set holds by construction (the free block is solved
exactly). Permanent indices (e.g. the library's edge-zeroed pixels) are seeded
into ``Z`` and never released, so the certificate is a certificate for the
restricted problem the caller asked for.

Budgets
-------
The search is a ``lax.while_loop`` that exits as soon as an iterate
certifies, so an easy system pays only the passes it needs. ``pass_budget``
caps the worst case. Measured passes to certification on source-only HST
inversions (autolens_profiling fixed-lens-light campaign, PyAutoArray#566):
rectangular <= 11, Delaunay <= 7, and a fiducial Euclid rectangular system
reaches 11. The default budget of 16 covers every measured draw with margin;
because the loop exits early the unused budget costs nothing. Inversions that
include dense linear light-profile / MGE coefficients converge far worse
(60-MGE + Delaunay-1500 failed to certify in 40 passes) and are therefore
never dispatched here — see ``AbstractInversion.positive_only_solver_used``.

Fallback
--------
:func:`solve_certified_with_fallback` returns the certified iterate when the
search certified within budget and otherwise the result of ``pdip_fn`` (the
library's PDIP solve), via ``lax.cond``. With ``fallback=False`` the last
iterate is returned with ``certified=False`` flagged — it is feasible on the
fixed set but not proven optimal.

**vmap caveat:** under ``jax.vmap`` a ``lax.cond`` with a batched predicate
is lowered to a ``select`` that executes *both* branches for every lane, so
the fallback PDIP solve runs for the whole batch even when every lane
certified (measured 44.6 vs 31.1 ms/lane at B=16 on an A100). Likewise the
``while_loop`` runs until the slowest lane certifies. The batched policy
(fallback ``"pdip"`` vs ``"none"`` under ``jit(vmap)``) and the production
default are phase B of the ``certified-positive-solver`` epic
(``PyAutoMind/draft/research/autolens_profiling/certified_solver_production_default.md``);
until then this solver is opt-in.

Gradient contract
-----------------
``lax.while_loop`` is not reverse-mode differentiable, and the active set is a
piecewise-constant function of ``(Q, q)`` anyway. The search therefore runs on
``jax.lax.stop_gradient`` copies of its inputs and returns only the boolean
fixed set. The returned solution is then recomputed by one final
:func:`masked_solve` on that set, **outside** the ``stop_gradient``, so
autodiff through that single Cholesky solve yields the exact implicit
derivative of the NNLS solution on its (locally constant) active set:
``dx_F = Q[F,F]^-1 (dq_F - dQ[F,:] x)`` and ``dx_Z = 0``. This is the true
derivative wherever the active set is locally stable (strict complementarity).
The PDIP path's ``custom_vjp`` instead differentiates a relaxed central-path
KKT system (``target_kappa``) and is an approximation to the same quantity, so
the two gradients agree closely but not bit-for-bit. The final solve costs one
extra Cholesky factorisation over the search.

Precision
---------
The solve runs in float64 (the inputs are promoted), consistent with the
PDIP path: active-set and interior-point solvers are sensitive to fp32 noise
on ill-conditioned source meshes, so ``use_mixed_precision`` never reaches the
NNLS.

Why the NumPy path is unchanged
-------------------------------
The NumPy backend keeps ``fnnls_cholesky`` with its cross-evaluation
warm-start memo: a NumPy port of this scheme measured 3-7 % slower than
fnnls, and factor reuse lost to the memo (PyAutoArray#566).

JAX is imported inside functions, never at module level (see
``docs/agents/jax_and_decorators.md``); this module must only be called on the
``xp=jnp`` path.
"""


def _as_float64(Q, q):
    import jax.numpy as jnp

    dtype = jnp.result_type(Q.dtype, q.dtype, jnp.float64)
    return jnp.asarray(Q, dtype=dtype), jnp.asarray(q, dtype=dtype)


def masked_solve(Q, q, fixed):
    """
    Solve ``Q x = q`` on the free set ``~fixed`` with ``x`` exactly zero on
    ``fixed``, at full static size.

    ``Q``'s fixed rows and columns are replaced by the identity and ``q`` is
    zeroed on ``fixed``, so the block-diagonal system returns
    ``x_F = Q[F, F]^-1 q_F`` and ``x_Z = 0`` exactly in one ``(n, n)``
    Cholesky factorisation.

    Parameters
    ----------
    Q
        The ``(n, n)`` symmetric positive-definite matrix.
    q
        The ``(n,)`` right-hand side.
    fixed
        Boolean ``(n,)`` mask of the indices held at zero.

    Returns
    -------
    The ``(n,)`` restricted solution.
    """
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve

    keep = ~fixed
    Q_masked = jnp.where(keep[:, None] & keep[None, :], Q, 0.0)
    Q_masked = Q_masked + jnp.diag(jnp.where(fixed, 1.0, 0.0).astype(Q.dtype))
    q_masked = jnp.where(fixed, 0.0, q)
    L = jnp.linalg.cholesky(Q_masked)
    return cho_solve((L, True), q_masked)


def certify(Q, q, x, fixed, permanent, tau_rel):
    """
    Check the KKT conditions of an active-set iterate.

    Parameters
    ----------
    Q, q
        The quadratic program.
    x
        The iterate, the :func:`masked_solve` of ``fixed``.
    fixed
        Boolean mask of the indices ``x`` holds at zero.
    permanent
        Boolean mask of fixed indices that may never be released.
    tau_rel
        Relative tolerance: primal violations are ``x < -tau_rel * max|x|`` on
        the free set, dual violations are ``g < -tau_rel * max|q|`` on the
        releasable fixed set, with ``g = Q x - q``.

    Returns
    -------
    ``(certified, primal_violations, dual_violations)``: a scalar boolean and
    two boolean ``(n,)`` masks.
    """
    import jax.numpy as jnp

    g = Q @ x - q
    tau_x = tau_rel * jnp.max(jnp.abs(x))
    tau_g = tau_rel * jnp.max(jnp.abs(q))

    free = ~fixed
    freeable = fixed & ~permanent
    primal_violations = free & (x < -tau_x)
    dual_violations = freeable & (g < -tau_g)
    certified = ~(jnp.any(primal_violations) | jnp.any(dual_violations))

    return certified, primal_violations, dual_violations


def active_set_search(Q, q, permanent=None, pass_budget=16, tau_rel=1.0e-9):
    """
    Find the certified fixed (active) set of the NNLS problem.

    Runs on ``stop_gradient`` copies of ``Q`` and ``q`` — no reverse-mode path
    goes through the ``while_loop``; see the module docstring's gradient
    contract.

    Parameters
    ----------
    Q, q
        The quadratic program (``Q`` symmetric positive-definite).
    permanent
        Optional boolean ``(n,)`` mask of indices held at zero and never
        released. ``None`` solves the pure problem.
    pass_budget
        Maximum number of restricted passes after pass 0 (static Python int).
    tau_rel
        Relative certification tolerance (see :func:`certify`).

    Returns
    -------
    ``(fixed, certified, passes)``: the final boolean fixed set (the certified
    one if ``certified``, otherwise the set the next pass would have solved),
    the certification flag, and the number of restricted passes run (``0``
    when the unconstrained solve was already non-negative).
    """
    import jax
    import jax.numpy as jnp

    Q, q = _as_float64(Q, q)
    Q = jax.lax.stop_gradient(Q)
    q = jax.lax.stop_gradient(q)

    n = q.shape[-1]

    if permanent is None:
        permanent = jnp.zeros(n, dtype=bool)
    else:
        permanent = jnp.asarray(permanent, dtype=bool)

    x0 = masked_solve(Q, q, permanent)
    negative0 = x0 < 0.0
    fixed0 = permanent | negative0

    # With no negative entry the pass-0 solve is the restricted solve of
    # `fixed0 == permanent`, it has no releasable fixed index and no primal
    # violation, so it is already certified and no restricted pass is needed.
    certified0 = ~jnp.any(negative0)

    def cond_fun(carry):
        _, certified, passes = carry
        return (~certified) & (passes < pass_budget)

    def body_fun(carry):
        fixed, _, passes = carry
        x = masked_solve(Q, q, fixed)
        certified, primal_violations, dual_violations = certify(
            Q, q, x, fixed, permanent, tau_rel
        )
        fixed_new = (fixed | primal_violations) & ~dual_violations
        fixed_next = jnp.where(certified, fixed, fixed_new)
        return fixed_next, certified, passes + 1

    init = (fixed0, certified0, jnp.asarray(0, dtype=jnp.int32))

    return jax.lax.while_loop(cond_fun, body_fun, init)


def solve_certified(Q, q, permanent=None, pass_budget=16, tau_rel=1.0e-9):
    """
    Solve the NNLS problem with the certified active-set scheme.

    The active set is found by :func:`active_set_search` (under
    ``stop_gradient``); the returned solution is one final differentiable
    :func:`masked_solve` on that set, so ``jax.grad`` yields the exact implicit
    active-set derivative. ``x`` is exactly zero on the fixed set.

    Parameters
    ----------
    Q, q
        The quadratic program (``Q`` symmetric positive-definite).
    permanent
        Optional boolean mask of indices held at zero and never released.
    pass_budget
        Maximum number of restricted passes after pass 0 (static Python int).
    tau_rel
        Relative certification tolerance.

    Returns
    -------
    ``(x, certified, passes)``. When ``certified`` is ``False`` the budget was
    exhausted and ``x`` is the (feasible on the fixed set, but unproven)
    solve of the last fixed set.
    """
    Q, q = _as_float64(Q, q)

    fixed, certified, passes = active_set_search(
        Q, q, permanent=permanent, pass_budget=pass_budget, tau_rel=tau_rel
    )

    x = masked_solve(Q, q, fixed)

    return x, certified, passes


def solve_certified_with_fallback(
    Q,
    q,
    pdip_fn,
    fallback=True,
    pass_budget=16,
    tau_rel=1.0e-9,
    permanent=None,
):
    """
    Certified active-set solve with a fallback for an exhausted budget.

    Parameters
    ----------
    Q, q
        The quadratic program (``Q`` symmetric positive-definite).
    pdip_fn
        A zero-argument callable returning the fallback solution of the same
        problem (the library's PDIP solve), with the same shape and dtype as
        ``q``.
    fallback
        If ``True``, an uncertified iterate is discarded and ``pdip_fn()`` is
        returned instead via ``lax.cond``. Under ``vmap`` that ``cond``
        executes both branches for every lane (see the module docstring). If
        ``False``, the last iterate is returned with ``certified=False``.
    pass_budget
        Maximum number of restricted passes after pass 0 (static Python int).
    tau_rel
        Relative certification tolerance.
    permanent
        Optional boolean mask of indices held at zero and never released.

    Returns
    -------
    ``(x, certified, passes)`` — ``certified`` is the search's flag, so with
    ``fallback=True`` a ``False`` means ``x`` came from ``pdip_fn``.
    """
    import jax

    x, certified, passes = solve_certified(
        Q, q, permanent=permanent, pass_budget=pass_budget, tau_rel=tau_rel
    )

    if fallback:
        x = jax.lax.cond(certified, lambda: x, pdip_fn)

    return x, certified, passes
