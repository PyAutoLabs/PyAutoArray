"""
Composable density-component framework for the adaptive rectangular CDF mesh.

The adaptive rectangular mesh in
``autoarray/inversion/mesh/interpolator/rectangular_spline.py`` drives a
per-axis CDF transform off a single ``mesh_weight_map`` of shape ``(N_base,)``.
The mesh class produces that weight map via ``mesh_weight_map_from(adapt_data,
xp)``.

This module generalises that single signal to a weighted sum of multiple
physically-motivated density components while preserving the exact
``(N_base,)`` shape and sum-to-one normalisation that downstream code expects.
The spline machinery requires no change; a multi-component mesh class would
only override ``mesh_weight_map_from`` to call ``compose_density``.

A density component is any callable matching::

    def component(traced_points, context, xp) -> Array  # shape (N_base,)

with an optional ``required_context_keys`` attribute (tuple of str) declaring
the ``context`` keys it needs. ``compose_density`` validates those keys exist
before evaluating so missing-context bugs surface before the JAX trace, not
deep inside it.

Phase-2 scope: framework only. Physical density bases (brightness,
magnification, residual, caustic-proximity) are deferred until the
ghost-peak experiment establishes whether separable-CDF is the binding
limitation — see ``ghost_peak_findings.md``.
"""
from typing import Callable, Mapping, Sequence

import numpy as np


_EPS = 1e-12


def compose_density(
    components: Sequence[Callable],
    weights,
    floor: float,
    traced_points,
    context: Mapping,
    xp=np,
):
    """Combine multiple density components into a single mesh weight map.

    Computes ``rho = floor + sum_k weights[k] * components[k](traced_points,
    context, xp)``, clips to a positive epsilon, and normalises to sum-to-one
    so the result is a drop-in replacement for ``mesh_weight_map`` in
    ``create_transforms_spline``.

    The floor is applied to the composite, not per-component — see
    ``cdf_audit.md`` §6.1 for the design rationale.

    Parameters
    ----------
    components
        Sequence of callables ``(traced_points, context, xp) -> (N,) Array``.
        Each may carry a ``required_context_keys`` attribute declaring which
        ``context`` keys it reads; missing keys raise ``KeyError`` before any
        component is evaluated.
    weights
        Array-like of length ``len(components)``. Per-component scalar
        multipliers. May be a Python list, a numpy array, or a JAX array
        (e.g. learnable adaptivity weights inside a likelihood).
    floor
        Scalar additive constant applied to the composite density before
        clipping. Ensures the empty-context / zero-signal regions still
        contribute non-vanishing pixel mass.
    traced_points
        Source-plane (y, x) coordinates the CDF will be built from. Shape
        ``(N_base, 2)``. Components may use these directly or ignore them.
    context
        Mapping of auxiliary fields each component may read (e.g.
        ``brightness``). Validated against component ``required_context_keys``
        before evaluation.
    xp
        Array library — ``numpy`` (default) or ``jax.numpy``.

    Returns
    -------
    Array
        Shape ``(N_base,)``, strictly positive, finite, sums to 1.
    """
    if len(components) != len(weights):
        raise ValueError(
            f"compose_density: got {len(components)} components but "
            f"{len(weights)} weights — must match"
        )

    for comp in components:
        required = getattr(comp, "required_context_keys", ())
        missing = [k for k in required if k not in context]
        if missing:
            raise KeyError(
                f"compose_density: component {comp!r} requires context keys "
                f"{missing!r}; context provides {list(context)!r}"
            )

    rho = xp.asarray(floor)
    for comp, w in zip(components, weights):
        rho = rho + w * comp(traced_points, context, xp)

    rho = xp.clip(rho, _EPS, None)
    return rho / xp.sum(rho)


def uniform_density_component(traced_points, context, xp=np):
    """Trivial component returning constant ``1.0`` per point.

    Useful as the first-iteration fallback before residual / brightness signals
    are available, or as a baseline to blend against more aggressive
    physically-motivated components.

    Carries ``required_context_keys = ()`` — never blocks compose validation.
    """
    n = traced_points.shape[0]
    return xp.ones(n)


uniform_density_component.required_context_keys = ()
