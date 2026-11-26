"""
The policy and gates of the lazy sparse operator (`Imaging.sparse_operator_for`,
`Interferometer.sparse_operator_for`).

Modeling applies its compute setup eagerly, before a search forks its workers, so a model-fit
always has the sparse operator it needs attached to its dataset. Calculations outside a search do
not: a plain `FitImaging(dataset, tracer)`, an aggregator reload or `analysis.fit_from(...)` before
a search take the dense mapping-matrix route on CPU, which is slow and whose memory grows as
n_image x n_source. With the policy on, a dataset without an attached operator builds the one its
NumPy pixelized fits need on first use, caches it privately and reuses it.

The hook is on the real dataset rather than the inversion factory: every fit wraps its data in a
fresh `DatasetInterface` with no setup and no back-reference, so a factory-side cache would be lost
after every fit.

This module holds what both dataset types share: the policy (`lazy_sparse_operator_enabled`), the
gates under which nothing is ever built (`lazy_build_blocked_reason`, `in_worker_process`) and the
interferometer visibility budget (`visibility_budget`).
"""

import logging
import os
import sys
from typing import Optional

from autonerves import conf

logger = logging.getLogger(__name__)

# The visibility count above which the lazy interferometer build accumulates the operator chunk by
# chunk (`apply_sparse_operator_from_chunks`) instead of in one shot. The unchunked NUFFT
# spreader's gather buffer is ~K x nspread^2 complex128 (~3 GB at 1e6 visibilities, ~15 GB at 5e6;
# see `nufft_precision_operator_from`), whereas the stored operator is extent-sized. Overridden by
# `general.yaml` `inversion: lazy_sparse_operator_visibility_budget`.
VISIBILITY_BUDGET = 1_000_000

# The number of visibilities per chunk of the chunked lazy build (~0.8 GB of spreader buffer at
# `eps = 1e-12`).
CHUNK_VISIBILITIES = 2**18


def lazy_sparse_operator_enabled(settings=None) -> bool:
    """
    Whether the lazy sparse operator policy is on.

    - An explicit `Settings(lazy_sparse_operator=...)` decides on its own (``"auto"`` on, ``"off"``
      off), so a script can switch the lazy build off whatever the environment says.
    - Otherwise the policy is on if `general.yaml` `inversion: lazy_sparse_operator` is ``"auto"``,
      (the packaged default), or the environment variable `PYAUTO_SETUP` is set to ``auto`` (parsed by
      `autonerves.environment.parse_setup_intent`). An unset or empty `PYAUTO_SETUP` does not turn
      on a config of ``"off"``.

    Parameters
    ----------
    settings
        The `Settings` of the fit, if any; `None` reads the config.
    """
    from autoarray.settings import Settings

    settings = settings if settings is not None else Settings()

    if settings._lazy_sparse_operator is not None:
        return settings.lazy_sparse_operator == "auto"

    if settings.lazy_sparse_operator == "auto":
        return True

    return _pyauto_setup_is_auto()


def lazy_sparse_operator_trigger(settings=None) -> str:
    """
    What turned the lazy sparse operator policy on, as the build notice says it:
    ``"lazy_sparse_operator: auto"`` (an explicit `Settings` or `general.yaml`), or
    ``"PYAUTO_SETUP=auto"`` when only the environment variable did.
    """
    from autoarray.settings import Settings

    settings = settings if settings is not None else Settings()

    if settings._lazy_sparse_operator is not None or settings.lazy_sparse_operator == "auto":
        return "lazy_sparse_operator: auto"

    return "PYAUTO_SETUP=auto" if _pyauto_setup_is_auto() else "lazy_sparse_operator: auto"


def _pyauto_setup_is_auto() -> bool:
    """
    Whether `PYAUTO_SETUP` is explicitly set to ``auto``. An invalid value is left for
    `Setup.auto` to report; here it simply does not turn the policy on.
    """
    from autonerves.environment import SETUP_ENV_VAR, parse_setup_intent

    value = os.environ.get(SETUP_ENV_VAR)

    if value is None or not value.strip():
        return False

    try:
        return parse_setup_intent(value) == "auto"
    except ValueError:
        return False


def visibility_budget() -> int:
    """
    The visibility count above which the lazy interferometer build is chunked: the config value
    `inversion: lazy_sparse_operator_visibility_budget` when set (not `null`), else
    `VISIBILITY_BUDGET`.
    """
    try:
        value = conf.instance["general"]["inversion"][
            "lazy_sparse_operator_visibility_budget"
        ]
    except KeyError:
        value = None

    if value is None:
        return VISIBILITY_BUDGET

    return int(value)


def is_jax_tracer(array) -> bool:
    """
    Whether `array` (or the raw array an autoarray structure wraps) is a JAX tracer, i.e. the call
    is inside `jax.jit` / `jax.grad` / `jax.vmap`. JAX is only consulted when it is already
    imported, so this never imports it.
    """
    if "jax" not in sys.modules:
        return False

    jax = sys.modules["jax"]

    tracer_type = getattr(getattr(jax, "core", None), "Tracer", None)

    if tracer_type is None:
        return False

    return isinstance(getattr(array, "array", array), tracer_type) or isinstance(
        getattr(array, "_array", None), tracer_type
    )


def in_worker_process() -> bool:
    """
    Whether this is a multiprocessing worker (or a pid-changed fork), where no operator is ever
    built: a lazy build there would be one operator per worker. Workers inherit the parent's cache
    instead, because PyAutoFit evaluates one likelihood before forking.
    """
    from autoarray import numba_util

    return not numba_util.parallel_kernel_allowed()


def lazy_build_blocked_reason(
    dataset, xp, needs_operator: bool, settings=None
) -> Optional[str]:
    """
    The reason the lazy sparse operator is neither built nor returned for this call, or `None` if
    it may be (a cached operator is then returned, or one is built when the call is not inside a
    worker).

    Nothing is built or returned when:

    - `needs_operator` is False (no pixelization, e.g. an MGE-only fit);
    - the policy is off (`lazy_sparse_operator_enabled`);
    - `xp` is not NumPy, or the noise-map is a JAX tracer (the JAX path stays modeling-applied or
      explicit);
    - the dataset has an explicit setup whose operator is not ``"sparse"`` (e.g. a dense setup).

    Parameters
    ----------
    dataset
        The real `Imaging` / `Interferometer` dataset.
    xp
        The array module of the fit (`numpy` or `jax.numpy`).
    needs_operator
        Whether the fit has a pixelization, i.e. would use a sparse operator at all.
    settings
        The `Settings` of the fit, if any; `None` reads the config.
    """
    import numpy as np

    if not needs_operator:
        return "the fit needs no sparse operator"

    if not lazy_sparse_operator_enabled(settings=settings):
        return "the lazy sparse operator policy is off"

    if xp is not np:
        return "the array backend is not NumPy"

    if is_jax_tracer(dataset.noise_map):
        return "the noise-map is a JAX tracer"

    setup = getattr(dataset, "setup", None)

    if setup is not None and setup.operator != "sparse":
        return f"the dataset has an explicit {setup.operator!r} setup"

    return None
