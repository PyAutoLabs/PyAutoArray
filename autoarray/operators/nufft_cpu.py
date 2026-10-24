"""
The CPU (NumPy) NUFFT backend of `TransformerNUFFT` and the NUFFT precision-operator builders.

Every eager NumPy NUFFT in PyAutoArray goes through the two functions below, which wrap `finufft`
(https://github.com/flatironinstitute/finufft), the CPU C++ library that `nufftax` ports to JAX.
They mirror `nufftax.nufft2d1` / `nufftax.nufft2d2` exactly -- same arguments, same output
orientation -- so a NumPy branch swaps one call for the other and nothing else changes.

The point of the split is that the NumPy path never imports JAX. Importing `nufftax` alone
initialises XLA (its spreader builds JAX arrays at import), and a parent process that has started
JAX's threads before a multiprocessing search forks its workers can crash those workers when numba
compiles in them. The array namespace therefore picks the backend: `xp=np` (eager NumPy) comes
here and runs `finufft`; `xp=jnp` / `use_jax=True` stays on `nufftax`, which `jax.jit`, `jax.grad`
and `jax.vmap` need. `finufft` is also 25-35x faster than `nufftax` on CPU, and the two agree to
~1e-14 relative.

Orientation: `finufft` returns (and takes) the mode grid in `n_modes` order, `(n_modes[0],
n_modes[1])`, with `x` paired to the first axis; `nufftax` uses `(n_modes[1], n_modes[0])` with
`x` paired to the last axis. The wrappers transpose the last two axes to give `nufftax`'s layout.

`finufft` has no wheel for every platform (notably Linux aarch64). Where it is not importable the
functions fall back to `nufftax` with a one-time warning, since that puts JAX back on the NumPy
path; where neither is importable they raise `nufft_cpu_exception`.
"""

import importlib.util
import logging
import os
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

# finufft is a compiled extension with no JAX dependency, but it is still only imported on first
# use so that sessions which never build an interferometer transformer do not pay for it.
_finufft = None
_finufft_loaded = False
_fallback_warned = False

# The pid of the process in which finufft first ran, i.e. whose OpenMP (libgomp) thread pool it
# started. The value is inherited by forked children, which is how they recognise themselves.
_openmp_pid = None


def _load_finufft():
    global _finufft, _finufft_loaded
    if _finufft_loaded:
        return _finufft
    _finufft_loaded = True
    try:
        import finufft
    except ImportError:
        return None
    _finufft = finufft
    return _finufft


def _threads_kwargs() -> dict:
    """
    The `nthreads` keyword for a finufft call: none (finufft's default, `OMP_NUM_THREADS` or all
    cores) in the process that first ran finufft, `nthreads=1` in a process forked from it.

    GNU OpenMP (libgomp, which the finufft wheels bundle) is not fork-safe: once a parent has run a
    multi-threaded OpenMP region, a forked child that opens another multi-threaded region
    deadlocks. This is exactly the multiprocessing-search pattern -- the parent builds the
    precision operator and dirty image with finufft, then Nautilus forks workers whose likelihood
    calls `TransformerNUFFT.visibilities_from` -- so forked children are pinned to one thread
    (which is what a worker sharing the machine with its siblings wants anyway). A child whose
    parent never ran finufft keeps finufft's default.
    """
    global _openmp_pid

    pid = os.getpid()

    if _openmp_pid is None:
        _openmp_pid = pid

    if pid == _openmp_pid:
        return {}

    return {"nthreads": 1}


def backend_available() -> bool:
    """
    Whether any NUFFT backend can run the NumPy path: `finufft`, or `nufftax` as the fallback.

    `nufftax` is only located (`importlib.util.find_spec`), never imported, because importing it
    starts JAX -- which is exactly what a NumPy user checking for a backend must not do.
    """
    if _load_finufft() is not None:
        return True
    return importlib.util.find_spec("nufftax") is not None


def nufft_cpu_exception():
    raise ModuleNotFoundError(
        "\n--------------------\n"
        "You are attempting to perform interferometer analysis with `TransformerNUFFT` or the "
        "NUFFT precision operator.\n\n"
        "However, neither of its NUFFT backends is installed: finufft "
        "(https://github.com/flatironinstitute/finufft), used on the NumPy path, nor nufftax "
        "(https://github.com/GragasLab/nufftax), used on the JAX path.\n\n"
        "Install finufft via the command `pip install finufft`.\n\n"
        "If your platform has no finufft wheel, install nufftax via the command "
        "`pip install nufftax` (it requires JAX), or use "
        "`transformer_class=TransformerDFT` instead.\n\n"
        "----------------------"
    )


def _nufftax_fallback():
    """
    Returns `nufftax` for the NumPy path when `finufft` is not importable, warning once that this
    starts JAX, or raises `nufft_cpu_exception` when `nufftax` is not importable either.
    """
    global _fallback_warned

    # Imported here: transformer.py imports this module at its top.
    from autoarray.operators.transformer import _load_nufftax

    nufftax = _load_nufftax()

    if nufftax is None:
        nufft_cpu_exception()

    if not _fallback_warned:
        _fallback_warned = True
        logger.warning(
            "INTERFEROMETER - `finufft` is not installed (it has no wheel for some platforms, "
            "e.g. Linux aarch64), so the NumPy NUFFT path falls back to `nufftax`, which starts "
            "JAX in this process. A multiprocessing search (e.g. Nautilus with "
            "`number_of_cores > 1`) that forks numba workers after JAX has started can then "
            "crash with a segmentation fault. Install `finufft` via `pip install finufft` to "
            "keep the NumPy path JAX-free."
        )

    return nufftax


def nufft2d1(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    n_modes: Tuple[int, int],
    eps: float = 1.0e-12,
    isign: int = 1,
) -> np.ndarray:
    """
    Type-1 (adjoint) 2D NUFFT on the CPU, a drop-in for `nufftax.nufft2d1`:

        f[..., m2, m1] = sum_k c[..., k] exp(isign * i(m1 * x_k + m2 * y_k))

    on the centred mode grid (mode 0 at index `N // 2`), returned with shape
    `(..., n_modes[1], n_modes[0])` -- `nufftax`'s orientation, not `finufft`'s.

    Runs `finufft.nufft2d1` on **one thread**. A multithreaded type-1 spread adds each thread's
    subgrid into the output in a non-deterministic order, so two identical calls can differ in
    the last bits; the products built from this transform (the dirty image and the NUFFT
    precision operator) are one-off precomputations that must be bitwise reproducible, and one
    thread is still ~10x faster than `nufftax` on the CPU (~0.9 s for 5.5M visibilities). It also
    keeps this transform out of the libgomp fork deadlock `_threads_kwargs` guards against.
    Falls back to `nufftax` (with a one-time warning) when `finufft` is not importable.

    Parameters
    ----------
    x, y
        `[K]` non-uniform frequencies in `[-pi, pi)`; `x` pairs with `n_modes[0]`.
    c
        `[K]` (or stacked `[n_trans, K]`) complex strengths.
    n_modes
        `(N1, N2)`, the mode counts along the `x` and `y` axes.
    eps
        The requested NUFFT precision.
    isign
        The sign of the exponent, `+1` or `-1`.

    Returns
    -------
    np.ndarray
        `[..., N2, N1]` complex128.
    """
    finufft = _load_finufft()

    if finufft is None:
        return np.asarray(_nufftax_fallback().nufft2d1(x, y, c, n_modes, eps, isign))

    f = finufft.nufft2d1(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(c, dtype=np.complex128),
        tuple(int(n) for n in n_modes),
        eps=float(eps),
        isign=int(isign),
        nthreads=1,
    )

    return np.swapaxes(f, -1, -2)


def nufft2d2(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    eps: float = 1.0e-12,
    isign: int = -1,
) -> np.ndarray:
    """
    Type-2 (forward) 2D NUFFT on the CPU, a drop-in for `nufftax.nufft2d2`:

        c[..., k] = sum_{m1, m2} f[..., m2, m1] exp(isign * i(m1 * x_k + m2 * y_k))

    for a mode grid `f` in `nufftax`'s orientation, shape `(..., N2, N1)`, centred (mode 0 at
    index `N // 2`). A stacked `f` of shape `(n_trans, N2, N1)` is transformed in one call and
    returns `(n_trans, K)`, as `TransformerNUFFT.transform_mapping_matrix` relies on.

    Runs `finufft.nufft2d2`, whose thread count is left to `finufft` (OpenMP, honouring
    `OMP_NUM_THREADS`; one thread in a forked child, see `_threads_kwargs`). Falls back to `nufftax` (with a one-time warning) when `finufft` is not
    importable.

    Parameters
    ----------
    x, y
        `[K]` non-uniform frequencies in `[-pi, pi)`; `x` pairs with the last axis of `f`.
    f
        `[N2, N1]` (or stacked `[n_trans, N2, N1]`) complex mode grid.
    eps
        The requested NUFFT precision.
    isign
        The sign of the exponent, `+1` or `-1`.

    Returns
    -------
    np.ndarray
        `[K]` (or `[n_trans, K]`) complex128.
    """
    finufft = _load_finufft()

    if finufft is None:
        return np.asarray(_nufftax_fallback().nufft2d2(x, y, f, eps, isign))

    f = np.asarray(f, dtype=np.complex128)

    return finufft.nufft2d2(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(np.swapaxes(f, -1, -2)),
        eps=float(eps),
        isign=int(isign),
        **_threads_kwargs(),
    )
