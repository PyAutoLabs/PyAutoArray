"""
Process pool for parallelising the scipy.spatial.Delaunay triangulation
across batch elements when running under ``jax.vmap``.

The split-callback path (see ``_jax_delaunay_split`` in ``delaunay.py``)
keeps only the triangulation in scipy via ``jax.pure_callback``. With
``vmap_method="sequential"`` (the default) JAX serialises the callback
across batch elements — at production batch=20 that's 20 sequential
~3 ms scipy.Delaunay calls, ~60 ms of wall time per batched likelihood
that could be parallelised across CPU cores.

This module manages a persistent ``ProcessPoolExecutor`` of N workers
that each have scipy pre-imported. The batched callback below dispatches
all B mesh-grids to the pool concurrently and stacks the results.

Notes on correctness and stability:

- ``multiprocessing`` start method is forced to ``"spawn"`` so workers
  do not inherit the parent's CUDA context. JAX's GPU state is fine
  to keep on the main process; the workers only need scipy.
- The pool is created lazily on first use and persists for the process
  lifetime. ``atexit`` registers a shutdown so workers are cleaned up.
- scipy/Qhull has C-level global state which is NOT thread-safe, but
  it IS safe across processes (each worker has its own address space).
- Worker count respects ``SLURM_CPUS_PER_TASK`` if set; otherwise falls
  back to ``os.cpu_count()`` capped by the optional autoconf flag
  ``inversion.delaunay_scipy_pool_workers``.
"""

import atexit
import os
import numpy as np


_POOL = None
_POOL_N_WORKERS = None


def _worker_init():
    """Pre-import scipy so the first triangulation in a worker doesn't pay
    the import cost on the critical path."""
    from scipy.spatial import Delaunay  # noqa: F401


def _triangulate_worker(points_np):
    """Worker entry point. Receives a single (N, 2) array of mesh
    vertices, returns ``(points, simplices)`` — both numpy arrays.

    Module-scope so it's picklable. Returns the raw scipy outputs;
    padding is done in the main process after gather.
    """
    from scipy.spatial import Delaunay

    tri = Delaunay(points_np)
    return tri.points.astype(points_np.dtype), tri.simplices.astype(np.int32)


def _resolve_n_workers(requested: int = 0) -> int:
    """Return the worker count to use.

    Priority:
      1. Explicit ``requested`` if > 0
      2. ``SLURM_CPUS_PER_TASK`` if set (HPC)
      3. ``os.cpu_count()``
      4. 1 (fallback)
    """
    if requested > 0:
        return requested
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit() and int(slurm) > 0:
        return int(slurm)
    cpus = os.cpu_count()
    return cpus or 1


def get_pool(requested_workers: int = 0):
    """Return the singleton pool, lazily initialising on first call.

    If the requested worker count differs from the live pool's count,
    shut down the existing pool and start a new one (rare; typically
    only happens once per process).
    """
    global _POOL, _POOL_N_WORKERS

    n = _resolve_n_workers(requested_workers)
    if _POOL is not None and _POOL_N_WORKERS == n:
        return _POOL

    if _POOL is not None:
        # Different worker count requested: tear down and recreate
        _POOL.shutdown(wait=True)
        _POOL = None

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    _POOL = ProcessPoolExecutor(
        max_workers=n,
        mp_context=ctx,
        initializer=_worker_init,
    )
    _POOL_N_WORKERS = n
    atexit.register(_shutdown_pool)
    return _POOL


def _shutdown_pool():
    global _POOL, _POOL_N_WORKERS
    if _POOL is not None:
        try:
            _POOL.shutdown(wait=True)
        except Exception:  # noqa: BLE001
            pass
        _POOL = None
        _POOL_N_WORKERS = None


def scipy_triangulate_batched(points_batched, max_simplices, n_workers: int = 0):
    """Host-side callback for ``vmap_method='parallel'`` scipy triangulation.

    Parameters
    ----------
    points_batched : numpy.ndarray
        Either ``(N, 2)`` (un-vmapped) or ``(B, N, 2)`` (under vmap).
    max_simplices : int
        Pad height for the output ``simplices_padded`` array. Must equal
        ``2 * N`` to match the rest of the autoarray Delaunay code.
    n_workers : int
        Pool size hint; see ``_resolve_n_workers``. 0 means auto.

    Returns
    -------
    (points_out, simplices_padded) :
        Shapes ``(N, 2), (max_simplices, 3)`` if un-vmapped, or
        ``(B, N, 2), (B, max_simplices, 3)`` if vmapped.
    """
    arr = np.asarray(points_batched)

    if arr.ndim == 2:
        # Un-vmapped path: single triangulation, no pool needed.
        pts, simps = _triangulate_worker(arr)
        out = -np.ones((max_simplices, 3), dtype=np.int32)
        out[: simps.shape[0]] = simps
        return pts, out

    if arr.ndim != 3:
        raise ValueError(
            f"scipy_triangulate_batched: expected ndim 2 or 3, got {arr.ndim}"
        )

    B, N, _ = arr.shape
    pool = get_pool(n_workers)

    futures = [pool.submit(_triangulate_worker, arr[i]) for i in range(B)]
    results = [f.result() for f in futures]

    pts_out = np.stack([r[0] for r in results])
    simps_out = -np.ones((B, max_simplices, 3), dtype=np.int32)
    for i, (_, simps) in enumerate(results):
        simps_out[i, : simps.shape[0]] = simps
    return pts_out, simps_out
