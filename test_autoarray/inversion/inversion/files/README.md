# Inversion test fixtures

- `mge_slam_nnls_systems.npz` — 8 positive-only systems `Q_<k>` (curvature_reg_matrix, 60x60) / `q_<k>`
  (data_vector) captured from the SLaM `source_lp[1]` MGE model on the autolens_profiling HST dataset
  (PyAutoArray#571): keys 0-4 never converge under JAX PDIP in 200 iterations, 5-6 hit the 50-iteration cap
  but converge by 200, 7 is healthy (19 iterations); `meta` holds the per-system JSON. Generator:
  `autolens_profiling/scripts/imaging/hazards/mge_nnls_capture.py` (autolens_profiling d6926af), run
  2026-09-24 on CPU fp64 with PyAutoArray 7fa8d2714f, PyAutoGalaxy 70a61e26cd, PyAutoLens 86054bbc19,
  PyAutoFit a736840127, jax 0.10.2.
- `mge_grad_nan_systems.npz` — 4 positive-only systems `Q_<k>` (20x20) / `q_<k>` captured from the
  autolens_workspace_test `scripts/imaging/jax_grad/mge.py` model (MGE source, NFWSph + ExternalShear) at
  `physical_values_from_prior_medians + jax.random.uniform(PRNGKey(p), minval=0.01, maxval=0.05)` for
  p = 2, 10, 12, 14 (PyAutoArray#573): on PyAutoArray 3de624b5 the `"raw"`-mode gradient is NaN on each (the
  relaxed-KKT backward solve diverges from the loose raw-forward iterate); `meta` holds the per-system JSON.
  Captured 2026-09-25 on CPU fp64 via a `jax.debug.callback` on `reconstruction_positive_only_from`, with
  autolens_workspace_test 5ec64413d2, PyAutoArray 3de624b5b9, PyAutoGalaxy 70a61e26cd, PyAutoLens 86054bbc19,
  PyAutoFit dd9fbe0aab, jax 0.10.2, numpy 2.5.3.
