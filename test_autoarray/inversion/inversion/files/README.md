# Inversion test fixtures

- `mge_slam_nnls_systems.npz` — 8 positive-only systems `Q_<k>` (curvature_reg_matrix, 60x60) / `q_<k>`
  (data_vector) captured from the SLaM `source_lp[1]` MGE model on the autolens_profiling HST dataset
  (PyAutoArray#571): keys 0-4 never converge under JAX PDIP in 200 iterations, 5-6 hit the 50-iteration cap
  but converge by 200, 7 is healthy (19 iterations); `meta` holds the per-system JSON. Generator:
  `autolens_profiling/scripts/imaging/hazards/mge_nnls_capture.py` (autolens_profiling d6926af), run
  2026-09-24 on CPU fp64 with PyAutoArray 7fa8d2714f, PyAutoGalaxy 70a61e26cd, PyAutoLens 86054bbc19,
  PyAutoFit a736840127, jax 0.10.2.
