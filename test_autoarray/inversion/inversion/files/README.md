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
- `mge_solver_reference_systems.npz` — 9 positive-only MGE systems (60x60 fp64) with their fnnls reference
  solutions, for the raw-forward PDIP amplitude regression (PyAutoArray#594): keys `Q_<name>` / `q_<name>` /
  `x_ref_<name>` for names `k0`..`k7` (the 8 #571 SLaM `source_lp[1]` systems, identical to
  `mge_slam_nnls_systems.npz`) and `euclid_vis_lp` (the euclid pipeline vis_lp system behind
  `test_latent_euclid_variables_traces_under_jax_jit`, captured with PyAutoArray d4298445). `meta` is a JSON
  string with per-system `group`, `source_column_index_list`, `max_abs_q`, `cond_Q`, `category` and reference
  active-column counts, plus provenance. Copied verbatim (no recomputation) from the autolens_profiling solver
  corpus `results/lens/solver/corpus/{slam_fixture_571,euclid_vis_lp}.npz` + `manifest.json` at
  autolens_profiling 3ad68af (phase-1 record `complete/2026/09/linear-solver-accuracy-study.md`) by a one-off
  script outside the repo that checks symmetry of `Q`, `x_ref >= 0` and `max|q|` against the manifest.
