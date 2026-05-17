# Adaptive Rectangular CDF — Phase 1 Audit

Reference document for issue [#322](https://github.com/PyAutoLabs/PyAutoArray/issues/322).
Records exactly how the current JAX-compatible adaptive rectangular source-plane
mesh works, so that Phase 2+ can extend the density signal without touching the
spline machinery.

This is a developer note. It lives under `PyAutoArray/files/` to keep package
namespaces clean. All paths are relative to the PyAutoArray repo root unless
explicitly absolute.

---

## 1. Class hierarchy

Two adaptive variants live in `autoarray/inversion/mesh/mesh/`:

```
RectangularAdaptDensity (rectangular_adapt_density.py:64)
└── RectangularAdaptImage (rectangular_adapt_image.py:9)
        └── RectangularSplineAdaptImage (rectangular_spline_adapt_image.py:16)
└── RectangularSplineAdaptDensity (rectangular_spline_adapt_density.py:16)
```

Naming caveat: `RectangularAdaptDensity` is *not* an adaptive-coordinate mesh.
Its `interpolator_from` returns a plain `InterpolatorRectangular` whose
`_mappings_sizes_weights` runs the linear-CDF helper
(`adaptive_rectangular_mappings_weights_via_interpolation_from`, in
`autoarray/inversion/mesh/interpolator/rectangular.py:113`). It is "adaptive"
only in the implicit sense — the linear empirical CDF *is* the warp; there is
no separate uniform path. The `RectangularSpline*` subclasses swap that linear
CDF for the polynomial+Hermite-spline implementation in `rectangular_spline.py`.

The class hierarchy carries one switching flag downstream:

- `interpolator_cls` (property) — `InterpolatorRectangular` vs
  `InterpolatorRectangularSpline`.
- `MeshGeometryRectangular.spline_deg` (`mesh_geometry/abstract.py:22`) — when
  non-None, the geometry's `areas_transformed` and `edges_transformed`
  properties route to the spline helpers instead of the linear helpers
  (`mesh_geometry/rectangular.py:478,526`).

---

## 2. `mesh_weight_map` data flow

`mesh_weight_map` is the single per-point weight signal that drives both
the linear and spline CDF transforms. It is shape `(N_base,)` — one scalar per
point in the (non-oversampled) source-plane data grid.

### 2.1 Where it is produced

| Mesh class                       | `mesh_weight_map_from(adapt_data, xp)` returns |
|----------------------------------|------------------------------------------------|
| `RectangularAdaptDensity`        | `None` (rectangular_adapt_density.py:159)      |
| `RectangularAdaptImage`          | normalised `adapt_data` (rectangular_adapt_image.py:82) |
| `RectangularSplineAdaptDensity`  | inherited → `None`                              |
| `RectangularSplineAdaptImage`    | inherited from `RectangularAdaptImage`         |

The `RectangularAdaptImage` recipe at `rectangular_adapt_image.py:82-107`:

```python
mesh_weight_map = adapt_data.array
mesh_weight_map = xp.clip(mesh_weight_map, 1e-12, None)
mesh_weight_map = mesh_weight_map ** self.weight_power
mesh_weight_map = xp.where(
    mesh_weight_map < self.weight_floor,
    self.weight_floor,
    mesh_weight_map,
)
mesh_weight_map = mesh_weight_map / xp.sum(mesh_weight_map)   # final normalisation
```

Key observations:

- `weight_floor` is applied *after* the power, *before* normalisation.
- The output sums to 1.0. Downstream uses interpret it as a per-point
  probability mass (CDF construction via `cumsum` — see §3).
- `adapt_data` is the user-supplied adapt image, sampled at the *base* data
  grid (one value per base grid point), not the oversampled grid.

### 2.2 Where it is consumed

All four entry points (linear / spline × areas / mappings) take
`mesh_weight_map` as a keyword and pass it unchanged into `create_transforms*`:

| Call site                                           | Function                                          | Line |
|-----------------------------------------------------|---------------------------------------------------|------|
| `InterpolatorRectangular._mappings_sizes_weights`   | `adaptive_rectangular_mappings_weights_via_interpolation_from` | rectangular.py:297 |
| `InterpolatorRectangularSpline._mappings_sizes_weights` | `adaptive_rectangular_mappings_weights_via_interpolation_from_spline` | rectangular_spline.py:618 |
| `MeshGeometryRectangular.areas_transformed`         | `adaptive_rectangular_areas_from[_spline]`        | mesh_geometry/rectangular.py:478-496 |
| `MeshGeometryRectangular.edges_transformed`         | `adaptive_rectangular_transformed_grid_from[_spline]` | mesh_geometry/rectangular.py:526-548 |

The interpolator and the mesh geometry independently build their own
transforms (no caching across the two). The CDF is re-fit per `mesh_weight_map`.

---

## 3. The CDF transform itself

### 3.1 Linear CDF (`rectangular.py:70`)

`create_transforms(traced_points, mesh_weight_map=None, xp=np)`:

```python
N = traced_points.shape[0]

if mesh_weight_map is None:
    t = xp.arange(1, N + 1) / (N + 1)          # uniform rank probabilities
    t = xp.stack([t, t], axis=1)               # same t for y and x axes
    sort_points = xp.sort(traced_points, axis=0)
else:
    sdx = xp.argsort(traced_points, axis=0)    # per-axis sort permutation
    sort_points = xp.take_along_axis(traced_points, sdx, axis=0)
    t = xp.stack([mesh_weight_map, mesh_weight_map], axis=1)  # broadcast w to both axes
    t = xp.take_along_axis(t, sdx, axis=0)     # weights aligned to per-axis sort
    t = xp.cumsum(t, axis=0)                   # weighted empirical CDF
```

This is the **separability assumption** in code form. `mesh_weight_map` is
the *same* (N,) weight vector applied independently to each axis, after
sorting traced points by that axis. The resulting CDF is the marginal CDF
along y for axis 0 and the marginal CDF along x for axis 1.

If two source-plane points have identical x but different y, they will be
adjacent in the y-CDF and far apart in the x-CDF — same weight contribution
in each, but interpreted as different cumulative positions per axis.

The forward / inverse transforms are then `np.interp` (numpy path) or a
vmapped `jnp.interp` (JAX path), producing piecewise-linear maps
`source plane ↔ [0,1]²`.

### 3.2 Spline CDF (`rectangular_spline.py`)

The spline variant replaces `np.interp` with:

1. **Empirical CDF** (same as §3.1): per-axis weighted cumsum after sort.
2. **Chebyshev-node resample** (rectangular_spline.py:316-342): the empirical
   `(t, sort_points)` is resampled at Chebyshev nodes via linear interp
   (`_interp1d_numpy` / `_interp1d_jax`) to control Runge's phenomenon. Resample
   width is `3 * deg` (default `33`).
3. **Weighted polynomial fit** (rectangular_spline.py:349-355): degree-`deg`
   polynomial fit of `cx = f(cy)` with weights `1/dcx/dcy`. This gives a
   closed-form **inverse CDF**: `polyval(coefs, y) = x`.
4. **Hermite-spline forward inverse** (rectangular_spline.py:122-247):
   `InvertPolySpline` evaluates the polynomial on a `low_res = 20*deg` grid in
   `[0,1]`, enforces strict monotonicity (cummax + per-position eps jitter),
   then for any new `x` does binary search + cubic-Hermite interpolation to
   recover `y`.

The pair `(fwd_transform, rev_transform)` are C¹ continuous and differentiable.
`InvertPolySpline` is registered as a JAX pytree lazily so it can flow through
`jax.jit` / `jax.vmap` / autograd.

**The weight signal enters exactly the same way** as in the linear variant —
through the `t = take_along_axis(stack([w, w]), sdx) → cumsum` step in
`_build_inv_poly_jax_impl` (rectangular_spline.py:390-399) and
`_build_inv_poly_numpy` (rectangular_spline.py:325-330). Everything after that
is a smoothing/interpolation pipeline that takes the empirical `(t,
sort_points)` and turns it into a differentiable map.

### 3.3 Pre-CDF normalisation (the µ/σ step)

Before CDF construction, every entry point normalises the source-plane grid:

```python
mu = data_grid.mean(axis=0)                     # (2,) per-axis mean
scale = data_grid.std(axis=0).min()             # scalar — min std over axes
source_grid_scaled = (data_grid - mu) / scale   # zero-mean, isotropic-scale
```

This is a *single shared scale*, not per-axis. Anisotropic source-plane
distributions therefore retain their aspect ratio. The same `(mu, scale)` pair
is used to scale the over-sampled grid before pushing it through the
forward CDF (rectangular_spline.py:519-528, rectangular.py:174-186).

The transform is fit on the *base* data grid; the oversampled grid is only
*evaluated* through the transform — never used in CDF construction.

---

## 4. Index-space scatter and the `(N-3) * t + 1` rescale

`adaptive_rectangular_mappings_weights_via_interpolation_from_spline`
(rectangular_spline.py:505-562) and its linear sibling do:

```python
grid_over_sampled_transformed = transform(grid_over_sampled_scaled)
                                # shape (N_over, 2), values in [0, 1]
grid_over_index = (source_grid_size - 3) * grid_over_sampled_transformed + 1
                                # values in [1, N - 2]
```

Then floor/ceil and bilinear weights:

```python
ix_down = floor(grid_over_index[:, 0])
ix_up   = ceil(grid_over_index[:, 0])
iy_down = floor(grid_over_index[:, 1])
iy_up   = ceil(grid_over_index[:, 1])

w_tl = (1 - t_row) * (1 - t_col)        # t_row = (ix_down - ix_down) / 1
w_tr = (1 - t_row) * t_col
w_bl = t_row * (1 - t_col)
w_br = t_row * t_col
```

Row-major flatten with a flip:

```python
flat = (source_grid_size - idx[:, 0]) * source_grid_size + idx[:, 1]
```

Consequences:

- Interior pixel indices live in `[1, N-2]`. With `N = source_grid_size`, the
  outermost rows/cols (indices 0 and N-1 along each axis) are never the
  bottom-left of any sample's bilinear cell.
- `zeroed_pixels` (`rectangular_adapt_density.py:126`) zeros the entire perimeter
  ring via `rectangular_edge_pixel_list_from`. Combined with the `(N-3)+1`
  buffer, **effective resolution is `(N-2) × (N-2)`**.
  - N=33 → 31×31 = 961 effective pixels.
  - N=50 → 48×48 = 2304 effective pixels.
  - N=70 → 68×68 = 4624 effective pixels (≈ the "~4000+" the issue cites).
- Bilinear weights sum to 1 by construction; `sizes = 4` for every sample
  (`rectangular_spline.py:628`).
- The `+ 1e-12` epsilon in the denominators of `t_row` / `t_col`
  (rectangular.py:217-218, rectangular_spline.py:553-554) protects against
  exact knot hits where `ix_down == ix_up`; under JAX this guards autograd too.

---

## 5. Mesh-geometry side (areas / edges)

`MeshGeometryRectangular.areas_transformed` computes per-pixel source-plane
areas by transforming uniform unit-square edges *through the inverse CDF*
(mesh_geometry/rectangular.py:466-496, rectangular_spline.py:476-502):

```python
edges_y = xp.linspace(1, 0, N + 1)        # (N+1,) unit-square y-edges
edges_x = xp.linspace(0, 1, N + 1)        # (N+1,) unit-square x-edges
pixel_edges  = inv_transform([edges_y, edges_x].T) * scale + mu
pixel_lengths = diff(pixel_edges, axis=0).squeeze()
dy, dx = pixel_lengths[:, 0], pixel_lengths[:, 1]
return abs(outer(dy, dx).flatten())
```

This is **also separable**: pixel area is the outer product of per-axis edge
spacings. A non-separable density would not produce the correct per-pixel
area through this code path — the area would still be `dy_i * dx_j` even if
the density at `(y_i, x_j)` was uncorrelated with that product. Phase 4's
"is separability sufficient?" question lands here directly.

`edges_transformed` is the pcolormesh path — it pushes the unit-square edges
through the inverse CDF and returns the warped (y, x) edge coordinates for
plotting.

---

## 6. Where Phase 2 plugs in

The composable density signal must arrive at `create_transforms_spline` as a
`mesh_weight_map` argument of shape `(N_base,)`, summed to 1.0, strictly
positive. **No change is needed below `mesh_weight_map_from`** — that method is
the only producer of the weight vector that every downstream call site reads
from.

Two clean integration shapes:

### 6.1 Subclass approach (preferred)

Add `RectangularMultiComponentAdapt` next to `RectangularSplineAdaptImage`,
overriding only `mesh_weight_map_from(adapt_data, xp) -> (N_base,)`. The
override calls `compose_density(components, weights, floor, xp, context)` and
ends with the same `clip → normalise` finish as the parent. This keeps
`interpolator_from` and the spline interpolator class untouched.

The `context` (magnification map, current reconstruction, residuals, etc.)
is the new wrinkle — it has to arrive at `mesh_weight_map_from`. Two
candidates:

- Stash it on the mesh instance at construction time
  (`RectangularMultiComponentAdapt(shape=..., context=...)`), making the mesh
  carry the auxiliary fields. Cleanest at the call site; awkward if `context`
  changes between fits (which it does — residuals update each iteration).
- Override `interpolator_from` to accept `context` as a kwarg and pass it
  through. Slightly more plumbing but matches how `adapt_data` already flows
  in the parent.

Recommend Option B: keep the mesh stateless, pass `context` through
`interpolator_from`. The PyAutoGalaxy/Lens caller already has the auxiliary
fields at hand when it constructs the interpolator.

### 6.2 Helper-injection approach (NOT recommended)

Change `mesh_weight_map_from` to accept the multi-component arguments
directly. This pollutes the base-class signature and forces every existing
caller to know about the new fields. Rejected — keep the subclass.

---

## 7. Open questions for Phase 4 (separability & low-rank)

The current architecture commits to **separable per-axis CDFs**. For a single
density signal that's well-approximated by `ρ(x,y) = a(x) · b(y)` (Gaussian-like
sources, point-density of regularly traced rays), separability is fine.

For multi-component densities the worry is:

1. **Caustic proximity** is anisotropic along curves — the level sets are
   1-D loci in the source plane. The marginal x-CDF and y-CDF of a caustic
   density will concentrate pixels along the x and y projections of the
   caustic, *not* along the caustic itself. Pixels get wasted on the convex
   hull of the projections rather than placed on the curve.
2. **Magnification density** is similarly anisotropic near critical curves.
3. **Source brightness × residual gradient** products may produce diagonal
   or curved support that no separable factorisation captures.

Phase 4 should measure how badly separability hurts vs a full 2-D CDF and
decide whether an outer-product factorisation
`ρ(x,y) ≈ a(x)·b(y) + Σ_k u_k(x)·v_k(y)` (sum of rank-1 outer products) is
worth the engineering cost. The current code has no obvious slot for a
non-separable CDF — would need a new transform class, not a new
`mesh_weight_map`.

A separate concern: the **areas calculation** (§5) is fundamentally
separable in this code. Even if Phase 2 ships a non-separable weight signal,
the areas reported to magnification/regularisation downstream would still be
`dy_i · dx_j`. Investigating non-separability requires touching
`areas_transformed` *and* `edges_transformed` *and* the CDF construction
itself — a much bigger change than the Phase 2 subclass.

---

## 8. Deferred to Phase 5: quantitative baseline

The issue calls for baseline reconstruction χ², log-evidence, and peak
per-pixel residual at 500 / 1000 / 4000 source pixels on a reference dataset.
That requires a full lens-model fit (image → tracer → mapper → inversion →
fit), which is `autolens_workspace` / `autolens_workspace_developer` territory
— this issue scope ends at the library prototype + unit tests.

Followup prompt: `autolens_workspace_developer/rectangular_adapt_cdf_benchmark.md`,
to be authored at the end of Phase 3 once the multi-component API is stable.
The benchmark script should:

- Use one of the existing `autolens_workspace_test/scripts/imaging/`
  pixelization examples as the reference dataset.
- Sweep N ∈ {25, 32, 40, 50, 70} (effective pixel counts 529, 900, 1444,
  2304, 4624) for both single-signal (current) and multi-component (Phase 3)
  meshes.
- Hold sampler budget and regularisation strategy fixed.
- Compare against a Delaunay run at the matched compute budget.

---

## 9. JAX compatibility checklist (what NOT to break)

The current spline path is fully JAX/JIT-compatible. Phase 2/3 components
must preserve every property:

- No dynamic shapes — every component returns `(N_base,)` regardless of
  trigger conditions. Branches that "skip" a component should multiply its
  contribution by 0, never resize.
- No scipy callbacks, no `pure_callback`, no kNN / Delaunay / RBF.
- No Python-level control flow on traced values. `jnp.where` for masking.
- Strictly positive output (`xp.clip(w, eps, None)` before normalisation).
- Finite under autograd — the `1e-12` epsilon trick in the bilinear weights is
  load-bearing; replicate when introducing new divisions.
- `InvertPolySpline` pytree registration is *lazy* (rectangular_spline.py:163-204);
  composing new components that wrap it must trigger
  `InvertPolySpline._register_pytree()` themselves or rely on the existing
  trigger in `_build_inv_poly_jax`.

Library unit tests stay numpy-only per
`feedback_no_jax_in_unit_tests` — cross-xp parity goes in
`autolens_workspace_test`.

---

## 10. Files touched in subsequent phases

- New: `autoarray/inversion/mesh/interpolator/density_components.py` —
  `compose_density` + per-component callables.
- New: `test_autoarray/inversion/pixelization/interpolator/test_density_components.py`
  — numpy-only unit tests.
- (Phase 6, conditional) New:
  `autoarray/inversion/mesh/mesh/rectangular_multi_component_adapt.py` and a
  parity test in `test_rectangular_spline.py`.

No edits to `rectangular_spline.py`, `rectangular.py`, or the existing mesh
classes are required for Phases 2–3. Phase 4's non-separable investigation
might force a deeper change, but only if separability is shown to be the
binding constraint.
