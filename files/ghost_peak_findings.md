# Ghost-Peak Findings — Phase 1.5

Empirical results from `files/ghost_peak_experiment.py`. Companion to
`files/cdf_audit.md`. Issue
[#322](https://github.com/PyAutoLabs/PyAutoArray/issues/322).

## Hypothesis

A separable per-axis CDF cannot adapt to multi-modal source planes (multiple
bright regions) without creating ghost peaks at the cross-product of marginal
modes.

## Result — **CONFIRMED**

| Configuration | mean(real) | mean(ghost) | ghost / real |
|---------------|-----------:|------------:|-------------:|
| K = 1 (control, one peak at origin) | 1006 px | — | n/a |
| K = 2 (diagonal: peaks at ±(0.5, 0.5)) | 226 px | 224 px | **0.99** |
| K = 3 (offset: 3 non-axis-aligned peaks) | 109 px | 105 px | **0.96** |

Visual proof: `files/ghost_peak.png`. The K=2 panel shows a 2×2 grid of dense
mesh zones at all four (±0.5, ±0.5) corners — two coincide with the bright
source peaks (green circles), two land on empty background (red ×). The
warped mesh wastes **roughly half the pixel budget** on ghosts.

The K=3 panel shows the corresponding 3×3 grid of dense zones, of which 3 are
real and 6 are ghosts — **two-thirds of the adaptive budget is wasted**.

## Why it happens (mechanism)

Per-axis CDF construction (`rectangular_spline.py:create_transforms_spline`)
sorts traced points independently along y and x. The two marginal CDFs
steepen at the y- and x-projections of every peak. When the inverse CDF is
evaluated on a uniform unit-square grid, both axes concentrate edges at the
projections — and the outer product places dense pixels at **all K²
combinations** of (y-projection × x-projection), only K of which are real
source peaks.

This is a structural property of any separable transform; no choice of
`weight_power`, `weight_floor`, or `spline_deg` can avoid it. Composing
multiple physics signals (magnification + brightness + residuals + caustic)
into the same `mesh_weight_map` slot — what the original Phase 3 plan was
building — inherits the failure because all signals still funnel through the
same separable transform.

## Implication for the project

The multi-component framework shipped in
`autoarray/inversion/mesh/interpolator/density_components.py` (Phase 2:
`compose_density` + `uniform_density_component`) is still useful for
combining physics signals on **single-mode** sources, but it does **not**
address the user's stated goal of adapting to multiple source-plane regions
simultaneously. That goal requires either:

1. A non-separable transform (breaks axis-aligned-rectangle pixel shape), or
2. Multiple meshes (one per source region) stitched together (breaks
   single-mesh assumption), or
3. A rotation hack that handles the K=2 case (most common in practice).

## Three concrete paths forward, smallest-first

### Path A — PCA rotation hack (smallest change, partial fix)

For any multi-modal brightness map, PCA on the brightness-weighted point
cloud gives a principal axis. Rotate the source plane into a frame where
that axis is x (or y), run the existing separable CDF, rotate the warped
mesh back.

- **K = 2 (binary peaks)**: peaks become collinear on the principal axis →
  marginal-y collapses (one mode), marginal-x has both modes → outer product
  gives **only 2 dense zones, both real, zero ghosts**.
- **K = 3 colinear**: same — perfect fix.
- **K = 3 non-colinear**: PCA finds the dominant axis but the off-axis peak
  still creates ghosts. Reduces ghost density but doesn't eliminate them.
- **K ≥ 4**: PCA helps proportionally less.

**Effort**: small. A new `RectangularRotatedAdaptImage` subclass that wraps
`RectangularSplineAdaptImage` with pre/post rotation steps.

**JAX**: trivial (rotation is a single matrix multiply, JAX-friendly).

**Honest limitation**: helps the most common case (binary / colinear) but
isn't a complete fix.

### Path B — Pre-segment + multi-sub-mesh (moderate change, full fix)

Detect K modes in the adapt image (Gaussian mixture, k-means on
brightness-weighted points, or quantile thresholding). Per detected mode,
construct a separate `RectangularSplineAdaptImage` covering that mode's
bounding box. Combine the K sub-meshes into a `MultiRectangularAdapt`
container that presents a single mesh interface to the inversion pipeline.

- **Each sub-mesh** uses unchanged CDF code.
- **No ghosts** within each sub-mesh because each is single-mode.
- **Coverage**: gaps between sub-meshes either remain unmodelled (acceptable
  for high-contrast multi-modal sources) or get a low-resolution fallback
  sub-mesh.

**Effort**: moderate. New container class, segmentation logic (probably
JAX-incompatible — would run as a numpy preprocessing step before the
JIT-compiled likelihood), per-sub-mesh inversion integration.

**JAX**: the per-sub-mesh inversions stay JAX-compatible; the segmentation
step (clustering / GMM fit) is once per fit, not per likelihood eval, so it
can be numpy-only.

**Honest limitation**: requires choosing K (or detecting it). Segmentation
quality determines reconstruction quality.

### Path C — Knothe-Rosenblatt with non-axis-aligned cells (largest change, principled)

Replace the separable transform with a Knothe-Rosenblatt 2D transport:
y = F₁⁻¹(u), x = F₂⁻¹(v | y). Conditional x-CDF depends on y, so x-edges vary
per y-stripe. Cells become quadrilaterals (still 4 corners, still
combinatorially-rectangular neighbours, but no longer axis-aligned).

- **No ghosts** — the transform follows the full joint density.
- **Topology**: rectangular *connectivity* preserved. Cell *geometry*
  becomes general quadrilateral.
- **Areas calculation** (Phase 1 audit §5) — needs Jacobian instead of
  `outer(dy, dx)`. Still tractable but ≠ outer product.
- **Bilinear scatter**: needs bilinear-on-quadrilateral, a bigger change to
  `adaptive_rectangular_mappings_weights_via_interpolation_from_spline`.
- **Spline machinery**: the conditional CDF requires a 2D function — a
  tensor-product spline or one polynomial per y-bin.

**Effort**: large. New CDF, new bilinear, new areas, new mesh class. The
test surface roughly doubles.

**JAX**: feasible, but the conditional CDF needs careful design to avoid
dynamic shapes.

**Honest limitation**: the "fixed rectangular topology" phrase in the prompt
might mean axis-aligned rectangles, in which case Path C violates the
constraint. Need user confirmation before committing.

## What I'd revert / keep from the work so far

- **Keep**: Phase 1 audit (`cdf_audit.md`), Phase 2 framework
  (`density_components.py` + tests, currently 9/9 passing), this findings doc
  and the experiment script.
- **Drop**: the four physical-density factory functions
  (`magnification_density`, `brightness_density`, `residual_density`,
  `caustic_proximity_density`) — already reverted out of
  `density_components.py`. They were aimed at the wrong target.

## Recommendation

Pick Path A first. It's small, it handles the most common multi-peak case
(binary / colinear sources) cleanly, and it provides a real-world reference
point for measuring how much Path B/C would buy on harder cases. We can
re-extend Path A → B → C as the user's actual source complexity demands,
without backing-out work each step.

## Path A empirical follow-up — **WORKS**

`files/pca_rotation_experiment.py` reproduces the ghost-peak diagnostic with
a brightness-weighted PCA pre-rotation. Compares baseline vs rotated for the
same three test cases.

| Case | ghost/real (baseline) | ghost/real (rotated) | real-peak density |
|------|----------------------:|---------------------:|------------------:|
| K = 2 diagonal           | 0.99 | **0.00** | +117% (226 → 490.5) |
| K = 2 axis-aligned       | 0.00 | 0.00     | unchanged (467 → 453) |
| K = 3 offset (triangle)  | 0.96 | **0.06** | +46% (109 → 159.3)  |

Visual proof: `files/pca_rotation.png`. The K = 2 diagonal mesh rotates by
+45.75° to align with the peaks — the cyan mesh grid hugs the diagonal
band between the real peaks, both ghost zones land in sparse regions. For
K = 3 the PCA axis runs through two of the three peaks; the third peak
still receives heavy pixel density and the off-axis ghost zones are nearly
empty. Even my pessimistic K=3 prediction in §"Path A" above was wrong by
an order of magnitude — non-colinear K=3 also gets a clean fix.

Practical implication: a `RectangularRotatedAdaptImage` subclass adding
brightness-weighted PCA rotation in front of the existing CDF would
materially fix the multi-modal-source case for all realistic K. The cost
is a 2×2 matrix multiply on the input grid plus a rotation-back step on
mesh-geometry outputs — JAX-trivial.
