"""
Ghost-peak diagnostic for the separable per-axis CDF in
``adaptive_rectangular_*`` (rectangular_spline.py).

Hypothesis under test
---------------------
A separable per-axis CDF cannot resolve a multi-modal adapt image (e.g. a
clumpy source with K bright peaks) without producing K**2 hot zones in the
warped mesh — K**2 - K of which are "ghost peaks" at the cross-product of
marginal modes where no source brightness exists.

Method
------
1. Sample N traced points uniformly across the source plane (proxy for a
   uniformly back-projected image-plane mask — no lensing physics needed
   for the geometric diagnostic).
2. Build a synthetic adapt image as a sum of K isotropic Gaussian peaks of
   matched amplitude.
3. Compute ``mesh_weight_map`` exactly as ``RectangularAdaptImage`` does:
   ``clip(b, eps, None) ** power`` then normalise.
4. Build the spline CDF transforms via ``create_transforms_spline`` (the same
   function ``RectangularSplineAdaptImage`` uses).
5. Evaluate the inverse CDF on the rectangular unit-square edge grid to
   recover the warped mesh edges in source-plane coordinates. Because the
   CDF is separable, the warped 2D mesh is the outer product of warped y
   and warped x edges.
6. Count pixel centres (mid-edges) inside each of the K**2 candidate hot
   zones (each peak location and each ghost cross-term).
7. Report the per-zone pixel-density ratios.

Outputs
-------
- ``files/ghost_peak.png`` — overlay of synthetic brightness, traced points,
  and the warped mesh edges, for K = 1, 2, 3 peaks side by side.
- console log: per-zone pixel-density ratios and ghost-to-real ratio.

Run inside the worktree with ``activate.sh`` sourced::

    source ~/Code/PyAutoLabs-wt/rectangular-adapt-cdf/activate.sh
    python PyAutoArray/files/ghost_peak_experiment.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoarray.inversion.mesh.interpolator.rectangular_spline import (
    SPLINE_CDF_DEFAULT_DEG,
    create_transforms_spline,
)


# ---------------------------------------------------------------------------
# Synthetic source
# ---------------------------------------------------------------------------


def sample_traced_points(n: int, seed: int = 0) -> np.ndarray:
    """N points uniformly in [-1, 1] x [-1, 1]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 2))


def brightness_at(points: np.ndarray, peaks, sigma: float = 0.1) -> np.ndarray:
    """Sum of isotropic Gaussian peaks, amplitude 1 each. ``peaks`` is
    a list of (y, x) tuples in the source-plane coordinate system."""
    b = np.zeros(points.shape[0])
    for (py, px) in peaks:
        dy = points[:, 0] - py
        dx = points[:, 1] - px
        b += np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))
    return b


# ---------------------------------------------------------------------------
# Run the existing pipeline geometry
# ---------------------------------------------------------------------------


def warped_mesh_edges(
    traced: np.ndarray,
    mesh_weight_map: np.ndarray,
    mesh_size: int,
    deg: int = SPLINE_CDF_DEFAULT_DEG,
):
    """Replicates the geometry path used by the live mesh.

    Mirrors ``adaptive_rectangular_transformed_grid_from_spline`` —
    same (mu, scale) normalisation, same per-axis spline CDF.

    Returns (warped_y_edges, warped_x_edges), each shape (mesh_size + 1,),
    in source-plane coordinates.
    """
    mu = traced.mean(axis=0)
    scale = traced.std(axis=0).min()
    traced_scaled = (traced - mu) / scale

    _, inv_transform = create_transforms_spline(
        traced_scaled, mesh_weight_map=mesh_weight_map, deg=deg, xp=np
    )

    # Per-axis unit-square edges — same as MeshGeometryRectangular.edges_transformed
    edges_y_unit = np.linspace(1.0, 0.0, mesh_size + 1)
    edges_x_unit = np.linspace(0.0, 1.0, mesh_size + 1)

    # The inverse transform takes (K, 2) -> (K, 2). Each column is processed
    # independently (column 0 = y, column 1 = x). To recover the warped 2D
    # mesh under the separability assumption, we feed the per-axis edges as
    # rows: row i is (y_unit[i], x_unit[i]) and we take col 0 of the output
    # for the warped-y edges and col 1 for the warped-x edges.
    grid = np.stack([edges_y_unit, edges_x_unit], axis=1)
    warped = inv_transform(grid) * scale + mu  # (mesh_size + 1, 2)

    return warped[:, 0], warped[:, 1]


# ---------------------------------------------------------------------------
# Quantify per-zone pixel density
# ---------------------------------------------------------------------------


def pixel_centres(warped_y_edges, warped_x_edges):
    """Cell centres are the outer product of consecutive-edge midpoints."""
    y_centres = 0.5 * (warped_y_edges[:-1] + warped_y_edges[1:])
    x_centres = 0.5 * (warped_x_edges[:-1] + warped_x_edges[1:])
    yy, xx = np.meshgrid(y_centres, x_centres, indexing="ij")
    return np.stack([yy.ravel(), xx.ravel()], axis=1)


def count_in_disc(centres, py, px, radius):
    d2 = (centres[:, 0] - py) ** 2 + (centres[:, 1] - px) ** 2
    return int(np.sum(d2 < radius ** 2))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_experiment(peaks_label, peaks, real_zones, ghost_zones,
                   mesh_size=40, n_traced=2000, sigma=0.1, deg=11,
                   probe_radius=0.15):
    traced = sample_traced_points(n_traced, seed=0)
    brightness = brightness_at(traced, peaks=peaks, sigma=sigma)
    # Mimic RectangularAdaptImage.mesh_weight_map_from with power=1, floor=0
    weight_map = np.clip(brightness, 1e-12, None)
    weight_map = weight_map / weight_map.sum()

    y_edges, x_edges = warped_mesh_edges(
        traced, weight_map, mesh_size=mesh_size, deg=deg
    )
    centres = pixel_centres(y_edges, x_edges)

    # Background pixel density per unit area (reference baseline)
    bbox_area = (y_edges.max() - y_edges.min()) * (x_edges.max() - x_edges.min())
    bg_density = centres.shape[0] / bbox_area
    probe_area = np.pi * probe_radius ** 2

    print(f"\n=== {peaks_label}: K = {len(peaks)} peaks ===")
    print(
        f"Total pixel centres: {centres.shape[0]}, "
        f"bbox area: {bbox_area:.3f}, baseline density: {bg_density:.2f} px/unit²"
    )

    real_counts = []
    for (py, px) in real_zones:
        n = count_in_disc(centres, py, px, probe_radius)
        d = n / probe_area
        ratio = d / bg_density
        real_counts.append(n)
        print(
            f"  REAL  peak at ({py:+.2f}, {px:+.2f}): "
            f"{n:3d} centres ({d:6.1f} px/unit², {ratio:.2f}x baseline)"
        )

    ghost_counts = []
    for (py, px) in ghost_zones:
        n = count_in_disc(centres, py, px, probe_radius)
        d = n / probe_area
        ratio = d / bg_density
        ghost_counts.append(n)
        print(
            f"  GHOST  cross at ({py:+.2f}, {px:+.2f}): "
            f"{n:3d} centres ({d:6.1f} px/unit², {ratio:.2f}x baseline)"
        )

    if real_counts and ghost_counts:
        mean_real = np.mean(real_counts)
        mean_ghost = np.mean(ghost_counts)
        ghost_to_real = mean_ghost / max(mean_real, 1e-12)
        print(
            f"  mean(real) = {mean_real:.1f}, mean(ghost) = {mean_ghost:.1f}, "
            f"ghost / real = {ghost_to_real:.2f}"
        )
        verdict = (
            "CONFIRMED ghost-peak failure (ghost density ≈ real density)"
            if ghost_to_real > 0.5
            else "no ghost-peak failure (real ≫ ghost)"
        )
        print(f"  verdict: {verdict}")

    return traced, brightness, weight_map, y_edges, x_edges, centres


def plot_experiment(ax, label, traced, brightness, weight_map, y_edges, x_edges,
                    centres, real_zones, ghost_zones):
    # Source brightness — scatter coloured by adapt image
    sc = ax.scatter(
        traced[:, 1], traced[:, 0], c=brightness, cmap="hot",
        s=4, alpha=0.45, marker="."
    )

    # Warped mesh as gridlines
    for ye in y_edges:
        ax.plot([x_edges.min(), x_edges.max()], [ye, ye], color="cyan",
                lw=0.25, alpha=0.5)
    for xe in x_edges:
        ax.plot([xe, xe], [y_edges.min(), y_edges.max()], color="cyan",
                lw=0.25, alpha=0.5)

    # Pixel centres
    ax.scatter(centres[:, 1], centres[:, 0], c="white", s=2,
               edgecolors="black", linewidths=0.2)

    # Real zones in green, ghost zones in red
    for (py, px) in real_zones:
        ax.plot(px, py, "o", color="lime", markersize=12,
                markeredgecolor="black", markeredgewidth=1.5)
    for (py, px) in ghost_zones:
        ax.plot(px, py, "x", color="red", markersize=12, markeredgewidth=2)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(label)
    ax.set_xlabel("source x")
    ax.set_ylabel("source y")
    return sc


def main():
    # K = 1 control (no separability problem possible)
    peaks_1 = [(0.0, 0.0)]
    real_1 = [(0.0, 0.0)]
    ghost_1 = []  # no cross-terms with K = 1

    # K = 2 diagonal — the textbook ghost-peak case
    peaks_2 = [(-0.5, -0.5), (0.5, 0.5)]
    real_2 = [(-0.5, -0.5), (0.5, 0.5)]
    ghost_2 = [(-0.5, 0.5), (0.5, -0.5)]

    # K = 3 — 9 candidate zones (3 real, 6 ghost)
    peaks_3 = [(-0.5, -0.5), (0.0, 0.5), (0.5, -0.2)]
    real_3 = list(peaks_3)
    ghost_3 = [
        (py, px) for py in (-0.5, 0.0, 0.5) for px in (-0.5, 0.5, -0.2)
        if (py, px) not in real_3
    ]

    cases = [
        ("K=1 (control)", peaks_1, real_1, ghost_1),
        ("K=2 (diagonal)", peaks_2, real_2, ghost_2),
        ("K=3 (offset)", peaks_3, real_3, ghost_3),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    last_sc = None
    for ax, (label, peaks, real, ghost) in zip(axes, cases):
        result = run_experiment(label, peaks, real, ghost)
        last_sc = plot_experiment(ax, label, *result,
                                  real_zones=real, ghost_zones=ghost)

    fig.colorbar(last_sc, ax=axes, label="brightness (adapt image)",
                 fraction=0.025, pad=0.02)
    out = os.path.join(os.path.dirname(__file__), "ghost_peak.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure written: {out}")


if __name__ == "__main__":
    sys.exit(main())
