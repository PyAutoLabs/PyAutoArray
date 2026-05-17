"""
Path-A diagnostic: does a PCA-aligned rotation of the source plane eliminate
the ghost-peak failure observed in ``ghost_peak_experiment.py``?

Hypothesis
----------
The separable per-axis CDF concentrates pixels at the outer product of
marginal modes. If a multi-modal source's peaks are aligned with the y/x
axes, there are no off-axis "ghost" combinations — all marginal modes line
up on the real peaks. PCA on the brightness-weighted point cloud finds the
principal axis; rotating the source plane into that frame makes K=2 peaks
exactly axis-aligned (and helps for higher K).

Method
------
1. Reuse the ghost-peak setup: synthetic 2-peak Gaussian brightness on
   uniformly sampled traced points.
2. Compute PCA on traced points weighted by brightness. Rotate everything
   into the principal frame.
3. Run the same spline-CDF + outer-product mesh as the baseline.
4. Quantify pixel density at the real peaks and the (now potentially
   collapsed) ghost positions, in the rotated frame.
5. Visualise rotated-frame warped mesh AND source-frame warped mesh
   (rotated back).

Run with::

    source ~/Code/PyAutoLabs-wt/rectangular-adapt-cdf/activate.sh
    python PyAutoArray/files/pca_rotation_experiment.py
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
# Synthetic source (identical to ghost_peak_experiment.py)
# ---------------------------------------------------------------------------


def sample_traced_points(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n, 2))


def brightness_at(points: np.ndarray, peaks, sigma: float = 0.1) -> np.ndarray:
    b = np.zeros(points.shape[0])
    for (py, px) in peaks:
        dy = points[:, 0] - py
        dx = points[:, 1] - px
        b += np.exp(-(dy ** 2 + dx ** 2) / (2 * sigma ** 2))
    return b


# ---------------------------------------------------------------------------
# Brightness-weighted PCA → rotation matrix
# ---------------------------------------------------------------------------


def pca_rotation_matrix(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return a 2x2 rotation matrix R such that R @ (point - centroid) is in
    the principal-axis frame: column 0 = direction of max brightness-weighted
    variance, column 1 = perpendicular.

    The matrix is constructed from the eigenvectors of the weighted covariance,
    so applying it to (y, x) row vectors (via ``points @ R.T``) rotates them
    into the PCA frame.
    """
    w = weights / weights.sum()
    centroid = (w[:, None] * points).sum(axis=0)
    centred = points - centroid
    cov = (w[:, None, None] * centred[:, :, None] * centred[:, None, :]).sum(axis=0)

    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending eigenvalues; we want the largest first.
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    # Make rotation deterministic by enforcing a sign convention.
    if eigvecs[0, 0] < 0:
        eigvecs[:, 0] = -eigvecs[:, 0]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 1] = -eigvecs[:, 1]
    # Rows of R are the new basis vectors (so R @ x rotates x into PCA frame).
    R = eigvecs.T
    return R, centroid


def rotate_points(points: np.ndarray, R: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return (points - centroid) @ R.T


def unrotate_points(points: np.ndarray, R: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    return points @ R + centroid


# ---------------------------------------------------------------------------
# Warped mesh edges in whatever frame the inputs live in
# ---------------------------------------------------------------------------


def warped_mesh_edges(traced, mesh_weight_map, mesh_size, deg=SPLINE_CDF_DEFAULT_DEG):
    mu = traced.mean(axis=0)
    scale = traced.std(axis=0).min()
    traced_scaled = (traced - mu) / scale

    _, inv_transform = create_transforms_spline(
        traced_scaled, mesh_weight_map=mesh_weight_map, deg=deg, xp=np
    )

    edges_y_unit = np.linspace(1.0, 0.0, mesh_size + 1)
    edges_x_unit = np.linspace(0.0, 1.0, mesh_size + 1)
    grid = np.stack([edges_y_unit, edges_x_unit], axis=1)
    warped = inv_transform(grid) * scale + mu

    return warped[:, 0], warped[:, 1]


def pixel_centres_from_edges(y_edges, x_edges):
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yy, xx = np.meshgrid(yc, xc, indexing="ij")
    return np.stack([yy.ravel(), xx.ravel()], axis=1)


def count_in_disc(centres, py, px, radius):
    d2 = (centres[:, 0] - py) ** 2 + (centres[:, 1] - px) ** 2
    return int(np.sum(d2 < radius ** 2))


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_one(label, peaks, mesh_size=40, n_traced=2000, sigma=0.1,
            deg=11, probe_radius=0.15):
    traced = sample_traced_points(n_traced, seed=0)
    brightness = brightness_at(traced, peaks, sigma=sigma)
    weight_map = np.clip(brightness, 1e-12, None)
    weight_map = weight_map / weight_map.sum()

    # ---- Baseline (no rotation) ----
    yE_b, xE_b = warped_mesh_edges(traced, weight_map, mesh_size, deg)
    centres_b = pixel_centres_from_edges(yE_b, xE_b)

    real_zones = list(peaks)
    ghost_zones = [(py, px) for py, _ in peaks for _, px in peaks
                   if (py, px) not in real_zones]

    real_b = [count_in_disc(centres_b, py, px, probe_radius) for py, px in real_zones]
    ghost_b = [count_in_disc(centres_b, py, px, probe_radius) for py, px in ghost_zones]

    # ---- PCA-rotated ----
    R, centroid = pca_rotation_matrix(traced, brightness)
    traced_rot = rotate_points(traced, R, centroid)
    yE_r, xE_r = warped_mesh_edges(traced_rot, weight_map, mesh_size, deg)
    centres_r_rotframe = pixel_centres_from_edges(yE_r, xE_r)
    centres_r_sourceframe = unrotate_points(centres_r_rotframe, R, centroid)

    real_r = [count_in_disc(centres_r_sourceframe, py, px, probe_radius)
              for py, px in real_zones]
    ghost_r = [count_in_disc(centres_r_sourceframe, py, px, probe_radius)
               for py, px in ghost_zones]

    mean_real_b = np.mean(real_b) if real_b else 0.0
    mean_ghost_b = np.mean(ghost_b) if ghost_b else 0.0
    mean_real_r = np.mean(real_r) if real_r else 0.0
    mean_ghost_r = np.mean(ghost_r) if ghost_r else 0.0

    print(f"\n=== {label} ===")
    angle_deg = np.degrees(np.arctan2(R[0, 1], R[0, 0]))
    print(f"  PCA principal-axis angle: {angle_deg:+.2f}°")
    print(f"  baseline:   mean(real) = {mean_real_b:.1f}, "
          f"mean(ghost) = {mean_ghost_b:.1f}, "
          f"ghost/real = {mean_ghost_b / max(mean_real_b, 1e-12):.2f}")
    print(f"  rotated:    mean(real) = {mean_real_r:.1f}, "
          f"mean(ghost) = {mean_ghost_r:.1f}, "
          f"ghost/real = {mean_ghost_r / max(mean_real_r, 1e-12):.2f}")

    return {
        "label": label,
        "traced": traced,
        "brightness": brightness,
        "peaks": peaks,
        "real_zones": real_zones,
        "ghost_zones": ghost_zones,
        "yE_b": yE_b, "xE_b": xE_b, "centres_b": centres_b,
        "R": R, "centroid": centroid,
        "yE_r": yE_r, "xE_r": xE_r,
        "centres_r_rotframe": centres_r_rotframe,
        "centres_r_sourceframe": centres_r_sourceframe,
        "ratios": (mean_ghost_b / max(mean_real_b, 1e-12),
                   mean_ghost_r / max(mean_real_r, 1e-12)),
    }


def plot_overlay(ax, traced, brightness, centres, real_zones, ghost_zones,
                 grid_lines=None, title=""):
    sc = ax.scatter(traced[:, 1], traced[:, 0], c=brightness, cmap="hot",
                    s=4, alpha=0.45, marker=".")
    if grid_lines is not None:
        ys_min, ys_max, xs_min, xs_max = grid_lines["bbox"]
        for ye in grid_lines["y_edges"]:
            ax.plot([xs_min, xs_max], [ye, ye], color="cyan", lw=0.25, alpha=0.5)
        for xe in grid_lines["x_edges"]:
            ax.plot([xe, xe], [ys_min, ys_max], color="cyan", lw=0.25, alpha=0.5)
    ax.scatter(centres[:, 1], centres[:, 0], c="white", s=2,
               edgecolors="black", linewidths=0.2)
    for (py, px) in real_zones:
        ax.plot(px, py, "o", color="lime", markersize=12,
                markeredgecolor="black", markeredgewidth=1.5)
    for (py, px) in ghost_zones:
        ax.plot(px, py, "x", color="red", markersize=12, markeredgewidth=2)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("source x")
    ax.set_ylabel("source y")
    return sc


def main():
    # Three test cases reusing the ghost-peak setup
    cases = [
        ("K=2 diagonal", [(-0.5, -0.5), (0.5, 0.5)]),
        ("K=2 axis-aligned (control)", [(0.0, -0.5), (0.0, 0.5)]),
        ("K=3 offset", [(-0.5, -0.5), (0.0, 0.5), (0.5, -0.2)]),
    ]

    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 5.0 * len(cases)))
    last_sc = None

    for row, (label, peaks) in zip(axes, cases):
        out = run_one(label, peaks)

        # Left: baseline (unrotated)
        last_sc = plot_overlay(
            row[0], out["traced"], out["brightness"], out["centres_b"],
            out["real_zones"], out["ghost_zones"],
            grid_lines={
                "y_edges": out["yE_b"], "x_edges": out["xE_b"],
                "bbox": (out["yE_b"].min(), out["yE_b"].max(),
                         out["xE_b"].min(), out["xE_b"].max()),
            },
            title=f"{label} — baseline\nghost/real = {out['ratios'][0]:.2f}",
        )

        # Right: PCA-rotated mesh, drawn in source-plane frame.
        # The mesh edges are in the rotated frame, so we have to construct the
        # full set of cell-corner points there and unrotate them all for the
        # axis-aligned plot. For visualisation we just plot the unrotated
        # pixel centres + the rotated bounding box, leaving the grid lines
        # implicit (drawing rotated lines via plain plot() means many segments).
        last_sc = plot_overlay(
            row[1], out["traced"], out["brightness"],
            out["centres_r_sourceframe"],
            out["real_zones"], out["ghost_zones"],
            grid_lines=None,
            title=f"{label} — PCA-rotated\nghost/real = {out['ratios'][1]:.2f}",
        )

        # Draw the rotated grid as line segments transformed to source frame.
        yE_r, xE_r = out["yE_r"], out["xE_r"]
        R, centroid = out["R"], out["centroid"]
        # Horizontal lines in rotated frame: y = yE_r[i], x in [xE_r.min, xE_r.max]
        for ye in yE_r:
            pts_rot = np.array([[ye, xE_r.min()], [ye, xE_r.max()]])
            pts_src = unrotate_points(pts_rot, R, centroid)
            row[1].plot(pts_src[:, 1], pts_src[:, 0], color="cyan",
                        lw=0.25, alpha=0.5)
        # Vertical lines in rotated frame: x = xE_r[j], y in [yE_r.min, yE_r.max]
        for xe in xE_r:
            pts_rot = np.array([[yE_r.min(), xe], [yE_r.max(), xe]])
            pts_src = unrotate_points(pts_rot, R, centroid)
            row[1].plot(pts_src[:, 1], pts_src[:, 0], color="cyan",
                        lw=0.25, alpha=0.5)

    fig.colorbar(last_sc, ax=axes, label="brightness",
                 fraction=0.025, pad=0.02)
    out_path = os.path.join(os.path.dirname(__file__), "pca_rotation.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nfigure written: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
