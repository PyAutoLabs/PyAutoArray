"""Unit tests for the brightness-weighted PCA-rotated adaptive rectangular mesh.

Numpy-only per the project rule (no JAX in library unit tests). Cross-xp
parity is left to workspace_test.
"""
import numpy as np
import pytest

import autoarray as aa
from autoarray.inversion.mesh.interpolator.rectangular_rotated_spline import (
    InterpolatorRectangularRotatedSpline,
)
from autoarray.inversion.mesh.mesh.rectangular_rotated_adapt_image import (
    RectangularRotatedAdaptImage,
    _GridLike,
    _RotatedGridShim,
    _pca_rotation_from,
)
from autoarray.inversion.mesh.mesh_geometry.rectangular_rotated import (
    MeshGeometryRectangularRotated,
)


# ---------------------------------------------------------------------------
# _pca_rotation_from
# ---------------------------------------------------------------------------


def test__pca_rotation__axis_aligned_brightness__principal_axis_is_x():
    """Two brightness peaks at (0, ±0.5) — variance lives along source x,
    so the principal axis must be x. R's first row (the principal-axis
    direction in rotated coords) should point along source-x (i.e.
    ``±[0, 1]``), the second row along source-y (``±[1, 0]``).
    """
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(2000, 2))
    sigma = 0.1
    b = np.exp(-(pts[:, 0] ** 2 + (pts[:, 1] - 0.5) ** 2) / (2 * sigma ** 2))
    b += np.exp(-(pts[:, 0] ** 2 + (pts[:, 1] + 0.5) ** 2) / (2 * sigma ** 2))

    R, centroid = _pca_rotation_from(pts, b, xp=np)

    assert R.shape == (2, 2)
    assert centroid.shape == (2,)
    np.testing.assert_allclose(centroid, [0.0, 0.0], atol=0.1)
    # Principal axis aligns with source-x: |R[0, 1]| ≈ 1, |R[0, 0]| ≈ 0.
    np.testing.assert_allclose(np.abs(R[0]), [0.0, 1.0], atol=0.05)
    np.testing.assert_allclose(np.abs(R[1]), [1.0, 0.0], atol=0.05)
    # Determinant +1 (proper rotation, no reflection)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test__pca_rotation__diagonal_brightness__rotation_is_45_degrees():
    """Brightness peaks at (-0.5, -0.5) and (+0.5, +0.5) → principal axis
    runs along the +x/+y diagonal → R rotates by ~45°.
    """
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1.0, 1.0, size=(2000, 2))
    sigma = 0.1
    b = np.exp(-((pts[:, 0] + 0.5) ** 2 + (pts[:, 1] + 0.5) ** 2) / (2 * sigma ** 2))
    b += np.exp(-((pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.5) ** 2) / (2 * sigma ** 2))

    R, _ = _pca_rotation_from(pts, b, xp=np)

    angle_deg = abs(np.degrees(np.arctan2(R[0, 1], R[0, 0])))
    # Angle is +45° within ~3° tolerance from random sampling noise.
    assert abs(angle_deg - 45.0) < 3.0, (
        f"expected ~45° rotation, got {angle_deg:.2f}°"
    )
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test__pca_rotation__round_trip__source_to_rotated_back_to_source():
    """Rotating into the PCA frame and back must reconstruct the original
    coordinates exactly (rotation is orthogonal).
    """
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1.0, 1.0, size=(500, 2))
    b = rng.uniform(0.1, 1.0, size=500)

    R, centroid = _pca_rotation_from(pts, b, xp=np)
    rotated = (pts - centroid) @ R.T
    back = rotated @ R + centroid

    np.testing.assert_allclose(back, pts, atol=1e-10)


# ---------------------------------------------------------------------------
# _GridLike + _RotatedGridShim duck-typing
# ---------------------------------------------------------------------------


def test__grid_like__exposes_array_and_basic_ndarray_ops():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    g = _GridLike(arr)

    np.testing.assert_array_equal(g.array, arr)
    np.testing.assert_allclose(g.mean(axis=0), [3.0, 4.0])
    np.testing.assert_allclose(g.std(axis=0), arr.std(axis=0))
    np.testing.assert_array_equal(g[1], np.array([3.0, 4.0]))
    np.testing.assert_array_equal(g - 1.0, arr - 1.0)
    np.testing.assert_array_equal(g / 2.0, arr / 2.0)
    assert g.shape == (3, 2)


class _OriginalGridStub:
    """Tiny stand-in for the un-rotated Grid2D the shim forwards mask /
    over_sampler from. Tests that don't run the full inversion can use
    sentinel values."""

    def __init__(self, mask="_mask_sentinel", over_sampler="_oversampler_sentinel"):
        self.mask = mask
        self.over_sampler = over_sampler


def test__rotated_grid_shim__exposes_array_and_over_sampled():
    base = np.array([[0.0, 0.0], [1.0, 1.0]])
    over = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    shim = _RotatedGridShim(_OriginalGridStub(), base, over)

    np.testing.assert_array_equal(shim.array, base)
    np.testing.assert_array_equal(shim.over_sampled.array, over)
    # mean / std on shim itself work (matches Grid2D-like behaviour)
    np.testing.assert_allclose(shim.mean(axis=0), [0.5, 0.5])
    # Mask + over_sampler forwarded unchanged
    assert shim.mask == "_mask_sentinel"
    assert shim.over_sampler == "_oversampler_sentinel"


# ---------------------------------------------------------------------------
# Construction + class wiring
# ---------------------------------------------------------------------------


def test__construction__inherits_parent_defaults():
    mesh = RectangularRotatedAdaptImage(shape=(7, 9))
    assert mesh.shape == (7, 9)
    assert mesh.weight_power == 1.0
    assert mesh.weight_floor == 0.0
    # Inherits SPLINE_CDF_DEFAULT_DEG
    from autoarray.inversion.mesh.interpolator.rectangular_spline import (
        SPLINE_CDF_DEFAULT_DEG,
    )
    assert mesh.spline_deg == SPLINE_CDF_DEFAULT_DEG


def test__construction__custom_parameters_pass_through():
    mesh = RectangularRotatedAdaptImage(
        shape=(5, 5), weight_power=2.0, weight_floor=0.05, spline_deg=9
    )
    assert mesh.weight_power == 2.0
    assert mesh.weight_floor == 0.05
    assert mesh.spline_deg == 9


def test__interpolator_cls__is_rotated_spline():
    mesh = RectangularRotatedAdaptImage(shape=(5, 5))
    assert mesh.interpolator_cls is InterpolatorRectangularRotatedSpline


def test__mesh_export__via_aa_dot_mesh():
    """Re-export through aa.mesh works (the __init__.py edit took)."""
    assert hasattr(aa.mesh, "RectangularRotatedAdaptImage")
    assert aa.mesh.RectangularRotatedAdaptImage is RectangularRotatedAdaptImage


# ---------------------------------------------------------------------------
# InterpolatorRectangularRotatedSpline._mappings_sizes_weights via shim
# ---------------------------------------------------------------------------


def test__rotated_interpolator__mappings_sizes_weights__shapes_and_norms():
    rng = np.random.default_rng(5)
    base = rng.standard_normal((64, 2))
    over = rng.standard_normal((100, 2)) * 0.9
    weights = rng.uniform(0.1, 1.0, size=64)
    weights = weights / weights.sum()

    shim = _RotatedGridShim(_OriginalGridStub(), base, over)

    # Identity rotation — the rotated interpolator should produce identical
    # mappings to the non-rotated case for this trivial choice.
    R = np.eye(2)
    centroid = np.zeros(2)

    mesh = RectangularRotatedAdaptImage(shape=(6, 6), spline_deg=7)
    interpolator = InterpolatorRectangularRotatedSpline(
        mesh=mesh,
        mesh_grid=aa.Grid2DIrregular(np.zeros((36, 2))),
        data_grid=shim,
        mesh_weight_map=weights,
        spline_deg=7,
        rotation_matrix=R,
        rotation_centroid=centroid,
        xp=np,
    )

    mappings, sizes, w = interpolator._mappings_sizes_weights
    # The bilinear scatter is evaluated on the over_sampled grid -> 100 rows.
    assert mappings.shape == (100, 4)
    assert sizes.shape == (100,)
    assert w.shape == (100, 4)
    assert np.all(sizes == 4)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-10)


def test__rotated_geometry__exposes_rotation_state_for_consumers():
    """The rotated mesh geometry must expose ``rotation_matrix`` and
    ``rotation_centroid`` so a rotation-aware plotter or analysis can
    convert per-axis edges back to source-frame 2D corners.
    """
    rng = np.random.default_rng(6)
    pts = rng.uniform(-1.0, 1.0, size=(300, 2))
    b = np.exp(-((pts[:, 0]) ** 2 + (pts[:, 1] / 0.4) ** 2) / 2)

    R, centroid = _pca_rotation_from(pts, b, xp=np)
    rotated = (pts - centroid) @ R.T
    shim = _RotatedGridShim(_OriginalGridStub(), rotated, rotated)

    mesh = RectangularRotatedAdaptImage(shape=(8, 8), spline_deg=7)
    interpolator = InterpolatorRectangularRotatedSpline(
        mesh=mesh,
        mesh_grid=aa.Grid2DIrregular(np.zeros((64, 2))),
        data_grid=shim,
        mesh_weight_map=None,
        spline_deg=7,
        rotation_matrix=R,
        rotation_centroid=centroid,
        xp=np,
    )

    geom = interpolator.mesh_geometry
    np.testing.assert_array_equal(geom.rotation_matrix, R)
    np.testing.assert_array_equal(geom.rotation_centroid, centroid)
    # And the per-axis edges still come out (in rotated frame) with the
    # parent's shape contract.
    assert geom.edges_transformed.shape == (9, 2)


# ---------------------------------------------------------------------------
# End-to-end: ghost-peak fix on a real Grid2D
# ---------------------------------------------------------------------------


def _build_two_peak_setup(peaks, sigma=0.1, shape_native=(50, 50), pixel_scales=0.04,
                          jitter_sigma=0.003, seed=0):
    """Real Grid2D + Array2D adapt-image with K Gaussian peaks.

    Adds a small random jitter to grid coordinates to break the lattice
    degeneracy. Without jitter, ``Grid2D.from_mask`` gives points with
    only ``shape_native[0]`` distinct y-values (each repeated ``shape_native[1]``
    times), which destabilises the spline-CDF's polynomial fit on the
    empirical CDF (pre-existing issue, see ``files/cdf_audit.md``). Real
    ray-traced grids in the lens pipeline are already scrambled by lensing
    and don't hit this — the jitter is a proxy.
    """
    mask = aa.Mask2D.all_false(
        shape_native=shape_native, pixel_scales=pixel_scales
    )
    grid_lattice = aa.Grid2D.from_mask(mask=mask)
    rng = np.random.default_rng(seed)
    jittered_values = grid_lattice.array + rng.normal(
        0.0, jitter_sigma, size=grid_lattice.array.shape
    )
    grid = aa.Grid2D(values=jittered_values, mask=mask, over_sample_size=1)
    pts = grid.array

    b = np.zeros(pts.shape[0])
    for (py, px) in peaks:
        b += np.exp(-((pts[:, 0] - py) ** 2 + (pts[:, 1] - px) ** 2) / (2 * sigma ** 2))

    adapt = aa.Array2D(values=b, mask=mask)
    return grid, adapt


def _count_in_disc(centres, py, px, radius):
    d2 = (centres[:, 0] - py) ** 2 + (centres[:, 1] - px) ** 2
    return int(np.sum(d2 < radius ** 2))


def _warped_centres_source_frame(interpolator):
    """Reconstruct the CDF-warped pixel-centre 2D grid in source frame from
    the rotated geometry.

    Mirrors the rotation-aware path that any source-frame plotter must do:
    per-axis warped edges (in rotated frame) → mid-points → 2D outer product
    → rotate back via geometry's rotation_matrix and rotation_centroid.
    """
    geom = interpolator.mesh_geometry
    edges = geom.edges_transformed
    y_edges_rot, x_edges_rot = edges[:, 0], edges[:, 1]
    y_centres_rot = 0.5 * (y_edges_rot[:-1] + y_edges_rot[1:])
    x_centres_rot = 0.5 * (x_edges_rot[:-1] + x_edges_rot[1:])
    yy, xx = np.meshgrid(y_centres_rot, x_centres_rot, indexing="ij")
    centres_rot = np.stack([yy.ravel(), xx.ravel()], axis=1)
    return centres_rot @ geom.rotation_matrix + geom.rotation_centroid


def test__end_to_end__k2_diagonal__rotated_mesh_eliminates_ghosts():
    """Production-class proof: with a K=2 diagonal multi-modal adapt image,
    RectangularRotatedAdaptImage places CDF-warped pixel centres at the
    real source peaks with negligible density at the ghost cross-products.

    Mirrors the ``pca_rotation_experiment.py`` K=2 case but exercises the
    real mesh + interpolator + mesh_geometry pipeline rather than the
    helper functions directly.
    """
    peaks = [(-0.5, -0.5), (0.5, 0.5)]
    grid, adapt = _build_two_peak_setup(peaks, sigma=0.1)

    mesh = RectangularRotatedAdaptImage(shape=(40, 40))
    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=None,
        border_relocator=None,
        adapt_data=adapt,
    )

    centres = _warped_centres_source_frame(interpolator)
    assert centres.shape == (1600, 2)

    real_density = np.mean([
        _count_in_disc(centres, -0.5, -0.5, 0.15),
        _count_in_disc(centres, 0.5, 0.5, 0.15),
    ])
    ghost_density = np.mean([
        _count_in_disc(centres, -0.5, 0.5, 0.15),
        _count_in_disc(centres, 0.5, -0.5, 0.15),
    ])

    # On the experiment, the rotated case gave ghost/real = 0.00. Threshold
    # generous to keep the test stable across small RNG / masking changes.
    assert real_density > 50, f"real-peak density too low: {real_density}"
    ratio = ghost_density / max(real_density, 1.0)
    assert ratio < 0.15, (
        f"ghost-peak density should be near-zero; got "
        f"real={real_density}, ghost={ghost_density}, ratio={ratio:.2f}"
    )


def test__end_to_end__k2_diagonal__rotated_beats_unrotated_baseline():
    """Sanity check: the unrotated parent class should produce significant
    ghost density on the same K=2 setup; the rotated subclass should not.
    Concrete regression check that the override is doing real work.
    """
    peaks = [(-0.5, -0.5), (0.5, 0.5)]
    grid, adapt = _build_two_peak_setup(peaks, sigma=0.1)

    # Unrotated baseline
    parent_mesh = aa.mesh.RectangularSplineAdaptImage(shape=(40, 40))
    parent_interp = parent_mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=None,
        border_relocator=None,
        adapt_data=adapt,
    )
    edges = parent_interp.mesh_geometry.edges_transformed
    y_edges, x_edges = edges[:, 0], edges[:, 1]
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    yy, xx = np.meshgrid(y_centres, x_centres, indexing="ij")
    parent_centres = np.stack([yy.ravel(), xx.ravel()], axis=1)

    parent_ghost = np.mean([
        _count_in_disc(parent_centres, -0.5, 0.5, 0.15),
        _count_in_disc(parent_centres, 0.5, -0.5, 0.15),
    ])
    parent_real = np.mean([
        _count_in_disc(parent_centres, -0.5, -0.5, 0.15),
        _count_in_disc(parent_centres, 0.5, 0.5, 0.15),
    ])
    parent_ratio = parent_ghost / max(parent_real, 1.0)

    # Rotated subclass
    rot_mesh = RectangularRotatedAdaptImage(shape=(40, 40))
    rot_interp = rot_mesh.interpolator_from(
        source_plane_data_grid=grid,
        source_plane_mesh_grid=None,
        border_relocator=None,
        adapt_data=adapt,
    )
    rot_centres = _warped_centres_source_frame(rot_interp)
    rot_ghost = np.mean([
        _count_in_disc(rot_centres, -0.5, 0.5, 0.15),
        _count_in_disc(rot_centres, 0.5, -0.5, 0.15),
    ])
    rot_real = np.mean([
        _count_in_disc(rot_centres, -0.5, -0.5, 0.15),
        _count_in_disc(rot_centres, 0.5, 0.5, 0.15),
    ])
    rot_ratio = rot_ghost / max(rot_real, 1.0)

    # Parent: severe ghost-peak (~1.0); subclass: near-zero (~0.0).
    assert parent_ratio > 0.5, (
        f"parent baseline expected severe ghost-peak failure, got "
        f"ghost/real = {parent_ratio:.2f}"
    )
    assert rot_ratio < 0.15, (
        f"rotated subclass expected near-zero ghosts, got "
        f"ghost/real = {rot_ratio:.2f}"
    )


def test__end_to_end__rejects_missing_adapt_data():
    """Without adapt_data, the rotation can't be computed; we surface a
    clear error instead of silently falling back to identity.
    """
    grid, _ = _build_two_peak_setup([(0.0, 0.0)])
    mesh = RectangularRotatedAdaptImage(shape=(8, 8))

    with pytest.raises(ValueError, match="adapt_data"):
        mesh.interpolator_from(
            source_plane_data_grid=grid,
            source_plane_mesh_grid=None,
            border_relocator=None,
            adapt_data=None,
        )
