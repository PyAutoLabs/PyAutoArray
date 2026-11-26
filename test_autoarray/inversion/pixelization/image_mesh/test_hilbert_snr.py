import numpy as np
import pytest

import autoarray as aa


def _reference_probabilities(s, background=0.25, peak=0.05, peak_power=8.0, s_noise=2.0, s0=3.0, power=2.0):
    """
    The `peak_share` weighting of the adapt-mesh-weighting study (PyAutoArray#714,
    `Science/development/adapt_snr_cap/scripts/mesh_weighting.py`), written out independently.
    """
    s = np.clip(np.asarray(s, dtype=float), 0.0, None)
    n = s.size
    x = np.clip(s - s_noise, 0.0, None)
    soft = (x / (x + s0)) ** power
    peak_term = (s / s.max()) ** peak_power
    return background / n + peak * peak_term / peak_term.sum() + (1.0 - background - peak) * soft / soft.sum()


def test__soft_snr_from__values():
    snr = np.array([-5.0, 0.0, 2.0, 3.0, 6.5, 12.0, 40.0])

    soft = aa.image_mesh.soft_snr_from(snr=snr)

    assert soft == pytest.approx(
        [0.0, 0.0, 0.0, 1.0 / 16.0, 0.36, (10.0 / 13.0) ** 2, (38.0 / 41.0) ** 2], 1.0e-8
    )


def test__soft_snr_map_from__floor_and_blank():
    snr = np.array([0.0, 1.0, 3.0, 40.0])

    adapt = aa.image_mesh.soft_snr_map_from(snr=snr)
    peak = (38.0 / 41.0) ** 2

    assert adapt == pytest.approx([0.01 * peak, 0.01 * peak, 1.0 / 16.0, peak], 1.0e-8)

    assert aa.image_mesh.soft_snr_map_from(snr=np.array([0.0, 1.0, 2.0])) == pytest.approx(
        [1.0, 1.0, 1.0]
    )


def test__weight_map_from__matches_reference_and_sums_to_one():
    rng = np.random.default_rng(1)
    s = np.abs(rng.normal(0.0, 1.0, size=500))
    s[100:140] += np.linspace(3.0, 60.0, 40)

    image_mesh = aa.image_mesh.HilbertSNR(pixels=100)
    weight_map = image_mesh.weight_map_from(adapt_data=s)

    assert np.sum(weight_map) == pytest.approx(1.0, 1.0e-10)
    assert weight_map == pytest.approx(_reference_probabilities(s), 1.0e-10)


def test__weight_map_from__scale_free_of_the_peak():
    # The source term uses absolute S/N: adding a much brighter peak elsewhere does not change how the
    # source share is split among the other cells.
    s = np.zeros(200)
    s[20:60] = 10.0
    s_bright = s.copy()
    s_bright[150] = 1000.0

    image_mesh = aa.image_mesh.HilbertSNR(pixels=100)
    w = image_mesh.weight_map_from(adapt_data=s)
    w_bright = image_mesh.weight_map_from(adapt_data=s_bright)

    soft_10 = (8.0 / 11.0) ** 2
    soft_1000 = (998.0 / 1001.0) ** 2

    # Every S/N 10 cell keeps the same soft weight; the bright cell takes one cell's (saturated) share,
    # not the whole source budget as `(s / max(s)) ** 3.5` would.
    assert w_bright[20:60].sum() == pytest.approx(
        0.25 * 40 / 200 + 0.70 * 40 * soft_10 / (40 * soft_10 + soft_1000), 1.0e-6
    )
    assert w_bright[20:60].sum() > 0.7
    assert w[20:60].sum() == pytest.approx(0.25 * 40 / 200 + 0.75, 1.0e-6)


def test__weight_map_from__blank_adapt_data_is_uniform():
    image_mesh = aa.image_mesh.HilbertSNR(pixels=10)

    assert image_mesh.weight_map_from(adapt_data=np.zeros(50)) == pytest.approx(np.full(50, 0.02))
    assert image_mesh.weight_map_from(adapt_data=-np.ones(50)) == pytest.approx(np.full(50, 0.02))


def test__shares_validated():
    with pytest.raises(ValueError):
        aa.image_mesh.HilbertSNR(pixels=10, background=0.9, peak=0.2)


def _arc_plus_compact(peak):
    mask = aa.Mask2D.circular(shape_native=(60, 60), radius=2.8, pixel_scales=0.1)
    y, x = np.indices(mask.shape_native)
    y = (29.5 - y) * 0.1
    x = (x - 29.5) * 0.1
    radius = np.sqrt(x**2 + y**2)
    arc = 10.0 * np.exp(-0.5 * ((radius - 1.5) / 0.1) ** 2) * (x > 0.0)
    compact = peak * np.exp(-0.5 * ((y - 1.5) ** 2 + x**2) / 0.05**2)
    snr = arc + compact
    adapt_data = aa.Array2D(values=snr[~np.asarray(mask)], mask=mask)
    return mask, adapt_data, arc, snr


def _shares(mesh, mask, arc, snr):
    rows = np.clip(np.round(29.5 - mesh[:, 0] / 0.1).astype(int), 0, 59)
    cols = np.clip(np.round(mesh[:, 1] / 0.1 + 29.5).astype(int), 0, 59)
    on_arc = arc[rows, cols] >= 3.0
    on_sky = snr[rows, cols] < 1.0
    return on_arc.mean(), on_sky.mean()


def test__image_plane_mesh_grid_from__bright_compact_component_keeps_arc_coverage():
    for peak in (25.0, 250.0, 1000.0):
        mask, adapt_data, arc, snr = _arc_plus_compact(peak)

        mesh = np.asarray(
            aa.image_mesh.HilbertSNR(pixels=500).image_plane_mesh_grid_from(
                mask=mask, adapt_data=adapt_data
            )
        )
        arc_share, sky_share = _shares(mesh, mask, arc, snr)

        assert arc_share > 0.5
        assert sky_share < 0.3

    # The peak-normalised `Hilbert` weighting loses the arc next to a peak S/N of 250.
    mask, adapt_data, arc, snr = _arc_plus_compact(250.0)
    mesh = np.asarray(
        aa.image_mesh.Hilbert(pixels=500, weight_power=3.5, weight_floor=0.01).image_plane_mesh_grid_from(
            mask=mask, adapt_data=adapt_data
        )
    )
    arc_share, _ = _shares(mesh, mask, arc, snr)
    assert arc_share < 0.2


def test__image_plane_mesh_grid_from__negative_values_treated_as_zero():
    mask, adapt_data, arc, snr = _arc_plus_compact(25.0)
    rng = np.random.default_rng(3)
    noisy = np.asarray(adapt_data.array) + rng.normal(0.0, 1.0, size=adapt_data.shape[0])

    image_mesh = aa.image_mesh.HilbertSNR(pixels=200)
    mesh_raw = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=aa.Array2D(values=noisy, mask=mask)
    )
    mesh_clipped = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=aa.Array2D(values=np.clip(noisy, 0.0, None), mask=mask)
    )

    assert np.asarray(mesh_raw) == pytest.approx(np.asarray(mesh_clipped), 1.0e-12)


@pytest.mark.parametrize(
    "image_mesh",
    [
        aa.image_mesh.HilbertSNR(pixels=50, edge_pixels=12),
        aa.image_mesh.Hilbert(pixels=50, weight_power=3.5, weight_floor=0.01, edge_pixels=12),
        aa.image_mesh.Overlay(shape=(10, 10), edge_pixels=12),
    ],
)
def test__edge_pixels__appended_on_circle_outside_mask(image_mesh):
    mask, adapt_data, arc, snr = _arc_plus_compact(25.0)

    grid = np.asarray(image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_data))

    edge = grid[-12:]
    radius = np.sqrt(edge[:, 0] ** 2 + edge[:, 1] ** 2)
    assert radius == pytest.approx(np.full(12, mask.circular_radius + 0.05), 1.0e-8)
    # Even-sized frame: the radius the mask was made with (2.8) plus half a pixel, as the workspace scripts place them.
    assert radius == pytest.approx(np.full(12, 2.8 + 0.05), 1.0e-8)

    without = type(image_mesh)(
        **{k: v for k, v in vars(image_mesh).items() if k in ("pixels", "shape")}
    )
    if isinstance(image_mesh, aa.image_mesh.Hilbert) and not isinstance(
        image_mesh, aa.image_mesh.HilbertSNR
    ):
        without = aa.image_mesh.Hilbert(pixels=50, weight_power=3.5, weight_floor=0.01)
    grid_without = np.asarray(without.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_data))

    assert grid.shape[0] == grid_without.shape[0] + 12
    assert grid[:-12] == pytest.approx(grid_without, 1.0e-12)


def _padded(mask, adapt_data, pad):
    mask_native = np.pad(np.asarray(mask), pad, constant_values=True)
    values = np.pad(np.asarray(adapt_data.native), pad)
    padded_mask = aa.Mask2D(mask=mask_native, pixel_scales=mask.pixel_scales)
    return padded_mask, aa.Array2D(values=values[~mask_native], mask=padded_mask)


@pytest.mark.parametrize(
    "image_mesh",
    [
        aa.image_mesh.HilbertSNR(pixels=300),
        aa.image_mesh.Hilbert(pixels=300, weight_power=3.5, weight_floor=0.01, frame_independent=True),
    ],
)
def test__frame_independent__same_mesh_in_larger_and_non_square_frames(image_mesh):
    mask, adapt_data, arc, snr = _arc_plus_compact(25.0)

    mesh = np.asarray(image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_data))

    for pad in (((10, 10), (10, 10)), ((0, 0), (8, 8)), ((6, 6), (0, 0))):
        padded_mask, padded_data = _padded(mask, adapt_data, pad)
        padded_mesh = np.asarray(
            image_mesh.image_plane_mesh_grid_from(mask=padded_mask, adapt_data=padded_data)
        )
        assert padded_mesh == pytest.approx(mesh, abs=1.0e-9)


def test__hilbert_default__unchanged_interpolation():
    mask, adapt_data, arc, snr = _arc_plus_compact(25.0)

    default = aa.image_mesh.Hilbert(pixels=300, weight_power=3.5, weight_floor=0.01)
    assert default.frame_independent is False

    from autoarray.inversion.mesh.image_mesh.hilbert import image_and_grid_from

    values_default, grid_default = image_and_grid_from(
        image=adapt_data, mask=mask, mask_radius=mask.circular_radius,
        pixel_scales=mask.pixel_scales, hilbert_length=193, mask_centre=mask.mask_centre,
    )
    values_bilinear, grid_bilinear = image_and_grid_from(
        image=adapt_data, mask=mask, mask_radius=mask.circular_radius,
        pixel_scales=mask.pixel_scales, hilbert_length=193, mask_centre=mask.mask_centre,
        frame_independent=True,
    )
    assert grid_bilinear == pytest.approx(grid_default, 1.0e-12)
    # The two interpolations agree closely, but are not identical between pixel centres.
    assert np.abs(values_bilinear - values_default).max() < 0.2 * np.abs(values_default).max()


def test__frame_independent__bilinear_is_exact_on_a_linear_image():
    from autoarray.inversion.mesh.image_mesh.hilbert import image_and_grid_from

    mask = aa.Mask2D.circular(shape_native=(40, 30), radius=1.0, pixel_scales=0.1)
    y, x = np.indices(mask.shape_native)
    y = (20 - 0.5 - y) * 0.1
    x = (x - 15 + 0.5) * 0.1
    plane = 3.0 + 2.0 * y - 1.5 * x
    image = aa.Array2D(values=plane[~np.asarray(mask)], mask=mask)

    values, grid = image_and_grid_from(
        image=image, mask=mask, mask_radius=mask.circular_radius,
        pixel_scales=mask.pixel_scales, hilbert_length=193, mask_centre=mask.mask_centre,
        frame_independent=True,
    )

    inner = np.sqrt(grid[:, 0] ** 2 + grid[:, 1] ** 2) < 0.8
    assert values[inner] == pytest.approx(3.0 + 2.0 * grid[inner, 0] - 1.5 * grid[inner, 1], 1.0e-10)


def test__capped_probabilities_from():
    from autoarray.inversion.mesh.image_mesh.hilbert_snr import capped_probabilities_from

    p = np.array([0.5, 0.2, 0.2, 0.1, 0.0])
    capped = capped_probabilities_from(probabilities=p, cap=0.3)

    assert capped.sum() == pytest.approx(1.0, 1.0e-12)
    assert capped.max() <= 0.3 + 1.0e-12
    assert capped[0] == pytest.approx(0.3, 1.0e-10)
    # The rest keep their proportions.
    assert capped[1] / capped[3] == pytest.approx(2.0, 1.0e-8)

    assert capped_probabilities_from(probabilities=p, cap=0.9) == pytest.approx(p)
    assert capped_probabilities_from(probabilities=p, cap=0.1) == pytest.approx(np.full(5, 0.2))


@pytest.mark.parametrize("max_per_pixel", [8.0, 3.0])
def test__max_per_pixel__caps_points_per_image_pixel(max_per_pixel):
    # A faint, compact source: without the cap most points pile onto a few pixels.
    mask = aa.Mask2D.circular(shape_native=(60, 60), radius=2.8, pixel_scales=0.1)
    y, x = np.indices(mask.shape_native)
    y = (29.5 - y) * 0.1
    x = (x - 29.5) * 0.1
    snr = 6.0 * np.exp(-0.5 * ((y - 1.0) ** 2 + x**2) / 0.08**2)
    adapt_data = aa.Array2D(values=snr[~np.asarray(mask)], mask=mask)

    def most_per_pixel(image_mesh):
        mesh = np.asarray(image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_data))
        rows = np.clip(np.round(29.5 - mesh[:, 0] / 0.1).astype(int), 0, 59)
        cols = np.clip(np.round(mesh[:, 1] / 0.1 + 29.5).astype(int), 0, 59)
        counts = np.zeros((60, 60), dtype=int)
        np.add.at(counts, (rows, cols), 1)
        return counts.max()

    uncapped = most_per_pixel(aa.image_mesh.HilbertSNR(pixels=1000, max_per_pixel=None))
    capped = most_per_pixel(aa.image_mesh.HilbertSNR(pixels=1000, max_per_pixel=max_per_pixel))

    assert uncapped > 3 * max_per_pixel
    # On average over a pixel's Hilbert cells; allow one for where cell and pixel boundaries fall.
    assert capped <= max_per_pixel + 1
