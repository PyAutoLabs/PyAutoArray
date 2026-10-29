"""The intro compositor uses the ordinary renderer, style and configuration."""
import numpy as np
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import AsinhNorm
from autonerves import conf
import autoarray as aa
from autoarray.plot.array import plot_array
from autoarray.plot.utils import norm_from, _colorbar_tick_values
from autoarray.dataset.plot.imaging_plots import subplot_imaging_dataset_intro, _intro_panel


def dataset():
    values = np.arange(81, dtype=float).reshape(9, 9) - 3
    return aa.Imaging(
        data=aa.Array2D.no_mask(values, pixel_scales=0.1),
        noise_map=aa.Array2D.no_mask(np.ones((9, 9)), pixel_scales=0.1),
        psf=aa.Convolver(kernel=aa.Array2D.no_mask(np.ones((3, 3)) / 9, pixel_scales=0.1)),
    )


def test_intro_inherits_config_and_standard_style(tmp_path, monkeypatch):
    general = conf.instance["visualize"]["general"]
    monkeypatch.setitem(general, "colormap", "cividis")
    d = dataset()
    original = d.data.native.array.copy()
    fig = subplot_imaging_dataset_intro(d, output_path=tmp_path, output_format="png")
    panels = [ax for ax in fig.axes if ax.images]
    assert [ax.get_title() for ax in panels] == ["Image", "Noise map", "Signal-to-noise map", "PSF"]
    assert all(ax.images[0].get_cmap().name == "cividis" for ax in panels)
    direct, ax = plt.subplots()
    values, extent = _intro_panel(d.data)
    plot_array(values, extent=extent, ax=ax, title="Image", stretch=0.01)
    assert ax.title.get_fontsize() == panels[0].title.get_fontsize()
    assert [t.get_text() for t in ax.get_xticklabels()] == [t.get_text() for t in panels[0].get_xticklabels()]
    assert ax.get_yticklabels()[0].get_rotation() == panels[0].get_yticklabels()[0].get_rotation()
    assert isinstance(panels[0].images[0].norm, AsinhNorm)
    np.testing.assert_array_equal(d.data.native.array, original)
    assert (tmp_path / 'dataset_intro.png').is_file()
    plt.close(direct)


def test_explicit_colormap_and_excluded_noise(tmp_path):
    d = dataset()
    excluded = np.zeros((9, 9), dtype=bool)
    excluded[2, 3] = True
    noise = np.ones((9, 9))
    noise[2, 3] = 1e8
    d = aa.Imaging(data=d.data, noise_map=aa.Array2D.no_mask(noise, pixel_scales=.1), psf=d.psf)
    fig = subplot_imaging_dataset_intro(d, colormap="magma", radius=0.25,
                                       excluded_pixels=excluded, output_path=tmp_path, output_format="png")
    panels = [ax for ax in fig.axes if ax.images]
    assert all(ax.images[0].get_cmap().name == "magma" for ax in panels)
    assert panels[1].images[0].norm.vmax < 2
    assert np.ma.getmaskarray(panels[0].images[0].get_array()).any()
    assert d.noise_map.native.array[2, 3] == 1e8


def test_crop_preserves_pixel_coordinates():
    d = dataset()
    values, extent = _intro_panel(d.data, radius=0.25)
    assert values.shape == (5, 5)
    np.testing.assert_allclose(extent, [-0.25, 0.25, -0.25, 0.25])
    assert values[2, 2] == d.data.native.array[4, 4]


@pytest.mark.parametrize('values', [np.zeros((2, 2)), np.full((2, 2), np.nan), np.array([-2., 0., 100.])])
def test_asinh_range_is_finite_and_invertible(values):
    norm = norm_from(values, stretch=0.01)
    assert np.isfinite([norm.vmin, norm.vmax]).all()
    assert norm.vmax > norm.vmin
    ticks = _colorbar_tick_values(norm)
    np.testing.assert_allclose(norm(ticks), [0, 0.5, 1], atol=1e-10)


@pytest.mark.parametrize('stretch', [0, -1, np.nan, np.inf])
def test_invalid_stretch_rejected(stretch):
    with pytest.raises(ValueError, match='stretch'):
        norm_from(np.ones(3), stretch=stretch)


def test_asinh_cannot_be_combined_with_log():
    with pytest.raises(ValueError, match='mutually exclusive'):
        norm_from([1, 2], stretch=.01, use_log10=True)


def test_ordinary_norm_remains_unchanged():
    assert norm_from([1, 2]) is None
    assert not isinstance(norm_from([1, 2], use_log10=True), AsinhNorm)
