from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import LogNorm, Normalize

from autonerves import conf

import autoarray.plot as aplt
from autoarray.plot import utils as plot_utils
from autoarray.plot.utils import _arcsec_labels, norm_from


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return Path(Path(__file__).resolve().parent) / "files" / "plots"


def test_arcsec_labels_default_suffix_format():
    ticks = conf.instance["visualize"]["general"]["ticks"]
    original = ticks.get("symbol_over_decimal", False)
    try:
        ticks["symbol_over_decimal"] = False

        assert _arcsec_labels([-1.0, 0.0, 1.0]) == ['-1"', '0"', '1"']
        assert _arcsec_labels([3.8]) == ['3.8"']
    finally:
        ticks["symbol_over_decimal"] = original


def test_arcsec_labels_symbol_over_decimal():
    ticks = conf.instance["visualize"]["general"]["ticks"]
    original = ticks.get("symbol_over_decimal", False)
    try:
        ticks["symbol_over_decimal"] = True

        assert _arcsec_labels([-1.69, 0.13, 1.95]) == [
            "-1.″69",
            "0.″13",
            "1.″95",
        ]
        assert _arcsec_labels([3.0]) == ["3″"]

        # Mixed whole-number and decimal ticks: the whole-number tick gains a
        # single decimal so it reads as 2.″0 rather than a bare 2″.
        assert _arcsec_labels([-2.1, -0.044, 2.0]) == [
            "-2.″1",
            "-0.″044",
            "2.″0",
        ]
    finally:
        ticks["symbol_over_decimal"] = original


def test_arcsec_labels_mixed_integer_and_decimal_default():
    ticks = conf.instance["visualize"]["general"]["ticks"]
    original = ticks.get("symbol_over_decimal", False)
    try:
        ticks["symbol_over_decimal"] = False

        # All-integer sets stay integer; a mixed set pads the whole number.
        assert _arcsec_labels([-1.0, 0.0, 1.0]) == ['-1"', '0"', '1"']
        assert _arcsec_labels([-2.1, -0.044, 2.0]) == ['-2.1"', '-0.044"', '2.0"']
    finally:
        ticks["symbol_over_decimal"] = original


def test_arcsec_labels_minus_in_math():
    ticks = conf.instance["visualize"]["general"]["ticks"]
    original_symbol = ticks.get("symbol_over_decimal", False)
    original_minus = ticks.get("minus_in_math", False)
    try:
        ticks["minus_in_math"] = True

        # Suffix format: the ASCII hyphen (U+002D) becomes the math minus (U+2212).
        ticks["symbol_over_decimal"] = False
        assert _arcsec_labels([-2.1, -0.044, 2.0]) == ["−2.1\"", "−0.044\"", "2.0\""]

        # Symbol-over-decimal format also uses the math minus.
        ticks["symbol_over_decimal"] = True
        assert _arcsec_labels([-2.1, -0.044, 2.0]) == ["−2.″1", "−0.″044", "2.″0"]

        # Positive-only tick sets are unchanged.
        assert _arcsec_labels([0.13, 1.95]) == ["0.″13", "1.″95"]

        # The emitted minus really is U+2212, not the ASCII hyphen U+002D.
        label = _arcsec_labels([-0.5])[0]
        assert "−" in label and "-" not in label
    finally:
        ticks["symbol_over_decimal"] = original_symbol
        ticks["minus_in_math"] = original_minus


class TestNormFrom:
    """The one colour-norm helper `plot_array`, `plot_inversion_reconstruction`
    and `autogalaxy.util.plot_utils.norm_from` all build their norms with.

    Each behaviour below was one of the three call sites' before it was merged;
    `norm_from`'s docstring records which won and why.
    """

    def test__no_limits_and_no_log__returns_none(self):
        assert norm_from(array=np.array([1.0, 2.0, 3.0])) is None

    def test__explicit_limits__returns_linear_norm(self):
        norm = norm_from(array=np.array([1.0, 2.0, 3.0]), vmin=0.5, vmax=2.5)

        assert isinstance(norm, Normalize)
        assert not isinstance(norm, LogNorm)
        assert norm.vmin == pytest.approx(0.5)
        assert norm.vmax == pytest.approx(2.5)

    def test__use_log10__explicit_limits_are_used_as_given(self):
        norm = norm_from(
            array=np.array([1.0, 2.0, 3.0]), use_log10=True, vmin=1.0e-3, vmax=1.0
        )

        assert isinstance(norm, LogNorm)
        assert norm.vmin == pytest.approx(1.0e-3)
        assert norm.vmax == pytest.approx(1.0)

    def test__use_log10__vmax_derived_from_the_values_being_coloured(self):
        norm = norm_from(array=np.array([1.0, 2.0, 7.0]), use_log10=True, vmin=1.0e-3)

        assert norm.vmax == pytest.approx(7.0)

    def test__use_log10__vmin_defaults_to_the_configured_floor(self):
        general = conf.instance["visualize"]["general"]["general"]
        original = general["log10_min_value"]
        try:
            general["log10_min_value"] = 3.0e-3

            norm = norm_from(array=np.array([1.0, 2.0, 7.0]), use_log10=True)

            assert norm.vmin == pytest.approx(3.0e-3)
        finally:
            general["log10_min_value"] = original

    def test__use_log10__values_are_clipped_to_the_configured_floor(self):
        """The floor is a floor on what is rendered, so `vmax` is derived from
        the clipped values — a whole array below the floor scales as the floor,
        not as its own (unrenderable) maximum."""
        general = conf.instance["visualize"]["general"]["general"]
        original = general["log10_min_value"]
        try:
            general["log10_min_value"] = 1.0e-2

            norm = norm_from(
                array=np.array([1.0e-5, 1.0e-4]), use_log10=True, vmin=1.0e-8
            )

            assert norm.vmax == pytest.approx(1.0e-2)
        finally:
            general["log10_min_value"] = original

    def test__use_log10__degenerate_range_is_widened_even_when_passed_explicitly(self):
        """`LogNorm(vmin=10, vmax=1)` is unusable however the pair arose, so the
        widening is not confined to the derived-`vmax` branch."""
        norm = norm_from(
            array=np.array([1.0, 2.0]), use_log10=True, vmin=10.0, vmax=1.0
        )

        assert norm.vmin == pytest.approx(10.0)
        assert norm.vmax == pytest.approx(100.0)

    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    def test__use_log10__all_nan_values_still_yield_a_finite_range(self):
        norm = norm_from(array=np.array([np.nan, np.nan]), use_log10=True)

        assert np.isfinite(norm.vmin)
        assert np.isfinite(norm.vmax)
        assert norm.vmax > norm.vmin

    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    def test__use_log10__no_values_at_all_takes_the_same_fallback(self):
        """`plot_inversion_reconstruction` may have no `pixel_values`; that is
        `array=None` here, and it must not raise."""
        norm = norm_from(array=None, use_log10=True)
        nan_norm = norm_from(array=np.array([np.nan]), use_log10=True)

        assert norm.vmin == pytest.approx(nan_norm.vmin)
        assert norm.vmax == pytest.approx(nan_norm.vmax)


class TestBothCallSitesHonourTheConfiguredFloor:
    """The regression this helper exists for.

    `plot_inversion_reconstruction` used to hardcode `1e-4` and never read
    `log10_min_value`, so a changed floor was honoured on array plots and
    silently ignored on inversion plots. Both call sites are exercised for real
    here — a spy wrapping the shared helper records the norm each one actually
    applied.
    """

    FLOOR = 3.0e-3

    @staticmethod
    def _spy_on(monkeypatch, module):
        """Wrap `module.norm_from` so the norm it returns can be inspected."""
        recorded = []

        def _recording_norm_from(**kwargs):
            norm = plot_utils.norm_from(**kwargs)
            recorded.append(norm)
            return norm

        monkeypatch.setattr(module, "norm_from", _recording_norm_from)
        return recorded

    def test__plot_array(self, array_2d_7x7, plot_path, plot_patch, monkeypatch):
        from autoarray.plot import array as array_module

        recorded = self._spy_on(monkeypatch, array_module)

        general = conf.instance["visualize"]["general"]["general"]
        original = general["log10_min_value"]
        try:
            general["log10_min_value"] = self.FLOOR

            aplt.plot_array(
                array=array_2d_7x7,
                use_log10=True,
                output_path=plot_path,
                output_filename="array_log10_floor",
                output_format="png",
            )
        finally:
            general["log10_min_value"] = original

        assert len(recorded) == 1
        assert recorded[0].vmin == pytest.approx(self.FLOOR)

    def test__plot_inversion_reconstruction(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch, monkeypatch
    ):
        from autoarray.plot import inversion as inversion_module

        recorded = self._spy_on(monkeypatch, inversion_module)

        general = conf.instance["visualize"]["general"]["general"]
        original = general["log10_min_value"]
        try:
            general["log10_min_value"] = self.FLOOR

            aplt.plot_inversion_reconstruction(
                pixel_values=np.ones(9),
                mapper=rectangular_mapper_7x7_3x3,
                use_log10=True,
                output_path=plot_path,
                output_filename="inversion_log10_floor",
                output_format="png",
            )
        finally:
            general["log10_min_value"] = original

        assert len(recorded) == 1
        assert recorded[0].vmin == pytest.approx(self.FLOOR)


class TestDefaultColormap:
    """The `visualize/general.yaml -> colormap` lever and its failure modes."""

    @staticmethod
    def _general():
        return conf.instance["visualize"]["general"]

    def test__config_value_is_returned(self):
        general = self._general()
        original = general["colormap"]
        try:
            general["colormap"] = "magma"

            assert plot_utils._default_colormap() == "magma"
        finally:
            general["colormap"] = original

    def test__absent_key_falls_back_quietly_to_autoarray(self):
        import matplotlib

        general = self._general()
        original = general["colormap"]
        try:
            del general["colormap"]

            assert plot_utils._default_colormap() == "autoarray"
        finally:
            general["colormap"] = original

        # The fallback also registers the bundled colormap, so it is usable.
        assert "autoarray" in matplotlib.colormaps

    def test__autoarray_value_registers_the_bundled_colormap(self):
        import matplotlib

        general = self._general()
        original = general["colormap"]
        try:
            general["colormap"] = "autoarray"

            assert plot_utils._default_colormap() == "autoarray"
        finally:
            general["colormap"] = original

        assert "autoarray" in matplotlib.colormaps

    def test__unknown_value_raises_instead_of_reverting_silently(self):
        general = self._general()
        original = general["colormap"]
        try:
            general["colormap"] = "magmaa"

            with pytest.raises(ValueError) as exc_info:
                plot_utils._default_colormap()
        finally:
            general["colormap"] = original

        message = str(exc_info.value)
        assert "magmaa" in message
        assert "colormap" in message

    def test__non_string_value_raises(self):
        general = self._general()
        original = general["colormap"]
        try:
            general["colormap"] = 3

            with pytest.raises(ValueError):
                plot_utils._default_colormap()
        finally:
            general["colormap"] = original

    def test__plot_array_uses_the_config_colormap(self, plot_path, monkeypatch):
        """One config edit moves the imaging figure's colormap."""
        from autoarray.structures.arrays.uniform_2d import Array2D

        recorded = {}

        import matplotlib.pyplot as plt

        original_imshow = plt.Axes.imshow

        def spy(self, *args, **kwargs):
            recorded.setdefault("cmap", kwargs.get("cmap"))
            return original_imshow(self, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "imshow", spy)

        general = self._general()
        original = general["colormap"]
        try:
            general["colormap"] = "magma"

            aplt.plot_array(
                array=Array2D.no_mask(
                    values=np.ones((7, 7)),
                    pixel_scales=1.0,
                ),
                output_path=plot_path,
                output_filename="config_colormap",
                output_format="png",
            )
        finally:
            general["colormap"] = original

        assert recorded["cmap"] == "magma"

    def test__per_figure_colormap_overrides_the_config(self, plot_path, monkeypatch):
        """`colormap=` on a plot function beats the config for that figure only."""
        from autoarray.structures.arrays.uniform_2d import Array2D

        recorded = {}

        import matplotlib.pyplot as plt

        original_imshow = plt.Axes.imshow

        def spy(self, *args, **kwargs):
            recorded.setdefault("cmap", kwargs.get("cmap"))
            return original_imshow(self, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "imshow", spy)

        general = self._general()
        original = general["colormap"]
        try:
            general["colormap"] = "magma"

            aplt.plot_array(
                array=Array2D.no_mask(
                    values=np.ones((7, 7)),
                    pixel_scales=1.0,
                ),
                colormap="viridis",
                output_path=plot_path,
                output_filename="override_colormap",
                output_format="png",
            )
        finally:
            general["colormap"] = original

        assert recorded["cmap"] == "viridis"

    def test__inversion_reconstruction_uses_the_config_colormap(
        self, rectangular_mapper_7x7_3x3, plot_path, monkeypatch
    ):
        """The same config edit moves the inversion reconstruction figure."""
        recorded = {}

        import matplotlib.pyplot as plt

        original_imshow = plt.Axes.imshow

        def spy(self, *args, **kwargs):
            recorded.setdefault("cmap", kwargs.get("cmap"))
            return original_imshow(self, *args, **kwargs)

        monkeypatch.setattr(plt.Axes, "imshow", spy)

        general = self._general()
        original = general["colormap"]
        try:
            general["colormap"] = "magma"

            aplt.plot_inversion_reconstruction(
                pixel_values=np.ones(9),
                mapper=rectangular_mapper_7x7_3x3,
                output_path=plot_path,
                output_filename="inversion_config_colormap",
                output_format="png",
            )
        finally:
            general["colormap"] = original

        assert recorded["cmap"] == "magma"


class TestConfImshowOrigin:
    @staticmethod
    def _general():
        return conf.instance["visualize"]["general"]["general"]

    def test__config_value_is_returned(self):
        general = self._general()
        original = general["imshow_origin"]
        try:
            general["imshow_origin"] = "lower"

            assert plot_utils._conf_imshow_origin() == "lower"
        finally:
            general["imshow_origin"] = original

    def test__absent_key_falls_back_quietly(self):
        general = self._general()
        original = general["imshow_origin"]
        try:
            del general["imshow_origin"]

            assert plot_utils._conf_imshow_origin() == "upper"
        finally:
            general["imshow_origin"] = original

    def test__invalid_value_raises(self):
        general = self._general()
        original = general["imshow_origin"]
        try:
            general["imshow_origin"] = "sideways"

            with pytest.raises(ValueError) as exc_info:
                plot_utils._conf_imshow_origin()
        finally:
            general["imshow_origin"] = original

        assert "sideways" in str(exc_info.value)
