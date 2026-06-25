from autoconf import conf

from autoarray.plot.utils import _arcsec_labels


def test_arcsec_labels_default_suffix_format():
    conf.instance["visualize"]["general"]["ticks"]["symbol_over_decimal"] = False

    assert _arcsec_labels([-1.0, 0.0, 1.0]) == ['-1"', '0"', '1"']
    assert _arcsec_labels([3.8]) == ['3.8"']


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
    finally:
        ticks["symbol_over_decimal"] = original
