from autoconf import conf

from autoarray.plot.utils import _arcsec_labels


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
