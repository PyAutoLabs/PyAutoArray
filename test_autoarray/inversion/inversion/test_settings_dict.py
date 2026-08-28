import os

import pytest
from pathlib import Path

import autoarray as aa
from autonerves.dictable import from_dict, to_dict, output_to_json, from_json


@pytest.fixture(name="settings_dict")
def make_settings_dict():
    return {
        "class_path": "autoarray.settings.Settings",
        "type": "instance",
        "arguments": {
            "use_positive_only_solver": False,
            "no_regularization_add_to_curvature_diag_value": 1e-08,
        },
    }


def test_settings_from_dict(settings_dict):
    assert isinstance(from_dict(settings_dict), aa.Settings)


def test_file():
    filename = Path("/tmp/temp.json")

    output_to_json(aa.Settings(), filename)

    try:
        assert isinstance(from_json(filename), aa.Settings)
    finally:
        os.remove(filename)


def test_settings_nnls_warm_start_memo_round_trips():
    # The field is serialised through the property (not the private attribute),
    # so a `None` default resolves to the packaged config value on the way out
    # and must come back as the same explicit boolean.
    assert aa.Settings().nnls_warm_start_memo is True
    assert aa.Settings(nnls_warm_start_memo=True).nnls_warm_start_memo is True
    assert aa.Settings(nnls_warm_start_memo=False).nnls_warm_start_memo is False

    settings = from_dict(to_dict(aa.Settings(nnls_warm_start_memo=True)))

    assert settings.nnls_warm_start_memo is True


def test_settings_nnls_warm_start_error_tolerance_round_trips():
    # The test config does not ship the key, so the default resolves through
    # the KeyError fallback -- which is also the production path whenever a
    # workspace shadows autoarray's general.yaml.
    assert aa.Settings().nnls_warm_start_error_tolerance == 1.5
    assert (
        aa.Settings(nnls_warm_start_error_tolerance=2.5).nnls_warm_start_error_tolerance
        == 2.5
    )
    assert aa.Settings(
        nnls_warm_start_error_tolerance=float("inf")
    ).nnls_warm_start_error_tolerance == float("inf")

    settings = from_dict(to_dict(aa.Settings(nnls_warm_start_error_tolerance=2.5)))

    assert settings.nnls_warm_start_error_tolerance == 2.5
