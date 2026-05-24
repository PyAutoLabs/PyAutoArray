"""Unit tests for ``SimulatorInterferometer(use_jax=True)`` constructor wiring.

Library unit tests stay NumPy-only per [[feedback_no_jax_in_unit_tests]];
cross-xp numerical parity lives in the workspace_test parity script.
"""
import numpy as np

import autoarray as aa


def test_use_jax_defaults_false():
    simulator = aa.SimulatorInterferometer(
        uv_wavelengths=np.array([[0.1, 0.2], [0.3, 0.4]]),
        exposure_time=300.0,
    )
    assert simulator.use_jax is False
    assert simulator._xp is np


def test_use_jax_true_flag_stored():
    simulator = aa.SimulatorInterferometer(
        uv_wavelengths=np.array([[0.1, 0.2], [0.3, 0.4]]),
        exposure_time=300.0,
        use_jax=True,
    )
    assert simulator.use_jax is True


def test_via_image_from_accepts_xp_param():
    """via_image_from now accepts xp= (signature symmetry with SimulatorImaging)."""
    import inspect

    sig = inspect.signature(aa.SimulatorInterferometer.via_image_from)
    assert "xp" in sig.parameters
    assert sig.parameters["xp"].default is None
