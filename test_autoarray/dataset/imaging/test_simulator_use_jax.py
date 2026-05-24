"""Unit tests for ``SimulatorImaging(use_jax=True)`` constructor wiring.

Per the PyAutoArray dependency-graph rule, library unit tests stay NumPy-only —
cross-xp numerical parity for the JAX execution path lives in
``autolens_workspace_test/scripts/imaging/simulator_use_jax_parity.py``.
"""
import numpy as np

import autoarray as aa


def test_use_jax_defaults_false():
    simulator = aa.SimulatorImaging(exposure_time=300.0)
    assert simulator.use_jax is False
    assert simulator._xp is np


def test_use_jax_true_flag_stored():
    simulator = aa.SimulatorImaging(exposure_time=300.0, use_jax=True)
    assert simulator.use_jax is True


def test_via_image_from_default_xp_falls_back_to_self_xp():
    """When xp is not passed, via_image_from must fall back to self._xp.

    Construct a no-noise simulator (so the RNG paths don't run) and confirm
    the call succeeds with the default xp=None resolving to numpy via _xp.
    """
    simulator = aa.SimulatorImaging(
        exposure_time=300.0,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
    )

    image = aa.Array2D.no_mask(
        values=np.ones((20, 20)),
        pixel_scales=0.1,
    )

    dataset = simulator.via_image_from(image=image)
    assert dataset.data.shape_native == (20, 20)
    assert isinstance(dataset.data.array, np.ndarray)
