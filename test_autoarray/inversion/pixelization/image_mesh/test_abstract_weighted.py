import numpy as np
import pytest

import autoarray as aa


def test__weight_map_from():
    adapt_data = np.array([-1.0, 1.0, 3.0])

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=1.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert weight_map == pytest.approx([0.33333, 0.33333, 1.0], 1.0e-4)

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=0.0, weight_power=2.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert weight_map == pytest.approx([0.11111, 0.11111, 1.0], 1.0e-4)

    pixelization = aa.image_mesh.KMeans(pixels=5, weight_floor=1.0, weight_power=1.0)

    weight_map = pixelization.weight_map_from(adapt_data=adapt_data)

    assert weight_map == pytest.approx([1.0, 1.0, 1.0], 1.0e-4)


def test__weight_map_from__blank_adapt_image_returns_uniform_not_nan():
    # A blank (all-zero) adapt image has ``np.max == 0``; the normalisation must
    # not divide by zero and produce NaN weights, which would propagate into NaN
    # mesh coordinates and crash the downstream Delaunay triangulation.
    pixelization = aa.image_mesh.Hilbert(
        pixels=5, weight_floor=0.01, weight_power=3.5
    )

    weight_map = pixelization.weight_map_from(adapt_data=np.zeros(4))

    assert np.all(np.isfinite(weight_map))
    assert weight_map == pytest.approx([1.0, 1.0, 1.0, 1.0], 1.0e-4)

    # An adapt image with no positive signal (non-positive peak) is equally
    # degenerate and must also fall back to a finite, uniform weight map.
    weight_map = pixelization.weight_map_from(adapt_data=np.array([-2.0, -1.0, -3.0]))

    assert np.all(np.isfinite(weight_map))
    assert weight_map == pytest.approx([1.0, 1.0, 1.0], 1.0e-4)
