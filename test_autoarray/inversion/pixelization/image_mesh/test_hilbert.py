import numpy as np
import pytest

import autoarray as aa


def test__image_plane_mesh_grid_from():
    mask = aa.Mask2D.circular(
        shape_native=(4, 4),
        radius=2.0,
        pixel_scales=1.0,
    )

    adapt_data = aa.Array2D.ones(
        shape_native=mask.shape_native,
        pixel_scales=1.0,
    )

    kmeans = aa.image_mesh.Hilbert(pixels=8)
    image_mesh = kmeans.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_data)

    assert image_mesh[0, :] == pytest.approx(
        [-1.02590674, -1.70984456],
        1.0e-4,
    )


def test__image_plane_mesh_grid_from__offset_centre__points_inside_mask_circle():
    centre = (0.5, 0.3)
    radius = 0.5

    mask = aa.Mask2D.circular(
        shape_native=(40, 40),
        radius=radius,
        pixel_scales=0.1,
        centre=centre,
    )

    adapt_data = aa.Array2D.ones(
        shape_native=mask.shape_native,
        pixel_scales=0.1,
    )

    kmeans = aa.image_mesh.Hilbert(pixels=16)
    image_mesh = kmeans.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_data)

    points = np.asarray(image_mesh)
    distances = np.sqrt(
        (points[:, 0] - centre[0]) ** 2 + (points[:, 1] - centre[1]) ** 2
    )

    assert np.all(distances <= radius + 1e-6)
