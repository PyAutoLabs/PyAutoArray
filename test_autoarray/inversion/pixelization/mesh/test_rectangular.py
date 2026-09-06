import numpy as np
import pytest

import autoarray as aa

from autoarray.inversion.mesh.interpolator.rectangular import (
    KERNEL_CDF_DEFAULT_BANDWIDTH,
    KERNEL_CDF_DEFAULT_KNOTS,
    InterpolatorRectangular,
)
from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
    InterpolatorRectangularUniform,
)
from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
    overlay_grid_from,
)


def test__overlay_grid_from__shape_native_and_pixel_scales():
    grid = aa.Grid2DIrregular(
        [
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    mesh = aa.mesh.RectangularUniform(shape=(3, 3))

    mesh_grid = overlay_grid_from(shape_native=mesh.shape, grid=grid, buffer=1e-8)

    mesh = aa.MeshGeometryRectangular(mesh=mesh, mesh_grid=mesh_grid, data_grid=None)

    assert mesh.shape_native == (3, 3)
    assert mesh.pixel_scales == pytest.approx((2.0 / 3.0, 2.0 / 3.0), 1e-2)

    grid = aa.Grid2DIrregular(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
        ]
    )

    mesh = aa.mesh.RectangularUniform(shape=(5, 4))

    mesh_grid = overlay_grid_from(shape_native=mesh.shape, grid=grid, buffer=1e-8)

    mesh = aa.MeshGeometryRectangular(mesh=mesh, mesh_grid=mesh_grid, data_grid=None)

    assert mesh.shape_native == (5, 4)
    assert mesh.pixel_scales == pytest.approx((2.0 / 5.0, 2.0 / 4.0), 1e-2)

    grid = aa.Grid2DIrregular([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])

    mesh = aa.mesh.RectangularUniform(shape=(3, 3))

    mesh_grid = overlay_grid_from(shape_native=mesh.shape, grid=grid, buffer=1e-8)

    mesh = aa.MeshGeometryRectangular(mesh=mesh, mesh_grid=mesh_grid, data_grid=None)

    assert mesh.shape_native == (3, 3)
    assert mesh.pixel_scales == pytest.approx((6.0 / 3.0, 6.0 / 3.0), 1e-2)


def test__overlay_grid_from__pixel_centres__3x3_grid__pixel_centres():
    grid = aa.Grid2DIrregular(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
        ]
    )

    mesh_grid = overlay_grid_from(shape_native=(3, 3), grid=grid, buffer=1e-8)

    assert mesh_grid == pytest.approx(
        np.array(
            [
                [2.0 / 3.0, -2.0 / 3.0],
                [2.0 / 3.0, 0.0],
                [2.0 / 3.0, 2.0 / 3.0],
                [0.0, -2.0 / 3.0],
                [0.0, 0.0],
                [0.0, 2.0 / 3.0],
                [-2.0 / 3.0, -2.0 / 3.0],
                [-2.0 / 3.0, 0.0],
                [-2.0 / 3.0, 2.0 / 3.0],
            ]
        )
    )

    grid = aa.Grid2DIrregular(
        [
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
        ]
    )

    mesh_grid = overlay_grid_from(shape_native=(4, 3), grid=grid, buffer=1e-8)

    assert mesh_grid == pytest.approx(
        np.array(
            [
                [0.75, -2.0 / 3.0],
                [0.75, 0.0],
                [0.75, 2.0 / 3.0],
                [0.25, -2.0 / 3.0],
                [0.25, 0.0],
                [0.25, 2.0 / 3.0],
                [-0.25, -2.0 / 3.0],
                [-0.25, 0.0],
                [-0.25, 2.0 / 3.0],
                [-0.75, -2.0 / 3.0],
                [-0.75, 0.0],
                [-0.75, 2.0 / 3.0],
            ]
        )
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test__construction__shape_and_kernel_kwargs__default():
    density = aa.mesh.RectangularRTUAdaptDensity(shape=(5, 7))
    image = aa.mesh.RectangularRTUAdaptImage(shape=(5, 7))

    assert density.shape == (5, 7)
    assert density.bandwidth == KERNEL_CDF_DEFAULT_BANDWIDTH
    assert density.n_knots == KERNEL_CDF_DEFAULT_KNOTS
    assert image.shape == (5, 7)
    assert image.bandwidth == KERNEL_CDF_DEFAULT_BANDWIDTH
    assert image.n_knots == KERNEL_CDF_DEFAULT_KNOTS
    # inherited
    assert image.weight_power == 1.0
    assert image.weight_floor == 0.0


def test__construction__kernel_kwargs__overridden():
    density = aa.mesh.RectangularRTUAdaptDensity(
        shape=(3, 3), bandwidth=0.5, n_knots=128
    )
    image = aa.mesh.RectangularRTUAdaptImage(
        shape=(3, 3), weight_power=2.0, weight_floor=0.1, bandwidth=2.0, n_knots=32
    )

    assert density.bandwidth == 0.5
    assert density.n_knots == 128
    assert image.bandwidth == 2.0
    assert image.n_knots == 32
    assert image.weight_power == 2.0
    assert image.weight_floor == 0.1


def test__construction__minimum_shape_raises():
    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularRTUAdaptDensity(shape=(2, 3))
    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularRTUAdaptImage(shape=(3, 2))
    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularUniform(shape=(2, 2))


# ---------------------------------------------------------------------------
# Interpolator dispatch
# ---------------------------------------------------------------------------


def test__interpolator_cls_and_kwargs():
    density = aa.mesh.RectangularRTUAdaptDensity(
        shape=(3, 3), bandwidth=0.5, n_knots=128
    )
    image = aa.mesh.RectangularRTUAdaptImage(shape=(3, 3))
    uniform = aa.mesh.RectangularUniform(shape=(3, 3))

    assert density.interpolator_cls is InterpolatorRectangular
    assert image.interpolator_cls is InterpolatorRectangular
    assert uniform.interpolator_cls is InterpolatorRectangularUniform

    assert density.interpolator_kwargs == {"bandwidth": 0.5, "n_knots": 128}
    assert image.interpolator_kwargs == {
        "bandwidth": KERNEL_CDF_DEFAULT_BANDWIDTH,
        "n_knots": KERNEL_CDF_DEFAULT_KNOTS,
    }
    # The uniform interpolator's constructor takes no kernel arguments.
    assert uniform.interpolator_kwargs == {}


# ---------------------------------------------------------------------------
# mesh_weight_map_from
# ---------------------------------------------------------------------------


def test__mesh_weight_map_from__density__returns_none():
    density = aa.mesh.RectangularRTUAdaptDensity(shape=(3, 3))
    assert density.mesh_weight_map_from(adapt_data=None) is None


def test__mesh_weight_map_from__image__returns_weighted_normalized():
    image = aa.mesh.RectangularRTUAdaptImage(
        shape=(3, 3), weight_power=2.0, weight_floor=0.0
    )

    class _Stub:
        def __init__(self, arr):
            self.array = arr

    adapt = _Stub(np.array([1.0, 2.0, 4.0, 0.0, 8.0]))
    w = image.mesh_weight_map_from(adapt_data=adapt)

    expected = np.array([1.0, 4.0, 16.0, 1e-24, 64.0])
    expected = expected / expected.sum()
    assert w == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Bilinear (rank-CDF) meshes
# ---------------------------------------------------------------------------


def test__bilinear__construction_and_interpolator_dispatch():
    density = aa.mesh.RectangularBilinearAdaptDensity(shape=(5, 7))
    image = aa.mesh.RectangularBilinearAdaptImage(
        shape=(5, 7), weight_power=2.0, weight_floor=0.1
    )

    assert density.shape == (5, 7)
    assert image.shape == (5, 7)
    assert image.weight_power == 2.0
    assert image.weight_floor == 0.1

    assert density.interpolator_cls is InterpolatorRectangular
    assert image.interpolator_cls is InterpolatorRectangular

    # The rank CDF has no kernel hyperparameters — only the transform selector.
    assert density.interpolator_kwargs == {"transform": "rank"}
    assert image.interpolator_kwargs == {"transform": "rank"}


def test__bilinear__no_kernel_hyperparameters_in_signature():
    with pytest.raises(TypeError):
        aa.mesh.RectangularBilinearAdaptDensity(shape=(3, 3), bandwidth=0.5)
    with pytest.raises(TypeError):
        aa.mesh.RectangularBilinearAdaptImage(shape=(3, 3), n_knots=128)


def test__bilinear__minimum_shape_raises():
    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularBilinearAdaptDensity(shape=(2, 3))
    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularBilinearAdaptImage(shape=(3, 2))


def test__bilinear__mesh_weight_map_matches_rtu():
    class _Stub:
        def __init__(self, arr):
            self.array = arr

    adapt = _Stub(np.array([1.0, 2.0, 4.0, 0.0, 8.0]))

    bilinear = aa.mesh.RectangularBilinearAdaptImage(
        shape=(3, 3), weight_power=2.0, weight_floor=0.0
    )
    rtu = aa.mesh.RectangularRTUAdaptImage(
        shape=(3, 3), weight_power=2.0, weight_floor=0.0
    )

    assert bilinear.mesh_weight_map_from(adapt_data=adapt) == pytest.approx(
        rtu.mesh_weight_map_from(adapt_data=adapt), rel=1e-12
    )

    density = aa.mesh.RectangularBilinearAdaptDensity(shape=(3, 3))
    assert density.mesh_weight_map_from(adapt_data=None) is None


def test__bilinear__split_regularization_not_supported():
    assert (
        aa.mesh.RectangularBilinearAdaptDensity(
            shape=(15, 15)
        ).supports_split_regularization
        is False
    )
    assert (
        aa.mesh.RectangularBilinearAdaptImage(
            shape=(15, 15)
        ).supports_split_regularization
        is False
    )


def test__small_datasets__mesh_shape_is_capped_per_axis(monkeypatch):
    # `Grid2D.uniform` and `Mask2D.circular` cap the DATA to 16x16 under
    # PYAUTO_SMALL_DATASETS=1 while a pixelization's `shape=` did not, so a
    # capped run reconstructed 1600-2500 source pixels from ~80 image pixels --
    # an inversion far larger than the data it fits.
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    for cls in [
        aa.mesh.RectangularUniform,
        aa.mesh.RectangularRTUAdaptDensity,
        aa.mesh.RectangularRTUAdaptImage,
        aa.mesh.RectangularBilinearAdaptDensity,
        aa.mesh.RectangularBilinearAdaptImage,
    ]:
        mesh = cls(shape=(50, 40))

        assert mesh.shape == (16, 16)
        assert mesh.pixels == 16 * 16

        # Per axis with `min`, never rewritten to the cap: a mesh already
        # smaller than the cap is a mesh the script chose.
        assert cls(shape=(8, 30)).shape == (8, 16)

        # The escape hatch, for a mesh whose resolution is load-bearing.
        assert cls(shape=(50, 40), respect_small_datasets=False).shape == (50, 40)


def test__small_datasets_unset__mesh_shape_is_untouched(monkeypatch):
    monkeypatch.delenv("PYAUTO_SMALL_DATASETS", raising=False)

    for cls in [
        aa.mesh.RectangularUniform,
        aa.mesh.RectangularRTUAdaptDensity,
        aa.mesh.RectangularRTUAdaptImage,
        aa.mesh.RectangularBilinearAdaptDensity,
        aa.mesh.RectangularBilinearAdaptImage,
    ]:
        mesh = cls(shape=(50, 40))

        assert mesh.shape == (50, 40)
        assert mesh.pixels == 50 * 40


def test__small_datasets__cap_still_respects_the_3x3_minimum(monkeypatch):
    # The cap only ever shrinks, and the mesh's own floor still applies to what
    # the caller asked for -- a capped run must not turn a 2x2 request into a
    # silently valid mesh.
    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    with pytest.raises(aa.exc.MeshException):
        aa.mesh.RectangularUniform(shape=(2, 2))
