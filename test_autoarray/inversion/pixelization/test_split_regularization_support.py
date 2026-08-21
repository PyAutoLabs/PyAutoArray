"""
Regression tests for @rhayes777's audit finding in PyAutoArray#332.

Rectangular meshes do not support split regularization -- their interpolators provide no
split-cross mappings. Before the fix this surfaced two different ways, both deep inside the
inversion and neither naming the real cause:

- ``RectangularUniform``  -> ``AttributeError: 'InterpolatorRectangularUniform' object has no
                              attribute '_mappings_sizes_weights_split'``
- ``RectangularRTUAdaptDensity`` / ``RectangularRTUAdaptImage``
                          -> ``IndexError: index 4 is out of bounds for axis 0 with size 4``
                             (``InterpolatorRectangular`` returned the plain 4-corner mappings
                             from a pass-through that claimed split "reuses the same mappings")

``Pixelization`` now rejects the combination at construction. These tests assert the *clear
failure*, not a successful fit -- the capability is deliberately absent.
"""

import pytest

import autoarray as aa
from autoarray import exc

RECTANGULAR_MESHES = [
    aa.mesh.RectangularUniform,
    aa.mesh.RectangularBilinearAdaptDensity,
    aa.mesh.RectangularBilinearAdaptImage,
    aa.mesh.RectangularRTUAdaptDensity,
    aa.mesh.RectangularRTUAdaptImage,
]

SPLIT_REGULARIZATIONS = [
    aa.reg.ConstantSplit,
    aa.reg.AdaptSplit,
    aa.reg.AdaptSplitZeroth,
]

ADAPTIVE_MESHES = [
    aa.mesh.Delaunay,
    aa.mesh.DelaunayNN,
    aa.mesh.KNNBarycentric,
]


@pytest.mark.parametrize("mesh_cls", RECTANGULAR_MESHES)
@pytest.mark.parametrize("regularization_cls", SPLIT_REGULARIZATIONS)
def test__rectangular_mesh_with_split_regularization__raises(
    mesh_cls, regularization_cls
):
    """All 15 rectangular-mesh x split-regularization combinations are rejected."""

    with pytest.raises(exc.PixelizationException) as error:
        aa.Pixelization(
            mesh=mesh_cls(shape=(15, 15)),
            regularization=regularization_cls(),
        )

    message = str(error.value)

    # the message must name both sides, so the user can act on it without a traceback
    assert mesh_cls.__name__ in message
    assert regularization_cls.__name__ in message


@pytest.mark.parametrize("mesh_cls", RECTANGULAR_MESHES)
def test__rectangular_mesh_with_non_split_regularization__is_allowed(mesh_cls):
    """The guard is specific to split schemes; `Constant` on the same meshes still builds."""

    pixelization = aa.Pixelization(
        mesh=mesh_cls(shape=(15, 15)),
        regularization=aa.reg.Constant(coefficient=1.0),
    )

    assert isinstance(pixelization.mesh, mesh_cls)


@pytest.mark.parametrize("mesh_cls", ADAPTIVE_MESHES)
@pytest.mark.parametrize("regularization_cls", SPLIT_REGULARIZATIONS)
def test__adaptive_mesh_with_split_regularization__is_allowed(
    mesh_cls, regularization_cls
):
    """Split regularization remains supported on the meshes that implement it."""

    pixelization = aa.Pixelization(
        mesh=mesh_cls(pixels=100),
        regularization=regularization_cls(),
    )

    assert isinstance(pixelization.mesh, mesh_cls)


@pytest.mark.parametrize("mesh_cls", RECTANGULAR_MESHES)
def test__rectangular_mesh_without_regularization__is_allowed(mesh_cls):
    """`regularization` is optional; a `None` value must not trip the guard."""

    pixelization = aa.Pixelization(mesh=mesh_cls(shape=(15, 15)))

    assert pixelization.regularization is None


def test__capability_flags():
    """The flags the guard reads, asserted directly so a future mesh can't silently regress."""

    assert (
        aa.mesh.RectangularUniform(shape=(15, 15)).supports_split_regularization
        is False
    )
    assert (
        aa.mesh.RectangularRTUAdaptDensity(shape=(15, 15)).supports_split_regularization
        is False
    )
    assert (
        aa.mesh.RectangularRTUAdaptImage(shape=(15, 15)).supports_split_regularization
        is False
    )
    assert aa.mesh.Delaunay(pixels=100).supports_split_regularization is True
    assert aa.mesh.DelaunayNN(pixels=100).supports_split_regularization is True
    assert aa.mesh.KNNBarycentric(pixels=100).supports_split_regularization is True

    assert aa.reg.ConstantSplit().is_split_regularization is True
    assert aa.reg.AdaptSplit().is_split_regularization is True
    assert aa.reg.AdaptSplitZeroth().is_split_regularization is True
    assert aa.reg.Constant().is_split_regularization is False
    assert aa.reg.Adapt().is_split_regularization is False


def test__interpolator_rectangular_has_no_split_mappings():
    """
    The pass-through that produced the `IndexError` is gone and must stay gone -- restoring it
    would reintroduce a silent-looking API that fails one frame later.
    """

    from autoarray.inversion.mesh.interpolator.rectangular import (
        InterpolatorRectangular,
    )
    from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
        InterpolatorRectangularUniform,
    )
    from autoarray.inversion.mesh.interpolator.delaunay import InterpolatorDelaunay
    from autoarray.inversion.mesh.interpolator.sibson import InterpolatorDelaunayNN

    assert not hasattr(InterpolatorRectangular, "_mappings_sizes_weights_split")
    assert not hasattr(InterpolatorRectangularUniform, "_mappings_sizes_weights_split")
    assert hasattr(InterpolatorDelaunay, "_mappings_sizes_weights_split")
    assert hasattr(InterpolatorDelaunayNN, "_mappings_sizes_weights_split")


def test__pixelization_exception_is_not_a_fit_exception():
    """
    An unsupported combination is a configuration error, not a bad model sample.

    `autogalaxy.exc.PixelizationException` subclasses `af.exc.FitException`, which
    `fitness.py` converts into a resample-reject. If the exception raised here were a
    `FitException`, a search would silently reject every sample instead of reporting the
    misconfiguration. `autoarray`'s own `PixelizationException` must stay a plain `Exception`.
    """

    assert issubclass(exc.PixelizationException, Exception)

    af = pytest.importorskip("autofit")
    assert not issubclass(exc.PixelizationException, af.exc.FitException)
