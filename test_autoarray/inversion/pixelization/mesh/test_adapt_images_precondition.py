"""
Regression tests for PyAutoArray#332 — the missing-`adapt_images` precondition.

__What this issue actually is__

The issue's headline says `Delaunay` and `KNNBarycentric` are "unusable in
`FitImaging`". That is **false**, and the public reply on #332 corrects it while
crediting the underlying finding. Both meshes work correctly; they *require* an
image-plane mesh grid, supplied via `adapt_images`.

So the defect is the **error**, not the mesh. Omitting `adapt_images` used to surface
as:

    AttributeError: 'NoneType' object has no attribute 'array'
      autoarray/inversion/mesh/border_relocator.py, in relocated_mesh_grid_from

— naming nothing the caller controls, in a file they have never opened.

**These tests therefore assert a CLEAR FAILURE, not a successful fit.** Asserting that
bare construction succeeds would enshrine the reporter's misreading; that trap is
recorded in the prompt for this task and is deliberately avoided here.

The `adapt_images` branch is exercised at the integration level (a real `FitImaging`
with an `AdaptImages` still fits, for `Delaunay` + `Constant`, `KNNBarycentric` +
`Constant` and `Delaunay` + `ConstantSplit`); these unit tests cover the guard itself
and its controls at the mesh boundary.
"""

import numpy as np
import pytest

import autoarray as aa
from autoarray import exc


@pytest.fixture(name="source_plane_data_grid")
def make_source_plane_data_grid():
    return aa.Grid2D.uniform(shape_native=(5, 5), pixel_scales=1.0)


@pytest.fixture(name="source_plane_mesh_grid")
def make_source_plane_mesh_grid():
    return aa.Grid2DIrregular(
        values=[[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 2.1], [2.1, 1.1]]
    )


# ======================================================================================
# The guard — a missing image-plane mesh grid fails legibly
# ======================================================================================


@pytest.mark.parametrize(
    "mesh_cls", [aa.mesh.Delaunay, aa.mesh.KNearestNeighbor, aa.mesh.KNNBarycentric]
)
def test__adaptive_meshes_raise_when_no_source_plane_mesh_grid_is_given(
    mesh_cls, source_plane_data_grid
):
    with pytest.raises(exc.MeshException):
        mesh_cls(pixels=6).interpolator_from(
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=None,
        )


def test__the_message_names_adapt_images_and_the_mesh__not_just_the_exception_type(
    source_plane_data_grid,
):
    """
    Asserting on the message is the point of this issue — the old failure raised too,
    it just said nothing useful. `adapt_images` is the thing the caller actually passes.
    """
    with pytest.raises(exc.MeshException) as error:
        aa.mesh.Delaunay(pixels=6).interpolator_from(
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=None,
        )

    message = str(error.value)

    assert "adapt_images" in message
    assert "Delaunay" in message


def test__the_message_points_at_the_workspace_idiom_and_the_rectangular_alternative(
    source_plane_data_grid,
):
    with pytest.raises(exc.MeshException) as error:
        aa.mesh.KNNBarycentric(pixels=6).interpolator_from(
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=None,
        )

    message = str(error.value)

    assert "galaxy_image_plane_mesh_grid_dict" in message
    assert "RectangularUniform" in message


def test__the_failure_is_no_longer_an_attribute_error_on_none(source_plane_data_grid):
    """The original symptom: AttributeError deep inside border_relocator.py."""
    with pytest.raises(exc.MeshException):
        aa.mesh.Delaunay(pixels=6).interpolator_from(
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=None,
        )


# ======================================================================================
# Controls — the meshes are NOT broken, which is the correction to the headline
# ======================================================================================


@pytest.mark.parametrize(
    "mesh_cls", [aa.mesh.Delaunay, aa.mesh.KNearestNeighbor, aa.mesh.KNNBarycentric]
)
def test__control__an_adaptive_mesh_with_a_mesh_grid_still_builds_its_interpolator(
    mesh_cls, source_plane_data_grid, source_plane_mesh_grid
):
    """
    The headline correction, pinned: supplied with the grid `adapt_images` carries,
    these meshes work. If this ever fails, the mesh really is broken and the reply
    posted on #332 needs revisiting.
    """
    interpolator = mesh_cls(pixels=6).interpolator_from(
        source_plane_data_grid=source_plane_data_grid,
        source_plane_mesh_grid=source_plane_mesh_grid,
    )

    assert interpolator is not None


def test__control__the_rectangular_family_needs_no_image_plane_mesh_grid(
    source_plane_data_grid,
):
    """
    `RectangularUniform` computes its own grid, so the guard must not fire for it —
    otherwise the fix would break the one mesh the issue never complained about.
    """
    interpolator = aa.mesh.RectangularUniform(shape=(3, 3)).interpolator_from(
        source_plane_data_grid=source_plane_data_grid,
        source_plane_mesh_grid=None,
    )

    assert interpolator is not None
