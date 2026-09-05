"""
The `Delaunay` mesh's zeroed edge ring.

The ring is the last `zeroed_pixels` vertices of the mesh grid the mapper is built
from, resolved against that grid's length (`zeroed_pixels_from`) and never against
the `pixels` the mesh was constructed with. Until 2026-09 the constructor inflated
`pixels` by `zeroed_pixels` and derived the ring from the inflated count, so a
caller passing the appended grid length (every workspace script did) produced a
mesh whose `pixels` overstated the mapper's parameter count by the ring size — a
mismatch that cancelled for one mapper and mis-placed the ring for two, and that
broke the `to_dict` round-trip (the inflated `pixels` and an index-array
`zeroed_pixels` were both written to `tracer.json`).
"""

import numpy as np
import pytest

import autoarray as aa
from autonerves.dictable import from_dict, to_dict


def test__pixels_and_zeroed_pixels__stored_as_passed__not_inflated():
    mesh = aa.mesh.Delaunay(pixels=500, zeroed_pixels=30)

    assert mesh.pixels == 500
    assert mesh.zeroed_pixels == 30
    assert mesh.total_pixels == 530

    mesh = aa.mesh.Delaunay(pixels=9)

    assert mesh.pixels == 9
    assert mesh.zeroed_pixels == 0
    assert mesh.total_pixels == 9


def test__zeroed_pixels_from__last_n_vertices_of_the_grid_whatever_pixels_says():
    """
    Both call forms in the wild — `pixels=<interior count>` (correct) and
    `pixels=<appended grid length>` (the workspace form) — must zero the same 30
    vertices: the appended ring at the end of a 530-point grid.
    """
    expected = np.arange(500, 530)

    mesh = aa.mesh.Delaunay(pixels=500, zeroed_pixels=30)
    assert (mesh.zeroed_pixels_from(pixels=530) == expected).all()

    mesh = aa.mesh.Delaunay(pixels=530, zeroed_pixels=30)
    assert (mesh.zeroed_pixels_from(pixels=530) == expected).all()

    mesh = aa.mesh.Delaunay(pixels=25, zeroed_pixels=5)
    assert (mesh.zeroed_pixels_from(pixels=40) == np.arange(35, 40)).all()


def test__zeroed_pixels_from__no_ring__empty():
    mesh = aa.mesh.Delaunay(pixels=9)

    zeroed = mesh.zeroed_pixels_from(pixels=9)

    assert zeroed.shape == (0,)
    assert zeroed.dtype == int

    mesh = aa.mesh.Delaunay(pixels=9, zeroed_pixels=None)

    assert mesh.zeroed_pixels == 0
    assert mesh.zeroed_pixels_from(pixels=9).shape == (0,)


def test__zeroed_pixels_from__subclasses_inherit_the_count_semantics():
    for cls in (aa.mesh.DelaunayNN, aa.mesh.KNearestNeighbor, aa.mesh.KNNBarycentric):
        mesh = cls(pixels=100, zeroed_pixels=10)

        assert mesh.pixels == 100
        assert mesh.zeroed_pixels == 10
        assert (mesh.zeroed_pixels_from(pixels=110) == np.arange(100, 110)).all()


def test__rectangular_mesh__zeroed_pixels_from__is_its_own_edge_ring():
    """
    The rectangular family knows its edge pixels from its shape; `pixels` is ignored.
    """
    mesh = aa.mesh.RectangularUniform(shape=(4, 4))

    assert (mesh.zeroed_pixels_from(pixels=16) == mesh.zeroed_pixels).all()
    assert (
        mesh.zeroed_pixels_from(pixels=16)
        == np.array([0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15])
    ).all()


def test__to_dict__round_trips__pixels_and_zeroed_pixels_are_ints():
    mesh = aa.mesh.Delaunay(pixels=500, zeroed_pixels=30, areas_factor=0.25)

    mesh_dict = to_dict(mesh)

    assert mesh_dict["arguments"]["pixels"] == 500
    assert mesh_dict["arguments"]["zeroed_pixels"] == 30

    mesh_reloaded = from_dict(mesh_dict)

    assert mesh_reloaded == mesh
    assert mesh_reloaded.pixels == 500
    assert mesh_reloaded.zeroed_pixels == 30
    assert (mesh_reloaded.zeroed_pixels_from(pixels=530) == np.arange(500, 530)).all()


def test__mapper__zeroed_pixels__resolved_against_params():
    """
    A mapper over a 530-point grid zeroes the last 30 vertices whether the mesh was
    told `pixels=500` or `pixels=530`.
    """
    for pixels in (500, 530):
        mapper = aa.m.MockMapper(
            mesh=aa.mesh.Delaunay(pixels=pixels, zeroed_pixels=30), parameters=530
        )

        assert (mapper.zeroed_pixels == np.arange(500, 530)).all()

    mapper = aa.m.MockMapper(mesh=aa.mesh.Delaunay(pixels=9), parameters=9)

    assert mapper.zeroed_pixels.shape == (0,)
