import autoarray as aa


def test__abstract_preloads__all_fields_default_none():
    preloads = aa.AbstractPreloads()

    assert preloads.curvature_matrix is None
    assert preloads.mapper_galaxy_dict is None
    assert preloads.source_plane_mesh_grid is None
    assert preloads.image_plane_mesh_grid is None


def test__preloads_imaging__carries_mesh_geometry_only():
    preloads = aa.PreloadsImaging(
        source_plane_mesh_grid=[["mesh"]], image_plane_mesh_grid=[["image_mesh"]]
    )

    assert preloads.source_plane_mesh_grid == [["mesh"]]
    assert preloads.image_plane_mesh_grid == [["image_mesh"]]
    assert preloads.curvature_matrix is None
    assert preloads.mapper_galaxy_dict is None


def test__preloads_interferometer__mesh_fields_available_via_abstract():
    preloads = aa.PreloadsInterferometer(curvature_matrix="F")

    assert preloads.curvature_matrix == "F"
    assert preloads.source_plane_mesh_grid is None
    assert preloads.image_plane_mesh_grid is None

    preloads = aa.PreloadsInterferometer(
        curvature_matrix="F", source_plane_mesh_grid=[["mesh"]], image_plane_mesh_grid=[["im"]]
    )
    assert preloads.source_plane_mesh_grid == [["mesh"]]
    assert preloads.image_plane_mesh_grid == [["im"]]
