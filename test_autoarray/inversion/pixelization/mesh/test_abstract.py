import autoarray as aa


def test__grid_is_relocated_via_border(grid_2d_7x7):
    mesh = aa.mesh.Delaunay(pixels=9)

    mask = aa.Mask2D.circular(
        shape_native=(60, 60),
        radius=1.0,
        pixel_scales=(0.1, 0.1),
        centre=(1.0, 1.0),
    )

    grid = aa.Grid2D.from_mask(mask=mask)

    image_mesh = aa.image_mesh.Overlay(
        shape=(3, 3),
    )
    image_mesh = image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=None)

    image_mesh = aa.Grid2DIrregular(image_mesh)

    grid[8, 0] = 100.0

    border_relocator = aa.BorderRelocator(mask=mask, sub_size=1)

    interpolator = mesh.interpolator_from(
        border_relocator=border_relocator,
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh,
    )

    assert grid[8, 0] != interpolator.data_grid[8, 0]
    assert interpolator.data_grid[8, 0] < 5.0

    grid[0, 0] = 0.0
    image_mesh[0, 0] = 100.0

    border_relocator = aa.BorderRelocator(mask=mask, sub_size=1)

    interpolator = mesh.interpolator_from(
        border_relocator=border_relocator,
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh,
    )

    assert image_mesh[0, 0] != interpolator.mesh_grid[0, 0]
    assert interpolator.mesh_grid[0, 0] < 5.0

    mesh = aa.mesh.Delaunay(pixels=9)

    border_relocator = aa.BorderRelocator(mask=mask, sub_size=1)

    interpolator = mesh.interpolator_from(
        border_relocator=border_relocator,
        source_plane_data_grid=grid,
        source_plane_mesh_grid=image_mesh,
    )

    assert image_mesh[0, 0] != interpolator.mesh_grid[0, 0]
    assert interpolator.mesh_grid[0, 0] < 5.0
