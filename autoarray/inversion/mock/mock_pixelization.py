from autoarray.mask.mask_2d import Mask2D
from autoarray.inversion.pixelization import Pixelization


class MockPixelization(Pixelization):
    def __init__(
        self,
        mesh=None,
        regularization=None,
        mapper=None,
        image_plane_mesh_grid=None,
    ):
        super().__init__(mesh=mesh, regularization=regularization)

        self.mapper = mapper
        self.image_plane_mesh_grid = image_plane_mesh_grid

    # noinspection PyUnusedLocal,PyShadowingNames
    def interpolator_from(
        self,
        mask,
        source_plane_data_grid,
        source_plane_mesh_grid,
        image_plane_mesh_grid=None,
        adapt_data=None,
        settings=None,
    ):
        return self.mapper

    def image_plane_mesh_grid_from(
        self,
        mask: Mask2D,
        adapt_data,
        settings=None,
    ):
        if adapt_data is not None and self.image_plane_mesh_grid is not None:
            return adapt_data * self.image_plane_mesh_grid

        return self.image_plane_mesh_grid
