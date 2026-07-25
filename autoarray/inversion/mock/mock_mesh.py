import numpy as np
from typing import Optional

from autoarray.inversion.mock.mock_mapper import MockMapper
from autoarray.mask.mask_2d import Mask2D
from autoarray.inversion.mesh.mesh.abstract import AbstractMesh
from autoarray.inversion.mesh.interpolator.abstract import AbstractInterpolator
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


class MockMesh(AbstractMesh):
    def __init__(self, image_plane_mesh_grid=None):
        super().__init__()

        self.image_plane_mesh_grid = image_plane_mesh_grid

    def interpolator_from(
        self,
        mask=None,
        source_plane_data_grid: Grid2D = None,
        border_relocator=None,
        source_plane_mesh_grid: Optional[AbstractInterpolator] = None,
        image_plane_mesh_grid: Optional[Grid2DIrregular] = None,
        adapt_data: Optional[np.ndarray] = None,
    ):
        return MockMapper(
            mask=mask,
            source_plane_data_grid=source_plane_data_grid,
            border_relocator=border_relocator,
            source_plane_mesh_grid=source_plane_mesh_grid,
            image_plane_mesh_grid=self.image_plane_mesh_grid,
            adapt_data=adapt_data,
        )

    def image_plane_mesh_grid_from(
        self,
        mask: Mask2D,
        adapt_data,
    ):
        if adapt_data is not None and self.image_plane_mesh_grid is not None:
            return adapt_data * self.image_plane_mesh_grid

        return self.image_plane_mesh_grid
