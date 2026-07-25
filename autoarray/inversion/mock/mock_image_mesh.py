import numpy as np
from typing import Optional

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autoarray.inversion.mesh.image_mesh.abstract import AbstractImageMesh


class MockImageMesh(AbstractImageMesh):
    def __init__(self, image_plane_mesh_grid=None):
        super().__init__()

        self.image_plane_mesh_grid = image_plane_mesh_grid
