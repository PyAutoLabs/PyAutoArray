"""Rotated-CDF variant of the spline rectangular interpolator.

Used by :class:`RectangularRotatedAdaptImage`. The mesh class rotates
``data_grid`` into a brightness-weighted PCA frame before constructing this
interpolator; everything else (bilinear scatter, CDF spline, mappings/sizes/
weights) runs unchanged inside that frame.

The only behavioural difference vs the parent
:class:`InterpolatorRectangularSpline` is the geometry: ``mesh_geometry`` is
replaced by :class:`MeshGeometryRectangularRotated`, which un-rotates the
CDF-warped edges back into the source frame for plotting.
"""
from typing import Optional

import numpy as np

from autoconf import cached_property

from autoarray.inversion.mesh.interpolator.rectangular_spline import (
    SPLINE_CDF_DEFAULT_DEG,
    InterpolatorRectangularSpline,
)


class InterpolatorRectangularRotatedSpline(InterpolatorRectangularSpline):
    """Spline-CDF rectangular interpolator with brightness-weighted PCA rotation.

    Constructed by :class:`RectangularRotatedAdaptImage` with a pre-rotated
    ``data_grid`` shim. Carries the rotation matrix and centroid so the
    rotated geometry can un-rotate edges for plotting.
    """

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        mesh_weight_map,
        rotation_matrix,
        rotation_centroid,
        adapt_data: Optional[np.ndarray] = None,
        spline_deg: int = SPLINE_CDF_DEFAULT_DEG,
        xp=np,
    ):
        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid=data_grid,
            mesh_weight_map=mesh_weight_map,
            adapt_data=adapt_data,
            spline_deg=spline_deg,
            xp=xp,
        )
        self.rotation_matrix = rotation_matrix
        self.rotation_centroid = rotation_centroid

    @cached_property
    def mesh_geometry(self):
        from autoarray.inversion.mesh.mesh_geometry.rectangular_rotated import (
            MeshGeometryRectangularRotated,
        )

        return MeshGeometryRectangularRotated(
            mesh=self.mesh,
            mesh_grid=self.mesh_grid,
            data_grid=self.data_grid,
            mesh_weight_map=self.mesh_weight_map,
            spline_deg=self.spline_deg,
            rotation_matrix=self.rotation_matrix,
            rotation_centroid=self.rotation_centroid,
            xp=self._xp,
        )
