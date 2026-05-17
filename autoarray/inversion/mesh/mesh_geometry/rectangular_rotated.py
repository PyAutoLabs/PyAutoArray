import numpy as np

from autoarray.inversion.mesh.mesh_geometry.rectangular import (
    MeshGeometryRectangular,
)


class MeshGeometryRectangularRotated(MeshGeometryRectangular):
    """Geometry for a rectangular mesh whose CDF is built in a brightness-weighted
    PCA frame instead of the native source-plane frame.

    The CDF transforms (areas, edges) live in the rotated frame; the
    bilinear-scatter math runs unchanged inside that frame. The inversion
    pipeline (mappings / sizes / weights) is correct as-is because every
    consumer of ``data_grid`` works in whatever frame ``data_grid`` already
    occupies.

    The geometry carries the rotation state (``rotation_matrix`` and
    ``rotation_centroid``) so downstream code that wants to plot or analyse
    the mesh in the source frame can convert pixel positions explicitly:

    .. code-block:: python

        y_edges_rot, x_edges_rot = geom.edges_transformed.T
        yy, xx = np.meshgrid(y_edges_rot, x_edges_rot, indexing="ij")
        corners_rot = np.stack([yy.ravel(), xx.ravel()], axis=1)
        corners_src = corners_rot @ geom.rotation_matrix + geom.rotation_centroid

    Note: existing ``pcolormesh``-based pixelization plotting in
    :mod:`autoarray.plot.inversion` calls ``edges_transformed.T`` and draws
    axis-aligned cells. For a rotated mesh those cells appear axis-aligned
    in the *rotated* frame which is visually incorrect when overlaid on
    source-frame data. A rotation-aware plot variant is a known follow-up.
    """

    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        mesh_weight_map=None,
        spline_deg=None,
        rotation_matrix=None,
        rotation_centroid=None,
        xp=np,
    ):
        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid=data_grid,
            mesh_weight_map=mesh_weight_map,
            spline_deg=spline_deg,
            xp=xp,
        )
        if rotation_matrix is None or rotation_centroid is None:
            raise ValueError(
                "MeshGeometryRectangularRotated requires rotation_matrix and "
                "rotation_centroid; use MeshGeometryRectangular for the "
                "non-rotated case."
            )
        self.rotation_matrix = rotation_matrix
        self.rotation_centroid = rotation_centroid
