import numpy as np
from typing import Optional

from autoarray import exc
from autoarray.settings import Settings
from autoarray.inversion.mesh.border_relocator import BorderRelocator
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


class AbstractMesh:
    supports_split_regularization = True
    """
    Whether this mesh supports "split" regularization schemes (e.g. ``ConstantSplit``, ``AdaptSplit``,
    ``AdaptSplitZeroth``).

    Split schemes regularize using a split-cross calculation, which requires the mesh's interpolator to
    provide ``_mappings_sizes_weights_split``. The adaptive meshes (``Delaunay``, ``KNNBarycentric``)
    compute this; the rectangular meshes do not, and set this to ``False`` so that ``Pixelization``
    rejects the combination at construction rather than failing deep inside the inversion.
    """

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ is other.__class__

    def relocated_grid_from(
        self, border_relocator: BorderRelocator, source_plane_data_grid: Grid2D, xp=np
    ) -> Grid2D:
        """
        Relocates all coordinates of the input `source_plane_data_grid` that are outside of a
        border (which is defined by a grid of (y,x) coordinates) to the edge of this border.

        The border is determined from the mask of the 2D data in the `data` frame before any transformations of the
        data's grid are performed. The border is all pixels in this mask that are pixels at its extreme edge. These
        pixel indexes are used to then determine a grid of (y,x) coordinates from the transformed `source_grid_grid` in
        the `source` reference frame, whereby points located outside of it are relocated to the border's edge.

        A full description of relocation is given in the method grid_2d.relocated_grid_from()`.

        This is used in the project PyAutoLens to relocate the coordinates that are ray-traced near the centre of mass
        of galaxies, which are heavily demagnified and may trace to outskirts of the source-plane well beyond the
        border.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D (y,x) grid of coordinates, whose coordinates outside the border are relocated to its edge.
        """
        if border_relocator is not None:
            return border_relocator.relocated_grid_from(
                grid=source_plane_data_grid, xp=xp
            )

        return Grid2D(
            values=source_plane_data_grid.array,
            mask=source_plane_data_grid.mask,
            over_sample_size=source_plane_data_grid.over_sampler.sub_size,
            over_sampled=source_plane_data_grid.over_sampled,
            over_sampler=source_plane_data_grid.over_sampler,
            xp=xp,
        )

    def _validate_source_plane_mesh_grid(self, source_plane_mesh_grid):
        """
        Raise if the mesh was given no source-plane mesh grid.

        The adaptive meshes (``Delaunay``, ``KNearestNeighbor``, ``KNNBarycentric``) do
        not compute their own image-plane mesh grid — it is a required input, supplied
        in PyAutoGalaxy / PyAutoLens through ``adapt_images``. Omitting it leaves this
        grid ``None`` and the failure previously landed several frames deeper as
        ``AttributeError: 'NoneType' object has no attribute 'array'`` inside
        ``border_relocator.py``, naming nothing the caller controls and no file the
        caller has opened.

        This raises at the point the precondition is known to be unmet, and names
        ``adapt_images`` so the message points at the thing the caller actually passes.

        Fail-fast is deliberate rather than having the mesh wire the grid up itself:
        constructing an image-plane mesh grid requires a weighting policy (which is
        exactly what ``adapt_images`` carries), so inventing one here would silently
        pick a science choice on the user's behalf. This matches how the
        rectangular-mesh / split-regularization combination was handled — an explicit
        "you must supply X" exception rather than implementing a missing capability.

        Parameters
        ----------
        source_plane_mesh_grid
            The source-plane mesh grid to check.
        """
        if source_plane_mesh_grid is None:
            raise exc.MeshException(
                f"The mesh `{type(self).__name__}` was not given an image-plane mesh "
                f"grid, so its source-plane mesh grid is None and the pixelization "
                f"cannot be built.\n\n"
                f"This mesh does not compute that grid itself — it is a required "
                f"input, supplied via `adapt_images`:\n\n"
                f"    adapt_images = al.AdaptImages(\n"
                f"        galaxy_image_plane_mesh_grid_dict={{source: image_plane_mesh_grid}}\n"
                f"    )\n"
                f"    fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)\n\n"
                f"See the `pixelization` feature scripts in the workspace (e.g. "
                f"`imaging/features/pixelization/delaunay.py`) for the full idiom. A "
                f"mesh in the rectangular family (e.g. `RectangularUniform`) builds "
                f"its own grid and needs no `adapt_images`."
            )

    def relocated_mesh_grid_from(
        self,
        border_relocator: Optional[BorderRelocator],
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        xp=np,
    ):
        """
        Relocates all coordinates of the input `source_plane_mesh_grid` that are outside of a border (which
        is defined by a grid of (y,x) coordinates) to the edge of this border.

        The border is determined from the mask of the 2D data in the `data` frame before any transformations of the
        data's grid are performed. The border is all pixels in this mask that are pixels at its extreme edge. These
        pixel indexes are used to then determine a grid of (y,x) coordinates from the transformed `source_grid_grid` in
        the `source` reference frame, whereby points located outside of it are relocated to the border's edge.

        A full description of relocation is given in the method grid_2d.relocated_grid_from()`.

        This is used in the project `PyAutoLens` to relocate the coordinates that are ray-traced near the centre of mass
        of galaxies, which are heavily demagnified and may trace to outskirts of the source-plane well beyond the
        border.

        Parameters
        ----------
        border_relocator
           The border relocator, which relocates coordinates outside the border of the source-plane data grid to its
           edge.
        source_plane_data_grid
            A 2D grid of (y,x) coordinates associated with the unmasked 2D data after it has been transformed to the
            `source` reference frame.
        source_plane_mesh_grid
            The centres of every pixel in the `source` frame, which are initially derived by computing a sparse
            set of (y,x) coordinates computed from the unmasked data in the `data` frame and applying a transformation
            to this.
        """
        if border_relocator is not None:
            return border_relocator.relocated_mesh_grid_from(
                grid=source_plane_data_grid, mesh_grid=source_plane_mesh_grid, xp=xp
            )
        return source_plane_mesh_grid

    def interpolator_from(
        self,
        source_plane_data_grid: Grid2D,
        source_plane_mesh_grid: Grid2DIrregular,
        border_relocator: Optional[BorderRelocator] = None,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        raise NotImplementedError

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def __repr__(self):
        return "{}\n{}".format(self.__class__.__name__, str(self))
