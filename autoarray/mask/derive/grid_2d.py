from __future__ import annotations
import logging
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D
    from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.structures.grids import grid_2d_util


logging.basicConfig()
logger = logging.getLogger(__name__)


class DeriveGrid2D:

    def __init__(self, mask: Mask2D, xp=np):
        """
        Derives ``Grid2D`` objects from a ``Mask2D``.

        A ``Mask2D`` masks values which are associated with a uniform 2D rectangular grid of pixels, where unmasked
        entries (which are ``False``) are used in subsequent calculations and masked values (which are ``True``) are
        omitted (for a full description see
        the :meth:`Mask2D class API documentation <autoarray.mask.mask_2d.Mask2D.__new__>`).

        From a ``Mask2D``,``Grid2D``s can be derived, which represent the (y,x) Cartesian coordinates of a subset of
        pixels with significance.

        For example:

        - An ``edge`` ``Grid2D``: which is the (y,x) coordinates wherever the ``Mask2D`` that it is derived from
          neighbors at least one ``True`` value (i.e. they are on the edge of the original mask).

        Parameters
        ----------
        mask
            The ``Mask2D`` from which new ``Grid2D`` objects are derived.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                    [True,  True,  True,  True, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True,  True,  True,  True, True],
                ],
                pixel_scales=1.0,
            )

            derive_grid_2d = aa.DeriveGrid2D(mask=mask_2d)

            print(derive_grid_2d.border)
        """
        self.mask = mask

        self.use_jax = xp is not np

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    def tree_flatten(self):
        return (self.mask,), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(mask=children[0])

    @property
    def all_false(self) -> Grid2D:
        """
        Returns a ``Grid2D`` which uses the ``Mask2D``
        geometry (``shape_native`` / ``pixel_scales`` / ``origin``) and every pixel in the ``Mask2D``
        irrespective of whether pixels are masked or unmasked (given by ``True`` or``False``).

        For example, for the following ``Mask2D``:

        ::
            mask_2d = aa.Mask2D(
                mask=[
                     [False, False],
                     [ True,  True]
                ],
                pixel_scales=1.0,
            )

        The ``all_false`` ``Grid2D`` (given via ``mask_2d.derive_grid.all_false``) is:

        ::
            [[ 0.5, -0.5], [ 0.5,  0.5],
             [-0.5, -0.5], [-0.5, -0.5]]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [False, False],
                     [ True,  True]
                ],
                pixel_scales=1.0,
            )

            derive_grid_2d = aa.DeriveGrid2D(mask=mask_2d)

            print(derive_grid_2d.all_false)
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_slim = grid_2d_util.grid_2d_slim_via_shape_native_from(
            shape_native=self.mask.shape,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

        return Grid2D(values=grid_slim, mask=self.mask.derive_mask.all_false)

    @property
    def unmasked(self) -> Grid2D:
        """
        Returns a ``Grid2D`` which uses the ``Mask2D``
        geometry (``shape_native`` / ``pixel_scales`` / ``origin``) and every unmasked
        pixel (given by ``False``), such that all masked entries (given by ``True``) are removed.

        For example, for the following ``Mask2D``:

        ::
            mask_2d = aa.Mask2D(
                mask=[
                     [False, False],
                     [ True,  True]
                ],
                pixel_scales=1.0,
            )

        The ``masked`` ``Grid2D`` (given via ``mask_2d.derive_grid.unmasked``) is:

        ::
            [[0.5, -0.5], [0.5, 0.5]]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [False, False],
                     [ True,  True]
                ],
                pixel_scales=1.0,
            )

            derive_grid_2d = aa.DeriveGrid2D(mask=mask_2d)

            print(derive_grid_2d.unmasked)
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        grid_2d = grid_2d_util.grid_2d_slim_via_mask_from(
            mask_2d=self.mask,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
            xp=self._xp,
        )
        return Grid2D(values=grid_2d, mask=self.mask)

    @property
    def edge(self) -> Grid2D:
        """
        Returns an edge ``Grid2D``, which uses all unmasked pixels (given by ``False``) which neighbor
        any masked value (give by ``True``) and therefore are on the edge of the 2D mask.

        For example, for the following ``Mask2D``:

        ::
            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

        The ``edge`` grid (given via ``mask_2d.derive_grid.edge``) is given by:

        ::
              [[1.0, -1.0], [ 1.0, 0.0], [ 1.0,  1.0],
               [0.0, -1.0],              [ 0.0,  1.0],
              [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0]]

        The central pixel, which does not neighbor a ``True`` value in any one of the eight neighboring directions,
        is not on the 2D grid.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True,  True,  True,  True, True]],
                pixel_scales=1.0,
            )

            derive_grid_2d = aa.DeriveGrid2D(mask=mask_2d)

            print(derive_grid_2d.edge)
        """

        from autoarray.structures.grids.uniform_2d import Grid2D

        edge_grid_1d = self.unmasked[self.mask.derive_indexes.edge_slim]
        return Grid2D(
            values=edge_grid_1d,
            mask=self.mask.derive_mask.edge,
        )

    @property
    def border(self) -> Grid2D:
        """
        Returns a border ``Grid2D``, which uses all unmasked pixels (given by ``False``) which neighbor
        any masked value (give by ``True``) and which are on the extreme exterior of the mask.

        For example, for the following ``Mask2D``:

        ::
            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True,  True,  True,  True,  True, True],
                      [True, False, False, False, False, False, False, False, True],
                      [True, False,  True,  True,  True,  True,  True, False, True],
                      [True, False,  True, False, False, False,  True, False, True],
                      [True, False,  True, False,  True, False,  True, False, True],
                      [True, False,  True, False, False, False,  True, False, True],
                      [True, False,  True,  True,  True,  True,  True, False, True],
                      [True, False, False, False, False, False, False, False, True],
                      [True,  True,  True,  True,  True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

        The ``border`` grid (given via ``mask_2d.derive_grid.border``) is given by:

        ::
            [[3.0, -3.0],  [3.0, -2.0],  [3.0, -1.0],  [3.0, 0.0],  [3.0, 1.0],  [3.0, 2.0],  [ 3.0, 3.0],
             [ 2.0, -3.0],                                                                    [ 2.0, 3.0],
             [ 1.0, -3.0],                                                                    [ 1.0, 3.0],
             [ 0.0, -3.0],                                                                    [ 0.0, 3.0],
             [-1.0, -3.0],                                                                    [-1.0, 3.0],
             [-2.0, -3.0],                                                                    [-2.0, 3.0],
             [-3.0, -3.0], [-3.0, -2.0], [-3.0, -1.0], [-3.0, 0.0], [-3.0, 1.0], [-3.0, 2.0], [-3.0, 3.0]]

        The central group of pixels, which neighbor ``True`` values, are omitted because they are not at an extreme
        edge of the mask.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True,  True,  True,  True,  True, True],
                      [True, False, False, False, False, False, False, False, True],
                      [True, False,  True,  True,  True,  True,  True, False, True],
                      [True, False,  True, False, False, False,  True, False, True],
                      [True, False,  True, False,  True, False,  True, False, True],
                      [True, False,  True, False, False, False,  True, False, True],
                      [True, False,  True,  True,  True,  True,  True, False, True],
                      [True, False, False, False, False, False, False, False, True],
                      [True,  True,  True,  True,  True,  True,  True,  True, True]],
                pixel_scales=1.0,
            )

            derive_grid_2d = aa.DeriveGrid2D(mask=mask_2d)

            print(derive_grid_2d.border)
        """
        from autoarray.structures.grids.uniform_2d import Grid2D

        border = self.unmasked[self.mask.derive_indexes.border_slim]
        return Grid2D(
            values=border,
            mask=self.mask.derive_mask.border,
        )
