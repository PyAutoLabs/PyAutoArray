from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask1D
    from autoarray.structures.grids.uniform_1d import Grid1D

from autoarray.structures.grids import grid_1d_util


logging.basicConfig()
logger = logging.getLogger(__name__)


class DeriveGrid1D:
    def __init__(self, mask: Mask1D):
        """
        Derives ``Grid1D`` objects from a ``Mask1D``.

        A ``Mask1D`` masks values which are associated with a 1D uniform grid of pixels, where unmasked
        entries (which are ``False``) are used in subsequent calculations and masked values (which are ``True``) are
        omitted (for a full description see
        the :meth:`Mask1D class API documentation <autoarray.mask.mask_1d.Mask1D.__new__>`).

        From a ``Mask1D``, ``Grid1D``s can be derived, which represent the (x,) Cartesian coordinates of a subset of
        pixels with significance.

        For example:

        - An ``all_false`` ``Grid1D``: the (x,) coordinates of every pixel in the ``Mask1D`` regardless of whether
          each pixel is masked or unmasked.

        Parameters
        ----------
        mask
            The ``Mask1D`` from which new ``Grid1D`` objects are derived.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_1d = aa.Mask1D(
                mask=[True, False, False, False, True],
                pixel_scales=1.0,
            )

            derive_grid_1d = aa.DeriveGrid1D(mask=mask_1d)

            print(derive_grid_1d.all_false)
        """
        self.mask = mask

    @property
    def all_false(self) -> Grid1D:
        """
        Returns a ``Grid1D`` which uses the ``Mask1D``
        geometry (``shape_native`` / ``pixel_scales`` / ``origin``) and includes every pixel in the ``Mask1D``,
        irrespective of whether pixels are masked or unmasked (given by ``True`` or ``False``).

        For example, for the following ``Mask1D``:

        ::
            mask_1d = aa.Mask1D(
                mask=[False, False, True,  True],
                pixel_scales=1.0,
            )

        The ``all_false`` ``Grid1D`` (given via ``mask_1d.derive_grid.all_false``) is:

        ::
            [-1.5, -0.5, 0.5, 1.5]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_1d = aa.Mask1D(
                mask=[False, False, True,  True],
                pixel_scales=1.0,
            )

            derive_grid_1d = aa.DeriveGrid1D(mask=mask_1d)

            print(derive_grid_1d.all_false)
        """
        from autoarray.structures.grids.uniform_1d import Grid1D

        grid_slim = grid_1d_util.grid_1d_slim_via_mask_from(
            mask_1d=self.mask,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

        return Grid1D(values=grid_slim, mask=self.mask.derive_mask.all_false)
