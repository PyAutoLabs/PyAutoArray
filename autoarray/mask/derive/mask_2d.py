from __future__ import annotations
import logging
import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from autoarray.mask.mask_2d import Mask2D

from autoarray import exc
from autoarray.mask.derive.indexes_2d import DeriveIndexes2D

from autoarray.mask import mask_2d_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class DeriveMask2D:
    def __init__(self, mask: Mask2D):
        """
        Derives ``Mask2D`` objects from a ``Mask2D``.

        A ``Mask2D`` masks values which are associated with a uniform 2D rectangular grid of pixels, where unmasked
        entries (which are ``False``) are used in subsequent calculations and masked values (which are ``True``) are
        omitted (for a full description see
        the :meth:`Mask2D class API documentation <autoarray.mask.mask_2d.Mask2D.__new__>`).

        From a ``Mask2D`` other ``Mask2D``s can be derived, which represent a subset of pixels with significance.
        For example:

        - An ``edge`` ``Mask2D``: has unmasked ``False`` values wherever the ``Mask2D`` that it is derived from
          neighbors at least one ``True`` value (i.e. they are on the edge of the original mask).

        - A ``recaled`` ``Mask``: The original mask rescaled to a larger or small 2D shape.

        Parameters
        ----------
        mask
            The ``Mask2D`` from which new ``Mask2D`` objects are derived.

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

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.edge)
        """
        self.mask = mask

    @property
    def derive_indexes(self) -> DeriveIndexes2D:
        """
        Derives ``ndarrays``s of useful indexes from a ``Mask2D``, which are often used in order to create the
        derived ``Mask2D`` objects.

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

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.derive_indexes.edge)
        """
        return self.mask.derive_indexes

    @property
    def all_false(self) -> Mask2D:
        """
        Returns a ``Mask2D`` which has the same
        geometry (``shape_native`` / ``pixel_scales`` / ``origin``) as this ``Mask2D`` but all
        entries are unmasked (given by``False``).

        For example, for the following ``Mask2D``:

        ::
            [[True,  True],
            [False, False]]

        The unmasked mask (given via ``mask_2d.derive_mask.all_false``) is:

        ::
            [[False, False],
            [False, False]]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[
                     [ True,  True],
                     [False, False]
                ],
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.all_false)
        """

        from autoarray.mask.mask_2d import Mask2D

        return Mask2D.all_false(
            shape_native=self.mask.shape_native,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    def blurring_from(
        self, kernel_shape_native: Tuple[int, int], allow_padding: bool = False
    ) -> Mask2D:
        """
        Returns a blurring ``Mask2D``, representing all masked pixels (given by ``True``) whose values are blurred
        into unmasked pixels (given by ``False``) when a 2D convolution is performed.

        This mask is used by the PSF to ensure that 2D convolution can be performed on masked data structures without
        missing values.

        For example, for the following ``Mask2D``:

        ::

            [[True,  True,  True,  True, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True,  True,  True,  True, True]]

        The blurring-mask for `kernel_shape_native=(3,3)` (given
        via ``mask_2d.derive_mask.blurring_from(kernel_shape_native=(3,3)``) is:

        ::

            [[False, False, False, False, False],
             [False,  True,  True,  True, False],
             [False,  True,  True,  True, False],
             [False,  True,  True,  True, False],
             [False, False, False, False, False]]

        Parameters
        ----------
        kernel_shape_native
           The 2D shape of the 2D convolution ``Convolver`` which defines the blurring region.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            blurring_mask = derive_mask_2d.blurring_from(kernel_shape_native=(3,3))

            print(blurring_mask)
        """

        from autoarray.mask.mask_2d import Mask2D

        blurring_mask = mask_2d_util.blurring_mask_2d_from(
            mask_2d=self.mask,
            kernel_shape_native=kernel_shape_native,
            allow_padding=allow_padding,
        )

        return Mask2D(
            mask=blurring_mask,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def edge(self) -> Mask2D:
        """
        Returns an edge ``Mask2D``, representing all unmasked pixels (given by ``False``) which neighbor any masked
        value (give by ``True``) and therefore are on the edge of the 2D mask.

        For example, for the following ``Mask2D``:

        ::
            [[True,  True,  True,  True, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True,  True,  True,  True, True]]

        The edge-mask (given via ``mask_2d.derive_mask.edge``) is given by:

        ::
             [[True,  True,  True,  True, True],
              [True, False, False, False, True],
              [True, False,  True, False, True],
              [True, False, False, False, True],
              [True, True,   True, True, True]]

        The central pixel, which does not neighbor a ``True`` value in any one of the eight neighboring directions,
        is switched to masked.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.edge)
        """

        from autoarray.mask.mask_2d import Mask2D

        mask = np.full(fill_value=True, shape=self.mask.shape)
        mask[
            self.derive_indexes.edge_native[:, 0], self.derive_indexes.edge_native[:, 1]
        ] = False
        return Mask2D(
            mask=mask,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def edge_buffed(self) -> Mask2D:
        """
        Returns a buffed edge ``Mask2D``, representing all unmasked pixels (given by ``False``) which neighbor any
        masked value (give by ``True``) and therefore are on the edge of the 2D mask, but with a buffer of 1 pixel
        applied such that everu pixel 1 pixel further out are included.

        For example, for the following ``Mask2D``:

        ::
            [[True,  True,  True,  True, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True, False, False, False, True],
             [True,  True,  True,  True, True]]

        The edge-mask (given via ``mask_2d.derive_mask.edge``) is given by:

        ::
             [[False, False, False,  False, False],
              [False, False, False,  False, False],
              [False, False,  True,  False, False],
              [False, False, False,  False, False],
              [False, False,  False, False, False]]

        The central pixel, which does not neighbor a ``True`` value in any one of the eight neighboring directions,
        is switched to masked.

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True,  True, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True, False, False, False, True],
                      [True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.edge_buffed)
        """
        from autoarray.mask.mask_2d import Mask2D

        edge_buffed_mask = mask_2d_util.buffed_mask_2d_from(mask_2d=self.mask).astype(
            "bool"
        )

        return Mask2D(
            mask=edge_buffed_mask,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )

    @property
    def border(self) -> Mask2D:
        """
        Returns border ``Mask2D``, representing all unmasked pixels (given by ``False``) which neighbor any masked
        value (give by ``True``) and which are on the extreme exterior of the mask.

        This is therefore close related to the edge mask, but may not include central pixels unlike the edge mask.

        For example, for the following ``Mask2D``:

        ::
            [[True,  True,  True,  True,  True,  True,  True,  True, True],
             [True, False, False, False, False, False, False, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True, False, False, False,  True, False, True],
             [True, False,  True, False,  True, False,  True, False, True],
             [True, False,  True, False, False, False,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False, False, False, False, False, False, False, True],
             [True,  True,  True,  True,  True,  True,  True,  True, True]]

        The border-mask (given via ``mask_2d.derive_mask.edge``) is given by:

        ::
            [[True,  True,  True,  True,  True,  True,  True,  True, True],
             [True, False, False, False, False, False, False, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False,  True,  True,  True,  True,  True, False, True],
             [True, False, False, False, False, False, False, False, True],
             [True,  True,  True,  True,  True,  True,  True,  True, True]]

        The central ``False`` pixels would be included in an edge-mask, but are not included in the border mask
        because they are not on the extreme outer edge.

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
                      [True,  True,  True,  True,  True,  True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_mask_2d = aa.DeriveMask2D(mask=mask_2d)

            print(derive_mask_2d.border)
        """

        from autoarray.mask.mask_2d import Mask2D

        mask = np.full(fill_value=True, shape=self.mask.shape)
        mask[
            self.derive_indexes.border_native[:, 0],
            self.derive_indexes.border_native[:, 1],
        ] = False
        return Mask2D(
            mask=mask,
            pixel_scales=self.mask.pixel_scales,
            origin=self.mask.origin,
        )
