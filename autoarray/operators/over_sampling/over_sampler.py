import numpy as np

from typing import Union

from autonerves import conf
from autonerves import cached_property

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D

from autoarray.operators.over_sampling import over_sample_util


class OverSampler:
    def __init__(self, mask: Mask2D, sub_size: Union[int, Array2D]):
        r"""
         Over samples grid calculations using a uniform sub-grid.

        When a 2D grid of (y,x) coordinates is input into a function, the result is evaluated at every coordinate
        on the grid. When the grid is paired to a 2D image (e.g. an `Array2D`) the solution needs to approximate
        the 2D integral of that function in each pixel. Over sample objects define how this over-sampling is performed.

        This object inputs a uniform sub-grid, where every image-pixel is split into a uniform grid of sub-pixels. The
        function is evaluated at every sub-pixel, and the final value in each pixel is computed by summing the
        contribution from all sub-pixels.

        This is the simplest over-sampling method, but may not provide precise solutions for functions that vary
        significantly within a pixel. To achieve precision in these pixels a high `sub_size` is required, which can
        be computationally expensive as it is applied to every pixel.

        **Example**

        If the mask's `sub_size` is > 1, the grid is defined as a sub-grid where each entry corresponds to the (y,x)
        coordinates at the centre of each sub-pixel of an unmasked pixel. The Grid2D is therefore stored as an ndarray
        of shape [total_unmasked_coordinates*sub_size**2, 2]

        The sub-grid indexes are ordered such that pixels begin from the first (top-left) sub-pixel in the first
        unmasked pixel. Indexes then go over the sub-pixels in each unmasked pixel, for every unmasked pixel.
        Therefore, the sub-grid is an ndarray of shape [total_unmasked_coordinates*(sub_grid_shape)**2, 2].

        For example:

        - grid[9, 1] - using a 2x2 sub-grid, gives the 3rd unmasked pixel's 2nd sub-pixel x-coordinate.
        - grid[9, 1] - using a 3x3 sub-grid, gives the 2nd unmasked pixel's 1st sub-pixel x-coordinate.
        - grid[27, 0] - using a 3x3 sub-grid, gives the 4th unmasked pixel's 1st sub-pixel y-coordinate.

        Below is a visual illustration of a sub grid. Indexing of each sub-pixel goes from the top-left corner. In
        contrast to the grid above, our illustration below restricts the mask to just 2 pixels, to keep the
        illustration brief.

        .. code-block:: bash

             x x x x x x x x x x
             x x x x x x x x x x     This is an example mask.Mask2D, where:
             x x x x x x x x x x
             x x x x x x x x x x     x = `True` (Pixel is masked and excluded from lens)
             x x x x O O x x x x     O = `False` (Pixel is not masked and included in lens)
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x
             x x x x x x x x x x

        Our grid with a sub-size looks like it did before:

        .. code-block:: bash

            pixel_scales = 1.0"

            <--- -ve  x  +ve -->

             x x x x x x x x x x  ^
             x x x x x x x x x x  I
             x x x x x x x x x x  I                        y     x
             x x x x x x x x x x +ve  grid[0] = [0.5,  -1.5]
             x x x x 0 1 x x x x  y   grid[1] = [0.5,  -0.5]
             x x x x x x x x x x -ve
             x x x x x x x x x x  I
             x x x x x x x x x x  I
             x x x x x x x x x x \/
             x x x x x x x x x x

        However, if the sub-size is 2, we go to each unmasked pixel and allocate sub-pixel coordinates for it. For
        example, for pixel 0, if *sub_size=2*, we use a 2x2 sub-grid:

        .. code-block:: bash

            Pixel 0 - (2x2):
                                y      x
                   grid[0] = [0.66, -1.66]
            I0I1I  grid[1] = [0.66, -1.33]
            I2I3I  grid[2] = [0.33, -1.66]
                   grid[3] = [0.33, -1.33]

        If we used a sub_size of 3, for the pixel we we would create a 3x3 sub-grid:

        .. code-block:: bash

                                  y      x
                     grid[0] = [0.75, -0.75]
                     grid[1] = [0.75, -0.5]
                     grid[2] = [0.75, -0.25]
            I0I1I2I  grid[3] = [0.5,  -0.75]
            I3I4I5I  grid[4] = [0.5,  -0.5]
            I6I7I8I  grid[5] = [0.5,  -0.25]
                     grid[6] = [0.25, -0.75]
                     grid[7] = [0.25, -0.5]
                     grid[8] = [0.25, -0.25]

        All sub-pixels in masked pixels have values (0.0, 0.0).

        __Adaptive Oversampling__

        By default, the sub-grid is the same size in every pixel (e.g. the value of `sub_size` is an integer that
        defines the size of the sub-grid for every pixel).

        However, the `sub_size` can also be input as an `Array2D`, with varying integer values for each pixel.
        This is called adaptive over-sampling and is used to adapt the over-sampling to the bright regions of the
        data, saving computational time.

        __Pixelization__

        For pixelizations performed in the inversion module, over sampling is equally important. Now, the over
        sampling maps multiple data sub-pixels to pixels in the pixelization, where mappings are performed fractionally
        based on the sub-grid sizes.

        The over sampling class has functions dedicated to mapping between the sub-grid and pixel-grid, for example
        `slim_for_sub_slim`.

         The class `OverSampling` is used for the high level API, whereby this is where users input their
         preferred over-sampling configuration. This class, `OverSampler`, contains the functionality
         which actually performs the over-sampling calculations, but is hidden from the user.

        Parameters
        ----------
        mask
            The mask defining the 2D region where the over-sampled grid is computed.
        sub_size
            The size (sub_size x sub_size) of each unmasked pixels sub-grid.
        """
        self.mask = mask

        self.sub_size = over_sample_util.over_sample_size_convert_to_array_2d_from(
            over_sample_size=sub_size, mask=mask
        )

        self.sub_total = int(np.sum(self.sub_size**2))
        self.sub_length = self.sub_size**self.mask.dimensions
        self.sub_fraction = Array2D(values=1.0 / self.sub_length.array, mask=self.mask)

        # Used for JAX based adaptive over sampling.

        # Define group sizes
        group_sizes = np.array(self.sub_size.array**2)

        # Compute the cumulative sum of group sizes to get split points
        self.split_indices = np.cumsum(group_sizes)

        # Ensure correct concatenation by making 0 a JAX array
        self.start_indices = np.concatenate((np.array([0]), self.split_indices[:-1]))

        # Compute segment ids for each element in the flattened array
        self.segment_ids = np.empty(np.sum(sub_size**2), dtype=np.int32)

        for seg_id, (start, end) in enumerate(
            zip(self.start_indices, self.split_indices)
        ):
            self.segment_ids[start:end] = seg_id

    @cached_property
    def sub_is_uniform(self) -> bool:
        """
        Returns True if the sub_size is uniform across all pixels in the mask.

        The `sub_size` is fixed when the `OverSampler` is created, so this is cached: it is evaluated once per
        instance instead of on every call to `binned_array_2d_from` (which is called once per light profile
        evaluation, and therefore many hundreds of times per likelihood evaluation).
        """
        return np.all(np.isclose(self.sub_size, self.sub_size[0]))

    @cached_property
    def binned_counts(self) -> np.ndarray:
        """
        The number of sub-pixels which are binned up into every pixel of the mask, used as the divisor when
        binning an over sampled array via `binned_array_2d_from` for a non-uniform `sub_size`.

        Segments which contain no sub-pixels have their count set to 1, so that the division by this array does
        not raise a divide-by-zero (the corresponding sum is zero, so the binned value is zero either way).

        This is a function of the `segment_ids` alone, which are fixed when the `OverSampler` is created, so it is
        computed once per instance instead of on every call to `binned_array_2d_from`.

        The returned array is read-only, because it is shared by every call and must never be mutated in place.
        """
        counts = np.bincount(self.segment_ids, minlength=self.mask.pixels_in_mask)

        # Avoid division by zero (on a copy, because `np.bincount` output is not reused elsewhere but the cached
        # array below is handed out to every caller).

        counts = np.maximum(counts, 1)
        counts.flags.writeable = False

        return counts

    def tree_flatten(self):
        return (self.mask, self.sub_size), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(mask=children[0], sub_size=children[1])

    @property
    def sub_pixel_areas(self) -> np.ndarray:
        """
        The area of every sub-pixel in the mask.
        """
        sub_pixel_areas = np.zeros(self.sub_total)

        k = 0

        pixel_area = self.mask.pixel_scales[0] * self.mask.pixel_scales[1]

        for i in range(self.sub_size.shape[0]):
            for j in range(self.sub_size[i] ** 2):
                sub_pixel_areas[k] = pixel_area / self.sub_size[i] ** 2
                k += 1

        return sub_pixel_areas

    def binned_array_2d_from(self, array: Array2D, xp=np) -> "Array2D":
        """
        Convenience method to access the binned-up array in its 1D representation, which is a Grid2D stored as an
        ``ndarray`` of shape [total_unmasked_pixels, 2].

        The binning up process converts a array from (y,x) values where each value is a coordinate on the sub-array to
        (y,x) values where each coordinate is at the centre of its mask (e.g. a array with a sub_size of 1). This is
        performed by taking the mean of all (y,x) values in each sub pixel.

        If the array is stored in 1D it is return as is. If it is stored in 2D, it must first be mapped from 2D to 1D.

        In **PyAutoCTI** all `Array2D` objects are used in their `native` representation without sub-gridding.
        Significant memory can be saved by only store this format, thus the `native_binned_only` config override
        can force this behaviour. It is recommended users do not use this option to avoid unexpected behaviour.

        Old docstring:

        For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into
        a finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
        scaled coordinates a the centre of every sub-pixel defined by this 2D mask array.

        The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size**2, 2). y coordinates are
        stored in the 0 index of the second dimension, x coordinates in the 1 index. Masked coordinates are therefore
        removed and not included in the slimmed grid.

        Grid2D are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0.
        Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
        sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.
        """

        if conf.instance["general"]["structures"]["native_binned_only"]:
            return self

        try:
            array = array.slim.array
        except AttributeError:
            pass

        if self.sub_is_uniform:

            binned_array_2d = array.reshape(
                self.mask.shape_slim, self.sub_size[0] ** 2
            ).mean(axis=1)

        else:

            if xp.__name__.startswith("jax"):

                import jax

                sums = jax.ops.segment_sum(
                    array, self.segment_ids, self.mask.pixels_in_mask
                )
                counts = jax.ops.segment_sum(
                    xp.ones_like(array), self.segment_ids, self.mask.pixels_in_mask
                )

            else:

                # Sum values per segment
                sums = np.bincount(
                    self.segment_ids, weights=array, minlength=self.mask.pixels_in_mask
                )

                # Count number of items per segment (cached, as it depends only on the segment ids)
                counts = self.binned_counts

            binned_array_2d = sums / counts

        return Array2D(
            values=binned_array_2d,
            mask=self.mask,
        )

    @property
    def slim_for_sub_slim(self) -> np.ndarray:
        """
        Derives a 1D ``ndarray`` which maps every subgridded 1D ``slim`` index of the ``Mask2D`` to its
        non-subgridded 1D ``slim`` index.

        For example, for the following ``Mask2D`` for ``sub_size=1``:

        ::
            [[True,  True,  True, True]
             [True, False, False, True],
             [True, False,  True, True],
             [True,  True,  True, True]]

        This has three unmasked (``False`` values) which have the ``slim`` indexes:

        ::
            [0, 1, 2]

        The array ``slim_for_sub_slim`` is therefore:

        ::
            [0, 1, 2]

        For a ``Mask2D`` with ``sub_size=2`` each unmasked ``False`` entry is split into a sub-pixel of size 2x2.
        Therefore the array ``slim_for_sub_slim`` becomes:

        ::
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

        Examples
        --------

        .. code-block:: python

            import autoarray as aa

            mask_2d = aa.Mask2D(
                mask=[[True,  True,  True, True]
                      [True, False, False, True],
                      [True, False,  True, True],
                      [True,  True,  True, True]]
                pixel_scales=1.0,
            )

            derive_indexes_2d = aa.DeriveIndexes2D(mask=mask_2d)

            print(derive_indexes_2d.slim_for_sub_slim)
        """
        return over_sample_util.slim_index_for_sub_slim_index_via_mask_2d_from(
            mask_2d=np.array(self.mask), sub_size=self.sub_size.array
        ).astype("int")

    @cached_property
    def uniform_over_sampled(self):
        """
        For a sub-grid, every unmasked pixel of its 2D mask with shape (total_y_pixels, total_x_pixels) is divided into
        a finer uniform grid of shape (total_y_pixels*sub_size, total_x_pixels*sub_size). This routine computes the (y,x)
        scaled coordinates at the centre of every sub-pixel defined by this 2D mask array.

        The sub-grid is returned on an array of shape (total_unmasked_pixels*sub_size**2, 2). y coordinates are
        stored in the 0 index of the second dimension, x coordinates in the 1 index. Masked coordinates are therefore
        removed and not included in the slimmed grid.

        Grid2D are defined from the top-left corner, where the first unmasked sub-pixel corresponds to index 0.
        Sub-pixels that are part of the same mask array pixel are indexed next to one another, such that the second
        sub-pixel in the first pixel has index 1, its next sub-pixel has index 2, and so forth.
        """

        from autoarray.structures.grids.irregular_2d import Grid2DIrregular

        grid = over_sample_util.grid_2d_slim_over_sampled_via_mask_from(
            mask_2d=np.array(self.mask),
            pixel_scales=self.mask.pixel_scales,
            sub_size=self.sub_size.array.astype("int"),
            origin=self.mask.origin,
        )

        return Grid2DIrregular(values=grid)
