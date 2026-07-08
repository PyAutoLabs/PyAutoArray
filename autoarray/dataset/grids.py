from typing import Optional, Union

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.operators.convolver import Convolver
from autoarray.structures.grids.uniform_2d import Grid2D

from autoarray.inversion.mesh.border_relocator import BorderRelocator

from autoarray import exc


class GridsDataset:
    def __init__(
        self,
        mask: Mask2D,
        over_sample_size_lp: Union[int, Array2D],
        over_sample_size_pixelization: Union[int, Array2D],
        psf: Optional[Convolver] = None,
    ):
        """
        Contains grids of (y,x) Cartesian coordinates at the centre of every pixel in the dataset's image and
        mask, which are used for performing calculations on the datas.

        The following grids are contained:

        - `lp`: A grids of (y,x) coordinates which aligns with the centre of every image pixel of the image data,
        which is used for most normal calculations (e.g. evaluating the amount of light that falls in an pixel
        from a light profile).

        - `pixelization`: A grid of (y,x) coordinates which again align with the centre of every image pixel of
        the image data. This grid is used specifically for pixelizations computed via the `inversion` module, which
        can benefit from using different oversampling schemes than the normal grid.

        - `blurring`: A grid of (y,x) coordinates which are used to compute the blurring of the image data. This image
        contains the light of all pixels that are masked, but are close enough to the unmasked pixels that their light
        will be convolved into the unmasked pixels by the PSF.

        Every grid has its own `over_sampling` class which defines how over sampling is performed for these grids and
        is described in the corresponding `over_sampling.py` module.

        This is used in the project PyAutoGalaxy to load imaging data of a galaxy and fit it with galaxy light profiles.
        It is used in PyAutoLens to load imaging data of a strong lens and fit it with a lens model.

        Parameters
        ----------
        mask
            The 2D mask defining which pixels are included in the dataset. All grids are constructed
            to align with the centres of the unmasked pixels in this mask.
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        psf
            The Point Spread Function kernel of the image which accounts for diffraction due to the telescope optics
            via 2D convolution. Required to compute the blurring grid; if `None` the blurring grid
            is not constructed.
        """
        self.mask = mask
        self.over_sample_size_lp = over_sample_size_lp
        self.over_sample_size_pixelization = over_sample_size_pixelization
        self.psf = psf

        self._lp = None
        self._pixelization = None
        self._blurring = None
        self._border_relocator = None

    @property
    def lp(self):
        """
        The light-profile grid: a `Grid2D` of (y,x) Cartesian coordinates at the centre of every
        unmasked image pixel, used for evaluating light profiles and other spatial calculations
        during model fitting.

        The grid uses `over_sample_size_lp` to perform over-sampled sub-pixel integration,
        approximating the 2D line integral of the light profile within each pixel. This grid is
        what most model-fitting calculations use (e.g. computing galaxy images).

        This property is lazily evaluated and cached on first access.
        """
        if self._lp is not None:
            return self._lp

        self._lp = Grid2D.from_mask(
            mask=self.mask,
            over_sample_size=self.over_sample_size_lp,
        )

        return self._lp

    @property
    def pixelization(self):
        """
        The pixelization grid: a `Grid2D` of (y,x) Cartesian coordinates at the centre of every
        unmasked image pixel, dedicated to pixelized source reconstructions via the `inversion` module.

        This grid uses `over_sample_size_pixelization` which can differ from `over_sample_size_lp`,
        allowing the pixelization to benefit from a different (e.g. lower) over-sampling resolution
        than the light-profile grid.

        This property is lazily evaluated and cached on first access.
        """
        if self._pixelization is not None:
            return self._pixelization

        self._pixelization = Grid2D.from_mask(
            mask=self.mask,
            over_sample_size=self.over_sample_size_pixelization,
        )

        return self._pixelization

    @property
    def blurring(self):
        """
        The blurring grid: a `Grid2D` of (y,x) coordinates for pixels that lie just outside the
        mask but whose light can be scattered into the unmasked region by the PSF.

        When convolving a model image with the PSF, pixels neighbouring the mask boundary can
        contribute flux to unmasked pixels. The blurring grid provides the coordinates of these
        border pixels so their light profile values can be evaluated and included in the convolution.

        Returns `None` if no PSF was supplied (i.e. no blurring is performed).

        This property is lazily evaluated and cached on first access.
        """
        if self._blurring is not None:
            return self._blurring

        if self.psf is None:
            self._blurring = None
        else:

            # For an oversampled PSF the blurring mask footprint is defined at
            # image resolution, and the blurring image must be evaluated on the
            # fine grid so it can enter the oversampled convolution.
            blurring_mask = self.mask.derive_mask.blurring_from(
                kernel_shape_native=self.psf.kernel_shape_image_resolution,
                allow_padding=True,
            )

            self._blurring = Grid2D.from_mask(
                mask=blurring_mask,
                over_sample_size=self.psf.convolve_over_sample_size,
            )

        return self._blurring

    @property
    def border_relocator(self) -> BorderRelocator:
        """
        The border relocator for the pixelization grid.

        During pixelized source reconstruction, source-plane coordinates that map outside the
        border of the pixelization mesh can cause numerical problems. The `BorderRelocator`
        detects these coordinates and relocates them to the border of the mesh, preventing
        ill-conditioned inversions.

        This property is lazily evaluated and cached on first access.
        """
        if self._border_relocator is not None:
            return self._border_relocator

        self._border_relocator = BorderRelocator(
            mask=self.mask,
            sub_size=self.over_sample_size_pixelization,
        )

        return self._border_relocator


class GridsInterface:
    def __init__(
        self,
        lp=None,
        pixelization=None,
        blurring=None,
        border_relocator=None,
    ):
        """
        A lightweight plain-data container for pre-constructed dataset grids.

        Unlike `GridsDataset`, this class performs no computation — it simply holds grids that have
        already been created elsewhere. It is used in test fixtures and mock datasets where a full
        `GridsDataset` is not needed, but code that accesses `dataset.grids.lp` or
        `dataset.grids.pixelization` still needs to work.

        Parameters
        ----------
        lp
            The light-profile `Grid2D` used for evaluating light profiles during model fitting.
        pixelization
            The pixelization `Grid2D` used for source reconstruction via the inversion module.
        blurring
            The blurring `Grid2D` for pixels outside the mask that contribute flux via PSF convolution.
        border_relocator
            The `BorderRelocator` used to remap out-of-bounds source-plane coordinates to the
            pixelization mesh border.
        """
        self.lp = lp
        self.pixelization = pixelization
        self.blurring = blurring
        self.border_relocator = border_relocator
