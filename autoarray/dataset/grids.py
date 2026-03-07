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
        over_sample_size_lp
            The over sampling scheme size, which divides the grid into a sub grid of smaller pixels when computing
            values (e.g. images) from the grid to approximate the 2D line integral of the amount of light that falls
            into each pixel.
        over_sample_size_pixelization
            How over sampling is performed for the grid which is associated with a pixelization, which is therefore
            passed into the calculations performed in the `inversion` module.
        psf
            The Point Spread Function kernel of the image which accounts for diffraction due to the telescope optics
            via 2D convolution.
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

        if self._lp is not None:
            return self._lp

        self._lp = Grid2D.from_mask(
            mask=self.mask,
            over_sample_size=self.over_sample_size_lp,
        )

        return self._lp

    @property
    def pixelization(self):
        if self._pixelization is not None:
            return self._pixelization

        self._pixelization = Grid2D.from_mask(
            mask=self.mask,
            over_sample_size=self.over_sample_size_pixelization,
        )

        return self._pixelization

    @property
    def blurring(self):

        if self._blurring is not None:
            return self._blurring

        if self.psf is None:
            self._blurring = None
        else:

            blurring_mask = self.mask.derive_mask.blurring_from(
                kernel_shape_native=self.psf.kernel.shape_native, allow_padding=True
            )

            self._blurring = Grid2D.from_mask(
                mask=blurring_mask,
                over_sample_size=1,
            )

        return self._blurring

    @property
    def border_relocator(self) -> BorderRelocator:

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
        self.lp = lp
        self.pixelization = pixelization
        self.blurring = blurring
        self.border_relocator = border_relocator
