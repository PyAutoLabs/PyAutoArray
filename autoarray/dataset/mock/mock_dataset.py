from autoarray.structures.grids.uniform_2d import Grid2D
from autoarray.dataset.grids import GridsInterface


class MockDataset:
    """
    A lightweight mock dataset for use in tests and unit testing fixtures.

    Provides a minimal dataset-like interface with `grids`, `psf` and `mask` attributes,
    without any real data arrays or numerical computation. Code that only needs to access
    `dataset.grids.pixelization` or similar can use this instead of constructing a full
    `Imaging` or `Interferometer` dataset.

    If no `grids` are provided, a default `GridsInterface` is constructed with a single
    pixelization pixel at coordinate (1.0, 1.0) and pixel_scales=1.0. This acts as a
    minimal placeholder sufficient for tests that exercise inversion or pixelization code
    paths without real imaging data.

    Parameters
    ----------
    grids
        A `GridsInterface` (or compatible object) providing `lp`, `pixelization`, `blurring`
        and `border_relocator` attributes. If `None`, a default single-pixel grid is used.
    psf
        An optional PSF kernel object. If `None`, no PSF is available on this mock dataset.
    mask
        An optional mask object. If `None`, no mask is available on this mock dataset.
    """

    def __init__(self, grids=None, psf=None, mask=None):
        if grids is None:
            self.grids = GridsInterface(
                lp=None,
                pixelization=Grid2D.no_mask(values=[[[1.0, 1.0]]], pixel_scales=1.0),
                blurring=None,
                border_relocator=None,
            )

        self.psf = psf
        self.mask = mask
