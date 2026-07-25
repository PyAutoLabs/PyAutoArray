class MaskException(Exception):
    """
    Raises exceptions associated with the `mask` modules and `Mask` classes.

    For example if a 2D mask's shape is not of length 2 (and thus not 2D).
    """

    pass


class ArrayException(Exception):
    """
    Raises exceptions associated with the `structures/array` modules and `Array` classes.

    For example if a 2D array's shape and its corresponding 2D mask shape do not match.
    """

    pass


class GridException(Exception):
    """
    Raises exceptions associated with the `structures/grid` modules and `Grid` classes.

    For example if a 2D grid's shape and its corresponding 2D mask shape do not match.
    """

    pass


class VectorYXException(Exception):
    """
    Raises exceptions associated with the `structures/vectors` modules and `VectorYX` classes.

    For example if a 2D vector's shape and its corresponding 2D mask shape do not match.
    """

    pass


class KernelException(Exception):
    """
    Raises exceptions associated with the `structures/arrays/kernel_2d.py` module and `Convolver` classes.

    For example if the kernel has an even-sized number of pixels.
    """

    pass


class RegionException(Exception):
    """
    Raises exceptions associated with the `layout/region` modules and `Region` classes.

    For example if a region is specified where the right-hand x coordinate is less than the left hand x coordinate.
    """

    pass


class DatasetException(Exception):
    """
    Raises exceptions associated with the `dataset` modules and `Imaging` / `Interferometer` classes.

    For example if a noise-map contains NaN values.
    """

    pass


class MeshException(Exception):
    """
    Raises exceptions associated with the `inversion/mesh` modules and `Mesh` classes.

    For example if a `RectangularAdaptDensity` mesh has dimensions below 3x3.
    """

    pass


class PixelizationException(Exception):
    """
    Raises exceptions associated with the `inversion/pixelization` modules and `Pixelization` classes.
    """

    pass


class InversionException(Exception):
    """
    Raises exceptions associated with the `inversion/inversion` modules and `Inversion` classes.

    For example many numpy linear algebra errors are overwritten with this exception for easy exception handling.
    """

    pass


class PlottingException(Exception):
    """
    Raises exceptions associated with the `plot` module and classes like `MatWrap` used for plotting.

    For example if the plotter type for certain plot objects is not a valid type.
    """

    pass
