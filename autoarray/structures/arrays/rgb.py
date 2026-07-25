from autoarray.abstract_ndarray import AbstractNDArray
from autoarray.structures.arrays.uniform_2d import Array2D


class Array2DRGB(Array2D):

    def __init__(self, values, mask):
        """
        A container for RGB images which have a final dimension of 3, which allows them to be visualized using
        the same functionality as `Array2D` objects.

        By passing an RGB image to this class, the following visualization functionality is used when the RGB
        image is used in plotting:

        - The RGB image is plotted using the `imshow` function of Matplotlib.
        - Functionality which sets the scale of the axis, zooms the image, and sets the axis limits is used.
        - The colorbar is set to the RGB image, which is a 3D array with a final dimension of 3.
        - The formatting of the image is identical to that of `Array2D` objects, which means the image is plotted
        with the same aspect ratio as the original image making for easy subplot formatting.

        This class always assumes the array is in its `native` representation, but with a final dimension of 3.

        Parameters
        ----------
        values
            The values of the RGB image, which is a 3D array with a final dimension of 3.
        mask
            The 2D mask associated with the array, defining the pixels each array value in its ``slim`` representation
            is paired with.
        """

        array = values

        while isinstance(array, AbstractNDArray):
            array = array.array

        self._array = array
        self.mask = mask

    @property
    def native(self) -> "Array2D":
        """
        Returns the RGB ndarray of shape [total_y_pixels, total_x_pixels, 3] in its `native` representation.
        """
        return self
