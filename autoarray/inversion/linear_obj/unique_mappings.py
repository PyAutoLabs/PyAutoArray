import numpy as np


class UniqueMappings:
    def __init__(
        self,
        data_to_pix_unique: np.ndarray,
        data_weights: np.ndarray,
        pix_lengths: np.ndarray,
    ):
        """
        Packages the unique mappings of every unmasked data pixel's (e.g. `grid_slim`) sub-pixels (e.g. `grid_sub_slim`)
        to their corresponding mesh pixels (e.g. `mesh_grid`).

        The following quantities are packaged in this class as ndarray:

        - `data_to_pix_unique`: the unique mapping of every data pixel's grouped sub-pixels to mesh pixels.
        - `data_weights`: the weights of each data pixel's grouped sub-pixels to mesh pixels (e.g. determined
        via their sub-size fractional mappings and interpolation weights).
        - `pix_lengths`: the number of unique mesh pixels each data pixel's grouped sub-pixels map too.

        The need to store separately the mappings and mesh lengths is so that they be easily iterated over when
        perform calculations for efficiency.

        See the mapper properties `unique_mappings()` for a description of the use of this object in mappers.

        Parameters
        ----------
        data_to_pix_unique
            The unique mapping of every data pixel's grouped sub-pixels to mesh pixels.
        data_weights
            The weights of each data pixel's grouped sub-pixels to mesh pixels
        pix_lengths
            The number of unique mesh pixels each data pixel's grouped sub-pixels map too.
        """
        self.data_to_pix_unique = data_to_pix_unique.astype("int")
        self.data_weights = data_weights
        self.pix_lengths = pix_lengths.astype("int")
