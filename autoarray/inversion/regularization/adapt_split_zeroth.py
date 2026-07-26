from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.adapt import Adapt
from autoarray.inversion.regularization.brightness_zeroth import BrightnessZeroth
from autoarray.inversion.regularization import regularization_util


class AdaptSplitZeroth(Adapt):
    def __init__(
        self,
        zeroth_coefficient: float = 1.0,
        zeroth_signal_scale: float = 1.0,
        inner_coefficient: float = 1.0,
        outer_coefficient: float = 1.0,
        signal_scale: float = 1.0,
    ):
        """
        Regularization which uses zeroth order regularization, the derivatives at a cross of four points around each
        pixel centre and values adapted to the data being fitted to smooth an inversion's solution.

        The size of this cross is determined via the size of the source-pixel, for example if the source pixel is a
        Delaunay pixel the area of the pixel is computed and the distance of each point of the cross is given by
        the area times 0.5.

        For the weighted regularization scheme, each pixel is given an 'effective regularization weight', which is
        applied when each set of pixel neighbors are regularized with one another. The motivation of this is that
        different regions of a pixelization's mesh require different levels of regularization (e.g., high smoothing where the
        no signal is present and less smoothing where it is, see (Nightingale, Dye and Massey 2018)).

        Unlike ``Constant`` regularization, neighboring pixels must now be regularized with one another
        in both directions (e.g. if pixel 0 regularizes pixel 1, pixel 1 must also regularize pixel 0). For example:

        B = [-1, 1]  [0->1]
            [-1, -1]  1 now also regularizes 0

        For ``Constant`` regularization this would NOT produce a positive-definite matrix. However, for
        the weighted scheme, it does!

        The regularize weight_list change the B matrix as shown below - we simply multiply each pixel's effective
        regularization weight by each row of B it has a -1 in, so:

        regularization_weights = [1, 2, 3, 4]

        B = [-1, 1, 0 ,0] # [0->1]
            [0, -2, 2 ,0] # [1->2]
            [0, 0, -3 ,3] # [2->3]
            [4, 0, 0 ,-4] # [3->0]

        If our -1's werent down the diagonal this would look like:

        B = [4, 0, 0 ,-4] # [3->0]
            [0, -2, 2 ,0] # [1->2]
            [-1, 1, 0 ,0] # [0->1]
            [0, 0, -3 ,3] # [2->3] This is valid!

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        **JAX & gradient support** (2026-07 gradient sweep): not currently
        JAX-differentiable on either gradient-capable mesh family — the split
        leg is structurally incompatible with the rectangular meshes (shape
        error) and the zeroth/adapt legs hit numpy-on-traced-array operations
        on the Delaunay mesh family (``TracerArrayConversionError``).

        Parameters
        ----------
        coefficients
            The regularization coefficients which controls the degree of smoothing of the inversion reconstruction in
            high and low signal regions of the reconstruction.
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
            low signal regions.
        """

        self.zeroth_coefficient = zeroth_coefficient
        self.zeroth_signal_scale = zeroth_signal_scale

        super().__init__(
            inner_coefficient=inner_coefficient,
            outer_coefficient=outer_coefficient,
            signal_scale=signal_scale,
        )

    def regularization_matrix_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization matrix with shape [pixels, pixels].

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses this matrix to perform regularization.

        Returns
        -------
        The regularization matrix.
        """
        regularization_weights = self.regularization_weights_from(
            linear_obj=linear_obj, xp=xp
        )

        mappings, sizes, weights = linear_obj.interpolator._mappings_sizes_weights_split

        (
            splitted_mappings,
            splitted_sizes,
            splitted_weights,
        ) = regularization_util.reg_split_from(
            splitted_mappings=mappings,
            splitted_sizes=sizes,
            splitted_weights=weights,
            xp=xp,
        )

        regularization_matrix = (
            regularization_util.pixel_splitted_regularization_matrix_from(
                regularization_weights=regularization_weights,
                splitted_mappings=splitted_mappings,
                splitted_sizes=splitted_sizes,
                splitted_weights=splitted_weights,
                xp=xp,
            )
        )

        brightness_zeroth = BrightnessZeroth(
            coefficient=self.zeroth_coefficient, signal_scale=self.zeroth_signal_scale
        )

        regularization_matrix_zeroth = brightness_zeroth.regularization_matrix_from(
            linear_obj=linear_obj, xp=xp
        )

        return regularization_matrix + regularization_matrix_zeroth
