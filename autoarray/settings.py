import logging
from typing import Optional

from autoconf import conf

logging.basicConfig()
logger = logging.getLogger(__name__)


class Settings:
    def __init__(
        self,
        use_mixed_precision: bool = False,
        use_positive_only_solver: Optional[bool] = None,
        use_edge_zeroed_pixels: Optional[bool] = None,
        use_border_relocator: Optional[bool] = None,
        no_regularization_add_to_curvature_diag_value: float = None,
    ):
        """
        The settings of an Inversion, customizing how a linear set of equations are solved for.

        An Inversion is used to reconstruct a dataset, for example the luminous emission of a galaxy.

        Parameters
        ----------
        use_mixed_precision
            If `True`, the linear algebra calculations of the inversion are performed using single precision on a
            targeted subset of functions which provide significant speed up when using a GPU (x4), reduces VRAM
            use and are expected to have minimal impact on the accuracy of the results. If `False`, all linear algebra
            calculations are performed using double precision, which is the default and is more accurate but
            slower on a GPU.
        use_positive_only_solver
            Whether to use a positive-only linear system solver, which requires that every reconstructed value is
            positive but is computationally much slower than the default solver (which allows for positive and
            negative values).
        use_border_relocator
            If `True`, all coordinates of all pixelization source mesh grids have pixels outside their border
            relocated to their edge.
        no_regularization_add_to_curvature_diag_value
            If a linear func object does not have a corresponding regularization, this value is added to its
            diagonal entries of the curvature regularization matrix to ensure the matrix is positive-definite.
        """
        self.use_mixed_precision = use_mixed_precision
        self._use_positive_only_solver = use_positive_only_solver
        self._use_edge_zeroed_pixels = use_edge_zeroed_pixels
        self._use_border_relocator = use_border_relocator
        self._no_regularization_add_to_curvature_diag_value = (
            no_regularization_add_to_curvature_diag_value
        )

    @property
    def use_positive_only_solver(self):
        if self._use_positive_only_solver is None:
            return conf.instance["general"]["inversion"]["use_positive_only_solver"]

        return self._use_positive_only_solver

    @property
    def use_edge_zeroed_pixels(self):
        if self._use_edge_zeroed_pixels is None:
            return conf.instance["general"]["inversion"]["use_edge_zeroed_pixels"]

        return self._use_edge_zeroed_pixels

    @property
    def use_border_relocator(self):
        if self._use_border_relocator is None:
            return conf.instance["general"]["inversion"]["use_border_relocator"]

        return self._use_border_relocator

    @property
    def no_regularization_add_to_curvature_diag_value(self):
        if self._no_regularization_add_to_curvature_diag_value is None:
            return conf.instance["general"]["inversion"][
                "no_regularization_add_to_curvature_diag_value"
            ]

        return self._no_regularization_add_to_curvature_diag_value
