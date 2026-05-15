from typing import Tuple


class DatasetModel:
    def __init__(
        self,
        background_sky_level: float = 0.0,
        grid_offset: Tuple[float, float] = (0.0, 0.0),
        grid_rotation_angle: float = 0.0,
    ):
        """
        Attributes which allow for parts of a dataset to be treated as a model, meaning they can be fitted
        for in the `fit` module.

        The following aspects of a dataset can be treated as a model:

         - `background_sky_level`: The data may have a constant signal in the background which is estimated
           and subtracted from the data beforehand with a degree of uncertainty. By including it in the model it can be
           marginalized over. Units are dimensionless and derived from the data.

        - `grid_offset`: Two datasets may be offset from one another, for example if they are taken with different
          pointing positions. This offset can be included in the model and marginalized over. Units are arc seconds.

        - `grid_rotation_angle`: Two datasets may also be rotated relative to one another (e.g. a different
          telescope roll angle). This rotation can be included in the model and marginalized over. Units are degrees,
          with positive angles applying a counter-clockwise rotation about the offset point.

        Parameters
        ----------
        background_sky_level
            Overall normalisation of the sky which is added or subtracted from the data. Units are dimensionless and
            derived from the data, which is expected to be electrons per second in Astronomy analyses.
        grid_offset
            Offset between two datasets, in arc seconds. This is used to align datasets which are taken at different
            pointing positions.
        grid_rotation_angle
            Rotation between two datasets, in degrees. Applied counter-clockwise about the offset point after the
            offset is subtracted. Used to align datasets which share a pointing centre but differ in roll angle.
        """
        self.background_sky_level = background_sky_level
        self.grid_offset = grid_offset
        self.grid_rotation_angle = grid_rotation_angle
