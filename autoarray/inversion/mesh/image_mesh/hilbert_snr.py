from __future__ import annotations
import numpy as np
from typing import Optional

from autoarray.mask.mask_2d import Mask2D
from autoarray.structures.arrays.uniform_2d import Array2D
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.inversion.mesh.image_mesh.hilbert import Hilbert
from autoarray.inversion.mesh.image_mesh.snr_transform import soft_snr_from


class HilbertSNR(Hilbert):
    def __init__(
        self,
        pixels=10.0,
        background_share: float = 0.25,
        peak_share: float = 0.05,
        peak_power: float = 8.0,
        snr_noise: float = 2.0,
        snr_soft: float = 3.0,
        soft_power: float = 2.0,
    ):
        """
        Computes an image-mesh by drawing points along a Hilbert curve, with the points shared out by the adapt
        image's absolute signal-to-noise (S/N) under a fixed point budget.

        This is the recommended default for adaptive pixelizations. It draws points exactly as `Hilbert` does (the
        same curve and deterministic inverse-transform sampling, so a region's share of the points equals its share
        of the weight) but replaces `Hilbert`'s weighting, ``(adapt / max(adapt)) ** weight_power`` floored at
        ``weight_floor``. That weighting is normalised by the brightest pixel, so the adaptation of the rest of the
        source depends on how bright that one pixel is: with a peak S/N of 250, ``weight_power=3.5`` and
        ``weight_floor=0.01``, every pixel below S/N ~67 got the floor weight, the same as empty sky. Capping the S/N
        map at 3 before the weighting avoided that, but treated all source structure above S/N 3 as equal.

        The weighting here gives each cell of the Hilbert curve, with S/N ``s``, the probability

            background_share / n
            + peak_share * (s / max(s)) ** peak_power / sum((s / max(s)) ** peak_power)
            + (1 - background_share - peak_share) * soft / sum(soft)

        where ``soft = (x / (x + snr_soft)) ** soft_power`` and ``x = max(s - snr_noise, 0)`` (`soft_snr_from`):

        - ``background_share`` (25%) of the points are spread evenly over the mask, so the background is always
          covered whatever the source looks like.
        - most of the points (70%) follow the soft S/N transform, which is in absolute S/N: zero at or below
          S/N ``snr_noise`` (noise), rising through S/N ``snr_noise + snr_soft`` and saturating for bright pixels.
          Faint but detected source emission always gets points; bright regions do not take them all.
        - ``peak_share`` (5%) of the points go to the cells within roughly 25% of the peak S/N
          (``peak_power=8``), giving a bright compact feature (a bulge, a compact clump) extra resolution.

        With no arguments other than ``pixels`` this gives the weighting chosen in the adapt-mesh-weighting study
        (PyAutoArray#714): fitted on 10 HST and 10 JWST (COWLS) lens datasets and checked offline on 19 lenses
        from a peak S/N of 5 to 1000.

        The ``adapt_data`` should be the source's S/N map *before* the minimum-percent floor PyAutoGalaxy applies to
        adapt images: above a peak S/N of 200 that floor lifts empty sky over ``snr_noise`` and the sky is then
        weighted as source. Negative values are treated as 0 before the map is interpolated onto the curve. To have
        an adaptive regularization (e.g. `AdaptSplit`) adapt to the same signal, give it `adapt_image_soft_from` of
        the same S/N map.

        Known limits, measured in that study:

        - For a very faint source (peak S/N ~5) only a few pixels exceed ``snr_noise``, so most points cluster on
          them (tens of points per image pixel) and the soft transform gives the regularization little contrast.
        - For a very bright source (peak S/N of several hundred) nearly all detected emission is past the
          saturation, so points spread evenly over it, arcs and faint wings alike.

        Parameters
        ----------
        pixels
            The total number of pixels in the image mesh and drawn from the Hilbert curve.
        background_share
            Fraction of the points spread evenly over every cell of the curve inside the mask.
        peak_share
            Fraction of the points given in proportion to ``(s / max(s)) ** peak_power``.
        peak_power
            Power of the peak term; 8 restricts it to cells within roughly 25% of the peak S/N.
        snr_noise
            S/N at or below which a cell counts as noise and receives only the background (and peak) shares.
        snr_soft
            S/N above ``snr_noise`` at which the soft term reaches ``0.5 ** soft_power``.
        soft_power
            Power of the soft term.
        """
        if background_share < 0.0 or peak_share < 0.0 or background_share + peak_share > 1.0:
            raise ValueError(
                "HilbertSNR requires background_share >= 0, peak_share >= 0 and "
                f"background_share + peak_share <= 1 (got {background_share}, {peak_share})."
            )

        super().__init__(pixels=pixels, weight_floor=0.0, weight_power=0.0)

        self.background_share = background_share
        self.peak_share = peak_share
        self.peak_power = peak_power
        self.snr_noise = snr_noise
        self.snr_soft = snr_soft
        self.soft_power = soft_power

    def weight_map_from(self, adapt_data: np.ndarray) -> np.ndarray:
        """
        Returns the probability of each cell of the Hilbert curve (summing to 1), from the adapt S/N values along
        the curve. See the `__init__` docstring for the formula.

        A share whose term is zero everywhere (no cell above ``snr_noise``, or no positive S/N) is given to the
        background instead, so a blank adapt image gives an even mesh.

        Parameters
        ----------
        adapt_data
            The adapt image's S/N values on the cells of the Hilbert curve.
        """
        snr = np.clip(np.asarray(adapt_data, dtype=float), 0.0, None)
        n = snr.size

        background = np.full(n, 1.0 / n)
        weight_map = self.background_share * background

        peak_value = np.max(snr) if n else 0.0
        if peak_value > 0.0:
            peak_term = (snr / peak_value) ** self.peak_power
            weight_map = weight_map + self.peak_share * peak_term / np.sum(peak_term)
        else:
            weight_map = weight_map + self.peak_share * background

        soft = soft_snr_from(
            snr=snr,
            snr_noise=self.snr_noise,
            snr_soft=self.snr_soft,
            soft_power=self.soft_power,
        )
        source_share = 1.0 - self.background_share - self.peak_share
        if np.sum(soft) > 0.0:
            weight_map = weight_map + source_share * soft / np.sum(soft)
        else:
            weight_map = weight_map + source_share * background

        return weight_map

    def image_plane_mesh_grid_from(
        self,
        mask: Mask2D,
        adapt_data: Optional[np.ndarray],
    ) -> Grid2DIrregular:
        """
        Returns the image mesh, drawn along the Hilbert curve with the `HilbertSNR` weighting.

        Negative S/N values are set to 0 on the native adapt image, before it is interpolated onto the curve, so
        noise below zero does not pull interpolated source values down; the rest is `Hilbert`'s.

        Parameters
        ----------
        mask
            The circular mask of the data the pixelization fits.
        adapt_data
            The source's S/N map (before the adapt-image minimum-percent floor), on ``mask``.
        """
        if adapt_data is not None:
            adapt_data = Array2D(
                values=np.clip(np.asarray(adapt_data.array, dtype=float), 0.0, None),
                mask=adapt_data.mask,
            )

        return super().image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_data)
