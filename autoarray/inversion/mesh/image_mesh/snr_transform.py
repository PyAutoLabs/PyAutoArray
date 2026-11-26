"""
Soft signal-to-noise transform shared by the `HilbertSNR` image mesh and the adapt image given to adaptive
regularization schemes (e.g. `AdaptSplit`).

Both the image mesh and the regularization adapt to the source's signal-to-noise (S/N) map. Earlier defaults
normalised that map by its brightest pixel (``(adapt / max(adapt)) ** weight_power``), so how well the rest of the
source was adapted to depended on how bright its single brightest pixel was: with a peak S/N of 250 and
``weight_power=3.5, weight_floor=0.01``, every pixel below S/N ~67 sat on the floor, the same as empty sky.

The transform here works in absolute S/N instead. A pixel's value depends only on its own S/N:

    x = max(S/N - snr_noise, 0)
    soft = (x / (x + snr_soft)) ** soft_power

With the defaults (``snr_noise=2``, ``snr_soft=3``, ``soft_power=2``) it is 0 at or below S/N 2 (noise), 0.06 at
S/N 3, 0.36 at S/N 6.5, 0.59 at S/N 12, 0.86 at S/N 40 and tends to 1 for bright pixels. Faint but detected source
emission therefore always counts, whatever the peak, and bright regions saturate rather than dominate.

The input should be the S/N map *before* the minimum-percent floor PyAutoGalaxy applies to adapt images
(``adapt_minimum_percent``, 1% of the maximum): above a peak S/N of 200 that floor lifts empty sky over
``snr_noise`` (sky reads S/N 2.5 at a peak of 250 and 10 at 1000) and the sky is then treated as source.
"""
import numpy as np


def soft_snr_from(
    snr: np.ndarray,
    snr_noise: float = 2.0,
    snr_soft: float = 3.0,
    soft_power: float = 2.0,
) -> np.ndarray:
    """
    Returns the soft S/N transform ``(x / (x + snr_soft)) ** soft_power`` with ``x = max(S/N - snr_noise, 0)``,
    applied value by value.

    Parameters
    ----------
    snr
        Signal-to-noise values (any shape). Negative values are treated as 0.
    snr_noise
        S/N at or below which a value counts as noise and maps to 0.
    snr_soft
        S/N above ``snr_noise`` at which the transform reaches ``0.5 ** soft_power``; it sets where the transform
        saturates.
    soft_power
        Power of the saturating term; higher values suppress faint values relative to bright ones.

    Returns
    -------
    The transformed values, in [0, 1), with the same shape as ``snr``.
    """
    snr = np.clip(np.asarray(snr, dtype=float), 0.0, None)
    x = np.clip(snr - snr_noise, 0.0, None)
    return (x / (x + snr_soft)) ** soft_power


def adapt_image_soft_from(
    snr: np.ndarray,
    snr_noise: float = 2.0,
    snr_soft: float = 3.0,
    soft_power: float = 2.0,
    minimum_percent: float = 0.01,
) -> np.ndarray:
    """
    Returns the adapt image to give an adaptive regularization (e.g. `AdaptSplit`) so that it adapts to the same
    signal as the `HilbertSNR` image mesh: `soft_snr_from` of the S/N map, floored at ``minimum_percent`` of its
    maximum (the floor PyAutoGalaxy puts on every adapt image, which keeps the regularization's per-pixel signal
    positive).

    The regularization scheme itself is unchanged: `AdaptSplit` still divides its pixel signals by their maximum and
    raises them to its ``signal_scale``. Only its input changes.

    Parameters
    ----------
    snr
        The source's signal-to-noise map before the adapt-image floor (any shape). Negative values are treated as 0.
    snr_noise, snr_soft, soft_power
        As in `soft_snr_from`; keep them equal to the image mesh's so mesh and regularization adapt to one signal.
    minimum_percent
        Fraction of the transformed maximum below which values are raised to that floor.

    Returns
    -------
    The adapt image values, with the same shape as ``snr``. A map with no value above ``snr_noise`` gives ones
    (nothing to adapt to).
    """
    soft = soft_snr_from(
        snr=snr, snr_noise=snr_noise, snr_soft=snr_soft, soft_power=soft_power
    )
    peak = np.max(soft) if soft.size else 0.0
    if peak <= 0.0:
        return np.ones_like(soft)
    return np.maximum(soft, minimum_percent * peak)
