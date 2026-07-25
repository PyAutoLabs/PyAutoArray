from __future__ import annotations
from copy import deepcopy
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from autoarray.layout.region import Region2D
    from autoarray.type import Region2DLike

import autoarray as aa


def rotate_array_via_roe_corner_from(
    array: np.ndarray, roe_corner: Tuple[int, int]
) -> np.ndarray:
    """
    Rotates an input array such that its read-out electronics corner (``roe_corner``) are positioned at the
    'bottom-left' (e.g. [1,0]) of the ndarray data structure.

    This is used to homogenize arrays to a common orientation, especially for the project **PyAutoCTI**
    which performs clocking for Charge Transfer Inefficiency.

    Parameters
    ----------
    array
        The array which is rotated.
    roe_corner
        The corner of the array at which the read-out electronics are located (e.g. (1, 1) is the bottom-right corner).
        The rotation is based on this such that the read-out electronics are in the bottom-left (e.g. (1, 0)).

    Returns
    -------
    ndarray
        The rotated array where the read out electronics are at the bottom left corner, (1, 0).
    """
    if roe_corner == (1, 0):
        return array.copy()
    elif roe_corner == (0, 0):
        return array[::-1, :].copy()
    elif roe_corner == (1, 1):
        return array[:, ::-1].copy()
    elif roe_corner == (0, 1):
        array = array[::-1, :].copy()
        return array[:, ::-1]


def rotate_region_via_roe_corner_from(
    region: Region2DLike,
    shape_native: Tuple[int, int],
    roe_corner: Tuple[int, int],
) -> Optional[Region2D]:
    """
    Rotates a (y0, y1, x0, x1) region such that its read-out electronics corner (``roe_corner``) are positioned at
    the 'bottom-left' (e.g. [1,0]).

    Parameters
    ----------
    region
        The coordinates on the image of the (y0, y1, x0, y1) ``Region2D`` that are rotated.
    shape_native
        The 2D shape of the `Array2D` the regions are located on, required to determine the rotated `region`.
    roe_corner
        The corner of the ``Array2D``at which the read-out electronics are located (e.g. (1, 1) is the bottom-right corner).
        The rotation is based on this such that the read-out electronics are in the bottom-left (e.g. (1, 0)).

    Returns
    -------
    aa.Region2D
        The rotated (y0, y1, x0, x1) ``Region2D`` where the read out electronics are at the bottom left corner, (1, 0).
    """
    if region is None:
        return None

    if roe_corner == (1, 0):
        return aa.Region2D(region=region)
    elif roe_corner == (0, 0):
        return aa.Region2D(
            region=(
                shape_native[0] - region[1],
                shape_native[0] - region[0],
                region[2],
                region[3],
            )
        )
    elif roe_corner == (1, 1):
        return aa.Region2D(
            region=(
                region[0],
                region[1],
                shape_native[1] - region[3],
                shape_native[1] - region[2],
            )
        )
    elif roe_corner == (0, 1):
        return aa.Region2D(
            region=(
                shape_native[0] - region[1],
                shape_native[0] - region[0],
                shape_native[1] - region[3],
                shape_native[1] - region[2],
            )
        )


def rotate_pattern_ci_via_roe_corner_from(
    pattern_ci, shape_native: Tuple[int, int], roe_corner: Tuple[int, int]
):
    """
    Rotates a ``ChargeInjectionPattern` such that its read-out electronics corner (``roe_corner``) are positioned at
    the 'bottom-left' (e.g. [1,0]).

    Parameters
    ----------
    pattern_ci : ac.CIPaattern
        The charge-injection pattern of the ``Array2D`` that is rotated.
    shape_native
        The 2D shape of the ``Array2D`` the regions are located on, required to determine the rotated ``region``.
    roe_corner
        The corner of the ``Array2D``at which the read-out electronics are located (e.g. (1, 1) is the bottom-right corner).
        The rotation is based on this such that the read-out electronics are in the bottom-left (e.g. (1, 0)).

    Returns
    -------
    aa.Region2D
        The rotated (y0, y1, x0, x1) ``Region2D`` where the read out electronics are at the bottom left corner, (1, 0).
    """
    new_pattern_ci = deepcopy(pattern_ci)

    new_pattern_ci.regions = [
        rotate_region_via_roe_corner_from(
            region=region, shape_native=shape_native, roe_corner=roe_corner
        )
        for region in pattern_ci.regions
    ]

    return new_pattern_ci


def region_after_extraction(
    original_region: Region2DLike, extraction_region: Region2DLike
) -> Optional[Region2D]:
    if original_region is None:
        return None

    y0, y1 = x0x1_after_extraction(
        x0o=original_region[0],
        x1o=original_region[1],
        x0e=extraction_region[0],
        x1e=extraction_region[1],
    )
    x0, x1 = x0x1_after_extraction(
        x0o=original_region[2],
        x1o=original_region[3],
        x0e=extraction_region[2],
        x1e=extraction_region[3],
    )

    if None in [y0, y1, x0, x1]:
        return None
    return aa.Region2D((y0, y1, x0, x1))


def x0x1_after_extraction(x0o: int, x1o: int, x0e: int, x1e: int):
    """
    When we extract an array, we also update the extracted array's regions by mapping each region from their
    coordinates on the original array (which has a shape_native) to the extracted array (which is a 2D section
    on this array).

    This function compares the 1D coordinates of a regions original coordinates on a array to the 1D coordinates of the
    extracted array, determining where the original region lies on the extracted array.

    For example, for a 1D array with shape 8 we may have a region whose 1D coordinates span x0o=2 -> x1o=6. From the
    original 1D array we then extract the region x0e=5 -> x1e = 7. This looks as follows:

                                eeeeeeeee
                                5        7      e = extracted region
          oooooooooooooooooooooooooo            o = original region
         2                          6           - = original array (which has shape = 8
      ------------------------------------
     0                                    8

     In the above example this function will recognise that the extracted region will contain a small section of the
     original region and for the extracted region give it coordinates (0, 1). This function covers all possible
     ways the original region and extracted array could over lap.

    If the extraction completely the region a None is returned.
    """

    if x0e >= x0o and x0e <= x1o:
        x0 = 0
    elif x0e <= x0o:
        x0 = x0o - x0e
    elif x0e >= x0o:
        x0 = 0

    if x1e >= x0o and x1e <= x1o:
        x1 = x1e - x0e
    elif x1e > x1o:
        x1 = x1o - x0e

    try:
        if x0 < 0 or x1 < 0 or x0 == x1:
            return None, None
        else:
            return x0, x1
    except UnboundLocalError:
        return None, None
