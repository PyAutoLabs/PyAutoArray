import numpy as np
import pytest

import autoarray as aa


def test__in_grid_2d__out_ndarray_2d():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_2d = obj.ndarray_2d_from(grid=grid_2d)

    assert isinstance(ndarray_2d, aa.Grid2D)
    assert (
        ndarray_2d.native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__in_grid_2d__out_ndarray_2d_list():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    grid_2d = aa.Grid2D.from_mask(mask=mask)

    obj = aa.m.MockGrid2DLikeObj()

    ndarray_2d_list = obj.ndarray_2d_list_from(grid=grid_2d)

    assert isinstance(ndarray_2d_list[0], aa.Grid2D)
    assert (
        ndarray_2d_list[0].native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.5, -0.5], [0.5, 0.5], [0.0, 0.0]],
                [[0.0, 0.0], [-0.5, -0.5], [-0.5, 0.5], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()

    assert isinstance(ndarray_2d_list[1], aa.Grid2D)
    assert (
        ndarray_2d_list[1].native
        == np.array(
            [
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
    ).all()


def test__in_grid_2d_irregular__out_ndarray_2d():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d_irregular = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_2d = obj.ndarray_2d_from(grid=grid_2d_irregular)

    assert ndarray_2d.in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]


def test__in_grid_2d_irregular__out_ndarray_2d_list():
    obj = aa.m.MockGrid2DLikeObj()

    grid_2d = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    ndarray_2d_list = obj.ndarray_2d_list_from(grid=grid_2d)

    assert ndarray_2d_list[0].in_list == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert ndarray_2d_list[1].in_list == [(2.0, 4.0), (6.0, 8.0), (10.0, 12.0)]


def test__in_ndarray__out_ndarray():
    obj = aa.m.MockGrid2DLikeObj()

    grid = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    result = obj.ndarray_2d_raw_from(grid=grid)

    assert isinstance(result, np.ndarray)
    assert not isinstance(result, aa.Grid2D)
    assert not isinstance(result, aa.Grid2DIrregular)


class MockGrid2DPassThroughObj:
    """
    Mimics the profile classes in **PyAutoGalaxy** whose `@to_grid` decorated methods
    delegate to another `@to_grid` decorated method, such that the value the decorator
    wraps up is already a `Grid2D`.
    """

    def __init__(self, over_sampled=None):
        self.centre = (0.0, 0.0)
        self.over_sampled = over_sampled

    @aa.decorators.to_grid
    def grid_2d_from(self, grid, *args, **kwargs):
        return aa.Grid2D(
            values=np.multiply(2.0, grid.array),
            mask=grid.mask,
            over_sample_size=grid.over_sample_size,
            over_sampled=self.over_sampled,
        )


def _mask_2x2():
    return aa.Mask2D(
        mask=[
            [True, True, True, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )


def test__to_grid__does_not_materialise_over_sampled_of_wrapped_grid(monkeypatch):
    """
    Reading the public `over_sampled` / `over_sampler` properties inside `to_grid`
    materialises them via a per-pixel Python loop, which dominates the runtime of
    deflection angle calculations. Only an explicitly set value may propagate.
    """

    def _fail(self):
        pytest.fail("over_sampled materialised inside to_grid")

    def _fail_sampler(self):
        pytest.fail("over_sampler materialised inside to_grid")

    monkeypatch.setattr(aa.Grid2D, "over_sampled", property(_fail))
    monkeypatch.setattr(aa.Grid2D, "over_sampler", property(_fail_sampler))

    grid_2d = aa.Grid2D.from_mask(mask=_mask_2x2(), over_sample_size=2)

    result = MockGrid2DPassThroughObj().grid_2d_from(grid=grid_2d)

    assert isinstance(result, aa.Grid2D)
    assert result._over_sampled is None
    assert result._over_sampler is None


def test__to_grid__propagates_explicitly_set_over_sampled():
    grid_2d = aa.Grid2D.from_mask(mask=_mask_2x2(), over_sample_size=2)

    sentinel = aa.Grid2DIrregular(values=[(1.0, 2.0), (3.0, 4.0)])

    result = MockGrid2DPassThroughObj(over_sampled=sentinel).grid_2d_from(grid=grid_2d)

    assert result._over_sampled is sentinel
    assert result.over_sampled is sentinel
