import importlib.util
import warnings

import numpy as np
import pytest

import autoarray as aa


@pytest.fixture(name="three_pixels")
def make_three_pixels():
    return np.array([[0, 0], [0, 1], [1, 0]])


@pytest.fixture(name="five_pixels")
def make_five_pixels():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])


def test__mapping_matrix(three_pixels, five_pixels):
    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    slim_index_for_sub_slim_index = np.array([0, 1, 2])

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_weights_for_sub_slim_index=np.ones((3, 1), dtype="int"),
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=np.ones(3, dtype="int"),
        pixels=6,
        total_mask_pixels=3,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([1.0, 1.0, 1.0]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [1, 0, 0, 0, 0, 0],  # Imaging pixel 0 maps to pix pixel 0.
                [0, 1, 0, 0, 0, 0],  # Imaging pixel 1 maps to pix pixel 1.
                [0, 0, 1, 0, 0, 0],
            ]
        )
    ).all()  # Imaging pixel 2 maps to pix pixel 2

    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2], [7], [6]])
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 3, 4])

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=np.ones(5, dtype="int"),
        pix_weights_for_sub_slim_index=np.ones((5, 1), dtype="int"),
        pixels=8,
        total_mask_pixels=5,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )
    ).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2]]
    ).T
    slim_index_for_sub_slim_index = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    )
    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_weights_for_sub_slim_index=np.ones((20, 1), dtype="int"),
        pix_size_for_sub_slim_index=np.ones(20, dtype="int"),
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pixels=8,
        total_mask_pixels=5,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([0.25, 0.25, 0.25, 0.25, 0.25]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0],
                [0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0],
                [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25],
            ]
        )
    ).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0, 0, 0, 1, 1, 1, 0, 0, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2]]
    ).T
    slim_index_for_sub_slim_index = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    )

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_weights_for_sub_slim_index=np.ones((20, 1), dtype="int"),
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=np.ones(20, dtype="int"),
        pixels=8,
        total_mask_pixels=5,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([0.25, 0.25, 0.25, 0.25, 0.25]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [0.75, 0.25, 0, 0, 0, 0, 0, 0],
                [0.5, 0.5, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25],
            ]
        )
    ).all()

    pix_indexes_for_sub_slim_index = np.array(
        [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
            ]
        ]
    ).T

    slim_index_for_sub_slim_index = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
    )

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        pix_weights_for_sub_slim_index=np.ones((48, 1), dtype="int"),
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=np.ones(48, dtype="int"),
        pixels=6,
        total_mask_pixels=3,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        sub_fraction=np.array([1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0]),
    )

    assert (
        mapping_matrix
        == np.array(
            [
                [0.75, 0.25, 0, 0, 0, 0],
                [0, 0, 1.0, 0, 0, 0],
                [0.1875, 0.1875, 0.1875, 0.1875, 0.125, 0.125],
            ]
        )
    ).all()


def test__data_to_pix_unique_from():
    image_pixels = 2
    sub_size = np.array([2, 2])

    pix_indexes_for_sub_slim_index = np.array(
        [[0, -1], [0, -1], [0, -1], [1, -1], [2, -1], [1, -1], [0, -1], [2, -1]]
    ).astype("int")
    pix_size_for_sub_slim_index = np.array([1, 1, 1, 1, 1, 1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array(
        [
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
        ]
    )

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper_numba.data_slim_to_pixelization_unique_from(
        data_pixels=image_pixels,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=sub_size,
    )

    assert (data_to_pix_unique[0, :] == np.array([0, 1, -1, -1])).all()
    assert (data_weights[0, :] == np.array([0.75, 0.25, 0.0, 0.0])).all()
    assert (data_to_pix_unique[1, :] == np.array([2, 1, 0, -1])).all()
    assert (data_weights[1, :] == np.array([0.5, 0.25, 0.25, 0.0])).all()
    assert (pix_lengths == np.array([2, 3])).all()

    pix_indexes_for_sub_slim_index = np.array(
        [[0, 1], [0, 1], [0, 2], [1, -1], [2, -1], [1, -1], [0, -1], [2, -1]]
    ).astype("int")
    pix_size_for_sub_slim_index = np.array([2, 2, 2, 1, 1, 1, 1, 1]).astype("int")
    pix_weights_for_sub_slim_index = np.array(
        [
            [0.5, 0.5],
            [0.25, 0.75],
            [0.75, 0.25],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
            [1.0, -1],
        ]
    )

    (
        data_to_pix_unique,
        data_weights,
        pix_lengths,
    ) = aa.util.mapper_numba.data_slim_to_pixelization_unique_from(
        data_pixels=image_pixels,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_sizes_for_sub_slim_index=pix_size_for_sub_slim_index,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        pix_pixels=3,
        sub_size=sub_size,
    )

    assert (data_to_pix_unique[0, :] == np.array([0, 1, 2, -1, -1, -1, -1, -1])).all()
    assert (
        data_weights[0, :] == np.array([0.375, 0.5625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0])
    ).all()
    assert (data_to_pix_unique[1, :] == np.array([2, 1, 0, -1, -1, -1, -1, -1])).all()
    assert (
        data_weights[1, :] == np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0])
    ).all()
    assert (pix_lengths == np.array([3, 3])).all()


def test__adaptive_pixel_signals_from():
    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pixel_weights = np.ones((3, 1), dtype="int")
    pixel_sizes = np.ones(3, dtype="int")
    slim_index_for_sub_slim_index = np.array([0, 1, 2])
    galaxy_image = np.array([1.0, 1.0, 1.0])

    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=3,
        signal_scale=1.0,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pixel_sizes,
        pixel_weights=pixel_weights,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        adapt_data=galaxy_image,
    )

    assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2], [0]])
    pixel_weights = np.ones((4, 1), dtype="int")
    pixel_sizes = np.ones(4, dtype="int")
    slim_index_for_sub_slim_index = np.array([0, 1, 2, 0])
    galaxy_image = np.array([1.0, 1.0, 1.0, 1.0])

    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=3,
        signal_scale=1.0,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pixel_sizes,
        pixel_weights=pixel_weights,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        adapt_data=galaxy_image,
    )

    assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pixel_weights = np.ones((3, 1), dtype="int")
    pixel_sizes = np.ones(3, dtype="int")
    slim_index_for_sub_slim_index = np.array([0, 1, 2])
    galaxy_image = np.array([2.0, 1.0, 1.0])

    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=3,
        signal_scale=1.0,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pixel_sizes,
        pixel_weights=pixel_weights,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        adapt_data=galaxy_image,
    )

    assert (pixel_signals == np.array([1.0, 0.5, 0.5])).all()

    pix_indexes_for_sub_slim_index = np.array([[0], [1], [2]])
    pixel_weights = np.ones((3, 1), dtype="int")
    pixel_sizes = np.ones(3, dtype="int")
    slim_index_for_sub_slim_index = np.array([0, 1, 2])
    galaxy_image = np.array([2.0, 1.0, 1.0])

    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=3,
        signal_scale=2.0,
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=pixel_sizes,
        pixel_weights=pixel_weights,
        slim_index_for_sub_slim_index=slim_index_for_sub_slim_index,
        adapt_data=galaxy_image,
    )

    assert (pixel_signals == np.array([1.0, 0.25, 0.25])).all()


def test_mapped_to_source_via_mapping_matrix_from():
    mapping_matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    array_slim = np.array([1.0, 2.0, 3.0])

    mapped_to_source = aa.util.mapper.mapped_to_source_via_mapping_matrix_from(
        mapping_matrix=mapping_matrix, array_slim=array_slim
    )

    assert (mapped_to_source == np.array([1.0, 2.5])).all()

    mapping_matrix = np.array(
        [
            [0.25, 0.5, 0.25],
            [0.0, 0.5, 0.5],
            [0.0, 0.25, 0.75],
            [0.5, 0.5, 0.0],
            [0.25, 0.75, 0.0],
        ]
    )

    array_slim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    mapped_to_source = aa.util.mapper.mapped_to_source_via_mapping_matrix_from(
        mapping_matrix=mapping_matrix, array_slim=array_slim
    )


#    assert (mapped_to_source == np.array([3.5, 8.0, 3.5])).all()


# ----------------------------------------------------------------------------
# Zero-signal adapt image: NumPy/JAX parity (PyAutoArray bug, 2026-09-06)
#
# `adaptive_pixel_signals_from` normalises by the maximum pixel signal. When the
# adapt image carries no signal where the pixels are -- e.g. an adapt image
# built from the *unlensed* source profile, a compact blob sitting where the
# Einstein ring is not -- every pixel signal is zero and so is the maximum.
#
# The normalisation used to read `xp.where(max_sig > 0, pixel_signals / max_sig,
# pixel_signals)`, which guards the *selection* but still evaluates the 0/0
# division. NumPy discarded the resulting NaN along with the unselected branch
# (emitting only a `RuntimeWarning`) while JAX propagated it, so the two
# backends returned different answers for the same input: a finite likelihood
# on NumPy, NaN on JAX, with `fitness._vmap` collapsing to the resample
# figure-of-merit.
# ----------------------------------------------------------------------------

# jax is an `[optional]` extra and is absent on the NumPy-only matrix env, so
# the JAX parity test skips rather than fails there (same convention as
# test_delaunay.py).
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)


def _zero_signal_kwargs():
    """Three pixels, each mapped by one sub-pixel, over an all-zero adapt image."""
    return dict(
        pixels=3,
        signal_scale=1.0,
        pix_indexes_for_sub_slim_index=np.array([[0], [1], [2]]),
        pix_size_for_sub_slim_index=np.ones(3, dtype="int"),
        pixel_weights=np.ones((3, 1), dtype="int"),
        slim_index_for_sub_slim_index=np.array([0, 1, 2]),
        adapt_data=np.zeros(3),
    )


def test__adaptive_pixel_signals_from__zero_signal_adapt_data__is_finite():
    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(**_zero_signal_kwargs())

    assert np.isfinite(np.asarray(pixel_signals)).all()
    assert np.asarray(pixel_signals) == pytest.approx(np.zeros(3), abs=1.0e-10)


def test__adaptive_pixel_signals_from__zero_signal_adapt_data__no_divide_warning():
    # The 0/0 that used to raise this warning is the same one that became a NaN
    # on the JAX path, so the clean-warning assertion pins the fix at its source.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        aa.util.mapper.adaptive_pixel_signals_from(**_zero_signal_kwargs())


@pytest.mark.parametrize("signal_scale", [0.5, 1.0, 2.0])
def test__adaptive_pixel_signals_from__zero_signal__finite_for_any_signal_scale(
    signal_scale,
):
    # `0.0 ** signal_scale` is finite forwards for every positive exponent, but
    # a fractional one made the *derivative* infinite; the guard has to hold for
    # all three without changing the value.
    kwargs = _zero_signal_kwargs()
    kwargs["signal_scale"] = signal_scale

    pixel_signals = aa.util.mapper.adaptive_pixel_signals_from(**kwargs)

    assert np.isfinite(np.asarray(pixel_signals)).all()
    assert np.asarray(pixel_signals) == pytest.approx(np.zeros(3), abs=1.0e-10)


@requires_jax
def test__adaptive_pixel_signals_from__zero_signal__jax_matches_numpy():
    import jax.numpy as jnp

    kwargs = _zero_signal_kwargs()

    numpy_signals = np.asarray(aa.util.mapper.adaptive_pixel_signals_from(**kwargs))
    jax_signals = np.asarray(
        aa.util.mapper.adaptive_pixel_signals_from(**kwargs, xp=jnp)
    )

    assert np.isfinite(jax_signals).all()
    assert jax_signals == pytest.approx(numpy_signals, abs=1.0e-10)


@requires_jax
def test__adaptive_pixel_signals_from__signal_present__jax_matches_numpy():
    # The parity has to hold where the function was already well-defined too,
    # otherwise the zero-signal guard could pass by breaking the ordinary path.
    import jax.numpy as jnp

    kwargs = _zero_signal_kwargs()
    kwargs["adapt_data"] = np.array([2.0, 1.0, 1.0])

    numpy_signals = np.asarray(aa.util.mapper.adaptive_pixel_signals_from(**kwargs))
    jax_signals = np.asarray(
        aa.util.mapper.adaptive_pixel_signals_from(**kwargs, xp=jnp)
    )

    assert numpy_signals == pytest.approx(np.array([1.0, 0.5, 0.5]), abs=1.0e-10)
    assert jax_signals == pytest.approx(numpy_signals, abs=1.0e-10)


@requires_jax
def test__adaptive_pixel_signals_from__zero_signal__grad_is_finite():
    # The NaN this task exists for reached the likelihood through `grad`/`vmap`,
    # not through the forward pass alone, so the guard is pinned there as well.
    import jax
    import jax.numpy as jnp

    kwargs = _zero_signal_kwargs()
    adapt_data = kwargs.pop("adapt_data")

    def total_signal(data):
        return jnp.sum(
            aa.util.mapper.adaptive_pixel_signals_from(
                **kwargs, adapt_data=data, xp=jnp
            )
        )

    gradient = jax.grad(total_signal)(jnp.asarray(adapt_data))

    assert np.isfinite(np.asarray(gradient)).all()


# ----------------------------------------------------------------------------
# Mixed precision is a JAX-only allocation choice (PyAutoArray#552).
#
# `use_mixed_precision` buys GPU throughput on the JAX path. Applying it on the
# NumPy path as well silently removed the fp64 reference a NumPy fit is
# supposed to be: the workspace smoke scripts compare a NumPy fit against a
# mixed-precision JAX fit and call the difference the mixed-precision error.
# ----------------------------------------------------------------------------


def _mixed_precision_mapping_kwargs():
    """
    Three image pixels, each mapped by two weighted sub-pixels, with weights
    that are not representable in fp32 so a downcast is visible in the values
    as well as in the dtype.
    """
    return dict(
        pix_indexes_for_sub_slim_index=np.array([[0, 1], [1, 2], [2, 0]]),
        pix_size_for_sub_slim_index=np.array([2, 2, 2]),
        pix_weights_for_sub_slim_index=np.array(
            [[1.0 / 3.0, 2.0 / 3.0], [0.7, 0.3], [0.1, 0.9]]
        ),
        pixels=3,
        total_mask_pixels=3,
        slim_index_for_sub_slim_index=np.array([0, 1, 2]),
        sub_fraction=np.array([1.0, 1.0, 1.0]),
    )


def test__mapping_matrix_from__mixed_precision__numpy_stays_float64():
    kwargs = _mixed_precision_mapping_kwargs()

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        **kwargs, use_mixed_precision=True
    )

    assert mapping_matrix.dtype == np.float64

    mapping_matrix = aa.util.mapper.mapping_matrix_from(
        **kwargs, use_mixed_precision=False
    )

    assert mapping_matrix.dtype == np.float64


@requires_jax
def test__mapping_matrix_from__mixed_precision__jax_is_float32_and_matches_numpy():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    kwargs = _mixed_precision_mapping_kwargs()

    numpy_matrix = aa.util.mapper.mapping_matrix_from(
        **kwargs, use_mixed_precision=True
    )
    jax_matrix = aa.util.mapper.mapping_matrix_from(
        **kwargs, use_mixed_precision=True, xp=jnp
    )

    assert numpy_matrix.dtype == np.float64
    assert jax_matrix.dtype == jnp.float32

    assert np.asarray(jax_matrix) == pytest.approx(numpy_matrix, abs=1.0e-6)


@requires_jax
def test__mapping_matrix_from__no_mixed_precision__float64_on_both_backends():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    kwargs = _mixed_precision_mapping_kwargs()

    numpy_matrix = aa.util.mapper.mapping_matrix_from(
        **kwargs, use_mixed_precision=False
    )
    jax_matrix = aa.util.mapper.mapping_matrix_from(
        **kwargs, use_mixed_precision=False, xp=jnp
    )

    assert numpy_matrix.dtype == np.float64
    assert jax_matrix.dtype == jnp.float64

    assert np.asarray(jax_matrix) == pytest.approx(numpy_matrix, abs=1.0e-14)


# ----------------------------------------------------------------------------
# Zero-weight mapping entries are not counted (PyAutoArray#627).
#
# A data point lying exactly on a Delaunay vertex has barycentric weights
# (0, 0, 1) in *every* triangle around that vertex, and a point on an edge has
# a zero weight in either neighbouring triangle. scipy's `find_simplex` and the
# JAX Delaunay walk legitimately choose different triangles for such points, so
# the two backends' mapping tables carry different zero-weight entries.
# `adaptive_pixel_signals_from` used to count those entries when normalising
# each pixel's signal, which made the pixel signals -- and through them the
# AdaptSplit regularization weights and matrix -- depend on the triangle chosen.
# Only non-zero-weight entries are counted now, and the set of non-zero
# (index, weight) pairs is unique.
# ----------------------------------------------------------------------------


def _vertex_coincident_delaunay_setup():
    """
    A 16x16 data grid with a smooth `0.01 * sin(7 * grid)` distortion, and a mesh that is
    the stride-2 subsample of the same points, so 64 data points sit exactly on mesh vertices
    (the geometry of a grid-subsample / `Overlay` image mesh traced through a lens).
    """
    n = 16

    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")

    data_grid = np.stack([yy.ravel(), xx.ravel()], axis=1)
    data_grid = data_grid + 0.01 * np.sin(7 * data_grid[:, ::-1])

    mesh_grid = data_grid.reshape(n, n, 2)[::2, ::2].reshape(-1, 2)

    adapt_data = np.exp(-np.sum(data_grid**2, axis=1) / 0.3)

    return data_grid, mesh_grid, adapt_data


def _delaunay_tables_from(delaunay_fn, data_grid, mesh_grid, xp):
    from autoarray.inversion.mesh.interpolator.delaunay import (
        pixel_weights_delaunay_from,
    )

    mappings = np.asarray(
        delaunay_fn(xp.asarray(mesh_grid), xp.asarray(data_grid), 0.5)[2]
    )

    sizes = (mappings >= 0).sum(axis=1)

    weights = np.asarray(
        pixel_weights_delaunay_from(
            data_grid=xp.asarray(data_grid),
            mesh_grid=xp.asarray(mesh_grid),
            pix_indexes_for_sub_slim_index=xp.asarray(mappings),
            xp=xp,
        )
    )

    return mappings, sizes, weights


@requires_jax
def test__adaptive_pixel_signals_from__scipy_and_jax_delaunay_tables__agree():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from autoarray.inversion.mesh.interpolator.delaunay import (
        jax_delaunay,
        scipy_delaunay,
    )

    data_grid, mesh_grid, adapt_data = _vertex_coincident_delaunay_setup()

    scipy_tables = _delaunay_tables_from(scipy_delaunay, data_grid, mesh_grid, xp=np)
    jax_tables = _delaunay_tables_from(jax_delaunay, data_grid, mesh_grid, xp=jnp)

    # The precondition the test exists for: the two point-location routines pick different
    # (equally valid) triangles for some of the vertex-coincident data points.
    rows_differ = np.any(
        np.sort(scipy_tables[0], axis=1) != np.sort(jax_tables[0], axis=1), axis=1
    )
    assert rows_differ.sum() > 0

    def pixel_signals_from(tables):
        mappings, sizes, weights = tables

        return aa.util.mapper.adaptive_pixel_signals_from(
            pixels=mesh_grid.shape[0],
            pixel_weights=weights,
            signal_scale=1.0,
            pix_indexes_for_sub_slim_index=mappings,
            pix_size_for_sub_slim_index=sizes,
            slim_index_for_sub_slim_index=np.arange(data_grid.shape[0]),
            adapt_data=adapt_data,
            exclude_zero_weight_mappings=True,
        )

    np.testing.assert_allclose(
        pixel_signals_from(scipy_tables),
        pixel_signals_from(jax_tables),
        rtol=0.0,
        atol=1.0e-12,
    )


def test__adaptive_pixel_signals_from__vertex_point_via_two_triangles__identical():
    """
    Pure NumPy: data point 0 lies exactly on vertex 0. Described through triangle (0, 1, 2)
    or through triangle (0, 3, 4) its weights are (1, 0, 0); the other two data points are
    ordinary interior points. Only the zero-weight neighbours differ, so the signals must not.
    """
    kwargs = dict(
        pixels=5,
        signal_scale=1.0,
        pix_size_for_sub_slim_index=np.array([3, 3, 3]),
        slim_index_for_sub_slim_index=np.array([0, 1, 2]),
        adapt_data=np.array([3.0, 1.0, 2.0]),
        exclude_zero_weight_mappings=True,
    )

    weights = np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])

    signals_via_012 = aa.util.mapper.adaptive_pixel_signals_from(
        pix_indexes_for_sub_slim_index=np.array([[0, 1, 2], [0, 1, 3], [0, 3, 4]]),
        pixel_weights=weights,
        **kwargs,
    )
    signals_via_034 = aa.util.mapper.adaptive_pixel_signals_from(
        pix_indexes_for_sub_slim_index=np.array([[0, 3, 4], [0, 1, 3], [0, 3, 4]]),
        pixel_weights=weights,
        **kwargs,
    )

    assert np.array_equal(signals_via_012, signals_via_034)

    # Pixel 0 is mapped by all three points with non-zero weight; the zero-weight entries of
    # point 0 add nothing to pixels 1 and 2 (or 3 and 4) and are not counted there either.
    pixel_sums = np.array([3.0 * 1.0 + 0.2 + 2.0 * 0.6, 0.3, 0.0, 0.5 + 0.2, 0.6])
    pixel_counts = np.array([3.0, 1.0, 1.0, 2.0, 1.0])
    expected = pixel_sums / pixel_counts

    assert signals_via_012 == pytest.approx(expected / expected.max(), rel=1.0e-14)


def test__adaptive_pixel_signals_from__default_flag__zero_weight_entries_still_counted():
    """
    Without `exclude_zero_weight_mappings` the util keeps its previous behaviour: every entry within
    a data point's size counts towards its pixel, including zero-weight ones. Same table as the
    two-triangle test above, described through triangle (0, 1, 2).
    """
    signals = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=5,
        signal_scale=1.0,
        pixel_weights=np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.6, 0.1, 0.3]]),
        pix_indexes_for_sub_slim_index=np.array([[0, 1, 2], [0, 1, 3], [0, 3, 4]]),
        pix_size_for_sub_slim_index=np.array([3, 3, 3]),
        slim_index_for_sub_slim_index=np.array([0, 1, 2]),
        adapt_data=np.array([3.0, 1.0, 2.0]),
    )

    # Point 0's zero-weight entries on pixels 1 and 2 add nothing to the sums but are counted.
    pixel_sums = np.array([3.0 * 1.0 + 0.2 + 2.0 * 0.6, 0.3, 0.0, 0.5 + 0.2, 0.6])
    pixel_counts = np.array([3.0, 2.0, 1.0, 2.0, 1.0])
    expected = pixel_sums / pixel_counts

    assert signals == pytest.approx(expected / expected.max(), rel=1.0e-14)


def _mapper_from(mesh, data_grid, mesh_grid, adapt_data):
    interpolator = mesh.interpolator_from(
        source_plane_data_grid=aa.Grid2D.no_mask(
            values=data_grid, shape_native=(data_grid.shape[0], 1), pixel_scales=1.0
        ),
        source_plane_mesh_grid=aa.Grid2DIrregular(mesh_grid),
        adapt_data=aa.Array2D.no_mask(
            values=adapt_data, shape_native=(adapt_data.shape[0], 1), pixel_scales=1.0
        ),
    )

    return aa.Mapper(interpolator=interpolator)


def _util_pixel_signals_from(mapper, adapt_data, exclude_zero_weight_mappings):
    return aa.util.mapper.adaptive_pixel_signals_from(
        pixels=mapper.pixels,
        signal_scale=1.0,
        pixel_weights=mapper.pix_weights_for_sub_slim_index,
        pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
        slim_index_for_sub_slim_index=mapper.over_sampler.slim_for_sub_slim,
        adapt_data=adapt_data,
        exclude_zero_weight_mappings=exclude_zero_weight_mappings,
    )


def test__mapper_pixel_signals_from__knn_barycentric_keeps_counting_clipped_zero_weights():
    """
    `KNNBarycentric` clips negative barycentric weights to exactly zero for every data point outside
    the triangle of its 3 nearest vertices, so its tables carry zero weights over whole regions. Those
    entries keep being counted (PyAutoArray#627 scopes the exclusion to Delaunay tables), so its pixel
    signals are exactly those of the default util path.

    The KNN interpolator computes its weights with JAX, so this test skips on the no-JAX leg.
    """
    pytest.importorskip("jax")

    data_grid, mesh_grid, adapt_data = _vertex_coincident_delaunay_setup()
    mesh_grid = mesh_grid + np.random.default_rng(1).normal(0.0, 0.03, mesh_grid.shape)

    mapper = _mapper_from(
        aa.mesh.KNNBarycentric(pixels=mesh_grid.shape[0]),
        data_grid,
        mesh_grid,
        adapt_data,
    )

    weights = np.asarray(mapper.pix_weights_for_sub_slim_index)
    in_size = (
        np.arange(weights.shape[1])[None, :]
        < np.asarray(mapper.pix_sizes_for_sub_slim_index)[:, None]
    )
    assert ((weights == 0.0) & in_size).sum() > 0

    assert mapper.interpolator.exclude_zero_weight_mappings is False

    signals = mapper.pixel_signals_from(signal_scale=1.0)

    assert np.array_equal(
        signals,
        _util_pixel_signals_from(
            mapper, adapt_data, exclude_zero_weight_mappings=False
        ),
    )
    assert not np.allclose(
        signals,
        _util_pixel_signals_from(mapper, adapt_data, exclude_zero_weight_mappings=True),
    )


def test__mapper_pixel_signals_from__delaunay_excludes_zero_weights():
    data_grid, mesh_grid, adapt_data = _vertex_coincident_delaunay_setup()

    for mesh in [
        aa.mesh.Delaunay(pixels=mesh_grid.shape[0]),
        aa.mesh.DelaunayNN(pixels=mesh_grid.shape[0]),
    ]:
        mapper = _mapper_from(mesh, data_grid, mesh_grid, adapt_data)

        assert mapper.interpolator.exclude_zero_weight_mappings is True

        assert np.array_equal(
            mapper.pixel_signals_from(signal_scale=1.0),
            _util_pixel_signals_from(
                mapper, adapt_data, exclude_zero_weight_mappings=True
            ),
        )
