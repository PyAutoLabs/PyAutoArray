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
