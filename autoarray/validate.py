"""
Shared constructor-input validation helpers.

**This module is the agreed home for the PyAuto validation guards.** PyAutoArray is
the floor that PyAutoGalaxy and PyAutoLens both build on, so the guards live here
and are imported downstream (``from autoarray import validate``) rather than being
re-invented per repo. That keeps one message shape for one class of mistake across
all three libraries — the failure mode this module exists to prevent is three repos
telling a user three different things about the same bad input.

__Message shape__

Every message names the parameter, states the rule, and shows what was received:

::

    pixel_scales must be a finite positive number; got -0.1

Callers may append a sentence of context (``extra``) when the rule alone is not
enough to act on, e.g. pointing out that swapped arguments are the usual cause.

__Tracer safety__

These constructors sit on JAX-traced paths — ``coefficient`` in particular is a
free model parameter, so under a traced fit a constructor is handed a JAX tracer
rather than a number. A plain ``if value < 0`` on a tracer raises
``TracerBoolConversionError``, so **every value guard here is gated on
:func:`is_concrete_scalar` first** and silently passes anything it does not
recognise. The guards therefore catch the mistakes a human makes when writing a
script by hand (which is where all of these were reported) and cost nothing inside
a trace.

The same reasoning already appears at ``autoarray/dataset/imaging/simulator.py``,
where a NaN check is guarded by ``if xp is np``. A constructor has no ``xp``
argument to test, so the value's own type is the gate instead.

Shape checks need no such gate: array shapes are static under JAX, so a shape is
always concrete.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple, Type, Union

import numpy as np


def is_concrete_scalar(value: Any) -> bool:
    """
    Returns ``True`` if ``value`` is a concrete Python or NumPy real scalar, and is
    therefore safe to compare against with a plain Python ``if``.

    This is the gate that makes every guard in this module tracer-safe. A JAX tracer,
    a ``numpy`` array, ``None``, a string or any other object returns ``False`` and is
    passed through unvalidated — a guard's job is to catch obvious hand-written
    mistakes, not to police types inside a trace.

    ``bool`` is excluded deliberately: it is a subclass of ``int`` in Python, and
    ``True`` reaching a numeric parameter is a different mistake than the ones these
    guards describe.

    Parameters
    ----------
    value
        The value to test.
    """
    if isinstance(value, bool):
        return False

    return isinstance(value, (int, float, np.integer, np.floating))


def _raise(
    name: str,
    rule: str,
    value: Any,
    exc_type: Type[Exception],
    extra: str = "",
):
    message = f"{name} must be {rule}; got {value!r}"

    if extra:
        message = f"{message}. {extra}"

    raise exc_type(message)


def validate_positive_finite(
    value: Any,
    name: str,
    exc_type: Type[Exception] = ValueError,
    extra: str = "",
):
    """
    Raise if ``value`` is a concrete scalar which is not finite and strictly positive.

    Used for quantities where zero is as meaningless as a negative — a ``pixel_scales``
    of ``0.0`` makes every pixel-to-scaled conversion a division by zero, and a
    negative one silently flips the geometry.

    Non-concrete values (e.g. a JAX tracer) are passed through — see the module
    docstring.

    Parameters
    ----------
    value
        The value to validate.
    name
        The parameter name, used in the error message so the caller knows what to fix.
    exc_type
        The exception type to raise, so a module can raise its own (e.g.
        ``exc.MaskException``) instead of the ``ValueError`` default.
    extra
        An optional sentence of extra guidance appended to the message.
    """
    if not is_concrete_scalar(value):
        return

    if not np.isfinite(value) or value <= 0:
        _raise(name, "a finite positive number", value, exc_type, extra)


def validate_non_negative_finite(
    value: Any,
    name: str,
    exc_type: Type[Exception] = ValueError,
    extra: str = "",
):
    """
    Raise if ``value`` is a concrete scalar which is not finite and non-negative.

    Used where zero is a meaningful (if degenerate) request but a negative value is
    not — a regularization ``coefficient`` of ``0.0`` legitimately means "do not
    regularize", whereas a negative one is unphysical.

    Non-concrete values (e.g. a JAX tracer) are passed through — see the module
    docstring.

    Parameters
    ----------
    value
        The value to validate.
    name
        The parameter name, used in the error message so the caller knows what to fix.
    exc_type
        The exception type to raise.
    extra
        An optional sentence of extra guidance appended to the message.
    """
    if not is_concrete_scalar(value):
        return

    if not np.isfinite(value) or value < 0:
        _raise(name, "a finite non-negative number", value, exc_type, extra)


def validate_pixel_scales(
    pixel_scales: Union[float, Tuple[float, ...]],
    name: str = "pixel_scales",
    exc_type: Type[Exception] = ValueError,
):
    """
    Raise if any entry of ``pixel_scales`` is a concrete scalar which is not finite and
    strictly positive.

    Accepts either the single-``float`` or the per-axis tuple form, because it is
    called at the conversion chokepoint which sees both.

    Parameters
    ----------
    pixel_scales
        The pixel scale to validate, as a scalar or a tuple of per-axis scalars.
    name
        The parameter name used in the error message.
    exc_type
        The exception type to raise.
    """
    extra = (
        "A pixel scale is the scaled-units size of one pixel, so it must be above "
        "zero: a value of 0.0 makes every pixel-to-scaled conversion a division by "
        "zero, and a negative value silently flips the coordinate system"
    )

    if isinstance(pixel_scales, (tuple, list)):
        for index, pixel_scale in enumerate(pixel_scales):
            validate_positive_finite(
                value=pixel_scale,
                name=f"{name}[{index}]",
                exc_type=exc_type,
                extra=extra,
            )
        return

    validate_positive_finite(
        value=pixel_scales, name=name, exc_type=exc_type, extra=extra
    )


def validate_shape_native(
    shape_native: Sequence[int],
    name: str = "shape_native",
    exc_type: Type[Exception] = ValueError,
):
    """
    Raise if any entry of ``shape_native`` is not a positive integer.

    A zero-length axis produces a structure with no pixels at all, which every
    downstream calculation then silently returns nothing for, rather than failing
    where the mistake was made.

    Parameters
    ----------
    shape_native
        The (y,x) shape in pixels to validate.
    name
        The parameter name used in the error message.
    exc_type
        The exception type to raise.
    """
    for index, length in enumerate(shape_native):
        if not is_concrete_scalar(length):
            continue

        if length <= 0:
            _raise(
                f"{name}[{index}]",
                "a positive number of pixels",
                length,
                exc_type,
                f"The full {name} input was {tuple(shape_native)!r}, which describes a "
                f"structure containing no pixels",
            )


def validate_radii_ordered(
    inner_radius: Any,
    outer_radius: Any,
    inner_name: str = "inner_radius",
    outer_name: str = "outer_radius",
    exc_type: Type[Exception] = ValueError,
):
    """
    Raise if a concrete ``inner_radius`` is not strictly less than a concrete
    ``outer_radius``, or if either is negative or non-finite.

    An annulus whose inner radius is the larger of the two contains no pixels at all.
    That is almost always the two arguments being passed the wrong way round, so the
    message says so.

    Parameters
    ----------
    inner_radius
        The inner radius of the annulus.
    outer_radius
        The outer radius of the annulus.
    inner_name
        The inner parameter's name, used in the error message.
    outer_name
        The outer parameter's name, used in the error message.
    exc_type
        The exception type to raise.
    """
    validate_non_negative_finite(value=inner_radius, name=inner_name, exc_type=exc_type)
    validate_positive_finite(value=outer_radius, name=outer_name, exc_type=exc_type)

    if not is_concrete_scalar(inner_radius) or not is_concrete_scalar(outer_radius):
        return

    if inner_radius >= outer_radius:
        raise exc_type(
            f"{inner_name} must be less than {outer_name}; got {inner_name}="
            f"{inner_radius!r} and {outer_name}={outer_radius!r}. An annulus whose "
            f"inner radius is not smaller than its outer radius contains no pixels — "
            f"the usual cause is the two arguments being passed the wrong way round"
        )
