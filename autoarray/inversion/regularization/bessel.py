# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# Modifications Copyright 2026 PyAutoLabs.
#
# Vendored from `tensorflow_probability/substrates/jax/math/bessel.py`
# (tfp-nightly 0.26.0.dev20260713) so that autoarray's JAX Matérn-kernel
# regularization no longer depends on tensorflow-probability. Changes from the
# original:
#
# - Only the modified-Bessel-K path (`bessel_kve`) is kept. The I_v branch
#   (`bessel_ive`, `bessel_iv_ratio`, `_bessel_ive_shared`,
#   `_compute_general_continued_fraction`) and the log-space variants are
#   dropped, as `_temme_expansion` only needs them to produce `ive`.
# - The negative-order corrections are dropped from `_olver_asymptotic_uniform`
#   and `_temme_expansion`: `bessel_kve` folds `K_{-v} = K_v` before calling
#   them, so they only ever see `v >= 0` and those corrections only affected
#   `ive`.
# - The `tf`-on-JAX shims are translated to plain `jax.numpy` / `jax.lax`, and
#   tfp's `custom_gradient` decorator is replaced by `jax.custom_jvp` with the
#   same JVP rule (gradient in `z`; no gradient in `v`, as in tfp).
"""
Exponentially scaled modified Bessel function of the second kind, ``Kve``, in JAX.

``Kve(v, z) = K_v(z) * exp(|z|)`` for real order ``v`` and real ``z >= 0``.
``jax.scipy.special`` has no modified-Bessel-K of arbitrary real order, and the
Matérn kernel's ``nu`` is a continuous prior, so the JAX path of
``matern_kernel.kv_xp`` uses this implementation. The NumPy path uses
``scipy.special.kv`` and never imports this module.

Algorithm (Temme 1975; Numerical Recipes in C, 2nd ed., section 6.7):

- ``v >= 50``: Olver's uniform asymptotic expansion.
- ``v < 50``: split ``v = n + u`` with ``|u| <= 0.5``; evaluate ``K_u`` and
  ``K_{u+1}`` with Temme's power series (``z <= 2``) or Steed's continued
  fraction (``z > 2``); then the forward recurrence
  ``K_{v+1} = (2v / z) K_v + K_{v-1}`` up to order ``v``.

Gradients: ``d Kve / dz = (z - v) / z * Kve(v, z) - Kve(v - 1, z)`` is
registered with ``jax.custom_jvp``, so forward and reverse mode both work and
higher derivatives follow by recursion. As in tfp, **no gradient is defined in
``v``** (its tangent is ignored, giving a zero gradient).
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

__all__ = ["bessel_kve", "kv"]


def _sqrt1px2(x):
    """``sqrt(1 + x**2)`` without overflow for large ``|x|``."""
    eps = float(np.finfo(np.dtype(x.dtype)).eps)
    return jnp.where(
        jnp.abs(x) * np.sqrt(eps) <= 1.0,
        jnp.exp(0.5 * jnp.log1p(jnp.square(x))),
        jnp.abs(x),
    )


_ASYMPTOTIC_OLVER_EXPANSION_COEFFICIENTS = [
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     -0.20833333333333334, 0., 0.125, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., 0., 0., 0., 0.3342013888888889, 0.,
     -0.40104166666666669, 0., 0.0703125, 0., 0.0],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0., 0., -1.0258125964506173, 0., 1.8464626736111112,
     0., -0.89121093750000002, 0., 0.0732421875, 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     0., 0., 0., 4.6695844234262474, 0., -11.207002616222995, 0.,
     8.78912353515625, 0., -2.3640869140624998, 0., 0.112152099609375,
     0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
     -28.212072558200244, 0., 84.636217674600744, 0., -91.818241543240035,
     0., 42.534998745388457, 0., -7.3687943594796312, 0., 0.22710800170898438,
     0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 212.5701300392171, 0.,
     -765.25246814118157, 0., 1059.9904525279999, 0., -699.57962737613275,
     0., 218.19051174421159, 0., -26.491430486951554, 0., 0.57250142097473145,
     0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., -1919.4576623184068, 0.,
     8061.7221817373083, 0., -13586.550006434136, 0., 11655.393336864536,
     0., -5305.6469786134048, 0., 1200.9029132163525, 0.,
     -108.09091978839464, 0., 1.7277275025844574, 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 20204.291330966149, 0., -96980.598388637503, 0.,
     192547.0012325315, 0., -203400.17728041555, 0., 122200.46498301747,
     0., -41192.654968897557, 0., 7109.5143024893641, 0.,
     -493.915304773088, 0., 6.074042001273483, 0., 0., 0., 0., 0.,
     0., 0., 0.],
    [0., 0., 0., -242919.18790055133, 0., 1311763.6146629769, 0.,
     -2998015.9185381061, 0., 3763271.2976564039, 0., -2813563.2265865342, 0.,
     1268365.2733216248, 0., -331645.17248456361, 0., 45218.768981362737, 0.,
     -2499.8304818112092, 0., 24.380529699556064, 0., 0., 0., 0., 0.,
     0., 0., 0., 0.0],
    [3284469.8530720375, 0., -19706819.11843222, 0., 50952602.492664628,
     0., -74105148.211532637, 0., 66344512.274729028, 0., -37567176.660763353,
     0., 13288767.166421819, 0., -2785618.1280864552, 0., 308186.40461266245,
     0., -13886.089753717039, 0., 110.01714026924674, 0., 0., 0., 0., 0.,
     0., 0., 0., 0., 0.]
]  # fmt: skip


def _olver_asymptotic_kve(v, z):
    """
    ``Kve(v, z)`` from Olver's uniform asymptotic expansion (large ``v >= 0``).

    ``K_v(v * w) ~ f(w, v) * sum_k (-1)**k U_k(1 / sqrt(1 + w**2)) / v**k``,
    using the first 10 Debye polynomials ``U_k`` (DLMF 10.41).
    """
    v_abs = jnp.abs(v)
    w = z / v_abs
    t = jnp.reciprocal(_sqrt1px2(w))

    divisor = v_abs
    kve_sum = 1.0
    for i, coeffs in enumerate(_ASYMPTOTIC_OLVER_EXPANSION_COEFFICIENTS):
        coeff = 0.0
        for c in coeffs:
            coeff = coeff * t + c
        term = coeff / divisor
        kve_sum = kve_sum + (term if i % 2 == 1 else -term)
        divisor = divisor * v_abs

    # Rearranged (as in tfp) for stability since the `exp(z)` scaling is
    # subtracted off analytically.
    shared_prefactor = (
        jnp.reciprocal(_sqrt1px2(w) + w) + jnp.log(w) - jnp.log1p(jnp.reciprocal(t))
    )
    log_k_prefactor = 0.5 * jnp.log(np.pi * t / (2 * v_abs)) - v_abs * shared_prefactor
    return jnp.exp(log_k_prefactor + jnp.log(kve_sum))


def _evaluate_temme_coeffs(v):
    """
    Numerically stable Chebyshev evaluation of the Temme gamma coefficients.

    Returns ``coeff1 = (1/Gamma(1-v) - 1/Gamma(1+v)) / 2v``,
    ``coeff2 = (1/Gamma(1-v) + 1/Gamma(1+v)) / 2``, ``1/Gamma(1+v)`` and
    ``1/Gamma(1-v)`` (Numerical Recipes 6.7.18), avoiding the catastrophic
    cancellation of a naive evaluation.
    """
    coeff1_coeffs = [-1.142022680371168e0, 6.5165112670737e-3,
                     3.087090173086e-4, -3.4706269649e-6, 6.9437664e-9,
                     3.67795e-11, -1.356e-13]  # fmt: skip
    coeff2_coeffs = [1.843740587300905e0, -7.68528408447867e-2,
                     1.2719271366546e-3, -4.9717367042e-6, -3.31261198e-8,
                     2.423096e-10, -1.702e-13, -1.49e-15]  # fmt: skip
    w = 8 * jnp.square(v) - 1.0

    # Clenshaw's recurrence for the Chebyshev series.
    y = 2 * w

    prev = 0.0
    current = 0.0
    for i in reversed(range(1, len(coeff1_coeffs))):
        temp = current
        current = y * current - prev + coeff1_coeffs[i]
        prev = temp
    coeff1 = w * current - prev + 0.5 * coeff1_coeffs[0]

    prev = 0.0
    current = 0.0
    for i in reversed(range(1, len(coeff2_coeffs))):
        temp = current
        current = y * current - prev + coeff2_coeffs[i]
        prev = temp
    coeff2 = w * current - prev + 0.5 * coeff2_coeffs[0]
    gamma1pv = coeff2 - v * coeff1
    gamma1mv = coeff2 + v * coeff1
    return coeff1, coeff2, gamma1pv, gamma1mv


def _temme_series(v, z):
    """
    ``Kve(v, z)`` and ``Kve(v + 1, z)`` via Temme's power series.

    Assumes ``|v| <= 0.5`` and ``0 < z <= 2`` for fast convergence (Temme 1975;
    Numerical Recipes 6.7.39).
    """
    dtype = jnp.result_type(v, z)
    tol = np.finfo(np.dtype(dtype)).eps

    coeff1, coeff2, gamma1pv_inv, gamma1mv_inv = _evaluate_temme_coeffs(v)

    z_sq = jnp.square(z)

    logzo2 = jnp.log(z / 2.0)
    mu = -v * logzo2
    sinc_v = jnp.where(v == 0.0, 1.0, jnp.sin(np.pi * v) / (np.pi * v))
    sinhc_mu = jnp.where(mu == 0.0, 1.0, jnp.sinh(mu) / mu)
    # Numerical Recipes 6.7.17.
    initial_f = (coeff1 * jnp.cosh(mu) + coeff2 * -logzo2 * sinhc_mu) / sinc_v
    initial_p = 0.5 * jnp.exp(mu) / gamma1pv_inv
    initial_q = 0.5 * jnp.exp(-mu) / gamma1mv_inv
    max_iterations = 1000

    def body_fn(state):
        should_stop, index, f, p, q, coeff, kv_sum, kvp1_sum = state
        f = jnp.where(
            should_stop,
            f,
            (index * f + p + q) / (jnp.square(index) - jnp.square(v)),
        )
        p = jnp.where(should_stop, p, p / (index - v))
        q = jnp.where(should_stop, q, q / (index + v))
        h = p - index * f
        # c_k = (z ** 2 / 4) ** k / (k!)
        coeff = jnp.where(should_stop, coeff, coeff * z_sq / (4 * index))
        kv_sum = jnp.where(should_stop, kv_sum, kv_sum + coeff * f)
        kvp1_sum = jnp.where(should_stop, kvp1_sum, kvp1_sum + coeff * h)
        index = index + 1
        should_stop = (jnp.abs(coeff * f) < jnp.abs(kv_sum) * tol) | (
            index > max_iterations
        )
        return should_stop, index, f, p, q, coeff, kv_sum, kvp1_sum

    init = (
        jnp.zeros_like(initial_f, dtype=bool),
        jnp.asarray(1.0, dtype=dtype),
        initial_f,
        initial_p,
        initial_q,
        jnp.ones_like(initial_p),
        initial_f,
        initial_p,
    )
    _, _, _, _, _, _, kv_sum, kvp1_sum = lax.while_loop(
        lambda state: jnp.any(~state[0]), body_fn, init
    )

    kve = jnp.exp(jnp.log(kv_sum) + z)
    kvep1 = jnp.exp(jnp.log(2.0 * kvp1_sum) + z - jnp.log(z))
    return kve, kvep1


def _continued_fraction_kv(v, z):
    """
    ``Kve(v, z)`` and ``Kve(v + 1, z)`` via Steed's continued fraction.

    With ``k_n(z) = (-1)**n U(v + n + 0.5, 2v + 1, 2z)`` (``U`` the confluent
    hypergeometric function), ``K_v(z) = sqrt(pi) (2z)**v exp(-z) k_0(z)`` and
    ``K_{v+1}(z) = K_v(z) (v + z + 0.5 - k_1(z) / k_0(z))``. The ratio
    ``k_1 / k_0`` is a continued fraction evaluated with Steed's algorithm.
    Assumes ``|v| <= 0.5`` and ``z > 2`` (Temme 1975; Campbell 1980;
    Numerical Recipes 6.7.30-6.7.36).
    """
    dtype = jnp.result_type(v, z)
    tol = np.finfo(np.dtype(dtype)).eps
    max_iterations = 1000

    def steeds_algorithm(state):
        (
            should_stop,
            index,
            partial_numerator,
            partial_denominator,
            denominator_ratio,
            convergent_difference,
            hypergeometric_ratio,
            k_0,  # Terms for the recurrence in NR 6.7.36.
            k_1,
            c,  # Intermediate coefficient in NR 6.7.30.
            q,  # Intermediate sum in NR 6.7.35.
            hypergeometric_sum,
        ) = state
        # The numerator is v**2 - (index - 0.5) ** 2
        partial_numerator = partial_numerator - 2.0 * (index - 1.0)
        c = jnp.where(should_stop, c, -c * partial_numerator / index)
        next_k = (k_0 - partial_denominator * k_1) / partial_numerator
        k_0 = jnp.where(should_stop, k_0, k_1)
        k_1 = jnp.where(should_stop, k_1, next_k)
        q = jnp.where(should_stop, q, q + c * next_k)
        partial_denominator = partial_denominator + 2.0
        denominator_ratio = 1.0 / (
            partial_denominator + partial_numerator * denominator_ratio
        )
        convergent_difference = jnp.where(
            should_stop,
            convergent_difference,
            convergent_difference * (partial_denominator * denominator_ratio - 1.0),
        )
        hypergeometric_ratio = jnp.where(
            should_stop,
            hypergeometric_ratio,
            hypergeometric_ratio + convergent_difference,
        )
        hypergeometric_sum = jnp.where(
            should_stop,
            hypergeometric_sum,
            hypergeometric_sum + q * convergent_difference,
        )
        index = index + 1
        should_stop = (
            jnp.abs(q * convergent_difference) < jnp.abs(hypergeometric_sum) * tol
        ) | (index > max_iterations)
        return (
            should_stop,
            index,
            partial_numerator,
            partial_denominator,
            denominator_ratio,
            convergent_difference,
            hypergeometric_ratio,
            k_0,
            k_1,
            c,
            q,
            hypergeometric_sum,
        )

    initial_numerator = jnp.square(v) - 0.25
    initial_denominator = 2 * (z + 1.0)
    initial_ratio = 1.0 / initial_denominator + jnp.zeros_like(v)
    initial_seq = -initial_numerator + jnp.zeros_like(z)

    init = (
        jnp.zeros_like(v + z, dtype=bool),
        jnp.asarray(2.0, dtype=dtype),
        initial_numerator,
        initial_denominator,
        initial_ratio,
        initial_ratio,
        initial_ratio,
        jnp.zeros_like(v + z),
        jnp.ones_like(v + z),
        initial_seq,
        initial_seq,
        1 - initial_numerator * initial_ratio,
    )
    # Broadcast every loop carry to the common shape so `while_loop` sees a
    # fixed carry signature.
    shape = jnp.shape(v + z)
    init = (init[0], init[1]) + tuple(
        jnp.broadcast_to(x, shape).astype(dtype) for x in init[2:]
    )
    _, _, _, _, _, _, hypergeometric_ratio, _, _, _, _, hypergeometric_sum = (
        lax.while_loop(lambda state: jnp.any(~state[0]), steeds_algorithm, init)
    )

    log_kve = 0.5 * jnp.log(np.pi / (2 * z)) - jnp.log(hypergeometric_sum)
    log_kvp1e = (
        log_kve
        + jnp.log1p(2 * (v + z + initial_numerator * hypergeometric_ratio))
        - jnp.log(z)
        - float(np.log(2.0))
    )
    return jnp.exp(log_kve), jnp.exp(log_kvp1e)


def _temme_expansion_kve(v, x):
    """
    ``Kve(v, x)`` for ``0 <= v < 50`` and ``x > 0`` using Temme's method.

    Splits ``v = n + u`` (``|u| <= 0.5``), computes ``Kve(u)`` and
    ``Kve(u + 1)`` by power series (``x <= 2``) or continued fraction
    (``x > 2``), then applies the forward recurrence
    ``K_{v+1}(x) = (2v / x) K_v(x) + K_{v-1}(x)``, which is numerically stable
    in the forward direction and is also satisfied by ``K_v(x) exp(x)``.
    """
    dtype = jnp.result_type(v, x)
    v = jnp.abs(v)
    n = jnp.round(v)
    u = v - n
    x_abs = jnp.abs(x)

    # Feed each branch a benign dummy argument outside its regime so the
    # unselected branch cannot produce NaNs that leak through gradients.
    small_x = jnp.where(x_abs <= 2.0, x_abs, jnp.asarray(0.1, dtype))
    large_x = jnp.where(x_abs > 2.0, x_abs, jnp.asarray(1000.0, dtype))
    temme_kue, temme_kuep1 = _temme_series(u, small_x)
    cf_kue, cf_kuep1 = _continued_fraction_kv(u, large_x)

    kue = jnp.where(x_abs <= 2.0, temme_kue, cf_kue)
    kuep1 = jnp.where(x_abs <= 2.0, temme_kuep1, cf_kuep1)

    shape = jnp.shape(kue)
    n = jnp.broadcast_to(n, shape)
    u = jnp.broadcast_to(u, shape)
    x_abs = jnp.broadcast_to(x_abs, shape)

    def bessel_recurrence(state):
        index, kve, kvep1 = state
        next_kvep1 = 2 * (u + index) * kvep1 / x_abs + kve
        kve = jnp.where(index > n, kve, kvep1)
        kvep1 = jnp.where(index > n, kvep1, next_kvep1)
        return index + 1.0, kve, kvep1

    _, kve, _ = lax.while_loop(
        lambda state: jnp.any(state[0] <= n),
        bessel_recurrence,
        (jnp.asarray(1.0, dtype=dtype), kue, kuep1),
    )

    kve = jnp.where(x == 0.0, np.inf, kve)
    return jnp.where(x < 0.0, np.nan, kve)


def _bessel_kve_shared(v, z):
    """Compute ``Kve(v, z)`` (no custom derivative)."""
    # K_{-v} == K_{v}.
    v = jnp.abs(v)
    z_abs = jnp.abs(z)

    small_v = jnp.where(v < 50.0, v, jnp.asarray(0.1, v.dtype))
    large_v = jnp.where(v >= 50.0, v, jnp.asarray(1000.0, v.dtype))

    olver_kve = _olver_asymptotic_kve(large_v, z_abs)
    temme_kve = _temme_expansion_kve(small_v, z_abs)
    kve = jnp.where(v >= 50.0, olver_kve, temme_kve)

    kve = jnp.where(z == 0.0, np.inf, kve)
    return jnp.where(z < 0.0, np.nan, kve)


@jax.custom_jvp
def _bessel_kve_custom_gradient(v, z):
    return _bessel_kve_shared(v, z)


@_bessel_kve_custom_gradient.defjvp
def _bessel_kve_jvp(primals, tangents):
    """
    JVP for ``Kve``: ``dKve/dz = (z - v) / z * Kve(v, z) - Kve(v - 1, z)``.

    The ``v`` tangent is ignored (no gradient in the order), matching tfp.
    """
    v, z = primals
    _, dz = tangents
    kve = _bessel_kve_custom_gradient(v, z)
    pz = (z - v) / z * kve - _bessel_kve_custom_gradient(v - 1.0, z)
    dz = jnp.broadcast_to(dz, jnp.shape(pz))
    return kve, pz * dz


def bessel_kve(v, z):
    """
    Exponentially scaled modified Bessel function of the second kind.

    ``Kve(v, z) = K_v(z) * exp(|z|)``, a drop-in replacement for
    ``tensorflow_probability.substrates.jax.math.bessel_kve``.

    Parameters
    ----------
    v
        Real order, broadcastable with ``z``. ``K_{-v} = K_v``, so the sign is
        ignored. No gradient is defined with respect to ``v``.
    z
        Real argument, broadcastable with ``v``. Returns ``inf`` at ``z == 0``
        and ``nan`` for ``z < 0``.

    Returns
    -------
    The exponentially scaled ``K_v(z)``, with dtype the promoted dtype of
    ``v`` and ``z`` (float64 when ``jax_enable_x64`` is on).
    """
    dtype = jnp.result_type(float, v, z)
    v, z = jnp.broadcast_arrays(jnp.asarray(v, dtype), jnp.asarray(z, dtype))
    return _bessel_kve_custom_gradient(v, z)


def kv(v, z):
    """
    Modified Bessel function of the second kind ``K_v(z)`` (unscaled).

    ``K_v(z) = Kve(v, z) * exp(-|z|)``; see ``bessel_kve``.
    """
    z = jnp.asarray(z)
    return bessel_kve(v, z) * jnp.exp(-jnp.abs(z))
