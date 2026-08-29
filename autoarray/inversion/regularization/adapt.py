from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj

from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.regularization.abstract import validate_coefficient


def adapt_regularization_weights_from(
    inner_coefficient: float,
    outer_coefficient: float,
    pixel_signals: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """
    Returns the regularization weights for the adaptive regularization scheme (e.g. ``Adapt``).

    The weights define the effective regularization coefficient of every mesh parameter (typically pixels
    of a ``Mapper``).

    They are computed using an estimate of the expected signal in each pixel.

    Two regularization coefficients are used, corresponding to the:

    1) pixel_signals: pixels with a high pixel-signal (i.e. where the signal is located in the pixelization).
    2) 1.0 - pixel_signals: pixels with a low pixel-signal (i.e. where the signal is not located in the pixelization).

    Parameters
    ----------
    inner_coefficient
        The inner regularization coefficients which controls the degree of smoothing of the inversion reconstruction
        in the inner regions of a mesh's reconstruction.
    outer_coefficient
        The outer regularization coefficients which controls the degree of smoothing of the inversion reconstruction
        in the outer regions of a mesh's reconstruction.
    pixel_signals
        The estimated signal in every pixelization pixel, used to change the regularization weighting of high signal
        and low signal pixelizations.
    power
        The exponent the interpolated coefficient is raised to. The matrix builders square the returned weights
        again, so the coefficient enters the regularization matrix at the power ``2 * power``.

        The default ``2.0`` is the historical ``Adapt`` convention (a fourth-power coefficient dependence) and is
        what the legacy ``Adapt``, ``AdaptSplit``, ``AdaptSplitZeroth`` and ``MaternAdaptKernel`` classes pass.
        ``power=1.0`` gives the squared-once convention shared with ``Constant`` and is what the ``*Power``
        classes (e.g. ``AdaptPower``) pass by default.

    Returns
    -------
    np.ndarray
        The adaptive regularization weights which act as the effective regularization coefficients of
        every source pixel.
    """
    return (
        inner_coefficient * pixel_signals + outer_coefficient * (1.0 - pixel_signals)
    ) ** power


def weighted_regularization_matrix_from(
    regularization_weights: np.ndarray,
    neighbors: np.ndarray,
    xp=np,
) -> np.ndarray:
    """
    Returns the regularization matrix of the adaptive regularization scheme (e.g. ``Adapt``).

    This matrix is computed using the regularization weights of every mesh pixel, which are computed using the
    function ``adapt_regularization_weights_from``. These act as the effective regularization coefficients of
    every mesh pixel.

    The regularization matrix is computed using the pixel-neighbors array, which is setup using the appropriate
    neighbor calculation of the corresponding ``Mapper`` class.

    Parameters
    ----------
    regularization_weights
        The regularization weight of each pixel, adaptively governing the degree of gradient regularization
        applied to each inversion parameter (e.g. mesh pixels of a ``Mapper``).
    neighbors
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the mesh grid (entries of -1 correspond to no neighbor).
    neighbors_sizes
        An array of length (total_pixels) which gives the number of neighbors of every pixel in the
        Delaunay grid.

    Returns
    -------
    np.ndarray
        The regularization matrix computed using an adaptive regularization scheme where the effective regularization
        coefficient of every source pixel is different.
    """
    S, P = neighbors.shape
    reg_w = regularization_weights**2

    # 1) Flatten the (i→j) neighbor pairs
    I = xp.repeat(xp.arange(S), P)  # (S*P,)
    J = neighbors.reshape(-1)  # (S*P,)

    # 2) Remap “no neighbor” entries to an extra slot S, whose weight=0
    OUT = S
    J = xp.where(J < 0, OUT, J)

    # 3) Build an extended weight vector with a zero at index S
    reg_w_ext = xp.concatenate([reg_w, xp.zeros((1,))], axis=0)
    w_ij = reg_w_ext[J]  # (S*P,)

    # 4) Start with zeros on an (S+1)x(S+1) canvas so we can scatter into row S safely
    mat = xp.zeros((S + 1, S + 1), dtype=regularization_weights.dtype)

    # 5) Scatter into the diagonal:
    #    - the tiny 1e-8 floor on each i < S
    #    - sum_j reg_w[j] into diag[i]
    #    - sum contributions reg_w[j] into diag[j]
    #    (diagonal at OUT=S picks up zeros only)
    diag_updates_i = xp.concatenate(
        [xp.full((S,), 1e-8), xp.zeros((1,))], axis=0  # out‐of‐bounds slot stays zero
    )

    if xp.__name__.startswith("jax"):
        mat = mat.at[xp.diag_indices(S + 1)].add(diag_updates_i)
        mat = mat.at[I, I].add(w_ij)
        mat = mat.at[J, J].add(w_ij)

        # 6) Scatter the off‐diagonal subtractions:
        mat = mat.at[I, J].add(-w_ij)
        mat = mat.at[J, I].add(-w_ij)
    else:
        np.add.at(mat, np.diag_indices(S + 1), diag_updates_i)

        xp.add.at(mat, (I, I), w_ij)
        xp.add.at(mat, (J, J), w_ij)

        np.add.at(mat, (I, J), -w_ij)
        np.add.at(mat, (J, I), -w_ij)

    # 7) Drop the extra row/column S and return the S×S result
    return mat[:S, :S]


def weighted_regularization_matrix_single_scatter_from(
    regularization_weights: np.ndarray,
    neighbors: np.ndarray,
    xp=np,
) -> np.ndarray:
    """
    Returns the regularization matrix of the adaptive regularization scheme, scattering every mesh edge
    **once** so that uniform weights reproduce ``Constant`` regularization exactly.

    This is the corrected sibling of ``weighted_regularization_matrix_from``, used by the ``*Power``
    classes (e.g. ``AdaptPower``). The two differ only in how often each mesh edge is scattered:

    - ``weighted_regularization_matrix_from`` adds every ordered neighbor pair to **both** the
      ``(i, j)`` and ``(j, i)`` entries. Because the neighbor list already holds each unordered edge
      twice (once in row ``i``, once in row ``j``), every edge lands four times, which is exactly twice
      what ``constant_regularization_matrix_from`` does. ``Adapt(inner=outer=c)`` is therefore
      ``2 x`` ``Constant(c)``, not equal to it.
    - This function adds every ordered neighbor pair once, to ``(i, i)`` and ``(i, j)`` only -- the same
      bookkeeping ``Constant`` uses -- so ``AdaptPower(inner=outer=c)`` equals ``Constant(c)`` exactly.

    The edge weight is the mean of the two endpoints' squared regularization weights,
    ``0.5 * (w_i ** 2 + w_j ** 2)``. This is symmetric in ``(i, j)`` (so the matrix is symmetric even
    for wildly varying adaptive weights), reduces to ``w ** 2`` when the weights are uniform, and makes
    the result a weighted graph Laplacian plus a ``1e-8`` diagonal floor -- so its rows sum to the floor
    and it is positive semi-definite by construction.

    The legacy builder is left untouched: halving it in place would silently change the effective
    regularization of every ``Adapt`` fit ever run.

    Parameters
    ----------
    regularization_weights
        The regularization weight of each pixel, adaptively governing the degree of gradient regularization
        applied to each inversion parameter (e.g. mesh pixels of a ``Mapper``).
    neighbors
        An array of length (total_pixels) which provides the index of all neighbors of every pixel in
        the mesh grid (entries of -1 correspond to no neighbor).

    Returns
    -------
    np.ndarray
        The regularization matrix computed using an adaptive regularization scheme where the effective
        regularization coefficient of every source pixel is different.
    """
    S, P = neighbors.shape

    reg_w = regularization_weights**2

    # 1) Flatten the (i->j) neighbor pairs
    I = xp.repeat(xp.arange(S), P)  # (S*P,)
    J_raw = neighbors.reshape(-1)  # (S*P,)

    # 2) Remap "no neighbor" entries to an extra slot S, whose weight = 0
    OUT = S
    valid = J_raw >= 0
    J = xp.where(valid, J_raw, OUT)

    # 3) Build an extended weight vector with a zero at index S
    reg_w_ext = xp.concatenate([reg_w, xp.zeros((1,))], axis=0)

    # 4) Symmetric per-edge weight: the mean of the two endpoints' squared weights. Padded entries are
    #    masked to zero (their I endpoint is a real pixel, so the mean alone would not vanish).
    w_ij = 0.5 * (reg_w_ext[I] + reg_w_ext[J]) * valid

    # 5) Start with zeros on an (S+1)x(S+1) canvas so we can scatter into row S safely
    mat = xp.zeros((S + 1, S + 1), dtype=regularization_weights.dtype)

    diag_updates_i = xp.concatenate(
        [xp.full((S,), 1e-8), xp.zeros((1,))], axis=0  # out-of-bounds slot stays zero
    )

    # 6) Scatter each ordered pair exactly once: onto the diagonal of i and the (i, j) off-diagonal
    if xp.__name__.startswith("jax"):
        mat = mat.at[xp.diag_indices(S + 1)].add(diag_updates_i)
        mat = mat.at[I, I].add(w_ij)
        mat = mat.at[I, J].add(-w_ij)
    else:
        np.add.at(mat, np.diag_indices(S + 1), diag_updates_i)
        np.add.at(mat, (I, I), w_ij)
        np.add.at(mat, (I, J), -w_ij)

    # 7) Drop the extra row/column S and return the SxS result
    return mat[:S, :S]


class Adapt(AbstractRegularization):
    def __init__(
        self,
        inner_coefficient: float = 1.0,
        outer_coefficient: float = 1.0,
        signal_scale: float = 1.0,
    ):
        """
        Regularization which uses the neighbors of the mesh (e.g. shared Delaunay vertexes) and values adaptred to the
        data being fitted to smooth an inversion's solution.

        For the weighted regularization scheme, each pixel is given an 'effective regularization weight', which is
        applied when each set of pixel neighbors are regularized with one another. The motivation of this is that
        different regions of a pixelization's mesh require different levels of regularization (e.g., high smoothing where the
        no signal is present and less smoothing where it is, see (Nightingale, Dye and Massey 2018)).

        Unlike ``Constant`` regularization, neighboring pixels must now be regularized with one another
        in both directions (e.g. if pixel 0 regularizes pixel 1, pixel 1 must also regularize pixel 0). For example:

        B = [-1, 1]  [0->1]
            [-1, -1]  1 now also regularizes 0

        For ``Constant`` regularization this would NOT produce a positive-definite matrix. However, for
        the weighted scheme, it does!

        The regularize weight_list change the B matrix as shown below - we simply multiply each pixel's effective
        regularization weight by each row of B it has a -1 in, so:

        regularization_weights = [1, 2, 3, 4]

        B = [-1, 1, 0 ,0] # [0->1]
            [0, -2, 2 ,0] # [1->2]
            [0, 0, -3 ,3] # [2->3]
            [4, 0, 0 ,-4] # [3->0]

        If our -1's werent down the diagonal this would look like:

        B = [4, 0, 0 ,-4] # [3->0]
            [0, -2, 2 ,0] # [1->2]
            [-1, 1, 0 ,0] # [0->1]
            [0, 0, -3 ,3] # [2->3] This is valid!

        A full description of regularization and this matrix can be found in the parent `AbstractRegularization` class.

        **JAX & gradient support** (2026-07 gradient sweep): as for
        ``Constant`` — JAX-differentiable and FD-certified on the rectangular
        mesh family (this is the rectangular production scheme), but raises
        ``TracerArrayConversionError`` on the Delaunay mesh family, whose
        neighbors come from a direct scipy call on the traced mesh grid (use
        ``AdaptSplit`` there). Note the defaults
        ``inner_coefficient == outer_coefficient == 1.0`` make the weighting
        uniform — but *not* numerically identical to ``Constant(coefficient=1.0)``;
        see the coefficient-convention note below.

        **Coefficient convention (legacy, ``lambda^4``).** The coefficients are squared twice before they
        reach the regularization matrix -- once by ``adapt_regularization_weights_from`` and once by the
        matrix builder -- so the matrix scales as the *fourth* power of the coefficient, while
        ``Constant`` scales as the second. Both carry the same ``LogUniform(1e-6, 1e6)`` prior, so this
        scheme explores a far wider effective smoothing range and reaches a numerically non
        positive-definite matrix from ``c ~ 1e4`` where ``Constant`` survives to ``c ~ 1e6``.

        **It is also 2x ``Constant``, not equal to it.** The matrix builder scatters every mesh edge in
        both directions, and the neighbor list already holds each unordered edge twice, so each edge
        lands four times where ``Constant`` lands it twice.
        ``Adapt(inner_coefficient=1.0, outer_coefficient=1.0)`` is therefore exactly ``2 x``
        ``Constant(coefficient=1.0)``.

        This behaviour is preserved deliberately: changing it would alter the coefficient scale of every
        adaptive fit ever run. **New work should prefer ``AdaptPower``**, which takes a ``power`` argument
        (default ``1.0``, giving the ``Constant``-matching ``lambda^2`` convention) and scatters each edge
        once, so ``AdaptPower(inner=outer=c)`` equals ``Constant(c)`` exactly and is more robust to
        gradient / NaN pathologies. The migration is ``c_new = c_old ** 2``, and
        ``AdaptPower(power=2.0)`` reproduces this class's coefficient scaling exactly.

        Parameters
        ----------
        coefficients
            The regularization coefficients which controls the degree of smoothing of the inversion reconstruction in
            high and low signal regions of the reconstruction.
        signal_scale
            A factor which controls how rapidly the smoothness of regularization varies from high signal regions to
            low signal regions.
        """

        super().__init__()

        validate_coefficient(coefficient=inner_coefficient, name="inner_coefficient")
        self.inner_coefficient = inner_coefficient
        validate_coefficient(coefficient=outer_coefficient, name="outer_coefficient")
        self.outer_coefficient = outer_coefficient
        self.signal_scale = signal_scale

    def regularization_weights_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization weights of this regularization scheme.

        The regularization weights define the level of regularization applied to each parameter in the linear object
        (e.g. the ``pixels`` in a ``Mapper``).

        For standard regularization (e.g. ``Constant``) are weights are equal, however for adaptive schemes
        (e.g. ``Adapt``) they vary to adapt to the data being reconstructed.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses these weights when performing regularization.

        Returns
        -------
        The regularization weights.
        """
        pixel_signals = linear_obj.pixel_signals_from(
            signal_scale=self.signal_scale, xp=xp
        )

        return adapt_regularization_weights_from(
            inner_coefficient=self.inner_coefficient,
            outer_coefficient=self.outer_coefficient,
            pixel_signals=pixel_signals,
        )

    def regularization_matrix_from(self, linear_obj: LinearObj, xp=np) -> np.ndarray:
        """
        Returns the regularization matrix with shape [pixels, pixels].

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) which uses this matrix to perform regularization.

        Returns
        -------
        The regularization matrix.
        """
        regularization_weights = self.regularization_weights_from(
            linear_obj=linear_obj, xp=xp
        )

        return weighted_regularization_matrix_from(
            regularization_weights=regularization_weights,
            neighbors=linear_obj.mesh_geometry.neighbors,
            xp=xp,
        )
