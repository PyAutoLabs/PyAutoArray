from __future__ import annotations
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from autoarray.inversion.linear_obj.linear_obj import LinearObj


class AbstractRegularization:
    def __init__(self):
        """
        Abstract base class for a regularization-scheme, which is applied to a pixelization to enforce a \
        smooth-source solution and prevent over-fitting. This is achieved by computing a \
        'regularization term' - which is the sum of differences in reconstructed flux between every set of neighboring \
        pixels. This regularization term is added to the solution's chi-squared as a penalty term. This effects \
        a pixelization in the following ways:

        1) The regularization matrix (see below) is added to the curvature matrix used by the inversion to \
           linearly invert and fit the data. Thus, it changes the pixelization in a linear manner, ensuring that \
           the minimum chi-squared solution is achieved accounting for the penalty term.

        2) The log likelihood of the pixelization's fit to the dataset changes from L = -0.5 *(chi^2 + noise_normalization) \
           to L = -0.5 (chi^2 + coefficients * regularization_term + noise_normalization). The regularization \
           coefficient is a parameter which determines how strongly we smooth the pixelization's reconstruction.

        The value of the coefficients(s) is set using the Bayesian framework of (Suyu 2006) and this \
        is described further in the (*inversion.inversion* class).

        The regularization matrix, H, is calculated by defining a set of B matrices which describe how the \
        pixels neighbor one another. For example, lets take a 3x3 square grid:
        ______
        I0I1I2I
        I3I4I5I
        I6I7I8I
        ^^^^^^^

        We want to regularize this grid such that each pixel is regularized with the pixel to its right and below it \
        (provided there are pixels in that direction). This means that:

        - pixel 0 is regularized with pixel 1 (to the right) and pixel 3 (below).
        - pixel 1 is regularized with pixel 2 (to the right) and pixel 4 (below),
        - Pixel 2 is only regularized with pixel 5, as there is no pixel to its right.
        - and so on.

        We make two 9 x 9 B matrices, which describe regularization in each direction (i.e. rightwards and downwards). \
        We simply put a -1 and 1 in each row of a pixel index where it has a neighbor, where the value 1 goes in the \
        column of its neighbor's index. Thus, the B matrix describing neighboring pixels to their right looks like:

        B_x = [-1,  1,  0,  0,  0,  0,  0,  0,  0] # [0->1]
              [ 0, -1,  1,  0,  0,  0,  0,  0,  0] # [1->2]
              [ 0,  0, -1,  0,  0,  0,  0,  0,  0] # [] NOTE - no pixel neighbor.
              [ 0,  0,  0, -1,  1,  0,  0,  0,  0] # [3->4]
              [ 0,  0,  0,  0, -1,  1,  0,  0,  0] # [4->5]
              [ 0,  0,  0,  0,  0, -1,  0,  0,  0] # [] NOTE - no pixel neighbor.
              [ 0,  0,  0,  0,  0,  0, -1,  1,  0] # [6->7]
              [ 0,  0,  0,  0,  0,  0,  0, -1,  1] # [7->8]
              [ 0,  0,  0,  0,  0,  0,  0,  0, -1] # [] NOTE - no pixel neighbor.

        We now make another B matrix for the regularization downwards:

        B_y = [-1,  0,  0,  1,  0,  0,  0,  0,  0] # [0->3]
              [ 0, -1,  0,  0,  1,  0,  0,  0,  0] # [1->4]
              [ 0,  0, -1,  0,  0,  1,  0,  0,  0] # [2->5]
              [ 0,  0,  0, -1,  0,  0,  1,  0,  0] # [3->6]
              [ 0,  0,  0,  0, -1,  0,  0,  1,  0] # [4->7]
              [ 0,  0,  0,  0,  0, -1,  0,  0,  1] # [5->8]
              [ 0,  0,  0,  0,  0,  0, -1,  0,  0] # [] NOTE - no pixel neighbor.
              [ 0,  0,  0,  0,  0,  0,  0, -1,  0] # [] NOTE - no pixel neighbor.
              [ 0,  0,  0,  0,  0,  0,  0,  0, -1] # [] NOTE - no pixel neighbor.

        After making the B matrices that represent our pixel neighbors, we can compute the regularization matrix, H, \
        of each direction as H = B * B.T (matrix multiplication).

        E.g.

        H_x = B_x.T, * B_x
        H_y = B_y.T * B_y
        H = H_x + H_y

        Whilst the example above used a square-grid with regularization to the right and downwards, this matrix \
        formalism can be extended to describe regularization in more directions (e.g. upwards, to the left).

        It can also describe irregular meshes, e.g. a Delaunay mesh, where a B matrix is \
        computed for every shared Delaunay vertex of each Delaunay pixel. The number of B matrices is now equal to the \
        number of Delaunay vertices in the pixel with the most Delaunay vertices. However, we describe below a scheme to \
        compute this solution more efficiently.

        ### COMBINING B MATRICES ###

        The B matrices above each had the -1's going down the diagonam. This is not necessary, and it is valid to put \
        each pixel pairing anywhere. So, if we had a 4x4 B matrix, where:

        - pixel 0 regularizes with pixel 1
        - pixel 2 regularizes with pixel 3
        - pixel 3 regularizes with pixel 0

        We can still set this up as one matrix (even though the pixel 0 comes up twice):

        B = [-1, 1,  0 , 0] # [0->1]
            [ 0, 0,  0 , 0] # We can skip rows by making them all zeros.
            [ 0, 0, -1 , 1] # [2->3]
            [ 1, 0,  0 ,-1] # [3->0] This is valid!

        So, for a Delaunay pixelzation, we don't have to make the same number of B matrices as Delaunay vertices,  \
        we can combine them into fewer B matrices as above.

        # SKIPPING THE B MATRIX CALCULATION #

        Infact, going through the rigmarole of computing and multiplying B matrices like this is uncessary. It is \
        more computationally efficiently to directly compute H. This is possible, provided you know know all of the \
        neighboring pixel pairs (which, by definition, you need to know to set up the B matrices anyway). Thus, the \
       'regularization_matrix_via_neighbors_from' functions in this module directly compute H from the pixel \
        neighbors.

        # POSITIVE DEFINITE MATRIX #

        The regularization matrix must be positive-definite, as the Bayesian framework of Suyu 2006 requires that we \
        use its determinant in the calculation.

        Parameters
        ----------
        shape
            The dimensions of the rectangular grid of pixels (x_pixels, y_pixel)
        coefficients : (float,)
            The regularization_matrix coefficients used to smooth the pix reconstructed_inversion_image.

        """

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ is other.__class__

    def __hash__(self):
        return id(self)

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
        raise NotImplementedError

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
        raise NotImplementedError

    def log_det_regularization_matrix_term_from(
        self, linear_obj: LinearObj, xp=np
    ) -> Optional[float]:
        """
        Returns ``log det H`` of this scheme's regularization matrix computed from a
        factorization the scheme itself knows about, or ``None`` when no such shortcut
        exists (the default).

        The kernel regularization schemes build ``H = coefficient * C^-1`` from a dense
        covariance ``C`` and can therefore return the analytically exact
        ``pixels * log(coefficient) - log det C`` from a single Cholesky of ``C`` —
        avoiding the explicit inverse, whose round-off (amplified by ``cond(C)``, which
        reaches ~1e9 on clustered traced mesh vertices) otherwise leaks into the
        evidence. ``C``'s conditioning does not depend on the regularization
        coefficient, so this path is also finite where factorizing the formed ``H``
        fails at extreme coefficients.

        The inversion consumes this ONLY when ``Settings.log_det_method == "slogdet"``
        (the opt-in gradient-safe log-det, PyAutoArray#391) — the default
        ``"cholesky"`` evidence path never calls it, so default likelihood values are
        unchanged. See ``AbstractInversion.log_det_regularization_matrix_term``.

        Parameters
        ----------
        linear_obj
            The linear object (e.g. a ``Mapper``) whose regularization matrix the
            log-determinant is of.

        Returns
        -------
        The log determinant of the regularization matrix, or ``None`` when this scheme
        has no factorization-aware shortcut.
        """
        return None
