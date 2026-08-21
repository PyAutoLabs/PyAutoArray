import numpy as np

from autonerves import cached_property

from autoarray.inversion.mesh.interpolator.abstract import AbstractInterpolator
from autoarray.inversion.regularization.regularization_util import (
    split_points_from,
)


def scipy_delaunay(points_np, query_points_np, areas_factor):
    """Compute Delaunay simplices (simplices_padded) and Voronoi areas in one call."""
    from scipy.spatial import Delaunay

    max_simplices = 2 * points_np.shape[0]

    # --- Delaunay mesh using source plane data grid ---
    tri = Delaunay(points_np)

    points = tri.points.astype(points_np.dtype)
    simplices = tri.simplices.astype(np.int32)

    # Pad simplices to max_simplices
    simplices_padded = -np.ones((max_simplices, 3), dtype=np.int32)
    simplices_padded[: simplices.shape[0]] = simplices

    # ---------- find_simplex for source plane data grid ----------
    simplex_idx = tri.find_simplex(query_points_np).astype(np.int32)  # (Q,)

    mappings = pix_indexes_for_sub_slim_index_delaunay_from(
        data_grid=query_points_np,
        simplex_index_for_sub_slim_index=simplex_idx,
        pix_indexes_for_simplex_index=simplices,
        delaunay_points=points_np,
    )

    # ---------- Barycentric Areas used to weight split points ----------

    areas = barycentric_dual_area_from(
        points,
        simplices,
        xp=np,
    )

    split_point_areas = areas_factor * np.sqrt(areas)

    # ---------- Compute split cross points for Split regularization ----------
    split_points = split_points_from(
        points=points_np,
        area_weights=split_point_areas,
    )

    # ---------- find_simplex for split cross points ----------
    split_points_idx = tri.find_simplex(split_points)

    splitted_mappings = pix_indexes_for_sub_slim_index_delaunay_from(
        data_grid=split_points,
        simplex_index_for_sub_slim_index=split_points_idx,
        pix_indexes_for_simplex_index=simplices,
        delaunay_points=points_np,
    )

    return points, simplices_padded, mappings, split_points, splitted_mappings


# Query points are located in chunks of this size so the (chunk, N) distance
# intermediate stays bounded on GPU when the likelihood is vmapped over many
# live points.
DELAUNAY_LOCATE_CHUNK = 1024

# Iteration cap of the visibility walk. On production Hilbert/Delaunay meshes
# every point resolves within ~64 steps from a nearest-vertex start; a query
# still unresolved at the cap (a would-be fp cycle on degenerate slivers)
# falls back to its nearest-vertex mapping, the outside-hull convention.
DELAUNAY_WALK_STEPS = 128


def scipy_delaunay_tri_only(points_np):
    """Minimal host-side Delaunay callback: the qhull triangulation plus the
    adjacency arrays the JAX-side visibility-walk point locator needs.

    Unlike ``scipy_delaunay`` this does NOT run ``find_simplex`` or assemble
    mappings — that work is fixed-shape array math once the simplices are
    known, and runs inside the JIT program (on GPU, batched across vmap
    lanes) instead of serializing per lane on the host.

    Returns
    -------
    simplices_padded : (2N, 3) int32, -1 padded (same convention as
        ``scipy_delaunay``).
    simplex_neighbors : (2N, 3) int32, -1 padded — qhull's ``tri.neighbors``:
        entry k of a row is the simplex opposite that row's vertex k, with
        -1 at the convex hull.
    vertex_simplex : (N,) int32 — one simplex incident to each vertex (the
        walk's start simplex); -1 for the rare vertex qhull excluded as
        coplanar/duplicate (the walk then starts from simplex 0 instead,
        which is equally valid — the walk converges from any start).
    """
    N = points_np.shape[0]
    simplices_padded = -np.ones((2 * N, 3), dtype=np.int32)
    simplex_neighbors = -np.ones((2 * N, 3), dtype=np.int32)
    vertex_simplex = -np.ones(N, dtype=np.int32)

    # This function is the body of a sequentially-vmapped host callback.  A
    # qhull exception in one lane aborts the entire batch, so keep non-finite
    # input lane-local and encode it with the callback's existing padding
    # convention.  The JAX-side walk and weights preserve the lane as NaN.
    if not np.isfinite(points_np).all():
        return simplices_padded, simplex_neighbors, vertex_simplex

    from scipy.spatial import Delaunay

    tri = Delaunay(points_np)
    simplices = tri.simplices.astype(np.int32)  # (T, 3)
    T = simplices.shape[0]

    simplices_padded[:T] = simplices
    simplex_neighbors[:T] = tri.neighbors.astype(np.int32)

    simplex_ids = np.arange(T, dtype=np.int32)
    for k in range(3):
        vertex_simplex[simplices[:, k]] = simplex_ids

    return simplices_padded, simplex_neighbors, vertex_simplex


def _jax_delaunay_tables(points):
    """Run the qhull-only callback with fixed output shapes.

    The callback input is wrapped in ``stop_gradient`` so ``jax.grad`` /
    ``jax.value_and_grad`` flow through the Delaunay likelihood
    (``pure_callback`` has no JVP rule and would otherwise raise). This is
    NOT an approximation: the callback returns only int32 connectivity
    tables, which are piecewise-constant in the vertex positions — their
    true derivative is exactly zero everywhere except the measure-zero
    re-wiring (triangle-flip) events. The barycentric interpolant is
    discontinuous there; the Sibson interpolant in ``DelaunayNN`` instead has
    matching limits across an ordinary flip. Every quantity
    with a non-zero derivative (point location via the visibility walk,
    barycentric weights, dual areas, split points) is computed in-graph
    from the traced ``points``, so the frozen-tables gradient is the exact
    almost-everywhere derivative. FD-certified 2026-07-26
    (autolens_workspace_test ``scripts/imaging/jax_grad/delaunay.py``).
    """
    import jax
    import jax.numpy as jnp

    N = points.shape[0]
    return jax.pure_callback(
        lambda pts: scipy_delaunay_tri_only(np.asarray(pts)),
        (
            jax.ShapeDtypeStruct((2 * N, 3), jnp.int32),
            jax.ShapeDtypeStruct((2 * N, 3), jnp.int32),
            jax.ShapeDtypeStruct((N,), jnp.int32),
        ),
        jax.lax.stop_gradient(points),
        vmap_method="sequential",
    )


def pix_indexes_delaunay_walk_from(
    query_points,
    points,
    simplices_padded,
    simplex_neighbors,
    vertex_simplex,
    xp=np,
    return_simplex_indexes=False,
):
    """JAX/NumPy point location replacing ``scipy.spatial.Delaunay.find_simplex``
    on the JAX likelihood path, via the same visibility-walk algorithm
    ``find_simplex`` itself uses — so the result is exact, not approximate.

    For each query point: find the nearest mesh vertex (brute-force distance
    argmin), start from a simplex incident to it, and walk: compute the
    signed barycentric weights of the query in the current simplex; if all
    are >= -1e-12 the simplex contains the point (done); otherwise step to
    the neighbor simplex opposite the most-negative vertex. Crossing a hull
    edge (neighbor -1) means the point is outside the triangulation and it
    falls back to the nearest-vertex mapping ``[v, -1, -1]`` — the identical
    convention ``scipy_delaunay`` applies via its KDTree.

    The walk is bounded at DELAUNAY_WALK_STEPS (production meshes resolve in
    <= ~64); the loop runs in lockstep over a chunk of queries with done/
    outside masks, so it is fixed-shape and JIT/vmap-safe. Chunking over
    query points bounds the (chunk, N) nearest-vertex intermediate under
    vmap (JAX path; the NumPy path — used by the unit tests — processes the
    whole array with early exit). Returns a (Q, 3) int32 mapping array with
    the same semantics as ``pix_indexes_for_sub_slim_index_delaunay_from``.

    When ``return_simplex_indexes`` is true, also return the containing
    simplex index for every query (or -1 outside the convex hull).  Sibson
    interpolation uses this as the seed of its circumcircle-cavity walk, so
    point location is not repeated.
    """

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    def weights_of(cur, q_chunk):
        verts = simplices_padded[cur]  # (chunk, 3)
        valid_simplex = (verts >= 0).all(axis=1)
        safe_verts = verts.clip(min=0)
        a = points[safe_verts[:, 0]]
        b = points[safe_verts[:, 1]]
        c = points[safe_verts[:, 2]]
        den = cross(b - a, c - a)
        den = xp.where(den != 0.0, den, 1.0)
        w = (
            xp.stack(
                [
                    cross(b - q_chunk, c - q_chunk),
                    cross(c - q_chunk, a - q_chunk),
                    cross(a - q_chunk, b - q_chunk),
                ],
                axis=1,
            )
            / den[:, None]
        )
        return verts, w, valid_simplex

    def walk_step(carry, q_chunk):
        cur, done, outside = carry
        _, w, valid_simplex = weights_of(cur, q_chunk)
        minw = w.min(axis=1)
        opposite = w.argmin(axis=1)
        outside = outside | (~done & ~valid_simplex)
        done = done | (~outside & (minw >= -1.0e-12))
        nxt = simplex_neighbors[cur, opposite]
        outside = outside | (~done & (nxt < 0))
        move = ~done & ~outside
        cur = xp.where(move, nxt.clip(min=0), cur)
        return cur, done, outside

    def locate_chunk(q_chunk):
        # nearest mesh vertex: (chunk, N) distance intermediate
        d2 = ((q_chunk[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        seed = xp.argmin(d2, axis=1).astype(xp.int32)

        cur = vertex_simplex[seed].clip(min=0).astype(xp.int32)
        done = xp.zeros(q_chunk.shape[0], dtype=bool)
        outside = xp.zeros(q_chunk.shape[0], dtype=bool)

        if xp is np:
            for _ in range(DELAUNAY_WALK_STEPS):
                cur, done, outside = walk_step((cur, done, outside), q_chunk)
                if (done | outside).all():
                    break
        else:
            import jax

            cur, done, outside = jax.lax.fori_loop(
                0,
                DELAUNAY_WALK_STEPS,
                lambda _, carry: walk_step(carry, q_chunk),
                (cur, done, outside),
            )

        verts = simplices_padded[cur]
        fallback = xp.stack([seed, -xp.ones_like(seed), -xp.ones_like(seed)], axis=1)
        mappings = xp.where(done[:, None], verts, fallback).astype(xp.int32)
        simplex_indexes = xp.where(done, cur, -1).astype(xp.int32)
        return mappings, simplex_indexes

    if xp is np:
        mappings, simplex_indexes = locate_chunk(query_points)
        if return_simplex_indexes:
            return mappings, simplex_indexes
        return mappings

    import jax

    Q = query_points.shape[0]
    chunk = DELAUNAY_LOCATE_CHUNK
    pad = (-Q) % chunk
    # pad rows sit far outside the mesh; their located rows are sliced away
    q_padded = xp.concatenate(
        [query_points, xp.full((pad, 2), 1.0e9, dtype=query_points.dtype)]
    )
    mappings, simplex_indexes = jax.lax.map(
        locate_chunk, q_padded.reshape(-1, chunk, 2)
    )
    mappings = mappings.reshape(-1, 3)[:Q]
    simplex_indexes = simplex_indexes.reshape(-1)[:Q]
    if return_simplex_indexes:
        return mappings, simplex_indexes
    return mappings


def jax_delaunay(points, query_points, areas_factor=0.5):
    """JAX-path Delaunay construction. Only the qhull triangulation runs on
    the host (via ``pure_callback``); point location, dual areas and split
    points run inside the JIT program. The return contract (values, shapes,
    dtypes and -1 padding conventions) is identical to ``scipy_delaunay``.
    """
    import jax.numpy as jnp

    simplices_padded, simplex_neighbors, vertex_simplex = _jax_delaunay_tables(points)

    mappings = pix_indexes_delaunay_walk_from(
        query_points=query_points,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        vertex_simplex=vertex_simplex,
        xp=jnp,
    )

    # dual areas via masked scatter-add over the padded simplices
    valid = simplices_padded[:, 0] >= 0
    s = simplices_padded.clip(min=0)
    p0, p1, p2 = points[s[:, 0]], points[s[:, 1]], points[s[:, 2]]
    tri_cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    contrib = jnp.where(valid, 0.5 * jnp.abs(tri_cross) / 3.0, 0.0)
    areas = jnp.zeros(points.shape[0], dtype=points.dtype)
    for k in range(3):
        areas = areas.at[s[:, k]].add(contrib)

    split_points = split_points_from(
        points=points,
        area_weights=areas_factor * jnp.sqrt(areas),
        xp=jnp,
    )

    # Split points are seeded at their own nearest vertex (not their parent
    # vertex): a split point can land outside the hull, and the fallback must
    # then match scipy_delaunay's KDTree nearest-vertex assignment.
    splitted_mappings = pix_indexes_delaunay_walk_from(
        query_points=split_points,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        vertex_simplex=vertex_simplex,
        xp=jnp,
    )

    return points, simplices_padded, mappings, split_points, splitted_mappings


def barycentric_dual_area_from(
    mesh_grid,  # (N_pix, 2) vertex positions
    simplices,  # (N_tri, 3) triangle vertex indices
    xp=np,  # xp = np or jnp
):
    """
    Compute barycentric dual area for each vertex in a Delaunay triangulation.

    Dual area A_i = sum over triangles containing vertex i of (triangle_area / 3).

    Parameters
    ----------
    mesh_grid : (N_pix, 2)
        Coordinates of all mesh vertices.
    simplices : (N_tri, 3)
        Vertex indices for each triangle.
    xp : module
        numpy or jax.numpy

    Returns
    -------
    dual_area : (N_pix,)
        Barycentric dual area for each vertex.
    """

    # -------------------------------
    # gather triangle vertices
    # -------------------------------
    p0 = mesh_grid[simplices[:, 0]]  # (N_tri, 2)
    p1 = mesh_grid[simplices[:, 1]]
    p2 = mesh_grid[simplices[:, 2]]

    # -------------------------------
    # triangle areas
    # -------------------------------
    # parallelogram area = |(p1 - p0) × (p2 - p0)|
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
        p2[:, 0] - p0[:, 0]
    )

    tri_area = 0.5 * xp.abs(cross)  # (N_tri,)

    # each triangle contributes area/3 to 3 vertices
    contrib = tri_area / 3.0

    # -------------------------------
    # scatter-add into dual area array
    # -------------------------------
    N_pix = mesh_grid.shape[0]
    dual_area = xp.zeros(N_pix)

    # xp.add.at works for np and jnp
    for k in range(3):
        xp.add.at(dual_area, simplices[:, k], contrib)

    return dual_area


def pix_indexes_for_sub_slim_index_delaunay_from(
    data_grid,  # (N_sub, 2)
    simplex_index_for_sub_slim_index,  # (N_sub,)
    pix_indexes_for_simplex_index,  # (M, 3)
    delaunay_points,  # (N_pix, 2)
):

    N_sub = data_grid.shape[0]

    inside_mask = simplex_index_for_sub_slim_index >= 0
    outside_mask = ~inside_mask

    # ---------------------------
    # Preallocate output
    # ---------------------------
    out = np.full((N_sub, 3), -1, dtype=np.int32)

    # ---------------------------
    # Case 1: Inside simplex (fast gather)
    # ---------------------------
    if inside_mask.any():
        out[inside_mask] = pix_indexes_for_simplex_index[
            simplex_index_for_sub_slim_index[inside_mask]
        ]

    # ---------------------------
    # Case 2: Outside → KDTree NN
    # ---------------------------
    if outside_mask.any():
        from scipy.spatial import cKDTree

        tree = cKDTree(delaunay_points)
        _, idx = tree.query(data_grid[outside_mask], k=1)
        out[outside_mask, 0] = idx.astype(np.int32)

    out = out.astype(np.int32)

    return out


def scipy_delaunay_matern(points_np, query_points_np):
    """
    Minimal SciPy Delaunay callback for Matérn regularization.

    Returns only what’s needed for mapping:
      - points (tri.points)
      - simplices_padded
      - mappings: integer array of pixel indices for each query point,
        typically of shape (Q, 3), where each row gives the indices of the
        Delaunay mesh vertices ("pixels") associated with that query point.
    """
    from scipy.spatial import Delaunay

    max_simplices = 2 * points_np.shape[0]

    # --- Delaunay mesh ---
    tri = Delaunay(points_np)

    points = tri.points.astype(points_np.dtype)
    simplices = tri.simplices.astype(np.int32)

    # --- Pad simplices to fixed shape for JAX ---
    simplices_padded = -np.ones((max_simplices, 3), dtype=np.int32)
    simplices_padded[: simplices.shape[0]] = simplices

    # --- find_simplex for query points ---
    simplex_idx = tri.find_simplex(query_points_np).astype(np.int32)  # (Q,)

    mappings = pix_indexes_for_sub_slim_index_delaunay_from(
        data_grid=query_points_np,
        simplex_index_for_sub_slim_index=simplex_idx,
        pix_indexes_for_simplex_index=simplices,
        delaunay_points=points_np,
    )

    return points, simplices_padded, mappings


def jax_delaunay_matern(points, query_points):
    """JAX-path Matérn variant: qhull-only callback + JAX point location,
    returning the same minimal (points, simplices_padded, mappings) contract
    as ``scipy_delaunay_matern``."""
    import jax.numpy as jnp

    simplices_padded, simplex_neighbors, vertex_simplex = _jax_delaunay_tables(points)

    mappings = pix_indexes_delaunay_walk_from(
        query_points=query_points,
        points=points,
        simplices_padded=simplices_padded,
        simplex_neighbors=simplex_neighbors,
        vertex_simplex=vertex_simplex,
        xp=jnp,
    )

    return points, simplices_padded, mappings


def triangle_area_xp(c0, c1, c2, xp):
    """
    Twice triangle area using vector cross product magnitude.
    Calling via xp ensures NumPy or JAX backend operation.
    """
    v0 = c1 - c0  # (..., 2)
    v1 = c2 - c0
    cross = v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]
    return xp.abs(cross)


def pixel_weights_delaunay_from(
    data_grid,  # (N_sub, 2)
    mesh_grid,  # (N_pix, 2)
    pix_indexes_for_sub_slim_index,  # (N_sub, 3), padded with -1
    xp=np,  # backend: np (default) or jnp
):
    """
    XP-compatible (NumPy/JAX) version of pixel_weights_delaunay_from.

    Computes barycentric weights for Delaunay triangle interpolation.
    """

    N_sub = pix_indexes_for_sub_slim_index.shape[0]

    # -----------------------------
    # CASE MASKS
    # -----------------------------
    # If pix_indexes_for_sub_slim_index[sub][1] == -1 → NOT in simplex
    has_simplex = pix_indexes_for_sub_slim_index[:, 1] != -1  # (N_sub,)

    # -----------------------------
    # GATHER TRIANGLE VERTICES
    # -----------------------------
    # Clip negatives (for padded entries) so that indexing doesn't crash
    safe_indices = pix_indexes_for_sub_slim_index.clip(min=0)

    # (N_sub, 3, 2)
    vertices = mesh_grid[safe_indices]

    p0 = vertices[:, 0]  # (N_sub, 2)
    p1 = vertices[:, 1]
    p2 = vertices[:, 2]

    # Query points
    q = data_grid  # (N_sub, 2)

    # -----------------------------
    # TRIANGLE AREAS (barycentric numerators)
    # -----------------------------
    a0 = triangle_area_xp(p1, p2, q, xp)
    a1 = triangle_area_xp(p0, p2, q, xp)
    a2 = triangle_area_xp(p0, p1, q, xp)

    area_sum = a0 + a1 + a2

    # (N_sub, 3)
    eps = xp.asarray(1e-12, dtype=area_sum.dtype)
    den = xp.where(xp.abs(area_sum) > eps, area_sum, 1.0)  # avoid 0 in denominator

    weights_bary = xp.stack([a0, a1, a2], axis=1) / den[:, None]

    # If degenerate, set weights to 0 (or something sensible)
    degenerate = xp.abs(area_sum) <= eps
    weights_bary = xp.where(degenerate[:, None], 0.0, weights_bary)

    # -----------------------------
    # NEAREST-NEIGHBOUR CASE
    # -----------------------------
    # For no-simplex: weight = [1,0,0]
    weights_nn = xp.stack(
        [
            xp.ones(N_sub),
            xp.zeros(N_sub),
            xp.zeros(N_sub),
        ],
        axis=1,
    )

    # -----------------------------
    # SELECT BETWEEN CASES
    # -----------------------------
    pixel_weights = xp.where(has_simplex[:, None], weights_bary, weights_nn)

    if xp is not np:
        mesh_is_finite = xp.isfinite(mesh_grid).all()
        pixel_weights = xp.where(mesh_is_finite, pixel_weights, xp.nan)

    return pixel_weights


class DelaunayInterface:

    def __init__(
        self, points, simplices, mappings, split_points, splitted_mappings, xp=np
    ):

        self.points = points
        self.simplices = simplices
        self.mappings = mappings
        self.split_points = split_points
        self.splitted_mappings = splitted_mappings

        self.xp = xp

    @cached_property
    def sizes(self):
        return self.xp.sum(self.mappings >= 0, axis=1).astype(np.int32)

    @cached_property
    def splitted_sizes(self):
        return self.xp.sum(self.splitted_mappings >= 0, axis=1).astype(np.int32)


class InterpolatorDelaunay(AbstractInterpolator):
    def __init__(
        self,
        mesh,
        mesh_grid,
        data_grid,
        adapt_data: np.ndarray = None,
        xp=np,
    ):
        """
        An irregular 2D grid of (y,x) coordinates which represents both a Delaunay triangulation and Voronoi mesh.

        The input irregular `2D` grid represents both of the following quantities:

        - The corners of the Delaunay triangulles used to construct a Delaunay triangulation.
        - The centers of a Voronoi pixels used to constract a Voronoi mesh.

        These reflect the closely related geometric properties of the Delaunay and Voronoi grids, whereby the corner
        points of Delaunay triangles by definition represent the centres of the corresponding Voronoi mesh.

        Different pixelizations, mappers and regularization schemes combine the Delaunay and Voronoi
        geometries in different ways to perform an Inversion. Thus, having all geometric methods contained in the
        single class here is necessary.

        The input `grid` of source pixel centres is ordered arbitrarily, given that there is no regular pattern
        for a Delaunay triangulation and Voronoi mesh's indexing to follow.

        This class is used in conjuction with the `inversion/pixelizations` package to create Voronoi meshs
        and mappers that perform an `Inversion`.

        Parameters
        ----------
        values
            The grid of (y,x) coordinates corresponding to the Delaunay triangle corners and Voronoi pixel centres.
        """

        super().__init__(
            mesh=mesh,
            mesh_grid=mesh_grid,
            data_grid=data_grid,
            adapt_data=adapt_data,
            xp=xp,
        )

    @cached_property
    def mesh_geometry(self):

        from autoarray.inversion.mesh.mesh_geometry.delaunay import (
            MeshGeometryDelaunay,
        )

        return MeshGeometryDelaunay(
            mesh=self.mesh,
            mesh_grid=self.mesh_grid,
            data_grid=self.data_grid,
            xp=self._xp,
        )

    @cached_property
    def mesh_grid_xy(self):
        """
        The default convention in `scipy.spatial` is to represent 2D coordinates as (x,y) pairs, whereas PyAutoArray
        represents 2D coordinates as (y,x) pairs.

        Therefore, this property simply converts the (y,x) grid of irregular coordinates into an (x,y) grid.
        """
        return self._xp.stack(
            [self.mesh_grid.array[:, 0], self.mesh_grid.array[:, 1]]
        ).T

    @cached_property
    def delaunay(self) -> "scipy.spatial.Delaunay":
        """
        Returns a `scipy.spatial.Delaunay` object from the 2D (y,x) grid of irregular coordinates, which correspond to
        the corner of every triangle of a Delaunay triangulation.

        This object contains numerous attributes describing a Delaunay triangulation. PyAutoArray uses the `ridge_points`
        attribute to determine the neighbors of every Voronoi pixel and the `vertices`, `regions` and `point_region`
        properties to determine the Voronoi pixel areas.

        There are numerous exceptions that `scipy.spatial.Voronoi` may raise when the input grid of coordinates used
        to compute the Voronoi mesh are ill posed. These exceptions are caught and combined into a single
        `MeshException`, which helps exception handling in the `inversion` package.
        """

        if not self.mesh.skip_areas:

            if self._xp.__name__.startswith("jax"):

                import jax.numpy as jnp

                points, simplices, mappings, split_points, splitted_mappings = (
                    jax_delaunay(
                        points=self.mesh_grid_xy,
                        query_points=self.data_grid.over_sampled.array,
                        areas_factor=self.mesh.areas_factor,
                    )
                )

            else:

                points, simplices, mappings, split_points, splitted_mappings = (
                    scipy_delaunay(
                        points_np=self.mesh_grid_xy,
                        query_points_np=self.data_grid.over_sampled.array,
                        areas_factor=self.mesh.areas_factor,
                    )
                )

        else:

            if self._xp.__name__.startswith("jax"):

                import jax.numpy as jnp

                points, simplices, mappings = jax_delaunay_matern(
                    points=self.mesh_grid_xy,
                    query_points=self.data_grid.over_sampled.array,
                )

            else:

                points, simplices, mappings = scipy_delaunay_matern(
                    points_np=self.mesh_grid_xy,
                    query_points_np=self.data_grid.over_sampled.array,
                )

            split_points = None
            splitted_mappings = None

        return DelaunayInterface(
            points=points,
            simplices=simplices,
            mappings=mappings,
            split_points=split_points,
            splitted_mappings=splitted_mappings,
            xp=self._xp,
        )

    @property
    def split_points(self) -> np.ndarray:
        """
        For every 2d (y,x) coordinate corresponding to a Voronoi pixel centre, this property splits them into a cross
        of four coordinates in the vertical and horizontal directions. The function therefore returns a irregular
        2D grid with four times the number of (y,x) coordinates.

        The distance between each centre and the 4 cross points is given by half the square root of its Voronoi
        pixel area.

        The reason for creating this grid is that the cross points allow one to estimate the gradient of the value of
        the Voronoi mesh, once the Voronoi pixels have values associated with them (e.g. after using the Voronoi
        mesh to fit data and perform an `Inversion`).

        The grid returned by this function is used by certain regularization schemes in the `Inversion` module to apply
        gradient regularization to an `Inversion` using a Delaunay triangulation or Voronoi mesh.
        """
        return self.delaunay.split_points

    @cached_property
    def _mappings_sizes_weights(self):
        """
        Computes the following three quantities describing the mappings between of every sub-pixel in the masked data
        and pixel in the `Delaunay` pixelization.

        - `pix_indexes_for_sub_slim_index`: the mapping of every data pixel (given its `sub_slim_index`)
        to pixelization pixels (given their `pix_indexes`).

        - `pix_sizes_for_sub_slim_index`: the number of mappings of every data pixel to pixelization pixels.

        - `pix_weights_for_sub_slim_index`: the interpolation weights of every data pixel's pixelization
        pixel mapping

        These are packaged into the attributes `mappings`, `sizes` and `weights`.

        The `sub_slim_index` refers to the masked data sub-pixels and `pix_indexes` the pixelization pixel indexes,
        for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the RectangularRTUAdaptDensity
        pixelization's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[2, 0] = 4`: The data's third (index 2) sub-pixel maps to the RectangularRTUAdaptDensity
        pixelization's fifth (index 4) pixel.

        The second dimension of the array `pix_indexes_for_sub_slim_index`, which is 0 in both examples above, is used
        for cases where a data pixel maps to more than one pixelization pixel.

        For a `Delaunay` pixelization each data pixel maps to 3 Delaunay triangles with interpolation, for example:

        - `pix_indexes_for_sub_slim_index[0, 0] = 2`: The data's first (index 0) sub-pixel maps to the Delaunay
        pixelization's third (index 2) pixel.

        - `pix_indexes_for_sub_slim_index[0, 1] = 5`: The data's first (index 0) sub-pixel also maps to the Delaunay
        pixelization's sixth (index 5) pixel.

        - `pix_indexes_for_sub_slim_index[0, 2] = 8`: The data's first (index 0) sub-pixel also maps to the Delaunay
        pixelization's ninth (index 8) pixel.

        The interpolation weights of these multiple mappings are stored in the array `pix_weights_for_sub_slim_index`.

        For the Delaunay pixelization these mappings are calculated using the Scipy spatial library
        (see `mapper_numba_util.pix_indexes_for_sub_slim_index_delaunay_from`).
        """
        mappings = self.delaunay.mappings.astype("int")
        sizes = self.delaunay.sizes.astype("int")

        weights = pixel_weights_delaunay_from(
            data_grid=self.data_grid.over_sampled.array,
            mesh_grid=self.mesh_grid.array,
            pix_indexes_for_sub_slim_index=mappings,
            xp=self._xp,
        )

        return mappings, sizes, weights

    @cached_property
    def _mappings_sizes_weights_split(self):
        """
        The property `_mappings_sizes_weightss` property describes the calculation of the mapping attributes, which contains
        numpy arrays describing how data-points and mapper pixels map to one another and the weights of these mappings.

        For certain regularization schemes (e.g. `ConstantSplit`, `AdaptSplit`) regularization uses
        mappings which are split in a cross configuration in order to factor in the derivative of the mapper
        reconstruction.

        This property returns a unique set of mapping values used for these regularization schemes which compute
        mappings and weights at each point on the split cross.
        """
        splitted_weights = pixel_weights_delaunay_from(
            data_grid=self.delaunay.split_points,
            mesh_grid=self.mesh_grid.array,
            pix_indexes_for_sub_slim_index=self.delaunay.splitted_mappings.astype(
                "int"
            ),
            xp=self._xp,
        )

        append_line_int = np.zeros((len(splitted_weights), 1), dtype="int") - 1
        append_line_float = np.zeros((len(splitted_weights), 1), dtype="float")

        mappings = self._xp.hstack(
            (self.delaunay.splitted_mappings.astype(self._xp.int32), append_line_int)
        )
        sizes = self.delaunay.splitted_sizes.astype(self._xp.int32)
        weights = self._xp.hstack((splitted_weights, append_line_float))

        return mappings, sizes, weights
