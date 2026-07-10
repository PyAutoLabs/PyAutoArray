class AbstractPreloads:
    def __init__(
        self,
        curvature_matrix=None,
        mapper_galaxy_dict=None,
        source_plane_mesh_grid=None,
        image_plane_mesh_grid=None,
    ):
        """
        Container for quantities that are *preloaded* into a fit / inversion: computed once and
        injected so that repeated evaluations reuse them instead of recomputing them.

        Preloading is an **advanced, opt-in** optimisation. A quantity may be preloaded only when it
        is genuinely invariant across the evaluations that reuse it; preloading a quantity that in
        fact changes silently corrupts the result. Establishing that invariance is the caller's
        responsibility — and the difficulty of knowing *when* a quantity is safe to preload is why
        preloads are **not applied as standard** (an earlier preload system was removed because this
        was bug-prone and hard to maintain).

        The natural, safe home for preloads is a **shared / combined likelihood**, where invariance
        is explicit and easy to verify:

        - In a datacube `FactorGraphModel` the lens model is identical for every spectral channel,
          so the channel-invariant inversion quantities are computed once (by
          `AnalysisInterferometer.shared_state_from`) and reused by every channel. It is obvious
          here exactly which quantities are invariant, which is what makes preloading safe.

        More general single-fit preloading (e.g. a source pixelization held fixed across a search)
        is possible but much harder to reason about, and is deliberately not the default.

        Every field is optional and defaults to `None`, so a caller preloads only the quantities it
        knows are invariant; each consumer reuses a field only when it is not `None`, otherwise
        falling back to the standard computation. New preloadable quantities are added as fields
        without changing the call signature threaded through the fit / inversion stack.

        This abstract base holds the quantities common to every dataset type. Dataset-specific
        subclasses (`PreloadsInterferometer`, and `PreloadsImaging` when needed) add the preloads
        unique to their formalism, mirroring the `AnalysisImaging` / `AnalysisInterferometer` split.

        Parameters
        ----------
        curvature_matrix
            The pre-computed curvature matrix `F = LᵀW̃L` — the dominant inversion-setup cost. When
            provided, the inversion returns it directly instead of rebuilding it.
        mapper_galaxy_dict
            The pre-computed mapping between each pixelization mapper and the source it reconstructs.
            Building a mapper is expensive (e.g. a Delaunay triangulation of the ray-traced source
            plane); when invariant across evaluations it is built once and reused here, so the
            `mapper` (and therefore the `mapping_matrix` and `regularization_matrix`) is not rebuilt.
            Stored opaquely — the consumer (e.g. PyAutoLens's `TracerToInversion`) populates and
            interprets it. Valid ONLY when the datasets sharing it have identical grids (e.g. the
            channels of a datacube) — a mapper embeds a dataset's own data-grid mappings.
        source_plane_mesh_grid
            The pre-computed source-plane mesh geometry (e.g. the ray-traced centres a Delaunay
            triangulation is built over), for consumers whose datasets share a lens model but NOT
            their grids (e.g. multi-exposure imaging with per-exposure pixel offsets and PSFs).
            Unlike `mapper_galaxy_dict`, this preloads only the mesh: each dataset still builds its
            own mapper by mapping its own (offset) data grid onto this shared mesh. Stored opaquely
            in the consumer's plane-grouped structure (PyAutoLens: the `traced_mesh_grid_pg_list`
            of the lead dataset). The mapping matrix, curvature matrix, blurred mapping matrix and
            regularization matrix are all deliberately NOT preloadable this way — offsets/PSFs make
            the first three per-dataset, and regularization may be data-adaptive.
        image_plane_mesh_grid
            The image-plane counterpart of `source_plane_mesh_grid` (the mesh centres before
            ray-tracing, in the lead dataset's frame), carried alongside it as metadata for
            downstream consumers (e.g. mapper plotting). Same opaque plane-grouped structure.
        """
        self.curvature_matrix = curvature_matrix
        self.mapper_galaxy_dict = mapper_galaxy_dict
        self.source_plane_mesh_grid = source_plane_mesh_grid
        self.image_plane_mesh_grid = image_plane_mesh_grid
