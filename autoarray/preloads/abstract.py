class AbstractPreloads:
    def __init__(self, curvature_matrix=None, mapper_galaxy_dict=None):
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
            interprets it.
        """
        self.curvature_matrix = curvature_matrix
        self.mapper_galaxy_dict = mapper_galaxy_dict
