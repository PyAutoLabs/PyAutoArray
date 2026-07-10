from autoarray.preloads.abstract import AbstractPreloads


class PreloadsImaging(AbstractPreloads):
    def __init__(self, source_plane_mesh_grid=None, image_plane_mesh_grid=None):
        """
        Preloaded quantities for an imaging fit / inversion (see `AbstractPreloads`).

        This is the consumer-facing preloads container for imaging data — for example the object
        returned by `AnalysisImaging.shared_state_from` for the multi-exposure shared-state path,
        where exposures of the same lens (at the same or different wavelengths, with per-exposure
        pixel offsets) share one source-plane mesh.

        The invariance contract differs from the datacube (`PreloadsInterferometer`) case, whose
        channels share a single real-space grid and can therefore preload the whole mapper and
        curvature matrix. Imaging exposures have per-exposure PSFs and pixel offsets, so:

        - the source-plane mesh geometry IS shareable (a pure function of the shared lens model and
          the lead exposure's image-mesh) — preloaded here;
        - the mapper, mapping matrix, blurred mapping matrix and curvature matrix are NOT (each
          exposure maps its own offset grid onto the shared mesh and blurs with its own PSF);
        - the regularization matrix is NOT (regularization may adapt to per-exposure data).

        Parameters
        ----------
        source_plane_mesh_grid
            The shared source-plane mesh geometry, traced once from the lead exposure. See
            `AbstractPreloads`.
        image_plane_mesh_grid
            Its image-plane counterpart in the lead exposure's frame. See `AbstractPreloads`.
        """
        super().__init__(
            source_plane_mesh_grid=source_plane_mesh_grid,
            image_plane_mesh_grid=image_plane_mesh_grid,
        )
