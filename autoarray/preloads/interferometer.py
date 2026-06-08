from autoarray.preloads.abstract import AbstractPreloads


class PreloadsInterferometer(AbstractPreloads):
    def __init__(self, curvature_matrix=None):
        """
        Preloaded quantities for an interferometer fit / inversion (see `AbstractPreloads`).

        This is the consumer-facing preloads container for interferometer data — for example the
        object returned by `AnalysisInterferometer.shared_state_from` for the datacube shared-state
        path, where the channel-invariant inversion quantities are preloaded once and reused by
        every spectral channel.

        It currently carries only the universal `curvature_matrix` (`F`) from `AbstractPreloads`,
        but it is the home for interferometer-specific preloads as they are added (e.g. w-tilde /
        NUFFT translation-invariant quantities), and it may also carry non-inversion preloads used
        higher up at the fit level.

        Parameters
        ----------
        curvature_matrix
            The pre-computed curvature matrix `F = LᵀW̃L`. See `AbstractPreloads`.
        """
        super().__init__(curvature_matrix=curvature_matrix)
