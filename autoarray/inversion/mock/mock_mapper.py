import numpy as np

from autoarray.inversion.mappers.abstract import Mapper


class MockMapper(Mapper):
    def __init__(
        self,
        mesh=None,
        mesh_geometry=None,
        interpolator=None,
        source_plane_data_grid=None,
        source_plane_mesh_grid=None,
        adapt_data=None,
        over_sampler=None,
        regularization=None,
        mapping_matrix=None,
        pixel_signals=None,
        parameters=None,
    ):

        super().__init__(
            interpolator=interpolator,
            regularization=regularization,
        )

        self._mesh = mesh
        self._source_plane_data_grid = source_plane_data_grid
        self._source_plane_mesh_grid = source_plane_mesh_grid
        self._adapt_data = adapt_data
        self._mesh_geometry = mesh_geometry
        self._over_sampler = over_sampler
        self._mapping_matrix = mapping_matrix
        self._parameters = parameters
        self._pixel_signals = pixel_signals

    def pixel_signals_from(self, signal_scale, xp=np):
        if self._pixel_signals is None:
            return super().pixel_signals_from(signal_scale=signal_scale)
        return self._pixel_signals

    @property
    def source_plane_data_grid(self):
        if self._source_plane_data_grid is None:
            return super().source_plane_data_grid
        return self._source_plane_data_grid

    @property
    def source_plane_mesh_grid(self):
        if self._source_plane_mesh_grid is None:
            return super().source_plane_mesh_grid
        return self._source_plane_mesh_grid

    @property
    def adapt_data(self):
        if self._adapt_data is None:
            return super().adapt_data
        return self._adapt_data

    @property
    def mesh(self):
        if self._mesh is None:
            return super().mesh
        return self._mesh

    @property
    def mesh_geometry(self):
        if self._mesh_geometry is None:
            return super().mesh_geometry
        return self._mesh_geometry

    @property
    def params(self):
        if self._parameters is None:
            return super().params
        return self._parameters

    @property
    def over_sampler(self):
        if self._over_sampler is None:
            return super().over_sampler
        return self._over_sampler

    @property
    def mapping_matrix(self):
        return self._mapping_matrix
