import numpy as np
from typing import Dict, List, Union

from autonerves import cached_property

from autoarray import exc
from autoarray.dataset.interferometer.dataset import Interferometer
from autoarray.inversion.inversion.dataset_interface import DatasetInterface
from autoarray.inversion.inversion.interferometer.sparse import (
    InversionInterferometerSparse,
)
from autoarray.inversion.inversion.interferometer import (
    inversion_interferometer_matrix_free_util as matrix_free_util,
)
from autoarray.inversion.linear_obj.func_list import AbstractLinearObjFuncList
from autoarray.inversion.linear_obj.linear_obj import LinearObj
from autoarray.inversion.mappers.abstract import Mapper
from autoarray.settings import Settings

_ROUTE_HINT = (
    'Use `Settings(interferometer_route="fft")` (or leave `interferometer_route` unset) '
    "for this model."
)

# The regularization methods the matrix-free route needs; only `aa.reg.MaternSparse` supplies
# all three.
_SPARSE_REGULARIZATION_METHODS = (
    "regularization_matrix_sparse_from",
    "regularization_factor_sparse_from",
    "log_det_regularization_matrix_term_from",
)


class InversionInterferometerMatrixFree(InversionInterferometerSparse):

    # The PCG relative residual tolerance and iteration cap of the unconstrained solve.
    pcg_rtol = 1.0e-10
    pcg_maxiter = 5000

    # The stochastic Lanczos quadrature settings of `ln det(F + H)` (the util module defaults).
    slq_n_probes = 10
    slq_n_steps = 40
    slq_seed = 20221

    def __init__(
        self,
        dataset: Union[Interferometer, DatasetInterface],
        linear_obj_list: List[LinearObj],
        settings: Settings = None,
        xp=np,
        preloads=None,
    ):
        """
        The single-mapper interferometer inversion solved matrix-free: neither the dense `(S, S)`
        curvature matrix `F = Aᵀ W~ A` nor the dense regularization matrix `H` is ever formed or
        factorised.

        `InversionInterferometerSparse` forms `F` and `H` densely and factorises `F + H`, which caps
        a pixelized interferometer source at roughly `100^2` mesh pixels. This class instead uses the
        operator form of `inversion_interferometer_matrix_free_util` (PyAutoArray #638, read that
        module's docstring for the method):

        - `F s = Mᵀ W~ (M s)` is applied by `MatrixFreeDataOperator.curvature_matvec`, with `M` the
          sparse mapping matrix and `W~` an FFT convolution, and the data vector is `D = Mᵀ d~`;
        - `H` is the natively sparse Matérn precision of `aa.reg.MaternSparse`, which already
          carries its coefficient (so the utilities' `λ` is `1.0`);
        - the reconstruction solves `(F + H) s = D` by preconditioned conjugate gradients (Powell
          preconditioner), or, when `settings.use_positive_only_solver` is `True`, by the
          interior-point positive solve with an exact active-set polish;
        - `ln det(F + H)` is the exact `ln det P` of the preconditioner plus a stochastic Lanczos
          quadrature estimate of `ln det(P⁻¹(F + H))` (10 probes, 40 steps, fixed seed 20221, so it
          is deterministic and smooth in the model parameters). It is the only approximate evidence
          term: within 0.5% of the dense value on the util module's test case, with an accuracy set
          by how well `P` approximates `F + H`;
        - `ln det H` is exact (`MaternSparse.log_det_regularization_matrix_term_from`).

        The mapping matrix carries any over-sampling (`sub_fraction`) and primary beam exactly as the
        parent's sparse triplets do, so both are supported and match the FFT route.

        The class raises on every configuration it cannot represent rather than silently working
        around it (see `_check_preconditions`), and the inversion factory routes here only when
        `Settings(interferometer_route="matrix_free")` is set, with no fallback. Unsupported:

        - a JAX (`xp=jnp`) array module -- the utilities are NumPy / scipy only;
        - linear-function lists (linear light profiles, MGE bases) and more than one mapper;
        - a mesh other than `RectangularUniform`;
        - a regularization other than `MaternSparse` (one without a sparse matrix, factor and exact
          log determinant);
        - edge-zeroed pixels (`settings.use_edge_zeroed_pixels`) with the positive-only solver: the
          base class's edge-zeroing subsets the dense system, which this class never forms. Pass
          `use_edge_zeroed_pixels=False`.

        Quantities that need the dense matrices -- `curvature_matrix`, `curvature_matrix_diag`,
        `curvature_reg_matrix`, `curvature_reg_matrix_reduced`, `reconstruction_covariance_matrix`,
        `reconstruction_noise_map` and `reconstruction_noise_map_with_covariance` -- raise an
        `InversionException`. Reconstruction uncertainty maps need a stochastic diagonal estimator
        of `(F + H)⁻¹`, which is future work.

        Parameters
        ----------
        dataset
            The interferometer dataset (or `DatasetInterface`) being reconstructed. It must carry a
            `sparse_operator`, whose `nufft_precision_operator` is the `W~` kernel.
        linear_obj_list
            The linear objects reconstructing the data. Exactly one `Mapper`, and nothing else.
        settings
            The inversion settings (`autoarray.settings.Settings`).
        xp
            The array module. Must be `numpy`.
        preloads
            Optional `AbstractPreloads`, forwarded to the parent unchanged.
        """
        if xp is not np:
            raise exc.InversionException(
                '`InversionInterferometerMatrixFree` (the `interferometer_route="matrix_free"` '
                f"route) was passed a non-NumPy array module ({xp!r}). The matrix-free "
                "utilities are NumPy / scipy only (scipy CG, sparse LU), with no JAX path.\n\n"
                + _ROUTE_HINT
            )

        super().__init__(
            dataset=dataset,
            linear_obj_list=linear_obj_list,
            settings=settings,
            xp=xp,
            preloads=preloads,
        )

        self._check_preconditions()

    def _check_preconditions(self) -> None:
        """
        Raise on every configuration the matrix-free route cannot represent.

        These are raised, not worked around: each would otherwise give a different (silently wrong)
        reconstruction or evidence, or none at all.
        """
        from autoarray.inversion.mesh.mesh.rectangular_uniform import (
            RectangularUniform,
        )

        if self.has(cls=AbstractLinearObjFuncList):
            raise exc.InversionException(
                "A linear-function list (e.g. a linear light profile or MGE basis) was passed to "
                '`InversionInterferometerMatrixFree` (the `interferometer_route="matrix_free"` '
                "route). The matrix-free operator represents a single mapper's sparse mapping "
                "matrix only and has no mapper x function or function x function block.\n\n"
                + _ROUTE_HINT
            )

        total_mappers = self.total(cls=Mapper)

        if total_mappers != 1:
            raise exc.InversionException(
                f'`InversionInterferometerMatrixFree` (the `interferometer_route="matrix_free"` '
                f"route) was passed {total_mappers} mappers. The matrix-free operator represents "
                "exactly one mapper and has no off-diagonal mapper x mapper block.\n\n"
                + _ROUTE_HINT
            )

        mapper = self.cls_list_from(cls=Mapper)[0]

        if not isinstance(mapper.mesh, RectangularUniform):
            raise exc.InversionException(
                '`InversionInterferometerMatrixFree` (the `interferometer_route="matrix_free"` '
                f"route) was passed a mapper with a {type(mapper.mesh).__name__} mesh. The route "
                "requires a `RectangularUniform` mesh, the mesh the natively sparse Matérn "
                "regularization (`aa.reg.MaternSparse`) is built and tested on.\n\n"
                + _ROUTE_HINT
            )

        regularization = mapper.regularization

        missing = [
            method
            for method in _SPARSE_REGULARIZATION_METHODS
            if not hasattr(regularization, method)
        ]

        if missing:
            raise exc.InversionException(
                '`InversionInterferometerMatrixFree` (the `interferometer_route="matrix_free"` '
                f"route) was passed a {type(regularization).__name__} regularization, which lacks "
                f"{', '.join(f'`{method}`' for method in missing)}. The route needs the "
                "regularization matrix as a sparse matrix, its sparse factor (for the evidence "
                "probes) and its exact log determinant; only `aa.reg.MaternSparse` supplies "
                "them.\n\n"
                "Use `aa.reg.MaternSparse`, or " + _ROUTE_HINT
            )

        if (
            self.settings.use_positive_only_solver
            and self.settings.use_edge_zeroed_pixels
        ):
            raise exc.InversionException(
                '`InversionInterferometerMatrixFree` (the `interferometer_route="matrix_free"` '
                "route) was passed `use_positive_only_solver=True` with "
                "`use_edge_zeroed_pixels=True`. Edge-zeroing subsets the dense "
                "`curvature_reg_matrix`, which this route never forms, so it is not applied "
                "here.\n\n"
                "Pass `Settings(..., use_edge_zeroed_pixels=False)` to solve the full positive "
                "system matrix-free, or " + _ROUTE_HINT
            )

    @property
    def _mapper(self) -> Mapper:
        return self.cls_list_from(cls=Mapper)[0]

    @cached_property
    def operator(self) -> matrix_free_util.MatrixFreeDataOperator:
        """
        The matrix-free data operator of the single mapper: `F s` by `curvature_matvec` and
        `D = Mᵀ d~` by `data_vector`.
        """
        return matrix_free_util.MatrixFreeDataOperator.from_inversion(
            inversion=self, mapper=self._mapper
        )

    @cached_property
    def regularization_matrix_sparse(self):
        """
        The regularization matrix `H` as a `scipy.sparse` matrix. It already carries the
        regularization coefficient, so every utility is called with `coefficient=1.0`.
        """
        mapper = self._mapper

        return mapper.regularization.regularization_matrix_sparse_from(mapper)

    @cached_property
    def regularization_factor_sparse(self):
        """
        The sparse factor `R` of exactly `H = RᵀR`: `MaternSparse.regularization_factor_sparse_from`
        returns the factor of the unscaled precision, so the coefficient is folded in as
        `sqrt(coefficient)`.
        """
        mapper = self._mapper
        regularization = mapper.regularization

        return np.sqrt(
            regularization.coefficient
        ) * regularization.regularization_factor_sparse_from(mapper)

    @cached_property
    def data_vector(self) -> np.ndarray:
        """
        The data vector `D = Mᵀ d~`, from the matrix-free operator.
        """
        return self.operator.data_vector

    @cached_property
    def reconstruction(self) -> np.ndarray:
        """
        The reconstruction solving `(F + H) s = D` without forming `F` or `H` densely.

        With `settings.use_positive_only_solver` it is the non-negative solution
        (`reconstruction_positive_via_interior_point_from`, exact zeros on the held pixels), stored
        with its diagnostics on `positive_result`; otherwise the unconstrained PCG solution
        (`reconstruction_via_pcg_from`), which raises an `InversionException` if CG does not converge
        within `pcg_maxiter` iterations.
        """
        if self.settings.use_positive_only_solver:
            result = matrix_free_util.reconstruction_positive_via_interior_point_from(
                operator=self.operator,
                regularization_matrix_sparse=self.regularization_matrix_sparse,
                coefficient=1.0,
            )

            self.positive_result = result

            return np.asarray(result.reconstruction, dtype=np.float64)

        reconstruction, info = matrix_free_util.reconstruction_via_pcg_from(
            operator=self.operator,
            regularization_matrix_sparse=self.regularization_matrix_sparse,
            coefficient=1.0,
            rtol=self.pcg_rtol,
            maxiter=self.pcg_maxiter,
        )

        if info != 0:
            raise exc.InversionException(
                "The matrix-free PCG solve of `InversionInterferometerMatrixFree` did not "
                f"converge to rtol={self.pcg_rtol} within {info} iterations."
            )

        return np.asarray(reconstruction, dtype=np.float64)

    @property
    def regularization_term(self) -> float:
        """
        The regularization term `sᵀ H s`, with the sparse `H` (which carries the coefficient, as the
        dense `regularization_matrix` does).
        """
        s = self.reconstruction

        return float(s @ (self.regularization_matrix_sparse @ s))

    @cached_property
    def log_det_curvature_reg_matrix_term(self) -> float:
        """
        `ln det(F + H)`: the exact `ln det P` of the Powell preconditioner plus the stochastic Lanczos
        quadrature estimate of `ln det(P⁻¹(F + H))`, both returned together by
        `log_det_curvature_reg_via_slq_from` (the estimate's per-probe values already include
        `ln det P`).
        """
        log_det, _ = matrix_free_util.log_det_curvature_reg_via_slq_from(
            operator=self.operator,
            regularization_matrix_sparse=self.regularization_matrix_sparse,
            regularization_factor_sparse=self.regularization_factor_sparse,
            coefficient=1.0,
            n_probes=self.slq_n_probes,
            n_steps=self.slq_n_steps,
            seed=self.slq_seed,
        )

        return log_det

    @cached_property
    def log_det_regularization_matrix_term(self) -> float:
        """
        The exact `ln det H` from the regularization's sparse factorisation
        (`MaternSparse.log_det_regularization_matrix_term_from`, which includes the coefficient).
        """
        mapper = self._mapper

        return float(
            mapper.regularization.log_det_regularization_matrix_term_from(mapper)
        )

    @property
    def fast_chi_squared(self):
        """
        The chi-squared `sᵀ F s - 2 sᵀ D + dᵀ N⁻¹ d`, as `AbstractInversionInterferometer.fast_chi_squared`
        but with `F s` applied matrix-free (`operator.curvature_matvec`).
        """
        s = self.reconstruction

        chi_squared_term_1 = float(s @ self.operator.curvature_matvec(s))

        chi_squared_term_2 = -2.0 * float(s @ self.data_vector)

        if self.dataset.data is None:
            # The interface carries no visibilities, so term 3 is a precomputed scalar: the
            # interface's own `data_term` (that of profile-subtracted visibilities) when given,
            # else the `d^T N^-1 d` the sparse operator cached when it was built from the raw data
            # (e.g. a pixelization-only fit, where nothing was subtracted).
            chi_squared_term_3 = getattr(self.dataset, "data_term", None)

            if chi_squared_term_3 is None:
                chi_squared_term_3 = getattr(
                    self.dataset.sparse_operator, "data_term", None
                )

            if chi_squared_term_3 is None:
                raise exc.InversionException(
                    "The dataset input to this interferometer inversion has `data=None`, which "
                    "is only valid when its `sparse_operator` carries a precomputed `data_term` "
                    "(sum(d_r^2/sigma_r^2) + sum(d_i^2/sigma_i^2)) -- e.g. one built by "
                    "`Interferometer.apply_sparse_operator` or "
                    "`Interferometer.apply_sparse_operator_from_chunks`. This dataset's "
                    "sparse operator has none, so `fast_chi_squared` cannot be computed. "
                    "Pass the visibilities as `data` instead."
                )
        else:
            chi_squared_term_3 = np.sum(
                self.dataset.data.array.real**2.0
                / self.dataset.noise_map.array.real**2.0
            ) + np.sum(
                self.dataset.data.array.imag**2.0
                / self.dataset.noise_map.array.imag**2.0
            )

        return chi_squared_term_1 + chi_squared_term_2 + chi_squared_term_3

    def _dense_unavailable(self, name: str):
        return exc.InversionException(
            f"`{name}` is not available on `InversionInterferometerMatrixFree` (the "
            '`interferometer_route="matrix_free"` route), which never forms the dense curvature '
            "or regularization matrices. Reconstruction uncertainties need a stochastic diagonal "
            "estimator of (F + H)^-1, which is not implemented yet.\n\n" + _ROUTE_HINT
        )

    @property
    def curvature_matrix(self):
        raise self._dense_unavailable("curvature_matrix")

    @property
    def curvature_matrix_diag(self):
        raise self._dense_unavailable("curvature_matrix_diag")

    @property
    def curvature_reg_matrix(self):
        raise self._dense_unavailable("curvature_reg_matrix")

    @property
    def curvature_reg_matrix_reduced(self):
        raise self._dense_unavailable("curvature_reg_matrix_reduced")

    @property
    def reconstruction_covariance_matrix(self):
        raise self._dense_unavailable("reconstruction_covariance_matrix")

    @property
    def reconstruction_noise_map_with_covariance(self):
        raise self._dense_unavailable("reconstruction_noise_map_with_covariance")

    @property
    def reconstruction_noise_map(self):
        raise self._dense_unavailable("reconstruction_noise_map")

    @property
    def reconstruction_noise_map_dict(self) -> Dict[LinearObj, np.ndarray]:
        raise self._dense_unavailable("reconstruction_noise_map_dict")
