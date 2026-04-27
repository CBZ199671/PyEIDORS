"""EIDORS-style adjoint Jacobian calculator with optional Torch accumulation."""

from __future__ import annotations

import numpy as np
import torch
from dolfinx import fem

from ._core import (
    assemble_jacobian_efficient_numpy,
    build_jacobian_geometry,
    compute_field_gradients,
    measurement_to_current_patterns,
)
from .base_jacobian import BaseJacobianCalculator
from .linearized import (
    JacobianLinearization,
    LazyAdjointJacobianLinearization,
    compute_sigma_fingerprint,
)


class EidorsStyleAdjointJacobian(BaseJacobianCalculator):
    """Adjoint Jacobian calculator with EIDORS canonical sign convention.

    Sign convention: EIDORS canonical physical ``J = -∂V/∂σ``
    (``sign=-1.0``). Mirrors the trailing ``J = -J;`` step of
    ``calc_jacobian_adjoint.m`` so that downstream EIDORS-style consumers
    (e.g. ``J' * W * (vi - vh)`` directly used as RHS without an extra
    negation) yield correct-sign reconstructions.

    The sibling :class:`pyeidors.inverse.jacobian.direct_jacobian.DirectJacobianCalculator`
    uses the opposite PyEIDORS runtime convention ``J = +∂V/∂σ``; that
    convention is what the production GN runtime (`gauss_newton_runtime.py:952`,
    `rhs = -jtr`) is paired with. The two calculators MUST satisfy
    ``Direct.calculate(σ) == -EidorsStyleAdjointJacobian.calculate(σ)``;
    the contract is frozen by V73 and exercised by
    ``tests/unit/test_jacobian_direct_adjoint_parity.py``. Do **not**
    swap calculators inside an existing GN/RM pipeline without
    compensating the sign at the consumer site.

    Unique features beyond DirectJacobianCalculator: ``use_torch`` /
    ``device`` / ``torch_dtype`` / ``torch_batch_all`` GPU controls and
    :meth:`linearize_lazy` for the matrix-free
    :class:`LazyAdjointJacobianLinearization` path. The eventual Path C
    refactor will move these into a shared core; until then this class
    coexists with the direct calculator by design.
    """

    sign_convention = "-dV/dsigma_eidors_canonical"

    def __init__(
        self,
        fwd_model,
        use_torch: bool = False,
        device: str | None = None,
        torch_dtype: str | torch.dtype | None = None,
        torch_batch_all: bool = False,
    ):
        super().__init__(fwd_model)
        self.use_torch = use_torch
        self.torch_batch_all = torch_batch_all
        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        if device is not None:
            self.torch_device = torch.device(device)
        elif torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
        else:
            self.torch_device = torch.device("cpu")
        self._setup()

    @staticmethod
    def _resolve_torch_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
        if dtype is None:
            return torch.float64
        if isinstance(dtype, torch.dtype):
            return dtype
        dtype_str = str(dtype).lower()
        if dtype_str in {"float32", "fp32", "f32", "torch.float32"}:
            return torch.float32
        if dtype_str in {"float64", "fp64", "f64", "double", "torch.float64"}:
            return torch.float64
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    def _setup(self):
        self._geometry = build_jacobian_geometry(self.fwd_model)
        self.mesh = self._geometry.mesh
        self.V = self._geometry.V
        self.V_sigma = self._geometry.V_sigma
        self.gdim = self._geometry.gdim
        self.Q_DG = self._geometry.Q_DG
        self.DG0 = self._geometry.DG0
        self.cell_areas = self._geometry.cell_areas

        if self.use_torch:
            self.cell_areas_t = torch.from_numpy(self.cell_areas).to(
                self.torch_device, dtype=self.torch_dtype
            )

    def calculate(self, sigma: fem.Function, **kwargs) -> np.ndarray:
        u_all, _ = self.fwd_model.forward_solve(sigma)
        grad_u_all = self._compute_field_gradients(u_all)

        meas_curr = self._measurement_to_current_patterns()
        adj_fields, _ = self.fwd_model.forward_solve(sigma, meas_curr)
        grad_adj_all = self._compute_field_gradients(adj_fields)

        if self.use_torch:
            J = self._assemble_torch(grad_u_all, grad_adj_all)
        else:
            J = self._assemble_numpy(grad_u_all, grad_adj_all)
        return J

    def linearize(self, sigma: fem.Function, **kwargs) -> JacobianLinearization:
        """Return EIDORS-style ``Jv``/``J^T r`` actions without dense ``J``."""
        u_all, _ = self.fwd_model.forward_solve(sigma)
        grad_u_all = self._compute_field_gradients(u_all)

        meas_curr = self._measurement_to_current_patterns()
        adj_fields, _ = self.fwd_model.forward_solve(sigma, meas_curr)
        grad_adj_all = self._compute_field_gradients(adj_fields)

        return JacobianLinearization(
            grad_u_all=tuple(grad_u_all),
            adjoint_gradients=tuple(grad_adj_all),
            cell_areas=np.asarray(self.cell_areas, dtype=np.float64),
            n_meas_per_stim=tuple(self.fwd_model.pattern_manager.n_meas_per_stim),
            sign=-1.0,
            sigma_fingerprint=compute_sigma_fingerprint(sigma),
        )

    def linearize_from_image(self, img, **kwargs) -> JacobianLinearization:
        sigma = fem.Function(self.fwd_model.V_sigma)
        sigma.x.array[:] = img.get_conductivity()
        return self.linearize(sigma, **kwargs)

    def linearize_lazy(
        self, sigma: fem.Function, **kwargs
    ) -> LazyAdjointJacobianLinearization:
        """Return lazy ``Jv``/``J^T r`` actions without per-measurement adjoints."""
        u_all, _ = self.fwd_model.forward_solve(sigma)
        grad_u_all = self._compute_field_gradients(u_all)
        return LazyAdjointJacobianLinearization(
            fwd_model=self.fwd_model,
            sigma_values=np.asarray(sigma.x.array, dtype=np.float64).copy(),
            u_all=tuple(np.asarray(u, dtype=np.float64).copy() for u in u_all),
            grad_u_all=tuple(grad_u_all),
            cell_areas=np.asarray(self.cell_areas, dtype=np.float64),
            n_meas_per_stim=tuple(self.fwd_model.pattern_manager.n_meas_per_stim),
            meas_matrices=tuple(
                np.asarray(matrix, dtype=np.float64)
                for matrix in self.fwd_model.pattern_manager.meas_matrices
            ),
            gradient_callback=self._compute_field_gradients,
            sign=-1.0,
            sigma_fingerprint=compute_sigma_fingerprint(sigma),
            diag_exact_max_measurements=int(
                kwargs.get("diag_exact_max_measurements", 512)
            ),
            diag_chunk_size=int(kwargs.get("diag_chunk_size", 128)),
        )

    def linearize_lazy_from_image(
        self, img, **kwargs
    ) -> LazyAdjointJacobianLinearization:
        sigma = fem.Function(self.fwd_model.V_sigma)
        sigma.x.array[:] = img.get_conductivity()
        return self.linearize_lazy(sigma, **kwargs)

    def _compute_field_gradients(self, field_solutions):
        return compute_field_gradients(field_solutions, self._geometry)

    def _measurement_to_current_patterns(self) -> np.ndarray:
        return measurement_to_current_patterns(self.fwd_model)

    def _assemble_numpy(
        self, grad_u_all: list[np.ndarray], grad_adj_all: list[np.ndarray]
    ) -> np.ndarray:
        """Assemble EIDORS-canonical ``J = -∂V/∂σ`` via shared core + sign flip.

        Stage 3 of T75 routes Adjoint's NumPy path through the same
        :func:`pyeidors.inverse.jacobian._core.assemble_jacobian_efficient_numpy`
        kernel that backs ``DirectJacobianCalculator``. The only difference
        between the two calculators on this path is the trailing sign
        convention (V73): Adjoint negates the shared result so it returns
        ``-Direct``. The Torch GPU path (``_assemble_torch`` / ``_assemble_torch_all``)
        remains inline because it is unique to this calculator and exercised
        through the explicit ``use_torch`` flag.
        """
        jacobian, _ = assemble_jacobian_efficient_numpy(
            grad_u_all=grad_u_all,
            adjoint_gradients=grad_adj_all,
            cell_areas=self.cell_areas,
            n_meas_per_stim=self.fwd_model.pattern_manager.n_meas_per_stim,
            block_size=int(len(self.cell_areas) or 1),
        )
        return -jacobian

    def _assemble_torch(
        self, grad_u_all: list[np.ndarray], grad_adj_all: list[np.ndarray]
    ) -> np.ndarray:
        if self.torch_batch_all:
            return self._assemble_torch_all(grad_u_all, grad_adj_all)

        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elem = len(self.cell_areas)
        J_t = torch.zeros(
            (n_meas, n_elem), device=self.torch_device, dtype=self.torch_dtype
        )

        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]
            grad_u_t = torch.from_numpy(grad_u).to(
                self.torch_device, dtype=self.torch_dtype
            )
            adj_block = np.stack(
                grad_adj_all[meas_idx : meas_idx + n_meas_this], axis=0
            )
            adj_block_t = torch.from_numpy(adj_block).to(
                self.torch_device, dtype=self.torch_dtype
            )
            sensitivity = (
                -(adj_block_t * grad_u_t.unsqueeze(0)).sum(dim=2) * self.cell_areas_t
            )
            J_t[meas_idx : meas_idx + n_meas_this, :] = sensitivity
            meas_idx += n_meas_this
        return J_t.cpu().numpy()

    def _assemble_torch_all(
        self, grad_u_all: list[np.ndarray], grad_adj_all: list[np.ndarray]
    ) -> np.ndarray:
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elem = len(self.cell_areas)
        np_dtype = np.float32 if self.torch_dtype == torch.float32 else np.float64

        adj_block = np.stack(grad_adj_all, axis=0).astype(np_dtype, copy=False)
        grad_u_block = np.zeros((n_meas, n_elem, adj_block.shape[2]), dtype=np_dtype)

        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]
            grad_u_block[meas_idx : meas_idx + n_meas_this] = grad_u.astype(
                np_dtype, copy=False
            )
            meas_idx += n_meas_this

        grad_u_t = torch.from_numpy(grad_u_block).to(
            self.torch_device, dtype=self.torch_dtype
        )
        adj_block_t = torch.from_numpy(adj_block).to(
            self.torch_device, dtype=self.torch_dtype
        )
        sensitivity = -(adj_block_t * grad_u_t).sum(dim=2) * self.cell_areas_t
        return sensitivity.cpu().numpy()
