"""Direct Jacobian calculator using DOLFINx function spaces."""

from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np
from dolfinx import fem

try:
    import torch
except Exception:  # pragma: no cover - optional in some unit stubs
    torch = None

from ...cache.object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
)
from ...femx import function_get_array
from ..solvers.gauss_newton_device import (
    normalize_runtime_device,
    normalize_runtime_device_label,
)
from ._core import (
    assemble_jacobian_efficient_numpy,
    assemble_jacobian_traditional,
    build_jacobian_geometry,
    calibrate_block_size_once,
    compute_field_gradients,
    convert_electrode_to_measurement_jacobian,
    measurement_to_current_patterns,
)
from .base_jacobian import BaseJacobianCalculator
from .linearized import JacobianLinearization, compute_sigma_fingerprint


class DirectJacobianCalculator(BaseJacobianCalculator):
    """Direct (forward + adjoint solves) Jacobian calculator.

    Sign convention: PyEIDORS runtime ``J = +∂V/∂σ`` (``sign=+1.0``).

    The dense Jacobian returned by :meth:`calculate` and the operator
    returned by :meth:`linearize` both encode this positive-sign
    convention. Combined with ``rhs = -jtr`` in
    :func:`pyeidors.inverse.solvers.gauss_newton_runtime` (line 952)
    the end-to-end Gauss-Newton step produces a physical δσ that
    matches the EIDORS reconstruction direction (``δσ > 0`` when the
    inhomogeneous conductivity exceeds the background).

    The sibling :class:`pyeidors.inverse.jacobian.adjoint_jacobian.EidorsJacobianAdapter`
    instead returns the EIDORS-canonical signed Jacobian
    ``J = -∂V/∂σ``. The two calculators differ only in overall sign
    (``DirectJacobianCalculator(...).calculate(σ)
    == -EidorsJacobianAdapter(...).calculate(σ)``); the contract is
    frozen by V73 and exercised by
    ``tests/unit/test_jacobian_direct_adjoint_parity.py``. Do **not**
    swap calculators inside an existing GN/RM pipeline without
    compensating the sign at the consumer site.

    The geometry setup, field-gradient interpolation, adjoint solve
    and the pure-numpy / traditional / electrode-to-measurement assembly
    helpers all live in :mod:`pyeidors.inverse.jacobian._core`. This
    class owns the cache-manager integration, the runtime-device state
    and the CUDA assembly orchestration; everything else delegates.
    """

    sign_convention = "+dV/dsigma_pyeidors_runtime"

    def __init__(
        self,
        fwd_model,
        *,
        block_tune_mode: str = "auto",
        block_size: int = 0,
        block_candidates: tuple[int, ...] | list[int] = (64, 128, 256, 512),
        runtime_device: str = "auto",
    ):
        super().__init__(fwd_model)
        self.block_tune_mode = str(block_tune_mode).strip().lower()
        if self.block_tune_mode not in {"auto", "off"}:
            raise ValueError(
                f"Unsupported block_tune_mode={block_tune_mode!r}. "
                "Expected one of: 'auto', 'off'."
            )
        self.block_size = int(max(0, block_size))
        self.block_candidates = tuple(
            sorted({int(v) for v in block_candidates if int(v) > 0})
        ) or (64, 128, 256, 512)
        self._resolved_block_size: int | None = None
        self._block_tune_source: str = "unset"
        self._last_assembly_elapsed_only: float = 0.0
        self._runtime_device_requested: str = normalize_runtime_device(
            runtime_device, default="auto"
        )
        self._runtime_device_effective: str = "cpu"
        self._runtime_cuda_device: str = "cuda"
        self._jacobian_backend_requested: str = normalize_runtime_device_label(
            self._runtime_device_requested, default="auto"
        )
        self._jacobian_backend_effective: str = "cpu"
        self._jacobian_block_backend: str = "numpy"
        self._jacobian_transfer_estimate: float = 0.0
        self._jacobian_cuda_threshold_hit: bool = False
        self._cell_areas_cuda = None
        self._setup_computation()

    def _setup_computation(self):
        self._geometry = build_jacobian_geometry(self.fwd_model)
        self.mesh = self._geometry.mesh
        self.V = self._geometry.V
        self.V_sigma = self._geometry.V_sigma
        self.gdim = self._geometry.gdim
        self.Q_DG = self._geometry.Q_DG
        self.DG0 = self._geometry.DG0
        self.cell_areas = self._geometry.cell_areas

    def _calibrate_block_size(
        self,
        *,
        grad_u_all,
        adjoint_gradients,
        n_elements: int,
    ) -> int:
        if self.block_size > 0:
            self._block_tune_source = "fixed"
            return int(min(self.block_size, n_elements))

        if self.block_tune_mode == "off":
            self._block_tune_source = "disabled"
            return int(n_elements)

        if n_elements <= 256:
            self._block_tune_source = "small-problem"
            return int(n_elements)

        cache_manager = getattr(self.fwd_model, "cache_manager", None)
        if cache_manager is None or not cache_manager.enabled:
            chosen = self._calibrate_block_size_once(
                grad_u_all, adjoint_gradients, n_elements
            )
            self._block_tune_source = "compute"
            return int(chosen)

        payload = {
            "model_signature": model_signature_from_forward_model(self.fwd_model),
            "pattern_signature": pattern_signature_from_forward_model(self.fwd_model),
            "backend_signature": backend_signature_from_forward_model(self.fwd_model),
            "n_elements": int(n_elements),
            "n_measurements": int(len(adjoint_gradients)),
            "gdim": int(self.gdim),
            "candidates": list(self.block_candidates),
            "mode": self.block_tune_mode,
        }
        chosen, lookup = cache_manager.get_or_compute_semantic(
            artifact="jacobian_block_tune",
            name="direct_jacobian_block_size",
            namespace="inverse",
            cache_obj=payload,
            payload=payload,
            compute_fn=lambda: int(
                self._calibrate_block_size_once(
                    grad_u_all, adjoint_gradients, n_elements
                )
            ),
            persist=True,
            cost=2.0,
            effort_seconds=0.5,
        )
        self._block_tune_source = (
            str(getattr(lookup, "layer", "cache"))
            if getattr(lookup, "hit", False)
            else "compute"
        )
        return int(max(1, min(int(chosen), n_elements)))

    def _calibrate_block_size_once(
        self,
        grad_u_all,
        adjoint_gradients,
        n_elements: int,
    ) -> int:
        return calibrate_block_size_once(
            grad_u_all=grad_u_all,
            adjoint_gradients=adjoint_gradients,
            n_elements=int(n_elements),
            candidates=self.block_candidates,
            sample_meas_count=int(self.fwd_model.pattern_manager.n_meas_per_stim[0]),
        )

    def _resolve_block_size(
        self, grad_u_all, adjoint_gradients, n_elements: int
    ) -> int:
        if self._resolved_block_size is None:
            self._resolved_block_size = self._calibrate_block_size(
                grad_u_all=grad_u_all,
                adjoint_gradients=adjoint_gradients,
                n_elements=n_elements,
            )
        return int(max(1, min(self._resolved_block_size, n_elements)))

    def set_runtime_device(
        self, requested: str, effective: str, *, torch_device=None
    ) -> None:
        self._runtime_device_requested = normalize_runtime_device(
            requested, default="auto"
        )
        self._runtime_device_effective = normalize_runtime_device_label(
            effective, default="cpu"
        )
        self._jacobian_backend_requested = normalize_runtime_device_label(
            self._runtime_device_requested, default="auto"
        )
        if torch_device is not None:
            self._runtime_cuda_device = str(torch_device)

    def _wants_cuda_contraction(self) -> bool:
        requested = getattr(self, "_runtime_device_requested", "auto")
        effective = getattr(self, "_runtime_device_effective", "cpu")
        self._runtime_device_requested = normalize_runtime_device(
            requested, default="auto"
        )
        self._runtime_device_effective = normalize_runtime_device_label(
            effective, default="cpu"
        )
        self._jacobian_backend_requested = normalize_runtime_device_label(
            self._runtime_device_requested, default="auto"
        )
        self._runtime_cuda_device = str(getattr(self, "_runtime_cuda_device", "cuda"))
        return (
            torch is not None
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
            and self._runtime_device_effective == "cuda"
        )

    def _should_use_cuda_contraction(
        self, *, n_measurements: int, n_elements: int
    ) -> bool:
        if not self._wants_cuda_contraction():
            self._jacobian_cuda_threshold_hit = False
            return False
        if self._runtime_device_requested.startswith("cuda"):
            self._jacobian_cuda_threshold_hit = True
            return True
        work = int(n_measurements) * int(n_elements) * int(max(self.gdim, 1))
        threshold_hit = int(n_elements) >= 1024 and work >= 2_000_000
        self._jacobian_cuda_threshold_hit = bool(threshold_hit)
        return bool(threshold_hit)

    def _get_cell_areas_cuda(self):
        if torch is None:
            return None
        if self._cell_areas_cuda is None:
            self._cell_areas_cuda = torch.from_numpy(
                np.asarray(self.cell_areas, dtype=np.float64)
            ).to(self._runtime_cuda_device, dtype=torch.float64)
        return self._cell_areas_cuda

    def block_tuning_info(self) -> dict[str, object]:
        """Expose current Jacobian block-size tuning state for diagnostics."""
        selected = (
            self._resolved_block_size
            if getattr(self, "_resolved_block_size", None) is not None
            else getattr(self, "block_size", 0)
        )
        if selected <= 0:
            selected = max(1, len(self.cell_areas))
        return {
            "selected_block_size": int(selected),
            "tune_mode": getattr(self, "block_tune_mode", "auto"),
            "tune_source": getattr(self, "_block_tune_source", "unset"),
            "candidates": list(getattr(self, "block_candidates", ()) or ()),
            "assembly_elapsed_only": float(
                getattr(self, "_last_assembly_elapsed_only", 0.0)
            ),
            "jacobian_backend_requested": getattr(
                self, "_jacobian_backend_requested", "auto"
            ),
            "jacobian_backend_effective": getattr(
                self, "_jacobian_backend_effective", "cpu"
            ),
            "jacobian_block_backend": getattr(self, "_jacobian_block_backend", "numpy"),
            "jacobian_transfer_estimate": float(
                getattr(self, "_jacobian_transfer_estimate", 0.0)
            ),
            "jacobian_cuda_threshold_hit": bool(
                getattr(self, "_jacobian_cuda_threshold_hit", False)
            ),
        }

    def calculate(
        self, sigma: fem.Function, method: str = "efficient", **kwargs
    ) -> np.ndarray:
        if method not in {"efficient", "traditional"}:
            raise ValueError(f"Unknown method: {method}")

        cache_manager = getattr(self.fwd_model, "cache_manager", None)
        if cache_manager is None or not cache_manager.enabled:
            jacobian = (
                self._calculate_efficient(sigma)
                if method == "efficient"
                else self._calculate_traditional(sigma)
            )
            self._last_block_tune_info = self.block_tuning_info()
            return jacobian

        sigma_values = np.ascontiguousarray(function_get_array(sigma), dtype=np.float64)
        model_signature = model_signature_from_forward_model(self.fwd_model)
        pattern_signature = pattern_signature_from_forward_model(self.fwd_model)
        backend_signature = backend_signature_from_forward_model(self.fwd_model)
        payload = {
            "method": method,
            "sigma_hash": hashlib.sha256(sigma_values.tobytes()).hexdigest(),
            "model_signature": model_signature,
            "pattern_signature": pattern_signature,
            "backend_signature": backend_signature,
        }
        jacobian, lookup = cache_manager.get_or_compute_semantic(
            artifact="jacobian",
            name="calc_jacobian",
            namespace="inverse",
            cache_obj=payload,
            payload=payload,
            compute_fn=(
                (lambda: self._calculate_efficient(sigma))
                if method == "efficient"
                else (lambda: self._calculate_traditional(sigma))
            ),
            persist=True,
            cost=12.0,
        )
        self._last_cache_lookup = {
            "hit": lookup.hit,
            "layer": lookup.layer,
            "artifact": lookup.artifact,
            "key": lookup.key,
        }
        self._last_block_tune_info = self.block_tuning_info()
        return jacobian

    def linearize(
        self, sigma: fem.Function, method: str = "efficient"
    ) -> JacobianLinearization:
        """Build a reusable ``Jv``/``J^T r`` sensitivity operator.

        ``method='efficient'`` follows the same EIDORS-style adjoint path used
        by :meth:`calculate`, but returns an operator instead of the dense
        measurement Jacobian.
        """
        if method != "efficient":
            raise ValueError(
                "Matrix-free linearization currently supports method='efficient'."
            )
        u_all, _ = self.fwd_model.forward_solve(sigma)
        grad_u_all = self._compute_field_gradients(u_all)
        adjoint_fields = self._compute_adjoint_fields_efficient(sigma)
        return JacobianLinearization(
            grad_u_all=tuple(grad_u_all),
            adjoint_gradients=tuple(adjoint_fields),
            cell_areas=np.asarray(self.cell_areas, dtype=np.float64),
            n_meas_per_stim=tuple(self.fwd_model.pattern_manager.n_meas_per_stim),
            sign=1.0,
            sigma_fingerprint=compute_sigma_fingerprint(sigma),
        )

    def _calculate_efficient(self, sigma: fem.Function) -> np.ndarray:
        u_all, _ = self.fwd_model.forward_solve(sigma)
        grad_u_all = self._compute_field_gradients(u_all)
        adjoint_fields = self._compute_adjoint_fields_efficient(sigma)
        return self._assemble_jacobian_efficient(grad_u_all, adjoint_fields)

    def _calculate_traditional(self, sigma: fem.Function) -> np.ndarray:
        u_all, _ = self.fwd_model.forward_solve(sigma)

        I2_all = np.eye(self.fwd_model.n_elec)
        bu_all, _ = self.fwd_model.forward_solve(sigma, I2_all)

        grad_u_all = self._compute_field_gradients(u_all)
        grad_bu_all = self._compute_field_gradients(bu_all)

        jacobian = self._assemble_jacobian_traditional(grad_u_all, grad_bu_all)
        return self._convert_to_measurement_jacobian(jacobian)

    def _compute_field_gradients(self, field_solutions):
        return compute_field_gradients(field_solutions, self._geometry)

    def _compute_adjoint_fields_efficient(self, sigma: fem.Function):
        adjoint_patterns = self._measurement_to_current_patterns()
        adjoint_fields, _ = self.fwd_model.forward_solve(sigma, adjoint_patterns)
        return self._compute_field_gradients(adjoint_fields)

    def _measurement_to_current_patterns(self):
        return measurement_to_current_patterns(self.fwd_model)

    def _assemble_jacobian_efficient(self, grad_u_all, adjoint_gradients):
        n_measurements = len(adjoint_gradients)
        n_elements = len(self.cell_areas)
        block_size = self._resolve_block_size(grad_u_all, adjoint_gradients, n_elements)

        use_cuda_blocks = self._should_use_cuda_contraction(
            n_measurements=n_measurements,
            n_elements=n_elements,
        )
        self._jacobian_backend_effective = "cuda" if use_cuda_blocks else "cpu"
        self._jacobian_block_backend = "torch-cuda" if use_cuda_blocks else "numpy"
        self._jacobian_transfer_estimate = 0.0

        if not use_cuda_blocks:
            jacobian, elapsed = assemble_jacobian_efficient_numpy(
                grad_u_all=grad_u_all,
                adjoint_gradients=adjoint_gradients,
                cell_areas=self.cell_areas,
                n_meas_per_stim=self.fwd_model.pattern_manager.n_meas_per_stim,
                block_size=block_size,
            )
            self._last_assembly_elapsed_only = elapsed
            return jacobian

        return self._assemble_jacobian_efficient_cuda(
            grad_u_all=grad_u_all,
            adjoint_gradients=adjoint_gradients,
            block_size=block_size,
        )

    def _assemble_jacobian_efficient_cuda(
        self, *, grad_u_all, adjoint_gradients, block_size: int
    ) -> np.ndarray:
        n_measurements = len(adjoint_gradients)
        n_elements = len(self.cell_areas)
        jacobian = np.zeros((n_measurements, n_elements), dtype=float)
        cell_areas_cuda = self._get_cell_areas_cuda()

        assembly_t0 = perf_counter()
        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this_stim = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]
            adjoint_block = np.asarray(
                adjoint_gradients[meas_idx : meas_idx + n_meas_this_stim],
                dtype=float,
            )
            grad_u_arr = np.asarray(grad_u, dtype=float)
            for start in range(0, n_elements, block_size):
                end = min(start + block_size, n_elements)
                grad_u_t = torch.from_numpy(
                    np.ascontiguousarray(grad_u_arr[start:end, :], dtype=np.float64)
                ).to(self._runtime_cuda_device, dtype=torch.float64)
                adjoint_block_t = torch.from_numpy(
                    np.ascontiguousarray(
                        adjoint_block[:, start:end, :], dtype=np.float64
                    )
                ).to(self._runtime_cuda_device, dtype=torch.float64)
                sensitivity_t = torch.einsum(
                    "eg,meg->me", grad_u_t, adjoint_block_t
                )
                out_t = sensitivity_t * cell_areas_cuda[start:end].unsqueeze(0)
                jacobian[meas_idx : meas_idx + n_meas_this_stim, start:end] = (
                    out_t.cpu().numpy()
                )
                bytes_h2d = (
                    grad_u_t.numel() + adjoint_block_t.numel() + out_t.numel()
                ) * 8
                self._jacobian_transfer_estimate += float(bytes_h2d)

            meas_idx += n_meas_this_stim

        self._last_assembly_elapsed_only = float(perf_counter() - assembly_t0)
        return jacobian

    def _assemble_jacobian_traditional(self, grad_u_all, grad_bu_all):
        jacobian, elapsed = assemble_jacobian_traditional(
            grad_u_all, grad_bu_all, self.cell_areas
        )
        self._last_assembly_elapsed_only = elapsed
        return jacobian

    def _convert_to_measurement_jacobian(self, electrode_jacobian):
        return convert_electrode_to_measurement_jacobian(
            electrode_jacobian,
            n_stim=self.fwd_model.pattern_manager.n_stim,
            n_elec=self.fwd_model.n_elec,
            meas_matrices=self.fwd_model.pattern_manager.meas_matrices,
        )
