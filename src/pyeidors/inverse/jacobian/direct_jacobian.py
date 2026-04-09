"""Direct Jacobian calculator using DOLFINx function spaces."""

from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc

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
from ..solvers.gauss_newton_device import normalize_runtime_device, normalize_runtime_device_label
from .base_jacobian import BaseJacobianCalculator


class DirectJacobianCalculator(BaseJacobianCalculator):
    """Direct method Jacobian calculator."""

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
        self._runtime_device_requested: str = normalize_runtime_device(runtime_device, default="auto")
        self._runtime_device_effective: str = "cpu"
        self._runtime_cuda_device: str = "cuda"
        self._jacobian_backend_requested: str = normalize_runtime_device_label(self._runtime_device_requested, default="auto")
        self._jacobian_backend_effective: str = "cpu"
        self._jacobian_block_backend: str = "numpy"
        self._jacobian_transfer_estimate: float = 0.0
        self._jacobian_cuda_threshold_hit: bool = False
        self._cell_areas_cuda = None
        self._setup_computation()

    def _setup_computation(self):
        self.mesh = self.fwd_model.mesh
        self.V = self.fwd_model.V
        self.V_sigma = self.fwd_model.V_sigma
        self.gdim = self.mesh.geometry.dim

        self.Q_DG = fem.functionspace(self.mesh, ("DG", 0, (self.gdim,)))
        self.DG0 = fem.functionspace(self.mesh, ("DG", 0))

        v = ufl.TestFunction(self.DG0)
        areas_vec = fem_petsc.assemble_vector(fem.form(v * ufl.dx))
        areas_vec.assemble()
        self.cell_areas = np.asarray(areas_vec.array, dtype=float)

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
            chosen = self._calibrate_block_size_once(grad_u_all, adjoint_gradients, n_elements)
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
                self._calibrate_block_size_once(grad_u_all, adjoint_gradients, n_elements)
            ),
            persist=True,
            cost=2.0,
            effort_seconds=0.5,
        )
        self._block_tune_source = str(getattr(lookup, "layer", "cache")) if getattr(lookup, "hit", False) else "compute"
        return int(max(1, min(int(chosen), n_elements)))

    def _calibrate_block_size_once(
        self,
        grad_u_all,
        adjoint_gradients,
        n_elements: int,
    ) -> int:
        if not grad_u_all or not adjoint_gradients:
            return int(min(n_elements, self.block_candidates[-1]))

        sample_grad_u = np.asarray(grad_u_all[0], dtype=float)
        if sample_grad_u.ndim != 2 or sample_grad_u.shape[0] == 0:
            return int(min(n_elements, self.block_candidates[-1]))

        local_meas = int(self.fwd_model.pattern_manager.n_meas_per_stim[0])
        sample_adjoint = np.asarray(adjoint_gradients[:local_meas], dtype=float)
        if sample_adjoint.ndim != 3 or sample_adjoint.shape[1] == 0:
            return int(min(n_elements, self.block_candidates[-1]))

        n_sample_elem = int(min(sample_grad_u.shape[0], 2048))
        sample_grad_u = sample_grad_u[:n_sample_elem, :]
        sample_adjoint = sample_adjoint[:, :n_sample_elem, :]

        candidates = sorted(
            {
                max(16, min(int(candidate), n_sample_elem))
                for candidate in self.block_candidates
            }
        )
        if not candidates:
            return int(min(n_elements, 256))

        best_size = candidates[0]
        best_elapsed = float("inf")
        for candidate in candidates:
            t0 = perf_counter()
            for start in range(0, n_sample_elem, candidate):
                end = min(start + candidate, n_sample_elem)
                _ = np.einsum(
                    "eg,meg->me",
                    sample_grad_u[start:end, :],
                    sample_adjoint[:, start:end, :],
                    optimize=True,
                )
            elapsed = perf_counter() - t0
            if elapsed < best_elapsed:
                best_elapsed = elapsed
                best_size = candidate
        return int(max(1, min(best_size, n_elements)))

    def _resolve_block_size(self, grad_u_all, adjoint_gradients, n_elements: int) -> int:
        if self._resolved_block_size is None:
            self._resolved_block_size = self._calibrate_block_size(
                grad_u_all=grad_u_all,
                adjoint_gradients=adjoint_gradients,
                n_elements=n_elements,
            )
        return int(max(1, min(self._resolved_block_size, n_elements)))

    def set_runtime_device(self, requested: str, effective: str, *, torch_device=None) -> None:
        self._runtime_device_requested = normalize_runtime_device(requested, default="auto")
        self._runtime_device_effective = normalize_runtime_device_label(effective, default="cpu")
        self._jacobian_backend_requested = normalize_runtime_device_label(self._runtime_device_requested, default="auto")
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

    def _should_use_cuda_contraction(self, *, n_measurements: int, n_elements: int) -> bool:
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
            self._cell_areas_cuda = torch.from_numpy(np.asarray(self.cell_areas, dtype=np.float64)).to(
                self._runtime_cuda_device, dtype=torch.float64
            )
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
            "assembly_elapsed_only": float(getattr(self, "_last_assembly_elapsed_only", 0.0)),
            "jacobian_backend_requested": getattr(self, "_jacobian_backend_requested", "auto"),
            "jacobian_backend_effective": getattr(self, "_jacobian_backend_effective", "cpu"),
            "jacobian_block_backend": getattr(self, "_jacobian_block_backend", "numpy"),
            "jacobian_transfer_estimate": float(getattr(self, "_jacobian_transfer_estimate", 0.0)),
            "jacobian_cuda_threshold_hit": bool(getattr(self, "_jacobian_cuda_threshold_hit", False)),
        }

    def calculate(self, sigma: fem.Function, method: str = "efficient", **kwargs) -> np.ndarray:
        if method not in {"efficient", "traditional"}:
            raise ValueError(f"Unknown method: {method}")

        cache_manager = getattr(self.fwd_model, "cache_manager", None)
        if cache_manager is None or not cache_manager.enabled:
            jacobian = self._calculate_efficient(sigma) if method == "efficient" else self._calculate_traditional(sigma)
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
            compute_fn=(lambda: self._calculate_efficient(sigma))
            if method == "efficient"
            else (lambda: self._calculate_traditional(sigma)),
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
        gradients = []
        interpolation_points = self.Q_DG.element.interpolation_points
        if callable(interpolation_points):
            interpolation_points = interpolation_points()
        for field in field_solutions:
            u_fun = fem.Function(self.V)
            u_fun.x.array[:] = field

            grad_expr = fem.Expression(ufl.grad(u_fun), interpolation_points)
            grad_u = fem.Function(self.Q_DG)
            grad_u.interpolate(grad_expr)
            grad_u_vec = grad_u.x.array.reshape(-1, self.gdim)
            gradients.append(grad_u_vec)

        return gradients

    def _compute_adjoint_fields_efficient(self, sigma: fem.Function):
        adjoint_patterns = self._measurement_to_current_patterns()
        adjoint_fields, _ = self.fwd_model.forward_solve(sigma, adjoint_patterns)
        return self._compute_field_gradients(adjoint_fields)

    def _measurement_to_current_patterns(self):
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elec = self.fwd_model.n_elec

        current_patterns = np.zeros((n_elec, n_meas), dtype=float)

        meas_idx = 0
        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            n_meas_this_stim = meas_matrix.shape[0]

            current_patterns[:, meas_idx : meas_idx + n_meas_this_stim] = meas_matrix.T
            meas_idx += n_meas_this_stim

        return current_patterns

    def _assemble_jacobian_efficient(self, grad_u_all, adjoint_gradients):
        n_measurements = len(adjoint_gradients)
        n_elements = len(self.cell_areas)
        block_size = self._resolve_block_size(grad_u_all, adjoint_gradients, n_elements)

        jacobian = np.zeros((n_measurements, n_elements), dtype=float)
        assembly_t0 = perf_counter()
        use_cuda_blocks = self._should_use_cuda_contraction(
            n_measurements=n_measurements,
            n_elements=n_elements,
        )
        self._jacobian_backend_effective = "cuda" if use_cuda_blocks else "cpu"
        self._jacobian_block_backend = "torch-cuda" if use_cuda_blocks else "numpy"
        self._jacobian_transfer_estimate = 0.0
        cell_areas_cuda = self._get_cell_areas_cuda() if use_cuda_blocks else None

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
                if use_cuda_blocks:
                    grad_u_t = torch.from_numpy(np.ascontiguousarray(grad_u_arr[start:end, :], dtype=np.float64)).to(
                        self._runtime_cuda_device, dtype=torch.float64
                    )
                    adjoint_block_t = torch.from_numpy(
                        np.ascontiguousarray(adjoint_block[:, start:end, :], dtype=np.float64)
                    ).to(self._runtime_cuda_device, dtype=torch.float64)
                    sensitivity_t = torch.einsum("eg,meg->me", grad_u_t, adjoint_block_t)
                    out_t = sensitivity_t * cell_areas_cuda[start:end].unsqueeze(0)
                    jacobian[meas_idx : meas_idx + n_meas_this_stim, start:end] = out_t.cpu().numpy()
                    bytes_h2d = (grad_u_t.numel() + adjoint_block_t.numel() + out_t.numel()) * 8
                    self._jacobian_transfer_estimate += float(bytes_h2d)
                else:
                    sensitivity_block = np.einsum(
                        "eg,meg->me",
                        grad_u_arr[start:end, :],
                        adjoint_block[:, start:end, :],
                        optimize=True,
                    )
                    jacobian[meas_idx : meas_idx + n_meas_this_stim, start:end] = (
                        sensitivity_block * self.cell_areas[None, start:end]
                    )

            meas_idx += n_meas_this_stim

        self._last_assembly_elapsed_only = float(perf_counter() - assembly_t0)
        return jacobian

    def _assemble_jacobian_traditional(self, grad_u_all, grad_bu_all):
        assembly_t0 = perf_counter()
        jacobian_blocks = []

        for grad_u in grad_u_all:
            derivatives = []
            for grad_bu in grad_bu_all:
                sensitivity = np.sum(grad_bu * grad_u, axis=1) * self.cell_areas
                derivatives.append(sensitivity)

            jacobian_blocks.append(np.array(derivatives))

        self._last_assembly_elapsed_only = float(perf_counter() - assembly_t0)
        return np.vstack(jacobian_blocks)

    def _convert_to_measurement_jacobian(self, electrode_jacobian):
        measurement_jacobian_blocks = []

        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            elec_start = stim_idx * self.fwd_model.n_elec
            elec_end = (stim_idx + 1) * self.fwd_model.n_elec
            electrode_jac_for_stim = electrode_jacobian[elec_start:elec_end, :]

            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            meas_jacobian_for_stim = meas_matrix @ electrode_jac_for_stim

            measurement_jacobian_blocks.append(meas_jacobian_for_stim)

        return np.vstack(measurement_jacobian_blocks)
