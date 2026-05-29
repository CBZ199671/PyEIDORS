"""Single-rank CUDA forward backend for generated structured 3D hex meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

from dolfinx import fem

try:  # pragma: no cover - optional in lean test stubs
    import meshio
except Exception:  # pragma: no cover
    meshio = None  # type: ignore[assignment]
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

try:  # pragma: no cover - optional in lean test stubs
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..cache.keys import hash_array_payload
from ..geometry.mesh3d_generator import (
    STRUCTURED_SIDECAR_VERSION,
    load_structured_sidecar,
    structured_sidecar_path_for_mesh,
)
from ..perf.policy import DEFAULT_3D_GENERATOR_REVISION
from ..utils.numeric_ops import all_finite_values

CUDA_STRUCTURED_BACKEND_VERSION = "cuda-structured-v1"


def _torch_cuda_available() -> bool:
    return bool(
        torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()
    )


def _norm(value: object) -> str:
    """Normalize a string-like value to lowercase stripped form."""
    return str(value).strip().lower()


def resolve_cuda_structured_runtime(
    *,
    mesh_dim: int,
    mesh_file: str | None,
    mesh_family: str | None,
    geometry_version: str | None,
    generator_revision: str | None,
    petsc_device_requested: str | None,
    scalar_type: str,
    mesh_comm_size: int,
) -> dict[str, Any]:
    if int(mesh_dim) != 3:
        raise ValueError(
            "forward_backend='cuda_structured' currently supports 3D meshes only."
        )
    if _norm(scalar_type) != "real":
        raise ValueError(
            "forward_backend='cuda_structured' currently supports real-valued "
            "conductivity only. For complex admittivity GPU CEM, use "
            "forward_backend='dolfinx' with petsc_device='cuda' in "
            "`nix develop .#complex-cuda` or `nix develop .#complex64-cuda`."
        )
    if int(mesh_comm_size) != 1:
        raise ValueError(
            "forward_backend='cuda_structured' supports single-rank execution only."
        )
    if not mesh_file:
        raise ValueError(
            "forward_backend='cuda_structured' requires a file-backed 3D mesh (.msh)."
        )
    if _norm(mesh_family) != "hex":
        raise ValueError(
            "forward_backend='cuda_structured' currently supports mesh_family='hex' only."
        )
    if _norm(geometry_version) != "geomv2":
        raise ValueError(
            "forward_backend='cuda_structured' currently supports geometry_version='geomv2' only."
        )
    resolved_revision = _norm(generator_revision) or DEFAULT_3D_GENERATOR_REVISION
    if resolved_revision != DEFAULT_3D_GENERATOR_REVISION:
        raise ValueError(
            "forward_backend='cuda_structured' currently supports "
            f"generator_revision={DEFAULT_3D_GENERATOR_REVISION!r} only."
        )
    if _norm(petsc_device_requested) != "cuda":
        raise ValueError(
            "forward_backend='cuda_structured' requires petsc_device='cuda' "
            "to make GPU selection explicit."
        )
    mesh_path = Path(mesh_file)
    if mesh_path.suffix.lower() != ".msh":
        raise ValueError(
            "forward_backend='cuda_structured' requires a Gmsh .msh mesh file."
        )
    if not _torch_cuda_available():
        raise RuntimeError(
            "forward_backend='cuda_structured' requires torch.cuda in the active runtime. "
            "Use `nix develop .#cuda` on this machine."
        )
    if meshio is None:
        raise RuntimeError(
            "forward_backend='cuda_structured' requires meshio for structured mesh metadata."
        )

    sidecar_path = structured_sidecar_path_for_mesh(mesh_path)
    if not sidecar_path.exists():
        raise ValueError(
            "forward_backend='cuda_structured' requires a structured sidecar "
            f"generated with the {DEFAULT_3D_GENERATOR_REVISION} hex mesh."
        )
    sidecar = load_structured_sidecar(sidecar_path)
    if _norm(sidecar.get("generator_revision")) != DEFAULT_3D_GENERATOR_REVISION:
        raise ValueError(
            "forward_backend='cuda_structured' requires a "
            f"{DEFAULT_3D_GENERATOR_REVISION} structured sidecar matching the mesh."
        )
    return {
        "forward_backend_requested": "cuda_structured",
        "forward_backend_effective": "cuda_structured",
        "mesh_file": str(mesh_path),
        "mesh_family": "hex",
        "geometry_version": "geomv2",
        "generator_revision": resolved_revision,
        "petsc_device_requested": "cuda",
        "petsc_device_effective": "cuda",
        "structured_sidecar_file": str(sidecar_path),
        "structured_sidecar_version": STRUCTURED_SIDECAR_VERSION,
        "structured_backend_version": CUDA_STRUCTURED_BACKEND_VERSION,
        "structured_sidecar_loaded": True,
        "operator_backend": "torch-cuda",
    }


@dataclass
class _CudaStructuredSigmaState:
    sigma_hash: str
    response_basis_gpu: Any
    schur_factor: tuple[np.ndarray, np.ndarray]
    pcg_iterations: int
    forward_reuse_state_hit: bool
    rhs_count: int


class CudaStructuredForwardBackend:
    """GPU batch-Schur backend for the narrow structured 3D hex contract."""

    def __init__(self, model, runtime: dict[str, Any]):
        self.model = model
        self.runtime = dict(runtime)
        self.device = torch.device("cuda")
        self.mesh_file = str(runtime["mesh_file"])
        self._sidecar_file = str(runtime["structured_sidecar_file"])
        self.sidecar = load_structured_sidecar(self._sidecar_file)
        self._structured_backend_version = str(runtime["structured_backend_version"])
        self._dof_bijection = self._build_dof_bijection()
        self._top_left_robin, self._coupling_columns, self._electrode_diag = (
            self._extract_cem_blocks()
        )
        self._mg_levels = self._estimate_mg_levels()
        self._sigma_state: _CudaStructuredSigmaState | None = None
        self._diagnostics: dict[str, object] = {
            "solve_mode": "cuda-structured-schur-pcg",
            "h1_solver": "pcg",
            "h1_preconditioner": "jacobi",
            "structured_backend_version": self._structured_backend_version,
            "structured_sidecar_loaded": True,
            "structured_sidecar_file": self._sidecar_file,
            "operator_backend": "torch-cuda",
            "mg_levels": self._mg_levels,
            "pcg_iterations": 0,
            "batched_rhs_count": 0,
            "forward_reuse_state_hit": False,
            "structured_dof_bijection_size": int(self._dof_bijection.size),
        }

    @staticmethod
    def _stable_hash(values: np.ndarray) -> str:
        arr = np.asarray(values, dtype=np.float64)
        return hash_array_payload(arr)

    def _estimate_mg_levels(self) -> int:
        dims = []
        for block in self.sidecar.get("blocks", []):
            logical = block.get("logical_cells", [1, 1, 1])
            if len(logical) >= 3:
                dims.append(min(int(logical[0]), int(logical[1]), int(logical[2])))
        if not dims:
            return 1
        min_dim = max(1, min(dims))
        levels = 1
        while min_dim >= 4:
            min_dim //= 2
            levels += 1
        return levels

    def _load_mesh_points(self) -> np.ndarray:
        mesh = meshio.read(self.mesh_file)
        return np.asarray(
            mesh.points[:, : self.model.mesh.geometry.dim], dtype=np.float64
        )

    def _build_dof_bijection(self) -> np.ndarray:
        structured_to_mesh = np.asarray(
            self.sidecar.get("structured_node_to_mesh_node", []),
            dtype=np.int32,
        )
        mesh_points = self._load_mesh_points()
        if structured_to_mesh.size == 0:
            raise RuntimeError(
                "cuda_structured sidecar is missing structured_node_to_mesh_node entries."
            )
        structured_points = mesh_points[structured_to_mesh]
        dolfinx_coords = np.asarray(
            self.model.V.tabulate_dof_coordinates()[:, : self.model.mesh.geometry.dim],
            dtype=np.float64,
        )
        if structured_points.shape != dolfinx_coords.shape:
            raise RuntimeError(
                "cuda_structured DOLFINx/structured node count mismatch: "
                f"structured={structured_points.shape}, dolfinx={dolfinx_coords.shape}"
            )
        tree = cKDTree(dolfinx_coords)
        dist, idx = tree.query(structured_points, k=1)
        if float(np.max(dist)) > 1e-10:
            raise RuntimeError(
                "cuda_structured DOLFINx dof mapping exceeds tolerance: "
                f"max={float(np.max(dist)):.3e}"
            )
        if np.unique(idx).size != idx.size:
            raise RuntimeError("cuda_structured DOLFINx dof mapping is not bijective.")
        return np.asarray(idx, dtype=np.int32)

    def _extract_cem_blocks(self) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
        full = self.model._ensure_electrode_matrix().tocsr()
        top_left = full[: self.model.dofs, : self.model.dofs].tocsr()
        coupling = np.asarray(
            full[
                : self.model.dofs, self.model.dofs : self.model.dofs + self.model.n_elec
            ].toarray(),
            dtype=np.float64,
        )
        electrode_diag = np.asarray(
            full[
                self.model.dofs : self.model.dofs + self.model.n_elec,
                self.model.dofs : self.model.dofs + self.model.n_elec,
            ].toarray(),
            dtype=np.float64,
        )
        return top_left, coupling, electrode_diag

    @staticmethod
    def _csr_to_torch(mat: csr_matrix, device) -> Any:
        csr = mat.tocsr()
        indptr = torch.as_tensor(csr.indptr.astype(np.int64, copy=False), device=device)
        indices = torch.as_tensor(
            csr.indices.astype(np.int64, copy=False), device=device
        )
        values = torch.as_tensor(csr.data.astype(np.float64, copy=False), device=device)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            return torch.sparse_csr_tensor(
                indptr,
                indices,
                values,
                size=csr.shape,
                device=device,
                dtype=torch.float64,
            )

    @staticmethod
    def _block_pcg(
        A,
        B,
        *,
        diag_inv,
        rtol: float = 1e-10,
        atol: float = 1e-12,
        max_it: int = 2000,
    ) -> tuple[Any, int]:
        X = torch.zeros_like(B)
        R = B.clone()
        Z = diag_inv * R
        P = Z.clone()
        rz_old = torch.sum(R * Z, dim=0)
        b_norm = torch.linalg.norm(B, dim=0)
        tol = torch.maximum(
            torch.full_like(b_norm, float(atol)),
            b_norm * float(rtol),
        )
        converged = torch.linalg.norm(R, dim=0) <= tol
        last_iter = 0

        for iteration in range(1, int(max_it) + 1):
            if bool(torch.all(converged).item()):
                last_iter = iteration - 1
                break
            AP = torch.sparse.mm(A, P)
            denom = torch.sum(P * AP, dim=0)
            safe = torch.where(
                torch.abs(denom) > 1e-30,
                denom,
                torch.ones_like(denom),
            )
            alpha = torch.where(converged, torch.zeros_like(rz_old), rz_old / safe)
            X = X + P * alpha.unsqueeze(0)
            R = R - AP * alpha.unsqueeze(0)
            residual_norm = torch.linalg.norm(R, dim=0)
            converged = converged | (residual_norm <= tol)
            if bool(torch.all(converged).item()):
                last_iter = iteration
                break
            Z = diag_inv * R
            rz_new = torch.sum(R * Z, dim=0)
            safe_old = torch.where(
                torch.abs(rz_old) > 1e-30,
                rz_old,
                torch.ones_like(rz_old),
            )
            beta = torch.where(converged, torch.zeros_like(rz_new), rz_new / safe_old)
            P = Z + P * beta.unsqueeze(0)
            rz_old = rz_new
            last_iter = iteration

        if not bool(torch.all(converged).item()):
            residual = torch.linalg.norm(R, dim=0).detach().cpu().numpy()
            raise RuntimeError(
                "cuda_structured PCG failed to converge within "
                f"{max_it} iterations (max residual={float(np.max(residual)):.3e})."
            )
        return X, int(last_iter)

    def _assemble_top_left_matrix(self, sigma_values: np.ndarray) -> csr_matrix:
        sigma = fem.Function(self.model.V_sigma)
        sigma.x.array[:] = np.asarray(sigma_values, dtype=np.float64)
        conductivity = self.model._petsc_to_csr(
            self.model._assemble_conductivity_matrix(sigma)
        )
        top_left = (conductivity + self._top_left_robin).tocsr()
        top_left = ((top_left + top_left.T) * 0.5).tocsr()
        top_left.eliminate_zeros()
        return top_left

    def _build_sigma_state(self, sigma_values: np.ndarray) -> _CudaStructuredSigmaState:
        sigma_hash = self._stable_hash(sigma_values)
        if self._sigma_state is not None and self._sigma_state.sigma_hash == sigma_hash:
            self._sigma_state.forward_reuse_state_hit = True
            self._diagnostics["forward_reuse_state_hit"] = True
            self._diagnostics["pcg_iterations"] = int(self._sigma_state.pcg_iterations)
            self._diagnostics["batched_rhs_count"] = int(self._sigma_state.rhs_count)
            return self._sigma_state

        top_left = self._assemble_top_left_matrix(sigma_values)
        diag = np.asarray(top_left.diagonal(), dtype=np.float64)
        if (
            diag.size != self.model.dofs
            or not all_finite_values(diag)
            or float(np.min(diag)) <= 0.0
        ):
            raise RuntimeError(
                "cuda_structured top-left system has an invalid diagonal."
            )

        A_gpu = self._csr_to_torch(top_left, self.device)
        coupling_gpu = torch.as_tensor(
            self._coupling_columns, device=self.device, dtype=torch.float64
        )
        diag_tensor = torch.as_tensor(diag, device=self.device, dtype=torch.float64)
        diag_inv = torch.reciprocal(diag_tensor).reshape(-1, 1)

        response_basis_gpu, pcg_iterations = self._block_pcg(
            A_gpu,
            coupling_gpu,
            diag_inv=diag_inv,
            rtol=1e-10,
            atol=1e-12,
            max_it=max(512, min(8192, 4 * int(self.model.dofs))),
        )

        response_basis_cpu = response_basis_gpu.detach().cpu().numpy()
        n_elec = self.model.n_elec
        schur = self._electrode_diag - (self._coupling_columns.T @ response_basis_cpu)
        schur_aug = np.zeros((n_elec + 1, n_elec + 1), dtype=np.float64)
        schur_aug[:n_elec, :n_elec] = schur
        schur_aug[:n_elec, n_elec] = 1.0
        schur_aug[n_elec, :n_elec] = 1.0
        schur_factor = lu_factor(schur_aug)

        state = _CudaStructuredSigmaState(
            sigma_hash=sigma_hash,
            response_basis_gpu=response_basis_gpu,
            schur_factor=schur_factor,
            pcg_iterations=int(pcg_iterations),
            forward_reuse_state_hit=False,
            rhs_count=n_elec,
        )
        self._sigma_state = state
        self._diagnostics.update(
            {
                "pcg_iterations": int(pcg_iterations),
                "batched_rhs_count": n_elec,
                "forward_reuse_state_hit": False,
            }
        )
        return state

    def backend_diagnostics(self) -> dict[str, object]:
        _RUNTIME_KEYS = (
            "structured_sidecar_version",
            "mesh_family",
            "geometry_version",
            "generator_revision",
            "petsc_device_requested",
            "petsc_device_effective",
        )
        result: dict[str, object] = {
            "forward_backend_requested": "cuda_structured",
            "forward_backend_effective": "cuda_structured",
        }
        result.update(self._diagnostics)
        for key in _RUNTIME_KEYS:
            result[key] = self.runtime.get(key)
        return result

    def solve_batch(
        self, sigma_values: np.ndarray, pattern_matrix: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        sigma_f64 = np.asarray(sigma_values, dtype=np.float64)
        state = self._build_sigma_state(sigma_f64)
        n_patterns = int(pattern_matrix.shape[0])

        rhs = np.zeros((self.model.n_elec + 1, n_patterns), dtype=np.float64)
        rhs[: self.model.n_elec, :] = np.asarray(pattern_matrix, dtype=np.float64).T

        schur_solution = lu_solve(state.schur_factor, rhs)
        electrode_block = np.asarray(
            schur_solution[: self.model.n_elec, :].T, dtype=np.float64
        )

        electrode_gpu = torch.as_tensor(
            electrode_block.T, device=self.device, dtype=torch.float64
        )
        potentials_gpu = -(state.response_basis_gpu @ electrode_gpu)
        potentials = (
            potentials_gpu.detach().cpu().numpy().astype(np.float64, copy=False)
        )

        self._diagnostics["batched_rhs_count"] = n_patterns
        self._diagnostics["forward_reuse_state_hit"] = bool(
            state.forward_reuse_state_hit
        )
        self._diagnostics["pcg_iterations"] = int(state.pcg_iterations)

        u_all = tuple(potentials[:, idx] for idx in range(n_patterns))
        return u_all, electrode_block
