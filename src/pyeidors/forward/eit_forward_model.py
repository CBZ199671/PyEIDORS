"""EIT Forward Model based on the Complete Electrode Model (CEM)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import warnings

import numpy as np
import ufl
from mpi4py import MPI
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import splu
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc

try:  # pragma: no cover - petsc4py is available in nix runtime, optional in unit stubs
    from petsc4py import PETSc
except ImportError:  # pragma: no cover
    PETSc = None

from ..data.structures import EITData, EITImage, EITMesh, PatternConfig
from ..electrodes.patterns import StimMeasPatternManager
from ..femx import create_ds_measure
from ..physics.current_drive import resolve_electrode_lengths_m


@dataclass(frozen=True)
class LinearBackendConfig:
    """Linear solve configuration for forward model backends."""

    ksp_type: str = "preonly"
    pc_type: str = "lu"
    rtol: float = 1e-10
    atol: float = 1e-12
    max_it: int = 2000
    reuse_preconditioner: bool = True
    monitor: bool = False
    mat_solve_mode: str = "off"
    use_mat_solve: bool = False
    petsc_device: str = "auto"

    @classmethod
    def from_dict(cls, payload: dict | None) -> "LinearBackendConfig":
        if not payload:
            return cls()
        return cls(
            ksp_type=str(payload.get("ksp_type", "preonly")),
            pc_type=str(payload.get("pc_type", "lu")),
            rtol=float(payload.get("rtol", 1e-10)),
            atol=float(payload.get("atol", 1e-12)),
            max_it=int(payload.get("max_it", 2000)),
            reuse_preconditioner=bool(payload.get("reuse_preconditioner", True)),
            monitor=bool(payload.get("monitor", False)),
            mat_solve_mode=str(payload.get("mat_solve_mode", "off")),
            use_mat_solve=bool(payload.get("use_mat_solve", False)),
            petsc_device=str(payload.get("petsc_device", "auto")),
        )


class EITForwardModel:
    """EIT Forward Model - Complete Electrode Model assembly and solve."""

    def __init__(
        self,
        n_elec: int,
        pattern_config: PatternConfig,
        z: np.ndarray,
        mesh: EITMesh,
        linear_backend: str = "petsc",
        backend_config: dict | LinearBackendConfig | None = None,
        cache_manager=None,
        performance_mode: str = "aggressive",
    ):
        self.n_elec = n_elec
        self.z = np.asarray(z, dtype=float)
        if not isinstance(mesh, EITMesh):
            raise TypeError("EITForwardModel expects an EITMesh instance")
        self.eit_mesh = mesh
        self.mesh = mesh.mesh

        if self.mesh.comm.size != 1:
            raise RuntimeError(
                "PyEIDORS phase-2 migration currently supports MPI size=1 only. "
                "Use single-rank execution in this stage."
            )
        if self.z.size != self.n_elec:
            raise ValueError(
                f"Contact impedance length ({self.z.size}) does not match electrode count ({self.n_elec})"
            )

        self.linear_backend = str(linear_backend).strip().lower()
        self.backend_config = (
            backend_config
            if isinstance(backend_config, LinearBackendConfig)
            else LinearBackendConfig.from_dict(backend_config)
        )
        self.performance_mode = str(performance_mode).strip().lower()
        self.cache_manager = cache_manager
        self._last_cache_lookup: dict[str, str | bool] = {}

        self.facet_tags = mesh.facet_tags
        self.association_table = mesh.association_table
        if self.facet_tags is None:
            raise ValueError("EITMesh lacks electrode facet tags, cannot assemble CEM")

        self.ds_electrodes = create_ds_measure(self.mesh, self.facet_tags)

        self.electrode_tags = self._resolve_electrode_tags()
        self.electrode_boundary_measures = self._compute_electrode_boundary_measures()
        self.geometry_scale_to_m = float(pattern_config.geometry_scale_to_m)
        self.mesh_tdim = int(self.mesh.topology.dim)
        self.boundary_scale_to_m = self.geometry_scale_to_m ** max(1, self.mesh_tdim - 1)
        self.electrode_lengths_m = resolve_electrode_lengths_m(
            electrode_lengths_mesh=[
                self.electrode_boundary_measures[tag] for tag in self.electrode_tags
            ],
            geometry_scale_to_m=self.boundary_scale_to_m,
            electrode_length_m_override=pattern_config.electrode_length_m_override,
            n_elec=self.n_elec,
        )
        self.pattern_manager = StimMeasPatternManager(
            pattern_config,
            electrode_lengths_m=self.electrode_lengths_m,
            mesh_tdim=self.mesh.topology.dim,
        )

        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))
        self.V_sigma = fem.functionspace(self.mesh, ("DG", 0))
        dofmap = self.V.dofmap.index_map
        self.dofs = int(dofmap.size_local * self.V.dofmap.index_map_bs)
        self.u = ufl.TrialFunction(self.V)
        self.phi = ufl.TestFunction(self.V)
        self.M = self._assemble_electrode_matrix()
        self._M_petsc = {}
        self._petsc_backend_info = self._resolve_petsc_backend_info()

    def _resolve_pattern_matrix(self, current_patterns=None) -> np.ndarray:
        """Return stimulation matrix with shape ``(n_patterns, n_elec)``."""
        if current_patterns is None:
            matrix = np.asarray(self.pattern_manager.stim_matrix, dtype=float)
        else:
            matrix = np.asarray(current_patterns, dtype=float)
            if matrix.ndim != 2:
                raise ValueError("current_patterns must be a 2D array")
            if matrix.shape[1] == self.n_elec:
                pass
            elif matrix.shape[0] == self.n_elec:
                matrix = matrix.T
            else:
                raise ValueError(
                    "current_patterns shape mismatch. Expected (n_patterns, n_elec) "
                    f"or (n_elec, n_patterns), got {matrix.shape}"
                )

        if matrix.shape[1] != self.n_elec:
            raise ValueError(f"Pattern width mismatch: expected {self.n_elec}, got {matrix.shape[1]}")
        return matrix

    def _resolve_electrode_tags(self):
        """Extract boundary tags sorted by electrode index from association table."""
        electrode_map = {}
        if isinstance(self.association_table, dict):
            for key, val in self.association_table.items():
                try:
                    tag_val = int(val)
                except (TypeError, ValueError):
                    continue
                if isinstance(key, str):
                    key_lower = key.lower()
                    if key_lower == "electrodes" and isinstance(val, dict):
                        for idx_str, tag in val.items():
                            try:
                                electrode_map[int(idx_str)] = int(tag)
                            except (TypeError, ValueError):
                                continue
                        continue
                    if key_lower.startswith("electrode"):
                        idx_str = key_lower.split("_")[-1]
                        if idx_str.isdigit():
                            electrode_map[int(idx_str)] = tag_val

        if len(electrode_map) < self.n_elec and isinstance(self.association_table, dict):
            candidates = []
            for key, val in self.association_table.items():
                try:
                    tag_val = int(val)
                except (TypeError, ValueError):
                    continue
                if isinstance(key, (int, np.integer)) and key >= 2:
                    candidates.append(tag_val)
            if candidates:
                for idx, tag_val in enumerate(sorted(set(candidates))[: self.n_elec], start=1):
                    electrode_map.setdefault(idx, tag_val)

        missing = [i for i in range(1, self.n_elec + 1) if i not in electrode_map]
        if missing:
            raise ValueError(f"Association table missing electrode tags {missing}, cannot assemble CEM")
        return [electrode_map[i] for i in range(1, self.n_elec + 1)]

    def _compute_electrode_boundary_measures(self):
        """Compute electrode boundary measure (2D length / 3D area)."""
        measures = {}
        one = fem.Constant(self.mesh, 1.0)
        for tag in self.electrode_tags:
            measure_local = fem.assemble_scalar(fem.form(one * self.ds_electrodes(tag)))
            measure = self.mesh.comm.allreduce(measure_local, op=MPI.SUM)
            measures[tag] = float(measure)
            if np.isclose(measure, 0.0):
                warnings.warn(
                    f"Electrode boundary tag {tag} has zero measure, check mesh markers",
                    RuntimeWarning,
                )
        return measures

    @staticmethod
    def _petsc_to_csr(mat) -> csr_matrix:
        indptr, indices, values = mat.getValuesCSR()
        shape = mat.getSize()
        return csr_matrix((values, indices, indptr), shape=shape)

    @staticmethod
    def _csr_to_petsc(system_matrix: csr_matrix):
        if PETSc is None:
            raise RuntimeError("petsc4py is not available for linear_backend='petsc'")
        matrix = system_matrix.tocsr()
        indptr = matrix.indptr.astype(np.int32, copy=False)
        indices = matrix.indices.astype(np.int32, copy=False)
        values = matrix.data.astype(np.float64, copy=False)
        A = PETSc.Mat().createAIJ(size=matrix.shape, csr=(indptr, indices, values))
        A.assemblyBegin()
        A.assemblyEnd()
        return A

    @staticmethod
    def _normalize_petsc_device(value: object) -> str:
        device = str(value).strip().lower()
        return device if device in {"auto", "cpu", "cuda"} else "auto"

    @staticmethod
    def _actionable_cuda_guidance() -> str:
        return (
            "Enter `nix develop .#cuda`, verify with "
            "`python scripts/diagnostics/probe_petsc_cuda.py --require cuda`, and retry."
        )

    def _stable_cpu_petsc_types(self) -> tuple[str | None, str | None]:
        if PETSc is None:
            return None, None

        comm = getattr(getattr(self, "mesh", None), "comm", None)
        comm_size = 1
        try:
            if comm is not None and hasattr(comm, "Get_size"):
                comm_size = int(comm.Get_size())
            elif comm is not None and hasattr(comm, "size"):
                comm_size = int(comm.size)
        except Exception:
            comm_size = 1

        mat_namespace = getattr(getattr(PETSc, "Mat", None), "Type", None)
        vec_namespace = getattr(getattr(PETSc, "Vec", None), "Type", None)

        def _first_available(namespace, candidates, fallback):
            if namespace is not None:
                for candidate in candidates:
                    if hasattr(namespace, candidate):
                        return str(getattr(namespace, candidate))
            return fallback

        mat_type = _first_available(
            mat_namespace,
            ("MPIAIJ", "AIJ") if comm_size > 1 else ("SEQAIJ", "AIJ"),
            "mpiaij" if comm_size > 1 else "seqaij",
        )
        vec_type = _first_available(
            vec_namespace,
            ("MPI",) if comm_size > 1 else ("SEQ",),
            "mpi" if comm_size > 1 else "seq",
        )
        return mat_type, vec_type

    def _resolve_petsc_backend_info(self) -> dict[str, object]:
        requested = self._normalize_petsc_device(getattr(self.backend_config, "petsc_device", "auto"))
        info: dict[str, object] = {
            "petsc_device_requested": requested,
            "petsc_device_effective": "cpu",
            "petsc_mat_type": None,
            "petsc_vec_type": None,
            "petsc_dense_mat_type": None,
            "gpu_fallback_reason": None,
            "forward_factor_backend": self.linear_backend,
            "forward_mat_solve_effective": None,
            "capability": {},
        }
        if self.linear_backend != "petsc":
            return info
        if PETSc is None:
            info["gpu_fallback_reason"] = "petsc_unavailable"
            if requested == "cuda":
                raise RuntimeError(
                    "petsc_device='cuda' requires petsc4py/PETSc support. "
                    + self._actionable_cuda_guidance()
                )
            return info

        try:
            from ..perf.capabilities import probe_petsc_cuda_runtime
        except Exception as exc:
            info["gpu_fallback_reason"] = f"capability_probe_failed: {exc}"
            if requested == "cuda":
                raise RuntimeError(
                    "petsc_device='cuda' requires a successful PETSc CUDA capability probe. "
                    + self._actionable_cuda_guidance()
                ) from exc
            return info

        capability = probe_petsc_cuda_runtime()
        info["capability"] = capability
        cuda_available = bool(capability.get("petsc_cuda", False))
        if requested == "cpu":
            return info
        if requested == "cuda" and not cuda_available:
            reason = capability.get("errors", {}) if isinstance(capability.get("errors"), dict) else capability
            raise RuntimeError(
                "petsc_device='cuda' requested, but the current PETSc/DOLFINx runtime "
                f"does not provide usable CUDA Mat/Vec types: {reason}. "
                + self._actionable_cuda_guidance()
            )
        if requested in {"auto", "cuda"} and cuda_available:
            info["petsc_device_effective"] = "cuda"
            info["petsc_mat_type"] = capability.get("mat_type_name")
            info["petsc_vec_type"] = capability.get("vec_type_name")
            info["petsc_dense_mat_type"] = capability.get("dense_mat_type_name")
            return info
        if requested == "auto":
            info["gpu_fallback_reason"] = "petsc_cuda_not_available"
        return info

    def _get_requested_petsc_mat_type(self):
        info = getattr(self, "_petsc_backend_info", {}) or {}
        if info.get("petsc_device_effective") != "cuda" or PETSc is None:
            return None
        mat_type = info.get("petsc_mat_type")
        if mat_type:
            return mat_type
        mat_namespace = getattr(getattr(PETSc, "Mat", None), "Type", None)
        return str(getattr(mat_namespace, "AIJCUSPARSE")) if mat_namespace is not None and hasattr(mat_namespace, "AIJCUSPARSE") else None

    def _get_requested_dense_mat_type(self):
        info = getattr(self, "_petsc_backend_info", {}) or {}
        if info.get("petsc_device_effective") != "cuda" or PETSc is None:
            return None
        dense_type = info.get("petsc_dense_mat_type")
        if dense_type:
            return dense_type
        mat_namespace = getattr(getattr(PETSc, "Mat", None), "Type", None)
        return str(getattr(mat_namespace, "DENSECUDA")) if mat_namespace is not None and hasattr(mat_namespace, "DENSECUDA") else None

    def _get_requested_petsc_vec_type(self):
        info = getattr(self, "_petsc_backend_info", {}) or {}
        if info.get("petsc_device_effective") != "cuda" or PETSc is None:
            return None
        vec_type = info.get("petsc_vec_type")
        if vec_type:
            return vec_type
        vec_namespace = getattr(getattr(PETSc, "Vec", None), "Type", None)
        return str(getattr(vec_namespace, "CUDA")) if vec_namespace is not None and hasattr(vec_namespace, "CUDA") else None

    @staticmethod
    def _mat_type_key(mat_type) -> str:
        return str(mat_type).strip().lower() if mat_type is not None else "cpu"

    @staticmethod
    def _vec_to_numpy(vec) -> np.ndarray:
        if hasattr(vec, "array"):
            return np.asarray(vec.array, dtype=float)
        return np.asarray(vec.getArray(readonly=True), dtype=float)

    @staticmethod
    def _ensure_mat_type(mat, mat_type):
        if PETSc is None or mat is None or mat_type is None:
            return mat
        try:
            current_type = str(mat.getType()).strip().lower()
        except Exception:
            current_type = None
        target_type = str(mat_type).strip().lower()
        if current_type == target_type:
            return mat
        try:
            converted = mat.convert(mat_type)
        except Exception:
            converted = None
        if converted is not None:
            if converted is not mat and hasattr(mat, "destroy"):
                try:
                    mat.destroy()
                except Exception:
                    pass
            return converted
        try:
            mat.setType(mat_type)
        except Exception:
            pass
        return mat

    @staticmethod
    def _ensure_vec_type(vec, vec_type):
        if PETSc is None or vec is None or vec_type is None:
            return vec
        try:
            current_type = str(vec.getType()).strip().lower()
        except Exception:
            current_type = None
        target_type = str(vec_type).strip().lower()
        if current_type == target_type:
            return vec
        try:
            vec.setType(vec_type)
        except Exception:
            pass
        return vec

    @staticmethod
    def _is_gpu_petsc_kind(kind) -> bool:
        if kind is None:
            return False
        label = str(kind).strip().lower()
        return "cuda" in label or "cusparse" in label

    def _assemble_form_matrix(self, form_obj, *, mat_kind=None):
        assembly_kind = None if self._is_gpu_petsc_kind(mat_kind) else mat_kind
        if assembly_kind is None:
            mat = fem_petsc.assemble_matrix(form_obj)
        else:
            try:
                mat = fem_petsc.assemble_matrix(form_obj, kind=assembly_kind)
            except TypeError:
                mat = fem_petsc.assemble_matrix(form_obj)
        mat.assemble()
        return self._ensure_mat_type(mat, mat_kind)

    def _assemble_form_vector(self, form_obj, *, vec_kind=None):
        assembly_kind = None if self._is_gpu_petsc_kind(vec_kind) else vec_kind
        if assembly_kind is None:
            vec = fem_petsc.assemble_vector(form_obj)
        else:
            try:
                vec = fem_petsc.assemble_vector(form_obj, kind=assembly_kind)
            except TypeError:
                vec = fem_petsc.assemble_vector(form_obj)
        vec.assemble()
        return self._ensure_vec_type(vec, vec_kind)

    def _set_backend_diagnostic(self, **updates) -> None:
        info = dict(getattr(self, "_petsc_backend_info", {}) or {})
        info.update(updates)
        self._petsc_backend_info = info

    def _ensure_structural_diagonal(self, mat) -> None:
        if PETSc is None or mat is None:
            return
        try:
            if hasattr(mat, "setOption"):
                mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            nrows, ncols = mat.getSize()
            for idx in range(min(int(nrows), int(ncols))):
                mat.setValue(idx, idx, 0.0)
        except Exception:
            return

    def _gpu_gauge_fix_enabled(self) -> bool:
        return bool(getattr(self, "_petsc_backend_info", {}).get("petsc_device_effective") == "cuda")

    def _cuda_gauge_indices(self) -> tuple[int, int]:
        return self.dofs + self.n_elec, self.dofs

    def _apply_cuda_gauge_fix_matrix(self, mat):
        if PETSc is None or mat is None or not self._gpu_gauge_fix_enabled():
            return mat
        try:
            gauge_matrix = self._petsc_to_csr(mat).tolil()
            nrows = gauge_matrix.shape[0]
            for idx in self._cuda_gauge_indices():
                for row in range(nrows):
                    if row == idx:
                        continue
                    row_cols = gauge_matrix.rows[row]
                    if idx in row_cols:
                        pos = row_cols.index(idx)
                        row_cols.pop(pos)
                        gauge_matrix.data[row].pop(pos)
                gauge_matrix.rows[idx] = [idx]
                gauge_matrix.data[idx] = [1.0]
            fixed = self._csr_to_petsc(gauge_matrix.tocsr())
            self._set_backend_diagnostic(gpu_constraint_strategy="electrode-zero")
            if fixed is not mat and hasattr(mat, "destroy"):
                try:
                    mat.destroy()
                except Exception:
                    pass
            return fixed
        except Exception:
            return mat

    def _apply_cuda_gauge_fix_rhs(self, rhs_matrix: np.ndarray) -> np.ndarray:
        if not self._gpu_gauge_fix_enabled():
            return rhs_matrix
        last, gauge = self._cuda_gauge_indices()
        rhs_matrix[last, :] = 0.0
        rhs_matrix[gauge, :] = 0.0
        return rhs_matrix

    def _recenter_cuda_gauge_solution(self, sol_matrix: np.ndarray) -> np.ndarray:
        if not self._gpu_gauge_fix_enabled():
            return sol_matrix
        sol = np.asarray(sol_matrix, dtype=float).copy()
        electrode_block = sol[self.dofs : self.dofs + self.n_elec, :]
        offsets = electrode_block.mean(axis=0, keepdims=True)
        sol[: self.dofs, :] -= offsets
        sol[self.dofs : self.dofs + self.n_elec, :] -= offsets
        sol[self.dofs + self.n_elec, :] = 0.0
        return sol

    def _make_petsc_dense_solver_bundle(self, system_matrix):
        if PETSc is None:
            raise RuntimeError("petsc4py is required for CUDA dense fallback")
        dense_type = self._get_requested_dense_mat_type()
        if dense_type is None:
            raise RuntimeError("CUDA dense PETSc Mat type is unavailable")
        cpu_mat_type, _ = self._stable_cpu_petsc_types()
        host_mat = self._ensure_mat_type(system_matrix.copy(), cpu_mat_type)
        if hasattr(host_mat, "assemble"):
            host_mat.assemble()
        solve_mat = self._ensure_mat_type(host_mat, dense_type)
        if hasattr(solve_mat, "assemble"):
            solve_mat.assemble()
        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOperators(solve_mat)
        ksp.setType(self.backend_config.ksp_type)
        pc = ksp.getPC()
        pc.setType(self.backend_config.pc_type)
        ksp.setTolerances(
            rtol=self.backend_config.rtol,
            atol=self.backend_config.atol,
            max_it=self.backend_config.max_it,
        )
        if hasattr(ksp, "setReusePreconditioner"):
            try:
                ksp.setReusePreconditioner(bool(self.backend_config.reuse_preconditioner))
            except Exception:
                pass
        ksp.setUp()
        return {
            "A": system_matrix,
            "solve_A": solve_mat,
            "ksp": ksp,
            "backend": f"petsc-ksp-{str(dense_type).lower()}-{self.backend_config.pc_type}",
            "ksp_type": str(ksp.getType()) if hasattr(ksp, "getType") else self.backend_config.ksp_type,
            "pc_type": str(pc.getType()) if hasattr(pc, "getType") else self.backend_config.pc_type,
            "factor_solver_type": None,
            "solve_mat_type": str(solve_mat.getType()) if hasattr(solve_mat, "getType") else None,
        }

    def _assemble_electrode_matrix(self):
        b_form = 0
        for i, electrode_tag in enumerate(self.electrode_tags):
            b_form += (self.boundary_scale_to_m / self.z[i]) * ufl.inner(self.u, self.phi) * self.ds_electrodes(
                electrode_tag
            )

        B = fem_petsc.assemble_matrix(fem.form(b_form))
        B.assemble()
        M = self._petsc_to_csr(B)
        M.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        M_lil = lil_matrix(M)

        for i, electrode_tag in enumerate(self.electrode_tags):
            c_form = (-self.boundary_scale_to_m / self.z[i]) * self.phi * self.ds_electrodes(electrode_tag)
            C_vec = fem_petsc.assemble_vector(fem.form(c_form))
            C_vec.assemble()
            C_i = np.asarray(C_vec.array, dtype=float)

            M_lil[self.dofs + i, : self.dofs] = C_i
            M_lil[: self.dofs, self.dofs + i] = C_i
            electrode_len_m = float(self.electrode_lengths_m[i])
            M_lil[self.dofs + i, self.dofs + i] = (1.0 / self.z[i]) * electrode_len_m
            M_lil[self.dofs + self.n_elec, self.dofs + i] = 1.0
            M_lil[self.dofs + i, self.dofs + self.n_elec] = 1.0

        return csr_matrix(M_lil)

    def _assemble_conductivity_matrix(self, sigma: fem.Function, *, mat_kind=None):
        """Assemble the conductivity-dependent stiffness matrix in PETSc form."""
        a_form = ufl.inner(sigma * ufl.grad(self.u), ufl.grad(self.phi)) * ufl.dx
        return self._assemble_form_matrix(fem.form(a_form), mat_kind=mat_kind)

    def _create_full_matrix_scipy(self, sigma: fem.Function) -> csr_matrix:
        """Build full system matrix for SciPy backend."""
        scipy_A = self._petsc_to_csr(self._assemble_conductivity_matrix(sigma))
        scipy_A.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        return scipy_A + self.M

    def _assemble_electrode_matrix_petsc(self, *, mat_type=None, vec_type=None):
        if PETSc is None:
            raise RuntimeError("petsc4py is not available for linear_backend='petsc'")

        b_form = 0
        for i, electrode_tag in enumerate(self.electrode_tags):
            b_form += (self.boundary_scale_to_m / self.z[i]) * ufl.inner(self.u, self.phi) * self.ds_electrodes(
                electrode_tag
            )
        top_left = self._assemble_form_matrix(fem.form(b_form), mat_kind=mat_type)
        full_matrix = self._expand_conductivity_csr_to_full(top_left, mat_type=mat_type)

        for i, electrode_tag in enumerate(self.electrode_tags):
            c_form = (-self.boundary_scale_to_m / self.z[i]) * self.phi * self.ds_electrodes(electrode_tag)
            c_vec = self._assemble_form_vector(fem.form(c_form), vec_kind=vec_type)
            c_i = self._vec_to_numpy(c_vec)
            nz = np.flatnonzero(c_i)
            row = self.dofs + i
            if nz.size > 0:
                full_matrix.setValues(row, nz.astype(np.int32), c_i[nz])
                full_matrix.setValues(nz.astype(np.int32), row, c_i[nz])
            electrode_len_m = float(self.electrode_lengths_m[i])
            full_matrix.setValue(row, row, (1.0 / self.z[i]) * electrode_len_m)
            full_matrix.setValue(self.dofs + self.n_elec, row, 1.0)
            full_matrix.setValue(row, self.dofs + self.n_elec, 1.0)
            if hasattr(c_vec, "destroy"):
                try:
                    c_vec.destroy()
                except Exception:
                    pass

        full_matrix.assemblyBegin()
        full_matrix.assemblyEnd()
        return self._ensure_mat_type(full_matrix, mat_type)

    def _get_electrode_matrix_petsc(self, mat_type=None):
        if PETSc is None:
            raise RuntimeError("petsc4py is not available for linear_backend='petsc'")
        key = self._mat_type_key(mat_type)
        if key not in self._M_petsc:
            electrode_matrix = self._csr_to_petsc(self.M)
            ground_row = self.dofs + self.n_elec
            try:
                if hasattr(electrode_matrix, "setOption"):
                    electrode_matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
                electrode_matrix.setValue(ground_row, ground_row, 0.0)
            except Exception:
                pass
            electrode_matrix = self._ensure_mat_type(electrode_matrix, mat_type)
            if hasattr(electrode_matrix, "assemble"):
                electrode_matrix.assemble()
            self._M_petsc[key] = electrode_matrix
        return self._M_petsc[key]

    def _expand_conductivity_csr_to_full(self, conductivity_mat, *, mat_type=None):
        indptr, indices, values = conductivity_mat.getValuesCSR()
        full_size = self.dofs + self.n_elec + 1
        full_indptr = np.empty(full_size + 1, dtype=np.int32)
        local_indptr = np.asarray(indptr, dtype=np.int32)
        full_indptr[: self.dofs + 1] = local_indptr
        full_indptr[self.dofs + 1 :] = int(local_indptr[-1])
        augmented = PETSc.Mat().createAIJ(
            size=(full_size, full_size),
            csr=(
                full_indptr,
                np.asarray(indices, dtype=np.int32),
                np.asarray(values, dtype=np.float64),
            ),
            comm=self.mesh.comm,
        )
        augmented = self._ensure_mat_type(augmented, mat_type)
        augmented.assemblyBegin()
        augmented.assemblyEnd()
        return augmented

    def _create_full_matrix_petsc(self, sigma: fem.Function):
        """Build full system matrix for PETSc backend without SciPy round-trips."""
        if PETSc is None:
            raise RuntimeError("petsc4py is not available for linear_backend='petsc'")
        mat_kind = self._get_requested_petsc_mat_type()
        if self._gpu_gauge_fix_enabled():
            scipy_full = self._create_full_matrix_scipy(sigma).tolil()
            for idx in self._cuda_gauge_indices():
                scipy_full[idx, :] = 0.0
                scipy_full[:, idx] = 0.0
                scipy_full[idx, idx] = 1.0
            full_matrix = self._csr_to_petsc(scipy_full.tocsr())
            self._set_backend_diagnostic(gpu_constraint_strategy="electrode-zero")
        else:
            conductivity_mat = self._assemble_conductivity_matrix(sigma, mat_kind=None)
            conductivity_augmented = self._expand_conductivity_csr_to_full(conductivity_mat, mat_type=None)
            full_matrix = self._get_electrode_matrix_petsc(mat_type=None).copy()
            full_matrix.axpy(1.0, conductivity_augmented, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            self._ensure_structural_diagonal(full_matrix)
            full_matrix.assemblyBegin()
            full_matrix.assemblyEnd()
            conductivity_augmented.destroy()
        full_matrix = self._ensure_mat_type(full_matrix, mat_kind)
        if hasattr(full_matrix, "assemble"):
            full_matrix.assemble()
        self._set_backend_diagnostic(
            petsc_mat_type=str(full_matrix.getType()) if hasattr(full_matrix, "getType") else mat_kind,
        )
        return full_matrix

    def create_full_matrix(self, sigma: fem.Function):
        """Build complete system matrix including conductivity term."""
        return self._create_full_matrix_scipy(sigma)

    def _sigma_fingerprint(self, sigma: fem.Function) -> str:
        values = np.ascontiguousarray(sigma.x.array, dtype=np.float64)
        return hashlib.sha256(values.tobytes()).hexdigest()

    def _predict_forward_mat_solve_effective(self, n_patterns: int) -> str:
        backend_cfg = getattr(self, "backend_config", None)
        if backend_cfg is None:
            mat_mode = "on"
        else:
            mat_mode = str(getattr(backend_cfg, "mat_solve_mode", "")).strip().lower()
            if mat_mode not in {"auto", "on", "off"}:
                mat_mode = "on" if bool(getattr(backend_cfg, "use_mat_solve", False)) else "off"

        if mat_mode == "on":
            use_mat_solve = True
        elif mat_mode == "off":
            use_mat_solve = False
        else:
            use_mat_solve = bool(self.mesh_tdim == 3 and n_patterns > 1 and self.performance_mode == "aggressive")

        backend_info = getattr(self, "_petsc_backend_info", {}) or {}
        effective_device = str(backend_info.get("petsc_device_effective", "cpu"))
        capability = backend_info.get("capability") if isinstance(backend_info.get("capability"), dict) else {}
        if effective_device == "cuda" and n_patterns > 1 and bool(capability.get("petsc_cuda_dense", False)):
            use_mat_solve = True
        if effective_device == "cuda" and use_mat_solve and not bool(capability.get("petsc_cuda_dense", False)):
            use_mat_solve = False
        return "matsolve" if use_mat_solve else "vec-loop"

    def _base_cache_payload(self, sigma_hash: str, n_patterns: int) -> dict[str, object]:
        petsc_backend = getattr(self, "_petsc_backend_info", {}) or {}
        effective_device = str(petsc_backend.get("petsc_device_effective", "cpu"))
        mat_type = petsc_backend.get("petsc_mat_type")
        vec_type = petsc_backend.get("petsc_vec_type")
        if effective_device == "cpu" and (mat_type is None or vec_type is None):
            stable_mat_type, stable_vec_type = self._stable_cpu_petsc_types()
            mat_type = mat_type or stable_mat_type or "cpu-default"
            vec_type = vec_type or stable_vec_type or "cpu-default"
        mat_solve_effective = petsc_backend.get("forward_mat_solve_effective") or self._predict_forward_mat_solve_effective(n_patterns)

        return {
            "backend": self.linear_backend,
            "sigma_hash": sigma_hash,
            "n_elec": self.n_elec,
            "n_patterns": n_patterns,
            "z_hash": hashlib.sha256(np.ascontiguousarray(self.z, dtype=np.float64).tobytes()).hexdigest(),
            "pattern_hash": hashlib.sha256(
                np.ascontiguousarray(self.pattern_manager.stim_matrix, dtype=np.float64).tobytes()
            ).hexdigest(),
            "backend_config": {
                "ksp_type": self.backend_config.ksp_type,
                "pc_type": self.backend_config.pc_type,
                "rtol": self.backend_config.rtol,
                "atol": self.backend_config.atol,
                "max_it": self.backend_config.max_it,
                "reuse_preconditioner": self.backend_config.reuse_preconditioner,
                "mat_solve_mode": self.backend_config.mat_solve_mode,
                "use_mat_solve": self.backend_config.use_mat_solve,
                "petsc_device": self.backend_config.petsc_device,
            },
            "petsc_backend": {
                "requested": petsc_backend.get("petsc_device_requested", "auto"),
                "effective": effective_device,
                "mat_type": mat_type,
                "vec_type": vec_type,
                "mat_solve_effective": mat_solve_effective,
            },
            "performance_mode": self.performance_mode,
        }

    def _solve_with_scipy(self, sigma: fem.Function, pattern_matrix: np.ndarray):
        n_patterns = pattern_matrix.shape[0]
        sigma_hash = self._sigma_fingerprint(sigma)
        payload = self._base_cache_payload(sigma_hash=sigma_hash, n_patterns=n_patterns)
        payload["solver"] = "splu"

        if self.cache_manager is not None and self.cache_manager.enabled:
            lu, lookup = self.cache_manager.get_or_compute(
                artifact="forward_factor",
                payload=payload,
                compute_fn=lambda: splu(self._create_full_matrix_scipy(sigma).tocsc()),
                persist=False,
                cost=16.0,
            )
            self._last_cache_lookup = {
                "key": lookup.key,
                "hit": lookup.hit,
                "layer": lookup.layer,
                "artifact": lookup.artifact,
            }
        else:
            system_matrix = self._create_full_matrix_scipy(sigma).tocsc()
            lu = splu(system_matrix)
            self._last_cache_lookup = {"hit": False, "layer": "disabled", "artifact": "forward_factor"}

        rhs_matrix = np.zeros((self.dofs + self.n_elec + 1, n_patterns), dtype=float)
        rhs_matrix[self.dofs : self.dofs + self.n_elec, :] = pattern_matrix.T
        rhs_matrix = self._apply_cuda_gauge_fix_rhs(rhs_matrix)
        sol_matrix = lu.solve(rhs_matrix)
        return np.asarray(sol_matrix, dtype=float)

    def _make_petsc_solver_bundle(self, system_matrix):
        if PETSc is None:
            raise RuntimeError("petsc4py is required for linear_backend='petsc'")
        if isinstance(system_matrix, csr_matrix):
            A = self._csr_to_petsc(system_matrix)
        else:
            A = system_matrix

        cuda_enabled = bool(
            getattr(self, "_petsc_backend_info", {}).get("petsc_device_effective") == "cuda"
        )
        requested_ksp_type = self.backend_config.ksp_type
        requested_pc_type = self.backend_config.pc_type

        def _configure(ksp_obj, mat_obj, *, factor_backend=None):
            ksp_obj.setOperators(mat_obj)
            ksp_obj.setType(requested_ksp_type)
            pc_obj = ksp_obj.getPC()
            pc_obj.setType(requested_pc_type)
            if factor_backend is not None and hasattr(pc_obj, "setFactorSolverType"):
                pc_obj.setFactorSolverType(factor_backend)
            ksp_obj.setTolerances(
                rtol=self.backend_config.rtol,
                atol=self.backend_config.atol,
                max_it=self.backend_config.max_it,
            )
            if hasattr(ksp_obj, "setReusePreconditioner"):
                try:
                    ksp_obj.setReusePreconditioner(bool(self.backend_config.reuse_preconditioner))
                except Exception:
                    pass
            if self.backend_config.monitor:
                ksp_obj.setMonitor(lambda _ksp, its, rnorm: print(f"[KSP] iter={its} rnorm={rnorm:.3e}"))
            return pc_obj

        def _bundle_from(ksp_obj, solve_mat_obj, *, backend_name, factor_solver_type=None):
            pc_final = ksp_obj.getPC()
            self._set_backend_diagnostic(
                forward_factor_backend=backend_name,
                petsc_mat_type=str(A.getType()) if hasattr(A, "getType") else getattr(self, "_petsc_backend_info", {}).get("petsc_mat_type"),
            )
            return {
                "A": A,
                "solve_A": solve_mat_obj,
                "ksp": ksp_obj,
                "backend": backend_name,
                "ksp_type": str(ksp_obj.getType()) if hasattr(ksp_obj, "getType") else requested_ksp_type,
                "pc_type": str(pc_final.getType()) if hasattr(pc_final, "getType") else requested_pc_type,
                "factor_solver_type": factor_solver_type,
                "solve_mat_type": str(solve_mat_obj.getType()) if hasattr(solve_mat_obj, "getType") else None,
            }

        direct_pc = requested_pc_type in {"lu", "cholesky"}
        if cuda_enabled and direct_pc:
            last_error = None
            for candidate in ("cusparse", "cuda"):
                try:
                    ksp = PETSc.KSP().create(self.mesh.comm)
                    _configure(ksp, A, factor_backend=candidate)
                    ksp.setUp()
                    return _bundle_from(
                        ksp,
                        A,
                        backend_name=f"petsc-ksp-{candidate}-{requested_pc_type}",
                        factor_solver_type=candidate,
                    )
                except Exception as exc:
                    last_error = exc
            dense_type = self._get_requested_dense_mat_type()
            if dense_type is not None:
                try:
                    solve_mat = self._ensure_mat_type(A.copy(), dense_type)
                    if hasattr(solve_mat, "assemble"):
                        solve_mat.assemble()
                    ksp = PETSc.KSP().create(self.mesh.comm)
                    _configure(ksp, solve_mat)
                    ksp.setUp()
                    return _bundle_from(
                        ksp,
                        solve_mat,
                        backend_name=f"petsc-ksp-{str(dense_type).lower()}-{requested_pc_type}",
                        factor_solver_type=None,
                    )
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                setup_error = last_error
            else:
                setup_error = RuntimeError("unknown PETSc CUDA direct setup failure")
        else:
            setup_error = None
            try:
                ksp = PETSc.KSP().create(self.mesh.comm)
                _configure(ksp, A)
                ksp.setUp()
                return _bundle_from(ksp, A, backend_name="petsc-ksp", factor_solver_type=None)
            except Exception as exc:
                setup_error = exc

        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOperators(A)
        ksp.setType("gmres")
        fallback_pc = "none"
        ksp.getPC().setType(fallback_pc)
        ksp.setTolerances(
            rtol=min(self.backend_config.rtol, 1e-12),
            atol=min(self.backend_config.atol, 1e-14),
            max_it=max(self.backend_config.max_it, 4000),
        )
        ksp.setUp()
        return _bundle_from(ksp, A, backend_name=f"petsc-ksp-gmres+{fallback_pc}", factor_solver_type=None)


    def _solve_with_petsc(self, sigma: fem.Function, pattern_matrix: np.ndarray):
        n_patterns = pattern_matrix.shape[0]
        sigma_hash = self._sigma_fingerprint(sigma)
        payload = self._base_cache_payload(sigma_hash=sigma_hash, n_patterns=n_patterns)
        payload["solver"] = "petsc-ksp"

        if self.cache_manager is not None and self.cache_manager.enabled:
            bundle, lookup = self.cache_manager.get_or_compute(
                artifact="forward_factor",
                payload=payload,
                compute_fn=lambda: self._make_petsc_solver_bundle(self._create_full_matrix_petsc(sigma)),
                persist=False,
                cost=24.0,
            )
            self._last_cache_lookup = {
                "key": lookup.key,
                "hit": lookup.hit,
                "layer": lookup.layer,
                "artifact": lookup.artifact,
            }
        else:
            system_matrix = self._create_full_matrix_petsc(sigma)
            bundle = self._make_petsc_solver_bundle(system_matrix)
            self._last_cache_lookup = {"hit": False, "layer": "disabled", "artifact": "forward_factor"}

        A = bundle["A"]
        solve_A = bundle.get("solve_A", A)
        ksp = bundle["ksp"]
        self._set_backend_diagnostic(
            forward_factor_backend=bundle.get("backend", "petsc-ksp"),
            petsc_mat_type=str(A.getType()) if hasattr(A, "getType") else getattr(self, "_petsc_backend_info", {}).get("petsc_mat_type"),
        )

        rhs_matrix = np.zeros((self.dofs + self.n_elec + 1, n_patterns), dtype=float)
        rhs_matrix[self.dofs : self.dofs + self.n_elec, :] = pattern_matrix.T
        rhs_matrix = self._apply_cuda_gauge_fix_rhs(rhs_matrix)

        backend_cfg = getattr(self, "backend_config", None)
        if backend_cfg is None:
            mat_mode = "on"
        else:
            mat_mode = str(getattr(backend_cfg, "mat_solve_mode", "")).strip().lower()
            if mat_mode not in {"auto", "on", "off"}:
                mat_mode = "on" if bool(getattr(backend_cfg, "use_mat_solve", False)) else "off"
        if mat_mode == "on":
            use_mat_solve = True
        elif mat_mode == "off":
            use_mat_solve = False
        else:
            use_mat_solve = bool(self.mesh_tdim == 3 and n_patterns > 1 and self.performance_mode == "aggressive")

        solve_mat_type = str(bundle.get("solve_mat_type") or "").strip().lower()
        if "dense" in solve_mat_type and hasattr(ksp, "matSolve"):
            use_mat_solve = True

        backend_info = getattr(self, "_petsc_backend_info", {}) or {}
        requested_device = str(backend_info.get("petsc_device_requested", "auto"))
        effective_device = str(backend_info.get("petsc_device_effective", "cpu"))
        capability = backend_info.get("capability") if isinstance(backend_info.get("capability"), dict) else {}
        dense_mat_type = self._get_requested_dense_mat_type()
        if effective_device == "cuda" and n_patterns > 1 and bool(capability.get("petsc_cuda_dense", False)) and hasattr(ksp, "matSolve"):
            use_mat_solve = True
        if effective_device == "cuda" and use_mat_solve and not bool(capability.get("petsc_cuda_dense", False)):
            use_mat_solve = False
            self._set_backend_diagnostic(
                gpu_fallback_reason="petsc_densecuda_unavailable",
                forward_mat_solve_effective="vec-loop",
            )

        if use_mat_solve and hasattr(ksp, "matSolve"):
            try:
                B = PETSc.Mat().createDense(
                    size=rhs_matrix.shape,
                    array=np.asfortranarray(rhs_matrix, dtype=np.float64),
                    comm=self.mesh.comm,
                )
                B = self._ensure_mat_type(B, dense_mat_type)
                X = PETSc.Mat().createDense(
                    size=rhs_matrix.shape,
                    comm=self.mesh.comm,
                )
                X = self._ensure_mat_type(X, dense_mat_type)
                ksp.matSolve(B, X)
                sol = np.array(X.getDenseArray(), dtype=float, copy=True)
                self._set_backend_diagnostic(
                    forward_factor_backend=f"{bundle.get('backend', 'petsc-ksp')}:matsolve",
                    petsc_dense_mat_type=str(B.getType()) if hasattr(B, "getType") else dense_mat_type,
                    forward_mat_solve_effective="matsolve",
                )
                B.destroy()
                X.destroy()
                return self._recenter_cuda_gauge_solution(sol)
            except Exception as exc:
                if effective_device == "cuda" and bool(capability.get("petsc_cuda_dense", False)):
                    dense_bundle = bundle.get("_dense_cuda_fallback")
                    if dense_bundle is None:
                        try:
                            dense_bundle = self._make_petsc_dense_solver_bundle(A)
                            bundle["_dense_cuda_fallback"] = dense_bundle
                        except Exception:
                            dense_bundle = None
                    if dense_bundle is not None and dense_bundle.get("backend") != bundle.get("backend"):
                        dense_ksp = dense_bundle["ksp"]
                        try:
                            B = PETSc.Mat().createDense(
                                size=rhs_matrix.shape,
                                array=np.asfortranarray(rhs_matrix, dtype=np.float64),
                                comm=self.mesh.comm,
                            )
                            B = self._ensure_mat_type(B, dense_mat_type)
                            X = PETSc.Mat().createDense(size=rhs_matrix.shape, comm=self.mesh.comm)
                            X = self._ensure_mat_type(X, dense_mat_type)
                            dense_ksp.matSolve(B, X)
                            sol = np.array(X.getDenseArray(), dtype=float, copy=True)
                            self._set_backend_diagnostic(
                                gpu_fallback_reason=f"matSolve_fallback:{exc}",
                                forward_factor_backend=f"{dense_bundle.get('backend', 'petsc-ksp')}:matsolve",
                                petsc_dense_mat_type=str(B.getType()) if hasattr(B, "getType") else dense_mat_type,
                                forward_mat_solve_effective="matsolve",
                            )
                            B.destroy()
                            X.destroy()
                            return self._recenter_cuda_gauge_solution(sol)
                        except Exception:
                            pass
                if effective_device == "cuda" and requested_device == "cuda":
                    raise RuntimeError(
                        f"PETSc CUDA matSolve failed ({exc}). {self._actionable_cuda_guidance()}"
                    ) from exc
                self._set_backend_diagnostic(
                    gpu_fallback_reason=f"matSolve_failed: {exc}",
                    forward_mat_solve_effective="vec-loop",
                )

        self._set_backend_diagnostic(forward_mat_solve_effective="vec-loop")
        sol_matrix = np.zeros_like(rhs_matrix)
        b = self._ensure_vec_type(solve_A.createVecRight(), self._get_requested_petsc_vec_type())
        x = self._ensure_vec_type(solve_A.createVecRight(), self._get_requested_petsc_vec_type())
        if hasattr(x, "getType"):
            self._set_backend_diagnostic(petsc_vec_type=str(x.getType()))
        b_array = b.getArray(readonly=False)
        for i in range(n_patterns):
            b_array[:] = rhs_matrix[:, i]
            ksp.solve(b, x)
            if ksp.getConvergedReason() < 0:
                reason = int(ksp.getConvergedReason())
                self._last_cache_lookup = {
                    "hit": False,
                    "layer": "compute",
                    "artifact": "forward_factor",
                    "petsc_reason": reason,
                }
                if effective_device == "cuda":
                    self._set_backend_diagnostic(
                        gpu_fallback_reason=f"petsc_ksp_failed:{reason}",
                        forward_mat_solve_effective="vec-loop",
                    )
                    raise RuntimeError(
                        "PETSc CUDA solve failed with a negative convergence reason "
                        f"({reason}). {self._actionable_cuda_guidance()}"
                    )
                return self._solve_with_scipy(sigma, pattern_matrix)
            sol_matrix[:, i] = x.getArray(readonly=True)
        return sol_matrix

    def forward_solve(self, sigma: fem.Function, current_patterns=None):
        """Forward solve for given conductivity and stimulation patterns."""
        pattern_matrix = self._resolve_pattern_matrix(current_patterns)
        if self.linear_backend == "scipy":
            sol_matrix = self._solve_with_scipy(sigma, pattern_matrix)
        elif self.linear_backend == "petsc":
            sol_matrix = self._solve_with_petsc(sigma, pattern_matrix)
        else:
            raise ValueError(
                f"Unsupported linear_backend: {self.linear_backend}. "
                "Expected one of: 'petsc', 'scipy'."
            )

        n_patterns = pattern_matrix.shape[0]
        potential_block = np.asarray(sol_matrix[: self.dofs, :], dtype=float)
        electrode_block = np.asarray(
            sol_matrix[self.dofs : self.dofs + self.n_elec, :].T,
            dtype=float,
        )
        u_views = []
        for i in range(n_patterns):
            column = potential_block[:, i]
            column.setflags(write=False)
            u_views.append(column)
        u_all = tuple(u_views)
        return u_all, electrode_block

    def fwd_solve(self, img: EITImage):
        """Forward solve interface for ``EITImage``."""
        sigma = fem.Function(self.V_sigma)
        sigma.x.array[:] = img.get_conductivity()
        u_all, U_all = self.forward_solve(sigma)
        meas = self.pattern_manager.apply_meas_pattern(U_all)
        data = EITData(
            meas=meas,
            stim_pattern=self.pattern_manager.stim_matrix,
            n_elec=self.n_elec,
            n_stim=self.pattern_manager.n_stim,
            n_meas=self.pattern_manager.n_meas_total,
            type="simulated",
        )
        return data, U_all
