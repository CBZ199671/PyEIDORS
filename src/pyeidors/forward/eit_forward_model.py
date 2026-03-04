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
        self.electrode_lengths = self._compute_electrode_lengths()
        self.geometry_scale_to_m = float(pattern_config.geometry_scale_to_m)
        self.electrode_lengths_m = resolve_electrode_lengths_m(
            electrode_lengths_mesh=[self.electrode_lengths[tag] for tag in self.electrode_tags],
            geometry_scale_to_m=self.geometry_scale_to_m,
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

    def _compute_electrode_lengths(self):
        """Compute boundary measure (length in 2D) for each electrode."""
        lengths = {}
        one = fem.Constant(self.mesh, 1.0)
        for tag in self.electrode_tags:
            length_local = fem.assemble_scalar(fem.form(one * self.ds_electrodes(tag)))
            length = self.mesh.comm.allreduce(length_local, op=MPI.SUM)
            lengths[tag] = float(length)
            if np.isclose(length, 0.0):
                warnings.warn(
                    f"Electrode boundary tag {tag} has zero measure, check mesh markers",
                    RuntimeWarning,
                )
        return lengths

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

    def _assemble_electrode_matrix(self):
        b_form = 0
        for i, electrode_tag in enumerate(self.electrode_tags):
            b_form += (self.geometry_scale_to_m / self.z[i]) * ufl.inner(self.u, self.phi) * self.ds_electrodes(
                electrode_tag
            )

        B = fem_petsc.assemble_matrix(fem.form(b_form))
        B.assemble()
        M = self._petsc_to_csr(B)
        M.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        M_lil = lil_matrix(M)

        for i, electrode_tag in enumerate(self.electrode_tags):
            c_form = (-self.geometry_scale_to_m / self.z[i]) * self.phi * self.ds_electrodes(electrode_tag)
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

    def create_full_matrix(self, sigma: fem.Function):
        """Build complete system matrix including conductivity term."""
        a_form = ufl.inner(sigma * ufl.grad(self.u), ufl.grad(self.phi)) * ufl.dx
        A = fem_petsc.assemble_matrix(fem.form(a_form))
        A.assemble()
        scipy_A = self._petsc_to_csr(A)
        scipy_A.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        return scipy_A + self.M

    def _sigma_fingerprint(self, sigma: fem.Function) -> str:
        values = np.ascontiguousarray(sigma.x.array, dtype=np.float64)
        return hashlib.sha256(values.tobytes()).hexdigest()

    def _base_cache_payload(self, sigma_hash: str, n_patterns: int) -> dict[str, object]:
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
            },
            "performance_mode": self.performance_mode,
        }

    def _solve_with_scipy(self, sigma: fem.Function, pattern_matrix: np.ndarray):
        n_patterns = pattern_matrix.shape[0]
        system_matrix = self.create_full_matrix(sigma).tocsc()
        sigma_hash = self._sigma_fingerprint(sigma)
        payload = self._base_cache_payload(sigma_hash=sigma_hash, n_patterns=n_patterns)
        payload["solver"] = "splu"

        if self.cache_manager is not None and self.cache_manager.enabled:
            lu, lookup = self.cache_manager.get_or_compute(
                artifact="forward_factor",
                payload=payload,
                compute_fn=lambda: splu(system_matrix),
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
            lu = splu(system_matrix)
            self._last_cache_lookup = {"hit": False, "layer": "disabled", "artifact": "forward_factor"}

        rhs_matrix = np.zeros((self.dofs + self.n_elec + 1, n_patterns), dtype=float)
        rhs_matrix[self.dofs : self.dofs + self.n_elec, :] = pattern_matrix.T
        sol_matrix = lu.solve(rhs_matrix)
        return np.asarray(sol_matrix, dtype=float)

    def _make_petsc_solver_bundle(self, system_matrix: csr_matrix):
        if PETSc is None:
            raise RuntimeError("petsc4py is required for linear_backend='petsc'")
        A = self._csr_to_petsc(system_matrix)
        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOperators(A)
        ksp.setType(self.backend_config.ksp_type)
        pc = ksp.getPC()
        pc.setType(self.backend_config.pc_type)
        ksp.setTolerances(
            rtol=self.backend_config.rtol,
            atol=self.backend_config.atol,
            max_it=self.backend_config.max_it,
        )
        if hasattr(ksp, "setReusePreconditioner"):
            ksp.setReusePreconditioner(bool(self.backend_config.reuse_preconditioner))
        if self.backend_config.monitor:
            ksp.setMonitor(lambda _ksp, its, rnorm: print(f"[KSP] iter={its} rnorm={rnorm:.3e}"))
        try:
            ksp.setUp()
        except Exception:
            # Augmented CEM matrices can miss structural diagonals for direct LU setup.
            # Fall back to an iterative PETSc solver while preserving backend='petsc'.
            ksp = PETSc.KSP().create(self.mesh.comm)
            ksp.setOperators(A)
            ksp.setType("gmres")
            ksp.getPC().setType("none")
            ksp.setTolerances(
                rtol=min(self.backend_config.rtol, 1e-12),
                atol=min(self.backend_config.atol, 1e-14),
                max_it=max(self.backend_config.max_it, 4000),
            )
            ksp.setUp()
        return {"A": A, "ksp": ksp}

    def _solve_with_petsc(self, sigma: fem.Function, pattern_matrix: np.ndarray):
        n_patterns = pattern_matrix.shape[0]
        system_matrix = self.create_full_matrix(sigma).tocsr()
        sigma_hash = self._sigma_fingerprint(sigma)
        payload = self._base_cache_payload(sigma_hash=sigma_hash, n_patterns=n_patterns)
        payload["solver"] = "petsc-ksp"

        if self.cache_manager is not None and self.cache_manager.enabled:
            bundle, lookup = self.cache_manager.get_or_compute(
                artifact="forward_factor",
                payload=payload,
                compute_fn=lambda: self._make_petsc_solver_bundle(system_matrix),
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
            bundle = self._make_petsc_solver_bundle(system_matrix)
            self._last_cache_lookup = {"hit": False, "layer": "disabled", "artifact": "forward_factor"}

        A = bundle["A"]
        ksp = bundle["ksp"]
        rhs_matrix = np.zeros((self.dofs + self.n_elec + 1, n_patterns), dtype=float)
        rhs_matrix[self.dofs : self.dofs + self.n_elec, :] = pattern_matrix.T

        sol_matrix = np.zeros_like(rhs_matrix)
        x = A.createVecRight()
        for i in range(n_patterns):
            b = PETSc.Vec().createWithArray(rhs_matrix[:, i], comm=self.mesh.comm)
            ksp.solve(b, x)
            if ksp.getConvergedReason() < 0:
                # Keep PETSc as default path, but fail over to SciPy for singular/ill-conditioned
                # edge cases to preserve runtime robustness and test stability.
                self._last_cache_lookup = {
                    "hit": False,
                    "layer": "compute",
                    "artifact": "forward_factor",
                    "fallback": "scipy",
                    "petsc_reason": int(ksp.getConvergedReason()),
                }
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
        u_all = [potential_block[:, i].copy() for i in range(n_patterns)]
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
