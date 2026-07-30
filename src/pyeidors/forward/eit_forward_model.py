"""EIT forward model with native complete- and point-electrode formulations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
import hashlib
import json
import time
from typing import Any, Mapping, Sequence
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
from ..cache.keys import update_digest_with_array_payload
from ..electrodes.patterns import StimMeasPatternManager
from ..femx import create_ds_measure
from ..interop.bridge_v3 import ElectrodeSpec
from ..utils.numeric_ops import (
    all_finite_values,
    has_nonzero_imaginary as _array_has_nonzero_imaginary,
)
from .cuda_structured_backend import (
    CudaStructuredForwardBackend,
    resolve_cuda_structured_runtime,
)
from .complex_support import petsc_scalar_dtype, runtime_scalar_summary
from .process_setup_cache import (
    ForwardStaticSetupBundle,
    build_process_forward_setup_key,
    get_process_forward_setup_bundle,
    put_process_forward_setup_bundle,
)
from ..perf.forward_solver_policy import (
    CUDA_HYPRE_BLACKLIST_REASON,
    is_hypre_cuda_blacklisted_solver,
)
from ..perf.policy import DEFAULT_FORWARD_BACKEND, normalize_forward_backend
from ..physics.current_drive import resolve_electrode_lengths_m


_NATIVE_AMGX_PRESETS = frozenset({"amgx", "cuda_amgx", "complex_cuda_amgx"})
_COMPLEX_BLOCK_REAL_AMGX_PRESETS = frozenset({"complex_block_real_amgx"})
_EXPLICIT_AMGX_PRESETS = _NATIVE_AMGX_PRESETS | _COMPLEX_BLOCK_REAL_AMGX_PRESETS
_NATIVE_COMPLEX_AMGX_CAVEAT = (
    "native complex PETSc PCAMGX is experimental for complex-admittance EIT; "
    "benchmarks found numerical differences versus CPU direct reference, so use "
    "it only for controlled parity experiments."
)
_COMPLEX_BLOCK_REAL_AMGX_CAVEAT = (
    "complex block-real AmgX solves the exported complex CEM system as a real "
    "2x2 block system in the real cuda-amgx runtime; use it for strict residual "
    "checks because large-grid benchmarks show it can be much slower than native "
    "complex64 CUDA GAMG."
)


@dataclass(frozen=True)
class _ForwardElectrodeSpec:
    """Zero-based electrode semantics used by the assembled forward operator."""

    kind: str
    source_nodes: tuple[int, ...] = ()
    node_weights: tuple[float | complex, ...] = ()
    boundary_kind: str = "exterior"
    contact_impedance: float | complex | None = None
    contact_impedance_present: bool = False


def _solver_route_metadata(solver_preset: str) -> dict[str, object]:
    preset = EITForwardModel._solver_token(solver_preset)
    if preset in _COMPLEX_BLOCK_REAL_AMGX_PRESETS:
        return {
            "solver_route_family": "complex_block_real_amgx",
            "solver_route_status": "strict_accuracy_complex_gpu",
            "solver_route_caveat": _COMPLEX_BLOCK_REAL_AMGX_CAVEAT,
        }
    if preset == "complex_cuda_amgx":
        return {
            "solver_route_family": "native_complex_amgx",
            "solver_route_status": "experimental_known_numeric_delta",
            "solver_route_caveat": _NATIVE_COMPLEX_AMGX_CAVEAT,
        }
    if preset in {"amgx", "cuda_amgx"}:
        return {
            "solver_route_family": "real_amgx",
            "solver_route_status": "real_gpu_candidate",
            "solver_route_caveat": "",
        }
    return {
        "solver_route_family": preset or "auto",
        "solver_route_status": "standard",
        "solver_route_caveat": "",
    }


def _nonzero_index_value_arrays(
    values: np.ndarray,
    *,
    index_dtype=np.int32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-zero indices and matching values without advanced indexing."""
    arr = np.asarray(values).reshape(-1)
    count = 0
    for raw_value in np.nditer(arr, flags=["refs_ok"], op_flags=["readonly"]):
        if raw_value.item() != 0:
            count += 1

    indices = np.empty(count, dtype=index_dtype)
    out_values = np.empty(count, dtype=arr.dtype)
    out_pos = 0
    for idx, raw_value in enumerate(
        np.nditer(arr, flags=["refs_ok"], op_flags=["readonly"])
    ):
        value = raw_value.item()
        if value == 0:
            continue
        indices[out_pos] = idx
        out_values[out_pos] = value
        out_pos += 1
    return indices, out_values


@dataclass(frozen=True)
class LinearBackendConfig:
    """Linear solve configuration for forward model backends."""

    solver_preset: str = "auto"
    ksp_type: str = "auto"
    pc_type: str = "auto"
    rtol: float = 1e-10
    atol: float = 1e-12
    max_it: int = 2000
    reuse_preconditioner: bool = True
    monitor: bool = False
    mat_solve_mode: str = "off"
    use_mat_solve: bool = False
    petsc_device: str = "auto"
    pc_factor_mat_solver_type: str | None = None
    pc_hypre_type: str | None = None
    pc_gamg_type: str | None = None
    petsc_options: dict[str, object] = field(default_factory=dict)
    forward_pc_refresh_policy: str = "auto"
    forward_pc_refresh_iter_threshold: int = 0
    forward_pc_refresh_lag: int = 0
    forward_mat_solve_min_patterns: int = 0
    forward_template_reuse: bool = False
    cuda_dense_fallback_max_gib: float = 2.0

    @classmethod
    def from_dict(cls, payload: dict | None) -> "LinearBackendConfig":
        if not payload:
            return cls()
        return cls(
            solver_preset=str(
                payload.get("solver_preset", payload.get("preset", "auto"))
            ),
            ksp_type=str(payload.get("ksp_type", "auto")),
            pc_type=str(payload.get("pc_type", "auto")),
            rtol=float(payload.get("rtol", 1e-10)),
            atol=float(payload.get("atol", 1e-12)),
            max_it=int(payload.get("max_it", 2000)),
            reuse_preconditioner=bool(payload.get("reuse_preconditioner", True)),
            monitor=bool(payload.get("monitor", False)),
            mat_solve_mode=str(payload.get("mat_solve_mode", "off")),
            use_mat_solve=bool(payload.get("use_mat_solve", False)),
            petsc_device=str(payload.get("petsc_device", "auto")),
            pc_factor_mat_solver_type=(
                None
                if payload.get("pc_factor_mat_solver_type") is None
                else str(payload.get("pc_factor_mat_solver_type"))
            ),
            pc_hypre_type=(
                None
                if payload.get("pc_hypre_type") is None
                else str(payload.get("pc_hypre_type"))
            ),
            pc_gamg_type=(
                None
                if payload.get("pc_gamg_type") is None
                else str(payload.get("pc_gamg_type"))
            ),
            petsc_options=dict(payload.get("petsc_options") or {}),
            forward_pc_refresh_policy=str(
                payload.get("forward_pc_refresh_policy", "auto")
            )
            .strip()
            .lower()
            or "auto",
            forward_pc_refresh_iter_threshold=int(
                payload.get("forward_pc_refresh_iter_threshold", 0) or 0
            ),
            forward_pc_refresh_lag=int(payload.get("forward_pc_refresh_lag", 0) or 0),
            forward_mat_solve_min_patterns=int(
                payload.get("forward_mat_solve_min_patterns", 0) or 0
            ),
            forward_template_reuse=bool(payload.get("forward_template_reuse", False)),
            cuda_dense_fallback_max_gib=float(
                payload.get("cuda_dense_fallback_max_gib", 2.0) or 2.0
            ),
        )


@dataclass
class ForwardKSPSession:
    """Long-lived PETSc KSP + PC bundle reused across GN iterations.

    The session stores a single KSP and lets subsequent solves call
    ``setOperators(A_new)`` + ``setReusePreconditioner`` instead of rebuilding
    the PC hierarchy for every new conductivity. It is owned by the
    ``EITForwardModel`` instance, *not* by ``cache_manager``.
    """

    ksp: object
    current_A: object
    current_solve_A: object
    backend_name: str
    ksp_type: str
    pc_type: str
    factor_solver_type: str | None
    solve_mat_type: str | None
    structural_fingerprint: str
    reuse_requested: bool = True
    reuse_applied: bool = False
    solves_since_setup: int = 0
    total_setups: int = 1
    total_solves: int = 0
    last_iter_count: int | None = None
    last_refresh_reason: str | None = "initial_setup"
    last_refresh_triggered: bool = True
    dense_cuda_fallback: dict | None = None

    def record_solve(self, iter_count: int | None) -> None:
        self.total_solves += 1
        self.solves_since_setup += 1
        if iter_count is not None:
            try:
                self.last_iter_count = int(iter_count)
            except (TypeError, ValueError):
                self.last_iter_count = None

    def mark_refresh(self, reason: str) -> None:
        self.last_refresh_reason = reason
        self.last_refresh_triggered = True
        self.solves_since_setup = 0
        self.total_setups += 1

    def mark_reuse(self) -> None:
        self.last_refresh_reason = None
        self.last_refresh_triggered = False

    def as_bundle(self) -> dict[str, object]:
        return {
            "A": self.current_A,
            "solve_A": self.current_solve_A,
            "ksp": self.ksp,
            "backend": self.backend_name,
            "ksp_type": self.ksp_type,
            "pc_type": self.pc_type,
            "factor_solver_type": self.factor_solver_type,
            "solve_mat_type": self.solve_mat_type,
            "ksp_setup_count": 1 if self.last_refresh_triggered else 0,
            "ksp_setup_attempts": int(self.total_setups),
            "reuse_preconditioner": bool(self.reuse_requested),
            "reuse_preconditioner_applied": bool(self.reuse_applied),
        }

    def as_observability(
        self,
        *,
        cache_hit: bool,
        session_reused: bool,
        setup_seconds: float,
        rhs_count: int,
        rhs_kind: str = "",
    ) -> dict[str, object]:
        return {
            "schema": "pyeidors-forward-ksp-session-telemetry-v1",
            "backend": self.backend_name,
            "ksp_type": self.ksp_type,
            "pc_type": self.pc_type,
            "factor_solver_type": self.factor_solver_type,
            "solve_mat_type": self.solve_mat_type,
            "structural_fingerprint": self.structural_fingerprint,
            "structural_fingerprint_short": self.structural_fingerprint[:12],
            "cache_hit": bool(cache_hit),
            "session_reused": bool(session_reused),
            "reuse_requested": bool(self.reuse_requested),
            "reuse_applied": bool(self.reuse_applied),
            "refresh_triggered": bool(self.last_refresh_triggered),
            "refresh_reason": self.last_refresh_reason,
            "solves_since_setup": int(self.solves_since_setup),
            "total_setups": int(self.total_setups),
            "total_solves": int(self.total_solves),
            "last_iter_count": self.last_iter_count,
            "setup_seconds": float(setup_seconds),
            "rhs_count": int(rhs_count),
            "rhs_kind": str(rhs_kind),
        }


def _hash_mesh_content(dolfinx_mesh) -> str:
    """Best-effort content hash for an in-memory DOLFINx mesh.

    Falls back to an empty string when the mesh object does not expose the
    expected geometry / topology arrays (e.g. unit-test fakes). Callers are
    responsible for treating an empty hash as "unavailable" and using a
    different stable identifier (such as ``mesh_file``) instead.
    """
    hasher = hashlib.sha256()
    touched = False
    try:
        geometry = getattr(dolfinx_mesh, "geometry", None)
        coords = getattr(geometry, "x", None) if geometry is not None else None
        if coords is not None:
            coord_arr = np.asarray(coords, dtype=np.float64)
            if coord_arr.size:
                update_digest_with_array_payload(hasher, coord_arr)
                touched = True
        topology = getattr(dolfinx_mesh, "topology", None)
        if topology is not None:
            tdim = int(getattr(topology, "dim", 0) or 0)
            create_connectivity = getattr(topology, "create_connectivity", None)
            if callable(create_connectivity):
                try:
                    create_connectivity(tdim, 0)
                except Exception:
                    pass
            connectivity_fn = getattr(topology, "connectivity", None)
            conn_obj = None
            if callable(connectivity_fn):
                try:
                    conn_obj = connectivity_fn(tdim, 0)
                except Exception:
                    conn_obj = None
            conn_arr = getattr(conn_obj, "array", None)
            if conn_arr is not None:
                cells = np.asarray(conn_arr, dtype=np.int64)
                if cells.size:
                    update_digest_with_array_payload(hasher, cells)
                    touched = True
    except Exception:
        pass
    return hasher.hexdigest() if touched else ""


def _has_nonzero_imaginary(values: np.ndarray) -> bool:
    """Return True when values carry a meaningful imaginary component."""

    return _array_has_nonzero_imaginary(values)


def _coerce_scalar_array(
    values,
    scalar_dtype: np.dtype,
    *,
    name: str,
    copy: bool = False,
) -> np.ndarray:
    """Coerce values to the active PETSc scalar dtype without dropping phase."""

    dtype = np.dtype(scalar_dtype)
    array = np.asarray(values)
    if not np.issubdtype(dtype, np.complexfloating) and _has_nonzero_imaginary(array):
        raise RuntimeError(
            f"{name} contains complex values, but the active PETSc/DOLFINx "
            f"scalar dtype is {dtype}. Enter `nix develop .#complex` "
            "or `nix develop .#complex64` and retry."
        )
    if not np.issubdtype(dtype, np.complexfloating) and np.iscomplexobj(array):
        array = np.real(array)
    out = np.asarray(array, dtype=dtype)
    if copy:
        out = out.copy()
    if not all_finite_values(out):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _hash_scalar_array(values, scalar_dtype: np.dtype) -> str:
    """Hash an array with the active PETSc scalar dtype and shape metadata."""

    array = np.ascontiguousarray(
        _coerce_scalar_array(values, np.dtype(scalar_dtype), name="cache array")
    )
    digest = hashlib.sha256()
    digest.update(f"{array.dtype}:{array.shape}:".encode("utf-8"))
    update_digest_with_array_payload(digest, array)
    return digest.hexdigest()


class EITForwardModel:
    """EIT forward model for ordered CEM, weighted PEM, and mixed assembly."""

    def __init__(
        self,
        n_elec: int,
        pattern_config: PatternConfig,
        z: np.ndarray | None,
        mesh: EITMesh,
        linear_backend: str = "petsc",
        backend_config: dict | LinearBackendConfig | None = None,
        forward_backend: str = DEFAULT_FORWARD_BACKEND,
        cache_manager=None,
        performance_mode: str = "aggressive",
        potential_order: int = 1,
        electrode_model: str = "cem",
        electrode_specs: Sequence[ElectrodeSpec | Mapping[str, Any]] | None = None,
    ):
        self.n_elec = n_elec
        self.scalar_dtype = petsc_scalar_dtype()
        self.is_complex = bool(np.issubdtype(self.scalar_dtype, np.complexfloating))
        requested_electrode_model = str(electrode_model or "cem").strip().lower()
        if requested_electrode_model not in {"cem", "pem", "mixed"}:
            raise ValueError("electrode_model must be 'cem', 'pem', or 'mixed'")
        if not isinstance(mesh, EITMesh):
            raise TypeError("EITForwardModel expects an EITMesh instance")
        self.eit_mesh = mesh
        self.mesh = mesh.mesh
        raw_electrode_specs = (
            electrode_specs
            if electrode_specs is not None
            else getattr(mesh, "electrode_specs", None)
        )
        self.electrode_specs = self._normalize_electrode_specs(
            requested_electrode_model,
            raw_electrode_specs,
        )
        self.cem_electrode_indices = tuple(
            index
            for index, spec in enumerate(self.electrode_specs)
            if spec.kind == "cem"
        )
        self.pem_electrode_indices = tuple(
            index
            for index, spec in enumerate(self.electrode_specs)
            if spec.kind == "pem"
        )
        self.has_cem = bool(self.cem_electrode_indices)
        self.has_pem = bool(self.pem_electrode_indices)
        self.electrode_model = (
            "mixed"
            if self.has_cem and self.has_pem
            else ("cem" if self.has_cem else "pem")
        )
        if requested_electrode_model != self.electrode_model:
            raise ValueError(
                "electrode_model does not match the ordered electrode specs: "
                f"{requested_electrode_model!r} != {self.electrode_model!r}"
            )
        self.contact_impedance_applicable = self.has_cem
        self.z, self.contact_impedance_source = (
            self._normalize_ordered_contact_impedance(z)
        )
        try:
            self.potential_order = int(potential_order)
        except (TypeError, ValueError) as exc:
            raise ValueError("potential_order must be a positive integer") from exc
        if self.potential_order < 1:
            raise ValueError("potential_order must be >= 1")
        if self.has_pem and self.potential_order != 1:
            raise ValueError("Weighted PEM currently requires potential_order=1")
        self.mesh_family = str(getattr(mesh, "mesh_family", None) or "tetra")
        self.geometry_version = str(getattr(mesh, "geometry_version", None) or "legacy")
        self.generator_revision = str(
            getattr(mesh, "generator_revision", None) or "g3d0"
        )

        self._mpi_backend_info = self._assert_supported_mpi_runtime()
        if self.z.size != self.n_elec:
            raise ValueError(
                f"Contact impedance length ({self.z.size}) does not match electrode count ({self.n_elec})"
            )

        self.linear_backend = str(linear_backend).strip().lower()
        self.forward_backend = normalize_forward_backend(
            forward_backend,
            default=DEFAULT_FORWARD_BACKEND,
        )
        if self.forward_backend == "cuda_structured" and self.is_complex:
            raise ValueError(
                "cuda_structured forward backend is real-only. "
                "Use forward_backend='dolfinx' with petsc_device='cuda' in "
                "`nix develop .#complex-cuda` or `nix develop .#complex64-cuda` "
                "for complex admittivity GPU CEM."
            )
        if self.forward_backend == "cuda_structured" and self.has_pem:
            raise ValueError(
                "Weighted or mixed PEM requires forward_backend='dolfinx'; "
                "cuda_structured currently implements CEM only."
            )
        if self.forward_backend == "cuda_structured" and self.potential_order != 1:
            raise ValueError(
                "potential_order > 1 is supported by the DOLFINx forward backend; "
                "cuda_structured currently supports only P1."
            )
        self.backend_config = (
            backend_config
            if isinstance(backend_config, LinearBackendConfig)
            else LinearBackendConfig.from_dict(backend_config)
        )
        if (
            self.has_pem
            and self._solver_token(self.backend_config.solver_preset)
            in _COMPLEX_BLOCK_REAL_AMGX_PRESETS
        ):
            raise ValueError(
                "complex_block_real_amgx is currently a CEM-only route; "
                "use the standard DOLFINx PETSc/SciPy backend for native PEM."
            )
        self.performance_mode = str(performance_mode).strip().lower()
        self.cache_manager = cache_manager
        self._last_cache_lookup: dict[str, str | bool] = {}
        self._forward_ksp_session: ForwardKSPSession | None = None
        self._full_matrix_template = None
        self._full_matrix_template_fingerprint: str | None = None

        self.facet_tags = mesh.facet_tags
        self.interior_facet_tags = getattr(mesh, "interior_facet_tags", None)
        self.association_table = mesh.association_table
        exterior_cem = any(
            self.electrode_specs[index].boundary_kind == "exterior"
            for index in self.cem_electrode_indices
        )
        interior_cem = any(
            self.electrode_specs[index].boundary_kind == "interior"
            for index in self.cem_electrode_indices
        )
        if exterior_cem and self.facet_tags is None:
            raise ValueError("EITMesh lacks electrode facet tags for exterior CEM")
        if interior_cem and self.interior_facet_tags is None:
            raise ValueError("EITMesh lacks interior CEM facet tags")
        if self.has_pem:
            (
                self.point_electrode_source_nodes,
                self.pem_source_node_lists,
                self.pem_source_weight_lists,
                self.ground_node_source,
            ) = self._resolve_pem_source_metadata()
        else:
            self.point_electrode_source_nodes = np.empty(0, dtype=np.int64)
            self.pem_source_node_lists = ()
            self.pem_source_weight_lists = ()
            self.ground_node_source = None

        mesh_file = getattr(mesh, "mesh_file", None)
        mesh_content_hash = None
        if not mesh_file:
            mesh_content_hash = _hash_mesh_content(self.mesh) or None
        self._static_setup_cache_key = build_process_forward_setup_key(
            mesh_file=mesh_file,
            mesh_content_hash=mesh_content_hash,
            n_elec=self.n_elec,
            z=self.z,
            pattern_config=deepcopy(pattern_config),
            potential_order=self.potential_order,
            scalar_dtype=self.scalar_dtype,
            electrode_model=self.electrode_model,
            point_node_ids=self.point_electrode_source_nodes,
            ground_node=self.ground_node_source,
            electrode_spec_signature=self._electrode_spec_signature(),
        )
        self._static_setup_lookup = {
            "hit": False,
            "layer": "compute",
            "artifact": "forward_static_setup",
        }
        self._initialize_static_setup(pattern_config)
        self.backend_config = self._resolve_linear_backend_config(self.backend_config)
        self._M_petsc = {}
        self._petsc_backend_info = self._resolve_petsc_backend_info()
        self._petsc_backend_info["forward_backend_requested"] = self.forward_backend
        self._petsc_backend_info["forward_backend_effective"] = self.forward_backend
        self._petsc_backend_info["mesh_family"] = self.mesh_family
        self._petsc_backend_info["geometry_version"] = self.geometry_version
        self._petsc_backend_info["generator_revision"] = self.generator_revision
        self._petsc_backend_info["potential_order"] = self.potential_order
        self._petsc_backend_info["potential_space_family"] = "Lagrange"
        self._petsc_backend_info["conductivity_order"] = 0
        self._petsc_backend_info["electrode_model"] = self.electrode_model
        self._petsc_backend_info["contact_impedance_applicable"] = bool(
            self.contact_impedance_applicable
        )
        self._petsc_backend_info["electrode_projection"] = (
            "none" if self.electrode_model == "pem" else "exact-surface-facets"
        )
        self._petsc_backend_info.update(runtime_scalar_summary())
        self._petsc_backend_info["static_setup_lookup"] = dict(
            self._static_setup_lookup
        )
        self._cuda_structured_runtime = None
        self._cuda_structured_backend = None
        if self.forward_backend == "cuda_structured":
            self._cuda_structured_runtime = resolve_cuda_structured_runtime(
                mesh_dim=self.mesh_tdim,
                mesh_file=getattr(self.eit_mesh, "mesh_file", None),
                mesh_family=self.mesh_family,
                geometry_version=self.geometry_version,
                generator_revision=self.generator_revision,
                petsc_device_requested=str(
                    self._petsc_backend_info.get("petsc_device_requested", "auto")
                ),
                scalar_type="real",
                mesh_comm_size=int(self.mesh.comm.size),
            )
            self._set_backend_diagnostic(**self._cuda_structured_runtime)
            self._cuda_structured_backend = CudaStructuredForwardBackend(
                self, self._cuda_structured_runtime
            )
            self._set_backend_diagnostic(
                **self._cuda_structured_backend.backend_diagnostics()
            )

    def _initialize_static_setup(self, pattern_config: PatternConfig) -> None:
        bundle = get_process_forward_setup_bundle(self._static_setup_cache_key)
        if bundle is not None:
            self._apply_static_setup_bundle(bundle)
            self._static_setup_lookup = {
                "key": self._static_setup_cache_key,
                "hit": True,
                "layer": "process",
                "artifact": "forward_static_setup",
            }
            return

        self.geometry_scale_to_m = float(pattern_config.geometry_scale_to_m)
        self.mesh_tdim = int(self.mesh.topology.dim)
        self.boundary_scale_to_m = self.geometry_scale_to_m ** max(
            1, self.mesh_tdim - 1
        )
        if self.has_pem:
            if str(pattern_config.drive_mode).strip().lower() == "line_current_density":
                raise ValueError(
                    "PEM has no electrode boundary length; use "
                    "drive_mode='total_current' (amperes) or 'normalized'."
                )
        if not self.has_cem:
            self.ds_electrodes = None
            self.dS_electrodes = None
            self.electrode_tags = []
            self.electrode_boundary_measures = {}
            self.electrode_lengths_m = np.ones(self.n_elec, dtype=float)
        else:
            has_exterior = any(
                self.electrode_specs[index].boundary_kind == "exterior"
                for index in self.cem_electrode_indices
            )
            has_interior = any(
                self.electrode_specs[index].boundary_kind == "interior"
                for index in self.cem_electrode_indices
            )
            self.ds_electrodes = (
                create_ds_measure(self.mesh, self.facet_tags) if has_exterior else None
            )
            self.dS_electrodes = (
                ufl.Measure(
                    "dS",
                    domain=self.mesh,
                    subdomain_data=self.interior_facet_tags,
                )
                if has_interior
                else None
            )
            self.electrode_tags = self._resolve_electrode_tags()
            self.electrode_boundary_measures = (
                self._compute_electrode_boundary_measures()
            )
            override = pattern_config.electrode_length_m_override
            if isinstance(override, (list, tuple, np.ndarray)):
                override_values = np.asarray(override).reshape(-1)
                if override_values.size == self.n_elec:
                    override = [
                        float(override_values[index])
                        for index in self.cem_electrode_indices
                    ]
            cem_lengths = resolve_electrode_lengths_m(
                electrode_lengths_mesh=[
                    self.electrode_boundary_measures[tag] for tag in self.electrode_tags
                ],
                geometry_scale_to_m=self.boundary_scale_to_m,
                electrode_length_m_override=override,
                n_elec=len(self.cem_electrode_indices),
            )
            self.electrode_lengths_m = np.ones(self.n_elec, dtype=float)
            self.electrode_lengths_m[
                np.asarray(self.cem_electrode_indices, dtype=np.int64)
            ] = cem_lengths
        cached_pattern_config = deepcopy(pattern_config)
        self.pattern_manager = StimMeasPatternManager(
            cached_pattern_config,
            electrode_lengths_m=self.electrode_lengths_m,
            mesh_tdim=self.mesh.topology.dim,
        )
        self.V = fem.functionspace(self.mesh, ("Lagrange", self.potential_order))
        self.V_sigma = fem.functionspace(self.mesh, ("DG", 0))
        dofmap = self.V.dofmap.index_map
        self.dofs = int(dofmap.size_local * self.V.dofmap.index_map_bs)
        self.u = ufl.TrialFunction(self.V)
        self.phi = ufl.TestFunction(self.V)
        if self.has_pem:
            (
                self.point_electrode_matrix,
                self.ground_dof,
            ) = self._build_point_electrode_matrix()
        else:
            self.point_electrode_matrix = None
            self.ground_dof = -1
        if self.has_cem:
            self.M = self._assemble_electrode_matrix()
        else:
            self.M = self._assemble_pem_auxiliary_matrix()

        bundle = ForwardStaticSetupBundle(
            ds_electrodes=self.ds_electrodes,
            dS_electrodes=self.dS_electrodes,
            electrode_tags=tuple(int(tag) for tag in self.electrode_tags),
            electrode_boundary_measures=dict(self.electrode_boundary_measures),
            geometry_scale_to_m=float(self.geometry_scale_to_m),
            mesh_tdim=int(self.mesh_tdim),
            boundary_scale_to_m=float(self.boundary_scale_to_m),
            electrode_lengths_m=np.asarray(self.electrode_lengths_m, dtype=float),
            pattern_manager=self.pattern_manager,
            V=self.V,
            V_sigma=self.V_sigma,
            dofs=int(self.dofs),
            electrode_matrix=self.M,
            electrode_model=self.electrode_model,
            point_electrode_matrix=self.point_electrode_matrix,
            ground_dof=int(self.ground_dof),
            cem_electrode_indices=self.cem_electrode_indices,
            pem_electrode_indices=self.pem_electrode_indices,
            cem_boundary_kinds=tuple(
                self.electrode_specs[index].boundary_kind
                for index in self.cem_electrode_indices
            ),
        )
        put_process_forward_setup_bundle(self._static_setup_cache_key, bundle)
        self._static_setup_lookup = {
            "key": self._static_setup_cache_key,
            "hit": False,
            "layer": "compute",
            "artifact": "forward_static_setup",
        }

    def _apply_static_setup_bundle(self, bundle: ForwardStaticSetupBundle) -> None:
        self.ds_electrodes = bundle.ds_electrodes
        self.dS_electrodes = bundle.dS_electrodes
        self.electrode_tags = [int(tag) for tag in bundle.electrode_tags]
        self.electrode_boundary_measures = {
            int(tag): float(value)
            for tag, value in bundle.electrode_boundary_measures.items()
        }
        self.geometry_scale_to_m = float(bundle.geometry_scale_to_m)
        self.mesh_tdim = int(bundle.mesh_tdim)
        self.boundary_scale_to_m = float(bundle.boundary_scale_to_m)
        self.electrode_lengths_m = np.asarray(bundle.electrode_lengths_m, dtype=float)
        self.pattern_manager = deepcopy(bundle.pattern_manager)
        self.V = bundle.V
        self.V_sigma = bundle.V_sigma
        self.dofs = int(bundle.dofs)
        self.u = ufl.TrialFunction(self.V)
        self.phi = ufl.TestFunction(self.V)
        self.M = bundle.electrode_matrix
        self.point_electrode_matrix = bundle.point_electrode_matrix
        self.ground_dof = int(bundle.ground_dof)
        if tuple(bundle.cem_electrode_indices) != self.cem_electrode_indices:
            raise RuntimeError("Forward static setup cache CEM electrode mismatch")
        if tuple(bundle.pem_electrode_indices) != self.pem_electrode_indices:
            raise RuntimeError("Forward static setup cache PEM electrode mismatch")
        if str(bundle.electrode_model) != self.electrode_model:
            raise RuntimeError(
                "Forward static setup cache electrode-model mismatch: "
                f"{bundle.electrode_model!r} != {self.electrode_model!r}"
            )

    def _active_scalar_dtype(self) -> np.dtype:
        """Return this model's PETSc scalar dtype, defaulting for unit stubs."""

        return np.dtype(getattr(self, "scalar_dtype", np.float64))

    def _active_scalar_is_complex(self) -> bool:
        return bool(
            getattr(
                self,
                "is_complex",
                np.issubdtype(self._active_scalar_dtype(), np.complexfloating),
            )
        )

    def _scalar_value(self, value):
        return self._active_scalar_dtype().type(value)

    def _as_scalar_array(self, values, *, name: str, copy: bool = False) -> np.ndarray:
        return _coerce_scalar_array(
            values,
            self._active_scalar_dtype(),
            name=name,
            copy=copy,
        )

    def _normalize_electrode_specs(
        self,
        requested_model: str,
        raw_specs: Sequence[ElectrodeSpec | Mapping[str, Any]] | None,
    ) -> tuple[_ForwardElectrodeSpec, ...]:
        if raw_specs is None:
            if requested_model == "mixed":
                raise ValueError("electrode_model='mixed' requires electrode_specs")
            source_nodes = np.asarray(
                getattr(self.eit_mesh, "point_electrode_source_nodes", []),
                dtype=np.int64,
            ).reshape(-1)
            if requested_model == "pem" and source_nodes.size != self.n_elec:
                raise ValueError(
                    "PEM requires one source node per electrode or explicit "
                    "weighted electrode_specs"
                )
            return tuple(
                _ForwardElectrodeSpec(
                    kind=requested_model,
                    source_nodes=(
                        (int(source_nodes[index]),) if requested_model == "pem" else ()
                    ),
                    node_weights=((1.0,) if requested_model == "pem" else ()),
                    boundary_kind=("none" if requested_model == "pem" else "exterior"),
                )
                for index in range(self.n_elec)
            )

        if len(raw_specs) != self.n_elec:
            raise ValueError(
                f"Expected {self.n_elec} electrode specs, got {len(raw_specs)}"
            )
        normalized: list[_ForwardElectrodeSpec] = []
        pem_signatures: set[tuple[tuple[int, ...], tuple[complex, ...]]] = set()
        for index, raw_spec in enumerate(raw_specs):
            if isinstance(raw_spec, ElectrodeSpec):
                mapping = {
                    name: getattr(raw_spec, name)
                    for name in raw_spec.__dataclass_fields__
                }
            elif isinstance(raw_spec, Mapping):
                mapping = dict(raw_spec)
            else:
                raise TypeError(
                    f"Electrode spec {index + 1} must be ElectrodeSpec or mapping"
                )
            kind = (
                str(mapping.get("kind", mapping.get("electrode_model", "")))
                .strip()
                .lower()
            )
            kind = {
                "point": "pem",
                "distributed_point": "pem",
                "cem_faces": "cem",
            }.get(kind, kind)
            if kind not in {"cem", "pem"}:
                raise ValueError(
                    f"Electrode spec {index + 1} has unsupported kind {kind!r}"
                )
            index_base = int(mapping.get("index_base", 0))
            if index_base not in {0, 1}:
                raise ValueError(
                    f"Electrode spec {index + 1} index_base must be 0 or 1"
                )
            source_nodes = tuple(
                int(value) - index_base
                for value in np.asarray(
                    mapping.get("source_nodes", ()),
                    dtype=np.int64,
                ).reshape(-1)
            )
            if any(value < 0 for value in source_nodes):
                raise ValueError(
                    f"Electrode spec {index + 1} contains invalid source nodes"
                )
            node_weights = tuple(
                complex(value)
                for value in np.asarray(
                    mapping.get("node_weights", ()),
                ).reshape(-1)
            )
            boundary_kind = (
                str(
                    mapping.get(
                        "boundary_kind",
                        "none" if kind == "pem" else "exterior",
                    )
                )
                .strip()
                .lower()
            )
            if boundary_kind not in {"exterior", "interior", "none"}:
                raise ValueError(
                    f"Electrode spec {index + 1} has invalid boundary_kind"
                )
            if kind == "pem":
                if boundary_kind != "none":
                    raise ValueError(
                        f"PEM electrode {index + 1} must use boundary_kind='none'"
                    )
                if not source_nodes or len(source_nodes) != len(node_weights):
                    raise ValueError(
                        f"PEM electrode {index + 1} requires matching nodes/weights"
                    )
                weights = np.asarray(node_weights, dtype=np.complex128)
                if not np.all(np.isfinite(weights)):
                    raise ValueError(
                        f"PEM electrode {index + 1} weights must be finite"
                    )
                if not np.isclose(np.sum(weights), 1.0):
                    raise ValueError(
                        f"PEM electrode {index + 1} weights must sum to one"
                    )
                if len(set(source_nodes)) != len(source_nodes):
                    raise ValueError(
                        f"PEM electrode {index + 1} contains duplicate source nodes"
                    )
                signature = (
                    source_nodes,
                    tuple(complex(value) for value in weights),
                )
                if signature in pem_signatures:
                    raise ValueError("Duplicate weighted PEM electrode definitions")
                pem_signatures.add(signature)
            elif boundary_kind == "none":
                raise ValueError(
                    f"CEM electrode {index + 1} requires exterior or interior faces"
                )
            normalized.append(
                _ForwardElectrodeSpec(
                    kind=kind,
                    source_nodes=source_nodes,
                    node_weights=node_weights,
                    boundary_kind=boundary_kind,
                    contact_impedance=mapping.get("contact_impedance"),
                    contact_impedance_present=bool(
                        mapping.get(
                            "contact_impedance_present",
                            mapping.get("contact_impedance") is not None,
                        )
                    ),
                )
            )
        return tuple(normalized)

    def _normalize_ordered_contact_impedance(
        self,
        z: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        provided = np.asarray([] if z is None else z).reshape(-1)
        if provided.size not in {0, self.n_elec}:
            raise ValueError(
                f"Contact impedance length ({provided.size}) does not match "
                f"electrode count ({self.n_elec})"
            )
        matrix_values = np.full(
            self.n_elec,
            np.nan,
            dtype=self._active_scalar_dtype(),
        )
        provenance: list[float | complex] = [float("nan")] * self.n_elec
        provenance_present = False
        for index, spec in enumerate(self.electrode_specs):
            candidate = (
                spec.contact_impedance
                if spec.contact_impedance_present
                else (provided[index] if provided.size else None)
            )
            if spec.kind == "cem":
                if candidate is None:
                    raise ValueError(
                        f"CEM electrode {index + 1} requires contact impedance"
                    )
                value = self._as_scalar_array(
                    [candidate],
                    name=f"CEM electrode {index + 1} contact impedance",
                )[0]
                if np.isclose(np.abs(value), 0.0):
                    raise ValueError("contact impedance values must be non-zero")
                matrix_values[index] = value
            elif candidate is not None:
                value = complex(candidate)
                if not np.isfinite(value):
                    raise ValueError(
                        f"PEM electrode {index + 1} contact provenance must be finite"
                    )
                provenance[index] = (
                    float(value.real) if np.isclose(value.imag, 0.0) else value
                )
                provenance_present = True
        return (
            np.ascontiguousarray(matrix_values),
            np.asarray(provenance) if provenance_present else None,
        )

    def _electrode_spec_signature(self) -> list[dict[str, Any]]:
        specs = getattr(self, "electrode_specs", None)
        if specs is None:
            return [
                {
                    "kind": "cem",
                    "source_nodes": [],
                    "node_weights": [],
                    "boundary_kind": "exterior",
                }
                for _ in range(int(self.n_elec))
            ]
        return [
            {
                "kind": spec.kind,
                "source_nodes": list(spec.source_nodes),
                "node_weights": [
                    [float(value.real), float(value.imag)]
                    for value in (complex(item) for item in spec.node_weights)
                ],
                "boundary_kind": spec.boundary_kind,
            }
            for spec in specs
        ]

    def _normalize_contact_impedance(self, z: np.ndarray) -> np.ndarray:
        impedance = self._as_scalar_array(z, name="contact impedance").reshape(-1)
        if impedance.size != int(self.n_elec):
            raise ValueError(
                f"Contact impedance length ({impedance.size}) does not match "
                f"electrode count ({self.n_elec})"
            )
        if np.any(np.isclose(np.abs(impedance), 0.0)):
            raise ValueError("contact impedance values must be non-zero")
        return np.ascontiguousarray(impedance, dtype=self._active_scalar_dtype())

    def _normalize_pem_contact_provenance(
        self, z: np.ndarray | None
    ) -> np.ndarray | None:
        if z is None:
            return None
        values = np.asarray(z).reshape(-1)
        if values.size == 1:
            values = np.full(self.n_elec, values[0], dtype=values.dtype)
        if values.size != self.n_elec:
            raise ValueError(
                f"Contact impedance provenance length ({values.size}) does not "
                f"match electrode count ({self.n_elec})"
            )
        return np.array(values, copy=True)

    def _resolve_pem_source_metadata(
        self,
    ) -> tuple[
        np.ndarray,
        tuple[np.ndarray, ...],
        tuple[np.ndarray, ...],
        int | None,
    ]:
        source_node_lists = tuple(
            np.asarray(
                self.electrode_specs[index].source_nodes,
                dtype=np.int64,
            )
            for index in self.pem_electrode_indices
        )
        source_weight_lists = tuple(
            self._as_scalar_array(
                self.electrode_specs[index].node_weights,
                name=f"PEM electrode {index + 1} weights",
            )
            for index in self.pem_electrode_indices
        )
        singleton_nodes = np.asarray(
            [
                int(nodes[0])
                for nodes in source_node_lists
                if np.asarray(nodes).size == 1
            ],
            dtype=np.int64,
        )
        ground_node = getattr(self.eit_mesh, "gnd_node_source", None)
        if not self.has_cem and ground_node is None:
            raise ValueError(
                "Pure PEM requires the exact EIDORS/PyEIDORS ground source node"
            )
        return (
            np.ascontiguousarray(singleton_nodes),
            source_node_lists,
            source_weight_lists,
            None if ground_node is None else int(ground_node),
        )

    def _source_vertex_to_p1_dof(self, source_node: int, *, label: str) -> int:
        self.mesh.topology.create_connectivity(0, self.mesh.topology.dim)
        vertex_map = self.mesh.topology.index_map(0)
        if vertex_map is None:
            raise ValueError("Unable to access the DOLFINx vertex map for native PEM")
        n_vertices = int(vertex_map.size_local + vertex_map.num_ghosts)
        local_to_source = np.asarray(
            self.mesh.geometry.input_global_indices,
            dtype=np.int64,
        ).reshape(-1)
        matches = np.flatnonzero(local_to_source[:n_vertices] == int(source_node))
        if matches.size != 1:
            raise ValueError(
                f"{label} source node {source_node + 1} maps to "
                f"{matches.size} local vertices; expected exactly one"
            )
        dofs = np.asarray(
            fem.locate_dofs_topological(
                self.V,
                0,
                np.asarray([int(matches[0])], dtype=np.int32),
            ),
            dtype=np.int64,
        ).reshape(-1)
        if dofs.size != 1:
            raise ValueError(
                f"{label} source node {source_node + 1} maps to "
                f"{dofs.size} P1 DOFs; expected exactly one"
            )
        return int(dofs[0])

    def _build_point_electrode_matrix(self) -> tuple[csr_matrix, int]:
        rows: list[int] = []
        columns: list[int] = []
        values: list[float | complex] = []
        for pem_position, electrode_index in enumerate(self.pem_electrode_indices):
            nodes = self.pem_source_node_lists[pem_position]
            weights = self.pem_source_weight_lists[pem_position]
            for source_node, weight in zip(nodes, weights, strict=True):
                rows.append(int(electrode_index))
                columns.append(
                    self._source_vertex_to_p1_dof(
                        int(source_node),
                        label=f"PEM electrode {electrode_index + 1}",
                    )
                )
                values.append(weight)
        matrix = csr_matrix(
            (
                np.asarray(values, dtype=self._active_scalar_dtype()),
                (
                    np.asarray(rows, dtype=np.int32),
                    np.asarray(columns, dtype=np.int32),
                ),
            ),
            shape=(self.n_elec, self.dofs),
            dtype=self._active_scalar_dtype(),
        )
        ground_dof = (
            -1
            if self.has_cem
            else self._source_vertex_to_p1_dof(
                int(self.ground_node_source),
                label="PEM ground",
            )
        )
        return matrix, ground_dof

    def _assemble_pem_auxiliary_matrix(self) -> csr_matrix:
        full_size = self.dofs + self.n_elec + 1
        matrix = lil_matrix(
            (full_size, full_size),
            dtype=self._active_scalar_dtype(),
        )
        for electrode_index in self.pem_electrode_indices:
            row = self.dofs + electrode_index
            matrix[row, row] = self._scalar_value(1.0)
        if not self.has_cem:
            matrix[full_size - 1, full_size - 1] = self._scalar_value(1.0)
        return csr_matrix(matrix)

    @staticmethod
    def _solver_token(value: object, default: str = "auto") -> str:
        token = str(value if value is not None else default).strip().lower()
        return token or default

    def _resolve_linear_backend_config(
        self, config: LinearBackendConfig
    ) -> LinearBackendConfig:
        """Resolve high-level PETSc presets into concrete KSP/PC settings.

        The default stays direct for 2D/small validation, while 3D defaults to
        PETSc's native AMG path. Users can still pin exact PETSc types through
        ``ksp_type`` and ``pc_type``.
        """
        preset = self._solver_token(config.solver_preset)
        ksp_type = self._solver_token(config.ksp_type)
        pc_type = self._solver_token(config.pc_type)
        explicit_solver = ksp_type != "auto" or pc_type != "auto"

        if preset == "auto":
            if explicit_solver:
                preset = "custom"
            elif int(getattr(self, "mesh_tdim", 2)) >= 3:
                preset = "3d_gamg"
            else:
                preset = "direct"

        presets: dict[str, dict[str, object]] = {
            "custom": {},
            "direct": {"ksp_type": "preonly", "pc_type": "lu"},
            "legacy_direct": {"ksp_type": "preonly", "pc_type": "lu"},
            "debug_direct": {"ksp_type": "preonly", "pc_type": "lu"},
            "3d_gamg": {
                "ksp_type": "fgmres",
                "pc_type": "gamg",
                "pc_gamg_type": "agg",
                "petsc_options": {
                    "mg_levels_ksp_type": "chebyshev",
                    "mg_levels_pc_type": "jacobi",
                },
            },
            "3d_amg": {
                "ksp_type": "fgmres",
                "pc_type": "gamg",
                "pc_gamg_type": "agg",
                "petsc_options": {
                    "mg_levels_ksp_type": "chebyshev",
                    "mg_levels_pc_type": "jacobi",
                },
            },
            "3d_hypre": {
                "ksp_type": "fgmres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
            "hypre_boomeramg": {
                "ksp_type": "fgmres",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
            "spd_gamg": {
                "ksp_type": "cg",
                "pc_type": "gamg",
                "pc_gamg_type": "agg",
                "petsc_options": {
                    "mg_levels_ksp_type": "chebyshev",
                    "mg_levels_pc_type": "jacobi",
                },
            },
            "spd_hypre": {
                "ksp_type": "cg",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
            "cg_hypre": {
                "ksp_type": "cg",
                "pc_type": "hypre",
                "pc_hypre_type": "boomeramg",
            },
            "amgx": {
                "ksp_type": "fgmres",
                "pc_type": "amgx",
                "petsc_options": {
                    "pc_amgx_smoother": "JACOBI_L1",
                    "pc_amgx_exact_coarse_solve": "0",
                    "pc_amgx_presweeps": "2",
                    "pc_amgx_postsweeps": "2",
                    "pc_amgx_coarse_solver": "NOSOLVER",
                },
            },
            "cuda_amgx": {
                "ksp_type": "fgmres",
                "pc_type": "amgx",
                "petsc_device": "cuda",
                "petsc_options": {
                    "pc_amgx_smoother": "JACOBI_L1",
                    "pc_amgx_exact_coarse_solve": "0",
                    "pc_amgx_presweeps": "2",
                    "pc_amgx_postsweeps": "2",
                    "pc_amgx_coarse_solver": "NOSOLVER",
                },
            },
            "complex_cuda_amgx": {
                "ksp_type": "fgmres",
                "pc_type": "amgx",
                "petsc_device": "cuda",
                "petsc_options": {
                    "pc_amgx_amg_method": "AGGREGATION",
                    "pc_amgx_selector": "SIZE_8",
                    "pc_amgx_smoother": "BLOCK_JACOBI",
                    "pc_amgx_exact_coarse_solve": "0",
                    "pc_amgx_presweeps": "2",
                    "pc_amgx_postsweeps": "2",
                    "pc_amgx_coarse_solver": "NOSOLVER",
                },
            },
            "complex_block_real_amgx": {
                "ksp_type": "fgmres",
                "pc_type": "gamg",
                "pc_gamg_type": "agg",
                "petsc_device": "cuda",
                "petsc_options": {
                    "block_real_amgx_profile": "real_jacobi_l1",
                    "block_real_amgx_ksp_type": "bcgs",
                    "block_real_amgx_rtol": "1e-6",
                    "block_real_amgx_atol": "1e-12",
                    "block_real_amgx_max_it": "4000",
                    "block_real_amgx_max_relative_residual": "1e-6",
                },
            },
        }
        if preset not in presets:
            raise ValueError(
                f"Unsupported PETSc solver_preset={config.solver_preset!r}. "
                f"Expected one of: {', '.join(sorted(presets))}."
            )

        template = presets[preset]
        resolved_options = dict(template.get("petsc_options") or {})
        resolved_options.update(dict(config.petsc_options or {}))
        if pc_type != "auto":
            default_ksp_for_pc = (
                "preonly" if pc_type in {"lu", "cholesky", "qr"} else "fgmres"
            )
        else:
            default_ksp_for_pc = "preonly"
        resolved_ksp_type = (
            ksp_type
            if ksp_type != "auto"
            else str(template.get("ksp_type", default_ksp_for_pc))
        )
        resolved_pc_type = (
            pc_type if pc_type != "auto" else str(template.get("pc_type", "lu"))
        )
        return replace(
            config,
            solver_preset=preset,
            ksp_type=resolved_ksp_type,
            pc_type=resolved_pc_type,
            pc_factor_mat_solver_type=(
                config.pc_factor_mat_solver_type
                if config.pc_factor_mat_solver_type is not None
                else (
                    None
                    if template.get("pc_factor_mat_solver_type") is None
                    else str(template.get("pc_factor_mat_solver_type"))
                )
            ),
            pc_hypre_type=(
                config.pc_hypre_type
                if config.pc_hypre_type is not None
                else (
                    None
                    if template.get("pc_hypre_type") is None
                    else str(template.get("pc_hypre_type"))
                )
            ),
            pc_gamg_type=(
                config.pc_gamg_type
                if config.pc_gamg_type is not None
                else (
                    None
                    if template.get("pc_gamg_type") is None
                    else str(template.get("pc_gamg_type"))
                )
            ),
            petsc_device=(
                config.petsc_device
                if self._solver_token(config.petsc_device) != "auto"
                else str(template.get("petsc_device", config.petsc_device))
            ),
            petsc_options=resolved_options,
        )

    def _resolve_pattern_matrix(self, current_patterns=None) -> np.ndarray:
        """Return stimulation matrix with shape ``(n_patterns, n_elec)``."""
        if current_patterns is None:
            matrix = self._as_scalar_array(
                self.pattern_manager.stim_matrix,
                name="stimulation matrix",
            )
        else:
            matrix = self._as_scalar_array(
                current_patterns,
                name="current_patterns",
            )
            if matrix.ndim != 2:
                raise ValueError("current_patterns must be a 2D array")
            if matrix.shape[1] == self.n_elec:
                pass  # already correct orientation
            elif matrix.shape[0] == self.n_elec:
                matrix = matrix.T
            else:
                raise ValueError(
                    "current_patterns shape mismatch. Expected (n_patterns, n_elec) "
                    f"or (n_elec, n_patterns), got {matrix.shape}"
                )

        if matrix.shape[1] != self.n_elec:
            raise ValueError(
                f"Pattern width mismatch: expected {self.n_elec}, got {matrix.shape[1]}"
            )
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

        if len(electrode_map) < self.n_elec and isinstance(
            self.association_table, dict
        ):
            candidates = []
            for key, val in self.association_table.items():
                try:
                    tag_val = int(val)
                except (TypeError, ValueError):
                    continue
                if isinstance(key, (int, np.integer)) and key >= 2:
                    candidates.append(tag_val)
            if candidates:
                for idx, tag_val in enumerate(
                    sorted(set(candidates))[: self.n_elec], start=1
                ):
                    electrode_map.setdefault(idx, tag_val)

        cem_indices = getattr(self, "cem_electrode_indices", None)
        if cem_indices is None:
            cem_indices = tuple(range(int(self.n_elec)))
        else:
            cem_indices = tuple(cem_indices)
        required = [index + 1 for index in cem_indices]
        missing = [index for index in required if index not in electrode_map]
        if missing:
            raise ValueError(
                f"Association table missing electrode tags {missing}, cannot assemble CEM"
            )
        return [electrode_map[index] for index in required]

    def _cem_trace_measure(self, electrode_index: int, electrode_tag: int):
        specs = getattr(self, "electrode_specs", None)
        boundary_kind = (
            specs[electrode_index].boundary_kind if specs is not None else "exterior"
        )
        if boundary_kind == "interior":
            if self.dS_electrodes is None:
                raise ValueError("Interior CEM measure is unavailable")
            return (
                self.dS_electrodes(electrode_tag),
                ufl.avg(self.u),
                ufl.avg(self.phi),
            )
        if self.ds_electrodes is None:
            raise ValueError("Exterior CEM measure is unavailable")
        return self.ds_electrodes(electrode_tag), self.u, self.phi

    def _compute_electrode_boundary_measures(self):
        """Compute electrode boundary measure (2D length / 3D area)."""
        measures = {}
        one = fem.Constant(self.mesh, self._scalar_value(1.0))
        cem_indices = getattr(self, "cem_electrode_indices", None)
        if cem_indices is None:
            electrode_count = int(getattr(self, "n_elec", len(self.electrode_tags)))
            cem_indices = tuple(range(electrode_count))
        else:
            cem_indices = tuple(cem_indices)
        specs = getattr(self, "electrode_specs", None)
        for electrode_index, tag in zip(
            cem_indices,
            self.electrode_tags,
            strict=True,
        ):
            boundary_kind = (
                specs[electrode_index].boundary_kind
                if specs is not None
                else "exterior"
            )
            if boundary_kind == "interior":
                if self.dS_electrodes is None:
                    raise ValueError("Interior CEM measure is unavailable")
                measure = self.dS_electrodes(tag)
            else:
                if self.ds_electrodes is None:
                    raise ValueError("Exterior CEM measure is unavailable")
                measure = self.ds_electrodes(tag)
            measure_local = fem.assemble_scalar(fem.form(one * measure))
            measure = self.mesh.comm.allreduce(measure_local, op=MPI.SUM)
            measure_real = np.real_if_close(measure)
            if np.iscomplexobj(measure_real):
                measure_real = np.real(measure_real)
            measure_float = float(measure_real)
            measures[tag] = measure_float
            if np.isclose(measure_float, 0.0):
                warnings.warn(
                    f"Electrode boundary tag {tag} has zero measure, check mesh markers",
                    RuntimeWarning,
                    stacklevel=2,
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
        scalar_type = getattr(PETSc, "ScalarType", matrix.data.dtype)
        values = np.asarray(matrix.data, dtype=np.dtype(scalar_type))
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
            "Enter `nix develop .#cuda` for real-valued CUDA, or "
            "`nix develop .#complex-cuda` / `nix develop .#complex64-cuda` "
            "for complex-admittivity CUDA. Verify with "
            "`python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty`, "
            "and retry."
        )

    def _resolve_mpi_backend_info(self) -> dict[str, object]:
        comm = getattr(getattr(self, "mesh", None), "comm", None)
        try:
            from ..perf.capabilities import probe_mpi_runtime
        except Exception as exc:
            return {
                "mpi_available": False,
                "mpi_source": "probe_import_failed",
                "mpi_size": 1,
                "mpi_rank": 0,
                "mpi_parallel": False,
                "mpi_parallel_supported": False,
                "mpi_size_supported": True,
                "mpi_fallback_reason": None,
                "mpi_guidance": f"mpi_probe_import_failed: {exc}",
            }
        return probe_mpi_runtime(comm=comm, supports_parallel=False)

    def _assert_supported_mpi_runtime(self) -> dict[str, object]:
        info = self._resolve_mpi_backend_info()
        if bool(info.get("mpi_size_supported", True)):
            return info
        raise RuntimeError(
            "PyEIDORS phase-2 migration currently supports MPI size=1 only. "
            f"Detected MPI size={int(info.get('mpi_size') or 1)}, "
            f"rank={int(info.get('mpi_rank') or 0)}. "
            f"mpi_fallback_reason={info.get('mpi_fallback_reason')}. "
            "Use single-rank execution in this stage; MPI production requires "
            "distributed Mat/Vec assembly and mpiexec smoke validation."
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
        requested = self._normalize_petsc_device(
            getattr(self.backend_config, "petsc_device", "auto")
        )
        forward_backend = str(getattr(self, "forward_backend", "dolfinx"))
        mpi_info = dict(
            getattr(self, "_mpi_backend_info", None) or self._resolve_mpi_backend_info()
        )
        info: dict[str, object] = {
            "petsc_device_requested": requested,
            "petsc_device_effective": "cpu",
            "petsc_mat_type": None,
            "petsc_vec_type": None,
            "petsc_dense_mat_type": None,
            "gpu_fallback_reason": None,
            "gpu_transfer_risk": None,
            "forward_factor_backend": self.linear_backend,
            "forward_mat_solve_effective": None,
            "solver_preset": getattr(self.backend_config, "solver_preset", "auto"),
            "ksp_type": getattr(self.backend_config, "ksp_type", None),
            "pc_type": getattr(self.backend_config, "pc_type", None),
            "pc_factor_mat_solver_type": getattr(
                self.backend_config,
                "pc_factor_mat_solver_type",
                None,
            ),
            "pc_hypre_type": getattr(self.backend_config, "pc_hypre_type", None),
            "pc_gamg_type": getattr(self.backend_config, "pc_gamg_type", None),
            "forward_reuse_preconditioner_requested": bool(
                getattr(self.backend_config, "reuse_preconditioner", True)
            ),
            "forward_reuse_preconditioner_applied": None,
            "forward_ksp_setup_count": None,
            "forward_ksp_setup_attempts": None,
            "forward_pc_refresh_policy": str(
                getattr(self.backend_config, "forward_pc_refresh_policy", "auto")
            ),
            "forward_pc_refresh_triggered": None,
            "forward_pc_refresh_reason": None,
            "forward_pc_session_reused": None,
            "forward_pc_session_solves": None,
            "forward_pc_session_total_setups": None,
            "forward_pc_last_iter_count": None,
            "forward_ksp_session": {},
            "capability": {},
            "petsc_hypre_available": False,
            "petsc_amgx_available": False,
            "petsc_amgx_cuda_candidate": False,
            "petsc_hypre_cuda_blacklisted": False,
            "petsc_scalar_type": str(self._active_scalar_dtype()),
            "petsc_scalar_is_complex": self._active_scalar_is_complex(),
        }
        info.update(mpi_info)

        def _with_stable_cpu_types() -> dict[str, object]:
            mat_type, vec_type = self._stable_cpu_petsc_types()
            info["petsc_mat_type"] = info.get("petsc_mat_type") or mat_type
            info["petsc_vec_type"] = info.get("petsc_vec_type") or vec_type
            return info

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
            return _with_stable_cpu_types()

        capability = probe_petsc_cuda_runtime()
        info["capability"] = capability
        info["petsc_hypre_available"] = bool(capability.get("petsc_hypre", False))
        info["petsc_amgx_available"] = bool(capability.get("petsc_amgx", False))
        info["petsc_amgx_cuda_candidate"] = bool(
            capability.get("petsc_amgx_cuda_candidate", False)
        )
        pc_type = self._solver_token(getattr(self.backend_config, "pc_type", ""))
        solver_preset = self._solver_token(
            getattr(self.backend_config, "solver_preset", "")
        )
        info.update(_solver_route_metadata(solver_preset))
        if (
            requested in {"auto", "cuda"}
            and bool(capability.get("petsc_cuda", False))
            and is_hypre_cuda_blacklisted_solver(
                solver_preset=solver_preset,
                pc_type=pc_type,
            )
        ):
            info["gpu_fallback_reason"] = CUDA_HYPRE_BLACKLIST_REASON
            info["petsc_hypre_cuda_blacklisted"] = True
            raise RuntimeError(
                "PETSc Hypre CUDA route is blacklisted after B4 SIGSEGV "
                "(pc_type='hypre' / spd_hypre+cuda). Use spd_gamg with "
                "petsc_device='cuda', or force petsc_device='cpu' for Hypre."
            )
        amgx_requested = pc_type == "amgx" or solver_preset in _NATIVE_AMGX_PRESETS
        if amgx_requested and not bool(
            capability.get("petsc_amgx_cuda_candidate", False)
        ):
            raise RuntimeError(
                "当前 PETSc PCAMGX 不能作为 CUDA 求解候选 "
                "(PETSc PCAMGX CUDA smoke unavailable); rebuild PETSc with "
                "AmgX support, or choose spd_gamg (Hypre CUDA is blacklisted "
                "after B4)."
            )
        cuda_available = bool(capability.get("petsc_cuda", False))
        if requested == "cpu":
            return _with_stable_cpu_types()
        if requested == "cuda" and not cuda_available:
            if forward_backend == "cuda_structured":
                info["gpu_fallback_reason"] = (
                    "petsc_cuda_not_required_for_cuda_structured"
                )
                return _with_stable_cpu_types()
            reason = (
                capability.get("errors", {})
                if isinstance(capability.get("errors"), dict)
                else capability
            )
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
            info["gpu_transfer_risk"] = (
                "mixed_dolfinx_assembly_to_petsc_cuda"
                if forward_backend != "cuda_structured"
                else "cuda_structured_runtime_reports_transfer_boundary"
            )
            return info
        if requested == "auto":
            info["gpu_fallback_reason"] = "petsc_cuda_not_available"
        return _with_stable_cpu_types()

    def _get_cuda_type(self, info_key: str, petsc_obj_name: str, type_attr: str):
        """Look up a PETSc CUDA type from backend info, falling back to PETSc constants."""
        info = getattr(self, "_petsc_backend_info", {}) or {}
        if info.get("petsc_device_effective") != "cuda" or PETSc is None:
            return None
        cached = info.get(info_key)
        if cached:
            return cached
        namespace = getattr(getattr(PETSc, petsc_obj_name, None), "Type", None)
        if namespace is not None and hasattr(namespace, type_attr):
            return str(getattr(namespace, type_attr))
        return None

    def _get_requested_petsc_mat_type(self):
        return self._get_cuda_type("petsc_mat_type", "Mat", "AIJCUSPARSE")

    def _get_requested_dense_mat_type(self):
        return self._get_cuda_type("petsc_dense_mat_type", "Mat", "DENSECUDA")

    def _get_requested_petsc_vec_type(self):
        return self._get_cuda_type("petsc_vec_type", "Vec", "CUDA")

    @staticmethod
    def _mat_type_key(mat_type) -> str:
        return str(mat_type).strip().lower() if mat_type is not None else "cpu"

    @staticmethod
    def _vec_to_numpy(vec) -> np.ndarray:
        if hasattr(vec, "array"):
            return np.asarray(vec.array)
        return np.asarray(vec.getArray(readonly=True))

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
    def _petsc_option_value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _configure_pc_from_backend_config(self, pc_obj, *, factor_backend=None) -> None:
        factor_type = factor_backend or getattr(
            self.backend_config,
            "pc_factor_mat_solver_type",
            None,
        )
        if factor_type is not None and hasattr(pc_obj, "setFactorSolverType"):
            pc_obj.setFactorSolverType(str(factor_type))

        hypre_type = getattr(self.backend_config, "pc_hypre_type", None)
        if hypre_type is not None and hasattr(pc_obj, "setHYPREType"):
            try:
                pc_obj.setHYPREType(str(hypre_type))
            except Exception:
                pass

        gamg_type = getattr(self.backend_config, "pc_gamg_type", None)
        if gamg_type is not None and hasattr(pc_obj, "setGAMGType"):
            try:
                pc_obj.setGAMGType(str(gamg_type))
            except Exception:
                pass

    def _apply_ksp_options_database(self, ksp_obj) -> None:
        if PETSc is None or not hasattr(ksp_obj, "setFromOptions"):
            return
        options = dict(getattr(self.backend_config, "petsc_options", {}) or {})
        hypre_type = getattr(self.backend_config, "pc_hypre_type", None)
        gamg_type = getattr(self.backend_config, "pc_gamg_type", None)
        factor_type = getattr(self.backend_config, "pc_factor_mat_solver_type", None)
        pc_type = str(getattr(self.backend_config, "pc_type", "") or "").strip().lower()
        if hypre_type is not None and pc_type == "hypre":
            options.setdefault("pc_hypre_type", str(hypre_type))
        if gamg_type is not None and pc_type == "gamg":
            options.setdefault("pc_gamg_type", str(gamg_type))
        if factor_type is not None and pc_type in {"lu", "cholesky", "qr"}:
            options.setdefault("pc_factor_mat_solver_type", str(factor_type))
        if not options:
            return

        prefix = f"pyeidors_forward_{id(ksp_obj):x}_"
        try:
            ksp_obj.setOptionsPrefix(prefix)
        except Exception:
            prefix = ""
        opts = PETSc.Options()
        written_keys: list[str] = []
        for key, value in options.items():
            option_key = str(key).strip().lstrip("-")
            if not option_key:
                continue
            full_key = prefix + option_key
            opts[full_key] = self._petsc_option_value(value)
            written_keys.append(full_key)
        try:
            ksp_obj.setFromOptions()
        finally:
            for full_key in written_keys:
                try:
                    if hasattr(opts, "delValue"):
                        opts.delValue(full_key)
                    else:
                        del opts[full_key]
                except Exception:
                    pass

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

    def get_backend_diagnostics(self) -> dict[str, object]:
        return dict(getattr(self, "_petsc_backend_info", {}) or {})

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
        return bool(
            getattr(
                self,
                "has_cem",
                getattr(self, "electrode_model", "cem") == "cem",
            )
            and getattr(self, "_petsc_backend_info", {}).get("petsc_device_effective")
            == "cuda"
        )

    def _cuda_gauge_rows(self) -> tuple[int, int]:
        """Return ``(constraint_row, reference_electrode_col)`` for CUDA CEM gauge.

        The CPU/EIDORS CEM system enforces ``sum(U)=0`` with a Lagrange
        multiplier. PETSc CUDA iterative routes are fragile on that saddle-point
        row, so the CUDA route uses the equivalent gauge ``U_0=0`` during the
        solve and recenters electrode voltages back to zero mean afterwards.
        """
        reference_index = (
            int(self.cem_electrode_indices[0])
            if getattr(self, "cem_electrode_indices", ())
            else 0
        )
        return self.dofs + self.n_elec, self.dofs + reference_index

    def _apply_cuda_gauge_fix_matrix(self, mat):
        if PETSc is None or mat is None or not self._gpu_gauge_fix_enabled():
            return mat
        try:
            gauge_matrix = self._petsc_to_csr(mat).tolil()
            constraint_row, reference_col = self._cuda_gauge_rows()
            gauge_matrix[constraint_row, :] = self._scalar_value(0.0)
            gauge_matrix[constraint_row, reference_col] = self._scalar_value(1.0)
            fixed = self._csr_to_petsc(gauge_matrix.tocsr())
            self._set_backend_diagnostic(
                gpu_constraint_strategy="reference-electrode-row"
            )
            if fixed is not mat and hasattr(mat, "destroy"):
                try:
                    mat.destroy()
                except Exception:
                    pass
            return fixed
        except Exception:
            return mat

    def _apply_cuda_gauge_fix_rhs(self, rhs_matrix: np.ndarray) -> np.ndarray:
        if getattr(self, "has_pem", getattr(self, "electrode_model", "") == "pem"):
            rhs_matrix = self._prepare_pem_rhs(rhs_matrix)
        if not self._gpu_gauge_fix_enabled():
            return rhs_matrix
        constraint_row, _reference_col = self._cuda_gauge_rows()
        rhs_matrix[constraint_row, :] = self._scalar_value(0.0)
        return rhs_matrix

    def _prepare_pem_rhs(self, rhs_matrix: np.ndarray) -> np.ndarray:
        rhs = self._as_scalar_array(
            rhs_matrix,
            name="PEM rhs_matrix",
            copy=True,
        )
        if rhs.ndim == 1:
            rhs = rhs.reshape(-1, 1)
        full_size = self.dofs + self.n_elec + 1
        if rhs.shape[0] != full_size:
            raise ValueError(
                f"PEM RHS row count mismatch: expected {full_size}, got {rhs.shape[0]}"
            )
        if not all_finite_values(rhs):
            raise ValueError("PEM RHS must contain only finite values")

        electrode_currents = rhs[
            self.dofs : self.dofs + self.n_elec,
            :,
        ].copy()
        current_scale = np.maximum(
            1.0,
            np.max(np.abs(electrode_currents), axis=0),
        )
        net_current = np.sum(electrode_currents, axis=0)
        unbalanced = np.flatnonzero(np.abs(net_current) > 1e-10 * current_scale)
        if unbalanced.size:
            pattern = int(unbalanced[0])
            raise ValueError(
                "Native PEM requires balanced electrode currents; "
                f"RHS column {pattern} has net current {net_current[pattern]!r}"
            )

        rhs[: self.dofs, :] += self.point_electrode_matrix.T @ electrode_currents
        pem_rows = self.dofs + np.asarray(
            self.pem_electrode_indices,
            dtype=np.int64,
        )
        rhs[pem_rows, :] = self._scalar_value(0.0)
        rhs[self.dofs + self.n_elec, :] = self._scalar_value(0.0)
        if not self.has_cem:
            rhs[self.ground_dof, :] = self._scalar_value(0.0)
        return rhs

    def _finalize_pem_solution(self, sol_matrix: np.ndarray) -> np.ndarray:
        if not getattr(
            self,
            "has_pem",
            getattr(self, "electrode_model", "cem") == "pem",
        ):
            return sol_matrix
        solution = self._as_scalar_array(
            sol_matrix,
            name="PEM solution",
            copy=True,
        )
        weighted_voltages = self.point_electrode_matrix @ solution[: self.dofs, :]
        pem_indices = np.asarray(self.pem_electrode_indices, dtype=np.int64)
        solution[self.dofs + pem_indices, :] = weighted_voltages[pem_indices, :]
        solution[self.dofs + self.n_elec, :] = self._scalar_value(0.0)
        return solution

    def _recenter_cuda_gauge_solution(self, sol_matrix: np.ndarray) -> np.ndarray:
        if not self._gpu_gauge_fix_enabled():
            return sol_matrix
        sol = self._as_scalar_array(
            sol_matrix,
            name="cuda gauge solution",
            copy=True,
        )
        cem_indices = np.asarray(
            getattr(self, "cem_electrode_indices", range(self.n_elec)),
            dtype=np.int64,
        )
        electrode_block = sol[self.dofs + cem_indices, :]
        offsets = electrode_block.mean(axis=0, keepdims=True)
        sol[: self.dofs, :] -= offsets
        sol[self.dofs : self.dofs + self.n_elec, :] -= offsets
        sol[self.dofs + self.n_elec, :] = self._scalar_value(0.0)
        return sol

    def _make_petsc_dense_solver_bundle(self, system_matrix):
        if PETSc is None:
            raise RuntimeError("petsc4py is required for CUDA dense fallback")
        reuse_requested = bool(
            getattr(self.backend_config, "reuse_preconditioner", True)
        )
        reuse_applied = False
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
        self._configure_pc_from_backend_config(pc)
        ksp.setTolerances(
            rtol=self.backend_config.rtol,
            atol=self.backend_config.atol,
            max_it=self.backend_config.max_it,
        )
        if hasattr(ksp, "setReusePreconditioner"):
            try:
                ksp.setReusePreconditioner(reuse_requested)
                reuse_applied = True
            except Exception:
                pass
        self._apply_ksp_options_database(ksp)
        ksp.setUp()
        return {
            "A": system_matrix,
            "solve_A": solve_mat,
            "ksp": ksp,
            "backend": f"petsc-ksp-{str(dense_type).lower()}-{self.backend_config.pc_type}",
            "ksp_type": (
                str(ksp.getType())
                if hasattr(ksp, "getType")
                else self.backend_config.ksp_type
            ),
            "pc_type": (
                str(pc.getType())
                if hasattr(pc, "getType")
                else self.backend_config.pc_type
            ),
            "factor_solver_type": None,
            "solve_mat_type": (
                str(solve_mat.getType()) if hasattr(solve_mat, "getType") else None
            ),
            "ksp_setup_count": 1,
            "reuse_preconditioner": reuse_requested,
            "reuse_preconditioner_applied": reuse_applied,
        }

    def _solve_with_cuda_dense_lu_fallback(
        self,
        system_matrix,
        rhs_matrix: np.ndarray,
        *,
        fallback_reason: str,
        solve_start: float,
    ) -> np.ndarray:
        rhs_count = (
            int(np.asarray(rhs_matrix).shape[1])
            if np.asarray(rhs_matrix).ndim > 1
            else 1
        )
        skip_reason = self._cuda_dense_lu_fallback_skip_reason(rhs_count=rhs_count)
        if skip_reason:
            raise RuntimeError(
                "PETSc CUDA dense LU fallback skipped: "
                f"{skip_reason}. {self._actionable_cuda_guidance()}"
            )
        if PETSc is None:
            raise RuntimeError("petsc4py is required for CUDA dense LU fallback")
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
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        ksp.setTolerances(
            rtol=self.backend_config.rtol,
            atol=self.backend_config.atol,
            max_it=self.backend_config.max_it,
        )
        ksp.setUp()

        rhs = self._as_scalar_array(rhs_matrix, name="rhs_matrix")
        sol_matrix = np.zeros_like(rhs)
        b = self._ensure_vec_type(
            solve_mat.createVecRight(), self._get_requested_petsc_vec_type()
        )
        x = self._ensure_vec_type(
            solve_mat.createVecRight(), self._get_requested_petsc_vec_type()
        )
        b_array = b.getArray(readonly=False)
        iterations_per_rhs: list[int | None] = []
        reason = None
        for i in range(rhs.shape[1]):
            b_array[:] = rhs[:, i]
            ksp.solve(b, x)
            try:
                iterations_per_rhs.append(int(ksp.getIterationNumber()))
            except Exception:
                iterations_per_rhs.append(None)
            try:
                reason = int(ksp.getConvergedReason())
            except Exception:
                reason = None
            if reason is not None and reason < 0:
                raise RuntimeError(
                    "PETSc CUDA dense LU fallback failed with a negative "
                    f"convergence reason ({reason})"
                )
            sol_matrix[:, i] = x.getArray(readonly=True)

        total_iterations = sum(
            int(value) for value in iterations_per_rhs if value is not None
        )
        self._set_backend_diagnostic(
            gpu_fallback_reason=f"cuda_dense_lu_fallback:{fallback_reason}",
            fallback_reason=f"cuda_dense_lu_fallback:{fallback_reason}",
            forward_factor_backend=f"petsc-ksp-{str(dense_type).lower()}-lu",
            petsc_solve_mat_type=str(solve_mat.getType())
            if hasattr(solve_mat, "getType")
            else str(dense_type),
            forward_mat_solve_effective="vec-loop",
            forward_ksp_mat_solve_count=0,
            forward_ksp_solve_count=int(rhs.shape[1]),
            forward_ksp_iterations_per_rhs=iterations_per_rhs,
            forward_ksp_iterations_total=total_iterations,
            forward_ksp_converged_reason=reason,
            forward_ksp_converged=None if reason is None else bool(reason > 0),
            forward_solve_seconds=float(time.perf_counter() - solve_start),
        )
        return self._recenter_cuda_gauge_solution(sol_matrix)

    def _cuda_dense_lu_fallback_skip_reason(self, *, rhs_count: int = 1) -> str:
        backend_info = getattr(self, "_petsc_backend_info", {}) or {}
        if str(backend_info.get("petsc_device_effective", "cpu")) != "cuda":
            return ""
        try:
            n_rows = int(self.dofs) + int(self.n_elec) + 1
        except Exception:
            n_rows = 0
        if n_rows <= 0:
            return ""
        try:
            dtype = np.dtype(self._active_scalar_dtype())
        except Exception:
            dtype = petsc_scalar_dtype()
        rhs_cols = max(1, int(rhs_count))
        dense_bytes = int(n_rows) * int(n_rows) * int(dtype.itemsize)
        rhs_bytes = int(n_rows) * rhs_cols * int(dtype.itemsize) * 2
        # LU factorization and PETSc dense conversion hold several buffers.  A
        # conservative multiplier prevents GUI-scale complex runs from trying a
        # dense CUDA matrix that cannot plausibly fit in VRAM.
        estimated_bytes = int(dense_bytes * 4 + rhs_bytes)
        estimated_gib = estimated_bytes / float(1024**3)
        limit_gib = float(
            getattr(self.backend_config, "cuda_dense_fallback_max_gib", 2.0) or 2.0
        )
        self._set_backend_diagnostic(
            cuda_dense_lu_fallback_estimated_gib=estimated_gib,
            cuda_dense_lu_fallback_max_gib=limit_gib,
            cuda_dense_lu_fallback_rows=int(n_rows),
            cuda_dense_lu_fallback_rhs_count=rhs_cols,
            cuda_dense_lu_fallback_scalar_dtype=str(dtype),
        )
        if estimated_gib > limit_gib:
            reason = (
                "cuda_dense_lu_estimated_memory_exceeds_limit"
                f":{estimated_gib:.2f}GiB>{limit_gib:.2f}GiB"
            )
            self._set_backend_diagnostic(
                cuda_dense_lu_fallback_skipped=True,
                cuda_dense_lu_fallback_skip_reason=reason,
            )
            return reason
        self._set_backend_diagnostic(
            cuda_dense_lu_fallback_skipped=False,
            cuda_dense_lu_fallback_skip_reason="",
        )
        return ""

    def _cuda_cem_requires_direct_solve(
        self, session: ForwardKSPSession, *, rhs_count: int = 1
    ) -> bool:
        if not getattr(
            self,
            "has_cem",
            getattr(self, "electrode_model", "cem") == "cem",
        ):
            return False
        backend_info = getattr(self, "_petsc_backend_info", {}) or {}
        if str(backend_info.get("petsc_device_effective", "cpu")) != "cuda":
            return False
        if not self._gpu_gauge_fix_enabled():
            return False
        ksp_type = str(session.ksp_type or "").strip().lower()
        pc_type = str(session.pc_type or "").strip().lower()
        solver_preset = self._solver_token(
            backend_info.get(
                "solver_preset",
                getattr(self.backend_config, "solver_preset", ""),
            )
        )
        capability = backend_info.get("capability", {})
        capability = capability if isinstance(capability, dict) else {}
        amgx_cuda_candidate = bool(
            backend_info.get(
                "petsc_amgx_cuda_candidate",
                capability.get("petsc_amgx_cuda_candidate", False),
            )
        )
        if (
            pc_type == "amgx" or solver_preset in _EXPLICIT_AMGX_PRESETS
        ) and amgx_cuda_candidate:
            self._set_backend_diagnostic(
                cuda_cem_direct_fallback_suppressed="pcamgx_explicit"
            )
            return False
        requires_direct = not (
            ksp_type == "preonly" and pc_type in {"lu", "cholesky", "qr"}
        )
        if not requires_direct:
            return False
        skip_reason = self._cuda_dense_lu_fallback_skip_reason(rhs_count=rhs_count)
        if skip_reason:
            self._set_backend_diagnostic(
                gpu_fallback_reason=f"cuda_dense_lu_fallback_skipped:{skip_reason}",
                fallback_reason=f"cuda_dense_lu_fallback_skipped:{skip_reason}",
            )
            return False
        return True

    def _is_complex_block_real_amgx_route(self) -> bool:
        return (
            self._solver_token(
                getattr(getattr(self, "backend_config", None), "solver_preset", "")
            )
            in _COMPLEX_BLOCK_REAL_AMGX_PRESETS
        )

    def _is_native_pcamgx_route(self) -> bool:
        cfg = getattr(self, "backend_config", None)
        solver_preset = self._solver_token(getattr(cfg, "solver_preset", ""))
        pc_type = self._solver_token(getattr(cfg, "pc_type", ""), "")
        return pc_type == "amgx" or solver_preset in _NATIVE_AMGX_PRESETS

    def _complex_block_real_matrix_and_rhs(
        self, sigma: fem.Function, rhs_matrix: np.ndarray
    ) -> tuple[csr_matrix, np.ndarray, str]:
        matrix = self._create_full_matrix_scipy(sigma).tocsr()
        rhs = self._as_scalar_array(rhs_matrix, name="rhs_matrix", copy=True)
        if rhs.ndim == 1:
            rhs = rhs.reshape(-1, 1)
        gauge = ""
        if self._gpu_gauge_fix_enabled():
            gauge_matrix = matrix.tolil()
            constraint_row, reference_col = self._cuda_gauge_rows()
            gauge_matrix[constraint_row, :] = self._scalar_value(0.0)
            gauge_matrix[constraint_row, reference_col] = self._scalar_value(1.0)
            matrix = gauge_matrix.tocsr()
            rhs = self._apply_cuda_gauge_fix_rhs(rhs)
            gauge = "reference-electrode-row"
            self._set_backend_diagnostic(
                gpu_constraint_strategy="reference-electrode-row"
            )
        return matrix, rhs, gauge

    def _solve_full_rhs_with_complex_block_real_amgx(
        self,
        sigma: fem.Function,
        rhs_matrix: np.ndarray,
        *,
        rhs_kind: str = "custom",
    ) -> np.ndarray:
        if not self._active_scalar_is_complex():
            raise RuntimeError(
                "complex_block_real_amgx requires a complex PETSc assembly runtime; "
                "real-valued GPU solves should use cuda_amgx."
            )
        rhs_preview = np.asarray(rhs_matrix)
        n_rhs = 1 if rhs_preview.ndim == 1 else int(rhs_preview.shape[1])
        setup_t0 = time.perf_counter()
        matrix, rhs, gauge = self._complex_block_real_matrix_and_rhs(sigma, rhs_matrix)
        setup_seconds = float(time.perf_counter() - setup_t0)
        solve_t0 = time.perf_counter()
        from .block_real_amgx import solve_complex_system_with_external_block_real_amgx

        petsc_options = dict(getattr(self.backend_config, "petsc_options", {}) or {})

        def _block_real_float_option(key: str, default: float) -> float:
            try:
                return float(petsc_options.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        def _block_real_int_option(key: str, default: int) -> int:
            try:
                return int(petsc_options.get(key, default))
            except (TypeError, ValueError):
                return int(default)

        block_real_rtol = _block_real_float_option("block_real_amgx_rtol", 1.0e-6)
        block_real_atol = _block_real_float_option("block_real_amgx_atol", 1.0e-12)
        block_real_max_it = max(
            4000,
            _block_real_int_option("block_real_amgx_max_it", 4000),
        )
        block_real_max_relative_residual = _block_real_float_option(
            "block_real_amgx_max_relative_residual",
            1.0e-6,
        )
        block_real_ksp_type = str(
            petsc_options.get("block_real_amgx_ksp_type", "bcgs") or "bcgs"
        )
        solution, report = solve_complex_system_with_external_block_real_amgx(
            matrix,
            rhs,
            potential_dofs=int(self.dofs),
            n_elec=int(self.n_elec),
            gauge=gauge,
            ksp_type=block_real_ksp_type,
            rtol=block_real_rtol,
            atol=block_real_atol,
            max_it=block_real_max_it,
            max_relative_residual=block_real_max_relative_residual,
        )
        solve_seconds = float(time.perf_counter() - solve_t0)
        solver = report.get("solver", {}) if isinstance(report, dict) else {}
        iterations = [
            int(value)
            for value in (solver.get("iterations_per_rhs", []) or [])
            if value is not None
        ]
        reasons = [
            int(value)
            for value in (solver.get("converged_reasons", []) or [])
            if value is not None
        ]
        residual = 0.0
        if isinstance(report, dict):
            residual = float(
                (report.get("complex_true_residual", {}) or {}).get("relative_max", 0.0)
                or 0.0
            )
        self._set_backend_diagnostic(
            forward_factor_backend="external-complex-block-real-amgx",
            forward_rhs_kind=str(rhs_kind),
            forward_rhs_count=n_rhs,
            forward_setup_seconds=setup_seconds,
            forward_solve_seconds=solve_seconds,
            forward_mat_solve_effective="vec-loop",
            forward_ksp_mat_solve_count=0,
            forward_ksp_solve_count=n_rhs,
            forward_ksp_iterations_per_rhs=iterations,
            forward_ksp_iterations_total=sum(iterations),
            forward_ksp_converged_reason=reasons[-1] if reasons else None,
            forward_ksp_converged=all(reason > 0 for reason in reasons)
            if reasons
            else None,
            petsc_solve_mat_type=str(solver.get("mat_type", "")),
            petsc_vec_type=str(solver.get("vec_type", "")),
            ksp_type=str(solver.get("ksp_type", "fgmres")),
            pc_type=str(solver.get("pc_type", "amgx")),
            solver_route_family="complex_block_real_amgx",
            solver_route_status="strict_accuracy_complex_gpu",
            solver_route_caveat=_COMPLEX_BLOCK_REAL_AMGX_CAVEAT,
            block_real_amgx_run_dir=str(report.get("run_dir", ""))
            if isinstance(report, dict)
            else "",
            block_real_amgx_external_worker_persistent=bool(
                report.get("external_worker_persistent", False)
            )
            if isinstance(report, dict)
            else False,
            block_real_amgx_external_worker_transport_error=str(
                report.get("external_worker_transport_error", "")
            )
            if isinstance(report, dict)
            else "",
            block_real_amgx_true_relative_residual_max=residual,
        )
        return self._recenter_cuda_gauge_solution(
            np.asarray(solution, dtype=self._active_scalar_dtype())
        )

    def _solve_with_complex_block_real_amgx(
        self, sigma: fem.Function, pattern_matrix: np.ndarray
    ) -> np.ndarray:
        rhs_matrix = np.zeros(
            (self.dofs + self.n_elec + 1, pattern_matrix.shape[0]),
            dtype=self._active_scalar_dtype(),
        )
        rhs_matrix[self.dofs : self.dofs + self.n_elec, :] = pattern_matrix.T
        return self._solve_full_rhs_with_complex_block_real_amgx(
            sigma,
            rhs_matrix,
            rhs_kind="forward_patterns",
        )

    def _assemble_electrode_matrix(self):
        b_form = 0
        for electrode_index, electrode_tag in zip(
            self.cem_electrode_indices,
            self.electrode_tags,
            strict=True,
        ):
            measure, trial, test = self._cem_trace_measure(
                electrode_index,
                electrode_tag,
            )
            b_form += (
                (self.boundary_scale_to_m / self.z[electrode_index])
                * ufl.inner(trial, test)
                * measure
            )

        B = fem_petsc.assemble_matrix(fem.form(b_form))
        B.assemble()
        M = self._petsc_to_csr(B).astype(self._active_scalar_dtype(), copy=False)
        M.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        M_lil = lil_matrix(M, dtype=self._active_scalar_dtype())

        for electrode_index, electrode_tag in zip(
            self.cem_electrode_indices,
            self.electrode_tags,
            strict=True,
        ):
            measure, _trial, test = self._cem_trace_measure(
                electrode_index,
                electrode_tag,
            )
            c_form = (
                (-self.boundary_scale_to_m / self.z[electrode_index])
                * ufl.conj(test)
                * measure
            )
            C_vec = fem_petsc.assemble_vector(fem.form(c_form))
            C_vec.assemble()
            C_i = self._as_scalar_array(C_vec.array, name="electrode coupling vector")

            row = self.dofs + electrode_index
            M_lil[row, : self.dofs] = C_i
            M_lil[: self.dofs, row] = C_i
            electrode_len_m = float(self.electrode_lengths_m[electrode_index])
            M_lil[row, row] = (1.0 / self.z[electrode_index]) * electrode_len_m
            M_lil[self.dofs + self.n_elec, row] = self._scalar_value(1.0)
            M_lil[row, self.dofs + self.n_elec] = self._scalar_value(1.0)
        for electrode_index in self.pem_electrode_indices:
            row = self.dofs + electrode_index
            M_lil[row, row] = self._scalar_value(1.0)

        return csr_matrix(M_lil)

    def _ensure_electrode_matrix(self):
        if self.M is None:
            self.M = self._assemble_electrode_matrix()
        return self.M

    def _assemble_conductivity_matrix(self, sigma: fem.Function, *, mat_kind=None):
        """Assemble the conductivity-dependent stiffness matrix in PETSc form."""
        a_form = ufl.inner(sigma * ufl.grad(self.u), ufl.grad(self.phi)) * ufl.dx
        return self._assemble_form_matrix(fem.form(a_form), mat_kind=mat_kind)

    def _create_full_matrix_scipy(self, sigma: fem.Function) -> csr_matrix:
        """Build full system matrix for SciPy backend."""
        scipy_A = self._petsc_to_csr(self._assemble_conductivity_matrix(sigma))
        if not getattr(
            self,
            "has_cem",
            getattr(self, "electrode_model", "cem") == "cem",
        ):
            grounded = scipy_A.tolil()
            grounded[self.ground_dof, :] = self._scalar_value(0.0)
            grounded[:, self.ground_dof] = self._scalar_value(0.0)
            grounded[self.ground_dof, self.ground_dof] = self._scalar_value(1.0)
            grounded.resize(
                self.dofs + self.n_elec + 1,
                self.dofs + self.n_elec + 1,
            )
            return grounded.tocsr() + self.M
        scipy_A.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        return scipy_A + self._ensure_electrode_matrix()

    def _assemble_electrode_matrix_petsc(self, *, mat_type=None, vec_type=None):
        if PETSc is None:
            raise RuntimeError("petsc4py is not available for linear_backend='petsc'")

        cem_indices = getattr(self, "cem_electrode_indices", None)
        if cem_indices is None:
            cem_indices = tuple(range(int(self.n_elec)))
        else:
            cem_indices = tuple(cem_indices)
        b_form = 0
        for electrode_index, electrode_tag in zip(
            cem_indices,
            self.electrode_tags,
            strict=True,
        ):
            measure, trial, test = self._cem_trace_measure(
                electrode_index,
                electrode_tag,
            )
            b_form += (
                (self.boundary_scale_to_m / self.z[electrode_index])
                * ufl.inner(trial, test)
                * measure
            )
        top_left = self._assemble_form_matrix(fem.form(b_form), mat_kind=mat_type)
        full_matrix = self._expand_conductivity_csr_to_full(top_left, mat_type=mat_type)

        for electrode_index, electrode_tag in zip(
            cem_indices,
            self.electrode_tags,
            strict=True,
        ):
            measure, _trial, test = self._cem_trace_measure(
                electrode_index,
                electrode_tag,
            )
            c_form = (
                (-self.boundary_scale_to_m / self.z[electrode_index])
                * ufl.conj(test)
                * measure
            )
            c_vec = self._assemble_form_vector(fem.form(c_form), vec_kind=vec_type)
            c_i = self._as_scalar_array(
                self._vec_to_numpy(c_vec),
                name="electrode coupling vector",
            )
            nz, nz_values = _nonzero_index_value_arrays(c_i)
            row = self.dofs + electrode_index
            if nz.size > 0:
                full_matrix.setValues(row, nz, nz_values)
                full_matrix.setValues(nz, row, nz_values)
            electrode_len_m = float(self.electrode_lengths_m[electrode_index])
            full_matrix.setValue(
                row,
                row,
                (1.0 / self.z[electrode_index]) * electrode_len_m,
            )
            full_matrix.setValue(
                self.dofs + self.n_elec,
                row,
                self._scalar_value(1.0),
            )
            full_matrix.setValue(
                row,
                self.dofs + self.n_elec,
                self._scalar_value(1.0),
            )
            if hasattr(c_vec, "destroy"):
                try:
                    c_vec.destroy()
                except Exception:
                    pass
        for electrode_index in getattr(self, "pem_electrode_indices", ()):
            row = self.dofs + electrode_index
            full_matrix.setValue(row, row, self._scalar_value(1.0))

        full_matrix.assemblyBegin()
        full_matrix.assemblyEnd()
        return self._ensure_mat_type(full_matrix, mat_type)

    def _get_electrode_matrix_petsc(self, mat_type=None):
        if PETSc is None:
            raise RuntimeError("petsc4py is not available for linear_backend='petsc'")
        key = self._mat_type_key(mat_type)
        if key not in self._M_petsc:
            electrode_matrix = self._csr_to_petsc(self._ensure_electrode_matrix())
            ground_row = self.dofs + self.n_elec
            try:
                if hasattr(electrode_matrix, "setOption"):
                    electrode_matrix.setOption(
                        PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False
                    )
                electrode_matrix.setValue(
                    ground_row,
                    ground_row,
                    self._scalar_value(0.0),
                )
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
                np.asarray(values, dtype=self._active_scalar_dtype()),
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
        if not getattr(
            self,
            "has_cem",
            getattr(self, "electrode_model", "cem") == "cem",
        ):
            full_matrix = self._csr_to_petsc(self._create_full_matrix_scipy(sigma))
            self._set_backend_diagnostic(
                pem_constraint_strategy="exact-source-node-ground"
            )
        elif self._gpu_gauge_fix_enabled():
            scipy_full = self._create_full_matrix_scipy(sigma).tolil()
            constraint_row, reference_col = self._cuda_gauge_rows()
            scipy_full[constraint_row, :] = self._scalar_value(0.0)
            scipy_full[constraint_row, reference_col] = self._scalar_value(1.0)
            full_matrix = self._csr_to_petsc(scipy_full.tocsr())
            self._set_backend_diagnostic(
                gpu_constraint_strategy="reference-electrode-row"
            )
        else:
            conductivity_mat = self._assemble_conductivity_matrix(sigma, mat_kind=None)
            conductivity_augmented = self._expand_conductivity_csr_to_full(
                conductivity_mat, mat_type=None
            )
            full_matrix = self._build_full_matrix_via_axpy(conductivity_augmented)
            conductivity_augmented.destroy()
        full_matrix = self._ensure_mat_type(full_matrix, mat_kind)
        if hasattr(full_matrix, "assemble"):
            full_matrix.assemble()
        self._set_backend_diagnostic(
            petsc_mat_type=(
                str(full_matrix.getType())
                if hasattr(full_matrix, "getType")
                else mat_kind
            ),
        )
        return full_matrix

    def _build_full_matrix_via_axpy(self, conductivity_augmented):
        """Combine cached electrode matrix M with conductivity term K(σ).

        Default (``forward_template_reuse=False``): copy M, AXPY K with
        ``DIFFERENT_NONZERO_PATTERN`` — original behavior, every iteration
        re-symbolic-allocates K's structure on top of M.

        Opt-in (``forward_template_reuse=True``): once a stable
        ``M ∪ K ∪ structural-diagonal`` template is bootstrapped, every
        subsequent call duplicates it (preallocated structure, zero values),
        AXPYs M and K with ``SUBSET_NONZERO_PATTERN`` (M and K each have
        nonzero structures that are subsets of the union template) so PETSc
        skips the symbolic phase. Template is invalidated whenever the
        structural fingerprint changes (mesh / electrode topology / backend
        config).
        """
        electrode_matrix = self._get_electrode_matrix_petsc(mat_type=None)
        reuse_enabled = bool(
            getattr(self.backend_config, "forward_template_reuse", False)
        )
        template = self._full_matrix_template if reuse_enabled else None
        if template is not None and reuse_enabled:
            current_fp = self._compute_forward_ksp_structural_fingerprint()
            if self._full_matrix_template_fingerprint != current_fp:
                self._dispose_full_matrix_template()
                template = None
        if template is None:
            full_matrix = electrode_matrix.copy()
            full_matrix.axpy(
                1.0,
                conductivity_augmented,
                structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN,
            )
            self._ensure_structural_diagonal(full_matrix)
            full_matrix.assemblyBegin()
            full_matrix.assemblyEnd()
            if reuse_enabled:
                self._full_matrix_template = full_matrix.duplicate(copy=False)
                self._full_matrix_template.zeroEntries()
                self._ensure_structural_diagonal(self._full_matrix_template)
                self._full_matrix_template.assemblyBegin()
                self._full_matrix_template.assemblyEnd()
                self._full_matrix_template_fingerprint = (
                    self._compute_forward_ksp_structural_fingerprint()
                )
            return full_matrix
        full_matrix = template.duplicate(copy=True)
        full_matrix.zeroEntries()
        full_matrix.axpy(
            1.0,
            electrode_matrix,
            structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
        )
        full_matrix.axpy(
            1.0,
            conductivity_augmented,
            structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN,
        )
        self._ensure_structural_diagonal(full_matrix)
        full_matrix.assemblyBegin()
        full_matrix.assemblyEnd()
        return full_matrix

    def _dispose_full_matrix_template(self) -> None:
        template = getattr(self, "_full_matrix_template", None)
        if template is not None:
            destroy = getattr(template, "destroy", None)
            if callable(destroy):
                try:
                    destroy()
                except Exception:
                    pass
        self._full_matrix_template = None
        self._full_matrix_template_fingerprint = None

    def create_full_matrix(self, sigma: fem.Function):
        """Build complete system matrix including conductivity term."""
        return self._create_full_matrix_scipy(sigma)

    def _sigma_fingerprint(self, sigma: fem.Function) -> str:
        return _hash_scalar_array(sigma.x.array, self._active_scalar_dtype())

    def _resolve_mat_solve_mode(self) -> str:
        """Normalize mat_solve_mode from backend config to 'on', 'off', or 'auto'."""
        backend_cfg = getattr(self, "backend_config", None)
        if backend_cfg is None:
            return "on"
        mat_mode = str(getattr(backend_cfg, "mat_solve_mode", "")).strip().lower()
        if mat_mode in {"auto", "on", "off"}:
            return mat_mode
        return "on" if bool(getattr(backend_cfg, "use_mat_solve", False)) else "off"

    def _should_use_mat_solve(self, n_patterns: int) -> bool:
        """Determine whether matSolve should be used for the given pattern count.

        ``forward_mat_solve_min_patterns`` adds an opt-in lower bound on the RHS
        batch size before enabling ``matSolve`` in ``auto`` mode. Very small
        batches allocate MATDENSE B/X buffers without amortizing the overhead
        compared to the vec-loop path, so users targeting small stim counts can
        raise the threshold to keep the vector-loop routing.
        """
        mat_mode = self._resolve_mat_solve_mode()
        try:
            min_patterns = int(
                getattr(
                    getattr(self, "backend_config", None),
                    "forward_mat_solve_min_patterns",
                    0,
                )
                or 0
            )
        except (TypeError, ValueError):
            min_patterns = 0
        min_patterns = max(0, min_patterns)

        if mat_mode == "on":
            use_mat_solve = True
        elif mat_mode == "off":
            use_mat_solve = False
        else:
            use_mat_solve = (
                self.mesh_tdim == 3
                and n_patterns > 1
                and self.performance_mode == "aggressive"
                and (min_patterns == 0 or n_patterns >= min_patterns)
            )

        backend_info = getattr(self, "_petsc_backend_info", {}) or {}
        effective_device = str(backend_info.get("petsc_device_effective", "cpu"))
        capability = (
            backend_info.get("capability")
            if isinstance(backend_info.get("capability"), dict)
            else {}
        )
        has_cuda_dense = bool(capability.get("petsc_cuda_dense", False))

        if effective_device == "cuda" and use_mat_solve and not has_cuda_dense:
            use_mat_solve = False
        return use_mat_solve

    def _predict_forward_mat_solve_effective(self, n_patterns: int) -> str:
        return "matsolve" if self._should_use_mat_solve(n_patterns) else "vec-loop"

    def _base_cache_payload(
        self, sigma_hash: str, n_patterns: int
    ) -> dict[str, object]:
        petsc_backend = getattr(self, "_petsc_backend_info", {}) or {}
        effective_device = str(petsc_backend.get("petsc_device_effective", "cpu"))
        mat_type = petsc_backend.get("petsc_mat_type")
        vec_type = petsc_backend.get("petsc_vec_type")
        if effective_device == "cpu" and (mat_type is None or vec_type is None):
            stable_mat_type, stable_vec_type = self._stable_cpu_petsc_types()
            mat_type = mat_type or stable_mat_type or "cpu-default"
            vec_type = vec_type or stable_vec_type or "cpu-default"
        mat_solve_effective = petsc_backend.get(
            "forward_mat_solve_effective"
        ) or self._predict_forward_mat_solve_effective(n_patterns)
        cem_indices = np.asarray(
            getattr(self, "cem_electrode_indices", range(self.n_elec)),
            dtype=np.int64,
        )
        cem_impedance = np.asarray(self.z)[cem_indices]

        return {
            "backend": self.linear_backend,
            "forward_backend": self.forward_backend,
            "electrode_model": str(getattr(self, "electrode_model", "cem")),
            "sigma_hash": sigma_hash,
            "scalar_dtype": str(self._active_scalar_dtype()),
            "scalar_is_complex": self._active_scalar_is_complex(),
            "n_elec": self.n_elec,
            "potential_order": int(getattr(self, "potential_order", 1)),
            "n_patterns": n_patterns,
            "z_hash": (
                _hash_scalar_array(cem_impedance, self._active_scalar_dtype())
                if cem_indices.size
                else "not-applicable"
            ),
            "electrode_specs": self._electrode_spec_signature(),
            "pattern_hash": _hash_scalar_array(
                self.pattern_manager.stim_matrix,
                self._active_scalar_dtype(),
            ),
            "backend_config": {
                "solver_preset": getattr(
                    self.backend_config, "solver_preset", "custom"
                ),
                "ksp_type": self.backend_config.ksp_type,
                "pc_type": self.backend_config.pc_type,
                "rtol": self.backend_config.rtol,
                "atol": self.backend_config.atol,
                "max_it": self.backend_config.max_it,
                "reuse_preconditioner": self.backend_config.reuse_preconditioner,
                "mat_solve_mode": self.backend_config.mat_solve_mode,
                "use_mat_solve": self.backend_config.use_mat_solve,
                "petsc_device": self.backend_config.petsc_device,
                "pc_factor_mat_solver_type": getattr(
                    self.backend_config,
                    "pc_factor_mat_solver_type",
                    None,
                ),
                "pc_hypre_type": getattr(self.backend_config, "pc_hypre_type", None),
                "pc_gamg_type": getattr(self.backend_config, "pc_gamg_type", None),
                "petsc_options": dict(
                    getattr(self.backend_config, "petsc_options", {}) or {}
                ),
                "forward_pc_refresh_policy": getattr(
                    self.backend_config, "forward_pc_refresh_policy", "auto"
                ),
                "forward_pc_refresh_iter_threshold": int(
                    getattr(self.backend_config, "forward_pc_refresh_iter_threshold", 0)
                    or 0
                ),
                "forward_pc_refresh_lag": int(
                    getattr(self.backend_config, "forward_pc_refresh_lag", 0) or 0
                ),
                "forward_mat_solve_min_patterns": int(
                    getattr(self.backend_config, "forward_mat_solve_min_patterns", 0)
                    or 0
                ),
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
            self._last_cache_lookup = {
                "hit": False,
                "layer": "disabled",
                "artifact": "forward_factor",
            }

        rhs_matrix = np.zeros(
            (self.dofs + self.n_elec + 1, n_patterns),
            dtype=self._active_scalar_dtype(),
        )
        rhs_matrix[self.dofs : self.dofs + self.n_elec, :] = pattern_matrix.T
        rhs_matrix = self._apply_cuda_gauge_fix_rhs(rhs_matrix)
        sol_matrix = lu.solve(rhs_matrix)
        return self._as_scalar_array(sol_matrix, name="SciPy solve result")

    def _solve_full_rhs_with_scipy(
        self,
        sigma: fem.Function,
        rhs_matrix: np.ndarray,
        *,
        rhs_kind: str = "custom",
    ) -> np.ndarray:
        n_rhs = int(rhs_matrix.shape[1])
        sigma_hash = self._sigma_fingerprint(sigma)
        payload = self._base_cache_payload(sigma_hash=sigma_hash, n_patterns=n_rhs)
        payload["solver"] = "splu"
        payload["rhs_kind"] = str(rhs_kind)

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
            lu = splu(self._create_full_matrix_scipy(sigma).tocsc())
            self._last_cache_lookup = {
                "hit": False,
                "layer": "disabled",
                "artifact": "forward_factor",
            }

        rhs = self._apply_cuda_gauge_fix_rhs(
            self._as_scalar_array(rhs_matrix, name="rhs_matrix", copy=True)
        )
        return self._as_scalar_array(lu.solve(rhs), name="SciPy solve result")

    def _cuda_dense_lu_memory_skip_reason(self, exc: Exception) -> str:
        marker = "cuda_dense_lu_estimated_memory_exceeds_limit"
        backend_info = getattr(self, "_petsc_backend_info", {}) or {}
        skip_reason = str(backend_info.get("cuda_dense_lu_fallback_skip_reason") or "")
        if marker in skip_reason:
            return skip_reason
        message = str(exc)
        if marker in message:
            return message
        return ""

    def _run_cpu_scipy_fallback(
        self,
        solve_fn,
        *,
        fallback_reason: str,
        solve_start: float,
    ) -> np.ndarray:
        original_info = dict(getattr(self, "_petsc_backend_info", {}) or {})
        temp_info = dict(original_info)
        temp_info["petsc_device_effective"] = "cpu"
        temp_info["gpu_fallback_reason"] = f"cpu_scipy_fallback:{fallback_reason}"
        temp_info["fallback_reason"] = f"cpu_scipy_fallback:{fallback_reason}"
        temp_info["forward_cpu_scipy_fallback_attempted"] = True
        self._petsc_backend_info = temp_info

        succeeded = False
        try:
            result = solve_fn()
            succeeded = True
            return result
        finally:
            current_info = dict(getattr(self, "_petsc_backend_info", {}) or {})
            restored_info = dict(current_info)
            restored_info.update(original_info)
            restored_info.update(
                {
                    "gpu_fallback_reason": f"cpu_scipy_fallback:{fallback_reason}",
                    "fallback_reason": f"cpu_scipy_fallback:{fallback_reason}",
                    "forward_cpu_scipy_fallback_attempted": True,
                    "forward_cpu_scipy_fallback": succeeded,
                    "cuda_scipy_fallback": succeeded,
                    "cuda_scipy_fallback_reason": fallback_reason,
                    "forward_mat_solve_effective": "vec-loop",
                    "forward_solve_seconds": float(time.perf_counter() - solve_start),
                }
            )
            if succeeded:
                restored_info["forward_factor_backend"] = "scipy-splu"
            self._petsc_backend_info = restored_info

    def _solve_with_cpu_scipy_fallback(
        self,
        sigma: fem.Function,
        pattern_matrix: np.ndarray,
        *,
        fallback_reason: str,
        solve_start: float,
    ) -> np.ndarray:
        return self._run_cpu_scipy_fallback(
            lambda: self._solve_with_scipy(sigma, pattern_matrix),
            fallback_reason=fallback_reason,
            solve_start=solve_start,
        )

    def _solve_full_rhs_with_cpu_scipy_fallback(
        self,
        sigma: fem.Function,
        rhs_matrix: np.ndarray,
        *,
        rhs_kind: str,
        fallback_reason: str,
        solve_start: float,
    ) -> np.ndarray:
        return self._run_cpu_scipy_fallback(
            lambda: self._solve_full_rhs_with_scipy(
                sigma,
                rhs_matrix,
                rhs_kind=rhs_kind,
            ),
            fallback_reason=fallback_reason,
            solve_start=solve_start,
        )

    def _make_petsc_solver_bundle(self, system_matrix):
        if PETSc is None:
            raise RuntimeError("petsc4py is required for linear_backend='petsc'")
        if isinstance(system_matrix, csr_matrix):
            A = self._csr_to_petsc(system_matrix)
        else:
            A = system_matrix

        cuda_enabled = bool(
            getattr(self, "_petsc_backend_info", {}).get("petsc_device_effective")
            == "cuda"
        )
        requested_ksp_type = self.backend_config.ksp_type
        requested_pc_type = self.backend_config.pc_type
        solver_preset = self._solver_token(
            getattr(self.backend_config, "solver_preset", "")
        )
        explicit_amgx_request = (
            self._solver_token(requested_pc_type, "") == "amgx"
            or solver_preset in _EXPLICIT_AMGX_PRESETS
        )
        reuse_requested = bool(
            getattr(self.backend_config, "reuse_preconditioner", True)
        )
        setup_attempts = 0
        reuse_applied_by_ksp: dict[int, bool] = {}
        last_setup_error: Exception | None = None

        def _configure(ksp_obj, mat_obj, *, factor_backend=None):
            reuse_applied_by_ksp[id(ksp_obj)] = False
            ksp_obj.setOperators(mat_obj)
            ksp_obj.setType(requested_ksp_type)
            pc_obj = ksp_obj.getPC()
            pc_obj.setType(requested_pc_type)
            self._configure_pc_from_backend_config(
                pc_obj, factor_backend=factor_backend
            )
            ksp_obj.setTolerances(
                rtol=self.backend_config.rtol,
                atol=self.backend_config.atol,
                max_it=self.backend_config.max_it,
            )
            if hasattr(ksp_obj, "setReusePreconditioner"):
                try:
                    ksp_obj.setReusePreconditioner(reuse_requested)
                    reuse_applied_by_ksp[id(ksp_obj)] = True
                except Exception:
                    pass
            if self.backend_config.monitor:
                ksp_obj.setMonitor(
                    lambda _ksp, its, rnorm: print(
                        f"[KSP] iter={its} rnorm={rnorm:.3e}"
                    )
                )
            self._apply_ksp_options_database(ksp_obj)
            return pc_obj

        def _setup(ksp_obj) -> None:
            nonlocal setup_attempts
            setup_attempts += 1
            ksp_obj.setUp()

        def _destroy_failed_candidate(*objects) -> None:
            for obj in objects:
                if obj is None:
                    continue
                destroy = getattr(obj, "destroy", None)
                if callable(destroy):
                    try:
                        destroy()
                    except Exception:
                        pass

        def _bundle_from(
            ksp_obj, solve_mat_obj, *, backend_name, factor_solver_type=None
        ):
            pc_final = ksp_obj.getPC()
            self._set_backend_diagnostic(
                forward_factor_backend=backend_name,
                petsc_mat_type=(
                    str(A.getType())
                    if hasattr(A, "getType")
                    else getattr(self, "_petsc_backend_info", {}).get("petsc_mat_type")
                ),
            )
            return {
                "A": A,
                "solve_A": solve_mat_obj,
                "ksp": ksp_obj,
                "backend": backend_name,
                "ksp_type": (
                    str(ksp_obj.getType())
                    if hasattr(ksp_obj, "getType")
                    else requested_ksp_type
                ),
                "pc_type": (
                    str(pc_final.getType())
                    if hasattr(pc_final, "getType")
                    else requested_pc_type
                ),
                "factor_solver_type": factor_solver_type,
                "solve_mat_type": (
                    str(solve_mat_obj.getType())
                    if hasattr(solve_mat_obj, "getType")
                    else None
                ),
                "ksp_setup_count": int(setup_attempts),
                "reuse_preconditioner": reuse_requested,
                "reuse_preconditioner_applied": reuse_applied_by_ksp.get(
                    id(ksp_obj), False
                ),
            }

        direct_pc = requested_pc_type in {"lu", "cholesky"}
        if cuda_enabled and direct_pc:
            for candidate in ("cusparse", "cuda"):
                ksp = None
                try:
                    ksp = PETSc.KSP().create(self.mesh.comm)
                    _configure(ksp, A, factor_backend=candidate)
                    _setup(ksp)
                    return _bundle_from(
                        ksp,
                        A,
                        backend_name=f"petsc-ksp-{candidate}-{requested_pc_type}",
                        factor_solver_type=candidate,
                    )
                except Exception as exc:
                    last_setup_error = exc
                    _destroy_failed_candidate(ksp)
            dense_type = self._get_requested_dense_mat_type()
            if dense_type is not None:
                ksp = None
                solve_mat = None
                try:
                    solve_mat = self._ensure_mat_type(A.copy(), dense_type)
                    if hasattr(solve_mat, "assemble"):
                        solve_mat.assemble()
                    ksp = PETSc.KSP().create(self.mesh.comm)
                    _configure(ksp, solve_mat)
                    _setup(ksp)
                    return _bundle_from(
                        ksp,
                        solve_mat,
                        backend_name=f"petsc-ksp-{str(dense_type).lower()}-{requested_pc_type}",
                        factor_solver_type=None,
                    )
                except Exception as exc:
                    last_setup_error = exc
                    _destroy_failed_candidate(ksp)
                    if solve_mat is not A:
                        _destroy_failed_candidate(solve_mat)
        else:
            ksp = None
            try:
                ksp = PETSc.KSP().create(self.mesh.comm)
                _configure(ksp, A)
                _setup(ksp)
                return _bundle_from(
                    ksp, A, backend_name="petsc-ksp", factor_solver_type=None
                )
            except Exception as exc:
                last_setup_error = exc
                _destroy_failed_candidate(ksp)

        if explicit_amgx_request:
            reason = "explicit_pcamgx_setup_failed_refused_fallback"
            self._set_backend_diagnostic(
                gpu_fallback_reason=reason,
                fallback_reason=reason,
            )
            raise RuntimeError(
                "Explicit PETSc PCAMGX setup failed; refusing gmres/none or "
                "dense-direct fallback because explicit PCAMGX preset / "
                "pc_type='amgx' must use AmgX."
            ) from last_setup_error

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
        if hasattr(ksp, "setReusePreconditioner"):
            reuse_applied_by_ksp[id(ksp)] = False
            try:
                ksp.setReusePreconditioner(reuse_requested)
                reuse_applied_by_ksp[id(ksp)] = True
            except Exception:
                pass
        _setup(ksp)
        return _bundle_from(
            ksp,
            A,
            backend_name=f"petsc-ksp-gmres+{fallback_pc}",
            factor_solver_type=None,
        )

    def _compute_forward_ksp_structural_fingerprint(self) -> str:
        """Fingerprint of everything that must match for safe KSP reuse."""
        cfg = getattr(self, "backend_config", None)
        petsc_backend = getattr(self, "_petsc_backend_info", {}) or {}
        payload = {
            "linear_backend": str(getattr(self, "linear_backend", "petsc")),
            "forward_backend": str(getattr(self, "forward_backend", "dolfinx")),
            "solver_preset": str(getattr(cfg, "solver_preset", "auto")),
            "ksp_type": str(getattr(cfg, "ksp_type", "auto")),
            "pc_type": str(getattr(cfg, "pc_type", "auto")),
            "pc_factor_mat_solver_type": getattr(
                cfg, "pc_factor_mat_solver_type", None
            ),
            "pc_hypre_type": getattr(cfg, "pc_hypre_type", None),
            "pc_gamg_type": getattr(cfg, "pc_gamg_type", None),
            "petsc_options": dict(getattr(cfg, "petsc_options", {}) or {}),
            "rtol": float(getattr(cfg, "rtol", 1e-10)),
            "atol": float(getattr(cfg, "atol", 1e-12)),
            "max_it": int(getattr(cfg, "max_it", 2000)),
            "petsc_device_requested": str(
                petsc_backend.get(
                    "petsc_device_requested",
                    getattr(cfg, "petsc_device", "auto"),
                )
            ),
            "petsc_device_effective": str(
                petsc_backend.get("petsc_device_effective", "cpu")
            ),
            "potential_order": int(getattr(self, "potential_order", 1)),
            "electrode_model": str(getattr(self, "electrode_model", "cem")),
            "scalar_dtype": str(self._active_scalar_dtype()),
            "scalar_is_complex": self._active_scalar_is_complex(),
            "dofs": int(getattr(self, "dofs", 0)),
            "n_elec": int(getattr(self, "n_elec", 0)),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _resolve_forward_pc_refresh_policy(self) -> tuple[str, int, int]:
        cfg = getattr(self, "backend_config", None)
        policy = (
            str(getattr(cfg, "forward_pc_refresh_policy", "auto")).strip().lower()
            or "auto"
        )
        if policy not in {"auto", "never", "always", "lag"}:
            policy = "auto"
        try:
            threshold = int(getattr(cfg, "forward_pc_refresh_iter_threshold", 0) or 0)
        except (TypeError, ValueError):
            threshold = 0
        try:
            lag = int(getattr(cfg, "forward_pc_refresh_lag", 0) or 0)
        except (TypeError, ValueError):
            lag = 0
        return policy, max(0, threshold), max(0, lag)

    def _decide_pc_reuse_for_session(
        self, session: "ForwardKSPSession"
    ) -> tuple[bool, str | None]:
        """Return ``(effective_reuse, refresh_reason)`` for the next solve.

        Correctness gate: ``KSPSetReusePreconditioner(True)`` skips ``PCSetUp``
        on subsequent solves. For iterative KSP + AMG/Hypre/ILU this is only a
        staleness penalty — Krylov iterations correct the residual. But for
        ``ksp_type="preonly"`` + ``pc_type ∈ {lu, cholesky, qr}`` the PC
        application IS the solve; skipping ``PCSetUp`` after sigma changes
        means applying ``A(σ_old)^{-1}`` to the new RHS, which is wrong, not
        slow. We therefore never reuse direct factorisations across sigma
        updates, regardless of ``forward_pc_refresh_policy``.
        """
        policy, threshold, lag = self._resolve_forward_pc_refresh_policy()
        cfg = getattr(self, "backend_config", None)
        if not bool(getattr(cfg, "reuse_preconditioner", True)):
            return False, "reuse_preconditioner_disabled"

        ksp_type_cfg = str(getattr(cfg, "ksp_type", "")).strip().lower()
        pc_type_cfg = str(getattr(cfg, "pc_type", "")).strip().lower()
        ksp_type_session = str(getattr(session, "ksp_type", "")).strip().lower()
        pc_type_session = str(getattr(session, "pc_type", "")).strip().lower()
        effective_ksp_type = ksp_type_cfg or ksp_type_session
        effective_pc_type = pc_type_cfg or pc_type_session
        direct_pc_types = {"lu", "cholesky", "qr"}
        if effective_ksp_type == "preonly" and effective_pc_type in direct_pc_types:
            return False, "direct_factor_requires_rebuild"

        if policy == "never":
            return False, "policy_never"
        if policy == "always":
            return True, None
        if policy == "lag" and lag > 0 and session.solves_since_setup >= lag:
            return False, f"policy_lag_{lag}_exceeded"
        if (
            threshold > 0
            and session.last_iter_count is not None
            and session.last_iter_count > threshold
        ):
            return (
                False,
                f"iter_count_{session.last_iter_count}_gt_threshold_{threshold}",
            )
        return True, None

    def _forward_session_structurally_compatible(
        self, session: "ForwardKSPSession | None", fingerprint: str
    ) -> bool:
        if session is None:
            return False
        if self._is_native_pcamgx_route():
            # PETSc PCAMGX keeps solve state inside the PC object and can raise
            # "AmgX solve state initialisation already called" when the same
            # KSP/PC session is set up against a new conductivity matrix.  Keep
            # the persistent backend worker hot, but rebuild native PCAMGX
            # KSP/PC bundles per sigma.
            return False
        if session.structural_fingerprint != fingerprint:
            return False
        # Dense CUDA fallback bundles rebuild fresh; session reuse only safe
        # when solve_A is A itself (primary AIJ/dense-first path).
        return session.current_solve_A is session.current_A

    def _dispose_forward_ksp_session(self, session: "ForwardKSPSession | None") -> None:
        self._dispose_full_matrix_template()
        if session is None:
            return
        ksp = getattr(session, "ksp", None)
        destroy = getattr(ksp, "destroy", None)
        if callable(destroy):
            try:
                destroy()
            except Exception:
                pass
        if getattr(self, "_forward_ksp_session", None) is session:
            self._forward_ksp_session = None

    def _acquire_forward_ksp_bundle(
        self, sigma: fem.Function
    ) -> tuple[dict[str, object], "ForwardKSPSession", bool]:
        fingerprint = self._compute_forward_ksp_structural_fingerprint()
        session = getattr(self, "_forward_ksp_session", None)
        policy, _threshold, _lag = self._resolve_forward_pc_refresh_policy()

        if policy != "never" and self._forward_session_structurally_compatible(
            session, fingerprint
        ):
            effective_reuse, refresh_reason = self._decide_pc_reuse_for_session(session)
            A_new = self._create_full_matrix_petsc(sigma)
            ksp = session.ksp
            if hasattr(ksp, "setOperators"):
                try:
                    ksp.setOperators(A_new)
                except Exception:
                    # Session KSP unusable with new operator; fall through to fresh.
                    self._dispose_forward_ksp_session(session)
                    session = None
            if session is not None:
                session.current_A = A_new
                session.current_solve_A = A_new
                applied = False
                if hasattr(ksp, "setReusePreconditioner"):
                    try:
                        ksp.setReusePreconditioner(bool(effective_reuse))
                        applied = True
                    except Exception:
                        applied = False
                session.reuse_requested = bool(effective_reuse)
                session.reuse_applied = bool(applied) and bool(effective_reuse)
                if effective_reuse:
                    session.mark_reuse()
                else:
                    session.mark_refresh(refresh_reason or "policy_refresh")
                self._last_cache_lookup = {
                    "hit": True,
                    "layer": "forward_ksp_session",
                    "artifact": "forward_factor",
                }
                return session.as_bundle(), session, True

        if session is not None:
            self._dispose_forward_ksp_session(session)
        system_matrix = self._create_full_matrix_petsc(sigma)
        bundle = self._make_petsc_solver_bundle(system_matrix)
        session = ForwardKSPSession(
            ksp=bundle["ksp"],
            current_A=bundle["A"],
            current_solve_A=bundle.get("solve_A", bundle["A"]),
            backend_name=str(bundle.get("backend", "petsc-ksp")),
            ksp_type=str(
                bundle.get("ksp_type")
                or getattr(getattr(self, "backend_config", None), "ksp_type", "")
            ),
            pc_type=str(
                bundle.get("pc_type")
                or getattr(getattr(self, "backend_config", None), "pc_type", "")
            ),
            factor_solver_type=bundle.get("factor_solver_type"),
            solve_mat_type=bundle.get("solve_mat_type"),
            structural_fingerprint=fingerprint,
            reuse_requested=bool(
                bundle.get(
                    "reuse_preconditioner",
                    getattr(
                        getattr(self, "backend_config", None),
                        "reuse_preconditioner",
                        True,
                    ),
                )
            ),
            reuse_applied=bool(bundle.get("reuse_preconditioner_applied", False)),
            total_setups=int(bundle.get("ksp_setup_count", 1) or 1),
            dense_cuda_fallback=bundle.get("_dense_cuda_fallback"),
        )
        self._forward_ksp_session = session
        self._last_cache_lookup = {
            "hit": False,
            "layer": "forward_ksp_session",
            "artifact": "forward_factor",
        }
        return session.as_bundle(), session, False

    def _solve_with_petsc(self, sigma: fem.Function, pattern_matrix: np.ndarray):
        if self._is_complex_block_real_amgx_route():
            return self._solve_with_complex_block_real_amgx(sigma, pattern_matrix)

        n_patterns = pattern_matrix.shape[0]
        sigma_hash = self._sigma_fingerprint(sigma)
        _ = sigma_hash  # retained for future signature-based invalidation hooks

        setup_t0 = time.perf_counter()
        bundle, session, session_reused = self._acquire_forward_ksp_bundle(sigma)
        setup_seconds = float(time.perf_counter() - setup_t0)

        A = bundle["A"]
        solve_A = bundle.get("solve_A", A)
        ksp = bundle["ksp"]
        refresh_triggered = bool(session.last_refresh_triggered)
        cache_hit = bool(session_reused and not refresh_triggered)
        current_setup_count = 0 if cache_hit else 1
        cumulative_setup_count = int(session.total_setups)
        policy_name = self._resolve_forward_pc_refresh_policy()[0]
        self._set_backend_diagnostic(
            forward_factor_backend=session.backend_name,
            forward_factor_cache_hit=cache_hit,
            forward_rhs_count=int(n_patterns),
            forward_ksp_setup_count=current_setup_count,
            forward_ksp_setup_attempts=cumulative_setup_count,
            forward_reuse_preconditioner_requested=bool(session.reuse_requested),
            forward_reuse_preconditioner_applied=bool(session.reuse_applied),
            forward_pc_refresh_triggered=refresh_triggered,
            forward_pc_refresh_reason=session.last_refresh_reason,
            forward_pc_refresh_policy=policy_name,
            forward_pc_session_reused=bool(session_reused),
            forward_pc_session_solves=int(session.solves_since_setup),
            forward_pc_session_total_setups=cumulative_setup_count,
            forward_pc_last_iter_count=session.last_iter_count,
            forward_ksp_session=session.as_observability(
                cache_hit=cache_hit,
                session_reused=bool(session_reused),
                setup_seconds=setup_seconds,
                rhs_count=int(n_patterns),
            ),
            ksp_type=session.ksp_type
            or getattr(getattr(self, "backend_config", None), "ksp_type", None),
            pc_type=session.pc_type
            or getattr(getattr(self, "backend_config", None), "pc_type", None),
            pc_factor_mat_solver_type=session.factor_solver_type
            or getattr(
                getattr(self, "backend_config", None),
                "pc_factor_mat_solver_type",
                None,
            ),
            petsc_solve_mat_type=session.solve_mat_type,
            petsc_mat_type=(
                str(A.getType())
                if hasattr(A, "getType")
                else getattr(self, "_petsc_backend_info", {}).get("petsc_mat_type")
            ),
            forward_setup_seconds=setup_seconds,
        )

        rhs_matrix = np.zeros(
            (self.dofs + self.n_elec + 1, n_patterns),
            dtype=self._active_scalar_dtype(),
        )
        rhs_matrix[self.dofs : self.dofs + self.n_elec, :] = pattern_matrix.T
        rhs_matrix = self._apply_cuda_gauge_fix_rhs(rhs_matrix)

        use_mat_solve = self._should_use_mat_solve(n_patterns)

        solve_mat_type = str(bundle.get("solve_mat_type") or "").strip().lower()
        if (
            self._resolve_mat_solve_mode() != "off"
            and "dense" in solve_mat_type
            and hasattr(ksp, "matSolve")
        ):
            use_mat_solve = True

        backend_info = getattr(self, "_petsc_backend_info", {}) or {}
        requested_device = str(backend_info.get("petsc_device_requested", "auto"))
        effective_device = str(backend_info.get("petsc_device_effective", "cpu"))
        solver_preset = self._solver_token(
            getattr(getattr(self, "backend_config", None), "solver_preset", "")
        )
        explicit_amgx_request = (
            self._solver_token(
                getattr(getattr(self, "backend_config", None), "pc_type", ""), ""
            )
            == "amgx"
            or solver_preset in _EXPLICIT_AMGX_PRESETS
        )
        capability = (
            backend_info.get("capability")
            if isinstance(backend_info.get("capability"), dict)
            else {}
        )
        dense_mat_type = self._get_requested_dense_mat_type()

        if (
            effective_device == "cuda"
            and use_mat_solve
            and not bool(capability.get("petsc_cuda_dense", False))
        ):
            use_mat_solve = False
            self._set_backend_diagnostic(
                gpu_fallback_reason="petsc_densecuda_unavailable",
                forward_mat_solve_effective="vec-loop",
            )

        def _ksp_iteration_number(ksp_obj) -> int | None:
            if not hasattr(ksp_obj, "getIterationNumber"):
                return None
            try:
                return int(ksp_obj.getIterationNumber())
            except Exception:
                return None

        def _ksp_converged_reason(ksp_obj) -> int | None:
            if not hasattr(ksp_obj, "getConvergedReason"):
                return None
            try:
                return int(ksp_obj.getConvergedReason())
            except Exception:
                return None

        solve_t0 = time.perf_counter()
        if self._cuda_cem_requires_direct_solve(session, rhs_count=n_patterns):
            self._dispose_forward_ksp_session(session)
            return self._solve_with_cuda_dense_lu_fallback(
                A,
                rhs_matrix,
                fallback_reason="cuda_cem_reference_gauge_requires_direct_solve",
                solve_start=solve_t0,
            )
        if use_mat_solve and hasattr(ksp, "matSolve"):
            try:
                B = PETSc.Mat().createDense(
                    size=rhs_matrix.shape,
                    array=np.asfortranarray(
                        rhs_matrix,
                        dtype=self._active_scalar_dtype(),
                    ),
                    comm=self.mesh.comm,
                )
                B = self._ensure_mat_type(B, dense_mat_type)
                X = PETSc.Mat().createDense(
                    size=rhs_matrix.shape,
                    comm=self.mesh.comm,
                )
                X = self._ensure_mat_type(X, dense_mat_type)
                ksp.matSolve(B, X)
                sol = np.array(
                    X.getDenseArray(),
                    dtype=self._active_scalar_dtype(),
                    copy=True,
                )
                mat_iterations = _ksp_iteration_number(ksp)
                mat_reason = _ksp_converged_reason(ksp)
                if mat_reason is not None and mat_reason < 0:
                    B.destroy()
                    X.destroy()
                    raise RuntimeError(
                        "PETSc matSolve failed with a negative convergence reason "
                        f"({mat_reason})"
                    )
                self._set_backend_diagnostic(
                    forward_factor_backend=f"{bundle.get('backend', 'petsc-ksp')}:matsolve",
                    petsc_dense_mat_type=(
                        str(B.getType()) if hasattr(B, "getType") else dense_mat_type
                    ),
                    forward_mat_solve_effective="matsolve",
                    forward_ksp_mat_solve_count=1,
                    forward_ksp_solve_count=0,
                    forward_ksp_iterations_per_rhs=(
                        [] if mat_iterations is None else [mat_iterations]
                    ),
                    forward_ksp_iterations_total=mat_iterations,
                    forward_ksp_converged_reason=mat_reason,
                    forward_ksp_converged=(
                        None if mat_reason is None else bool(mat_reason > 0)
                    ),
                    forward_solve_seconds=float(time.perf_counter() - solve_t0),
                )
                B.destroy()
                X.destroy()
                session.record_solve(mat_iterations)
                self._set_backend_diagnostic(
                    forward_pc_last_iter_count=session.last_iter_count
                )
                return self._recenter_cuda_gauge_solution(sol)
            except Exception as exc:
                if effective_device == "cuda" and bool(
                    capability.get("petsc_cuda_dense", False)
                ):
                    dense_bundle = bundle.get("_dense_cuda_fallback")
                    if dense_bundle is None:
                        try:
                            dense_bundle = self._make_petsc_dense_solver_bundle(A)
                            bundle["_dense_cuda_fallback"] = dense_bundle
                        except Exception:
                            dense_bundle = None
                    if dense_bundle is not None and dense_bundle.get(
                        "backend"
                    ) != bundle.get("backend"):
                        dense_ksp = dense_bundle["ksp"]
                        try:
                            B = PETSc.Mat().createDense(
                                size=rhs_matrix.shape,
                                array=np.asfortranarray(
                                    rhs_matrix,
                                    dtype=self._active_scalar_dtype(),
                                ),
                                comm=self.mesh.comm,
                            )
                            B = self._ensure_mat_type(B, dense_mat_type)
                            X = PETSc.Mat().createDense(
                                size=rhs_matrix.shape, comm=self.mesh.comm
                            )
                            X = self._ensure_mat_type(X, dense_mat_type)
                            dense_ksp.matSolve(B, X)
                            sol = np.array(
                                X.getDenseArray(),
                                dtype=self._active_scalar_dtype(),
                                copy=True,
                            )
                            dense_iterations = _ksp_iteration_number(dense_ksp)
                            dense_reason = _ksp_converged_reason(dense_ksp)
                            if dense_reason is not None and dense_reason < 0:
                                B.destroy()
                                X.destroy()
                                raise RuntimeError(
                                    "PETSc dense matSolve fallback failed with a negative "
                                    f"convergence reason ({dense_reason})"
                                )
                            self._set_backend_diagnostic(
                                gpu_fallback_reason=f"matSolve_fallback:{exc}",
                                forward_mat_solve_fallback_reason=str(exc),
                                forward_factor_backend=f"{dense_bundle.get('backend', 'petsc-ksp')}:matsolve",
                                petsc_dense_mat_type=(
                                    str(B.getType())
                                    if hasattr(B, "getType")
                                    else dense_mat_type
                                ),
                                forward_mat_solve_effective="matsolve",
                                forward_ksp_mat_solve_count=1,
                                forward_ksp_solve_count=0,
                                forward_ksp_iterations_per_rhs=(
                                    []
                                    if dense_iterations is None
                                    else [dense_iterations]
                                ),
                                forward_ksp_iterations_total=dense_iterations,
                                forward_ksp_converged_reason=dense_reason,
                                forward_ksp_converged=(
                                    None
                                    if dense_reason is None
                                    else bool(dense_reason > 0)
                                ),
                                forward_solve_seconds=float(
                                    time.perf_counter() - solve_t0
                                ),
                            )
                            B.destroy()
                            X.destroy()
                            session.record_solve(dense_iterations)
                            self._set_backend_diagnostic(
                                forward_pc_last_iter_count=session.last_iter_count
                            )
                            return self._recenter_cuda_gauge_solution(sol)
                        except Exception:
                            pass
                if effective_device == "cuda" and requested_device == "cuda":
                    self._dispose_forward_ksp_session(session)
                    raise RuntimeError(
                        f"PETSc CUDA matSolve failed ({exc}). {self._actionable_cuda_guidance()}"
                    ) from exc
                self._set_backend_diagnostic(
                    gpu_fallback_reason=f"matSolve_failed: {exc}",
                    fallback_reason=f"matSolve_failed: {exc}",
                    forward_mat_solve_fallback_reason=str(exc),
                    forward_mat_solve_effective="vec-loop",
                )

        self._set_backend_diagnostic(
            forward_mat_solve_effective="vec-loop",
            forward_ksp_mat_solve_count=0,
            forward_ksp_solve_count=0,
        )
        sol_matrix = np.zeros_like(rhs_matrix)
        b = self._ensure_vec_type(
            solve_A.createVecRight(), self._get_requested_petsc_vec_type()
        )
        x = self._ensure_vec_type(
            solve_A.createVecRight(), self._get_requested_petsc_vec_type()
        )
        if hasattr(x, "getType"):
            self._set_backend_diagnostic(petsc_vec_type=str(x.getType()))
        b_array = b.getArray(readonly=False)
        iterations_per_rhs: list[int | None] = []
        for i in range(n_patterns):
            b_array[:] = rhs_matrix[:, i]
            ksp.solve(b, x)
            iterations_per_rhs.append(_ksp_iteration_number(ksp))
            reason = int(ksp.getConvergedReason())
            if reason < 0:
                self._last_cache_lookup = {
                    "hit": False,
                    "layer": "compute",
                    "artifact": "forward_factor",
                    "petsc_reason": reason,
                }
                if effective_device == "cuda":
                    self._set_backend_diagnostic(
                        gpu_fallback_reason=f"petsc_ksp_failed:{reason}",
                        fallback_reason=f"petsc_ksp_failed:{reason}",
                        forward_mat_solve_effective="vec-loop",
                        forward_ksp_solve_count=int(i + 1),
                        forward_ksp_iterations_per_rhs=iterations_per_rhs,
                        forward_ksp_iterations_total=sum(
                            int(value)
                            for value in iterations_per_rhs
                            if value is not None
                        ),
                        forward_ksp_converged_reason=reason,
                        forward_ksp_converged=False,
                        forward_solve_seconds=float(time.perf_counter() - solve_t0),
                    )
                    self._dispose_forward_ksp_session(session)
                    if explicit_amgx_request:
                        fallback_reason = (
                            f"explicit_pcamgx_solve_failed_refused_fallback:{reason}"
                        )
                        self._set_backend_diagnostic(
                            gpu_fallback_reason=fallback_reason,
                            fallback_reason=fallback_reason,
                        )
                        raise RuntimeError(
                            "Explicit PETSc PCAMGX solve failed with a negative "
                            f"convergence reason ({reason}); refusing dense-direct "
                            "or CPU fallback because explicit PCAMGX preset / "
                            "pc_type='amgx' must use AmgX."
                        )
                    try:
                        return self._solve_with_cuda_dense_lu_fallback(
                            A,
                            rhs_matrix,
                            fallback_reason=f"petsc_ksp_failed:{reason}",
                            solve_start=solve_t0,
                        )
                    except Exception as fallback_exc:
                        skip_reason = self._cuda_dense_lu_memory_skip_reason(
                            fallback_exc
                        )
                        cpu_fallback_reason = (
                            f"petsc_ksp_failed:{reason};dense_lu_skipped:{skip_reason}"
                            if skip_reason
                            else (
                                f"petsc_ksp_failed:{reason};"
                                f"dense_lu_failed:{fallback_exc}"
                            )
                        )
                        try:
                            return self._solve_with_cpu_scipy_fallback(
                                sigma,
                                pattern_matrix,
                                fallback_reason=cpu_fallback_reason,
                                solve_start=solve_t0,
                            )
                        except Exception as cpu_fallback_exc:
                            raise RuntimeError(
                                "PETSc CUDA solve failed with a negative convergence "
                                f"reason ({reason}), dense LU fallback failed "
                                f"({fallback_exc}), and CPU SciPy fallback failed "
                                f"({cpu_fallback_exc}). "
                                f"{self._actionable_cuda_guidance()}"
                            ) from cpu_fallback_exc
                self._set_backend_diagnostic(
                    fallback_reason=f"petsc_ksp_failed:{reason}",
                    forward_ksp_solve_count=int(i + 1),
                    forward_ksp_iterations_per_rhs=iterations_per_rhs,
                    forward_ksp_iterations_total=sum(
                        int(value) for value in iterations_per_rhs if value is not None
                    ),
                    forward_ksp_converged_reason=reason,
                    forward_ksp_converged=False,
                    forward_solve_seconds=float(time.perf_counter() - solve_t0),
                )
                self._dispose_forward_ksp_session(session)
                return self._solve_with_scipy(sigma, pattern_matrix)
            sol_matrix[:, i] = x.getArray(readonly=True)
        self._set_backend_diagnostic(
            forward_ksp_solve_count=int(n_patterns),
            forward_ksp_iterations_per_rhs=iterations_per_rhs,
            forward_ksp_iterations_total=sum(
                int(value) for value in iterations_per_rhs if value is not None
            ),
            forward_ksp_converged_reason=reason if n_patterns else None,
            forward_ksp_converged=None if n_patterns == 0 else bool(reason > 0),
            forward_solve_seconds=float(time.perf_counter() - solve_t0),
        )
        session.record_solve(
            sum(int(value) for value in iterations_per_rhs if value is not None)
            if n_patterns
            else None
        )
        self._set_backend_diagnostic(
            forward_pc_last_iter_count=session.last_iter_count,
            forward_ksp_session=session.as_observability(
                cache_hit=cache_hit,
                session_reused=bool(session_reused),
                setup_seconds=setup_seconds,
                rhs_count=int(n_patterns),
            ),
        )
        return self._recenter_cuda_gauge_solution(sol_matrix)

    def _solve_full_rhs_with_petsc(
        self,
        sigma: fem.Function,
        rhs_matrix: np.ndarray,
        *,
        rhs_kind: str = "custom",
    ) -> np.ndarray:
        if PETSc is None:
            raise RuntimeError("petsc4py is not available for linear_backend='petsc'")
        rhs = self._as_scalar_array(rhs_matrix, name="rhs_matrix")
        if rhs.ndim == 1:
            rhs = rhs.reshape(-1, 1)
        full_size = self.dofs + self.n_elec + 1
        if rhs.shape[0] != full_size:
            raise ValueError(
                f"full RHS row count mismatch: expected {full_size}, got {rhs.shape[0]}"
            )
        if self._is_complex_block_real_amgx_route():
            return self._solve_full_rhs_with_complex_block_real_amgx(
                sigma,
                rhs,
                rhs_kind=rhs_kind,
            )
        rhs_for_cpu_scipy = rhs.copy()
        rhs = self._apply_cuda_gauge_fix_rhs(rhs.copy())
        n_rhs = int(rhs.shape[1])

        setup_t0 = time.perf_counter()
        bundle, session, session_reused = self._acquire_forward_ksp_bundle(sigma)
        setup_seconds = float(time.perf_counter() - setup_t0)

        A = bundle["A"]
        solve_A = bundle.get("solve_A", A)
        ksp = bundle["ksp"]
        refresh_triggered = bool(session.last_refresh_triggered)
        cache_hit = bool(session_reused and not refresh_triggered)
        policy_name = self._resolve_forward_pc_refresh_policy()[0]
        self._set_backend_diagnostic(
            forward_factor_backend=session.backend_name,
            forward_factor_cache_hit=cache_hit,
            forward_rhs_kind=str(rhs_kind),
            forward_rhs_count=n_rhs,
            forward_ksp_setup_count=0 if cache_hit else 1,
            forward_ksp_setup_attempts=int(session.total_setups),
            forward_reuse_preconditioner_requested=bool(session.reuse_requested),
            forward_reuse_preconditioner_applied=bool(session.reuse_applied),
            forward_pc_refresh_triggered=refresh_triggered,
            forward_pc_refresh_reason=session.last_refresh_reason,
            forward_pc_refresh_policy=policy_name,
            forward_pc_session_reused=bool(session_reused),
            forward_pc_session_solves=int(session.solves_since_setup),
            forward_pc_session_total_setups=int(session.total_setups),
            forward_pc_last_iter_count=session.last_iter_count,
            forward_ksp_session=session.as_observability(
                cache_hit=cache_hit,
                session_reused=bool(session_reused),
                setup_seconds=setup_seconds,
                rhs_count=n_rhs,
                rhs_kind=str(rhs_kind),
            ),
            ksp_type=session.ksp_type
            or getattr(getattr(self, "backend_config", None), "ksp_type", None),
            pc_type=session.pc_type
            or getattr(getattr(self, "backend_config", None), "pc_type", None),
            petsc_mat_type=(
                str(A.getType())
                if hasattr(A, "getType")
                else getattr(self, "_petsc_backend_info", {}).get("petsc_mat_type")
            ),
            forward_setup_seconds=setup_seconds,
            forward_mat_solve_effective="vec-loop",
            forward_ksp_mat_solve_count=0,
        )

        def _ksp_iteration_number(ksp_obj) -> int | None:
            if not hasattr(ksp_obj, "getIterationNumber"):
                return None
            try:
                return int(ksp_obj.getIterationNumber())
            except Exception:
                return None

        solve_t0 = time.perf_counter()
        sol_matrix = np.zeros_like(rhs)
        b = self._ensure_vec_type(
            solve_A.createVecRight(), self._get_requested_petsc_vec_type()
        )
        x = self._ensure_vec_type(
            solve_A.createVecRight(), self._get_requested_petsc_vec_type()
        )
        if hasattr(x, "getType"):
            self._set_backend_diagnostic(petsc_vec_type=str(x.getType()))
        b_array = b.getArray(readonly=False)
        backend_info = getattr(self, "_petsc_backend_info", {}) or {}
        effective_device = str(backend_info.get("petsc_device_effective", "cpu"))
        solver_preset = self._solver_token(
            getattr(getattr(self, "backend_config", None), "solver_preset", "")
        )
        explicit_amgx_request = (
            self._solver_token(
                getattr(getattr(self, "backend_config", None), "pc_type", ""), ""
            )
            == "amgx"
            or solver_preset in _EXPLICIT_AMGX_PRESETS
        )
        iterations_per_rhs: list[int | None] = []
        reason = None
        if self._cuda_cem_requires_direct_solve(session, rhs_count=n_rhs):
            self._dispose_forward_ksp_session(session)
            return self._solve_with_cuda_dense_lu_fallback(
                A,
                rhs,
                fallback_reason="full_rhs_cuda_cem_reference_gauge_requires_direct_solve",
                solve_start=solve_t0,
            )
        for i in range(n_rhs):
            b_array[:] = rhs[:, i]
            ksp.solve(b, x)
            iterations_per_rhs.append(_ksp_iteration_number(ksp))
            reason = int(ksp.getConvergedReason())
            if reason < 0:
                self._last_cache_lookup = {
                    "hit": False,
                    "layer": "compute",
                    "artifact": "forward_factor",
                    "petsc_reason": reason,
                }
                self._set_backend_diagnostic(
                    fallback_reason=f"petsc_ksp_failed:{reason}",
                    forward_ksp_solve_count=int(i + 1),
                    forward_ksp_iterations_per_rhs=iterations_per_rhs,
                    forward_ksp_iterations_total=sum(
                        int(value) for value in iterations_per_rhs if value is not None
                    ),
                    forward_ksp_converged_reason=reason,
                    forward_ksp_converged=False,
                    forward_solve_seconds=float(time.perf_counter() - solve_t0),
                )
                self._dispose_forward_ksp_session(session)
                if effective_device == "cuda":
                    if explicit_amgx_request:
                        fallback_reason = (
                            f"explicit_pcamgx_solve_failed_refused_fallback:{reason}"
                        )
                        self._set_backend_diagnostic(
                            gpu_fallback_reason=fallback_reason,
                            fallback_reason=fallback_reason,
                        )
                        raise RuntimeError(
                            "Explicit PETSc PCAMGX full RHS solve failed with a "
                            f"negative convergence reason ({reason}); refusing "
                            "dense-direct or CPU fallback because "
                            "explicit PCAMGX preset / pc_type='amgx' must use "
                            "AmgX."
                        )
                    try:
                        return self._solve_with_cuda_dense_lu_fallback(
                            A,
                            rhs,
                            fallback_reason=f"full_rhs_petsc_ksp_failed:{reason}",
                            solve_start=solve_t0,
                        )
                    except Exception as fallback_exc:
                        skip_reason = self._cuda_dense_lu_memory_skip_reason(
                            fallback_exc
                        )
                        if skip_reason:
                            return self._solve_full_rhs_with_cpu_scipy_fallback(
                                sigma,
                                rhs_for_cpu_scipy,
                                rhs_kind=rhs_kind,
                                fallback_reason=(
                                    f"full_rhs_petsc_ksp_failed:{reason};"
                                    f"dense_lu_skipped:{skip_reason}"
                                ),
                                solve_start=solve_t0,
                            )
                        raise RuntimeError(
                            "PETSc CUDA full RHS solve failed with a negative "
                            f"convergence reason ({reason}) and dense LU fallback "
                            f"failed ({fallback_exc}). "
                            f"{self._actionable_cuda_guidance()}"
                        ) from fallback_exc
                return self._solve_full_rhs_with_scipy(sigma, rhs, rhs_kind=rhs_kind)
            sol_matrix[:, i] = x.getArray(readonly=True)

        total_iterations = sum(
            int(value) for value in iterations_per_rhs if value is not None
        )
        self._set_backend_diagnostic(
            forward_ksp_solve_count=n_rhs,
            forward_ksp_iterations_per_rhs=iterations_per_rhs,
            forward_ksp_iterations_total=total_iterations,
            forward_ksp_converged_reason=reason,
            forward_ksp_converged=None if reason is None else bool(reason > 0),
            forward_solve_seconds=float(time.perf_counter() - solve_t0),
        )
        session.record_solve(total_iterations if n_rhs else None)
        self._set_backend_diagnostic(
            forward_pc_last_iter_count=session.last_iter_count,
            forward_ksp_session=session.as_observability(
                cache_hit=cache_hit,
                session_reused=bool(session_reused),
                setup_seconds=setup_seconds,
                rhs_count=n_rhs,
                rhs_kind=str(rhs_kind),
            ),
        )
        return self._recenter_cuda_gauge_solution(sol_matrix)

    def solve_full_rhs(
        self,
        sigma: fem.Function,
        rhs_matrix: np.ndarray,
        *,
        rhs_kind: str = "custom",
    ) -> np.ndarray:
        """Solve the active CEM/PEM system for arbitrary full RHS columns.

        This is the low-level hook used by matrix-free sensitivity actions:
        ``Jv`` has potential-space RHS columns, while ``J^T r`` can use
        combined adjoint electrode-current RHS columns. The method deliberately
        shares the forward KSP session so these auxiliary solves reuse the same
        matrix and preconditioner lifecycle as ordinary CEM forward solves.
        """
        rhs = self._as_scalar_array(rhs_matrix, name="rhs_matrix")
        if rhs.ndim == 1:
            rhs = rhs.reshape(-1, 1)
        full_size = self.dofs + self.n_elec + 1
        if rhs.shape[0] != full_size:
            raise ValueError(
                f"full RHS row count mismatch: expected {full_size}, got {rhs.shape[0]}"
            )
        if self.linear_backend == "scipy":
            solution = self._solve_full_rhs_with_scipy(
                sigma,
                rhs,
                rhs_kind=rhs_kind,
            )
        elif self.linear_backend == "petsc":
            solution = self._solve_full_rhs_with_petsc(
                sigma,
                rhs,
                rhs_kind=rhs_kind,
            )
        else:
            raise ValueError(
                f"Unsupported linear_backend: {self.linear_backend}. "
                "Expected one of: 'petsc', 'scipy'."
            )
        return self._finalize_pem_solution(solution)

    def forward_solve(self, sigma: fem.Function, current_patterns=None):
        """Forward solve for given conductivity and stimulation patterns."""
        pattern_matrix = self._resolve_pattern_matrix(current_patterns)
        if self.forward_backend == "cuda_structured":
            if self._cuda_structured_backend is None:
                raise RuntimeError("cuda_structured backend was not initialized")
            self._set_backend_diagnostic(
                **self._cuda_structured_backend.backend_diagnostics()
            )
            return self._cuda_structured_backend.solve_batch(
                self._as_scalar_array(sigma.x.array, name="admittivity").astype(
                    np.float64,
                    copy=False,
                ),
                pattern_matrix,
            )
        if self.linear_backend == "scipy":
            sol_matrix = self._solve_with_scipy(sigma, pattern_matrix)
        elif self.linear_backend == "petsc":
            sol_matrix = self._solve_with_petsc(sigma, pattern_matrix)
        else:
            raise ValueError(
                f"Unsupported linear_backend: {self.linear_backend}. "
                "Expected one of: 'petsc', 'scipy'."
            )
        sol_matrix = self._finalize_pem_solution(sol_matrix)

        n_patterns = pattern_matrix.shape[0]
        potential_block = self._as_scalar_array(
            sol_matrix[: self.dofs, :],
            name="potential solution",
        )
        electrode_block = np.asarray(
            sol_matrix[self.dofs : self.dofs + self.n_elec, :].T,
            dtype=self._active_scalar_dtype(),
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
        admittivity = self._as_scalar_array(
            img.get_conductivity(),
            name="image admittivity",
        ).reshape(-1)
        sigma.x.array[:] = admittivity
        u_all, U_all = self.forward_solve(sigma)
        meas = self.pattern_manager.apply_meas_pattern(U_all)
        data_type = "complex_simulated" if np.iscomplexobj(meas) else "simulated"
        data = EITData(
            meas=meas,
            stim_pattern=self.pattern_manager.stim_matrix,
            n_elec=self.n_elec,
            n_stim=self.pattern_manager.n_stim,
            n_meas=self.pattern_manager.n_meas_total,
            type=data_type,
        )
        return data, U_all
