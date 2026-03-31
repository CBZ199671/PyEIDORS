"""Shared policy for 3D performance defaults, experimental switches, and benchmark contracts."""

from __future__ import annotations

from collections.abc import Iterable

FEATURE_MODE_OFF = "off"
FEATURE_MODE_AUTO = "auto"
FEATURE_MODE_ON = "on"
FEATURE_MODE_VALUES = (FEATURE_MODE_OFF, FEATURE_MODE_AUTO, FEATURE_MODE_ON)

PETSC_DEVICE_AUTO = "auto"
PETSC_DEVICE_CPU = "cpu"
PETSC_DEVICE_CUDA = "cuda"
PETSC_DEVICE_VALUES = (PETSC_DEVICE_AUTO, PETSC_DEVICE_CPU, PETSC_DEVICE_CUDA)

FORWARD_BACKEND_DOLFINX = "dolfinx"
FORWARD_BACKEND_CUDA_STRUCTURED = "cuda_structured"
FORWARD_BACKEND_VALUES = (
    FORWARD_BACKEND_DOLFINX,
    FORWARD_BACKEND_CUDA_STRUCTURED,
)

MESH_FAMILY_TETRA = "tetra"
MESH_FAMILY_HEX = "hex"
MESH_FAMILY_VALUES = (MESH_FAMILY_TETRA, MESH_FAMILY_HEX)

GEOMETRY_VERSION_LEGACY = "legacy"
GEOMETRY_VERSION_GEOMV2 = "geomv2"
LEGACY_3D_GENERATOR_REVISION = "g3d0"
SQUARE_TO_DISK_3D_GENERATOR_REVISION = "g3d2"
DEFAULT_3D_GENERATOR_REVISION = "g3d3"

DEFAULT_ROM_MODE = FEATURE_MODE_OFF
DEFAULT_INEXACT_MODE = FEATURE_MODE_OFF
DEFAULT_LOWRANK_MODE = FEATURE_MODE_OFF

DEFAULT_SOLVER_MODE_2D = "strict"
DEFAULT_SOLVER_MODE_3D = "fast"
DEFAULT_LINE_SEARCH_MODE_2D = "full"
DEFAULT_LINE_SEARCH_MODE_3D = "fast"
DEFAULT_FORWARD_MAT_SOLVE_2D = "off"
DEFAULT_FORWARD_MAT_SOLVE_3D_FAST = "auto"

DEFAULT_LINEAR_SOLVER = "auto"
DEFAULT_PRECONDITIONER = "auto"
DEFAULT_FAST_LINEAR_PATH = "auto"
DEFAULT_FORWARD_MAT_SOLVE = "auto"
DEFAULT_PETSC_DEVICE = PETSC_DEVICE_AUTO
DEFAULT_ABSOLUTE_STARTUP_CACHE = "on"
DEFAULT_FORWARD_BACKEND = FORWARD_BACKEND_DOLFINX
DEFAULT_MESH_FAMILY = MESH_FAMILY_TETRA
DEFAULT_3D_GEOMETRY_VERSION = GEOMETRY_VERSION_GEOMV2

DEFAULT_ROM_RANK_GLOBAL = 32
DEFAULT_ROM_RANK_ADAPTIVE = 16
DEFAULT_ROM_REFRESH_EVERY = 2
DEFAULT_ROM_SNAPSHOT_SOURCE = "hybrid"
DEFAULT_INEXACT_FORCING = "eisenstat-walker"
DEFAULT_INEXACT_ETA0 = 0.2
DEFAULT_INEXACT_ETA_MIN = 1e-3
DEFAULT_INEXACT_ETA_MAX = 0.5
DEFAULT_LOWRANK_RANK = 16
DEFAULT_LOWRANK_METHOD = "tsvd"
DEFAULT_LOWRANK_ENERGY = 0.995
DEFAULT_CHOLMOD_MAX_N = 50000
DEFAULT_CHOLMOD_MAX_MEMORY_GIB = 4.0
DEFAULT_JACOBIAN_BLOCK_TUNE = "auto"
DEFAULT_JACOBIAN_BLOCK_SIZE = 0
DEFAULT_JACOBIAN_BLOCK_CANDIDATES = (64, 128, 256, 512)

PROFILE_A_BASELINE = "A_baseline"
PROFILE_B_CHOLMOD_ONLY = "B_cholmod_only"
PROFILE_C_AUTOTUNE_ONLY = "C_autotune_only"
PROFILE_D_COMBINED = "D_combined"
PROFILE_E_FUSED = "E_fused"

PRIMARY_PERF_PROFILE = PROFILE_D_COMBINED
EXPERIMENTAL_PERF_PROFILES = (PROFILE_E_FUSED,)
QUICK_PERF_PROFILES = (PROFILE_A_BASELINE, PROFILE_D_COMBINED)
FULL_PERF_PROFILES = (
    PROFILE_A_BASELINE,
    PROFILE_B_CHOLMOD_ONLY,
    PROFILE_C_AUTOTUNE_ONLY,
    PROFILE_D_COMBINED,
    PROFILE_E_FUSED,
)

QUICK_BENCHMARK_PEAK_OVERHEAD_LIMIT = 0.10
PERF_GATE_PEAK_MEMORY_LIMIT_RATIO = 1.10
PERF_GATE_AUTOTUNE_JACOBIAN_SPEEDUP_REF2 = 1.10
PERF_GATE_COMBINED_TOTAL_TARGETS = {
    1: 0.99,
    2: 1.01,
}


def normalize_mode(value: object, *, valid: tuple[str, ...], default: str) -> str:
    """Normalize a string mode against an explicit allow-list."""
    mode = str(value).strip().lower()
    return mode if mode in valid else default


def normalize_feature_mode(value: object, *, default: str = FEATURE_MODE_OFF) -> str:
    """Normalize experimental feature modes to ``off|auto|on``."""
    return normalize_mode(value, valid=FEATURE_MODE_VALUES, default=default)


def normalize_petsc_device(value: object, *, default: str = DEFAULT_PETSC_DEVICE) -> str:
    """Normalize PETSc FEM device policy to ``auto|cpu|cuda``."""
    return normalize_mode(value, valid=PETSC_DEVICE_VALUES, default=default)


def normalize_forward_backend(value: object, *, default: str = DEFAULT_FORWARD_BACKEND) -> str:
    """Normalize forward discretization backend to a supported runtime label."""
    return normalize_mode(value, valid=FORWARD_BACKEND_VALUES, default=default)


def normalize_mesh_family(value: object, *, default: str = DEFAULT_MESH_FAMILY) -> str:
    """Normalize 3D mesh cell family to a supported runtime label."""
    return normalize_mode(value, valid=MESH_FAMILY_VALUES, default=default)


def resolve_solver_mode(value: object, *, mesh_dim: int) -> str:
    """Resolve CLI/runtime solver mode, keeping 3D fast and 2D strict as defaults."""
    mode = normalize_mode(value, valid=("auto", "strict", "fast"), default="auto")
    if mode != "auto":
        return mode
    return DEFAULT_SOLVER_MODE_3D if int(mesh_dim) == 3 else DEFAULT_SOLVER_MODE_2D


def resolve_line_search_mode(value: object, *, mesh_dim: int) -> str:
    """Resolve line-search mode with 3D fast and 2D full defaults."""
    mode = normalize_mode(value, valid=("auto", "full", "fast"), default="auto")
    if mode != "auto":
        return mode
    return DEFAULT_LINE_SEARCH_MODE_3D if int(mesh_dim) == 3 else DEFAULT_LINE_SEARCH_MODE_2D


def resolve_experimental_mode(value: object) -> str:
    """Keep experimental features opt-in only; ``auto`` is coerced to ``off``."""
    mode = normalize_feature_mode(value, default=FEATURE_MODE_OFF)
    return FEATURE_MODE_OFF if mode == FEATURE_MODE_AUTO else mode


def resolve_forward_mat_solve(value: object, *, mesh_dim: int, solver_mode: str) -> str:
    """Resolve the forward matSolve policy for the current mesh/solver mode."""
    mode = normalize_mode(value, valid=("auto", "off", "on"), default="auto")
    if mode != "auto":
        return mode
    if int(mesh_dim) == 3 and str(solver_mode).strip().lower() == "fast":
        return DEFAULT_FORWARD_MAT_SOLVE_3D_FAST
    return DEFAULT_FORWARD_MAT_SOLVE_2D


def parse_block_size_candidates(value: object) -> list[int]:
    """Parse a comma-separated or iterable block-size candidate list."""
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
    elif isinstance(value, Iterable):
        tokens = [str(token).strip() for token in value if str(token).strip()]
    elif value is None:
        tokens = []
    else:
        token = str(value).strip()
        tokens = [token] if token else []

    try:
        candidates = sorted({int(token) for token in tokens if int(token) > 0})
    except ValueError as exc:
        raise ValueError(f"invalid block-size candidate list: {value!r}") from exc
    if not candidates:
        raise ValueError("block-size candidate list must include at least one positive integer")
    return candidates


def is_experimental_profile(name: str) -> bool:
    """Return True when a benchmark profile is experimental-only."""
    return str(name) in EXPERIMENTAL_PERF_PROFILES
