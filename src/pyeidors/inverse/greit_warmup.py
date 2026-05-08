"""Common 3D GREIT RM warmup artifacts for known hardware layouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from pyeidors.inverse.greit import GREITRM, GREIT_RM_HDF5_SCHEMA, load_greit_rm

GREIT_COMMON_CONFIG_WARMUP_SCHEMA = "pyeidors-greit-common-config-warmup-v1"
GREIT_COMMON_CONFIG_ENV = "PYEIDORS_GREIT_COMMON_CONFIG_DIR"


@dataclass(frozen=True)
class GREITCommonConfig:
    """Known GREIT hardware layout with an offline RM artifact slot."""

    config_id: str
    total_electrodes: int
    n_rings: int
    n_measurements: int
    voxel_shape: tuple[int, int, int]
    radius: float
    height: float
    description: str
    electrode_layout: str = "cylindrical-rings"
    mesh_dimension: int = 3

    @property
    def n_parameters(self) -> int:
        return int(np.prod(self.voxel_shape))

    @property
    def n_elec_per_ring(self) -> int:
        return max(1, self.total_electrodes // max(1, self.n_rings))

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["voxel_shape"] = [int(v) for v in self.voxel_shape]
        payload["n_parameters"] = self.n_parameters
        payload["n_elec_per_ring"] = self.n_elec_per_ring
        return payload


@dataclass(frozen=True)
class GREITCommonWarmupResult:
    """Loaded or built common-config GREIT artifact."""

    config: GREITCommonConfig
    artifact_path: Path
    greit: GREITRM
    built: bool
    loaded: bool
    prepared_online: bool

    @property
    def metadata(self) -> MappingProxyType:
        return self.greit.metadata

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": GREIT_COMMON_CONFIG_WARMUP_SCHEMA,
            "config": self.config.metadata(),
            "artifact_path": str(self.artifact_path),
            "artifact_suffix": self.artifact_path.suffix,
            "built": bool(self.built),
            "loaded": bool(self.loaded),
            "prepared_online": bool(self.prepared_online),
            "rm_shape": [int(v) for v in self.greit.rm.shape],
            "voxel_shape": [
                int(v) for v in (self.greit.voxel_shape or self.config.voxel_shape)
            ],
            "online_hot_path": self.greit.metadata.get("online_hot_path"),
            "artifact_format": self.greit.metadata.get("artifact_format"),
            "artifact_schema": self.greit.metadata.get("artifact_schema"),
            "common_config_id": self.greit.metadata.get("common_config_id"),
            "eidors_parity": bool(self.greit.metadata.get("eidors_parity", False)),
        }


_COMMON_CONFIGS = {
    "16e": GREITCommonConfig(
        config_id="16e",
        total_electrodes=16,
        n_rings=1,
        n_measurements=208,
        voxel_shape=(6, 6, 3),
        radius=0.18,
        height=0.16,
        description="16-electrode 3D cylinder baseline",
    ),
    "32e": GREITCommonConfig(
        config_id="32e",
        total_electrodes=32,
        n_rings=2,
        n_measurements=928,
        voxel_shape=(7, 7, 4),
        radius=0.18,
        height=0.16,
        description="32-electrode two-ring 3D cylinder baseline",
    ),
    "48e": GREITCommonConfig(
        config_id="48e",
        total_electrodes=48,
        n_rings=3,
        n_measurements=5936,
        voxel_shape=(8, 8, 5),
        radius=0.18,
        height=0.16,
        description="48-electrode three-ring 3D cylinder, 5936-measurement runtime gate",
    ),
}


def greit_common_config_ids() -> tuple[str, ...]:
    """Return supported common GREIT config ids."""

    return tuple(_COMMON_CONFIGS)


def normalize_greit_common_config_id(config_id: Any) -> str:
    """Normalize ids such as ``16`` or ``16e`` to the canonical form."""

    text = str(config_id or "").strip().lower().replace("_", "-")
    if text.startswith("greit-"):
        text = text.removeprefix("greit-")
    if text.endswith("-electrode"):
        text = text.removesuffix("-electrode")
    if text.isdigit():
        text = f"{text}e"
    if text.startswith("e") and text[1:].isdigit():
        text = f"{text[1:]}e"
    if text not in _COMMON_CONFIGS:
        choices = ", ".join(greit_common_config_ids())
        raise ValueError(
            f"Unknown GREIT common config {config_id!r}; choices: {choices}"
        )
    return text


def greit_common_config(config_id: Any) -> GREITCommonConfig:
    """Return a common GREIT config by id."""

    return _COMMON_CONFIGS[normalize_greit_common_config_id(config_id)]


def greit_common_config_dir(path: str | Path | None = None) -> Path:
    """Resolve the common-config artifact directory."""

    if path is not None:
        return Path(path).expanduser()
    env_value = os.environ.get(GREIT_COMMON_CONFIG_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return Path(".pyeidors_cache") / "greit_common_configs"


def greit_common_config_artifact_path(
    config_id: Any,
    artifact_dir: str | Path | None = None,
) -> Path:
    """Return the canonical HDF5 artifact path for a common GREIT config."""

    cfg = greit_common_config(config_id)
    return greit_common_config_dir(artifact_dir) / f"greit3d_common_{cfg.config_id}.h5"


def resolve_greit_common_config_artifact_path(
    config_id: Any,
    artifact_dir: str | Path | None = None,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a common-config HDF5 artifact without building it."""

    path = greit_common_config_artifact_path(config_id, artifact_dir)
    if path.exists() or not must_exist:
        return path
    raise FileNotFoundError(
        "GREIT common-config artifact does not exist; precompute/register it offline "
        f"before GUI use: {path}"
    )


def resolve_greit_common_config_artifact_path_from_meta(
    meta: Mapping[str, Any],
    *,
    must_exist: bool = True,
) -> Path | None:
    """Resolve a common GREIT artifact from GUI/runtime metadata."""

    explicit_path = _first_meta_value(
        meta,
        "greit_common_config_artifact_path",
        "greit_common_config_path",
        "common_greit_rm_path",
    )
    if explicit_path is not None:
        path = Path(str(explicit_path)).expanduser()
        if path.exists() or not must_exist:
            return path
        raise FileNotFoundError(
            f"GREIT common-config artifact path does not exist: {path}"
        )

    config_id = _first_meta_value(
        meta,
        "greit_common_config",
        "greit_common_config_id",
        "common_greit_config",
        "common_config",
    )
    if config_id is None or not str(config_id).strip():
        return None
    artifact_dir = _first_meta_value(
        meta,
        "greit_common_config_dir",
        "greit_common_artifact_dir",
        "common_greit_artifact_dir",
        "rm_artifact_dir",
    )
    return resolve_greit_common_config_artifact_path(
        config_id,
        artifact_dir=artifact_dir,
        must_exist=must_exist,
    )


def precompute_greit_common_config(
    config_id: Any,
    *,
    artifact_dir: str | Path | None = None,
    overwrite: bool = False,
    prepare_online: bool = False,
    device: str = "auto",
    dtype: str | np.dtype[Any] = "float64",
) -> GREITCommonWarmupResult:
    """Materialize a common-config GREIT HDF5 artifact.

    The default builder is a deterministic linearized fixture so T48 can test
    the warm path without a forward/Jacobian cold build. Official EIDORS
    artifacts should be registered with :func:`register_greit_common_config_artifact`.
    """

    cfg = greit_common_config(config_id)
    path = greit_common_config_artifact_path(cfg.config_id, artifact_dir)
    if path.exists() and not overwrite:
        return load_greit_common_config(
            cfg.config_id,
            artifact_dir=artifact_dir,
            prepare_online=prepare_online,
            device=device,
            dtype=dtype,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    rm = _deterministic_fixture_rm(cfg)
    metadata = _common_metadata(
        cfg,
        {
            "artifact_schema": GREIT_RM_HDF5_SCHEMA,
            "eidors_parity": False,
            "training_mode": "deterministic-linearized-fixture",
            "warmup_builder": "deterministic_fixture_rm",
            "fixture_only": True,
            "official_parity_gate": "T49",
        },
    )
    greit = GREITRM(
        rm=rm,
        metadata=MappingProxyType(metadata),
        voxel_shape=cfg.voxel_shape,
    )
    saved = greit.save(path)
    loaded = load_greit_rm(saved)
    if prepare_online:
        loaded = loaded.prepare_online(
            device=device,
            dtype=dtype,
            cache_key=str(saved),
        )
    return GREITCommonWarmupResult(
        config=cfg,
        artifact_path=saved,
        greit=loaded,
        built=True,
        loaded=True,
        prepared_online=bool(prepare_online),
    )


def register_greit_common_config_artifact(
    config_id: Any,
    source_artifact: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    overwrite: bool = False,
    prepare_online: bool = False,
    device: str = "auto",
    dtype: str | np.dtype[Any] = "float64",
    strict_shape: bool = True,
) -> GREITCommonWarmupResult:
    """Register an externally built GREIT RM artifact under a common config id."""

    cfg = greit_common_config(config_id)
    target = greit_common_config_artifact_path(cfg.config_id, artifact_dir)
    if target.exists() and not overwrite:
        raise FileExistsError(f"GREIT common-config artifact already exists: {target}")

    greit = load_greit_rm(source_artifact)
    _validate_common_config_rm_shape(cfg, greit.rm.shape, strict_shape=strict_shape)
    metadata = _common_metadata(
        cfg,
        {
            **dict(greit.metadata),
            "registered_from_artifact": str(Path(source_artifact)),
            "warmup_builder": "registered_external_greit_rm",
            "fixture_only": bool(greit.metadata.get("fixture_only", False)),
        },
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    saved = replace(greit, metadata=MappingProxyType(metadata)).save(target)
    loaded = load_greit_rm(saved)
    if prepare_online:
        loaded = loaded.prepare_online(
            device=device,
            dtype=dtype,
            cache_key=str(saved),
        )
    return GREITCommonWarmupResult(
        config=cfg,
        artifact_path=saved,
        greit=loaded,
        built=True,
        loaded=True,
        prepared_online=bool(prepare_online),
    )


def load_greit_common_config(
    config_id: Any,
    *,
    artifact_dir: str | Path | None = None,
    prepare_online: bool = False,
    device: str = "auto",
    dtype: str | np.dtype[Any] = "float64",
) -> GREITCommonWarmupResult:
    """Load a precomputed common-config artifact and optionally prepare matmul."""

    cfg = greit_common_config(config_id)
    path = resolve_greit_common_config_artifact_path(cfg.config_id, artifact_dir)
    greit = load_greit_rm(path)
    _validate_common_config_rm_shape(cfg, greit.rm.shape, strict_shape=True)
    if prepare_online:
        greit = greit.prepare_online(device=device, dtype=dtype, cache_key=str(path))
    return GREITCommonWarmupResult(
        config=cfg,
        artifact_path=path,
        greit=greit,
        built=False,
        loaded=True,
        prepared_online=bool(prepare_online),
    )


def common_config_runtime_metadata(
    config_id: Any,
    *,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return metadata for GUI cached-RM hot-path requests."""

    cfg = greit_common_config(config_id)
    path = resolve_greit_common_config_artifact_path(
        cfg.config_id,
        artifact_dir=artifact_dir,
        must_exist=True,
    )
    return {
        "reconstruction_runtime": "single_step_cached",
        "greit_common_config": cfg.config_id,
        "greit_common_config_artifact_path": str(path),
        "greit_common_config_dir": str(path.parent),
        "n_elec": cfg.total_electrodes,
        "n_rings": cfg.n_rings,
        "mesh_dimension": cfg.mesh_dimension,
        "radius": cfg.radius,
        "height": cfg.height,
        "rm_voxel_shape": cfg.voxel_shape,
        "online_hot_path": "rm_matmul",
    }


def _common_metadata(
    cfg: GREITCommonConfig,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    meta = dict(metadata or {})
    meta.update(
        {
            "algorithm": "greit-3d",
            "artifact_format": "hdf5",
            "common_config_schema": GREIT_COMMON_CONFIG_WARMUP_SCHEMA,
            "common_config_id": cfg.config_id,
            "known_hardware_config": True,
            "online_hot_path": "rm_matmul",
            "n_elec": cfg.total_electrodes,
            "total_electrodes": cfg.total_electrodes,
            "n_rings": cfg.n_rings,
            "n_elec_per_ring": cfg.n_elec_per_ring,
            "n_measurements": cfg.n_measurements,
            "n_parameters": cfg.n_parameters,
            "voxel_shape": cfg.voxel_shape,
            "mesh_dimension": cfg.mesh_dimension,
            "radius": cfg.radius,
            "height": cfg.height,
            "electrode_layout": cfg.electrode_layout,
            "description": cfg.description,
            "gui_cold_build_allowed": False,
            "official_fixture_scope": "48e official fixture passed",
            "protocol_5936_official_status": "pending_T97",
            "official_equivalence_claim_allowed": False,
            "official_equivalence_scope": (
                "48e official fixture only; 5936 protocol official fixture pending T97"
            ),
        }
    )
    meta.setdefault("artifact_schema", GREIT_RM_HDF5_SCHEMA)
    return meta


def _deterministic_fixture_rm(cfg: GREITCommonConfig) -> np.ndarray:
    rows = np.arange(cfg.n_parameters, dtype=np.float64)[:, None] + 1.0
    cols = np.arange(cfg.n_measurements, dtype=np.float64)[None, :] + 1.0
    scale = max(float(np.sqrt(cfg.n_measurements)), 1.0)
    matrix = (
        np.sin(rows * (cols + 3.0) * 0.00037) + np.cos((rows + 5.0) * cols * 0.00019)
    ) / (2.0 * scale)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _validate_common_config_rm_shape(
    cfg: GREITCommonConfig,
    shape: tuple[int, ...],
    *,
    strict_shape: bool,
) -> None:
    if len(shape) != 2 or 0 in shape:
        raise ValueError(f"GREIT RM must be non-empty 2D, got {shape}.")
    expected = (cfg.n_parameters, cfg.n_measurements)
    if strict_shape and tuple(int(v) for v in shape) != expected:
        raise ValueError(
            f"GREIT common config {cfg.config_id} expects RM shape {expected}, "
            f"got {shape}."
        )


def _first_meta_value(meta: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = meta.get(key)
        if value is not None and str(value).strip():
            return value
    return None


__all__ = [
    "GREIT_COMMON_CONFIG_ENV",
    "GREIT_COMMON_CONFIG_WARMUP_SCHEMA",
    "GREITCommonConfig",
    "GREITCommonWarmupResult",
    "common_config_runtime_metadata",
    "greit_common_config",
    "greit_common_config_artifact_path",
    "greit_common_config_dir",
    "greit_common_config_ids",
    "load_greit_common_config",
    "normalize_greit_common_config_id",
    "precompute_greit_common_config",
    "register_greit_common_config_artifact",
    "resolve_greit_common_config_artifact_path",
    "resolve_greit_common_config_artifact_path_from_meta",
]
