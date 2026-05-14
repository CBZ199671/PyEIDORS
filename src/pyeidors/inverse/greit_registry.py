"""Config-driven GREIT artifact registry.

This module is intentionally a thin orchestration layer around
``pyeidors.inverse.greit``.  The numerical GREIT components stay there; this
file owns V92 signatures, manifest bookkeeping, and native artifact builds.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from pyeidors.inverse.greit import (
    GREIT_EIDORS_HDF5_SCHEMA,
    GREITRM,
    build_greit3d_distribution,
    build_native_greit_training_pipeline,
    load_greit_rm,
)

GREIT_ARTIFACT_REGISTRY_SCHEMA = "pyeidors-greit-artifact-registry-v1"
GREIT_ARTIFACT_SIGNATURE_SCHEMA = "pyeidors-greit-artifact-signature-v1"
GREIT_NATIVE_BUILDER_VERSION = "native-greit-finite-target-v2"
GREIT_REGISTRY_ENV = "PYEIDORS_GREIT_ARTIFACT_REGISTRY_DIR"


@dataclass(frozen=True)
class GREITRegistryLookup:
    """Resolved GREIT artifact registry entry."""

    signature: str
    signature_payload: MappingProxyType
    artifact_path: Path
    manifest_path: Path
    greit: GREITRM
    built: bool
    registered: bool
    cache_status: str
    backend: str


GREITArtifactBuilder = Callable[[Mapping[str, Any], Mapping[str, Any], Path], GREITRM]


def greit_registry_dir(path: str | Path | None = None) -> Path:
    """Resolve the on-disk GREIT registry directory."""

    if path is not None:
        return Path(path).expanduser()
    import os

    env_value = os.environ.get(GREIT_REGISTRY_ENV)
    if env_value:
        return Path(env_value).expanduser()
    return Path(".pyeidors_cache") / "greit_artifacts"


def greit_registry_manifest_path(path: str | Path | None = None) -> Path:
    """Return registry manifest path."""

    return greit_registry_dir(path) / "manifest.json"


def greit_artifact_path_for_signature(
    signature: str,
    *,
    registry_dir: str | Path | None = None,
) -> Path:
    """Return canonical artifact path for a GREIT signature."""

    sig = str(signature).strip().lower()
    if not sig:
        raise ValueError("GREIT signature is required.")
    return greit_registry_dir(registry_dir) / f"greit3d_{sig[:24]}.h5"


def greit_artifact_signature_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build V92 hard-field signature payload from a mapping."""

    cfg = dict(config)
    measurement_count = _int_first(
        cfg,
        "measurement_count",
        "n_measurements",
        "points_per_frame",
    )
    channel_order = cfg.get("channel_order")
    if (
        channel_order is None
        and measurement_count is not None
        and measurement_count > 0
    ):
        channel_order = np.arange(int(measurement_count), dtype=np.int64)
    imgsz = _shape_tuple(cfg.get("imgsz", cfg.get("greit_imgsz")))
    xvec = cfg.get("xvec", cfg.get("greit_xvec"))
    yvec = cfg.get("yvec", cfg.get("greit_yvec"))
    zvec = cfg.get("zvec", cfg.get("greit_zvec"))
    downsample = cfg.get("downsample", cfg.get("greit_downsample"))
    mesh_dimension = _int_or_none(cfg.get("mesh_dimension"))
    rec_mask = cfg.get("rec_mask", cfg.get("greit_rec_mask"))
    if rec_mask is None and mesh_dimension == 3:
        rec_mask = "cylindrical_fem_volume_v1"
    point_in_volume_signature = cfg.get("point_in_volume_signature")
    if point_in_volume_signature is None and mesh_dimension == 3:
        point_in_volume_signature = "analytic_cylinder_radius_height_v1"
    rec_grid = {
        "imgsz": imgsz,
        "xvec": xvec,
        "yvec": yvec,
        "zvec": zvec,
        "downsample": downsample,
        "mask": rec_mask,
        "point_in_volume_signature": point_in_volume_signature,
    }
    target_radius_effective = _greit_target_radius_from_config(
        cfg,
        radius=_float_or_none(cfg.get("radius")) or 1.0,
    )
    stim_payload = {
        "stim_pattern": cfg.get("stim_pattern"),
        "custom_stim_matrix": cfg.get("custom_stim_matrix"),
        "drive_mode": cfg.get("drive_mode"),
        "drive_value": cfg.get("drive_value"),
    }
    meas_payload = {
        "meas_pattern": cfg.get("meas_pattern"),
        "custom_meas_matrices": cfg.get("custom_meas_matrices"),
        "rotate_meas": cfg.get("rotate_meas"),
        "use_meas_current": cfg.get("use_meas_current"),
        "use_meas_current_next": cfg.get("use_meas_current_next"),
        "stim_direction": cfg.get("stim_direction"),
        "meas_direction": cfg.get("meas_direction"),
        "stim_first_positive": cfg.get("stim_first_positive"),
        "measurement_protocol": cfg.get("measurement_protocol"),
    }
    payload = {
        "schema": GREIT_ARTIFACT_SIGNATURE_SCHEMA,
        "mesh_dimension": mesh_dimension,
        "mesh_hash": str(
            cfg.get("mesh_hash")
            or cfg.get("fwd_model_signature")
            or _digest_json(
                {
                    "mesh_dimension": cfg.get("mesh_dimension"),
                    "mesh_refinement": cfg.get("mesh_refinement", cfg.get("mesh_size")),
                    "mesh_family": cfg.get("mesh_family"),
                    "geometry_version": cfg.get("geometry_version"),
                    "radius": cfg.get("radius"),
                    "height": cfg.get("height"),
                }
            )
        ),
        "fwd_model_hash": str(
            cfg.get("fwd_model_hash")
            or cfg.get("fwd_model_signature")
            or cfg.get("mesh_hash")
            or ""
        ),
        "n_elec": _int_or_none(cfg.get("n_elec", cfg.get("n_electrodes"))),
        "n_rings": _int_or_none(cfg.get("n_rings")),
        "n_layers": _int_or_none(cfg.get("n_layers", cfg.get("n_rings"))),
        "radius": _float_or_none(cfg.get("radius")),
        "height": _float_or_none(cfg.get("height")),
        "electrode_length_2d": _canonical_value(
            cfg.get("electrode_length_2d", cfg.get("electrode_length_m_override"))
        ),
        "electrode_area_3d": _float_or_none(
            cfg.get("electrode_area_3d", cfg.get("electrode_area_m2_override"))
        ),
        "electrode_height_ratio": _float_or_none(cfg.get("electrode_height_ratio")),
        "electrode_level_fractions": _canonical_value(
            cfg.get("electrode_level_fractions")
        ),
        "electrode_numbering_layout": str(
            cfg.get("electrode_numbering_layout") or cfg.get("electrode_layout") or ""
        ),
        "stim_pattern": _canonical_value(cfg.get("stim_pattern")),
        "stim_pattern_hash": _digest_json(stim_payload),
        "meas_pattern": _canonical_value(cfg.get("meas_pattern")),
        "meas_pattern_hash": _digest_json(meas_payload),
        "measurement_count": measurement_count,
        "channel_order_hash": _digest_json(channel_order),
        "bad_channel_mask": _canonical_value(
            cfg.get("bad_channel_mask", cfg.get("channel_mask"))
        ),
        "background_ref": {
            "background_conductivity": _float_or_none(
                cfg.get("background_conductivity", cfg.get("sigma0"))
            ),
            "contact_impedance": _canonical_value(
                cfg.get("contact_impedance", cfg.get("z0"))
            ),
            "reference_policy": cfg.get("reference_policy", "homogeneous"),
        },
        "normalize_measurements": _bool_value(
            cfg.get(
                "normalize_measurements",
                cfg.get("normalize", cfg.get("difference_mode", "normalized")),
            )
        ),
        "rec_model_signature": str(
            cfg.get("rec_model_signature") or _digest_json(rec_grid)
        ),
        "greit_rec_grid": _canonical_value(rec_grid),
        "target_distribution": _canonical_value(
            cfg.get("target_distribution", cfg.get("distr"))
        ),
        "target_size": _float_or_none(
            cfg.get("target_size", cfg.get("greit_target_size"))
        ),
        "target_radius": _float_or_none(
            cfg.get("target_radius", cfg.get("greit_target_radius"))
        ),
        "target_size_semantics": str(
            cfg.get("target_size_semantics") or "fraction_of_tank_radius"
        ),
        "target_radius_effective": target_radius_effective,
        "target_contrast": _canonical_value(
            cfg.get("target_contrast", cfg.get("greit_target_contrast", 1.0))
        ),
        "desired_solution_fn": str(
            cfg.get("desired_solution_fn")
            or cfg.get("greit_desired_solution_fn")
            or "GREIT_desired_img_sigmoid"
        ),
        "desired_solution_params": _canonical_value(
            cfg.get("desired_solution_params", cfg.get("greit_desired_options"))
        ),
        "noise_covar": _canonical_value(cfg.get("noise_covar", cfg.get("Sn", 1.0))),
        "weight": _float_or_none(cfg.get("weight", cfg.get("greit_weight"))),
        "noise_figure": _float_or_none(
            cfg.get("noise_figure", cfg.get("greit_noise_figure"))
        ),
        "image_SNR": _float_or_none(cfg.get("image_SNR", cfg.get("image_snr"))),
        "training_mode": str(cfg.get("training_mode", "forward")).strip().lower(),
        "artifact_schema": str(cfg.get("artifact_schema") or GREIT_EIDORS_HDF5_SCHEMA),
        "builder_backend": str(cfg.get("builder_backend", "native")).strip().lower(),
        "builder_semantic_version": str(
            cfg.get("builder_semantic_version") or GREIT_NATIVE_BUILDER_VERSION
        ),
    }
    return payload


def greit_artifact_signature(config: Mapping[str, Any]) -> str:
    """Hash V92 GREIT artifact signature payload."""

    return _digest_json(greit_artifact_signature_payload(config))


def load_greit_registry_manifest(
    registry_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load registry manifest; missing file returns an empty manifest."""

    manifest_path = greit_registry_manifest_path(registry_dir)
    if not manifest_path.exists():
        return {
            "schema": GREIT_ARTIFACT_REGISTRY_SCHEMA,
            "entries": {},
        }
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != GREIT_ARTIFACT_REGISTRY_SCHEMA:
        raise ValueError(
            f"Unsupported GREIT registry manifest schema {manifest.get('schema')!r}."
        )
    manifest.setdefault("entries", {})
    return manifest


def write_greit_registry_manifest(
    manifest: Mapping[str, Any],
    registry_dir: str | Path | None = None,
) -> Path:
    """Write registry manifest."""

    manifest_path = greit_registry_manifest_path(registry_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": GREIT_ARTIFACT_REGISTRY_SCHEMA,
        "entries": dict(manifest.get("entries", {})),
    }
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(_canonical_value(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest_path


def resolve_greit_artifact(
    config: Mapping[str, Any],
    *,
    registry_dir: str | Path | None = None,
    prepare_online: bool = False,
    device: str = "auto",
    dtype: str | np.dtype[Any] = "float64",
) -> GREITRegistryLookup | None:
    """Resolve an exact-match GREIT artifact, returning ``None`` on miss."""

    payload = greit_artifact_signature_payload(config)
    signature = _digest_json(payload)
    manifest = load_greit_registry_manifest(registry_dir)
    entry = dict(manifest.get("entries", {}).get(signature, {}))
    candidate_path = None
    if entry.get("artifact_path"):
        candidate_path = Path(str(entry["artifact_path"])).expanduser()
    if candidate_path is None:
        candidate_path = greit_artifact_path_for_signature(
            signature,
            registry_dir=registry_dir,
        )
    if not candidate_path.exists():
        return None
    greit = load_greit_rm(candidate_path)
    _validate_loaded_signature(greit, signature=signature)
    if prepare_online:
        greit = greit.prepare_online(
            device=device, dtype=dtype, cache_key=str(candidate_path)
        )
    return GREITRegistryLookup(
        signature=signature,
        signature_payload=MappingProxyType(payload),
        artifact_path=candidate_path,
        manifest_path=greit_registry_manifest_path(registry_dir),
        greit=greit,
        built=False,
        registered=False,
        cache_status="disk_hit",
        backend=str(payload["builder_backend"]),
    )


def register_greit_artifact(
    config: Mapping[str, Any],
    artifact_path: str | Path,
    *,
    registry_dir: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> GREITRegistryLookup:
    """Register an externally or freshly built artifact under exact signature."""

    payload = greit_artifact_signature_payload(config)
    signature = _digest_json(payload)
    path = Path(artifact_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"GREIT artifact path does not exist: {path}")
    greit = load_greit_rm(path)
    _validate_loaded_signature(greit, signature=signature, allow_missing=True)
    manifest = load_greit_registry_manifest(registry_dir)
    entry = {
        "signature": signature,
        "artifact_path": str(path),
        "artifact_schema": str(greit.metadata.get("artifact_schema", "")),
        "builder_backend": str(payload["builder_backend"]),
        "builder_semantic_version": str(payload["builder_semantic_version"]),
        "n_measurements": int(np.asarray(greit.rm).shape[1]),
        "n_parameters": int(np.asarray(greit.rm).shape[0]),
        "signature_payload": payload,
        "provenance": dict(provenance or {}),
    }
    manifest.setdefault("entries", {})[signature] = entry
    manifest_path = write_greit_registry_manifest(manifest, registry_dir)
    return GREITRegistryLookup(
        signature=signature,
        signature_payload=MappingProxyType(payload),
        artifact_path=path,
        manifest_path=manifest_path,
        greit=greit,
        built=False,
        registered=True,
        cache_status="registered",
        backend=str(payload["builder_backend"]),
    )


def resolve_or_build_greit_artifact(
    config: Mapping[str, Any],
    *,
    registry_dir: str | Path | None = None,
    auto_build: bool = True,
    fwd_model: Any | None = None,
    builder: GREITArtifactBuilder | None = None,
    prepare_online: bool = False,
    device: str = "auto",
    dtype: str | np.dtype[Any] = "float64",
) -> GREITRegistryLookup:
    """Resolve exact artifact or build/register it with the requested backend."""

    use_cached = _bool_value(config.get("greit_use_cached_rm", True))
    force_rebuild = _bool_value(config.get("greit_rebuild_rm", False))
    resolved = None
    if use_cached and not force_rebuild:
        resolved = resolve_greit_artifact(
            config,
            registry_dir=registry_dir,
            prepare_online=prepare_online,
            device=device,
            dtype=dtype,
        )
    if resolved is not None:
        return resolved
    if not auto_build:
        signature = greit_artifact_signature(config)
        raise FileNotFoundError(
            "GREIT artifact registry miss and auto_build=False "
            f"for signature {signature}."
        )

    payload = greit_artifact_signature_payload(config)
    signature = _digest_json(payload)
    artifact_path = greit_artifact_path_for_signature(
        signature,
        registry_dir=registry_dir,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    backend = str(payload["builder_backend"])
    if builder is not None:
        greit = builder(config, payload, artifact_path)
    elif backend == "native":
        if fwd_model is None:
            raise ValueError("native GREIT artifact build requires fwd_model.")
        greit = build_native_greit_artifact(
            config,
            fwd_model=fwd_model,
            artifact_path=artifact_path,
            signature=signature,
            signature_payload=payload,
        )
    else:
        raise NotImplementedError(
            f"GREIT builder backend {backend!r} is not available in-process."
        )
    saved_path = artifact_path if artifact_path.exists() else greit.save(artifact_path)
    registered = register_greit_artifact(
        config,
        saved_path,
        registry_dir=registry_dir,
        provenance={"built_by": "resolve_or_build_greit_artifact"},
    )
    if prepare_online:
        prepared = registered.greit.prepare_online(
            device=device,
            dtype=dtype,
            cache_key=str(registered.artifact_path),
        )
    else:
        prepared = registered.greit
    return GREITRegistryLookup(
        signature=signature,
        signature_payload=MappingProxyType(payload),
        artifact_path=registered.artifact_path,
        manifest_path=registered.manifest_path,
        greit=prepared,
        built=True,
        registered=True,
        cache_status="built",
        backend=backend,
    )


def build_native_greit_artifact(
    config: Mapping[str, Any],
    *,
    fwd_model: Any,
    artifact_path: str | Path,
    signature: str | None = None,
    signature_payload: Mapping[str, Any] | None = None,
) -> GREITRM:
    """Build a native PyEIDORS finite-target GREIT artifact."""

    cfg = dict(config)
    payload = dict(signature_payload or greit_artifact_signature_payload(cfg))
    sig = str(signature or _digest_json(payload))
    imgsz = _shape_tuple(cfg.get("imgsz", cfg.get("greit_imgsz"))) or _default_imgsz(
        cfg
    )
    radius = _float_or_none(cfg.get("radius")) or 1.0
    height = _float_or_none(cfg.get("height")) or 2.0 * radius
    bounds = np.asarray(
        [
            [-radius, -radius, -0.5 * height],
            [radius, radius, 0.5 * height],
        ],
        dtype=np.float64,
    )
    point_in_volume = cfg.get("point_in_volume")
    if point_in_volume is None:
        point_in_volume = _cylindrical_point_in_volume_from_config(
            cfg,
            radius=radius,
            height=height,
        )
    distribution = build_greit3d_distribution(
        fwd_model,
        imgsz=imgsz,
        xvec=cfg.get("xvec", cfg.get("greit_xvec")),
        yvec=cfg.get("yvec", cfg.get("greit_yvec")),
        zvec=cfg.get("zvec", cfg.get("greit_zvec")),
        downsample=cfg.get("downsample", cfg.get("greit_downsample")),
        bounds=bounds,
        point_in_volume=point_in_volume,
    )
    normalize = bool(payload["normalize_measurements"])
    target_radius = _greit_target_radius_from_config(cfg, radius=radius)
    weight = cfg.get("weight", cfg.get("greit_weight"))
    if weight is None:
        weight = cfg.get("noise_figure", cfg.get("greit_noise_figure", 0.5))
    metadata = {
        "greit_registry_schema": GREIT_ARTIFACT_REGISTRY_SCHEMA,
        "greit_registry_signature": sig,
        "greit_registry_signature_payload": payload,
        "builder_backend": "native",
        "builder_semantic_version": GREIT_NATIVE_BUILDER_VERSION,
        "fixture_only": False,
        "eidors_parity": True,
        "rec_model_source": "GREIT3D_distribution",
        "voxel_shape": tuple(int(v) for v in distribution.volume_mask.shape),
        "rec_mask": cfg.get("rec_mask", "cylindrical_fem_volume_v1"),
        "point_in_volume_signature": cfg.get(
            "point_in_volume_signature", "analytic_cylinder_radius_height_v1"
        ),
        "target_size_semantics": "fraction_of_tank_radius",
        "target_radius_effective": float(target_radius),
    }
    pipeline = build_native_greit_training_pipeline(
        fwd_model,
        distribution=distribution,
        rec_model=distribution,
        target_radius=target_radius,
        target_contrast=cfg.get(
            "target_contrast", cfg.get("greit_target_contrast", 1.0)
        ),
        background_conductivity=cfg.get(
            "background_conductivity",
            cfg.get("sigma0", 1.0),
        ),
        normalize=normalize,
        measurement_order=cfg.get("measurement_order", cfg.get("channel_order")),
        channel_mask=cfg.get("channel_mask", cfg.get("bad_channel_mask")),
        measurement_weights=cfg.get("measurement_weights"),
        batch_size=_int_or_none(cfg.get("greit_response_batch_size")) or 8,
        desired_radius=cfg.get("desired_radius", target_radius),
        desired_solution_fn=cfg.get("desired_solution_fn"),
        desired_options=cfg.get("desired_solution_params")
        or cfg.get("greit_desired_options"),
        weight=0.5 if weight is None else float(weight),
        noise_covar=cfg.get("noise_covar", 1.0),
        artifact_path=None,
        keep_model_components=True,
        fwd_model_signature=str(payload["mesh_hash"]),
        metadata=metadata,
    )
    greit = pipeline.greit
    greit = replace(
        greit,
        voxel_shape=tuple(int(v) for v in distribution.volume_mask.shape),
    )
    saved = greit.save(artifact_path)
    greit_meta = dict(greit.metadata)
    greit_meta["artifact_path"] = str(saved)
    return replace(greit, metadata=MappingProxyType(greit_meta))


def _greit_target_radius_from_config(
    cfg: Mapping[str, Any],
    *,
    radius: float,
) -> float:
    explicit = _float_or_none(cfg.get("target_radius", cfg.get("greit_target_radius")))
    if explicit is not None:
        return float(explicit)
    size = _float_or_none(cfg.get("target_size", cfg.get("greit_target_size")))
    if size is None:
        size = 0.20
    semantics = (
        str(cfg.get("target_size_semantics") or "fraction_of_tank_radius")
        .strip()
        .lower()
    )
    if semantics in {"fraction_of_tank_radius", "radius_fraction", "fraction"}:
        return float(size) * float(radius)
    if semantics in {"absolute", "meters", "metres", "m"}:
        return float(size)
    raise ValueError(
        "target_size_semantics must be 'fraction_of_tank_radius' or 'absolute'."
    )


def _cylindrical_point_in_volume_from_config(
    cfg: Mapping[str, Any],
    *,
    radius: float,
    height: float,
) -> Callable[[np.ndarray], np.ndarray] | None:
    mesh_dim = _int_or_none(cfg.get("mesh_dimension"))
    if mesh_dim != 3:
        return None
    z_center = _float_or_none(cfg.get("z_center")) or 0.0
    eps = max(float(radius), float(height), 1.0) * 1.0e-9

    def _inside(points: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("point_in_volume points must have shape (N, >=3).")
        radial = np.hypot(arr[:, 0], arr[:, 1])
        z_rel = arr[:, 2] - float(z_center)
        return (radial <= float(radius) + eps) & (
            np.abs(z_rel) <= 0.5 * float(height) + eps
        )

    return _inside


def _validate_loaded_signature(
    greit: GREITRM,
    *,
    signature: str,
    allow_missing: bool = False,
) -> None:
    meta = dict(greit.metadata)
    actual = str(meta.get("greit_registry_signature") or "").strip()
    if not actual and allow_missing:
        return
    if actual != str(signature):
        raise ValueError(
            "GREIT artifact registry signature mismatch: "
            f"expected {signature}, got {actual or '<missing>'}."
        )


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_value(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return _canonical_value(dict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": [int(v) for v in arr.shape],
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    return repr(value)


def _int_first(cfg: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = cfg.get(key)
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"raw", "false", "0", "no", "off", "none"}:
        return False
    return True


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        text = value.strip().lower().replace("x", ",")
        try:
            return tuple(int(part.strip()) for part in text.split(",") if part.strip())
        except ValueError:
            return ()
    try:
        array = np.asarray(value, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return ()
    return tuple(int(v) for v in array if int(v) > 0)


def _default_imgsz(config: Mapping[str, Any]) -> tuple[int, int, int]:
    n_elec = max(_int_or_none(config.get("n_elec")) or 16, 1)
    n_rings = max(_int_or_none(config.get("n_rings")) or 1, 1)
    total = n_elec * n_rings
    if total >= 48:
        return (8, 8, 5)
    if total >= 32:
        return (7, 7, 4)
    return (6, 6, 3)


__all__ = [
    "GREIT_ARTIFACT_REGISTRY_SCHEMA",
    "GREIT_ARTIFACT_SIGNATURE_SCHEMA",
    "GREIT_NATIVE_BUILDER_VERSION",
    "GREIT_REGISTRY_ENV",
    "GREITRegistryLookup",
    "build_native_greit_artifact",
    "greit_artifact_path_for_signature",
    "greit_artifact_signature",
    "greit_artifact_signature_payload",
    "greit_registry_dir",
    "greit_registry_manifest_path",
    "load_greit_registry_manifest",
    "register_greit_artifact",
    "resolve_greit_artifact",
    "resolve_or_build_greit_artifact",
    "write_greit_registry_manifest",
]
