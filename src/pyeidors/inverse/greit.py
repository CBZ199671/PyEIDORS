"""3D GREIT reconstruction-matrix helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.data.channels import apply_measurement_contract_to_jacobian
from pyeidors.inverse.dual_mesh import CellMesh, VoxelGrid
from pyeidors.inverse.reconstruction_matrix import reconstruct_difference_batch
from pyeidors.perf.gpu_kernels import RMMatmulHandle, RMMatmulResult, prepare_rm_matmul

GREIT_METRIC_KEYS = ("AR", "PE", "RES", "SD", "RNG")


@dataclass(frozen=True)
class GREITTrainingTargets:
    """Synthetic 3D GREIT training targets on the inverse grid."""

    values: np.ndarray
    masks: np.ndarray
    centers: np.ndarray
    radii: np.ndarray
    metadata: MappingProxyType

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


@dataclass(frozen=True)
class GREIT3DDistribution:
    """EIDORS ``GREIT3D_distribution`` target-center volume."""

    centers: np.ndarray
    distr: np.ndarray
    candidate_centers: np.ndarray
    inside_mask: np.ndarray
    volume_mask: np.ndarray
    x_pts: np.ndarray
    y_pts: np.ndarray
    z_pts: np.ndarray
    xvec: np.ndarray
    yvec: np.ndarray
    zvec: np.ndarray
    metadata: MappingProxyType

    def cell_centers(self) -> np.ndarray:
        return np.ascontiguousarray(self.centers, dtype=np.float64)

    def num_cells(self) -> int:
        return int(self.centers.shape[0])


@dataclass(frozen=True)
class GREITRM:
    """Precomputed GREIT RM plus artifact metadata.

    Online reconstruction deliberately delegates to the RM hot path:
    normalized difference data are optionally prepared, then a single
    ``RM @ dv`` matmul is applied.
    """

    rm: np.ndarray
    metadata: MappingProxyType
    voxel_shape: tuple[int, ...] | None = None
    channel_mask: np.ndarray | None = None
    measurement_weights: np.ndarray | None = None
    training_targets: np.ndarray | None = None
    training_responses: np.ndarray | None = None
    rm_handle: RMMatmulHandle | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.rm.shape

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.rm, dtype=dtype)

    def prepare_online(
        self,
        *,
        device: str = "auto",
        dtype: str | np.dtype[Any] = "float64",
        cache_key: str | None = None,
    ) -> "GREITRM":
        """Return a GREIT RM with its online matmul matrix preloaded."""

        handle = prepare_rm_matmul(
            self.rm,
            device=device,
            dtype=dtype,
            cache_key=cache_key,
        )
        return replace(self, rm_handle=handle)

    def reconstruct(
        self,
        dv,
        *,
        normalize: bool = True,
        v_ref=None,
        floor: float | None = None,
        channel_mask: Any | None = None,
        measurement_weights: Any | None = None,
        device: str = "auto",
        dtype: str | np.dtype[Any] = "float64",
        return_metadata: bool = False,
    ) -> np.ndarray | RMMatmulResult:
        """Apply this GREIT RM to one frame or a frame batch."""

        resolved_mask = self.channel_mask if channel_mask is None else channel_mask
        resolved_weights = (
            self.measurement_weights
            if measurement_weights is None
            else measurement_weights
        )
        result = reconstruct_difference_batch(
            self.rm_handle if self.rm_handle is not None else self.rm,
            dv,
            normalize=normalize,
            v_ref=v_ref,
            floor=floor,
            channel_mask=resolved_mask,
            measurement_weights=resolved_weights,
            device=device,
            dtype=dtype,
            return_metadata=True,
        )
        values = _reshape_reconstruction(np.asarray(result.values), self.voxel_shape)
        if return_metadata:
            meta = dict(result.metadata)
            meta.update(
                {
                    "algorithm": "greit-3d",
                    "online_hot_path": "rm_matmul",
                    "voxel_shape": self.voxel_shape,
                }
            )
            return RMMatmulResult(values=values, metadata=MappingProxyType(meta))
        return values

    def save(self, path: str | Path) -> Path:
        """Persist the RM and offline training artifact to HDF5."""

        from pyeidors.io.hdf5_artifacts import write_hdf5_artifact

        target = _greit_hdf5_path(path)
        metadata = dict(self.metadata)
        metadata.setdefault("legacy_artifact_schema", metadata.get("artifact_schema"))
        metadata["artifact_schema"] = "pyeidors-greit-rm-hdf5-v1"
        metadata["artifact_format"] = "hdf5"
        write_hdf5_artifact(
            target,
            {
                "rm": np.asarray(self.rm, dtype=np.float64),
                "voxel_shape": np.asarray(self.voxel_shape or (), dtype=np.int64),
                "channel_mask": _optional_array(self.channel_mask, dtype=bool),
                "measurement_weights": _optional_array(self.measurement_weights),
                "training_targets": _optional_array(self.training_targets),
                "training_responses": _optional_array(self.training_responses),
            },
            metadata,
            schema="pyeidors-greit-rm-hdf5-v1",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "GREITRM":
        """Load a GREIT RM artifact written by :meth:`save`."""

        source = Path(path)
        suffix = source.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            return cls._load_hdf5(source)
        if suffix == ".npz":
            return cls._load_legacy_npz(source)
        raise ValueError(f"Unsupported GREIT RM suffix {suffix!r}; expected .h5.")

    @classmethod
    def _load_hdf5(cls, path: Path) -> "GREITRM":
        from pyeidors.io.hdf5_artifacts import read_hdf5_artifact

        artifact = read_hdf5_artifact(path)
        arrays = dict(artifact.arrays)
        if "rm" not in arrays:
            raise ValueError(f"GREIT artifact is missing 'rm': {path}")
        voxel_raw = np.asarray(arrays.get("voxel_shape", ()), dtype=np.int64)
        return cls(
            rm=np.asarray(arrays["rm"], dtype=np.float64),
            metadata=MappingProxyType(dict(artifact.metadata)),
            voxel_shape=tuple(int(v) for v in voxel_raw) if voxel_raw.size else None,
            channel_mask=_empty_to_none_array(arrays.get("channel_mask"), dtype=bool),
            measurement_weights=_empty_to_none_array(arrays.get("measurement_weights")),
            training_targets=_empty_to_none_array(arrays.get("training_targets")),
            training_responses=_empty_to_none_array(arrays.get("training_responses")),
        )

    @classmethod
    def _load_legacy_npz(cls, path: Path) -> "GREITRM":
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            metadata.setdefault("artifact_format", "legacy-npz")
            metadata.setdefault("legacy_read_only", True)
            voxel_raw = np.asarray(payload["voxel_shape"], dtype=np.int64)
            return cls(
                rm=np.asarray(payload["rm"], dtype=np.float64),
                metadata=MappingProxyType(metadata),
                voxel_shape=(
                    tuple(int(v) for v in voxel_raw) if voxel_raw.size else None
                ),
                channel_mask=_empty_to_none(payload["channel_mask"], dtype=bool),
                measurement_weights=_empty_to_none(payload["measurement_weights"]),
                training_targets=_empty_to_none(payload["training_targets"]),
                training_responses=_empty_to_none(payload["training_responses"]),
            )


def generate_spherical_targets(
    inverse_mesh: Any,
    *,
    centers: Any | None = None,
    radius: float | None = None,
    amplitude: float = 1.0,
    kind: str = "sphere",
) -> GREITTrainingTargets:
    """Generate sphere or Gaussian-blob GREIT training targets."""

    cell_centers = _cell_centers(inverse_mesh)
    if cell_centers.shape[1] != 3:
        raise ValueError("3D GREIT targets require 3D inverse cell centers.")
    target_centers = cell_centers if centers is None else _as_centers(centers)
    if target_centers.shape[1] != 3:
        raise ValueError("GREIT target centers must be 3D.")
    resolved_kind = str(kind).strip().lower()
    if resolved_kind not in {"sphere", "blob"}:
        raise ValueError("kind must be one of: 'sphere', 'blob'.")
    resolved_radius = _default_radius(cell_centers) if radius is None else float(radius)
    if resolved_radius <= 0.0 or not np.isfinite(resolved_radius):
        raise ValueError("radius must be finite and positive.")
    amp = float(amplitude)
    if not np.isfinite(amp):
        raise ValueError("amplitude must be finite.")

    values = []
    masks = []
    for center in target_centers:
        distance = np.linalg.norm(cell_centers - center.reshape(1, -1), axis=1)
        mask = distance <= resolved_radius
        if not np.any(mask):
            mask[int(np.argmin(distance))] = True
        if resolved_kind == "sphere":
            target = mask.astype(np.float64) * amp
        else:
            target = amp * np.exp(-0.5 * (distance / resolved_radius) ** 2)
            target[~mask] = 0.0
        values.append(target)
        masks.append(mask)

    voxel_shape = tuple(int(v) for v in getattr(inverse_mesh, "shape", ()))
    metadata = MappingProxyType(
        {
            "kind": resolved_kind,
            "radius": resolved_radius,
            "amplitude": amp,
            "n_targets": int(len(values)),
            "n_parameters": int(cell_centers.shape[0]),
            "voxel_shape": voxel_shape or None,
        }
    )
    return GREITTrainingTargets(
        values=np.asarray(values, dtype=np.float64),
        masks=np.asarray(masks, dtype=bool),
        centers=np.asarray(target_centers, dtype=np.float64),
        radii=np.full(len(values), resolved_radius, dtype=np.float64),
        metadata=metadata,
    )


def build_greit3d_distribution(
    fwd_model: Any | None = None,
    *,
    imgsz: Any | None = None,
    xvec: Any | None = None,
    yvec: Any | None = None,
    zvec: Any | None = None,
    bounds: Any | None = None,
    downsample: Any | None = None,
    point_in_volume: Any | None = None,
) -> GREIT3DDistribution:
    """Build EIDORS-style 3D GREIT target centers.

    EIDORS creates a voxel rec model, extracts ``x_pts/y_pts/z_pts``,
    optionally downsamples them, then flattens ``ndgrid`` output with the
    x-axis changing fastest.  This helper mirrors that ordering and returns
    both the full candidate volume mask and the valid target-center list.
    """

    bounds_arr = _resolve_distribution_bounds(fwd_model, bounds)
    x_edges, y_edges, z_edges, axis_source = _resolve_distribution_edges(
        imgsz=imgsz,
        xvec=xvec,
        yvec=yvec,
        zvec=zvec,
        bounds=bounds_arr,
    )
    raw_x_pts = _edge_centers(x_edges, name="xvec")
    raw_y_pts = _edge_centers(y_edges, name="yvec")
    raw_z_pts = _edge_centers(z_edges, name="zvec")
    factors, phases = _parse_downsample(downsample)
    x_pts = _downsample_axis(raw_x_pts, factor=factors[0], phase=phases[0], name="x")
    y_pts = _downsample_axis(raw_y_pts, factor=factors[1], phase=phases[1], name="y")
    z_pts = _downsample_axis(raw_z_pts, factor=factors[2], phase=phases[2], name="z")

    grids = np.meshgrid(x_pts, y_pts, z_pts, indexing="ij")
    candidate_shape = tuple(int(grid.shape[axis]) for axis, grid in enumerate(grids))
    candidate_centers = np.stack(
        [grid.ravel(order="F") for grid in grids],
        axis=1,
    )
    inside_mask = _distribution_inside_mask(
        candidate_centers,
        candidate_shape=candidate_shape,
        fwd_model=fwd_model,
        point_in_volume=point_in_volume,
    )
    if not bool(np.any(inside_mask)):
        raise ValueError(
            "GREIT3D_distribution produced no target centers inside volume."
        )

    centers = np.ascontiguousarray(candidate_centers[inside_mask], dtype=np.float64)
    distr = np.ascontiguousarray(centers.T, dtype=np.float64)
    volume_mask = np.ascontiguousarray(
        inside_mask.reshape(candidate_shape, order="F"),
        dtype=bool,
    )
    metadata = MappingProxyType(
        {
            "builder": "GREIT3D_distribution",
            "eidors_component_parity": True,
            "parameter_order": "eidors_ndgrid_x_fastest",
            "axis_source": axis_source,
            "downsample_factors": tuple(int(v) for v in factors),
            "downsample_phases": tuple(int(v) for v in phases),
            "candidate_shape": candidate_shape,
            "n_candidate_voxels": int(candidate_centers.shape[0]),
            "n_targets": int(centers.shape[0]),
            "xvec": tuple(float(v) for v in x_edges),
            "yvec": tuple(float(v) for v in y_edges),
            "zvec": tuple(float(v) for v in z_edges),
            "x_pts": tuple(float(v) for v in x_pts),
            "y_pts": tuple(float(v) for v in y_pts),
            "z_pts": tuple(float(v) for v in z_pts),
        }
    )
    return GREIT3DDistribution(
        centers=centers,
        distr=distr,
        candidate_centers=np.ascontiguousarray(candidate_centers, dtype=np.float64),
        inside_mask=np.ascontiguousarray(inside_mask, dtype=bool),
        volume_mask=volume_mask,
        x_pts=np.ascontiguousarray(x_pts, dtype=np.float64),
        y_pts=np.ascontiguousarray(y_pts, dtype=np.float64),
        z_pts=np.ascontiguousarray(z_pts, dtype=np.float64),
        xvec=np.ascontiguousarray(x_edges, dtype=np.float64),
        yvec=np.ascontiguousarray(y_edges, dtype=np.float64),
        zvec=np.ascontiguousarray(z_edges, dtype=np.float64),
        metadata=metadata,
    )


def build_3d_greit_rm(
    fwd_model: Any = None,
    targets: Any | None = None,
    noise_figure: float = 0.5,
    regularisation: Any | None = None,
    *,
    jacobian: Any | None = None,
    inverse_mesh: Any | None = None,
    target_centers: Any | None = None,
    target_radius: float | None = None,
    target_amplitude: float = 1.0,
    target_kind: str = "sphere",
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    artifact_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> GREITRM:
    """Build an offline 3D GREIT RM from synthetic targets.

    The first v1 implementation accepts a linearized forward response
    ``J`` with shape ``(n_measurements, n_inverse_parameters)``. Synthetic
    targets ``T`` are projected to measurement responses ``Y = T @ J.T`` and
    the GREIT RM is built as ``T.T @ Y @ (Y.T @ Y + nf^2 Rn)^-1`` in
    measurement space.
    """

    raw_j = _resolve_jacobian(fwd_model, jacobian)
    weighted_j, measurement_contract = apply_measurement_contract_to_jacobian(
        raw_j,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    target_bundle = _resolve_targets(
        targets,
        inverse_mesh=inverse_mesh,
        centers=target_centers,
        radius=target_radius,
        amplitude=target_amplitude,
        kind=target_kind,
    )
    target_values = np.asarray(target_bundle.values, dtype=np.float64)
    if target_values.shape[1] != weighted_j.shape[1]:
        raise ValueError(
            "targets parameter dimension "
            f"{target_values.shape[1]} does not match J columns {weighted_j.shape[1]}."
        )
    nf = float(noise_figure)
    if nf < 0.0 or not np.isfinite(nf):
        raise ValueError("noise_figure must be finite and non-negative.")

    responses = np.asarray(target_values @ weighted_j.T, dtype=np.float64)
    response_cols = responses.T
    target_cols = target_values.T
    rn, rn_source = _measurement_regularisation(
        regularisation,
        n_measurements=weighted_j.shape[0],
    )
    lhs = response_cols @ response_cols.T + (nf * nf) * rn
    rhs_t = (target_cols @ response_cols.T).T
    try:
        rm = np.linalg.solve(lhs.T, rhs_t).T
        solver = "solve"
    except np.linalg.LinAlgError:
        rm = target_cols @ response_cols.T @ np.linalg.pinv(lhs)
        solver = "pinv"
    rm = np.asarray(rm, dtype=np.float64)
    if not np.isfinite(rm).all():
        raise FloatingPointError("GREIT RM contains non-finite values.")

    voxel_shape = _voxel_shape(inverse_mesh) or _metadata_voxel_shape(
        target_bundle.metadata
    )
    meta = {
        "algorithm": "greit-3d",
        "target_kind": target_bundle.metadata["kind"],
        "synthetic_target_count": int(target_values.shape[0]),
        "n_measurements": int(weighted_j.shape[0]),
        "n_parameters": int(weighted_j.shape[1]),
        "noise_figure": nf,
        "regularisation_source": rn_source,
        "bad_channel_count": int(measurement_contract.bad_channel_count),
        "measurement_weight_kind": measurement_contract.weight_kind,
        "system_shape": tuple(int(v) for v in lhs.shape),
        "rm_shape": tuple(int(v) for v in rm.shape),
        "solver": solver,
        "online_hot_path": "rm_matmul",
        "artifact_schema": "pyeidors-greit-rm-hdf5-v1",
        "artifact_format": "hdf5",
        "eidors_parity": False,
        "training_mode": "linearized",
        "voxel_shape": voxel_shape,
    }
    if metadata:
        meta.update(metadata)

    result = GREITRM(
        rm=rm,
        metadata=MappingProxyType(meta),
        voxel_shape=voxel_shape,
        channel_mask=measurement_contract.channel_mask,
        measurement_weights=_stored_measurement_weights(measurement_weights),
        training_targets=target_values,
        training_responses=responses,
    )
    if artifact_path is not None:
        saved = result.save(artifact_path)
        meta = dict(result.metadata)
        meta["artifact_path"] = str(saved)
        result = GREITRM(
            rm=result.rm,
            metadata=MappingProxyType(meta),
            voxel_shape=result.voxel_shape,
            channel_mask=result.channel_mask,
            measurement_weights=result.measurement_weights,
            training_targets=result.training_targets,
            training_responses=result.training_responses,
        )
    return result


def load_greit_rm(path: str | Path) -> GREITRM:
    """Load a persisted GREIT RM artifact."""

    return GREITRM.load(path)


def migrate_greit_rm_to_hdf5(src: str | Path, dst: str | Path | None = None) -> Path:
    """Migrate a legacy GREIT NPZ artifact into HDF5 without deleting source."""

    source = Path(src)
    target = _greit_hdf5_path(dst if dst is not None else source.with_suffix(".h5"))
    greit = GREITRM.load(source)
    meta = dict(greit.metadata)
    meta.update(
        {
            "migrated_from": str(source),
            "legacy_format": source.suffix.lower().lstrip("."),
        }
    )
    greit = replace(greit, metadata=MappingProxyType(meta))
    return greit.save(target)


def greit_metrics(
    voxel_image: Any,
    target_mask: Any,
    *,
    centers: Any | None = None,
    target_values: Any | None = None,
    cell_volumes: Any | None = None,
    threshold_fraction: float = 0.25,
) -> dict[str, float]:
    """Compute EIDORS-style GREIT figures of merit.

    The original GREIT evaluation reports amplitude response, position error,
    resolution, shape deformation, and ringing. This helper applies the same
    quarter-max idea to a discrete 3D voxel/cell image.
    """

    image, original_shape = _as_flat_image(voxel_image, name="voxel_image")
    mask = np.asarray(target_mask, dtype=bool).reshape(-1)
    if mask.size != image.size:
        raise ValueError(
            f"target_mask size {mask.size} does not match image size {image.size}."
        )
    if not np.any(mask):
        raise ValueError("target_mask must mark at least one target cell.")
    weights = _as_cell_volumes(cell_volumes, n_cells=image.size)
    coords = _metric_centers(centers, original_shape, n_cells=image.size)
    target = _as_target_values(target_values, mask=mask)

    target_integral = float(np.sum(target * weights))
    if abs(target_integral) <= np.finfo(np.float64).eps:
        raise ValueError("target amplitude integral must be non-zero.")
    ar = float(np.sum(image * weights) / target_integral)

    signal_sign = 1.0 if target_integral >= 0.0 else -1.0
    signed_image = signal_sign * image
    positive = np.maximum(signed_image, 0.0)
    centroid_weights = positive * weights
    if float(np.sum(centroid_weights)) <= np.finfo(np.float64).eps:
        recon_center = coords[int(np.argmax(np.abs(image)))]
    else:
        recon_center = _weighted_centroid(coords, centroid_weights)
    target_center = _weighted_centroid(coords, np.abs(target) * weights)
    pe = float(np.linalg.norm(recon_center - target_center))

    threshold = _quarter_max_threshold(signed_image, threshold_fraction)
    qmi = signed_image >= threshold
    if not np.any(qmi):
        qmi[int(np.argmax(signed_image))] = True
    qmi_volume = float(np.sum(weights[qmi]))
    domain_volume = float(np.sum(weights))
    dimension = _metric_dimension(coords)
    res = float((qmi_volume / domain_volume) ** (1.0 / dimension))

    equivalent_ball = _equivalent_ball_mask(
        coords,
        weights,
        center=recon_center,
        target_volume=qmi_volume,
    )
    outside_ball_volume = float(np.sum(weights[qmi & ~equivalent_ball]))
    sd = float(outside_ball_volume / qmi_volume) if qmi_volume > 0.0 else 0.0

    qmi_signal = float(np.sum(signed_image[qmi] * weights[qmi]))
    opposite = (signed_image < 0.0) & ~qmi
    if qmi_signal <= np.finfo(np.float64).eps:
        rng = 0.0
    else:
        rng = float(-np.sum(signed_image[opposite] * weights[opposite]) / qmi_signal)

    metrics = {
        "AR": ar,
        "PE": pe,
        "RES": res,
        "SD": sd,
        "RNG": max(rng, 0.0),
    }
    _ensure_greit_metric_keys(metrics)
    return metrics


def write_greit_metrics_artifact(
    metrics: Any,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write GREIT metrics to JSON or CSV.

    The writer fails fast if any record lacks one of
    ``{AR, PE, RES, SD, RNG}``, which is the validation gate in V41.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = _as_metric_records(metrics)
    suffix = target.suffix.lower()
    if suffix == ".json":
        payload = {
            "schema": "pyeidors-greit-metrics-v1",
            "metric_keys": list(GREIT_METRIC_KEYS),
            "metadata": _json_ready(metadata or {}),
            "records": [_json_ready(record) for record in records],
        }
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    elif suffix == ".csv":
        extra_keys = sorted(
            {
                key
                for record in records
                for key in record
                if key not in GREIT_METRIC_KEYS
            }
        )
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(GREIT_METRIC_KEYS) + extra_keys
            )
            writer.writeheader()
            writer.writerows(records)
    else:
        raise ValueError("GREIT metrics artifact path must end with .json or .csv.")
    return target


def _resolve_jacobian(fwd_model: Any, jacobian: Any | None) -> np.ndarray:
    source = jacobian if jacobian is not None else fwd_model
    if source is None:
        raise ValueError("build_3d_greit_rm requires jacobian or fwd_model.")
    for attr_name in ("jacobian", "J"):
        attr = getattr(source, attr_name, None)
        if attr is not None:
            source = attr() if callable(attr) else attr
            break
    if sparse.issparse(source):
        matrix = np.asarray(source.toarray(), dtype=np.float64)
    else:
        matrix = np.asarray(source, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("J must be a 2D measurement-by-parameter matrix.")
    if 0 in matrix.shape:
        raise ValueError("J must be non-empty.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("J contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _resolve_targets(
    targets: Any | None,
    *,
    inverse_mesh: Any | None,
    centers: Any | None,
    radius: float | None,
    amplitude: float,
    kind: str,
) -> GREITTrainingTargets:
    if isinstance(targets, GREITTrainingTargets):
        return targets
    if targets is not None:
        values = np.asarray(targets, dtype=np.float64)
        if values.ndim != 2 or 0 in values.shape:
            raise ValueError("targets must be a non-empty 2D target matrix.")
        if not np.isfinite(values).all():
            raise FloatingPointError("targets contain non-finite values.")
        masks = np.asarray(values != 0.0, dtype=bool)
        metadata = MappingProxyType(
            {
                "kind": "provided",
                "radius": None,
                "amplitude": None,
                "n_targets": int(values.shape[0]),
                "n_parameters": int(values.shape[1]),
                "voxel_shape": None,
            }
        )
        return GREITTrainingTargets(
            values=np.ascontiguousarray(values, dtype=np.float64),
            masks=masks,
            centers=np.empty((values.shape[0], 0), dtype=np.float64),
            radii=np.zeros(values.shape[0], dtype=np.float64),
            metadata=metadata,
        )
    if inverse_mesh is None:
        raise ValueError("inverse_mesh is required when targets are not provided.")
    return generate_spherical_targets(
        inverse_mesh,
        centers=centers,
        radius=radius,
        amplitude=amplitude,
        kind=kind,
    )


def _resolve_distribution_bounds(
    fwd_model: Any | None,
    bounds: Any | None,
) -> np.ndarray | None:
    if bounds is not None:
        arr = np.asarray(bounds, dtype=np.float64)
        if arr.shape != (2, 3):
            raise ValueError("bounds must have shape (2, 3).")
        if not np.isfinite(arr).all():
            raise FloatingPointError("bounds contain non-finite values.")
        if np.any(arr[1] <= arr[0]):
            raise ValueError("bounds upper row must exceed lower row.")
        return np.ascontiguousarray(arr, dtype=np.float64)

    nodes = _model_nodes(fwd_model)
    if nodes is None:
        return None
    lower = np.min(nodes[:, :3], axis=0)
    upper = np.max(nodes[:, :3], axis=0)
    if np.any(upper <= lower):
        return None
    return np.ascontiguousarray(np.vstack([lower, upper]), dtype=np.float64)


def _resolve_distribution_edges(
    *,
    imgsz: Any | None,
    xvec: Any | None,
    yvec: Any | None,
    zvec: Any | None,
    bounds: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    img_counts = _parse_imgsz(imgsz)
    x_edges, x_source = _resolve_axis_edges(
        xvec,
        axis=0,
        name="xvec",
        imgsz_count=img_counts[0],
        bounds=bounds,
    )
    y_edges, y_source = _resolve_axis_edges(
        yvec,
        axis=1,
        name="yvec",
        imgsz_count=img_counts[1],
        bounds=bounds,
    )
    z_edges, z_source = _resolve_axis_edges(
        zvec,
        axis=2,
        name="zvec",
        imgsz_count=img_counts[2],
        bounds=bounds,
    )
    return x_edges, y_edges, z_edges, f"{x_source},{y_source},{z_source}"


def _parse_imgsz(imgsz: Any | None) -> tuple[int | None, int | None, int | None]:
    if imgsz is None:
        return None, None, None
    arr = np.asarray(imgsz, dtype=np.int64).reshape(-1)
    if arr.size not in {2, 3}:
        raise ValueError("imgsz must contain 2 or 3 entries.")
    if np.any(arr <= 0):
        raise ValueError("imgsz entries must be positive.")
    if arr.size == 2:
        return int(arr[0]), int(arr[1]), None
    return int(arr[0]), int(arr[1]), int(arr[2])


def _resolve_axis_edges(
    value: Any | None,
    *,
    axis: int,
    name: str,
    imgsz_count: int | None,
    bounds: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if value is None:
        if imgsz_count is None:
            raise ValueError(f"{name} or imgsz entry is required.")
        if bounds is None:
            raise ValueError(f"bounds or fwd_model nodes are required for {name}.")
        return (
            np.linspace(
                float(bounds[0, axis]),
                float(bounds[1, axis]),
                int(imgsz_count) + 1,
                dtype=np.float64,
            ),
            "imgsz",
        )

    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        if bounds is None:
            raise ValueError(
                f"bounds or fwd_model nodes are required for scalar {name}."
            )
        n_planes = int(round(float(arr[0])))
        if n_planes < 2 or not np.isclose(float(arr[0]), float(n_planes)):
            raise ValueError(f"scalar {name} must be an integer >= 2.")
        return (
            np.linspace(
                float(bounds[0, axis]),
                float(bounds[1, axis]),
                n_planes,
                dtype=np.float64,
            ),
            f"{name}:scalar",
        )
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two cut planes.")
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} cut planes must be strictly increasing.")
    return np.ascontiguousarray(arr, dtype=np.float64), f"{name}:explicit"


def _edge_centers(edges: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two cut planes.")
    return np.ascontiguousarray((arr[:-1] + arr[1:]) * 0.5, dtype=np.float64)


def _parse_downsample(downsample: Any | None) -> tuple[np.ndarray, np.ndarray]:
    if downsample is None:
        return np.ones(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    arr = np.asarray(downsample, dtype=np.int64)
    if arr.ndim == 0:
        factors = np.full(3, int(arr), dtype=np.int64)
        phases = np.zeros(3, dtype=np.int64)
    elif arr.shape == (2,):
        factors = np.full(3, int(arr[0]), dtype=np.int64)
        phases = np.full(3, int(arr[1]), dtype=np.int64)
    elif arr.shape == (3,):
        factors = np.asarray(arr, dtype=np.int64)
        phases = np.zeros(3, dtype=np.int64)
    elif arr.shape == (3, 2):
        factors = np.asarray(arr[:, 0], dtype=np.int64)
        phases = np.asarray(arr[:, 1], dtype=np.int64)
    elif arr.size == 6:
        reshaped = arr.reshape(3, 2)
        factors = np.asarray(reshaped[:, 0], dtype=np.int64)
        phases = np.asarray(reshaped[:, 1], dtype=np.int64)
    else:
        raise ValueError("downsample must be scalar, [N, PHASE], length-3, or 3x2.")
    if np.any(factors <= 0):
        raise ValueError("downsample factors must be positive.")
    if np.any(phases < 0) or np.any(phases >= factors):
        raise ValueError("downsample phases must satisfy 0 <= phase < factor.")
    return factors, phases


def _downsample_axis(
    values: np.ndarray,
    *,
    factor: int,
    phase: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    selected = arr[int(phase) :: int(factor)]
    if selected.size == 0:
        raise ValueError(f"downsample removed all {name}-axis points.")
    return np.ascontiguousarray(selected, dtype=np.float64)


def _distribution_inside_mask(
    candidate_centers: np.ndarray,
    *,
    candidate_shape: tuple[int, int, int],
    fwd_model: Any | None,
    point_in_volume: Any | None,
) -> np.ndarray:
    centers = np.asarray(candidate_centers, dtype=np.float64)
    source = point_in_volume
    if source is None:
        source = getattr(fwd_model, "point_in_volume", None)
        if source is None and isinstance(fwd_model, dict):
            source = fwd_model.get("point_in_volume")
    if callable(source):
        mask = np.asarray(source(centers), dtype=bool).reshape(-1)
    elif source is not None:
        mask = np.asarray(source, dtype=bool)
        if mask.shape == candidate_shape:
            mask = mask.reshape(-1, order="F")
        else:
            mask = mask.reshape(-1)
    else:
        mask = _inside_mask_from_model_nodes(fwd_model, centers)
    if mask.size != centers.shape[0]:
        raise ValueError(
            f"point_in_volume mask length {mask.size} does not match "
            f"{centers.shape[0]} candidate centers."
        )
    return np.ascontiguousarray(mask, dtype=bool)


def _model_nodes(fwd_model: Any | None) -> np.ndarray | None:
    if fwd_model is None:
        return None
    raw = None
    if isinstance(fwd_model, dict):
        raw = fwd_model.get("nodes")
    if raw is None:
        for name in ("nodes", "coordinates"):
            attr = getattr(fwd_model, name, None)
            if attr is not None:
                raw = attr() if callable(attr) else attr
                break
    if raw is None:
        mesh = getattr(fwd_model, "mesh", None)
        if mesh is not None:
            raw = getattr(mesh, "coordinates", None)
            if raw is None and hasattr(mesh, "geometry"):
                raw = getattr(mesh.geometry, "x", None)
    if raw is None:
        return None
    nodes = np.asarray(raw, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[0] == 0 or nodes.shape[1] < 3:
        return None
    if not np.isfinite(nodes).all():
        return None
    return np.ascontiguousarray(nodes[:, :3], dtype=np.float64)


def _inside_mask_from_model_nodes(
    fwd_model: Any | None,
    centers: np.ndarray,
) -> np.ndarray:
    nodes = _model_nodes(fwd_model)
    if nodes is None:
        return np.ones(centers.shape[0], dtype=bool)
    try:
        from scipy.spatial import Delaunay

        hull = Delaunay(nodes)
        return np.asarray(hull.find_simplex(centers[:, :3]) >= 0, dtype=bool)
    except Exception:
        lower = np.min(nodes, axis=0)
        upper = np.max(nodes, axis=0)
        eps = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(upper - lower))))
        return np.asarray(
            np.all(
                (centers[:, :3] >= lower - eps) & (centers[:, :3] <= upper + eps),
                axis=1,
            ),
            dtype=bool,
        )


def _cell_centers(mesh: Any) -> np.ndarray:
    if isinstance(mesh, GREIT3DDistribution):
        return mesh.cell_centers()
    if isinstance(mesh, VoxelGrid):
        return mesh.cell_centers()
    if isinstance(mesh, CellMesh):
        return mesh.cell_centers()
    attr = getattr(mesh, "cell_centers", None)
    if callable(attr):
        centers = attr()
    else:
        centers = getattr(mesh, "centers", None)
    if centers is None:
        raise TypeError(f"Cannot extract cell centers from mesh type {type(mesh)!r}.")
    array = np.asarray(centers, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("inverse mesh cell centers must be a non-empty 2D array.")
    if not np.isfinite(array).all():
        raise FloatingPointError("inverse mesh cell centers contain non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_centers(values: Any) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] == 0:
        raise ValueError("centers must be a non-empty 2D array.")
    if not np.isfinite(centers).all():
        raise FloatingPointError("centers contain non-finite values.")
    return np.ascontiguousarray(centers, dtype=np.float64)


def _default_radius(centers: np.ndarray) -> float:
    if centers.shape[0] <= 1:
        return 1.0
    distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :],
        axis=2,
    )
    distances[distances == 0.0] = np.inf
    nearest = float(np.min(distances))
    if not np.isfinite(nearest):
        return 1.0
    return max(0.51 * nearest, np.finfo(np.float64).eps)


def _measurement_regularisation(
    values: Any | None,
    *,
    n_measurements: int,
) -> tuple[np.ndarray, str]:
    if values is None:
        return np.eye(n_measurements, dtype=np.float64), "identity"
    if sparse.issparse(values):
        matrix = np.asarray(values.toarray(), dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
        matrix = np.diag(array) if array.ndim == 1 else array
    if matrix.shape != (n_measurements, n_measurements):
        raise ValueError(
            "regularisation must have shape "
            f"{(n_measurements, n_measurements)}, got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("regularisation contains non-finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("regularisation matrix must be symmetric.")
    return np.ascontiguousarray(matrix, dtype=np.float64), "provided"


def _stored_measurement_weights(values: Any | None) -> np.ndarray | None:
    if values is None:
        return None
    if sparse.issparse(values):
        return np.asarray(values.toarray(), dtype=np.float64)
    return np.asarray(values, dtype=np.float64).copy()


def _optional_array(values: Any | None, *, dtype=np.float64) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _empty_to_none(values: Any, *, dtype=np.float64) -> np.ndarray | None:
    array = np.asarray(values, dtype=dtype)
    if array.size == 0:
        return None
    return array


def _empty_to_none_array(values: Any | None, *, dtype=np.float64) -> np.ndarray | None:
    if values is None:
        return None
    return _empty_to_none(values, dtype=dtype)


def _greit_hdf5_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix == "":
        return target.with_suffix(".h5")
    if target.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError(
            f"GREIT RM artifacts are written as HDF5 .h5 files, got {target}"
        )
    return target


def _json_ready(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return _json_ready(dict(value))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _voxel_shape(mesh: Any | None) -> tuple[int, ...] | None:
    shape = getattr(mesh, "shape", None)
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def _metadata_voxel_shape(metadata: MappingProxyType) -> tuple[int, ...] | None:
    shape = metadata.get("voxel_shape")
    if shape is None:
        return None
    return tuple(int(v) for v in shape)


def _reshape_reconstruction(
    values: np.ndarray,
    voxel_shape: tuple[int, ...] | None,
) -> np.ndarray:
    if voxel_shape is None:
        return values
    expected = int(np.prod(voxel_shape))
    if values.ndim == 1:
        if values.size != expected:
            return values
        return values.reshape(voxel_shape, order="C")
    if values.ndim == 2 and values.shape[1] == expected:
        return values.reshape((values.shape[0],) + voxel_shape, order="C")
    return values


def _as_flat_image(values: Any, *, name: str) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array.reshape(-1), dtype=np.float64), tuple(array.shape)


def _as_cell_volumes(values: Any | None, *, n_cells: int) -> np.ndarray:
    if values is None:
        return np.ones(n_cells, dtype=np.float64)
    volumes = np.asarray(values, dtype=np.float64).reshape(-1)
    if volumes.size != n_cells:
        raise ValueError(
            f"cell_volumes length {volumes.size} does not match {n_cells}."
        )
    if not np.isfinite(volumes).all():
        raise FloatingPointError("cell_volumes contain non-finite values.")
    if np.any(volumes <= 0.0):
        raise ValueError("cell_volumes entries must be positive.")
    return np.ascontiguousarray(volumes, dtype=np.float64)


def _metric_centers(
    centers: Any | None,
    original_shape: tuple[int, ...],
    *,
    n_cells: int,
) -> np.ndarray:
    if centers is not None:
        coords = np.asarray(centers, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[0] != n_cells or coords.shape[1] == 0:
            raise ValueError(
                "centers must have shape (n_cells, dimension); "
                f"got {coords.shape}, expected first dimension {n_cells}."
            )
        if not np.isfinite(coords).all():
            raise FloatingPointError("centers contain non-finite values.")
        return np.ascontiguousarray(coords, dtype=np.float64)
    if len(original_shape) <= 1:
        return np.arange(n_cells, dtype=np.float64).reshape(-1, 1)
    axes = [np.arange(size, dtype=np.float64) for size in original_shape]
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack([grid.ravel(order="C") for grid in grids], axis=1)


def _as_target_values(values: Any | None, *, mask: np.ndarray) -> np.ndarray:
    if values is None:
        return mask.astype(np.float64)
    target = np.asarray(values, dtype=np.float64).reshape(-1)
    if target.size != mask.size:
        raise ValueError(
            f"target_values size {target.size} does not match mask size {mask.size}."
        )
    if not np.isfinite(target).all():
        raise FloatingPointError("target_values contain non-finite values.")
    return np.ascontiguousarray(target, dtype=np.float64)


def _weighted_centroid(coords: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= np.finfo(np.float64).eps:
        raise ValueError("centroid weights must have positive sum.")
    return np.asarray((coords * weights[:, None]).sum(axis=0) / total, dtype=np.float64)


def _quarter_max_threshold(image: np.ndarray, fraction: float) -> float:
    frac = float(fraction)
    if frac <= 0.0 or frac > 1.0 or not np.isfinite(frac):
        raise ValueError("threshold_fraction must be in (0, 1].")
    peak = float(np.max(image))
    if peak <= np.finfo(np.float64).eps:
        return peak
    return frac * peak


def _metric_dimension(coords: np.ndarray) -> int:
    if coords.shape[1] <= 1:
        return 1
    span = np.ptp(coords, axis=0)
    return max(1, int(np.count_nonzero(span > np.finfo(np.float64).eps)))


def _equivalent_ball_mask(
    coords: np.ndarray,
    weights: np.ndarray,
    *,
    center: np.ndarray,
    target_volume: float,
) -> np.ndarray:
    order = np.argsort(np.linalg.norm(coords - center.reshape(1, -1), axis=1))
    selected = np.zeros(coords.shape[0], dtype=bool)
    cumulative = 0.0
    for idx in order:
        selected[int(idx)] = True
        cumulative += float(weights[int(idx)])
        if cumulative + np.finfo(np.float64).eps >= target_volume:
            break
    return selected


def _ensure_greit_metric_keys(record: dict[str, Any]) -> None:
    missing = [key for key in GREIT_METRIC_KEYS if key not in record]
    if missing:
        raise ValueError(f"GREIT metrics record missing keys: {missing}.")
    for key in GREIT_METRIC_KEYS:
        value = float(record[key])
        if not np.isfinite(value):
            raise FloatingPointError(f"GREIT metric {key} is non-finite.")


def _as_metric_records(metrics: Any) -> list[dict[str, Any]]:
    if isinstance(metrics, dict):
        records = [dict(metrics)]
    else:
        records = [dict(record) for record in metrics]
    if not records:
        raise ValueError("metrics must contain at least one record.")
    for record in records:
        _ensure_greit_metric_keys(record)
    return records


__all__ = [
    "GREIT3DDistribution",
    "GREIT_METRIC_KEYS",
    "GREITRM",
    "GREITTrainingTargets",
    "build_3d_greit_rm",
    "build_greit3d_distribution",
    "generate_spherical_targets",
    "greit_metrics",
    "load_greit_rm",
    "migrate_greit_rm_to_hdf5",
    "write_greit_metrics_artifact",
]
