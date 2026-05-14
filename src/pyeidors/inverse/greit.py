"""3D GREIT reconstruction-matrix helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy import sparse
from scipy.stats import qmc

from pyeidors.data.channels import (
    apply_measurement_contract_to_jacobian,
    apply_measurement_contract_to_vector,
)
from pyeidors.data.structures import EITImage
from pyeidors.inverse.dual_mesh import CellMesh, VoxelGrid
from pyeidors.inverse.reconstruction_matrix import reconstruct_difference_batch
from pyeidors.perf.gpu_kernels import RMMatmulHandle, RMMatmulResult, prepare_rm_matmul

GREIT_METRIC_KEYS = ("AR", "PE", "RES", "SD", "RNG")
GREIT_RM_HDF5_SCHEMA = "pyeidors-greit-rm-hdf5-v1"
GREIT_EIDORS_HDF5_SCHEMA = "pyeidors-greit-eidors-hdf5-v1"
GREIT_CACHE_SIGNATURE_SCHEMA = "pyeidors-greit-cache-signature-v1"
GREIT_DESIRED_IMAGE_DEFAULT_SAMPLING = "gauss"


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
class GREITFiniteTargetResponses:
    """Finite-target GREIT forward responses for EIDORS parity mode."""

    vh: np.ndarray
    vi: np.ndarray
    y: np.ndarray
    contracted_y: np.ndarray
    xyzr: np.ndarray
    conductivities: np.ndarray
    metadata: MappingProxyType

    @property
    def n_targets(self) -> int:
        return int(self.vi.shape[1])

    @property
    def n_measurements(self) -> int:
        return int(self.vh.size)


@dataclass(frozen=True)
class GREITDesiredImages:
    """Desired image stack ``D`` for EIDORS-parity GREIT RM construction."""

    values: np.ndarray
    xyz: np.ndarray
    radii: np.ndarray
    rec_centers: np.ndarray
    metadata: MappingProxyType

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


@dataclass(frozen=True)
class GREITRMComponents:
    """EIDORS ``calc_GREIT_RM`` component bundle."""

    rm: np.ndarray
    pjt: np.ndarray
    m: np.ndarray
    sn: np.ndarray
    noiselev: float
    weight: float
    y: np.ndarray
    d: np.ndarray
    metadata: MappingProxyType


@dataclass(frozen=True)
class GREITNativeTrainingPipeline:
    """Native PyEIDORS GREIT training pipeline artifact.

    The bundle makes the full offline path explicit: target distribution,
    PyEIDORS forward responses ``vh/vi/Y``, desired images ``D``, and the final
    RM built by the shared GREIT algebra.
    """

    distribution: GREIT3DDistribution | None
    responses: GREITFiniteTargetResponses
    desired_images: GREITDesiredImages
    greit: GREITRM
    metadata: MappingProxyType

    @property
    def rm(self) -> np.ndarray:
        return self.greit.rm

    @property
    def y(self) -> np.ndarray:
        return self.responses.contracted_y

    @property
    def d(self) -> np.ndarray:
        return self.desired_images.values


@dataclass(frozen=True)
class GREITWeightSearchResult:
    """Scalar GREIT weight search result over ``log10(weight)``."""

    weight: float
    log10_weight: float
    target_metric: float
    achieved_metric: float
    objective_value: float
    initial_bracket: tuple[float, float]
    bracket: tuple[float, float]
    evaluations: int
    metadata: MappingProxyType


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
    pjt: np.ndarray | None = None
    m: np.ndarray | None = None
    sn: np.ndarray | None = None
    y: np.ndarray | None = None
    d: np.ndarray | None = None
    vh: np.ndarray | None = None
    vi: np.ndarray | None = None
    xyzr: np.ndarray | None = None
    rec_model: np.ndarray | None = None
    fwd_model_signature: str | None = None
    cache_signature: str | None = None
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

        from pyeidors.io.hdf5_artifacts import write_large_cache_hdf5_artifact

        target = _greit_hdf5_path(path)
        metadata = dict(self.metadata)
        metadata.setdefault("legacy_artifact_schema", metadata.get("artifact_schema"))
        schema = _greit_artifact_schema(metadata)
        metadata["artifact_schema"] = schema
        metadata["artifact_format"] = "hdf5"
        metadata.setdefault("component_storage", "eidors_components")
        write_large_cache_hdf5_artifact(
            target,
            _greit_artifact_arrays(self, schema=schema, metadata=metadata),
            metadata,
            schema=schema,
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
        rm_array = _array_from_aliases(arrays, "rm", "RM")
        if rm_array is None:
            raise ValueError(f"GREIT artifact is missing 'rm': {path}")
        voxel_raw = np.asarray(arrays.get("voxel_shape", ()), dtype=np.int64)
        metadata = dict(artifact.metadata)
        return cls(
            rm=np.asarray(rm_array, dtype=np.float64),
            metadata=MappingProxyType(metadata),
            voxel_shape=tuple(int(v) for v in voxel_raw) if voxel_raw.size else None,
            channel_mask=_empty_to_none_array(arrays.get("channel_mask"), dtype=bool),
            measurement_weights=_empty_to_none_array(arrays.get("measurement_weights")),
            training_targets=_empty_to_none_array(arrays.get("training_targets")),
            training_responses=_empty_to_none_array(arrays.get("training_responses")),
            pjt=_empty_to_none_array(_array_from_aliases(arrays, "pjt", "PJt")),
            m=_empty_to_none_array(_array_from_aliases(arrays, "m", "M")),
            sn=_empty_to_none_array(_array_from_aliases(arrays, "sn", "Sn")),
            y=_empty_to_none_array(_array_from_aliases(arrays, "y", "Y")),
            d=_empty_to_none_array(_array_from_aliases(arrays, "d", "D")),
            vh=_empty_to_none_array(arrays.get("vh")),
            vi=_empty_to_none_array(arrays.get("vi")),
            xyzr=_empty_to_none_array(arrays.get("xyzr")),
            rec_model=_empty_to_none_array(arrays.get("rec_model")),
            fwd_model_signature=_utf8_bytes_to_string(arrays.get("fwd_model_signature"))
            or metadata.get("fwd_model_signature"),
            cache_signature=metadata.get("cache_signature_hash"),
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


def build_greit_finite_target_responses(
    fwd_model: Any,
    *,
    distribution: GREIT3DDistribution | None = None,
    targets: GREITTrainingTargets | None = None,
    centers: Any | None = None,
    target_radius: float | None = None,
    target_size: float | None = None,
    target_plane: Any | None = None,
    target_offset: Any | None = None,
    target_contrast: Any = 1.0,
    background_conductivity: Any = 1.0,
    normalize: bool = True,
    measurement_order: Any | None = None,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    batch_size: int | None = None,
    response_cache: dict[str, GREITFiniteTargetResponses] | None = None,
    cache_key: str | None = None,
) -> GREITFiniteTargetResponses:
    """Simulate finite-target ``vh``/``vi`` training responses.

    This is the EIDORS-parity training path for T42: homogeneous data ``vh``
    are solved once, each finite target produces one inhomogeneous column in
    ``vi``, and ``Y`` follows EIDORS ``calc_difference_data`` orientation
    ``(n_measurements, n_targets)``.
    """

    if (
        response_cache is not None
        and cache_key is not None
        and cache_key in response_cache
    ):
        cached = response_cache[cache_key]
        meta = dict(cached.metadata)
        meta["cache_hit"] = True
        return replace(cached, metadata=MappingProxyType(meta))

    fwd_centers = _forward_cell_centers(fwd_model)
    background = _resolve_background_conductivity(
        background_conductivity,
        n_cells=fwd_centers.shape[0],
    )
    target_centers, target_radii, target_source = _resolve_finite_target_geometry(
        distribution=distribution,
        targets=targets,
        centers=centers,
        target_radius=target_radius,
        target_size=target_size,
    )
    target_centers, plane_metadata = _apply_target_plane_offset(
        target_centers,
        target_plane=target_plane,
        target_offset=target_offset,
    )
    target_contrasts = _as_target_contrasts(
        target_contrast,
        n_targets=target_centers.shape[0],
    )
    resolved_batch_size = _resolve_batch_size(
        batch_size, n_targets=target_centers.shape[0]
    )

    conductivities = _build_finite_target_conductivities(
        fwd_centers,
        background=background,
        target_centers=target_centers,
        target_radii=target_radii,
        target_contrasts=target_contrasts,
    )
    vh = _solve_measurement_vector(fwd_model, background)
    vi = _solve_measurement_batch(
        fwd_model,
        conductivities,
        batch_size=resolved_batch_size,
    )
    if vi.shape[0] != vh.size:
        raise ValueError(
            f"vi measurement rows {vi.shape[0]} do not match vh length {vh.size}."
        )
    order, order_metadata = _resolve_measurement_order(
        measurement_order,
        n_measurements=vh.size,
    )
    if order is not None:
        vh = np.ascontiguousarray(vh[order], dtype=np.float64)
        vi = np.ascontiguousarray(vi[order, :], dtype=np.float64)
    y = _calc_greit_difference_data(vh, vi, normalize=normalize)
    contracted_y, measurement_contract = _contract_training_responses(
        y,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    xyzr = np.vstack(
        [
            target_centers.T,
            target_radii.reshape(1, -1),
        ]
    )

    metadata = MappingProxyType(
        {
            "training_mode": "forward",
            "eidors_parity": True,
            "response_orientation": "measurements_by_targets",
            "difference_normalization": "ratio" if normalize else "raw",
            "target_source": target_source,
            "target_radius_source": "target_radius"
            if target_radius is not None
            else "target_size"
            if target_size is not None
            else "targets_or_default",
            "target_contrast_mode": "additive_conductivity_delta",
            "n_measurements": int(vh.size),
            "n_targets": int(target_centers.shape[0]),
            "n_forward_parameters": int(fwd_centers.shape[0]),
            "batch_size": resolved_batch_size,
            "cache_key": cache_key,
            "cache_hit": False,
            **order_metadata,
            "bad_channel_count": int(measurement_contract.bad_channel_count),
            "measurement_weight_kind": measurement_contract.weight_kind,
            **plane_metadata,
        }
    )
    responses = GREITFiniteTargetResponses(
        vh=vh,
        vi=vi,
        y=y,
        contracted_y=contracted_y,
        xyzr=np.ascontiguousarray(xyzr, dtype=np.float64),
        conductivities=conductivities,
        metadata=metadata,
    )
    if response_cache is not None and cache_key is not None:
        response_cache[cache_key] = responses
    return responses


def build_greit_desired_images(
    rec_model: Any,
    *,
    xyz: Any | None = None,
    radius: Any | None = None,
    responses: GREITFiniteTargetResponses | None = None,
    distribution: GREIT3DDistribution | None = None,
    desired_solution_fn: Any | None = None,
    desired_options: dict[str, Any] | None = None,
    target_values: Any | None = None,
) -> GREITDesiredImages:
    """Build the GREIT desired image matrix ``D``.

    ``D`` is independent from raw synthetic target matrix ``T`` by default.
    Passing ``desired_solution_fn="target_values"`` is the explicit opt-in
    escape hatch for legacy ``D≈T`` experiments.
    """

    rec_centers = _desired_rec_centers(rec_model)
    xyz_matrix, radii, xyz_source = _resolve_desired_xyz_radius(
        xyz=xyz,
        radius=radius,
        responses=responses,
        distribution=distribution,
    )
    options = dict(desired_options or {})
    options.setdefault("rec_model", rec_model)
    options.setdefault("rec_centers", rec_centers)
    options.setdefault("n_rec_parameters", int(rec_centers.shape[0]))
    options.setdefault("dimension", int(rec_centers.shape[1]))

    if desired_solution_fn is None:
        mode = _desired_sampling_mode(options)
        options["desired_img_sampling"] = mode
        raw_values = greit_desired_image_sigmoid(xyz_matrix, radii, options)
        fn_label = f"GREIT_desired_img_sigmoid:{mode}"
        parity_default = True
        target_values_used = False
    elif isinstance(desired_solution_fn, str):
        token = desired_solution_fn.strip().lower().replace("-", "_")
        if token in {
            "greit_desired_img",
            "greit_desired_img_sigmoid",
            "sigmoid",
            "default",
        }:
            mode = _desired_sampling_mode(options)
            options["desired_img_sampling"] = mode
            raw_values = greit_desired_image_sigmoid(xyz_matrix, radii, options)
            fn_label = f"GREIT_desired_img_sigmoid:{mode}"
            parity_default = True
            target_values_used = False
        elif token in _DESIRED_SAMPLING_MODE_ALIASES:
            mode = _desired_sampling_mode(options, explicit=token)
            options["desired_img_sampling"] = mode
            raw_values = greit_desired_image_sigmoid(xyz_matrix, radii, options)
            fn_label = f"GREIT_desired_img_sigmoid:{mode}"
            parity_default = mode != "center"
            target_values_used = False
        elif token in {"target_values", "target", "raw_target", "t"}:
            options["desired_img_sampling"] = "target_values"
            raw_values = _desired_from_target_values(
                target_values,
                n_rec_parameters=rec_centers.shape[0],
                n_targets=xyz_matrix.shape[1],
            )
            fn_label = "target_values_explicit_opt_in"
            parity_default = False
            target_values_used = True
        else:
            raise ValueError(
                "desired_solution_fn string must be one of: "
                "'sigmoid', 'center', 'gauss', 'adaptive_gauss', "
                "'sobol_qmc', or 'target_values'."
            )
    elif callable(desired_solution_fn):
        options["desired_img_sampling"] = "custom_callable"
        raw_values = desired_solution_fn(xyz_matrix, radii, MappingProxyType(options))
        fn_label = getattr(desired_solution_fn, "__name__", "custom_callable")
        parity_default = False
        target_values_used = False
    else:
        raise TypeError("desired_solution_fn must be None, a string, or callable.")

    values = _validate_desired_matrix(
        raw_values,
        n_rec_parameters=rec_centers.shape[0],
        n_targets=xyz_matrix.shape[1],
    )
    metadata = MappingProxyType(
        {
            "builder": "GREIT_desired_img",
            "desired_solution_fn": fn_label,
            "desired_image_sampling": options.get(
                "desired_img_sampling", "custom_callable"
            ),
            "desired_image_default_sampling": GREIT_DESIRED_IMAGE_DEFAULT_SAMPLING,
            "eidors_component_parity": parity_default,
            "target_values_used": target_values_used,
            "target_values_requires_explicit_opt_in": True,
            "xyz_source": xyz_source,
            "d_shape": tuple(int(v) for v in values.shape),
            "xyz_shape": tuple(int(v) for v in xyz_matrix.shape),
            "n_rec_parameters": int(rec_centers.shape[0]),
            "n_targets": int(xyz_matrix.shape[1]),
            "radius_min": float(np.min(radii)),
            "radius_max": float(np.max(radii)),
            "coordinate_mode": "rec_model_physical",
        }
    )
    return GREITDesiredImages(
        values=values,
        xyz=xyz_matrix,
        radii=radii,
        rec_centers=rec_centers,
        metadata=metadata,
    )


def greit_desired_image_sigmoid(
    xyz: Any,
    radius: Any,
    options: Any,
) -> np.ndarray:
    """Default EIDORS-like sigmoid desired image function.

    The public signature mirrors EIDORS' ``desired_solution_fn(xyz, radius,
    options)`` hook.  The implementation can either sample cell centres or
    approximate the element-average target image by quadrature/QMC sampling,
    producing ``D`` with shape ``n_rec_parameters × n_targets``.
    """

    opts = dict(options or {})
    if "desired_img_radius" in opts and opts["desired_img_radius"] is not None:
        radius = opts["desired_img_radius"]
    rec_model = opts.get("rec_model")
    rec_centers = _desired_rec_centers(
        opts.get("rec_centers") if opts.get("rec_centers") is not None else rec_model
    )
    xyz_matrix, embedded_radii = _as_eidors_xyz(xyz)
    radii = _desired_radii(
        radius if radius is not None else embedded_radii,
        n_targets=xyz_matrix.shape[1],
        xyz=xyz_matrix,
    )
    steepness = _desired_steepness(
        opts.get("desired_img_steepness", opts.get("sigmoid_steepness", 10.0)),
        n_targets=xyz_matrix.shape[1],
    )
    threshold = float(opts.get("desired_img_threshold", 1e-4))
    if threshold < 0.0 or threshold >= 0.5 or not np.isfinite(threshold):
        raise ValueError("desired_img_threshold must be finite in [0, 0.5).")

    mode = _desired_sampling_mode(opts)
    if mode == "center":
        desired = _greit_sigmoid_from_centers(
            rec_centers,
            xyz_matrix,
            radii,
            steepness,
        )
    elif mode == "adaptive_gauss":
        desired = _greit_sigmoid_adaptive_gauss(
            rec_model,
            rec_centers,
            xyz_matrix,
            radii,
            steepness,
            opts,
        )
    else:
        samples, weights, _ = _desired_cell_samples(
            rec_model,
            rec_centers,
            mode=mode,
            options=opts,
        )
        desired = _greit_sigmoid_average_over_samples(
            samples,
            weights,
            xyz_matrix,
            radii,
            steepness,
        )
    desired = _postprocess_desired_image(desired, threshold=threshold, options=opts)

    return np.ascontiguousarray(desired, dtype=np.float64)


_DESIRED_SAMPLING_MODE_ALIASES = {
    "center": "center",
    "centre": "center",
    "center_sample": "center",
    "center_sampled": "center",
    "centre_sampled": "center",
    "cell_center": "center",
    "cell_centers": "center",
    "fast": "center",
    "gauss": "gauss",
    "gaussian": "gauss",
    "gauss_quadrature": "gauss",
    "quadrature": "gauss",
    "fixed_gauss": "gauss",
    "element_integrated": "gauss",
    "adaptive": "adaptive_gauss",
    "adaptive_gauss": "adaptive_gauss",
    "adaptive_gaussian": "adaptive_gauss",
    "adaptive_quadrature": "adaptive_gauss",
    "sobol": "sobol_qmc",
    "sobol_qmc": "sobol_qmc",
    "qmc": "sobol_qmc",
    "quasi_monte_carlo": "sobol_qmc",
}


def _desired_sampling_mode(
    options: dict[str, Any] | None,
    *,
    explicit: Any | None = None,
) -> str:
    opts = dict(options or {})
    raw = explicit
    if raw is None:
        for key in (
            "desired_img_sampling",
            "desired_image_sampling",
            "desired_img_integration",
            "desired_integration",
            "integration_mode",
            "sampling_mode",
        ):
            if opts.get(key) is not None:
                raw = opts[key]
                break
    if raw is None or str(raw).strip() == "":
        raw = GREIT_DESIRED_IMAGE_DEFAULT_SAMPLING
    token = str(raw).strip().lower().replace("-", "_")
    try:
        return _DESIRED_SAMPLING_MODE_ALIASES[token]
    except KeyError as exc:
        valid = ", ".join(sorted(set(_DESIRED_SAMPLING_MODE_ALIASES.values())))
        raise ValueError(
            f"unknown desired image sampling mode {raw!r}; use {valid}."
        ) from exc


def _greit_sigmoid_from_centers(
    rec_centers: np.ndarray,
    xyz_matrix: np.ndarray,
    radii: np.ndarray,
    steepness: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(
        rec_centers[:, None, :3] - xyz_matrix.T[None, :, :],
        axis=2,
    )
    return _greit_sigmoid_from_distances(distances, radii, steepness)


def _greit_sigmoid_from_distances(
    distances: np.ndarray,
    radii: np.ndarray,
    steepness: np.ndarray,
) -> np.ndarray:
    scaled = steepness.reshape(1, -1) * (distances / radii.reshape(1, -1) - 1.0)
    return 1.0 / (1.0 + np.exp(np.clip(scaled, -700.0, 700.0)))


def _greit_sigmoid_average_over_samples(
    samples: np.ndarray,
    weights: np.ndarray,
    xyz_matrix: np.ndarray,
    radii: np.ndarray,
    steepness: np.ndarray,
) -> np.ndarray:
    n_cells, n_samples, _ = samples.shape
    flat_samples = samples.reshape(n_cells * n_samples, 3)
    desired = np.empty((n_cells, xyz_matrix.shape[1]), dtype=np.float64)
    for target_idx in range(xyz_matrix.shape[1]):
        distances = np.linalg.norm(
            flat_samples - xyz_matrix[:, target_idx].reshape(1, 3),
            axis=1,
        ).reshape(n_cells, n_samples)
        values = _greit_sigmoid_from_distances(
            distances,
            radii[target_idx : target_idx + 1],
            steepness[target_idx : target_idx + 1],
        )
        desired[:, target_idx] = values @ weights
    return desired


def _greit_sigmoid_adaptive_gauss(
    rec_model: Any,
    rec_centers: np.ndarray,
    xyz_matrix: np.ndarray,
    radii: np.ndarray,
    steepness: np.ndarray,
    options: dict[str, Any],
) -> np.ndarray:
    base_options = dict(options)
    base_options["desired_img_gauss_order"] = int(
        options.get("desired_img_adaptive_base_order", 2)
    )
    base_samples, base_weights, extents = _desired_cell_samples(
        rec_model,
        rec_centers,
        mode="gauss",
        options=base_options,
    )
    desired = _greit_sigmoid_average_over_samples(
        base_samples,
        base_weights,
        xyz_matrix,
        radii,
        steepness,
    )
    half_diagonal = 0.5 * np.linalg.norm(extents, axis=1)
    center_distances = np.linalg.norm(
        rec_centers[:, None, :3] - xyz_matrix.T[None, :, :],
        axis=2,
    )
    band = _desired_adaptive_band(options, steepness)
    fine_order = int(options.get("desired_img_adaptive_fine_order", 5))
    if fine_order <= 0:
        raise ValueError("desired_img_adaptive_fine_order must be positive.")
    fine_offsets, fine_weights = _gauss_offsets_for_extents(extents, fine_order)
    for target_idx in range(xyz_matrix.shape[1]):
        boundary_distance = np.abs(center_distances[:, target_idx] - radii[target_idx])
        refine = (
            boundary_distance <= half_diagonal + band[target_idx] * radii[target_idx]
        )
        if not np.any(refine):
            continue
        fine_samples = (
            rec_centers[refine, None, :3]
            + fine_offsets[None, :, :] * extents[refine, None, :]
        )
        refined = _greit_sigmoid_average_over_samples(
            fine_samples,
            fine_weights,
            xyz_matrix[:, target_idx : target_idx + 1],
            radii[target_idx : target_idx + 1],
            steepness[target_idx : target_idx + 1],
        )
        desired[refine, target_idx] = refined[:, 0]
    return desired


def _gauss_offsets_for_extents(
    extents: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.any(extents > np.finfo(np.float64).eps, axis=0)
    if not np.any(active):
        return (
            np.zeros((1, 3), dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )
    offsets_active, weights = _gauss_reference_offsets(int(np.sum(active)), order)
    offsets = np.zeros((offsets_active.shape[0], 3), dtype=np.float64)
    offsets[:, active] = offsets_active
    return np.ascontiguousarray(offsets), weights


def _desired_cell_samples(
    rec_model: Any,
    rec_centers: np.ndarray,
    *,
    mode: str,
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    extents = _desired_cell_extents(rec_model, rec_centers, options)
    active = np.any(extents > np.finfo(np.float64).eps, axis=0)
    if not np.any(active):
        samples = rec_centers[:, None, :3].copy()
        weights = np.ones(1, dtype=np.float64)
        return samples, weights, extents

    if mode == "gauss":
        order = int(options.get("desired_img_gauss_order", 3))
        if order <= 0:
            raise ValueError("desired_img_gauss_order must be positive.")
        offsets_active, weights = _gauss_reference_offsets(int(np.sum(active)), order)
    elif mode == "sobol_qmc":
        requested = int(options.get("desired_img_sobol_samples", 32))
        if requested <= 0:
            raise ValueError("desired_img_sobol_samples must be positive.")
        seed = int(options.get("desired_img_sobol_seed", 0))
        scramble = bool(options.get("desired_img_sobol_scramble", True))
        offsets_active, weights = _sobol_reference_offsets(
            int(np.sum(active)),
            requested,
            seed=seed,
            scramble=scramble,
        )
    else:
        raise ValueError(f"unsupported desired image sampling mode {mode!r}.")

    offsets = np.zeros((offsets_active.shape[0], 3), dtype=np.float64)
    offsets[:, active] = offsets_active
    samples = rec_centers[:, None, :3] + offsets[None, :, :] * extents[:, None, :]
    return (
        np.ascontiguousarray(samples, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(extents, dtype=np.float64),
    )


def _desired_cell_extents(
    rec_model: Any,
    rec_centers: np.ndarray,
    options: dict[str, Any],
) -> np.ndarray:
    explicit = options.get("desired_img_cell_extents")
    if explicit is None:
        explicit = options.get("cell_extents")
    if explicit is not None:
        return _as_desired_cell_extents(explicit, n_cells=rec_centers.shape[0])

    spacing = options.get("desired_img_cell_spacing")
    if spacing is None:
        spacing = options.get("cell_spacing")
    if spacing is not None:
        extent = _as_extent_vector(spacing)
        return np.broadcast_to(extent, (rec_centers.shape[0], 3)).copy()

    if isinstance(rec_model, VoxelGrid):
        extent = _as_extent_vector(rec_model.spacing)
        return np.broadcast_to(extent, (rec_centers.shape[0], 3)).copy()

    if isinstance(rec_model, CellMesh):
        vertices = rec_model.coordinates[rec_model.cells]
        extents = vertices.max(axis=1) - vertices.min(axis=1)
        return _as_desired_cell_extents(extents, n_cells=rec_centers.shape[0])

    if isinstance(rec_model, dict):
        for key in ("cell_extents", "extents", "cell_spacing", "spacing"):
            if rec_model.get(key) is not None:
                raw = rec_model[key]
                if "extent" in key:
                    return _as_desired_cell_extents(raw, n_cells=rec_centers.shape[0])
                extent = _as_extent_vector(raw)
                return np.broadcast_to(extent, (rec_centers.shape[0], 3)).copy()

    for attr_name in ("cell_extents", "extents", "cell_spacing", "spacing"):
        attr = getattr(rec_model, attr_name, None)
        if attr is None:
            continue
        raw = attr() if callable(attr) else attr
        if "extent" in attr_name:
            return _as_desired_cell_extents(raw, n_cells=rec_centers.shape[0])
        extent = _as_extent_vector(raw)
        return np.broadcast_to(extent, (rec_centers.shape[0], 3)).copy()

    extent = _infer_center_spacing(rec_centers)
    return np.broadcast_to(extent, (rec_centers.shape[0], 3)).copy()


def _as_desired_cell_extents(values: Any, *, n_cells: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        extent = _as_extent_vector(array)
        array = np.broadcast_to(extent, (n_cells, 3)).copy()
    elif array.ndim == 2:
        if array.shape[0] != n_cells:
            raise ValueError(
                f"cell_extents rows {array.shape[0]} do not match {n_cells} cells."
            )
        if array.shape[1] > 3:
            raise ValueError("cell_extents must have at most 3 columns.")
        if array.shape[1] < 3:
            array = np.pad(array, ((0, 0), (0, 3 - array.shape[1])))
        else:
            array = array[:, :3]
    else:
        raise ValueError("cell_extents must be a vector or 2D matrix.")
    if not np.isfinite(array).all():
        raise FloatingPointError("cell_extents contain non-finite values.")
    if np.any(array < 0.0):
        raise ValueError("cell_extents entries must be non-negative.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_extent_vector(values: Any) -> np.ndarray:
    extent = np.asarray(values, dtype=np.float64).reshape(-1)
    if extent.size == 1:
        extent = np.repeat(float(extent[0]), 3)
    elif extent.size < 3:
        extent = np.pad(extent, (0, 3 - extent.size))
    elif extent.size > 3:
        raise ValueError("cell spacing/extents must have at most 3 entries.")
    if not np.isfinite(extent).all():
        raise FloatingPointError("cell spacing/extents contain non-finite values.")
    if np.any(extent < 0.0):
        raise ValueError("cell spacing/extents entries must be non-negative.")
    return np.ascontiguousarray(extent[:3], dtype=np.float64)


def _infer_center_spacing(rec_centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(rec_centers, dtype=np.float64)
    spacing = np.zeros(3, dtype=np.float64)
    for axis in range(min(3, centers.shape[1])):
        coords = np.unique(np.round(centers[:, axis], decimals=12))
        diffs = np.diff(np.sort(coords))
        diffs = diffs[diffs > np.finfo(np.float64).eps]
        if diffs.size:
            spacing[axis] = float(np.median(diffs))
    if not np.any(spacing > 0.0):
        nearest = _nearest_center_distance(centers[:, :3])
        if nearest > 0.0:
            spacing[:] = nearest
    return np.ascontiguousarray(spacing, dtype=np.float64)


def _nearest_center_distance(centers: np.ndarray) -> float:
    if centers.shape[0] <= 1:
        return 0.0
    distances = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :],
        axis=2,
    )
    distances[distances <= np.finfo(np.float64).eps] = np.inf
    nearest = float(np.min(distances))
    return 0.0 if not np.isfinite(nearest) else nearest


def _gauss_reference_offsets(
    dimension: int,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights_1d = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * nodes
    weights_1d = 0.5 * weights_1d
    grids = np.meshgrid(*([nodes] * dimension), indexing="ij")
    weight_grids = np.meshgrid(*([weights_1d] * dimension), indexing="ij")
    offsets = np.stack([grid.ravel(order="C") for grid in grids], axis=1)
    weights = np.prod(
        np.stack([grid.ravel(order="C") for grid in weight_grids], axis=1),
        axis=1,
    )
    weights = weights / float(np.sum(weights))
    return np.ascontiguousarray(offsets), np.ascontiguousarray(weights)


def _sobol_reference_offsets(
    dimension: int,
    n_samples: int,
    *,
    seed: int,
    scramble: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_power = 1 << int(np.ceil(np.log2(n_samples)))
    sampler = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
    points = sampler.random_base2(int(np.log2(n_power)))
    offsets = points - 0.5
    weights = np.full(offsets.shape[0], 1.0 / offsets.shape[0], dtype=np.float64)
    return np.ascontiguousarray(offsets), weights


def _desired_adaptive_band(
    options: dict[str, Any],
    steepness: np.ndarray,
) -> np.ndarray:
    value = options.get("desired_img_adaptive_band")
    if value is None:
        band = 1.0 / steepness
    else:
        band = np.asarray(value, dtype=np.float64).reshape(-1)
        if band.size == 1:
            band = np.full(steepness.size, float(band[0]), dtype=np.float64)
        elif band.size != steepness.size:
            raise ValueError(
                "desired_img_adaptive_band length must be one or n_targets."
            )
    if not np.isfinite(band).all():
        raise FloatingPointError(
            "desired_img_adaptive_band contains non-finite values."
        )
    if np.any(band < 0.0):
        raise ValueError("desired_img_adaptive_band entries must be non-negative.")
    return np.ascontiguousarray(band, dtype=np.float64)


def _postprocess_desired_image(
    desired: np.ndarray,
    *,
    threshold: float,
    options: dict[str, Any],
) -> np.ndarray:
    desired = np.asarray(desired, dtype=np.float64)
    if threshold > 0.0:
        desired = desired.copy()
        desired[desired < threshold] = 0.0
        desired[desired > 1.0 - threshold] = 1.0
    if bool(options.get("normalize_peak", False)):
        peaks = np.max(desired, axis=0)
        good = peaks > np.finfo(np.float64).eps
        desired[:, good] = desired[:, good] / peaks[good].reshape(1, -1)
    return np.ascontiguousarray(desired, dtype=np.float64)


def calc_greit_rm(
    y: Any,
    d: Any,
    *,
    weight: Any = 0.5,
    noise_covar: Any = 1.0,
    pjt_cache: Any | None = None,
) -> GREITRMComponents:
    """Replicate EIDORS ``calc_GREIT_RM`` matrix algebra.

    Inputs follow EIDORS component orientation: ``Y`` is
    ``n_measurements x n_targets`` and ``D`` is
    ``n_rec_parameters x n_targets``.  The scalar ``weight`` is converted to
    effective ``noiselev`` via ``weight * mean(abs(Y))`` before forming
    ``M``.
    """

    y_matrix = _validate_training_response_matrix(y)
    d_matrix = _validate_desired_component_matrix(d, n_targets=y_matrix.shape[1])
    scalar_weight = _as_scalar_weight(weight)
    sn, sn_source = _noise_covar_matrix(
        noise_covar,
        n_measurements=y_matrix.shape[0],
    )

    component_dtype = np.result_type(y_matrix, d_matrix, sn)
    if pjt_cache is None:
        pjt = np.ascontiguousarray(d_matrix @ y_matrix.T, dtype=component_dtype)
        pjt_source = "computed"
    else:
        pjt = _validate_pjt_cache(
            pjt_cache,
            n_rec_parameters=d_matrix.shape[0],
            n_measurements=y_matrix.shape[0],
            dtype=component_dtype,
        )
        pjt_source = "provided_cache"
    noiselev = float(scalar_weight * np.mean(np.abs(y_matrix)))
    m = np.ascontiguousarray(
        y_matrix @ y_matrix.T + (noiselev * noiselev) * sn,
        dtype=component_dtype,
    )

    rhs_t = pjt.T
    try:
        rm = np.linalg.solve(m.T, rhs_t).T
        solver = "solve"
        singular_fallback = False
    except np.linalg.LinAlgError:
        rm = (np.linalg.pinv(m.T) @ rhs_t).T
        solver = "pinv"
        singular_fallback = True
    rm = np.ascontiguousarray(rm, dtype=component_dtype)
    if not np.isfinite(rm).all():
        raise FloatingPointError("GREIT RM contains non-finite values.")

    try:
        condition = float(np.linalg.cond(m))
    except np.linalg.LinAlgError:
        condition = float("inf")
    rank = int(np.linalg.matrix_rank(m))
    metadata = MappingProxyType(
        {
            "algorithm": "calc_GREIT_RM",
            "eidors_component_parity": not singular_fallback,
            "pjt_shape": tuple(int(v) for v in pjt.shape),
            "pjt_source": pjt_source,
            "y_shape": tuple(int(v) for v in y_matrix.shape),
            "d_shape": tuple(int(v) for v in d_matrix.shape),
            "sn_shape": tuple(int(v) for v in sn.shape),
            "m_shape": tuple(int(v) for v in m.shape),
            "rm_shape": tuple(int(v) for v in rm.shape),
            "weight": scalar_weight,
            "noiselev": noiselev,
            "noise_covar_source": sn_source,
            "solver": solver,
            "singular_fallback": singular_fallback,
            "matrix_rank": rank,
            "matrix_condition": condition,
            "transpose_semantics": "matlab_nonconjugate_dot_transpose",
        }
    )
    return GREITRMComponents(
        rm=rm,
        pjt=pjt,
        m=m,
        sn=sn,
        noiselev=noiselev,
        weight=scalar_weight,
        y=y_matrix,
        d=d_matrix,
        metadata=metadata,
    )


def search_greit_weight_for_metric(
    metric_fn,
    *,
    target_metric: float,
    bracket: tuple[float, float] = (-2.0, 2.0),
    tolerance: float = 1.0e-3,
    maxiter: int = 64,
    max_expand: int = 3,
    boundary_margin: float | None = None,
) -> GREITWeightSearchResult:
    """Choose scalar GREIT weight by bounded search over ``log10(weight)``.

    ``metric_fn`` is intentionally a tiny injectable objective seam: it
    receives ``log10(weight)`` and returns the achieved NF/image-SNR scalar.
    It does not know about ``calc_greit_rm`` or GREIT matrices.
    """

    target = float(target_metric)
    if not np.isfinite(target):
        raise ValueError("target_metric must be finite.")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive.")
    if maxiter < 3:
        raise ValueError("maxiter must be at least 3.")
    lo, hi = _validate_log10_bracket(bracket)
    initial_bracket = (lo, hi)
    records: list[tuple[float, float, float]] = []

    def objective(log10_weight: float) -> float:
        metric = float(metric_fn(float(log10_weight)))
        if not np.isfinite(metric):
            raise FloatingPointError("GREIT weight metric returned non-finite value.")
        value = float((metric - target) ** 2)
        records.append((float(log10_weight), metric, value))
        return value

    expansions = 0
    result = None
    success = False
    message = ""
    for attempt in range(max_expand + 1):
        result = _bounded_minimize(
            objective,
            lo=lo,
            hi=hi,
            tolerance=tolerance,
            maxiter=maxiter,
        )
        success = bool(result["success"])
        message = str(result["message"])
        best_x = float(result["x"])
        if attempt >= max_expand or _inside_bracket(
            best_x,
            lo=lo,
            hi=hi,
            boundary_margin=boundary_margin,
        ):
            break
        width = hi - lo
        if best_x <= lo + 0.1 * width:
            hi = lo
            lo = lo - width
        else:
            lo = hi
            hi = hi + width
        expansions += 1

    if result is None:  # pragma: no cover - defensive
        raise RuntimeError("GREIT weight search did not run.")
    log10_weight = float(result["x"])
    achieved = float(metric_fn(log10_weight))
    objective_value = float((achieved - target) ** 2)
    records.append((log10_weight, achieved, objective_value))
    metadata = MappingProxyType(
        {
            "algorithm": "bounded_log10_weight_search",
            "search_variable": "log10_weight",
            "initial_bracket": initial_bracket,
            "bracket": (float(lo), float(hi)),
            "bracket_expansions": expansions,
            "target_metric": target,
            "achieved_metric": achieved,
            "objective_value": objective_value,
            "tolerance": float(tolerance),
            "maxiter": int(maxiter),
            "boundary_margin": None
            if boundary_margin is None
            else float(boundary_margin),
            "evaluations": len(records),
            "success": success,
            "message": message,
            "best_log10_weight": log10_weight,
            "best_weight": float(10.0**log10_weight),
        }
    )
    return GREITWeightSearchResult(
        weight=float(10.0**log10_weight),
        log10_weight=log10_weight,
        target_metric=target,
        achieved_metric=achieved,
        objective_value=objective_value,
        initial_bracket=initial_bracket,
        bracket=(float(lo), float(hi)),
        evaluations=len(records),
        metadata=metadata,
    )


def optimize_greit_weight_for_metric(
    y: Any,
    d: Any,
    *,
    target_metric: float,
    metric: str = "noise_figure",
    noise_covar: Any = 1.0,
    measurement_noise: Any | None = None,
    pjt_cache: Any | None = None,
    bracket: tuple[float, float] = (-2.0, 2.0),
    tolerance: float = 1.0e-3,
    maxiter: int = 64,
) -> GREITWeightSearchResult:
    """Optimize scalar GREIT weight against simulated NF/image-SNR metric."""

    metric_name = _normalize_weight_metric(metric)
    y_matrix = _validate_training_response_matrix(y)
    d_matrix = _validate_desired_component_matrix(d, n_targets=y_matrix.shape[1])
    noise = _measurement_noise_matrix(measurement_noise, y=y_matrix)
    pjt = (
        np.ascontiguousarray(
            d_matrix @ y_matrix.T,
            dtype=np.result_type(y_matrix, d_matrix),
        )
        if pjt_cache is None
        else _validate_pjt_cache(
            pjt_cache,
            n_rec_parameters=d_matrix.shape[0],
            n_measurements=y_matrix.shape[0],
            dtype=np.result_type(y_matrix, d_matrix),
        )
    )

    def metric_fn(log10_weight: float) -> float:
        components = calc_greit_rm(
            y_matrix,
            d_matrix,
            weight=10.0 ** float(log10_weight),
            noise_covar=noise_covar,
            pjt_cache=pjt,
        )
        return _greit_noise_metric(
            y_matrix,
            d_matrix,
            components.rm,
            noise=noise,
            metric=metric_name,
        )

    result = search_greit_weight_for_metric(
        metric_fn,
        target_metric=target_metric,
        bracket=bracket,
        tolerance=tolerance,
        maxiter=maxiter,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "algorithm": "greit_weight_metric_search",
            "metric": metric_name,
            "noise_source": "provided"
            if measurement_noise is not None
            else "deterministic_unit_std",
            "uses_calc_greit_rm_as_black_box": True,
            "pjt_cache_source": "computed_once" if pjt_cache is None else "provided",
            "pjt_cache_reused_across_weight_search": True,
            "pjt_shape": tuple(int(v) for v in pjt.shape),
        }
    )
    return GREITWeightSearchResult(
        weight=result.weight,
        log10_weight=result.log10_weight,
        target_metric=result.target_metric,
        achieved_metric=result.achieved_metric,
        objective_value=result.objective_value,
        initial_bracket=result.initial_bracket,
        bracket=result.bracket,
        evaluations=result.evaluations,
        metadata=MappingProxyType(metadata),
    )


def optimize_greit_weight_eidors_nf(
    y: Any,
    d: Any,
    *,
    vh: Any,
    vi_nf: Any | None = None,
    signal_y: Any | None = None,
    volume_weights: Any | None = None,
    normalize: bool = False,
    target_noise_figure: float = 1.0,
    noise_covar: Any = 1.0,
    pjt_cache: Any | None = None,
    bracket: tuple[float, float] = (-2.0, 2.0),
    tolerance: float = 1.0e-4,
    maxiter: int = 96,
) -> GREITWeightSearchResult:
    """Optimize GREIT weight with EIDORS ``calc_noise_figure`` semantics.

    EIDORS fixes the desired noise figure, then searches the scalar GREIT
    ``weight`` used by ``calc_GREIT_RM``.  The metric here mirrors the linear
    ``solve_use_matrix`` branch of ``calc_noise_figure``: measurement noise is
    ``0.01 * std(vh) * I``, image/data SNR use mean absolute signal divided by
    sample standard deviation noise, and optional reconstruction-element volume
    weights are applied before measuring image SNR.
    """

    target = float(target_noise_figure)
    if target <= 0.0 or not np.isfinite(target):
        raise ValueError("target_noise_figure must be finite and positive.")
    y_matrix = _validate_training_response_matrix(y)
    d_matrix = _validate_desired_component_matrix(d, n_targets=y_matrix.shape[1])
    vh_vector = _eidors_nf_vh_vector(vh, n_measurements=y_matrix.shape[0])
    signal_matrix, signal_source = _eidors_nf_signal_matrix(
        vi_nf=vi_nf,
        signal_y=signal_y,
        vh=vh_vector,
        normalize=normalize,
    )
    volumes, volume_source = _eidors_nf_volume_weights(
        volume_weights,
        n_rec_parameters=d_matrix.shape[0],
    )
    pjt = (
        np.ascontiguousarray(
            d_matrix @ y_matrix.T,
            dtype=np.result_type(y_matrix, d_matrix),
        )
        if pjt_cache is None
        else _validate_pjt_cache(
            pjt_cache,
            n_rec_parameters=d_matrix.shape[0],
            n_measurements=y_matrix.shape[0],
            dtype=np.result_type(y_matrix, d_matrix),
        )
    )

    def metric_fn(log10_weight: float) -> float:
        components = calc_greit_rm(
            y_matrix,
            d_matrix,
            weight=10.0 ** float(log10_weight),
            noise_covar=noise_covar,
            pjt_cache=pjt,
        )
        metric, _ = _eidors_noise_figure_metric(
            components.rm,
            vh_vector,
            signal_matrix,
            volume_weights=volumes,
            normalize=normalize,
        )
        return metric

    result = search_greit_weight_for_metric(
        metric_fn,
        target_metric=target,
        bracket=bracket,
        tolerance=tolerance,
        maxiter=maxiter,
        boundary_margin=0.1,
    )
    final_components = calc_greit_rm(
        y_matrix,
        d_matrix,
        weight=result.weight,
        noise_covar=noise_covar,
        pjt_cache=pjt,
    )
    achieved, nf_metadata = _eidors_noise_figure_metric(
        final_components.rm,
        vh_vector,
        signal_matrix,
        volume_weights=volumes,
        normalize=normalize,
    )
    objective_value = float((achieved - target) ** 2)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "algorithm": "eidors_greit_noise_figure_search",
            "metric": "noise_figure",
            "eidors_reference": "mk_GREIT_model opt.noise_figure -> calc_noise_figure linear solve_use_matrix",
            "target_noise_figure": target,
            "achieved_noise_figure": achieved,
            "objective_value": objective_value,
            "uses_calc_greit_rm_as_black_box": True,
            "pjt_cache_source": "computed_once" if pjt_cache is None else "provided",
            "pjt_cache_reused_across_weight_search": True,
            "pjt_shape": tuple(int(v) for v in pjt.shape),
            "signal_source": signal_source,
            "normalize": bool(normalize),
            "volume_weight_source": volume_source,
            "final_noiselev": float(final_components.noiselev),
            **nf_metadata,
        }
    )
    return GREITWeightSearchResult(
        weight=result.weight,
        log10_weight=result.log10_weight,
        target_metric=target,
        achieved_metric=achieved,
        objective_value=objective_value,
        initial_bracket=result.initial_bracket,
        bracket=result.bracket,
        evaluations=result.evaluations,
        metadata=MappingProxyType(metadata),
    )


def greit_cache_signature_payload(
    *,
    target_distribution_grid: Any,
    desired_solution_fn: str,
    normalize: bool,
    noise_covar: Any,
    training_mode: str,
    fwd_model_signature: str,
    keep_model_components: bool,
    target_distribution_downsample: Any | None = None,
    finite_target_inputs: Any | None = None,
    desired_solution_params: Any | None = None,
    scalar_weight: Any | None = None,
    target_noise_figure: float | None = None,
    image_snr: float | None = None,
    model_components: Any | None = None,
) -> dict[str, Any]:
    """Return canonical GREIT RM cache-signature payload for V55..V61 inputs."""

    mode = str(training_mode).strip().lower()
    if not mode:
        raise ValueError("training_mode is required for GREIT cache signature.")
    fwd_signature = str(fwd_model_signature or "").strip()
    if not fwd_signature:
        raise ValueError("fwd_model_signature is required for GREIT cache signature.")
    return {
        "schema": GREIT_CACHE_SIGNATURE_SCHEMA,
        "target_distribution_grid": _canonical_signature_value(
            target_distribution_grid
        ),
        "target_distribution_downsample": _canonical_signature_value(
            target_distribution_downsample
        ),
        "finite_target_inputs": _canonical_signature_value(finite_target_inputs),
        "desired_solution_fn": str(desired_solution_fn),
        "desired_solution_params": _canonical_signature_value(desired_solution_params),
        "normalize": bool(normalize),
        "noise_covar": _canonical_signature_value(noise_covar),
        "scalar_weight": _canonical_signature_value(scalar_weight),
        "target_noise_figure": _canonical_signature_value(target_noise_figure),
        "image_snr": _canonical_signature_value(image_snr),
        "training_mode": mode,
        "fwd_model_signature": fwd_signature,
        "keep_model_components": bool(keep_model_components),
        "model_components": _canonical_signature_value(model_components),
    }


def greit_cache_signature(**kwargs) -> str:
    """Hash the canonical GREIT RM cache-signature payload."""

    return _signature_hash(greit_cache_signature_payload(**kwargs))


def build_3d_greit_rm(
    fwd_model: Any = None,
    targets: Any | None = None,
    noise_figure: float | None = 0.5,
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
    target_noise_figure: float | None = None,
    image_snr: float | None = None,
    weight_search_bracket: tuple[float, float] = (-2.0, 2.0),
    weight_search_tolerance: float = 1.0e-3,
    weight_search_maxiter: int = 64,
    weight_search_noise: Any | None = None,
    artifact_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    keep_model_components: bool = False,
) -> GREITRM:
    """Build an offline 3D GREIT RM from synthetic targets.

    The current production path still accepts a linearized forward response
    ``J`` with shape ``(n_measurements, n_inverse_parameters)``. Synthetic
    targets ``T`` are projected to measurement responses, then the shared
    ``calc_greit_rm`` parity core builds ``RM`` from EIDORS-oriented
    component matrices ``Y`` and ``D``.
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
    responses = np.asarray(target_values @ weighted_j.T, dtype=np.float64)
    response_cols = responses.T
    target_cols = target_values.T
    pjt_cache = np.ascontiguousarray(
        target_cols @ response_cols.T,
        dtype=np.result_type(response_cols, target_cols),
    )
    if regularisation is None:
        noise_covar: Any = 1.0
        rn_source = "identity"
    else:
        noise_covar, rn_source = _measurement_regularisation(
            regularisation,
            n_measurements=weighted_j.shape[0],
        )
    search_result: GREITWeightSearchResult | None = None
    metric_targets = [target is not None for target in (target_noise_figure, image_snr)]
    if sum(metric_targets) > 1:
        raise ValueError("Only one of target_noise_figure or image_snr may be set.")
    if target_noise_figure is not None or image_snr is not None:
        if noise_figure is not None:
            raise ValueError(
                "Set noise_figure=None when optimizing scalar GREIT weight."
            )
        metric_name = "noise_figure" if target_noise_figure is not None else "image_snr"
        target_metric = (
            float(target_noise_figure)
            if target_noise_figure is not None
            else float(image_snr)
        )
        search_result = optimize_greit_weight_for_metric(
            response_cols,
            target_cols,
            target_metric=target_metric,
            metric=metric_name,
            noise_covar=noise_covar,
            measurement_noise=weight_search_noise,
            pjt_cache=pjt_cache,
            bracket=weight_search_bracket,
            tolerance=weight_search_tolerance,
            maxiter=weight_search_maxiter,
        )
        nf = search_result.weight
        weight_source = "metric_search"
    else:
        if noise_figure is None:
            raise ValueError(
                "noise_figure=None requires target_noise_figure or image_snr."
            )
        nf = float(noise_figure)
        if nf < 0.0 or not np.isfinite(nf):
            raise ValueError("noise_figure must be finite and non-negative.")
        weight_source = "explicit"
    components = calc_greit_rm(
        response_cols,
        target_cols,
        weight=nf,
        noise_covar=noise_covar,
        pjt_cache=pjt_cache,
    )

    voxel_shape = _voxel_shape(inverse_mesh) or _metadata_voxel_shape(
        target_bundle.metadata
    )
    fwd_signature = _greit_forward_model_signature(fwd_model, raw_j)
    cache_payload = greit_cache_signature_payload(
        target_distribution_grid=target_bundle.metadata,
        target_distribution_downsample=target_bundle.metadata.get("downsample_factors"),
        finite_target_inputs={
            "mode": "linearized",
            "target_radius": target_radius,
            "target_amplitude": target_amplitude,
            "target_kind": target_kind,
            "target_values": target_values,
        },
        desired_solution_fn="target_values_explicit_opt_in",
        desired_solution_params={
            "D": target_cols,
            "legacy_linearized_d_approx_t": True,
        },
        normalize=True,
        noise_covar=noise_covar,
        scalar_weight=nf,
        target_noise_figure=target_noise_figure,
        image_snr=image_snr,
        training_mode="linearized",
        fwd_model_signature=fwd_signature,
        keep_model_components=keep_model_components,
        model_components={
            "Y": response_cols,
            "D": target_cols,
            "PJt": pjt_cache,
        }
        if keep_model_components
        else None,
    )
    cache_hash = _signature_hash(cache_payload)
    meta = {
        "algorithm": "greit-3d",
        "target_kind": target_bundle.metadata["kind"],
        "synthetic_target_count": int(target_values.shape[0]),
        "n_measurements": int(weighted_j.shape[0]),
        "n_parameters": int(weighted_j.shape[1]),
        "noise_figure": nf,
        "weight": components.weight,
        "weight_source": weight_source,
        "noiselev": components.noiselev,
        "regularisation_source": rn_source,
        "bad_channel_count": int(measurement_contract.bad_channel_count),
        "measurement_weight_kind": measurement_contract.weight_kind,
        "system_shape": tuple(int(v) for v in components.m.shape),
        "rm_shape": tuple(int(v) for v in components.rm.shape),
        "pjt_shape": tuple(int(v) for v in components.pjt.shape),
        "pjt_cache_source": components.metadata["pjt_source"],
        "pjt_cache_reused_across_weight_search": search_result is not None,
        "sn_shape": tuple(int(v) for v in components.sn.shape),
        "matrix_rank": components.metadata["matrix_rank"],
        "matrix_condition": components.metadata["matrix_condition"],
        "solver": components.metadata["solver"],
        "singular_fallback": components.metadata["singular_fallback"],
        "transpose_semantics": components.metadata["transpose_semantics"],
        "online_hot_path": "rm_matmul",
        "artifact_schema": "pyeidors-greit-rm-hdf5-v1",
        "artifact_format": "hdf5",
        "eidors_parity": False,
        "calc_greit_rm_parity_core": True,
        "training_mode": "linearized",
        "keep_model_components": bool(keep_model_components),
        "fwd_model_signature": fwd_signature,
        "cache_signature_schema": GREIT_CACHE_SIGNATURE_SCHEMA,
        "cache_signature_payload": cache_payload,
        "cache_signature_hash": cache_hash,
        "voxel_shape": voxel_shape,
    }
    if search_result is not None:
        meta.update(
            {
                "weight_search": dict(search_result.metadata),
                "target_noise_figure": target_noise_figure,
                "target_image_snr": image_snr,
                "achieved_metric": search_result.achieved_metric,
            }
        )
    if metadata:
        meta.update(metadata)

    result = GREITRM(
        rm=components.rm,
        metadata=MappingProxyType(meta),
        voxel_shape=voxel_shape,
        channel_mask=measurement_contract.channel_mask,
        measurement_weights=_stored_measurement_weights(measurement_weights),
        training_targets=target_values,
        training_responses=responses,
        pjt=components.pjt if keep_model_components else None,
        m=components.m if keep_model_components else None,
        sn=components.sn if keep_model_components else None,
        y=components.y if keep_model_components else None,
        d=components.d if keep_model_components else None,
        rec_model=_rec_model_array(inverse_mesh) if keep_model_components else None,
        fwd_model_signature=fwd_signature,
        cache_signature=cache_hash,
    )
    if artifact_path is not None:
        saved = result.save(artifact_path)
        meta = dict(result.metadata)
        meta["artifact_path"] = str(saved)
        result = replace(result, metadata=MappingProxyType(meta))
    return result


def build_native_greit_training_pipeline(
    fwd_model: Any,
    *,
    distribution: GREIT3DDistribution | None = None,
    rec_model: Any | None = None,
    imgsz: Any | None = None,
    xvec: Any | None = None,
    yvec: Any | None = None,
    zvec: Any | None = None,
    bounds: Any | None = None,
    downsample: Any | None = None,
    point_in_volume: Any | None = None,
    centers: Any | None = None,
    target_radius: float | None = None,
    target_size: float | None = None,
    target_plane: Any | None = None,
    target_offset: Any | None = None,
    target_contrast: Any = 1.0,
    background_conductivity: Any = 1.0,
    normalize: bool = True,
    measurement_order: Any | None = None,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    batch_size: int | None = None,
    desired_radius: Any | None = None,
    desired_solution_fn: Any | None = None,
    desired_options: dict[str, Any] | None = None,
    weight: Any = 0.5,
    noise_covar: Any = 1.0,
    artifact_path: str | Path | None = None,
    keep_model_components: bool = True,
    fwd_model_signature: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GREITNativeTrainingPipeline:
    """Build a GREIT RM from native PyEIDORS finite-target forward solves.

    This is the public orchestration entry for fair PyEIDORS/EIDORS GREIT
    comparisons: PyEIDORS generates ``V_h``, ``V_i`` and ``Y`` itself, then
    uses the same desired-image and ``calc_GREIT_RM`` algebra as the parity
    component path.
    """

    built_distribution = distribution
    if built_distribution is None and rec_model is None:
        built_distribution = build_greit3d_distribution(
            fwd_model,
            imgsz=imgsz,
            xvec=xvec,
            yvec=yvec,
            zvec=zvec,
            bounds=bounds,
            downsample=downsample,
            point_in_volume=point_in_volume,
        )
    resolved_rec_model = rec_model if rec_model is not None else built_distribution
    if resolved_rec_model is None:
        raise ValueError(
            "native GREIT training requires rec_model or a buildable distribution."
        )
    if centers is None and built_distribution is None:
        raise ValueError(
            "native GREIT training requires centers or GREIT3D distribution targets."
        )

    responses = build_greit_finite_target_responses(
        fwd_model,
        distribution=built_distribution,
        centers=centers,
        target_radius=target_radius,
        target_size=target_size,
        target_plane=target_plane,
        target_offset=target_offset,
        target_contrast=target_contrast,
        background_conductivity=background_conductivity,
        normalize=normalize,
        measurement_order=measurement_order,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
        batch_size=batch_size,
    )
    desired = build_greit_desired_images(
        resolved_rec_model,
        responses=responses,
        distribution=built_distribution,
        radius=desired_radius,
        desired_solution_fn=desired_solution_fn,
        desired_options=desired_options,
    )
    resolved_signature = (
        str(fwd_model_signature)
        if fwd_model_signature is not None
        else _greit_forward_model_signature(fwd_model, responses.conductivities)
    )
    pipeline_meta = {
        "training_data_source": "native_pyeidors_forward",
        "uses_eidors_exported_vh_vi_d": False,
        "fairness_contract": {
            "target_distribution": dict(built_distribution.metadata)
            if built_distribution is not None
            else {"target_source": "centers"},
            "forward_model_signature": resolved_signature,
            "difference_normalization": responses.metadata["difference_normalization"],
            "measurement_order_hash": responses.metadata["measurement_order_hash"],
            "bad_channel_count": responses.metadata["bad_channel_count"],
            "measurement_weight_kind": responses.metadata["measurement_weight_kind"],
            "desired_solution_fn": desired.metadata["desired_solution_fn"],
            "desired_image_sampling": desired.metadata["desired_image_sampling"],
        },
    }
    if metadata:
        pipeline_meta.update(metadata)
    greit = build_greit_rm_from_eidors_components(
        responses,
        desired,
        weight=weight,
        noise_covar=noise_covar,
        artifact_path=artifact_path,
        keep_model_components=keep_model_components,
        fwd_model_signature=resolved_signature,
        rec_model=resolved_rec_model,
        metadata=pipeline_meta,
    )
    return GREITNativeTrainingPipeline(
        distribution=built_distribution,
        responses=responses,
        desired_images=desired,
        greit=greit,
        metadata=MappingProxyType(pipeline_meta),
    )


def build_greit_rm_from_eidors_components(
    responses: GREITFiniteTargetResponses,
    desired_images: GREITDesiredImages,
    *,
    weight: Any = 0.5,
    noise_covar: Any = 1.0,
    artifact_path: str | Path | None = None,
    keep_model_components: bool = True,
    fwd_model_signature: str,
    rec_model: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> GREITRM:
    """Build GREIT RM from EIDORS-oriented ``vh/vi/Y/D`` components."""

    if not isinstance(responses, GREITFiniteTargetResponses):
        raise TypeError("responses must be GREITFiniteTargetResponses.")
    if not isinstance(desired_images, GREITDesiredImages):
        raise TypeError("desired_images must be GREITDesiredImages.")
    response_meta = dict(responses.metadata)
    desired_meta = dict(desired_images.metadata)
    y_matrix = _validate_training_response_matrix(responses.contracted_y)
    d_matrix = _validate_desired_component_matrix(
        desired_images.values,
        n_targets=y_matrix.shape[1],
    )
    pjt_cache = np.ascontiguousarray(
        d_matrix @ y_matrix.T,
        dtype=np.result_type(y_matrix, d_matrix),
    )
    components = calc_greit_rm(
        y_matrix,
        d_matrix,
        weight=weight,
        noise_covar=noise_covar,
        pjt_cache=pjt_cache,
    )
    rec_model_array = (
        _rec_model_array(rec_model)
        if rec_model is not None
        else np.asarray(desired_images.rec_centers, dtype=np.float64)
    )
    normalize = response_meta.get("difference_normalization") == "ratio"
    cache_payload = greit_cache_signature_payload(
        target_distribution_grid={
            "rec_centers": desired_images.rec_centers,
            "xyz": desired_images.xyz,
            "radii": desired_images.radii,
        },
        target_distribution_downsample=response_meta.get("downsample_factors"),
        finite_target_inputs={
            "metadata": response_meta,
            "xyzr": responses.xyzr,
            "conductivities": responses.conductivities,
        },
        desired_solution_fn=str(
            desired_meta.get("desired_solution_fn", "unknown_desired_solution_fn")
        ),
        desired_solution_params={"metadata": desired_meta},
        normalize=normalize,
        noise_covar=noise_covar,
        scalar_weight=components.weight,
        training_mode=str(response_meta.get("training_mode", "forward")),
        fwd_model_signature=fwd_model_signature,
        keep_model_components=keep_model_components,
        model_components={
            "Y": y_matrix,
            "D": d_matrix,
            "PJt": pjt_cache,
            "M": components.m,
            "vh": responses.vh,
            "vi": responses.vi,
            "xyzr": responses.xyzr,
        }
        if keep_model_components
        else None,
    )
    cache_hash = _signature_hash(cache_payload)
    meta = {
        "algorithm": "greit-3d",
        "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
        "artifact_format": "hdf5",
        "eidors_parity": bool(
            response_meta.get("eidors_parity")
            and desired_meta.get("eidors_component_parity")
        ),
        "calc_greit_rm_parity_core": True,
        "training_mode": str(response_meta.get("training_mode", "forward")),
        "difference_normalization": response_meta.get("difference_normalization"),
        "desired_solution_fn": desired_meta.get("desired_solution_fn"),
        "keep_model_components": bool(keep_model_components),
        "component_storage": "eidors_components",
        "n_measurements": int(y_matrix.shape[0]),
        "n_parameters": int(d_matrix.shape[0]),
        "synthetic_target_count": int(y_matrix.shape[1]),
        "weight": components.weight,
        "noise_figure": components.weight,
        "noiselev": components.noiselev,
        "system_shape": tuple(int(v) for v in components.m.shape),
        "rm_shape": tuple(int(v) for v in components.rm.shape),
        "pjt_shape": tuple(int(v) for v in components.pjt.shape),
        "sn_shape": tuple(int(v) for v in components.sn.shape),
        "matrix_rank": components.metadata["matrix_rank"],
        "matrix_condition": components.metadata["matrix_condition"],
        "solver": components.metadata["solver"],
        "singular_fallback": components.metadata["singular_fallback"],
        "transpose_semantics": components.metadata["transpose_semantics"],
        "pjt_cache_source": components.metadata["pjt_source"],
        "pjt_cache_reused_across_weight_search": False,
        "online_hot_path": "rm_matmul",
        "fwd_model_signature": str(fwd_model_signature),
        "cache_signature_schema": GREIT_CACHE_SIGNATURE_SCHEMA,
        "cache_signature_payload": cache_payload,
        "cache_signature_hash": cache_hash,
    }
    if metadata:
        meta.update(metadata)
    result = GREITRM(
        rm=components.rm,
        metadata=MappingProxyType(meta),
        training_responses=y_matrix.T,
        pjt=components.pjt if keep_model_components else None,
        m=components.m if keep_model_components else None,
        sn=components.sn if keep_model_components else None,
        y=components.y if keep_model_components else None,
        d=components.d if keep_model_components else None,
        vh=responses.vh if keep_model_components else None,
        vi=responses.vi if keep_model_components else None,
        xyzr=responses.xyzr if keep_model_components else None,
        rec_model=rec_model_array if keep_model_components else None,
        fwd_model_signature=str(fwd_model_signature),
        cache_signature=cache_hash,
    )
    if artifact_path is not None:
        saved = result.save(artifact_path)
        meta = dict(result.metadata)
        meta["artifact_path"] = str(saved)
        result = replace(result, metadata=MappingProxyType(meta))
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


def _resolve_finite_target_geometry(
    *,
    distribution: GREIT3DDistribution | None,
    targets: GREITTrainingTargets | None,
    centers: Any | None,
    target_radius: float | None,
    target_size: float | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    if centers is not None:
        target_centers = _as_centers(centers)
        source = "centers"
    elif distribution is not None:
        target_centers = np.asarray(distribution.centers, dtype=np.float64)
        source = "GREIT3D_distribution"
    elif targets is not None and targets.centers.size:
        target_centers = _as_centers(targets.centers)
        source = "GREITTrainingTargets.centers"
    else:
        raise ValueError(
            "finite-target GREIT responses require centers, distribution, "
            "or GREITTrainingTargets with 3D centers."
        )
    if target_centers.shape[1] != 3:
        raise ValueError("finite-target GREIT centers must be 3D.")

    n_targets = int(target_centers.shape[0])
    if target_radius is not None and target_size is not None:
        raise ValueError("Use either target_radius or target_size, not both.")
    radius_source = target_radius if target_radius is not None else target_size
    if radius_source is None and targets is not None and targets.radii.size:
        radii = np.asarray(targets.radii, dtype=np.float64).reshape(-1)
        if radii.size != n_targets:
            raise ValueError(
                f"targets radii length {radii.size} does not match {n_targets}."
            )
    elif radius_source is None:
        radii = np.full(n_targets, _default_radius(target_centers), dtype=np.float64)
    else:
        radii = np.asarray(radius_source, dtype=np.float64).reshape(-1)
        if radii.size == 1:
            radii = np.full(n_targets, float(radii[0]), dtype=np.float64)
        elif radii.size != n_targets:
            raise ValueError(
                f"target radius length {radii.size} does not match {n_targets}."
            )
    if not np.isfinite(radii).all():
        raise FloatingPointError("target radii contain non-finite values.")
    if np.any(radii <= 0.0):
        raise ValueError("target radii must be positive.")
    return (
        np.ascontiguousarray(target_centers, dtype=np.float64),
        np.ascontiguousarray(radii, dtype=np.float64),
        source,
    )


def _apply_target_plane_offset(
    centers: np.ndarray,
    *,
    target_plane: Any | None,
    target_offset: Any | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    shifted = np.asarray(centers, dtype=np.float64).copy()
    metadata: dict[str, Any] = {
        "target_plane": None,
        "target_offset": None,
    }
    if target_plane is None and target_offset is None:
        return shifted, metadata

    if target_plane is None:
        offset = _as_3d_offset(target_offset)
        shifted += offset.reshape(1, 3)
        metadata["target_offset"] = tuple(float(v) for v in offset)
        return np.ascontiguousarray(shifted, dtype=np.float64), metadata

    axis = _target_plane_axis(target_plane)
    metadata["target_plane"] = ("x", "y", "z")[axis]
    if target_offset is None:
        return np.ascontiguousarray(shifted, dtype=np.float64), metadata

    offset_values = np.asarray(target_offset, dtype=np.float64).reshape(-1)
    if offset_values.size == 1:
        if not np.isfinite(offset_values[0]):
            raise FloatingPointError("target_offset contains non-finite values.")
        shifted[:, axis] = float(offset_values[0])
        metadata["target_offset"] = float(offset_values[0])
        return np.ascontiguousarray(shifted, dtype=np.float64), metadata

    offset = _as_3d_offset(offset_values)
    shifted += offset.reshape(1, 3)
    metadata["target_offset"] = tuple(float(v) for v in offset)
    return np.ascontiguousarray(shifted, dtype=np.float64), metadata


def _target_plane_axis(value: Any) -> int:
    if isinstance(value, str):
        token = value.strip().lower()
        aliases = {
            "x": 0,
            "yz": 0,
            "xplane": 0,
            "y": 1,
            "xz": 1,
            "yplane": 1,
            "z": 2,
            "xy": 2,
            "zplane": 2,
        }
        if token in aliases:
            return aliases[token]
    axis = int(value)
    if axis not in {0, 1, 2}:
        raise ValueError("target_plane must resolve to axis 0, 1, or 2.")
    return axis


def _as_3d_offset(value: Any) -> np.ndarray:
    offset = np.asarray(value, dtype=np.float64).reshape(-1)
    if offset.size == 1:
        offset = np.full(3, float(offset[0]), dtype=np.float64)
    if offset.size != 3:
        raise ValueError("target_offset must be scalar or length-3.")
    if not np.isfinite(offset).all():
        raise FloatingPointError("target_offset contains non-finite values.")
    return np.ascontiguousarray(offset, dtype=np.float64)


def _resolve_background_conductivity(values: Any, *, n_cells: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 1:
        array = np.full(n_cells, float(array[0]), dtype=np.float64)
    if array.size != n_cells:
        raise ValueError(
            f"background_conductivity length {array.size} does not match {n_cells}."
        )
    if not np.isfinite(array).all():
        raise FloatingPointError("background_conductivity contains non-finite values.")
    if np.any(array <= 0.0):
        raise ValueError("background_conductivity entries must be positive.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_target_contrasts(values: Any, *, n_targets: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 1:
        array = np.full(n_targets, float(array[0]), dtype=np.float64)
    if array.size != n_targets:
        raise ValueError(
            f"target_contrast length {array.size} does not match {n_targets}."
        )
    if not np.isfinite(array).all():
        raise FloatingPointError("target_contrast contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _resolve_batch_size(value: int | None, *, n_targets: int) -> int:
    if value is None:
        return max(1, int(n_targets))
    batch_size = int(value)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return batch_size


def _build_finite_target_conductivities(
    fwd_centers: np.ndarray,
    *,
    background: np.ndarray,
    target_centers: np.ndarray,
    target_radii: np.ndarray,
    target_contrasts: np.ndarray,
) -> np.ndarray:
    conductivities = []
    for center, radius, contrast in zip(
        target_centers,
        target_radii,
        target_contrasts,
        strict=True,
    ):
        distance = np.linalg.norm(fwd_centers - center.reshape(1, 3), axis=1)
        mask = distance <= float(radius)
        if not np.any(mask):
            mask[int(np.argmin(distance))] = True
        sigma = background.copy()
        sigma[mask] = sigma[mask] + float(contrast)
        if np.any(sigma <= 0.0):
            raise ValueError("finite-target conductivity must stay positive.")
        conductivities.append(sigma)
    return np.ascontiguousarray(conductivities, dtype=np.float64)


def _solve_measurement_vector(fwd_model: Any, conductivity: np.ndarray) -> np.ndarray:
    data = _call_fwd_solve(fwd_model, conductivity)
    vector = _measurement_vector_from_result(data)
    if vector.size == 0:
        raise ValueError("forward solve returned an empty measurement vector.")
    return vector


def _solve_measurement_batch(
    fwd_model: Any,
    conductivities: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    batch_solver = getattr(fwd_model, "fwd_solve_batch", None)
    columns = []
    if callable(batch_solver):
        for start in range(0, conductivities.shape[0], batch_size):
            chunk = conductivities[start : start + batch_size]
            images = [_eit_image(fwd_model, sigma) for sigma in chunk]
            results = batch_solver(images)
            for result in results:
                columns.append(_measurement_vector_from_result(result))
    else:
        for sigma in conductivities:
            columns.append(_solve_measurement_vector(fwd_model, sigma))
    if not columns:
        raise ValueError("finite-target response batch produced no columns.")
    first_size = columns[0].size
    if any(column.size != first_size for column in columns):
        raise ValueError("finite-target response columns have inconsistent lengths.")
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64)


def _resolve_measurement_order(
    measurement_order: Any | None,
    *,
    n_measurements: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if measurement_order is None:
        identity = np.arange(int(n_measurements), dtype=np.int64)
        return None, {
            "measurement_order_source": "identity",
            "measurement_order_hash": _array_digest(identity),
            "measurement_order_first_indices": tuple(int(v) for v in identity[:8]),
        }
    order = np.asarray(measurement_order, dtype=np.int64).reshape(-1)
    if order.size != int(n_measurements):
        raise ValueError(
            f"measurement_order length {order.size} does not match {n_measurements}."
        )
    if np.any((order < 0) | (order >= int(n_measurements))):
        raise ValueError("measurement_order indices are out of range.")
    if np.unique(order).size != order.size:
        raise ValueError("measurement_order must be a permutation.")
    identity = np.arange(int(n_measurements), dtype=np.int64)
    if np.array_equal(order, identity):
        return None, {
            "measurement_order_source": "identity",
            "measurement_order_hash": _array_digest(identity),
            "measurement_order_first_indices": tuple(int(v) for v in identity[:8]),
        }
    return np.ascontiguousarray(order, dtype=np.int64), {
        "measurement_order_source": "provided",
        "measurement_order_hash": _array_digest(order),
        "measurement_order_first_indices": tuple(int(v) for v in order[:8]),
    }


def _call_fwd_solve(fwd_model: Any, conductivity: np.ndarray) -> Any:
    solver = getattr(fwd_model, "fwd_solve", None)
    if not callable(solver):
        raise TypeError(
            "fwd_model must provide fwd_solve(EITImage) for T42 parity mode."
        )
    return solver(_eit_image(fwd_model, conductivity))


def _eit_image(fwd_model: Any, conductivity: np.ndarray) -> EITImage:
    return EITImage(
        elem_data=np.ascontiguousarray(conductivity, dtype=np.float64),
        fwd_model=fwd_model,
    )


def _measurement_vector_from_result(result: Any) -> np.ndarray:
    data = result[0] if isinstance(result, tuple) else result
    values = getattr(data, "meas", data)
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(vector).all():
        raise FloatingPointError(
            "forward solve measurement contains non-finite values."
        )
    return np.ascontiguousarray(vector, dtype=np.float64)


def _calc_greit_difference_data(
    vh: np.ndarray,
    vi: np.ndarray,
    *,
    normalize: bool,
) -> np.ndarray:
    if normalize:
        if np.any(np.abs(vh) <= np.finfo(np.float64).eps):
            raise ValueError("normalize=True requires non-zero homogeneous vh entries.")
        y = vi / vh.reshape(-1, 1) - 1.0
    else:
        y = vi - vh.reshape(-1, 1)
    if not np.isfinite(y).all():
        raise FloatingPointError(
            "GREIT training response Y contains non-finite values."
        )
    return np.ascontiguousarray(y, dtype=np.float64)


def _contract_training_responses(
    y: np.ndarray,
    *,
    channel_mask: Any | None,
    measurement_weights: Any | None,
):
    columns = []
    contract = None
    for column in y.T:
        weighted, contract = apply_measurement_contract_to_vector(
            column,
            channel_mask=channel_mask,
            measurement_weights=measurement_weights,
        )
        columns.append(weighted)
    assert contract is not None
    return np.ascontiguousarray(np.column_stack(columns), dtype=np.float64), contract


def _forward_cell_centers(fwd_model: Any) -> np.ndarray:
    centers = None
    if isinstance(fwd_model, dict):
        centers = fwd_model.get("cell_centers")
        if centers is None:
            centers = fwd_model.get("centers")
    if centers is None:
        for name in ("cell_centers", "centers"):
            attr = getattr(fwd_model, name, None)
            if attr is not None:
                centers = attr() if callable(attr) else attr
                break
    if centers is None:
        space = getattr(fwd_model, "V_sigma", None)
        tabulate = getattr(space, "tabulate_dof_coordinates", None)
        if callable(tabulate):
            centers = tabulate()
    if centers is None:
        raise TypeError(
            "fwd_model must expose fine target centers via cell_centers, centers, "
            "or V_sigma.tabulate_dof_coordinates()."
        )
    array = np.asarray(centers, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 3:
        raise ValueError("forward target centers must have shape (n_cells, >=3).")
    if not np.isfinite(array).all():
        raise FloatingPointError("forward target centers contain non-finite values.")
    return np.ascontiguousarray(array[:, :3], dtype=np.float64)


def _desired_rec_centers(rec_model: Any) -> np.ndarray:
    if rec_model is None:
        raise ValueError("rec_model or rec_centers is required for desired images.")
    if isinstance(rec_model, (list, tuple, np.ndarray)):
        centers = np.asarray(rec_model, dtype=np.float64)
    else:
        centers = _cell_centers(rec_model)
    if centers.ndim != 2 or centers.shape[0] == 0 or centers.shape[1] < 3:
        raise ValueError("GREIT desired images require 3D rec-model centers.")
    if not np.isfinite(centers).all():
        raise FloatingPointError("rec-model centers contain non-finite values.")
    return np.ascontiguousarray(centers[:, :3], dtype=np.float64)


def _resolve_desired_xyz_radius(
    *,
    xyz: Any | None,
    radius: Any | None,
    responses: GREITFiniteTargetResponses | None,
    distribution: GREIT3DDistribution | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    embedded_radii = None
    if xyz is not None:
        xyz_matrix, embedded_radii = _as_eidors_xyz(xyz)
        source = "xyz"
    elif responses is not None:
        xyz_matrix, embedded_radii = _as_eidors_xyz(responses.xyzr)
        source = "GREITFiniteTargetResponses.xyzr"
    elif distribution is not None:
        xyz_matrix = np.asarray(distribution.centers, dtype=np.float64).T
        source = "GREIT3D_distribution"
    else:
        raise ValueError(
            "desired images require xyz, finite-target responses, or distribution."
        )
    radii = _desired_radii(
        radius if radius is not None else embedded_radii,
        n_targets=xyz_matrix.shape[1],
        xyz=xyz_matrix,
    )
    return xyz_matrix, radii, source


def _as_eidors_xyz(values: Any) -> tuple[np.ndarray, np.ndarray | None]:
    array = np.asarray(values, dtype=np.float64)
    embedded_radii = None
    if array.ndim == 1:
        if array.size not in {3, 4}:
            raise ValueError("xyz vector must have 3 or 4 entries.")
        xyz = array[:3].reshape(3, 1)
        if array.size == 4:
            embedded_radii = array[3:].reshape(1)
    elif array.ndim == 2:
        if array.shape[0] == 3:
            xyz = array
        elif array.shape[0] == 4:
            xyz = array[:3, :]
            embedded_radii = array[3, :]
        elif array.shape[1] == 3:
            xyz = array.T
        elif array.shape[1] == 4:
            xyz = array[:, :3].T
            embedded_radii = array[:, 3]
        else:
            raise ValueError("xyz must have shape 3xN, 4xN, Nx3, or Nx4.")
    else:
        raise ValueError("xyz must be a vector or 2D matrix.")
    if xyz.size == 0 or xyz.shape[0] != 3 or xyz.shape[1] == 0:
        raise ValueError("xyz must contain at least one 3D target center.")
    if not np.isfinite(xyz).all():
        raise FloatingPointError("xyz contains non-finite values.")
    if embedded_radii is not None:
        embedded_radii = np.asarray(embedded_radii, dtype=np.float64).reshape(-1)
        if not np.isfinite(embedded_radii).all():
            raise FloatingPointError("embedded xyz radii contain non-finite values.")
    return np.ascontiguousarray(xyz, dtype=np.float64), embedded_radii


def _desired_radii(
    values: Any | None,
    *,
    n_targets: int,
    xyz: np.ndarray,
) -> np.ndarray:
    if values is None:
        radii = np.full(n_targets, _default_radius(xyz.T), dtype=np.float64)
    else:
        radii = np.asarray(values, dtype=np.float64).reshape(-1)
        if radii.size == 1:
            radii = np.full(n_targets, float(radii[0]), dtype=np.float64)
        elif radii.size != n_targets:
            raise ValueError(f"radius length {radii.size} does not match {n_targets}.")
    if not np.isfinite(radii).all():
        raise FloatingPointError("desired image radius contains non-finite values.")
    if np.any(radii <= 0.0):
        raise ValueError("desired image radii must be positive.")
    return np.ascontiguousarray(radii, dtype=np.float64)


def _desired_steepness(values: Any, *, n_targets: int) -> np.ndarray:
    steepness = np.asarray(values, dtype=np.float64).reshape(-1)
    if steepness.size == 1:
        steepness = np.full(n_targets, float(steepness[0]), dtype=np.float64)
    elif steepness.size != n_targets:
        raise ValueError(
            f"desired image steepness length {steepness.size} does not match {n_targets}."
        )
    if not np.isfinite(steepness).all():
        raise FloatingPointError("desired image steepness contains non-finite values.")
    if np.any(steepness <= 0.0):
        raise ValueError("desired image steepness must be positive.")
    return np.ascontiguousarray(steepness, dtype=np.float64)


def _desired_from_target_values(
    target_values: Any,
    *,
    n_rec_parameters: int,
    n_targets: int,
) -> np.ndarray:
    if target_values is None:
        raise ValueError(
            "target_values desired mode requires explicit target_values input."
        )
    values = (
        np.asarray(target_values.values, dtype=np.float64)
        if isinstance(target_values, GREITTrainingTargets)
        else np.asarray(target_values, dtype=np.float64)
    )
    if values.shape == (n_targets, n_rec_parameters):
        return np.ascontiguousarray(values.T, dtype=np.float64)
    if values.shape == (n_rec_parameters, n_targets):
        return np.ascontiguousarray(values, dtype=np.float64)
    raise ValueError(
        "target_values must have shape "
        f"{(n_targets, n_rec_parameters)} or {(n_rec_parameters, n_targets)}."
    )


def _validate_desired_matrix(
    values: Any,
    *,
    n_rec_parameters: int,
    n_targets: int,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (n_rec_parameters, n_targets):
        raise ValueError(
            "desired image matrix D must have shape "
            f"{(n_rec_parameters, n_targets)}, got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("desired image matrix D contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64)


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


def _validate_log10_bracket(bracket: tuple[float, float]) -> tuple[float, float]:
    if len(bracket) != 2:
        raise ValueError("bracket must contain exactly two log10 bounds.")
    lo = float(bracket[0])
    hi = float(bracket[1])
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("bracket bounds must be finite and increasing.")
    return lo, hi


def _bounded_minimize(
    objective,
    *,
    lo: float,
    hi: float,
    tolerance: float,
    maxiter: int,
) -> dict[str, Any]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    x1 = hi - inv_phi * (hi - lo)
    x2 = lo + inv_phi * (hi - lo)
    f1 = float(objective(x1))
    f2 = float(objective(x2))
    iterations = 0
    for iterations in range(1, int(maxiter) + 1):
        if abs(hi - lo) <= tolerance:
            break
        if f1 > f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + inv_phi * (hi - lo)
            f2 = float(objective(x2))
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - inv_phi * (hi - lo)
            f1 = float(objective(x1))
    if f1 <= f2:
        x = x1
        fun = f1
    else:
        x = x2
        fun = f2
    return {
        "x": float(x),
        "fun": float(fun),
        "success": bool(np.isfinite(fun)),
        "message": "bounded golden-section search completed",
        "nit": int(iterations),
    }


def _inside_bracket(
    value: float,
    *,
    lo: float,
    hi: float,
    boundary_margin: float | None = None,
) -> bool:
    width = hi - lo
    if width <= 0.0:
        return True
    if boundary_margin is None:
        margin = max(0.1 * width, 1.0e-12)
    else:
        margin = max(float(boundary_margin), 1.0e-12)
    return (lo + margin) < value < (hi - margin)


def _normalize_weight_metric(metric: str) -> str:
    token = str(metric).strip().lower().replace("-", "_")
    aliases = {
        "nf": "noise_figure",
        "noisefigure": "noise_figure",
        "noise_figure": "noise_figure",
        "image_snr": "image_snr",
        "imagesnr": "image_snr",
        "snr": "image_snr",
    }
    if token not in aliases:
        raise ValueError("metric must be 'noise_figure' or 'image_snr'.")
    return aliases[token]


def _measurement_noise_matrix(values: Any | None, *, y: np.ndarray) -> np.ndarray:
    if values is not None:
        noise = np.asarray(values, dtype=np.float64)
        if noise.shape != y.shape:
            raise ValueError(
                f"measurement_noise must have shape {y.shape}, got {noise.shape}."
            )
    else:
        idx = np.arange(y.size, dtype=np.float64).reshape(y.shape)
        noise = np.sin(idx + 1.0) + 0.5 * np.cos(2.0 * idx + 0.25)
        noise = noise - float(np.mean(noise))
        std = float(np.std(noise))
        if std <= np.finfo(np.float64).eps:
            noise = np.ones_like(y, dtype=np.float64)
            noise.flat[::2] *= -1.0
            noise = noise - float(np.mean(noise))
            std = float(np.std(noise))
        scale = max(float(np.mean(np.abs(y))), 1.0e-12) * 1.0e-2
        noise = noise / max(std, 1.0e-12) * scale
    if not np.isfinite(noise).all():
        raise FloatingPointError("measurement_noise contains non-finite values.")
    if float(np.std(noise)) <= 1.0e-15:
        raise ValueError("measurement_noise must have non-zero standard deviation.")
    return np.ascontiguousarray(noise, dtype=np.float64)


def _greit_noise_metric(
    y: np.ndarray,
    d: np.ndarray,
    rm: np.ndarray,
    *,
    noise: np.ndarray,
    metric: str,
) -> float:
    clean_recon = rm @ y
    noisy_recon = rm @ (y + noise)
    recon_noise = noisy_recon - clean_recon
    input_signal = float(np.mean(np.abs(y)))
    input_noise = float(np.std(noise))
    output_signal = float(np.mean(np.abs(clean_recon)))
    if output_signal <= 1.0e-15:
        output_signal = float(np.mean(np.abs(d)))
    output_noise = float(np.std(recon_noise))
    input_snr = input_signal / max(input_noise, 1.0e-15)
    image_snr = output_signal / max(output_noise, 1.0e-15)
    if metric == "image_snr":
        return float(image_snr)
    return float(input_snr / max(image_snr, 1.0e-15))


def _eidors_nf_vh_vector(values: Any, *, n_measurements: int) -> np.ndarray:
    raw = np.asarray(values)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    vector = np.asarray(values, dtype=dtype).reshape(-1)
    if vector.size != n_measurements:
        raise ValueError(
            f"vh must contain {n_measurements} measurements, got {vector.size}."
        )
    if vector.size < 2:
        raise ValueError("EIDORS noise-figure search requires at least two channels.")
    if not np.isfinite(vector).all():
        raise FloatingPointError("vh contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=dtype)


def _eidors_nf_signal_matrix(
    *,
    vi_nf: Any | None,
    signal_y: Any | None,
    vh: np.ndarray,
    normalize: bool,
) -> tuple[np.ndarray, str]:
    if (vi_nf is None) == (signal_y is None):
        raise ValueError("Provide exactly one of vi_nf or signal_y.")
    if signal_y is not None:
        return (
            _eidors_nf_measurement_matrix(
                signal_y,
                n_measurements=vh.size,
                name="signal_y",
            ),
            "provided_signal_y",
        )
    vi_matrix = _eidors_nf_measurement_matrix(
        vi_nf,
        n_measurements=vh.size,
        name="vi_nf",
    )
    if normalize:
        _ensure_nonzero_vh_for_normalization(vh)
        signal = vi_matrix / vh.reshape(-1, 1) - 1.0
    else:
        signal = vi_matrix - vh.reshape(-1, 1)
    if not np.isfinite(signal).all():
        raise FloatingPointError("EIDORS NF signal_y contains non-finite values.")
    return np.ascontiguousarray(signal), "computed_from_vi_nf"


def _eidors_nf_measurement_matrix(
    values: Any,
    *,
    n_measurements: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    matrix = np.asarray(values, dtype=dtype)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] != n_measurements:
        raise ValueError(
            f"{name} must have shape ({n_measurements}, n_targets), got {matrix.shape}."
        )
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one target column.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=dtype)


def _eidors_nf_volume_weights(
    values: Any | None,
    *,
    n_rec_parameters: int,
) -> tuple[np.ndarray, str]:
    if values is None:
        return np.ones(n_rec_parameters, dtype=np.float64), "unit"
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != n_rec_parameters:
        raise ValueError(
            f"volume_weights must contain {n_rec_parameters} entries, "
            f"got {vector.size}."
        )
    if not np.isfinite(vector).all():
        raise FloatingPointError("volume_weights contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=np.float64), "provided"


def _ensure_nonzero_vh_for_normalization(vh: np.ndarray) -> None:
    if np.any(np.abs(vh) <= np.finfo(np.float64).eps):
        raise ValueError("normalize=True requires non-zero vh entries.")


def _eidors_noise_figure_metric(
    rm: np.ndarray,
    vh: np.ndarray,
    signal_y: np.ndarray,
    *,
    volume_weights: np.ndarray,
    normalize: bool,
) -> tuple[float, dict[str, Any]]:
    rm_matrix = np.asarray(rm)
    if rm_matrix.ndim != 2 or rm_matrix.shape[1] != vh.size:
        raise ValueError(
            f"rm must have shape (n_rec, {vh.size}), got {rm_matrix.shape}."
        )
    if rm_matrix.shape[0] < 2:
        raise ValueError("EIDORS image noise standard deviation requires >=2 pixels.")
    if volume_weights.shape != (rm_matrix.shape[0],):
        raise ValueError(
            "volume_weights shape must match RM rows: "
            f"{volume_weights.shape} vs {rm_matrix.shape[0]}."
        )
    if signal_y.shape[0] != vh.size:
        raise ValueError(f"signal_y must have {vh.size} rows, got {signal_y.shape[0]}.")
    if normalize:
        _ensure_nonzero_vh_for_normalization(vh)
    vh_std = float(np.std(vh, ddof=1))
    if vh_std <= 0.0 or not np.isfinite(vh_std):
        raise ValueError("EIDORS NF noise amplitude requires non-zero std(vh).")
    noise_amplitude = 0.01 * vh_std
    channel_noise = np.full(
        vh.size, noise_amplitude, dtype=np.result_type(vh, rm_matrix)
    )
    if normalize:
        channel_noise = channel_noise / vh

    weighted_rm = rm_matrix * volume_weights.reshape(-1, 1)
    signal_x = weighted_rm @ signal_y
    noise_x = weighted_rm * channel_noise.reshape(1, -1)

    signal_x_amp = np.mean(np.abs(signal_x), axis=0)
    noise_x_amp = float(np.mean(np.std(noise_x, axis=0, ddof=1)))
    signal_y_amp = np.mean(np.abs(signal_y), axis=0)
    noise_y_amp = float(np.mean(np.abs(channel_noise)) / np.sqrt(float(vh.size)))
    if noise_x_amp <= 0.0 or noise_y_amp <= 0.0:
        raise ValueError("EIDORS NF noise standard deviation must be non-zero.")
    snr_x = signal_x_amp / noise_x_amp
    snr_y = signal_y_amp / noise_y_amp
    nf_values = np.asarray(snr_y / snr_x, dtype=np.float64).reshape(-1)
    if not np.isfinite(nf_values).all():
        raise FloatingPointError("EIDORS NF calculation produced non-finite values.")
    nf = float(np.mean(nf_values))
    metadata = {
        "nf_formula": "eidors_calc_noise_figure_linear",
        "nf_values": [float(v) for v in nf_values],
        "nf_value_count": int(nf_values.size),
        "vh_std_ddof": 1,
        "noise_amplitude": float(noise_amplitude),
        "noise_y_std_mean": noise_y_amp,
        "noise_x_std_mean": noise_x_amp,
        "signal_y_abs_mean": float(np.mean(signal_y_amp)),
        "signal_x_abs_mean": float(np.mean(signal_x_amp)),
    }
    return nf, metadata


def _validate_training_response_matrix(values: Any) -> np.ndarray:
    raw = np.asarray(values)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    matrix = np.asarray(values, dtype=dtype)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("GREIT response matrix Y must be non-empty 2D.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("GREIT response matrix Y contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=dtype)


def _validate_desired_component_matrix(values: Any, *, n_targets: int) -> np.ndarray:
    raw = np.asarray(values)
    dtype = np.complex128 if np.iscomplexobj(raw) else np.float64
    matrix = np.asarray(values, dtype=dtype)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != n_targets:
        raise ValueError(
            "desired image matrix D must have shape "
            f"(n_rec_parameters, {n_targets}), got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("desired image matrix D contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=dtype)


def _as_scalar_weight(value: Any) -> float:
    weight = np.asarray(value, dtype=np.float64)
    if weight.size != 1:
        raise NotImplementedError(
            "calc_greit_rm currently supports scalar weight only; "
            "matrix/NF weight search belongs to T45."
        )
    scalar = float(weight.reshape(-1)[0])
    if scalar < 0.0 or not np.isfinite(scalar):
        raise ValueError("weight must be finite and non-negative.")
    return scalar


def _noise_covar_matrix(values: Any, *, n_measurements: int) -> tuple[np.ndarray, str]:
    if sparse.issparse(values):
        matrix = np.asarray(values.toarray(), dtype=np.float64)
        source = "matrix"
    else:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 1:
            scalar = float(array.reshape(-1)[0])
            if scalar < 0.0 or not np.isfinite(scalar):
                raise ValueError("noise_covar scalar must be finite and non-negative.")
            matrix = scalar * np.eye(n_measurements, dtype=np.float64)
            source = "scalar"
        else:
            matrix = np.asarray(array, dtype=np.float64)
            source = "matrix"
    if matrix.shape != (n_measurements, n_measurements):
        raise ValueError(
            "noise_covar must be scalar or have shape "
            f"{(n_measurements, n_measurements)}, got {matrix.shape}."
        )
    if not np.isfinite(matrix).all():
        raise FloatingPointError("noise_covar contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=np.float64), source


def _stored_measurement_weights(values: Any | None) -> np.ndarray | None:
    if values is None:
        return None
    if sparse.issparse(values):
        return np.asarray(values.toarray(), dtype=np.float64)
    return np.asarray(values, dtype=np.float64).copy()


def _validate_pjt_cache(
    values: Any,
    *,
    n_rec_parameters: int,
    n_measurements: int,
    dtype: Any,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=dtype)
    expected = (int(n_rec_parameters), int(n_measurements))
    if matrix.shape != expected:
        raise ValueError(f"PJt cache must have shape {expected}, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("PJt cache contains non-finite values.")
    return np.ascontiguousarray(matrix, dtype=dtype)


def _greit_artifact_schema(metadata: dict[str, Any]) -> str:
    schema = str(metadata.get("artifact_schema") or "")
    if schema in {GREIT_RM_HDF5_SCHEMA, GREIT_EIDORS_HDF5_SCHEMA}:
        return schema
    if bool(metadata.get("eidors_parity")) or bool(
        metadata.get("keep_model_components")
    ):
        return GREIT_EIDORS_HDF5_SCHEMA
    return GREIT_RM_HDF5_SCHEMA


def _greit_artifact_arrays(
    greit: GREITRM,
    *,
    schema: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    eidors_schema = schema == GREIT_EIDORS_HDF5_SCHEMA
    rm_name = "RM" if eidors_schema else "rm"
    pjt_name = "PJt" if eidors_schema else "pjt"
    m_name = "M" if eidors_schema else "m"
    sn_name = "Sn" if eidors_schema else "sn"
    y_name = "Y" if eidors_schema else "y"
    d_name = "D" if eidors_schema else "d"
    return {
        rm_name: np.asarray(greit.rm, dtype=np.float64),
        "voxel_shape": np.asarray(greit.voxel_shape or (), dtype=np.int64),
        "channel_mask": _optional_array(greit.channel_mask, dtype=bool),
        "measurement_weights": _optional_array(greit.measurement_weights),
        "training_targets": _optional_array(greit.training_targets),
        "training_responses": _optional_array(greit.training_responses),
        pjt_name: _optional_raw_array(greit.pjt),
        m_name: _optional_raw_array(greit.m),
        sn_name: _optional_raw_array(greit.sn),
        y_name: _optional_raw_array(greit.y),
        d_name: _optional_raw_array(greit.d),
        "noiselev": _optional_scalar_array(metadata.get("noiselev")),
        "weight": _optional_scalar_array(metadata.get("weight")),
        "vh": _optional_raw_array(greit.vh),
        "vi": _optional_raw_array(greit.vi),
        "xyzr": _optional_raw_array(greit.xyzr),
        "rec_model": _optional_raw_array(greit.rec_model),
        "fwd_model_signature": _optional_utf8_bytes(
            greit.fwd_model_signature or metadata.get("fwd_model_signature")
        ),
    }


def _array_from_aliases(
    arrays: dict[str, np.ndarray], *names: str
) -> np.ndarray | None:
    for name in names:
        if name in arrays:
            return arrays[name]
    return None


def _optional_raw_array(values: Any | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(values)


def _optional_scalar_array(value: Any | None) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    scalar = float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    return np.asarray([scalar], dtype=np.float64)


def _optional_utf8_bytes(value: Any | None) -> np.ndarray:
    text = "" if value is None else str(value)
    if text == "":
        return np.asarray([], dtype=np.uint8)
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def _utf8_bytes_to_string(values: Any | None) -> str | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.size == 0:
        return None
    if array.dtype.kind in {"S", "U", "O"}:
        raw = array.reshape(-1)[0]
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return bytes(np.asarray(array, dtype=np.uint8).reshape(-1).tolist()).decode("utf-8")


def _rec_model_array(model: Any | None) -> np.ndarray | None:
    if model is None:
        return None
    try:
        return np.asarray(_cell_centers(model), dtype=np.float64)
    except (TypeError, ValueError, FloatingPointError):
        array = np.asarray(model, dtype=np.float64)
        if array.size == 0:
            return None
        if not np.isfinite(array).all():
            raise FloatingPointError("rec_model contains non-finite values.")
        return np.ascontiguousarray(array, dtype=np.float64)


def _greit_forward_model_signature(fwd_model: Any | None, jacobian: Any) -> str:
    for name in (
        "fwd_model_signature",
        "model_signature",
        "signature",
        "_semantic_model_signature",
    ):
        value = getattr(fwd_model, name, None)
        if callable(value):
            value = value()
        if value:
            return str(value)
    if isinstance(fwd_model, dict):
        for key in ("fwd_model_signature", "model_signature", "signature"):
            if fwd_model.get(key):
                return str(fwd_model[key])
    return "jacobian:" + _array_digest(jacobian)


def _canonical_signature_value(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return _canonical_signature_value(dict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _canonical_signature_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_signature_value(item) for item in value]
    if sparse.issparse(value):
        coo = value.tocoo()
        return {
            "kind": "sparse",
            "shape": [int(v) for v in value.shape],
            "row": _array_digest(coo.row),
            "col": _array_digest(coo.col),
            "data": _array_digest(coo.data),
        }
    if isinstance(value, np.ndarray):
        return _array_signature(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "metadata"):
        return _canonical_signature_value(getattr(value, "metadata"))
    return repr(value)


def _array_signature(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "kind": "ndarray",
        "dtype": str(array.dtype),
        "shape": [int(v) for v in array.shape],
        "sha256": _array_digest(array),
    }


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.kind in {"U", "O"}:
        array = np.ascontiguousarray(array.astype(str))
    encoded = (
        str(array.dtype).encode("utf-8")
        + b"|"
        + json.dumps([int(v) for v in array.shape], sort_keys=True).encode("utf-8")
        + b"|"
        + array.tobytes()
    )
    return hashlib.sha256(encoded).hexdigest()


def _signature_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    "GREIT_CACHE_SIGNATURE_SCHEMA",
    "GREITDesiredImages",
    "GREIT_EIDORS_HDF5_SCHEMA",
    "GREITFiniteTargetResponses",
    "GREIT_METRIC_KEYS",
    "GREIT_RM_HDF5_SCHEMA",
    "GREITRM",
    "GREITRMComponents",
    "GREITNativeTrainingPipeline",
    "GREITTrainingTargets",
    "GREITWeightSearchResult",
    "build_3d_greit_rm",
    "build_greit_desired_images",
    "build_greit_finite_target_responses",
    "build_greit_rm_from_eidors_components",
    "build_greit3d_distribution",
    "build_native_greit_training_pipeline",
    "calc_greit_rm",
    "generate_spherical_targets",
    "greit_cache_signature",
    "greit_cache_signature_payload",
    "greit_metrics",
    "greit_desired_image_sigmoid",
    "load_greit_rm",
    "migrate_greit_rm_to_hdf5",
    "optimize_greit_weight_eidors_nf",
    "optimize_greit_weight_for_metric",
    "search_greit_weight_for_metric",
    "write_greit_metrics_artifact",
]
