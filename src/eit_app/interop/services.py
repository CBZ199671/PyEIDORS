"""High-level services for the EIDORS <-> PyEIDORS interop hub."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from eit_app.models.forward_model_config import ForwardModelConfig
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.interop import STANDARD_INTEROP_FORMAT

from .bridge_package import (
    CAPTURE_SCRIPT_NAME,
    CONFIG_NAME,
    GEOMETRY_NAME,
    LoadedBridgePackage,
    default_manifest,
    load_bridge_package,
    save_bridge_package,
)
from .environment import (
    EidorsEnvironmentDetector,
    _run_command_capture,
    matlab_command_for_execution,
    matlab_runtime_path,
    to_posix_path,
    to_windows_path,
)
from .matlab_templates import CAPTURE_SCRIPT_TEMPLATE
from .models import (
    EidorsEnvironment,
    EidorsExportJob,
    EidorsImportPreview,
    InteropCapabilityReport,
    ReconstructionPreset,
)

log = logging.getLogger(__name__)

BRIDGE_RUNTIME_NAME = "bridge_runtime.json"
BRIDGE_MANIFEST_ALIAS = "bridge_manifest.json"
CAPTURE_REQUEST_NAME = "capture_request.json"
CAPTURE_REPORT_NAME = "capture_report.json"
RUN_IMPORT_FROM_PYEIDORS_NAME = "run_import_from_pyeidors.m"

_SCRIPT_HINT_PATTERNS: dict[str, re.Pattern[str]] = {
    "mk_common_model": re.compile(r"\bmk_common_model\b", re.IGNORECASE),
    "mk_stim_patterns": re.compile(r"\bmk_stim_patterns\b", re.IGNORECASE),
    "inv_solve": re.compile(r"\binv_solve\b", re.IGNORECASE),
    "fwd_solve": re.compile(r"\bfwd_solve\b", re.IGNORECASE),
    "z_contact": re.compile(r"\bz_contact\b", re.IGNORECASE),
    "n_elec": re.compile(r"\bn_elec\b|\bnelec\b", re.IGNORECASE),
    "vh": re.compile(r"\bvh\b", re.IGNORECASE),
    "vi": re.compile(r"\bvi\b", re.IGNORECASE),
}


def _read_text_if_possible(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def detect_script_hints(path: str | Path) -> dict[str, Any]:
    """Shallow-parse a MATLAB script to infer likely EIDORS content."""

    source = Path(path)
    text = _read_text_if_possible(source)
    hints: dict[str, Any] = {}
    for key, pattern in _SCRIPT_HINT_PATTERNS.items():
        hints[key] = bool(pattern.search(text))

    if hints["inv_solve"] and (hints["vh"] or hints["vi"]):
        script_kind = "difference_inverse"
    elif hints["fwd_solve"] or hints["mk_common_model"]:
        script_kind = "forward_or_model"
    elif "show_fem" in text or "show_slices" in text or "plot(" in text:
        script_kind = "plot_only"
    else:
        script_kind = "unknown"

    stim_pattern = "{ad}" if "{ad}" in text else "{op}" if "{op}" in text else ""
    meas_pattern = stim_pattern

    hints.update(
        {
            "script_kind": script_kind,
            "stim_pattern": stim_pattern,
            "meas_pattern": meas_pattern,
            "path": str(source),
        }
    )
    return hints


def _measurement_summary(measurements: dict[str, np.ndarray] | None) -> dict[str, str]:
    if not measurements:
        return {"status": "No boundary-voltage arrays found in this package."}
    keys = sorted(measurements)
    first_key = next(
        (
            key
            for key in ("target", "ground_truth", "difference", "homogeneous")
            if key in measurements
        ),
        keys[0],
    )
    first = np.asarray(measurements[first_key]).reshape(-1)
    return {
        "arrays": ", ".join(keys),
        "points": str(first.size),
        "difference": "difference" if "difference" in measurements else "n/a",
    }


def _geometry_summary(geometry_payload: dict[str, Any] | None) -> dict[str, str]:
    if not geometry_payload:
        return {"status": "No geometry payload found."}
    nodes = np.asarray(geometry_payload.get("nodes", np.zeros((0, 2))))
    elems = np.asarray(geometry_payload.get("elems", np.zeros((0, 0))))
    n_elec_raw = geometry_payload.get("n_elec", 0)
    try:
        n_elec = int(np.asarray(n_elec_raw).reshape(-1)[0])
    except Exception:
        n_elec = 0
    dimension = "3D" if nodes.ndim == 2 and nodes.shape[1] >= 3 else "2D"
    return {
        "dimension": dimension,
        "nodes": str(int(nodes.shape[0])),
        "elements": str(int(elems.shape[0])),
        "electrodes": str(n_elec),
    }


def _normalize_contact_impedance(value: Any) -> float | list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    if array.size == 1:
        return float(array.reshape(-1)[0])
    return array.reshape(-1).tolist()


def _config_from_loaded_package(loaded: LoadedBridgePackage) -> ForwardModelConfig:
    if loaded.forward_model_config is not None:
        return loaded.forward_model_config
    geometry = loaded.geometry_payload or {}
    nodes = np.asarray(geometry.get("nodes", np.zeros((0, 2))), dtype=float)
    return ForwardModelConfig.from_mapping(
        {
            "mesh_dimension": 3 if nodes.ndim == 2 and nodes.shape[1] >= 3 else 2,
            "n_elec": int(np.asarray(geometry.get("n_elec", 16)).reshape(-1)[0]),
            "contact_impedance": _normalize_contact_impedance(
                geometry.get("contact_impedance")
            ),
        }
    )


def _build_preview(loaded: LoadedBridgePackage) -> EidorsImportPreview:
    geometry_summary = _geometry_summary(loaded.geometry_payload)
    measurement_summary = _measurement_summary(loaded.measurements)
    capability = InteropCapabilityReport(
        can_import_geometry=loaded.geometry_payload is not None,
        can_import_measurements=loaded.measurements is not None,
    )
    forward_cfg = _config_from_loaded_package(loaded)

    recognized = {
        "n_elec": forward_cfg.n_elec,
        "n_rings": forward_cfg.n_rings,
        "stim_pattern": forward_cfg.stim_pattern,
        "meas_pattern": forward_cfg.meas_pattern,
        "rotate_meas": forward_cfg.rotate_meas,
        "use_meas_current": forward_cfg.use_meas_current,
        "use_meas_current_next": forward_cfg.use_meas_current_next,
        "mesh_dimension": forward_cfg.mesh_dimension,
        "mesh_refinement": forward_cfg.mesh_refinement,
        "contact_impedance": forward_cfg.contact_impedance,
        "point_count": forward_cfg.point_count(),
    }
    inferred: dict[str, Any] = {}
    if "config" not in loaded.manifest.files:
        inferred["source"] = (
            "No config.json found; values were inferred from geometry and defaults."
        )

    missing: list[str] = []
    if loaded.geometry_payload is None:
        missing.append("geometry")
    if loaded.measurements is None:
        missing.append("measurements")

    warnings = list(loaded.manifest.notes)
    if loaded.measurements:
        try:
            measurements = loaded.measurements
            if "homogeneous" in measurements:
                MeasurementDataset.from_metadata(
                    np.asarray(measurements["homogeneous"]).reshape(1, -1),
                    forward_cfg.to_mapping(),
                    data_type="real",
                )
            if "target" in measurements:
                MeasurementDataset.from_metadata(
                    np.asarray(measurements["target"]).reshape(1, -1),
                    forward_cfg.to_mapping(),
                    data_type="real",
                )
            capability.can_import_measurements = True
        except Exception as exc:
            capability.can_import_measurements = False
            warnings.append(f"边界电压数据已找到，但当前无法直接按配置解释：{exc}")

    return EidorsImportPreview(
        forward_model_config=forward_cfg,
        capability_report=capability,
        geometry_summary=geometry_summary,
        measurement_summary=measurement_summary,
        recognized_fields=recognized,
        inferred_fields=inferred,
        missing_fields=missing,
        warnings=warnings,
    )


def _boundary_entities_from_cells(cell_connectivity: np.ndarray) -> np.ndarray:
    cells = np.asarray(cell_connectivity, dtype=int)
    if cells.ndim != 2 or cells.size == 0:
        return np.empty((0, 0), dtype=np.int64)

    verts_per_cell = cells.shape[1]
    if verts_per_cell == 3:
        local_facets = ((0, 1), (1, 2), (0, 2))
    elif verts_per_cell == 4:
        local_facets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    else:
        raise ValueError("当前导出器只支持三角形 2D 网格或四面体 3D 网格。")

    counts: dict[tuple[int, ...], tuple[int, ...]] = {}
    repeats: dict[tuple[int, ...], int] = {}
    for cell in cells:
        for facet in local_facets:
            original = tuple(int(cell[index]) for index in facet)
            key = tuple(sorted(original))
            counts[key] = original
            repeats[key] = repeats.get(key, 0) + 1

    boundary = [counts[key] for key, count in repeats.items() if count == 1]
    if not boundary:
        return np.empty((0, len(local_facets[0])), dtype=np.int64)
    return np.asarray(boundary, dtype=np.int64) + 1


def _infer_electrode_node_groups(
    node_coords: np.ndarray, boundary_entities: np.ndarray, n_elec: int
) -> tuple[np.ndarray, np.ndarray]:
    if node_coords.shape[1] < 2:
        raise ValueError("至少需要二维节点坐标才能推断电极节点。")

    boundary_nodes = np.unique(np.asarray(boundary_entities, dtype=int).reshape(-1)) - 1
    if boundary_nodes.size == 0:
        raise ValueError("无法从几何中推断边界节点。")

    coords = node_coords[boundary_nodes, :2]
    center = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    order = np.argsort(angles)
    ordered_nodes = boundary_nodes[order]
    chunks = np.array_split(ordered_nodes, max(int(n_elec), 1))
    max_len = max(len(chunk) for chunk in chunks)
    padded = np.zeros((len(chunks), max_len), dtype=np.int64)
    counts = np.zeros(len(chunks), dtype=np.int64)
    for index, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        padded[index, : len(chunk)] = chunk + 1
        counts[index] = len(chunk)
    return padded, counts


def build_geometry_payload_from_result(
    *,
    node_coords: np.ndarray,
    cell_connectivity: np.ndarray,
    forward_model_config: ForwardModelConfig,
    truth_elem_data: np.ndarray | None = None,
    background: float | None = None,
    source_framework: str = "pyeidors",
    mesh_name: str = "pyeidors_export",
    scenario_name: str = "bridge_export",
) -> dict[str, Any]:
    """Build a standard-ish geometry payload from a simulation result."""

    nodes = np.asarray(node_coords, dtype=float)
    elems = np.asarray(cell_connectivity, dtype=np.int64) + 1
    boundary_entities = _boundary_entities_from_cells(cell_connectivity)
    electrode_nodes, electrode_counts = _infer_electrode_node_groups(
        nodes,
        boundary_entities,
        forward_model_config.n_elec * max(forward_model_config.n_rings, 1),
    )
    if truth_elem_data is None:
        truth_elem = np.full(
            elems.shape[0],
            float(background or forward_model_config.background_conductivity),
        )
    else:
        truth_elem = np.asarray(truth_elem_data, dtype=float).reshape(-1)
    return {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "source_framework": source_framework,
        "nodes": nodes,
        "elems": elems,
        "boundary_edges": boundary_entities,
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "n_elec": int(
            forward_model_config.n_elec * max(forward_model_config.n_rings, 1)
        ),
        "background": float(background or forward_model_config.background_conductivity),
        "truth_elem_data": truth_elem,
        "contact_impedance": (
            float(forward_model_config.contact_impedance)
            if isinstance(forward_model_config.contact_impedance, (int, float))
            else np.asarray(
                forward_model_config.contact_impedance or [0.01], dtype=float
            )
        ),
        "mesh_name": mesh_name,
        "mesh_level": "bridge_export",
        "scenario_name": scenario_name,
    }


class EidorsBridgeRunner:
    """Run a controlled MATLAB capture to produce a Bridge Package v2."""

    def run_capture(
        self,
        environment: EidorsEnvironment,
        script_path: str | Path,
        output_dir: str | Path,
    ) -> Path:
        if not environment.matlab_command:
            raise RuntimeError("尚未配置 MATLAB 可执行路径。")
        if not environment.eidors_startup:
            raise RuntimeError("尚未配置 EIDORS startup.m。")

        source = Path(script_path)
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        hints = detect_script_hints(source)
        (root / CAPTURE_SCRIPT_NAME).write_text(
            CAPTURE_SCRIPT_TEMPLATE, encoding="utf-8"
        )
        request_payload = {
            "eidors_startup": matlab_runtime_path(
                environment.eidors_startup, environment
            ),
            "target_script": matlab_runtime_path(source, environment),
            "output_dir": matlab_runtime_path(root, environment),
            "script_kind": hints["script_kind"],
        }
        request_path = root / CAPTURE_REQUEST_NAME
        request_path.write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        runtime_root = matlab_runtime_path(root, environment)
        runtime_request = matlab_runtime_path(request_path, environment)
        escaped_root = runtime_root.replace("'", "''")
        escaped_request = runtime_request.replace("'", "''")
        expression = (
            f"addpath('{escaped_root}'); run_capture_from_eidors('{escaped_request}');"
        )
        returncode, stdout, stderr = _run_command_capture(
            [matlab_command_for_execution(environment), "-batch", expression],
            timeout=180,
        )
        if returncode != 0:
            raise RuntimeError(
                stderr.strip() or stdout.strip() or "MATLAB bridge capture failed."
            )

        loaded = load_bridge_package(root)
        geometry_payload = loaded.geometry_payload
        measurements = loaded.measurements
        if geometry_payload is None:
            geometry_path = root / GEOMETRY_NAME
            if geometry_path.exists():
                geometry_payload = loadmat(
                    geometry_path, squeeze_me=True, struct_as_record=False
                )
        existing_cfg = loaded.forward_model_config or _config_from_loaded_package(
            loaded
        )
        forward_cfg = existing_cfg.with_overrides(
            stim_pattern=hints.get("stim_pattern") or existing_cfg.stim_pattern,
            meas_pattern=hints.get("meas_pattern") or existing_cfg.meas_pattern,
        )

        notes: list[str] = []
        report_path = root / CAPTURE_REPORT_NAME
        if report_path.exists():
            notes.append(
                "已生成 capture_report.json，可在 Diagnostics 中查看原始采集结果。"
            )
        manifest = default_manifest(
            source_framework="eidors",
            package_kind="captured_script",
            environment=environment.to_mapping(),
            capabilities={
                "can_import_geometry": geometry_payload is not None,
                "can_import_measurements": measurements is not None,
                "can_capture_script": True,
            },
            script_path=str(source),
            script_kind=str(hints.get("script_kind", "unknown")),
            script_hints=hints,
            notes=notes,
        )
        save_bridge_package(
            root,
            manifest,
            geometry_payload=geometry_payload,
            measurements=measurements,
            forward_model_config=forward_cfg,
            include_capture_script=True,
        )
        return root


class EidorsScriptCaptureService:
    """Facade that detects environments and captures user EIDORS scripts."""

    def __init__(
        self,
        detector: EidorsEnvironmentDetector | None = None,
        runner: EidorsBridgeRunner | None = None,
    ) -> None:
        self._detector = detector or EidorsEnvironmentDetector()
        self._runner = runner or EidorsBridgeRunner()

    def detect_environments(
        self,
    ) -> tuple[list[EidorsEnvironment], InteropCapabilityReport]:
        return self._detector.detect()

    def load_profiles(self) -> list[EidorsEnvironment]:
        return self._detector.load_profiles()

    def save_profiles(self, profiles: list[EidorsEnvironment]) -> None:
        self._detector.save_profiles(profiles)

    def save_last_environment(self, environment: EidorsEnvironment) -> None:
        self._detector.save_last_environment(environment)

    def test_matlab(self, environment: EidorsEnvironment) -> tuple[bool, str]:
        return self._detector.test_matlab_launch(environment)

    def test_startup(self, environment: EidorsEnvironment) -> tuple[bool, str]:
        return self._detector.test_eidors_startup(environment)

    def infer_startup_from_source(self, source_path: str | Path) -> str:
        return self._detector.infer_startup_from_source(source_path)

    def capture_or_load(
        self,
        source_path: str | Path,
        environment: EidorsEnvironment | None = None,
        output_dir: str | Path | None = None,
    ) -> LoadedBridgePackage:
        source = Path(to_posix_path(source_path))
        if source.is_dir() and (
            (source / "manifest.json").exists()
            or (source / GEOMETRY_NAME).exists()
            or (source / CONFIG_NAME).exists()
        ):
            return load_bridge_package(source)
        if source.is_file() and source.suffix.lower() in {".mat", ".json"}:
            return load_bridge_package(source)
        if source.is_file() and source.suffix.lower() == ".m":
            if environment is None:
                raise RuntimeError(
                    "采集 EIDORS 脚本前需要先选择一个 MATLAB/EIDORS 环境。"
                )
            target_dir = (
                Path(to_posix_path(output_dir))
                if output_dir
                else source.parent / f"{source.stem}_bridge"
            )
            bundle_dir = self._runner.run_capture(environment, source, target_dir)
            return load_bridge_package(bundle_dir)
        raise RuntimeError(
            "当前仅支持导入 bridge 目录、legacy .mat 文件或 EIDORS .m 脚本。"
        )


class InteropBundleImporter:
    """Build previews and normalized assets from a bridge package."""

    def load_package(self, path: str | Path) -> LoadedBridgePackage:
        return load_bridge_package(path)

    def preview_package(
        self, path: str | Path
    ) -> tuple[LoadedBridgePackage, EidorsImportPreview]:
        loaded = load_bridge_package(path)
        return loaded, _build_preview(loaded)

    def preview_loaded_package(
        self, loaded: LoadedBridgePackage
    ) -> EidorsImportPreview:
        return _build_preview(loaded)


class InteropBundleExporter:
    """Export PyEIDORS runtime state to a Bridge Package v2 directory."""

    def export_bundle(
        self,
        job: EidorsExportJob,
        *,
        forward_model_config: ForwardModelConfig,
        environment: EidorsEnvironment | None = None,
        geometry_payload: dict[str, Any] | None = None,
        measurements: dict[str, np.ndarray] | None = None,
        reconstruction_preset: ReconstructionPreset | None = None,
        notes: list[str] | None = None,
    ) -> Path:
        root = Path(to_posix_path(job.output_dir))
        root.mkdir(parents=True, exist_ok=True)

        effective_geometry = geometry_payload if job.include_geometry else None
        effective_measurements = measurements if job.include_measurements else None
        effective_notes = list(notes or [])
        if job.include_geometry and effective_geometry is None:
            effective_notes.append(
                "本次导出未包含 geometry.mat；当前来源尚未提供可复用的几何载荷。"
            )
        if job.include_measurements and effective_measurements is None:
            effective_notes.append(
                "本次导出未包含边界电压数据；当前来源没有可导出的测量数组。"
            )

        manifest = default_manifest(
            source_framework="pyeidors",
            package_kind="export_project",
            environment=environment.to_mapping() if environment else {},
            capabilities={
                "can_import_geometry": effective_geometry is not None,
                "can_import_measurements": effective_measurements is not None,
            },
            notes=effective_notes,
        )
        save_bridge_package(
            root,
            manifest,
            geometry_payload=effective_geometry,
            measurements=effective_measurements,
            forward_model_config=forward_model_config,
            reconstruction_preset=reconstruction_preset,
            include_run_in_eidors_script=job.include_scripts
            and effective_geometry is not None,
        )

        runtime_payload = {
            "eidors_startup": to_windows_path(environment.eidors_startup)
            if environment
            else "",
            "geometry_mat": to_windows_path((root / GEOMETRY_NAME).resolve()),
            "measurements_csv": to_windows_path((root / "measurements.csv").resolve()),
            "stim_pattern": forward_model_config.stim_pattern,
            "meas_pattern": forward_model_config.meas_pattern,
            "rotate_meas": forward_model_config.rotate_meas,
            "use_meas_current": forward_model_config.use_meas_current,
            "drive_value": forward_model_config.drive_value,
        }
        (root / BRIDGE_RUNTIME_NAME).write_text(
            json.dumps(runtime_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            (root / BRIDGE_MANIFEST_ALIAS).write_text(
                manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        run_script = root / "run_in_eidors.m"
        if run_script.exists():
            (root / RUN_IMPORT_FROM_PYEIDORS_NAME).write_text(
                run_script.read_text(encoding="utf-8"), encoding="utf-8"
            )
        return root


class InteropSmokeValidator:
    """Run a lightweight import smoke check, optionally including inverse solve."""

    def validate(
        self,
        loaded: LoadedBridgePackage,
        *,
        reconstruction_preset: ReconstructionPreset | None = None,
    ) -> dict[str, Any]:
        config = _config_from_loaded_package(loaded)
        measurements = loaded.measurements or {}
        homogeneous = measurements.get("homogeneous")
        target = measurements.get("target")
        difference = measurements.get("difference")

        if target is None and homogeneous is not None and difference is not None:
            target = np.asarray(homogeneous, dtype=float).reshape(-1) + np.asarray(
                difference, dtype=float
            ).reshape(-1)
        if homogeneous is None and target is not None and difference is not None:
            homogeneous = np.asarray(target, dtype=float).reshape(-1) - np.asarray(
                difference, dtype=float
            ).reshape(-1)
        if homogeneous is None or target is None:
            raise RuntimeError("冒烟验证需要 homogeneous 与 target 两组边界电压。")

        homogeneous = np.asarray(homogeneous, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        metadata = {
            **config.to_mapping(),
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
        }

        MeasurementDataset.from_metadata(
            homogeneous.reshape(1, -1), metadata, data_type="real"
        )
        MeasurementDataset.from_metadata(
            target.reshape(1, -1), metadata, data_type="real"
        )

        result: dict[str, Any] = {
            "status": "compatible",
            "n_measurements": int(target.size),
            "mesh_dimension": int(config.mesh_dimension),
            "message": (
                f"数据兼容性检查通过：{int(target.size)} 个边界电压点可按当前布局解释。"
            ),
        }

        if int(config.mesh_dimension) == 3:
            result["message"] += (
                " 当前为 3D 配置，自动烟测默认跳过 full inverse，只完成输入兼容性验证。"
            )
            return result

        preset = (
            reconstruction_preset
            or loaded.reconstruction_preset
            or ReconstructionPreset()
        )
        from pyeidors import EITSystem
        from pyeidors.data import PatternConfig

        pattern = PatternConfig(
            n_elec=config.n_elec,
            n_rings=config.n_rings,
            stim_pattern=config.stim_pattern,
            meas_pattern=config.meas_pattern,
            drive_mode=config.drive_mode,
            drive_value=config.drive_value,
            geometry_scale_to_m=config.geometry_scale_to_m,
            electrode_length_m_override=config.electrode_length_m_override,
            use_meas_current=config.use_meas_current,
            use_meas_current_next=config.use_meas_current_next,
            rotate_meas=config.rotate_meas,
            stim_direction=config.stim_direction,
            meas_direction=config.meas_direction,
            stim_first_positive=config.stim_first_positive,
        )
        system = EITSystem(
            n_elec=config.n_elec,
            pattern_config=pattern,
            regularization_alpha=float(preset.regularization_alpha),
            difference_mode=str(preset.difference_mode),
            difference_orientation=str(preset.difference_orientation),
            potential_order=config.potential_order,
        )
        system.setup(
            mesh_source="generated",
            dimension=config.mesh_dimension,
            mesh_size=config.mesh_refinement,
            radius=config.radius,
            height=config.height,
            electrode_height_ratio=config.electrode_height_ratio,
            electrode_level_fractions=config.electrode_level_fractions,
            z_center=config.z_center,
            mesh_family=config.mesh_family,
            geometry_version=config.geometry_version,
        )

        ref_data = MeasurementDataset.from_metadata(
            homogeneous.reshape(1, -1), metadata, data_type="real"
        ).to_eit_data(0)
        tgt_data = MeasurementDataset.from_metadata(
            target.reshape(1, -1), metadata, data_type="real"
        ).to_eit_data(0)

        method = str(preset.method or "").strip().lower()
        if method in {"gn-absolute", "eidors_abs_gn"}:
            recon = system.absolute_reconstruct(measurement_data=tgt_data)
        else:
            recon = system.difference_reconstruct(
                measurement_data=tgt_data, reference_data=ref_data
            )

        conductivity = np.asarray(
            getattr(recon, "conductivity", np.asarray([])), dtype=float
        ).reshape(-1)
        result.update(
            {
                "status": "inverse_ok",
                "n_elements": int(conductivity.size),
                "message": (
                    f"逆问题烟测通过：{int(target.size)} 点边界电压已成功跑通 "
                    f"{preset.method}，得到 {int(conductivity.size)} 个单元的重构结果。"
                ),
            }
        )
        return result
