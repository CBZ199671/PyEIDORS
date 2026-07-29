#!/usr/bin/env python3
"""Run the real MATLAB/EIDORS source-semantics acceptance matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.io import loadmat

from eit_app.interop import (
    EidorsExportJob,
    EidorsBridgeRunner,
    EidorsEnvironment,
    InteropBundleImporter,
    InteropBundleExporter,
    build_geometry_payload_from_result,
    validate_bridge_package,
)
from eit_app.interop.environment import (
    _run_command_capture,
    matlab_command_for_execution,
    matlab_runtime_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matlab", required=True, help="MATLAB executable")
    parser.add_argument("--eidors-startup", required=True, help="EIDORS startup.m")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory for packages and the JSON report",
    )
    return parser.parse_args()


def _public_mat(path: Path) -> dict[str, Any]:
    return {
        key: value
        for key, value in loadmat(
            path,
            squeeze_me=True,
            struct_as_record=False,
        ).items()
        if not key.startswith("__")
    }


def _bool_scalar(value: Any) -> bool:
    return bool(np.asarray(value).reshape(-1)[0])


def _float_scalar(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def _run_import(
    repo_root: Path,
    package: Path,
) -> tuple[int, dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "pyeidors.interop",
        "import-geometry",
        str(package),
        "--forward-smoke",
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout if result.returncode == 0 else result.stderr
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Interop CLI did not return JSON for {package}: {output.strip()}"
        ) from exc
    return result.returncode, payload


def _capture(
    runner: EidorsBridgeRunner,
    environment: EidorsEnvironment,
    script: Path,
    output: Path,
    *,
    selectors: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runner.run_capture(environment, script, output, selectors=selectors)
    report = validate_bridge_package(output)
    geometry = _public_mat(output / "geometry.mat")
    return report, geometry


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(actual) - np.asarray(reference)))
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).eps)
    return numerator / denominator


def _native_pem_parity(package: Path) -> dict[str, Any]:
    from pyeidors import EITSystem
    from pyeidors.data import PatternConfig
    from pyeidors.interop import build_mesh_from_exchange_mat

    loaded, preview = InteropBundleImporter().preview_package(package)
    config = preview.forward_model_config
    measurements = loaded.measurements or {}
    config.require_interop_forward_ready()
    if config.electrode_model != "pem":
        raise RuntimeError(
            f"Expected native PEM config, got {config.electrode_model!r}"
        )

    pattern = PatternConfig(
        n_elec=config.n_elec,
        n_rings=config.n_rings,
        stim_pattern=config.stim_pattern,
        meas_pattern=config.meas_pattern,
        electrode_layout=config.electrode_layout,
        measurement_protocol=config.measurement_protocol,
        custom_stim_matrix=config.custom_stim_matrix,
        custom_meas_matrices=config.custom_meas_matrices,
        drive_mode=config.drive_mode,
        drive_value=config.drive_value,
        geometry_scale_to_m=config.geometry_scale_to_m,
        use_meas_current=config.use_meas_current,
        use_meas_current_next=config.use_meas_current_next,
        rotate_meas=config.rotate_meas,
        stim_direction=config.stim_direction,
        meas_direction=config.meas_direction,
        stim_first_positive=config.stim_first_positive,
    )
    mesh, local_geometry = build_mesh_from_exchange_mat(package / "geometry.mat")
    background = np.asarray(local_geometry["background_elem_data"]).reshape(-1)
    target = np.asarray(local_geometry["target_elem_data"]).reshape(-1)
    eidors_background = np.asarray(measurements["homogeneous"]).reshape(-1)
    eidors_target = np.asarray(measurements["target"]).reshape(-1)

    pyeidors_measurements: list[tuple[np.ndarray, np.ndarray]] = []
    for contact_impedance in (config.contact_impedance, np.full(config.n_elec, 1e9)):
        system = EITSystem(
            n_elec=config.total_electrodes(),
            pattern_config=pattern,
            electrode_model="pem",
            contact_impedance=contact_impedance,
            base_conductivity=config.background_conductivity,
            linear_backend="scipy",
            forward_backend="dolfinx",
            petsc_device="cpu",
            potential_order=1,
        )
        system.setup(mesh=mesh, initialize_inverse=False)
        pyeidors_measurements.append(
            (
                np.asarray(system.forward_solve(background).meas).reshape(-1),
                np.asarray(system.forward_solve(target).meas).reshape(-1),
            )
        )

    py_background, py_target = pyeidors_measurements[0]
    alternate_background, alternate_target = pyeidors_measurements[1]
    return {
        "background_relative_l2": _relative_l2(
            py_background,
            eidors_background,
        ),
        "target_relative_l2": _relative_l2(py_target, eidors_target),
        "z_contact_background_max_abs": float(
            np.max(np.abs(py_background - alternate_background), initial=0.0)
        ),
        "z_contact_target_max_abs": float(
            np.max(np.abs(py_target - alternate_target), initial=0.0)
        ),
        "n_measurements": int(py_background.size),
        "electrode_model": config.electrode_model,
        "electrode_projection": config.interop_semantics["electrode_projection"],
    }


def _build_pem_roundtrip_export(
    captured_package: Path,
    output: Path,
    environment: EidorsEnvironment,
) -> Path:
    loaded, preview = InteropBundleImporter().preview_package(captured_package)
    geometry = loaded.geometry_payload or {}
    config = preview.forward_model_config.with_overrides(contact_impedance=None)
    export_geometry = build_geometry_payload_from_result(
        node_coords=np.asarray(geometry["nodes"]),
        cell_connectivity=np.asarray(geometry["elems"], dtype=np.int64) - 1,
        forward_model_config=config,
        truth_elem_data=np.asarray(geometry["target_elem_data"]).reshape(-1),
        background=float(np.asarray(geometry["background"]).reshape(-1)[0]),
        boundary_facets=np.asarray(geometry["boundary_facets"], dtype=np.int64),
        electrode_nodes=np.asarray(geometry["electrode_nodes"], dtype=np.int64),
        electrode_node_counts=np.asarray(
            geometry["electrode_node_counts"],
            dtype=np.int64,
        ),
        source_framework="pyeidors",
        mesh_name="pyeidors_native_pem_roundtrip",
        scenario_name="native_pem_roundtrip",
    )
    return InteropBundleExporter().export_bundle(
        EidorsExportJob(
            source_kind="simulation",
            source_name="native PEM roundtrip",
            output_dir=str(output),
        ),
        forward_model_config=config,
        environment=environment,
        geometry_payload=export_geometry,
        measurements=loaded.measurements,
    )


def _validate_roundtrip_in_eidors(
    *,
    environment: EidorsEnvironment,
    repo_root: Path,
    package: Path,
) -> dict[str, Any]:
    examples_dir = matlab_runtime_path(repo_root / "examples" / "interop", environment)
    run_script = matlab_runtime_path(package / "run_in_eidors.m", environment)
    report_path = package / "eidors_import_report.json"
    runtime_report = matlab_runtime_path(report_path, environment)
    escaped_examples_dir = examples_dir.replace("'", "''")
    escaped_run_script = run_script.replace("'", "''")
    escaped_runtime_report = runtime_report.replace("'", "''")
    expression = (
        f"addpath('{escaped_examples_dir}');"
        "validate_bridge_in_eidors("
        f"'{escaped_run_script}',"
        f"'{escaped_runtime_report}');"
    )
    returncode, stdout, stderr = _run_command_capture(
        [matlab_command_for_execution(environment), "-batch", expression],
        timeout=180,
    )
    if returncode != 0:
        raise RuntimeError(
            stderr.strip()
            or stdout.strip()
            or "MATLAB/EIDORS PEM roundtrip validation failed."
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output.resolve()
    case_paths = {
        "cem_2d": output_root / "cem_2d",
        "cem_surface": output_root / "cem_surface",
        "pem_current_density": output_root / "pem_current_density",
        "pem_roundtrip_export": output_root / "pem_roundtrip_export",
        "missing_fields": output_root / "missing_fields",
        "ambiguous_auto": output_root / "ambiguous_auto",
        "explicit_selector": output_root / "explicit_selector",
    }
    occupied = [path for path in case_paths.values() if path.exists()]
    report_path = output_root / "eidors_source_semantics_acceptance.json"
    if occupied or report_path.exists():
        names = ", ".join(
            str(path) for path in [*occupied, report_path] if path.exists()
        )
        raise FileExistsError(
            "Acceptance output must be new; refusing to overwrite: " + names
        )
    output_root.mkdir(parents=True, exist_ok=True)

    environment = EidorsEnvironment(
        name="real MATLAB/EIDORS source-semantics acceptance",
        matlab_command=args.matlab,
        eidors_startup=args.eidors_startup,
    )
    runner = EidorsBridgeRunner()
    scripts = {
        "cem_2d": repo_root / "examples/interop/eidors_2d_quickstart.m",
        "cem_surface": repo_root / "examples/interop/eidors_3d_quickstart.m",
        "pem_current_density": (
            repo_root / "examples/interop/eidors_3d_point_electrode_quickstart.m"
        ),
        "missing_fields": (
            repo_root / "examples/interop/eidors_missing_fields_semantics.m"
        ),
        "multiple_models": (
            repo_root / "examples/interop/eidors_multiple_models_requires_selector.m"
        ),
    }
    reports: dict[str, dict[str, Any]] = {}
    geometries: dict[str, dict[str, Any]] = {}
    for name in ("cem_2d", "cem_surface", "pem_current_density", "missing_fields"):
        reports[name], geometries[name] = _capture(
            runner,
            environment,
            scripts[name],
            case_paths[name],
        )
    try:
        runner.run_capture(
            environment,
            scripts["multiple_models"],
            case_paths["ambiguous_auto"],
        )
    except RuntimeError as exc:
        ambiguity_error = str(exc)
    else:
        ambiguity_error = ""
    reports["explicit_selector"], geometries["explicit_selector"] = _capture(
        runner,
        environment,
        scripts["multiple_models"],
        case_paths["explicit_selector"],
        selectors={"fwd_model_var": "fmdl_b"},
    )
    pem_roundtrip_path = _build_pem_roundtrip_export(
        case_paths["pem_current_density"],
        case_paths["pem_roundtrip_export"],
        environment,
    )
    pem_eidors_roundtrip = _validate_roundtrip_in_eidors(
        environment=environment,
        repo_root=repo_root,
        package=pem_roundtrip_path,
    )

    cem_2d_code, cem_2d_import = _run_import(
        repo_root,
        case_paths["cem_2d"],
    )
    cem_code, cem_import = _run_import(
        repo_root,
        case_paths["cem_surface"],
    )
    pem_code, pem_import = _run_import(
        repo_root,
        case_paths["pem_current_density"],
    )

    pem = geometries["pem_current_density"]
    pem_parity = _native_pem_parity(case_paths["pem_current_density"])
    missing = geometries["missing_fields"]
    pem_raw = np.asarray(pem["stim_matrix_raw"], dtype=float)
    pem_effective = np.asarray(pem["stim_matrix"], dtype=float)
    pem_target = np.asarray(pem["target_elem_data"], dtype=float).reshape(-1)
    missing_contact = np.asarray(missing["contact_impedance"]).reshape(-1)
    missing_contact_present = np.asarray(
        missing["contact_impedance_present"],
        dtype=bool,
    ).reshape(-1)

    checks = {
        "cem_2d_geometry_valid": reports["cem_2d"]["valid"] is True,
        "cem_2d_forward_ready": reports["cem_2d"]["forward_ready"] is True,
        "cem_2d_models_preserved": set(reports["cem_2d"]["electrode_models"])
        <= {"cem", "cem_faces"},
        "cem_2d_forward_smoke": (
            cem_2d_code == 0
            and cem_2d_import.get("dimension") == 2
            and cem_2d_import.get("mesh_family") == "triangle"
            and cem_2d_import.get("forward_smoke") == "passed"
            and cem_2d_import.get("electrode_projection") == "exact_surface_nodes"
        ),
        "cem_geometry_valid": reports["cem_surface"]["valid"] is True,
        "cem_forward_ready": reports["cem_surface"]["forward_ready"] is True,
        "cem_models_preserved": set(reports["cem_surface"]["electrode_models"])
        <= {"cem", "cem_faces"},
        "cem_contact_impedance_present": all(
            reports["cem_surface"]["contact_impedance_present"]
        ),
        "cem_forward_smoke": (
            cem_code == 0
            and cem_import.get("forward_smoke") == "passed"
            and cem_import.get("electrode_projection") == "exact_surface_nodes"
        ),
        "pem_geometry_valid": reports["pem_current_density"]["valid"] is True,
        "pem_models_preserved": set(reports["pem_current_density"]["electrode_models"])
        == {"point"},
        "pem_current_density_recorded": (
            _bool_scalar(pem["current_density_present"])
            and _bool_scalar(pem["current_density_applied"])
            and np.isclose(_float_scalar(pem["current_density"]), 2.0)
        ),
        "pem_raw_current_is_0_02_a": np.allclose(
            np.max(np.abs(pem_raw), axis=1),
            0.02,
        ),
        "pem_effective_current_is_0_01_a": np.allclose(
            np.max(np.abs(pem_effective), axis=1),
            0.01,
        ),
        "pem_resistivity_mapped_to_conductivity": (
            np.isclose(_float_scalar(pem["background"]), 2.0)
            and np.isclose(pem_target[0], 4.0)
        ),
        "pem_native_forward_ready": (
            reports["pem_current_density"]["forward_ready"] is True
        ),
        "pem_native_forward_smoke": (
            pem_code == 0
            and pem_import.get("forward_smoke") == "passed"
            and pem_import.get("electrode_model") == "pem"
            and pem_import.get("electrode_projection") == "none"
        ),
        "pem_native_eidors_background_parity": (
            pem_parity["background_relative_l2"] <= 5e-4
        ),
        "pem_native_eidors_target_parity": (pem_parity["target_relative_l2"] <= 5e-4),
        "pem_z_contact_is_nonphysical": (
            pem_parity["z_contact_background_max_abs"] == 0.0
            and pem_parity["z_contact_target_max_abs"] == 0.0
        ),
        "pem_pyeidors_to_eidors_roundtrip": (
            pem_eidors_roundtrip["status"] == "passed"
            and pem_eidors_roundtrip["electrode_model"] == "pem"
            and pem_eidors_roundtrip["electrodes_exact"] is True
            and pem_eidors_roundtrip["protocol_exact"] is True
            and pem_eidors_roundtrip["pem_singleton_exact"] is True
            and pem_eidors_roundtrip["pem_no_projection"] is True
            and pem_eidors_roundtrip["pem_contact_marked_not_applicable"] is True
            and pem_eidors_roundtrip["pem_z_contact_invariant"] is True
        ),
        "missing_geometry_still_valid": reports["missing_fields"]["valid"] is True,
        "missing_contact_not_fabricated": (
            not np.any(missing_contact_present) and np.all(np.isnan(missing_contact))
        ),
        "missing_contact_blocks_forward": (
            reports["missing_fields"]["forward_ready"] is False
            and "contact_impedance_missing_no_eidors_default"
            in reports["missing_fields"]["forward_blockers"]
        ),
        "missing_example_conductivity_preserved": (
            np.isclose(_float_scalar(missing["background"]), 1.5)
            and np.isclose(
                np.asarray(missing["target_elem_data"], dtype=float).reshape(-1)[0],
                2.5,
            )
        ),
        "missing_runtime_fields_are_provenanced": (
            not _bool_scalar(missing["gnd_node_present"])
            and not _bool_scalar(missing["normalize_measurements_present"])
            and "derived_eidors_fwd_solve_1st_order"
            in str(np.asarray(missing["effective_gnd_node_source"]).reshape(-1)[0])
            and "eidors_runtime_default"
            in str(np.asarray(missing["normalize_measurements_source"]).reshape(-1)[0])
        ),
        "multiple_models_fail_without_selector": (
            "Multiple EIDORS forward model objects were discovered" in ambiguity_error
            and "fwd_model_var" in ambiguity_error
        ),
        "explicit_model_selector_is_deterministic": (
            reports["explicit_selector"]["valid"] is True
            and reports["explicit_selector"]["n_electrodes"] == 16
            and reports["explicit_selector"]["background_present"] is False
            and "background_image_missing_or_unmappable"
            in reports["explicit_selector"]["forward_blockers"]
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "schema": "eidors_pyeidors_source_semantics_acceptance_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "packages": reports,
        "imports": {
            "cem_2d": cem_2d_import,
            "cem_surface": cem_import,
            "pem_native": pem_import,
        },
        "discovery": {
            "ambiguous_auto_error": ambiguity_error,
            "explicit_selector": "fmdl_b",
        },
        "source_assertions": {
            "pem_raw_drive_amperes": 0.02,
            "pem_current_density_divisor": 2.0,
            "pem_effective_drive_amperes": 0.01,
            "pem_background_conductivity": 2.0,
            "pem_target_first_element_conductivity": 4.0,
        },
        "pem_native_parity": pem_parity,
        "pem_eidors_roundtrip": pem_eidors_roundtrip,
    }
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(report_path)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
