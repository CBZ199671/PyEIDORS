"""Novice-facing EIDORS <-> PyEIDORS migration command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _emit(payload: dict[str, Any], *, stream: Any = sys.stdout) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        file=stream,
    )


def _geometry_path(path: Path) -> tuple[Path, Any | None]:
    if path.is_file() and path.suffix.lower() == ".mat":
        return path, None
    from eit_app.interop import InteropBundleImporter

    importer = InteropBundleImporter()
    loaded = importer.load_package(path)
    preview = importer.preview_loaded_package(loaded)
    mesh_path = Path(preview.forward_model_config.mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Bridge Package geometry file was not found: {mesh_path}"
        )
    return mesh_path, preview.forward_model_config


def _cmd_validate(args: argparse.Namespace) -> int:
    from eit_app.interop import validate_bridge_package

    report = validate_bridge_package(args.path)
    _emit(report)
    return 0 if report["valid"] else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    from eit_app.interop import validate_bridge_package

    report = validate_bridge_package(args.path)
    _emit(report)
    return 0 if report["valid"] else 1


def _forward_smoke(mesh: Any, config: Any) -> dict[str, Any]:
    from pyeidors import EITSystem
    from pyeidors.data import PatternConfig

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
        electrode_length_m_override=config.electrode_length_m_override,
        use_meas_current=config.use_meas_current,
        use_meas_current_next=config.use_meas_current_next,
        rotate_meas=config.rotate_meas,
        stim_direction=config.stim_direction,
        meas_direction=config.meas_direction,
        stim_first_positive=config.stim_first_positive,
    )
    system = EITSystem(
        n_elec=config.total_electrodes(),
        pattern_config=pattern,
        contact_impedance=config.contact_impedance,
        base_conductivity=config.background_conductivity,
        potential_order=config.potential_order,
    )
    system.setup(mesh=mesh, initialize_inverse=False)
    sigma = np.full(mesh.num_cells(), config.background_conductivity)
    data = system.forward_solve(sigma)
    measurements = np.asarray(data.meas).reshape(-1)
    if measurements.size == 0 or not np.isfinite(measurements).all():
        raise RuntimeError(
            "The imported mesh forward smoke solve produced no finite measurements"
        )
    return {
        "forward_smoke": "passed",
        "n_forward_measurements": int(measurements.size),
        "forward_measurements_finite": True,
    }


def _cmd_import_geometry(args: argparse.Namespace) -> int:
    from pyeidors.interop import build_mesh_from_exchange_mat

    source = Path(args.path)
    geometry_path, config = _geometry_path(source)
    mesh, payload = build_mesh_from_exchange_mat(geometry_path)
    report: dict[str, Any] = {
        "schema": "pyeidors_imported_geometry_summary_v1",
        "status": "imported",
        "geometry_path": str(geometry_path.resolve()),
        "geometry_format": str(np.asarray(payload["exchange_format"]).reshape(-1)[0]),
        "dimension": int(mesh.topology.dim),
        "mesh_family": str(mesh.mesh_family),
        "n_nodes": mesh.num_vertices(),
        "n_elements": mesh.num_cells(),
        "n_boundary_facets": int(mesh.facet_tags.indices.size),
        "n_electrodes": int(mesh.n_electrodes),
        "electrode_projection": str(getattr(mesh, "electrode_projection", "unknown")),
    }
    if args.forward_smoke:
        if config is None:
            from eit_app.interop import InteropBundleImporter

            loaded = InteropBundleImporter().load_package(geometry_path)
            config = (
                InteropBundleImporter()
                .preview_loaded_package(loaded)
                .forward_model_config
            )
        report.update(_forward_smoke(mesh, config))
    _emit(report)
    return 0


def _cmd_capture(args: argparse.Namespace) -> int:
    from eit_app.interop import (
        EidorsBridgeRunner,
        EidorsEnvironment,
        validate_bridge_package,
    )

    environment = EidorsEnvironment(
        name=args.profile_name,
        matlab_command=args.matlab,
        eidors_startup=args.eidors_startup,
    )
    root = EidorsBridgeRunner().run_capture(
        environment,
        Path(args.script),
        Path(args.output),
    )
    report = validate_bridge_package(root)
    report["capture_output"] = str(root.resolve())
    _emit(report)
    return 0 if report["valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyeidors-interop",
        description=(
            "Capture, validate, inspect, and import EIDORS/PyEIDORS "
            "Bridge Package v2 models."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text, handler in (
        ("validate", "Validate a Bridge Package or geometry MAT.", _cmd_validate),
        ("inspect", "Print a deterministic package/model summary.", _cmd_inspect),
    ):
        subparser = subparsers.add_parser(name, help=help_text)
        subparser.add_argument("path", help="Bridge directory or geometry.mat path")
        subparser.set_defaults(handler=handler)

    import_parser = subparsers.add_parser(
        "import-geometry",
        help="Build the exact 2D/3D DOLFINx EITMesh.",
    )
    import_parser.add_argument("path", help="Bridge directory or geometry.mat path")
    import_parser.add_argument(
        "--forward-smoke",
        action="store_true",
        help="Also run one homogeneous forward solve on the imported mesh.",
    )
    import_parser.set_defaults(handler=_cmd_import_geometry)

    capture_parser = subparsers.add_parser(
        "capture",
        help="Run an arbitrary EIDORS script and create Bridge Package v2.",
    )
    capture_parser.add_argument("script", help="EIDORS .m script to run")
    capture_parser.add_argument("--output", required=True, help="Output directory")
    capture_parser.add_argument(
        "--matlab",
        required=True,
        help="MATLAB executable path or command",
    )
    capture_parser.add_argument(
        "--eidors-startup",
        required=True,
        help="EIDORS startup.m path",
    )
    capture_parser.add_argument(
        "--profile-name",
        default="CLI MATLAB / EIDORS",
        help="Human-readable environment name",
    )
    capture_parser.set_defaults(handler=_cmd_capture)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        _emit(
            {
                "schema": "pyeidors_interop_error_v1",
                "status": "error",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            stream=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
