#!/usr/bin/env python3
"""Export a native PyEIDORS 3D model as Bridge Package v3."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eit_app.interop import (
    EidorsEnvironment,
    EidorsExportJob,
    InteropBundleExporter,
    build_geometry_payload_from_result,
)
from eit_app.models.forward_model_config import ForwardModelConfig
from pyeidors import EITSystem
from pyeidors.data import PatternConfig
from pyeidors.interop import build_boundary_facets, build_electrode_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eidors-startup", default="")
    parser.add_argument("--n-electrodes", type=int, default=8)
    parser.add_argument("--mesh-size", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ForwardModelConfig(
        mesh_dimension=3,
        mesh_refinement=args.mesh_size,
        mesh_family="tetra",
        n_elec=args.n_electrodes,
        n_rings=1,
        radius=1.0,
        height=1.0,
        electrode_height_ratio=0.2,
        electrode_level_fractions=(0.25, 0.75),
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        rotate_meas=True,
        use_meas_current=False,
        drive_mode="total_current",
        drive_value=1.0,
        contact_impedance=[0.01] * args.n_electrodes,
    )
    pattern = PatternConfig(
        n_elec=config.n_elec,
        n_rings=config.n_rings,
        stim_pattern=config.stim_pattern,
        meas_pattern=config.meas_pattern,
        drive_mode=config.drive_mode,
        drive_value=config.drive_value,
        rotate_meas=config.rotate_meas,
        use_meas_current=config.use_meas_current,
    )
    system = EITSystem(
        n_elec=config.total_electrodes(),
        pattern_config=pattern,
        contact_impedance=config.contact_impedance,
        base_conductivity=config.background_conductivity,
    )
    system.setup(
        mesh_source="generated",
        dimension=3,
        mesh_size=config.mesh_refinement,
        radius=config.radius,
        height=config.height,
        electrode_height_ratio=config.electrode_height_ratio,
        electrode_level_fractions=config.electrode_level_fractions,
        mesh_family=config.mesh_family,
        geometry_version=config.geometry_version,
        initialize_inverse=False,
    )
    mesh = system.mesh
    if mesh is None or system.fwd_model is None:
        raise RuntimeError("PyEIDORS did not create the requested 3D model")

    pattern_manager = system.fwd_model.pattern_manager
    export_config = config.with_overrides(
        measurement_protocol="custom",
        custom_stim_matrix=pattern_manager.stim_matrix,
        custom_meas_matrices=pattern_manager.meas_matrices,
    )
    sigma = np.full(mesh.num_cells(), config.background_conductivity)
    data = system.forward_solve(sigma)
    measurements = np.asarray(data.meas).reshape(-1)
    boundary_facets = build_boundary_facets(mesh)
    electrode_nodes, electrode_counts = build_electrode_arrays(mesh)
    geometry = build_geometry_payload_from_result(
        node_coords=mesh.coordinates(),
        cell_connectivity=mesh.cells(),
        forward_model_config=export_config,
        truth_elem_data=sigma,
        boundary_facets=boundary_facets,
        electrode_nodes=electrode_nodes,
        electrode_node_counts=electrode_counts,
        mesh_name="pyeidors_native_3d",
        scenario_name="pyeidors_3d_quickstart",
    )

    InteropBundleExporter().export_bundle(
        EidorsExportJob(
            source_kind="simulation",
            output_dir=str(args.output),
            include_geometry=True,
            include_measurements=True,
            include_scripts=True,
            source_name="PyEIDORS native 3D",
        ),
        forward_model_config=export_config,
        environment=EidorsEnvironment(
            name="EIDORS",
            eidors_startup=args.eidors_startup,
        ),
        geometry_payload=geometry,
        measurements={
            "homogeneous": measurements,
            "target": measurements,
        },
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
