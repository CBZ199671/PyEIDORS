#!/usr/bin/env python3
"""Export a minimal exact mixed CEM/weighted-PEM Bridge v3 package."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eit_app.interop import (
    EidorsEnvironment,
    EidorsExportJob,
    InteropBundleExporter,
    build_geometry_payload_from_result,
    save_bridge_package,
)
from eit_app.interop.bridge_package import default_manifest
from eit_app.models.forward_model_config import ForwardModelConfig
from pyeidors import EITSystem
from pyeidors.data import PatternConfig
from pyeidors.interop import ElectrodeSpec, build_mesh_from_exchange_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eidors-startup", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    background = np.asarray([1.0, 1.0])
    target = np.asarray([1.0, 1.5])
    stim = np.asarray([[1.0, -1.0]])
    measurements = [np.asarray([[1.0, -1.0]])]
    config = ForwardModelConfig(
        mesh_dimension=2,
        n_elec=2,
        electrode_model="mixed",
        measurement_protocol="custom",
        custom_stim_matrix=stim,
        custom_meas_matrices=measurements,
        contact_impedance=None,
        drive_mode="total_current",
        interop_semantics={"effective_gnd_node": 1},
    )
    specs = (
        ElectrodeSpec(
            kind="cem",
            index_base=0,
            source_nodes=(0, 1),
            source_faces=((0, 1),),
            boundary_kind="exterior",
            contact_impedance=0.02,
            contact_impedance_present=True,
            contact_impedance_applicable=True,
        ),
        ElectrodeSpec(
            kind="pem",
            index_base=0,
            source_nodes=(2, 3),
            node_weights=(0.25, 0.75),
            boundary_kind="none",
            contact_impedance=None,
            contact_impedance_present=False,
            contact_impedance_applicable=False,
        ),
    )
    geometry = build_geometry_payload_from_result(
        node_coords=nodes,
        cell_connectivity=cells,
        forward_model_config=config,
        background_elem_data=background,
        truth_elem_data=target,
        electrode_specs=specs,
        mesh_name="mixed_cem_weighted_pem",
        scenario_name="bridge_v3_mixed_quickstart",
    )

    save_bridge_package(
        args.output,
        default_manifest(source_framework="pyeidors", package_kind="staging"),
        geometry_payload=geometry,
        forward_model_config=config,
    )
    mesh, imported = build_mesh_from_exchange_mat(args.output / "geometry.mat")
    pattern = PatternConfig(
        n_elec=2,
        measurement_protocol="custom",
        custom_stim_matrix=stim,
        custom_meas_matrices=measurements,
        drive_mode="total_current",
    )
    system = EITSystem(
        n_elec=2,
        pattern_config=pattern,
        electrode_model="mixed",
        contact_impedance=None,
        base_conductivity=1.0,
    )
    system.setup(mesh=mesh, initialize_inverse=False)
    homogeneous_data = system.forward_solve(imported["background_elem_data"])
    target_data = system.forward_solve(imported["target_elem_data"])

    exporter = InteropBundleExporter()
    exporter.export_bundle(
        EidorsExportJob(
            source_kind="simulation",
            output_dir=str(args.output),
            include_geometry=True,
            include_measurements=True,
            include_scripts=True,
            source_name="mixed_cem_weighted_pem",
        ),
        forward_model_config=config,
        environment=(
            EidorsEnvironment(
                name="mixed bridge acceptance",
                eidors_startup=args.eidors_startup,
            )
            if args.eidors_startup
            else None
        ),
        geometry_payload=geometry,
        measurements={
            "homogeneous": np.asarray(homogeneous_data.meas).reshape(-1),
            "target": np.asarray(target_data.meas).reshape(-1),
        },
        notes=[
            "Logical electrode 1 is CEM; logical electrode 2 is weighted PEM.",
            "EIDORS expands the weighted PEM into two physical point electrodes.",
        ],
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
