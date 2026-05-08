#!/usr/bin/env python3
"""Generate or run a MATLAB/EIDORS GREIT artifact export script.

The native registry builder is the GUI default.  This helper is the optional
official-parity backend: it prepares a MATLAB script that calls
``GREIT3D_distribution`` and ``mk_GREIT_model`` for an exact config, then
exports GREIT components for HDF5 import/registration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any


def _matlab_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + " ".join(_matlab_literal(item) for item in value) + "]"
    if value is None:
        return "[]"
    text = str(value).replace("'", "''")
    return f"'{text}'"


def build_matlab_script(config: dict[str, Any], output: Path, eidors_root: str) -> str:
    imgsz = config.get("imgsz") or config.get("greit_imgsz") or [6, 6, 4]
    target_size = config.get("target_size") or config.get("greit_target_size") or 0.20
    weight = config.get("weight") or config.get("greit_weight") or 0.5
    radius = config.get("radius", 1.0)
    height = config.get("height", 1.0)
    n_elec = config.get("n_elec", 16)
    n_rings = config.get("n_rings", 1)
    return f"""
addpath({_matlab_literal(eidors_root)});
eidors_startup;
out_file = {_matlab_literal(str(output))};
imgsz = {_matlab_literal(imgsz)};
target_size = {_matlab_literal(target_size)};
weight = {_matlab_literal(weight)};
radius = {_matlab_literal(radius)};
height = {_matlab_literal(height)};
n_elec = {_matlab_literal(n_elec)};
n_rings = {_matlab_literal(n_rings)};

% Build an EIDORS canonical cylindrical model.  Exact PyEIDORS mesh import can
% replace this block while preserving the same registry signature payload.
levels = linspace(0.15, 0.85, max(n_rings, 2));
elec_per_ring = n_elec;
elec_pos = [];
for iz = 1:numel(levels)
    for ie = 1:elec_per_ring
        elec_pos(end+1,:) = [levels(iz) ie-1]; %#ok<AGROW>
    end
end
[fmdl, ~] = ng_mk_cyl_models([height, radius, 0.05], elec_pos, [0.05]);
stim = mk_stim_patterns(numel(fmdl.electrode), 1, '{{ad}}', '{{ad}}', ...
    {{'no_meas_current'}}, 1);
fmdl.stimulation = stim;
img = mk_image(fmdl, 1);

vopt.imgsz = imgsz;
[imdl, opt_distr] = GREIT3D_distribution(fmdl, vopt);
opt.distr = opt_distr;
opt.keep_model_components = true;
imdl = mk_GREIT_model(imdl, target_size, weight, opt);

RM = imdl.solve_use_matrix.RM;
PJt = imdl.solve_use_matrix.PJt;
M = imdl.solve_use_matrix.M;
noiselev = imdl.solve_use_matrix.noiselev;
rec_model = opt_distr';
save(out_file, 'RM', 'PJt', 'M', 'noiselev', 'rec_model', 'imdl', '-v7.3');
exit;
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-json", required=True, help="GREIT registry config JSON"
    )
    parser.add_argument("--output", required=True, help="MATLAB .mat output path")
    parser.add_argument("--eidors-root", required=True, help="EIDORS toolbox root")
    parser.add_argument("--matlab", default="matlab", help="MATLAB executable")
    parser.add_argument("--script-out", help="Write generated MATLAB script here")
    parser.add_argument(
        "--run", action="store_true", help="Run MATLAB after script generation"
    )
    args = parser.parse_args()

    config_path = Path(args.config_json)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    output = Path(args.output).expanduser()
    script = build_matlab_script(config, output, args.eidors_root)

    if args.script_out:
        script_path = Path(args.script_out).expanduser()
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script + "\n", encoding="utf-8")
    else:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".m",
            prefix="pyeidors_greit_eidors_",
            delete=False,
            encoding="utf-8",
        )
        with handle:
            handle.write(script + "\n")
        script_path = Path(handle.name)

    if not args.run:
        print(str(script_path))
        return 0

    subprocess.run(
        [args.matlab, "-batch", f"run('{str(script_path)}')"],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
