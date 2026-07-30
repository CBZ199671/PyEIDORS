#!/usr/bin/env python3
"""注册真实 float64 Nix 内核。 / Register the real-float64 Nix kernel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


KERNEL_NAME = "pyeidors-real-float64-nix"
DISPLAY_NAME = "PyEIDORS real float64 (Nix)"
PACKAGE_DIR = Path(__file__).resolve().parent
LAUNCHER = PACKAGE_DIR / "launch_pyeidors_float64_kernel.sh"


def _jupyter_data_dir() -> Path:
    configured = os.environ.get("JUPYTER_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".local" / "share" / "jupyter"


def build_kernel_spec() -> dict[str, object]:
    return {
        "argv": [
            "/usr/bin/env",
            "bash",
            str(LAUNCHER),
            "-f",
            "{connection_file}",
        ],
        "display_name": DISPLAY_NAME,
        "language": "python",
        "metadata": {
            "debugger": True,
            "pyeidors_profile": "default",
            "petsc_scalar_type": "real",
            "numpy_float_type": "float64",
        },
    }


def install_kernel(*, replace: bool = True) -> Path:
    target = _jupyter_data_dir() / "kernels" / KERNEL_NAME
    if target.exists():
        if not replace:
            raise FileExistsError(f"Kernel already exists: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    (target / "kernel.json").write_text(
        json.dumps(build_kernel_spec(), indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Fail instead of replacing an existing kernel registration.",
    )
    args = parser.parse_args()
    target = install_kernel(replace=not args.no_replace)
    print(f"已注册 / Registered: {DISPLAY_NAME}")
    print(f"内核配置 / Kernel spec: {target / 'kernel.json'}")
    print("VS Code：选择内核 / Select Kernel > Jupyter Kernel > " + DISPLAY_NAME)


if __name__ == "__main__":
    main()
