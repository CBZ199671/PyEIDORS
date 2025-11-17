#!/usr/bin/env python3
"""
Compare MATLAB-exported and PyEidors-generated linear systems for single-step GN.

This helper loads:
  * MATLAB's `eidors_linear_system.mat` (v7.3 HDF5) containing A, b, J, dv, …
  * PyEidors's `linear_system.npz` generated via --dump-linear-system
  * A measured difference vector (e.g., eit_difference_reconstruction.csv column 0)

It solves each system, projects back to measurement space, and reports RMSE/correlation
against the measured difference. Optional outputs are written for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import h5py


def load_measured_vector(path: Path, column: int) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        return data
    if not (0 <= column < data.shape[1]):
        raise IndexError(f"measured column {column} out of range for {path} with {data.shape[1]} columns")
    return data[:, column]


def solve_dense_system(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.linalg.solve(lhs, rhs)


def project_prediction(jacobian: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if jacobian.shape[1] == delta.size:
        return jacobian @ delta
    if jacobian.shape[0] == delta.size:
        return jacobian.T @ delta
    raise ValueError(f"Jacobian shape {jacobian.shape} incompatible with delta size {delta.size}")


def analyze_solution(tag: str, measured: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    error = predicted - measured
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    corr = float(np.corrcoef(predicted, measured)[0, 1]) if np.std(measured) > 0 else np.nan
    return {"tag": tag, "rmse": rmse, "mae": mae, "corr": corr}


def solve_matlab_system(mat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(mat_path, "r") as f:
        lhs = f["A"][()]
        rhs = f["b"][()].reshape(-1)
        jacobian = f["J"][()]
        delta = solve_dense_system(lhs, rhs)
        predicted = project_prediction(jacobian, delta)
    return predicted, delta


def solve_pyeidors_system(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    lhs = data["lhs"]
    rhs = data["rhs"]
    jacobian = data["jacobian"]
    delta = solve_dense_system(lhs, rhs)
    predicted = project_prediction(jacobian, delta)
    return predicted, delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--measured", type=Path, required=True, help="CSV containing measured difference (column specified below)")
    parser.add_argument("--measured-column", type=int, default=0, help="Which column of measured CSV to use")
    parser.add_argument("--matlab-mat", type=Path, help="Path to eidors_linear_system.mat")
    parser.add_argument("--pyeidors-npz", type=Path, help="Path to linear_system.npz saved via --dump-linear-system")
    parser.add_argument("--output-dir", type=Path, help="Optional directory to store predicted vectors for inspection")
    parser.add_argument("--report-components", action="store_true", help="Print norms/differences for J, W, RtR when both systems are provided.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    measured = load_measured_vector(args.measured, args.measured_column)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    matlab_components = None
    if args.matlab_mat:
        pred, delta = solve_matlab_system(args.matlab_mat)
        stats = analyze_solution("matlab", measured, pred)
        print(f"[MATLAB] corr={stats['corr']:.6f} rmse={stats['rmse']:.6e} mae={stats['mae']:.6e}")
        if args.output_dir:
            np.savetxt(args.output_dir / "matlab_predicted.csv", pred, delimiter=",")
            np.savetxt(args.output_dir / "matlab_delta.csv", delta, delimiter=",")
        with h5py.File(args.matlab_mat, "r") as f:
            matlab_components = {
                "J": np.array(f["J"]),
                "W": np.array(f["W/data"]),
                "RtR": np.array(f["RtR/data"]),
            }

    pyeidors_components = None
    if args.pyeidors_npz:
        pred, delta = solve_pyeidors_system(args.pyeidors_npz)
        stats = analyze_solution("pyeidors", measured, pred)
        print(f"[PyEidors] corr={stats['corr']:.6f} rmse={stats['rmse']:.6e} mae={stats['mae']:.6e}")
        if args.output_dir:
            np.savetxt(args.output_dir / "pyeidors_predicted.csv", pred, delimiter=",")
            np.savetxt(args.output_dir / "pyeidors_delta.csv", delta, delimiter=",")
        data = np.load(args.pyeidors_npz)
        pyeidors_components = {
            "J": data["jacobian"],
            "W": data["weights"],
            "RtR": data["noser_diag"],
        }

    if args.report_components and matlab_components and pyeidors_components:
        def report(name: str, transpose_py: bool = False) -> None:
            mat = matlab_components[name]
            py = pyeidors_components[name]
            if transpose_py:
                py = py.T
            if mat.shape != py.shape:
                print(f"[Diff] {name}: shape mismatch {mat.shape} vs {py.shape}")
                return
            diff = np.linalg.norm(mat - py)
            print(f"[Diff] {name}: ||mat||={np.linalg.norm(mat):.6e} ||py||={np.linalg.norm(py):.6e} ||mat-py||={diff:.6e}")

        report("J", transpose_py=True)
        report("W")
        report("RtR")


if __name__ == "__main__":
    main()
