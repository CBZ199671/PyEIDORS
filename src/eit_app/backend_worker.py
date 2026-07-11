"""Profile-isolated GUI backend worker entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import time


def _forward_runtime():
    from eit_app.backend_worker_protocol import (
        read_forward_request,
        write_forward_result,
    )
    from eit_app.controllers.forward_solver_controller import (
        execute_forward_request,
        prime_forward_setup_request,
    )

    return (
        read_forward_request,
        write_forward_result,
        execute_forward_request,
        prime_forward_setup_request,
    )


def _reconstruction_runtime():
    from eit_app.backend_worker_protocol import (
        read_reconstruction_request,
        write_reconstruction_result,
    )
    from eit_app.controllers.reconstruction_controller import run_reconstruction_request

    return (
        read_reconstruction_request,
        write_reconstruction_result,
        run_reconstruction_request,
    )


def _ecd_cwr_runtime():
    from eit_app.ecd_cwr_simulation import run_ecd_cwr_simulation_request_file

    return run_ecd_cwr_simulation_request_file


def _dynamic_kalman_runtime():
    from eit_app.dynamic_kalman_runtime import (
        apply_dynamic_kalman_to_reconstruction,
        dynamic_kalman_registry_command,
    )

    return apply_dynamic_kalman_to_reconstruction, dynamic_kalman_registry_command


def _prime_runtime() -> dict[str, object]:
    """Import heavyweight solver modules without running a solve or FFCx JIT."""

    modules = (
        "eit_app.controllers.forward_solver_controller",
        "pyeidors.core_system",
        "pyeidors.forward.eit_forward_model",
        "pyeidors.geometry.optimized_mesh_generator",
        "pyeidors.perf.capabilities",
        "pyeidors.perf.forward_solver_policy",
    )
    timings: dict[str, float] = {}
    errors: dict[str, str] = {}
    for module_name in modules:
        started = time.perf_counter()
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised in full runtime
            errors[module_name] = str(exc)
        finally:
            timings[module_name] = time.perf_counter() - started

    scalar_summary: dict[str, object] = {}
    try:
        from pyeidors.forward.complex_support import runtime_scalar_summary

        scalar_summary = runtime_scalar_summary()
    except Exception as exc:  # pragma: no cover - optional runtime guard
        errors["pyeidors.forward.complex_support"] = str(exc)

    mpi_summary: dict[str, object] = {}
    try:
        from pyeidors.perf.capabilities import probe_mpi_runtime

        mpi_summary = probe_mpi_runtime()
    except Exception as exc:  # pragma: no cover - optional runtime guard
        errors["pyeidors.perf.capabilities.probe_mpi_runtime"] = str(exc)

    petsc_cuda_probe: dict[str, object] = {}
    try:
        from pyeidors.perf.capabilities import probe_petsc_cuda_runtime

        petsc_cuda_probe = probe_petsc_cuda_runtime()
    except Exception as exc:  # pragma: no cover - optional runtime guard
        errors["pyeidors.perf.capabilities.probe_petsc_cuda_runtime"] = str(exc)

    return {
        "modules": list(modules),
        "timings_seconds": timings,
        "errors": errors,
        "scalar": scalar_summary,
        "mpi": mpi_summary,
        "petsc_cuda_probe": petsc_cuda_probe,
    }


def _run_forward(args: argparse.Namespace) -> int:
    read_forward_request, write_forward_result, execute_forward_request = (
        _forward_runtime()[:3]
    )
    request = read_forward_request(args.input)

    def progress(message: str) -> None:
        print(f"[backend-worker] {message}", file=sys.stderr, flush=True)

    result = execute_forward_request(request, progress_cb=progress)
    write_forward_result(args.output, result)
    return 0


def _run_reconstruct(args: argparse.Namespace) -> int:
    (
        read_reconstruction_request,
        write_reconstruction_result,
        run_reconstruction_request,
    ) = _reconstruction_runtime()
    request = read_reconstruction_request(args.input)

    def progress(message: str) -> None:
        print(f"[backend-worker] {message}", file=sys.stderr, flush=True)

    result = run_reconstruction_request(request, progress_cb=progress)
    write_reconstruction_result(args.output, result)
    return 0


def _serve(_args: argparse.Namespace) -> int:
    protocol_out = sys.stdout

    def send(payload: dict[str, object]) -> None:
        print(json.dumps(payload, sort_keys=True), file=protocol_out, flush=True)

    for raw in sys.stdin:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            send({"id": "", "type": "done", "status": "error", "error": str(exc)})
            continue
        request_id = str(message.get("id", ""))
        command = str(message.get("command", ""))
        if command == "shutdown":
            try:
                _dynamic_kalman_runtime()[1]("clear")
            except Exception:
                pass
            send({"id": request_id, "type": "done", "status": "ok"})
            return 0

        def progress(text: str, *, request_id: str = request_id) -> None:
            send({"id": request_id, "type": "progress", "message": str(text)})

        try:
            metadata: dict[str, object] | None = None
            with contextlib.redirect_stdout(sys.stderr):
                if command == "forward":
                    (
                        read_forward_request,
                        write_forward_result,
                        execute_forward_request,
                        _prime_forward_setup_request,
                    ) = _forward_runtime()
                    input_path = str(message["input"])
                    output_path = str(message["output"])
                    request = read_forward_request(input_path)
                    result = execute_forward_request(request, progress_cb=progress)
                    write_forward_result(output_path, result)
                elif command == "prime_forward_setup":
                    (
                        read_forward_request,
                        _write_forward_result,
                        _execute_forward_request,
                        prime_forward_setup_request,
                    ) = _forward_runtime()
                    input_path = str(message["input"])
                    request = read_forward_request(input_path)
                    metadata = prime_forward_setup_request(
                        request,
                        progress_cb=progress,
                    )
                elif command in {"reconstruct", "reconstruction"}:
                    (
                        read_reconstruction_request,
                        write_reconstruction_result,
                        run_reconstruction_request,
                    ) = _reconstruction_runtime()
                    input_path = str(message["input"])
                    output_path = str(message["output"])
                    request = read_reconstruction_request(input_path)
                    result = run_reconstruction_request(request, progress_cb=progress)
                    result = _dynamic_kalman_runtime()[0](request, result)
                    write_reconstruction_result(output_path, result)
                elif command in {
                    "dynamic_kalman_reset",
                    "dynamic_kalman_close",
                    "dynamic_kalman_status",
                    "dynamic_kalman_clear",
                }:
                    operation = command.removeprefix("dynamic_kalman_")
                    metadata = _dynamic_kalman_runtime()[1](
                        operation,
                        str(message.get("session_id", "")) or None,
                    )
                elif command == "ecd_cwr_simulate_cem":
                    run_ecd_cwr_simulation_request_file = _ecd_cwr_runtime()
                    input_path = str(message["input"])
                    metadata = run_ecd_cwr_simulation_request_file(
                        input_path,
                        progress_cb=progress,
                    )
                elif command == "prime_runtime":
                    metadata = _prime_runtime()
                else:
                    raise ValueError(f"unknown backend worker command: {command!r}")
        except Exception as exc:
            print(
                f"[backend-worker] request failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            send(
                {
                    "id": request_id,
                    "type": "done",
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue
        response: dict[str, object] = {
            "id": request_id,
            "type": "done",
            "status": "ok",
        }
        if metadata is not None:
            response["metadata"] = metadata
        send(response)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    forward = sub.add_parser("forward", help="Run one forward solve request.")
    forward.add_argument("--input", required=True)
    forward.add_argument("--output", required=True)
    reconstruct = sub.add_parser("reconstruct", help="Run one reconstruction request.")
    reconstruct.add_argument("--input", required=True)
    reconstruct.add_argument("--output", required=True)
    ecd_cwr = sub.add_parser(
        "ecd-cwr-simulate-cem",
        help="Run one ECD-CWR CEM simulation request.",
    )
    ecd_cwr.add_argument("--input", required=True)
    sub.add_parser("serve", help="Run a persistent JSON-lines worker.")

    args = parser.parse_args(argv)
    if args.command == "forward":
        return _run_forward(args)
    if args.command == "reconstruct":
        return _run_reconstruct(args)
    if args.command == "ecd-cwr-simulate-cem":
        run_ecd_cwr_simulation_request_file = _ecd_cwr_runtime()
        run_ecd_cwr_simulation_request_file(args.input)
        return 0
    if args.command == "serve":
        return _serve(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
