from __future__ import annotations

import io
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from eit_app.backend_routing import (
    select_forward_backend_route,
    select_reconstruction_backend_route,
)
import eit_app.backend_routing as backend_routing_module
from eit_app.backend_worker_protocol import (
    _read_dataset_array,
    read_forward_result,
    read_forward_request,
    read_reconstruction_result,
    read_reconstruction_request,
    write_forward_result,
    write_reconstruction_result,
)
from eit_app.backend_worker_runtime import (
    backend_worker_command,
    backend_worker_env,
    prepare_inprocess_backend_runtime,
)
from eit_app.controllers.forward_solver_controller import (
    ForwardSolverRequest,
    ForwardSolverResult,
    execute_forward_request,
    execute_forward_request_in_backend,
)
import eit_app.controllers.forward_solver_controller as forward_controller_module
from eit_app.controllers.reconstruction_controller import (
    ReconstructionRequest,
    ReconstructionResult,
    execute_reconstruction_request_in_backend,
)
from eit_app.models.frame_model import FrameData
from eit_app.models.simulation_state import InhomogeneitySpec


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v136_real_3d_forward_routes_from_complex_gui_to_real_cuda_worker(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    request = ForwardSolverRequest(
        mesh_dimension=3,
        mesh_refinement=0.1,
        background_conductivity=1.0,
        inhomogeneities=[InhomogeneitySpec(conductivity=2.0)],
        forward_model_config={
            "mesh_dimension": 3,
            "mesh_refinement": 0.1,
            "background_conductivity": 1.0,
            "acceleration_profile": "default",
        },
    )

    route = select_forward_backend_route(request)

    assert route.profile == "cuda-amgx"
    assert route.external is True
    assert route.reason.startswith("real_input_uses_real_amgx_petsc_runtime")


def test_sm61_real_3d_forward_routes_to_legacy_cuda_worker(monkeypatch) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda-sm61")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    request = ForwardSolverRequest(
        mesh_dimension=3,
        mesh_refinement=0.1,
        background_conductivity=1.0,
        inhomogeneities=[InhomogeneitySpec(conductivity=2.0)],
        forward_model_config={
            "mesh_dimension": 3,
            "mesh_refinement": 0.1,
            "background_conductivity": 1.0,
            "acceleration_profile": "default",
        },
    )

    route = select_forward_backend_route(request)

    assert route.profile == "cuda-sm61"
    assert route.external is True
    assert route.reason.startswith("real_input_uses_real_cuda_petsc_runtime")


def test_v136_complex_3d_forward_uses_current_profile_inprocess(monkeypatch) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    request = ForwardSolverRequest(
        mesh_dimension=3,
        background_conductivity=1.0 + 0.2j,
        forward_model_config={
            "mesh_dimension": 3,
            "background_conductivity": "1+0.2j",
        },
    )

    route = select_forward_backend_route(request)

    assert route.profile == "complex64-cuda"
    assert route.external is False
    assert "target_profile_matches_current_runtime" in route.reason


def test_sm61_complex_3d_forward_uses_current_profile_inprocess(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda-sm61")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    request = ForwardSolverRequest(
        mesh_dimension=3,
        background_conductivity=1.0 + 0.2j,
        forward_model_config={
            "mesh_dimension": 3,
            "background_conductivity": "1+0.2j",
        },
    )

    route = select_forward_backend_route(request)

    assert route.profile == "complex64-cuda-sm61"
    assert route.external is False
    assert "target_profile_matches_current_runtime" in route.reason


@pytest.mark.parametrize(
    "runtime_profile",
    ["cuda-sm61", "complex64-cuda-sm61"],
)
def test_sm61_complex128_3d_forward_routes_to_cpu_complex_runtime(
    monkeypatch,
    runtime_profile: str,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", runtime_profile)
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setenv("EIT_APP_GUI_PRECISION", "complex128")
    request = ForwardSolverRequest(
        mesh_dimension=3,
        background_conductivity=1.0 + 0.2j,
        forward_model_config={
            "mesh_dimension": 3,
            "background_conductivity": "1+0.2j",
        },
    )

    route = select_forward_backend_route(request)

    assert route.profile == "complex"
    assert route.external is True
    assert "complex128_gpu_unsupported_on_sm61_fallback_cpu" in route.reason


def test_non_sm61_complex128_3d_forward_still_routes_to_complex_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setenv("EIT_APP_GUI_PRECISION", "complex128")
    request = ForwardSolverRequest(
        mesh_dimension=3,
        background_conductivity=1.0 + 0.2j,
        forward_model_config={
            "mesh_dimension": 3,
            "background_conductivity": "1+0.2j",
        },
    )

    route = select_forward_backend_route(request)

    assert route.profile == "complex-cuda"
    assert route.external is True
    assert "complex128_gpu_unsupported_on_sm61_fallback_cpu" not in route.reason


def test_sm61_complex64_3d_forward_still_routes_to_complex64_legacy_cuda(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda-sm61")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setenv("EIT_APP_GUI_PRECISION", "complex64")
    request = ForwardSolverRequest(
        mesh_dimension=3,
        background_conductivity=1.0 + 0.2j,
        forward_model_config={
            "mesh_dimension": 3,
            "background_conductivity": "1+0.2j",
        },
    )

    route = select_forward_backend_route(request)

    assert route.profile == "complex64-cuda-sm61"
    assert route.external is False
    assert "target_profile_matches_current_runtime" in route.reason


def test_v624_packaged_profile_command_wins_over_direct_profile_mismatch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_LAUNCH_MODE", "direct")
    monkeypatch.setenv(
        "EIT_APP_BACKEND_WORKER_COMMAND_CUDA",
        "/nix/store/pyeidors-cuda/bin/eit-backend-worker",
    )

    cmd, launch_mode = backend_worker_command(
        profile="cuda",
        worker_args=["forward", "--input", "in.h5", "--output", "out.h5"],
    )

    assert launch_mode == "profile_command"
    assert cmd == [
        "/nix/store/pyeidors-cuda/bin/eit-backend-worker",
        "forward",
        "--input",
        "in.h5",
        "--output",
        "out.h5",
    ]


def test_v638_backend_worker_nix_develop_does_not_wrap_uv(monkeypatch) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_COMMAND_CUDA", raising=False)

    cmd, launch_mode = backend_worker_command(
        profile="cuda",
        worker_args=["forward", "--input", "in.h5", "--output", "out.h5"],
    )

    assert launch_mode == "nix_develop"
    assert cmd[:5] == ["nix", "--option", "warn-dirty", "false", "develop"]
    assert ".#cuda" in cmd
    assert cmd[-1].startswith("python -m eit_app.backend_worker ")
    assert "uv run" not in cmd[-1]


def test_v624_real_3d_cpu_forward_routes_from_complex_gui_to_default_worker(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "cpu")
    request = ForwardSolverRequest(
        mesh_dimension=3,
        mesh_refinement=0.1,
        background_conductivity=1.0,
        inhomogeneities=[InhomogeneitySpec(conductivity=2.0)],
        forward_model_config={
            "mesh_dimension": 3,
            "mesh_refinement": 0.1,
            "background_conductivity": 1.0,
            "acceleration_profile": "default",
        },
    )

    route = select_forward_backend_route(request)

    assert route.profile == "default"
    assert route.external is True
    assert route.reason.startswith("real_input_uses_real_petsc_runtime")


def test_v475_backend_routing_complex_scan_uses_shared_bounded_helper() -> None:
    source = inspect.getsource(backend_routing_module._has_nonzero_imag)

    assert "has_nonzero_imaginary(arr, tol=1.0e-12)" in source
    assert "np.any(np.abs(np.imag" not in source


def test_v624_real_2d_forward_routes_from_complex_gui_to_default_worker(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    request = ForwardSolverRequest(
        mesh_dimension=2,
        mesh_refinement=0.1,
        background_conductivity=1.0,
        inhomogeneities=[InhomogeneitySpec(conductivity=2.0)],
        forward_model_config={
            "mesh_dimension": 2,
            "mesh_refinement": 0.1,
            "background_conductivity": 1.0,
        },
    )

    route = select_forward_backend_route(request)

    assert route.profile == "default"
    assert route.external is True
    assert "real_2d_prefers_real_profile_isolation" in route.reason


def test_v139_process_mode_can_force_2d_forward_external_worker(monkeypatch) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_MODE", "process")
    request = ForwardSolverRequest(
        mesh_dimension=2,
        background_conductivity=1.0,
        forward_model_config={"mesh_dimension": 2, "background_conductivity": 1.0},
    )

    route = select_forward_backend_route(request)

    assert route.profile == "default"
    assert route.external is True
    assert "forced_process" in route.reason


def test_v136_backend_worker_runner_uses_selected_nix_profile(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_PERSISTENT", "0")
    captured: dict[str, object] = {}

    def fake_run(cmd, *, cwd, env, text, capture_output, check):
        captured.update(
            {
                "cmd": list(cmd),
                "cwd": cwd,
                "env": dict(env),
                "text": text,
                "capture_output": capture_output,
                "check": check,
            }
        )
        output_path = list(cmd)[-1].split("--output ", 1)[1].strip().split()[0]
        request_path = list(cmd)[-1].split("--input ", 1)[1].split(" --output", 1)[0]
        request = read_forward_request(request_path)
        assert request.background_conductivity == 1.0
        write_forward_result(
            output_path,
            ForwardSolverResult(
                boundary_voltages=np.array([1.0, 2.0], dtype=np.float32),
                ground_truth_conductivity=np.array([1.0], dtype=np.float32),
                node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
                cell_connectivity=np.array([[0]], dtype=np.int32),
                n_elements=1,
                n_measurements=2,
                homogeneous_voltages=np.array([0.5, 1.0], dtype=np.float32),
                forward_model_config={"runtime_diagnostics": {"ok": True}},
            ),
        )
        return SimpleNamespace(returncode=0, stderr="[backend-worker] ok\n", stdout="")

    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.subprocess.run",
        fake_run,
    )
    progress: list[str] = []

    result = execute_forward_request_in_backend(
        ForwardSolverRequest(mesh_dimension=3, background_conductivity=1.0),
        profile="cuda",
        route_reason="real_input_uses_real_petsc_runtime",
        progress_cb=progress.append,
    )

    assert captured["cmd"][:5] == [
        "nix",
        "--option",
        "warn-dirty",
        "false",
        "develop",
    ]
    assert ".#cuda" in captured["cmd"]
    shell_command = captured["cmd"][-1]
    assert shell_command.startswith("python -m eit_app.backend_worker ")
    assert "uv run" not in shell_command
    assert captured["env"]["EIT_APP_GUI_RUNTIME_PROFILE"] == "cuda"
    assert captured["env"]["EIT_APP_GUI_PROFILE"] == "gpu"
    assert captured["env"]["PYEIDORS_ENV_SYNC_CACHE"] == "1"
    assert captured["env"]["XDG_CACHE_HOME"].endswith("/v1/cuda/xdg-cache")
    assert result.forward_model_config["backend_worker_profile"] == "cuda"
    assert result.forward_model_config["backend_worker_process_isolated"] is True
    assert result.forward_model_config["backend_worker_launch_mode"] == "nix_develop"
    assert result.forward_model_config["backend_worker_cache_home"].endswith(
        "/v1/cuda/xdg-cache"
    )
    assert result.forward_model_config["backend_worker_request_write_ms"] >= 0.0
    assert result.forward_model_config["backend_worker_subprocess_duration_ms"] >= 0.0
    assert result.forward_model_config["backend_worker_result_read_ms"] >= 0.0
    assert any("profile=cuda" in item for item in progress)


def test_v137_backend_worker_uses_current_python_when_profile_matches(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "cuda")
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_PERSISTENT", "0")
    captured: dict[str, object] = {}

    def fake_run(cmd, *, cwd, env, text, capture_output, check):
        captured.update(
            {
                "cmd": list(cmd),
                "cwd": cwd,
                "env": dict(env),
                "text": text,
                "capture_output": capture_output,
                "check": check,
            }
        )
        request_path = list(cmd)[list(cmd).index("--input") + 1]
        output_path = list(cmd)[list(cmd).index("--output") + 1]
        request = read_forward_request(request_path)
        assert request.background_conductivity == 1.0
        write_forward_result(
            output_path,
            ForwardSolverResult(
                boundary_voltages=np.array([1.0], dtype=np.float32),
                ground_truth_conductivity=np.array([1.0], dtype=np.float32),
                node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
                cell_connectivity=np.array([[0]], dtype=np.int32),
                n_elements=1,
                n_measurements=1,
                forward_model_config={},
            ),
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.subprocess.run",
        fake_run,
    )

    result = execute_forward_request_in_backend(
        ForwardSolverRequest(mesh_dimension=3, background_conductivity=1.0),
        profile="cuda",
        route_reason="profile_match_fast_launch",
    )

    assert captured["cmd"][:3] == [sys.executable, "-m", "eit_app.backend_worker"]
    assert "nix" not in captured["cmd"]
    assert captured["env"]["EIT_APP_GUI_RUNTIME_PROFILE"] == "cuda"
    assert result.forward_model_config["backend_worker_launch_mode"] == "current_python"


def test_v591_backend_worker_env_propagates_installed_site_packages(
    monkeypatch,
    tmp_path,
) -> None:
    cache_root = tmp_path / "backend-cache"
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("PYTHONPATH", "/tmp/original-pythonpath")
    installed_site = tmp_path / "nix-store" / "python3.13" / "site-packages"
    installed_site.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(installed_site))
    repo = tmp_path / "installed-app"
    repo.mkdir()

    env, _cache = backend_worker_env(repo=repo, profile="default")

    pythonpath = env["PYTHONPATH"].split(os.pathsep)
    assert str(repo) in pythonpath
    assert str(installed_site) in pythonpath
    assert "/tmp/original-pythonpath" in pythonpath
    assert pythonpath.index(str(installed_site)) < pythonpath.index(
        "/tmp/original-pythonpath"
    )


def test_v138_forward_uses_persistent_worker_pool_by_default(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_persistent_backend_worker_request(
        *, repo, profile, command, input_path, output_path, progress_cb
    ):
        captured.update(
            {
                "repo": repo,
                "profile": profile,
                "command": command,
                "input_path": input_path,
                "output_path": output_path,
            }
        )
        assert read_forward_request(input_path).background_conductivity == 1.0
        write_forward_result(
            output_path,
            ForwardSolverResult(
                boundary_voltages=np.array([1.0], dtype=np.float32),
                ground_truth_conductivity=np.array([1.0], dtype=np.float32),
                node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
                cell_connectivity=np.array([[0]], dtype=np.int32),
                n_elements=1,
                n_measurements=1,
                forward_model_config={},
            ),
        )
        return SimpleNamespace(
            launch_mode="current_python",
            cache_home=Path(tmp_path / "cache" / "v1" / "cuda" / "xdg-cache"),
            pid=12345,
            reused_process=True,
            stale_jit_locks_removed=0,
            primed_runtime=True,
            prime_command="prime_runtime",
            prime_duration_ms=3.5,
            request_duration_ms=12.0,
        )

    monkeypatch.setattr(
        "eit_app.backend_worker_pool.run_persistent_backend_worker_request",
        fake_run_persistent_backend_worker_request,
    )

    def fail_subprocess_run(*_args, **_kwargs):
        raise AssertionError("one-shot worker should not run")

    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.subprocess.run",
        fail_subprocess_run,
    )

    result = execute_forward_request_in_backend(
        ForwardSolverRequest(mesh_dimension=3, background_conductivity=1.0),
        profile="cuda",
        route_reason="persistent_pool",
    )

    assert captured["profile"] == "cuda"
    assert captured["command"] == "forward"
    assert result.forward_model_config["backend_worker_persistent"] is True
    assert result.forward_model_config["backend_worker_reused_process"] is True
    assert result.forward_model_config["backend_worker_pid"] == 12345
    assert result.forward_model_config["backend_worker_primed_runtime"] is True
    assert (
        result.forward_model_config["backend_worker_prime_command"] == "prime_runtime"
    )
    assert result.forward_model_config["backend_worker_prime_duration_ms"] == 3.5
    assert result.forward_model_config["backend_worker_request_duration_ms"] == 12.0
    assert result.forward_model_config["backend_worker_result_read_ms"] >= 0.0


def test_v666_amgx_profiles_keep_persistent_worker_by_default(monkeypatch) -> None:
    import eit_app.backend_worker_pool as worker_pool

    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_PERSISTENT", raising=False)
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_PERSISTENT_AMGX", raising=False)

    assert worker_pool.persistent_backend_workers_enabled("cuda") is True
    assert worker_pool.persistent_backend_workers_enabled("complex64-cuda") is True
    assert worker_pool.persistent_backend_workers_enabled("cuda-amgx") is True
    assert worker_pool.persistent_backend_workers_enabled("complex-cuda-amgx") is True

    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_PERSISTENT_AMGX", "0")
    assert worker_pool.persistent_backend_workers_enabled("cuda-amgx") is False

    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_PERSISTENT", "0")
    assert worker_pool.persistent_backend_workers_enabled("cuda-amgx") is False


def test_v666_forward_cuda_amgx_uses_persistent_worker_by_default(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_PERSISTENT", raising=False)
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_PERSISTENT_AMGX", raising=False)
    captured: dict[str, object] = {}

    def fake_run_persistent_backend_worker_request(
        *, repo, profile, command, input_path, output_path, progress_cb
    ):
        captured.update(
            {
                "repo": repo,
                "profile": profile,
                "command": command,
                "input_path": input_path,
                "output_path": output_path,
                "progress_cb": progress_cb,
            }
        )
        assert read_forward_request(input_path).background_conductivity == 1.0
        write_forward_result(
            output_path,
            ForwardSolverResult(
                boundary_voltages=np.array([1.0], dtype=np.float32),
                ground_truth_conductivity=np.array([1.0], dtype=np.float32),
                node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
                cell_connectivity=np.array([[0]], dtype=np.int32),
                n_elements=1,
                n_measurements=1,
                forward_model_config={},
            ),
        )
        return SimpleNamespace(
            launch_mode="profile_command",
            cache_home=Path(tmp_path / "cache" / "v1" / "cuda-amgx" / "xdg-cache"),
            pid=23456,
            reused_process=True,
            stale_jit_locks_removed=0,
            primed_runtime=True,
            prime_command="prime_runtime",
            prime_duration_ms=4.5,
            request_duration_ms=10.0,
        )

    monkeypatch.setattr(
        "eit_app.backend_worker_pool.run_persistent_backend_worker_request",
        fake_run_persistent_backend_worker_request,
    )

    def fail_subprocess_run(*_args, **_kwargs):
        raise AssertionError("cuda-amgx should use the persistent backend worker")

    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.subprocess.run",
        fail_subprocess_run,
    )

    result = execute_forward_request_in_backend(
        ForwardSolverRequest(mesh_dimension=3, background_conductivity=1.0),
        profile="cuda-amgx",
        route_reason="real_input_uses_real_amgx_petsc_runtime",
    )

    assert captured["profile"] == "cuda-amgx"
    assert captured["command"] == "forward"
    assert result.forward_model_config["backend_worker_profile"] == "cuda-amgx"
    assert result.forward_model_config["backend_worker_persistent"] is True
    assert result.forward_model_config["backend_worker_reused_process"] is True
    assert result.forward_model_config["backend_worker_pid"] == 23456
    assert result.forward_model_config["backend_worker_process_isolated"] is True


def test_v666_forward_cuda_amgx_can_disable_persistent_worker(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_PERSISTENT_AMGX", "0")
    captured: dict[str, object] = {}

    def fail_persistent_backend_worker_request(*_args, **_kwargs):
        raise AssertionError("cuda-amgx persistent worker was explicitly disabled")

    monkeypatch.setattr(
        "eit_app.backend_worker_pool.run_persistent_backend_worker_request",
        fail_persistent_backend_worker_request,
    )

    def fake_run(cmd, *, cwd, env, text, capture_output, check):
        captured.update(
            {
                "cmd": list(cmd),
                "cwd": cwd,
                "env": dict(env),
                "text": text,
                "capture_output": capture_output,
                "check": check,
            }
        )
        shell_command = list(cmd)[-1]
        output_path = shell_command.split("--output ", 1)[1].strip().split()[0]
        request_path = shell_command.split("--input ", 1)[1].split(" --output", 1)[0]
        request = read_forward_request(request_path)
        assert request.background_conductivity == 1.0
        write_forward_result(
            output_path,
            ForwardSolverResult(
                boundary_voltages=np.array([1.0], dtype=np.float32),
                ground_truth_conductivity=np.array([1.0], dtype=np.float32),
                node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
                cell_connectivity=np.array([[0]], dtype=np.int32),
                n_elements=1,
                n_measurements=1,
                forward_model_config={},
            ),
        )
        return SimpleNamespace(returncode=0, stderr="[backend-worker] ok\n", stdout="")

    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.subprocess.run",
        fake_run,
    )

    result = execute_forward_request_in_backend(
        ForwardSolverRequest(mesh_dimension=3, background_conductivity=1.0),
        profile="cuda-amgx",
        route_reason="real_input_uses_real_amgx_petsc_runtime",
    )

    assert ".#cuda-amgx" in captured["cmd"]
    assert captured["env"]["EIT_APP_GUI_RUNTIME_PROFILE"] == "cuda-amgx"
    assert result.forward_model_config["backend_worker_profile"] == "cuda-amgx"
    assert result.forward_model_config["backend_worker_persistent"] is False


def test_v319_forward_timing_metadata_schema(monkeypatch) -> None:
    monkeypatch.setattr(
        forward_controller_module.time,
        "perf_counter",
        lambda: 10.25,
    )

    metadata = forward_controller_module._forward_timing_metadata(
        timings_ms={"setup_mesh_and_forward_model": 125.0},
        phase_order=["setup_mesh_and_forward_model"],
        total_started=10.0,
        mesh_dimension=3,
    )

    assert metadata["forward_timing_schema"] == "eit_app_forward_timing_v1"
    assert metadata["forward_timing_mesh_dimension"] == 3
    assert metadata["forward_timing_phase_order"] == [
        "setup_mesh_and_forward_model",
        "total",
    ]
    assert metadata["forward_timing_ms"]["setup_mesh_and_forward_model"] == 125.0
    assert metadata["forward_timing_total_ms"] == pytest.approx(250.0)
    assert metadata["forward_timing_ms"]["total"] == pytest.approx(250.0)


def test_v319_record_forward_visualization_timing_metadata() -> None:
    import eit_app.ui.forward_timing as forward_timing_module

    result = ForwardSolverResult(
        boundary_voltages=np.array([1.0], dtype=np.float32),
        ground_truth_conductivity=np.array([1.0], dtype=np.float32),
        node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 0, 0]], dtype=np.int32),
        n_elements=1,
        n_measurements=1,
        forward_model_config={
            "forward_timing_ms": {"total": 4.0},
            "forward_timing_phase_order": ["total"],
        },
    )

    forward_timing_module._record_forward_visualization_timing(result, visual_ms=12.5)

    assert result.forward_model_config["forward_timing_ms"][
        "gui_visualization_update"
    ] == pytest.approx(12.5)
    assert result.forward_model_config["gui_forward_visualization_update_ms"] == (
        pytest.approx(12.5)
    )
    assert (
        "gui_visualization_update"
        in result.forward_model_config["forward_timing_phase_order"]
    )


def test_v639_gui_helper_imports_stay_lightweight() -> None:
    code = (
        "import sys\n"
        "import eit_app.ui.forward_timing\n"
        "import eit_app.ui.forward_prewarm\n"
        "for name in ('pyqtgraph', 'OpenGL', 'eit_app.ui.main_window'):\n"
        "    if name in sys.modules:\n"
        "        raise SystemExit(f'unexpected heavy import: {name}')\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_v650_gui_helper_modules_are_git_tracked_for_nix_flake_source() -> None:
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("requires Git checkout to verify flake source tracking")

    for rel_path in (
        "src/eit_app/ui/forward_prewarm.py",
        "src/eit_app/ui/forward_timing.py",
    ):
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel_path],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr or completed.stdout


def test_v651_simulation_mesh_density_copy_avoids_exact_size_claim() -> None:
    from eit_app.i18n.en import TRANSLATIONS as EN_TRANSLATIONS
    from eit_app.i18n.zh import TRANSLATIONS as ZH_TRANSLATIONS

    assert ZH_TRANSLATIONS["sim.mesh.size_label"] == "网格细度："
    assert "网格尺寸" not in ZH_TRANSLATIONS["sim.mesh.size_label"]
    assert "特征长度" not in ZH_TRANSLATIONS["sim.mesh.size_label"]
    assert EN_TRANSLATIONS["sim.mesh.size_label"] == "Mesh density:"
    assert "Mesh size" not in EN_TRANSLATIONS["sim.mesh.size_label"]
    assert "Target length" not in EN_TRANSLATIONS["sim.mesh.size_label"]

    zh_tooltip = ZH_TRANSLATIONS["sim.mesh.refinement_tooltip"]
    en_tooltip = EN_TRANSLATIONS["sim.mesh.refinement_tooltip"]
    assert "粗/中/细/很细" in zh_tooltip
    assert "D/18" in zh_tooltip
    assert "生成尺度 h" in zh_tooltip
    assert "细化参数" in zh_tooltip
    assert "边长/直径统计" in zh_tooltip
    assert "domain diameter D" in en_tooltip
    assert "generation scale h" in en_tooltip
    assert "integer refinement" in en_tooltip
    assert "D/18" in en_tooltip
    assert "edge-length or diameter statistics" in en_tooltip

    assert ZH_TRANSLATIONS["sim.mesh.density_summary"].startswith("D/{density}")
    assert "refinement" in ZH_TRANSLATIONS["sim.mesh.density_summary"]
    assert "预估元素数" in ZH_TRANSLATIONS["sim.mesh.density_summary"]
    assert "estimated elements" in EN_TRANSLATIONS["sim.mesh.density_summary"]
    assert ZH_TRANSLATIONS["sim.mesh.density_advanced_toggle"] == "高级输入"
    assert "求解时间" in ZH_TRANSLATIONS["sim.mesh.density_warning"]

    source = (REPO_ROOT / "src/eit_app/ui/simulation/mesh_setup_panel.py").read_text(
        encoding="utf-8"
    )
    assert "QSlider" in source
    assert "_SliderTickLabels" in source
    assert "_mesh_density_spin" in source
    assert "_mesh_density_warning" in source


def test_v326_forward_mesh_geometry_arrays_stream_connectivity_once() -> None:
    class _FakeIndexMap:
        def __init__(self, size_local: int) -> None:
            self.size_local = int(size_local)

    class _FakeConnectivity:
        def __init__(self, rows: np.ndarray) -> None:
            self.rows = np.asarray(rows, dtype=np.int32)
            self.calls: list[int] = []

        def links(self, idx: int) -> np.ndarray:
            self.calls.append(int(idx))
            return self.rows[int(idx)]

    class _FakeTopology:
        dim = 3

        def __init__(self, rows: np.ndarray) -> None:
            self.connectivity_obj = _FakeConnectivity(rows)
            self.created: list[tuple[int, int]] = []

        def create_connectivity(self, from_dim: int, to_dim: int) -> None:
            self.created.append((int(from_dim), int(to_dim)))

        def connectivity(self, from_dim: int, to_dim: int):
            if (int(from_dim), int(to_dim)) == (3, 0):
                return self.connectivity_obj
            return None

        def index_map(self, dim: int):
            if int(dim) == 3:
                return _FakeIndexMap(self.connectivity_obj.rows.shape[0])
            return None

    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    topology = _FakeTopology(cells)
    mesh = SimpleNamespace(
        geometry=SimpleNamespace(
            x=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
        ),
        topology=topology,
    )

    centers, node_coords, cell_connectivity, n_cells = (
        forward_controller_module._forward_mesh_geometry_arrays(
            mesh,
            mesh_dimension=3,
        )
    )

    assert n_cells == 2
    assert topology.created == [(3, 0)]
    assert topology.connectivity_obj.calls == [0, 1]
    assert node_coords.dtype == np.float32
    np.testing.assert_array_equal(cell_connectivity, cells)
    np.testing.assert_allclose(
        centers,
        [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )
    source = inspect.getsource(forward_controller_module._forward_mesh_geometry_arrays)
    assert "cell_midpoints" not in source
    assert "[connectivity.links" not in source
    assert "work = np.empty_like(centers)" not in source
    assert "work = np.empty(n_cells, dtype=node_coords.dtype)" in source
    assert "np.take(node_coords[:, axis], indices, out=work)" in source


def test_v576_forward_mesh_geometry_arrays_uses_flat_connectivity_array() -> None:
    class _FakeIndexMap:
        def __init__(self, size_local: int) -> None:
            self.size_local = int(size_local)

    class _FlatConnectivity:
        def __init__(self, rows: np.ndarray) -> None:
            self.rows = np.asarray(rows, dtype=np.int64)
            self.array = self.rows.reshape(-1)
            self.offsets = np.arange(
                0,
                self.array.size + 1,
                self.rows.shape[1],
                dtype=np.int64,
            )

        def links(self, idx: int) -> np.ndarray:
            raise AssertionError(f"flat connectivity path should not call links({idx})")

    class _FakeTopology:
        dim = 3

        def __init__(self, rows: np.ndarray) -> None:
            self.connectivity_obj = _FlatConnectivity(rows)

        def create_connectivity(self, from_dim: int, to_dim: int) -> None:
            assert (int(from_dim), int(to_dim)) == (3, 0)

        def connectivity(self, from_dim: int, to_dim: int):
            if (int(from_dim), int(to_dim)) == (3, 0):
                return self.connectivity_obj
            return None

        def index_map(self, dim: int):
            if int(dim) == 3:
                return _FakeIndexMap(self.connectivity_obj.rows.shape[0])
            return None

    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    mesh = SimpleNamespace(
        geometry=SimpleNamespace(
            x=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
        ),
        topology=_FakeTopology(cells),
    )

    centers, node_coords, cell_connectivity, n_cells = (
        forward_controller_module._forward_mesh_geometry_arrays(
            mesh,
            mesh_dimension=3,
        )
    )

    assert n_cells == 2
    assert node_coords.dtype == np.float32
    assert cell_connectivity.dtype == np.int32
    np.testing.assert_array_equal(cell_connectivity, cells.astype(np.int32))
    np.testing.assert_allclose(
        centers,
        [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )
    source = inspect.getsource(forward_controller_module._forward_mesh_geometry_arrays)
    assert "flat_arr[start:stop].reshape" in source


def test_v577_forward_measurement_values_copies_only_when_noise_is_added() -> None:
    values = np.linspace(1.0, 4.0, 4, dtype=np.float32)

    clean = forward_controller_module._forward_measurement_values(
        values,
        noise_level=0.0,
    )
    noisy = forward_controller_module._forward_measurement_values(
        values,
        noise_level=0.1,
        rng=np.random.default_rng(123),
    )

    assert clean is values
    assert clean.dtype == np.float32
    assert not np.shares_memory(noisy, values)
    assert noisy.dtype == np.float32
    assert np.array_equal(values, np.linspace(1.0, 4.0, 4, dtype=np.float32))
    assert not np.array_equal(noisy, values)


def test_v320_prime_forward_setup_request_stops_before_solve(monkeypatch, tmp_path):
    class _FakeIndexMap:
        size_local = 7

    class _FakeTopology:
        dim = 3

        def index_map(self, _dim):
            return _FakeIndexMap()

    fake_system = SimpleNamespace(
        mesh=SimpleNamespace(
            geometry=SimpleNamespace(x=np.zeros((11, 3), dtype=np.float64)),
            topology=_FakeTopology(),
        ),
        forward_solve=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("setup prime must not run a forward solve")
        ),
    )
    setup_calls: list[object] = []
    monkeypatch.setattr(
        "eit_app.backend_worker_runtime.prepare_inprocess_backend_runtime",
        lambda repo: SimpleNamespace(
            xdg_cache_home=tmp_path / "xdg-cache",
            removed_stale_jit_locks=[],
        ),
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_forward_config_from_request",
        lambda _req: SimpleNamespace(mesh_dimension=3),
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_pattern_and_electrode_count",
        lambda _forward_cfg: (object(), 16),
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_resolve_forward_runtime",
        lambda _forward_cfg: {"forward_backend": "dolfinx"},
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_create_forward_system",
        lambda **_kwargs: fake_system,
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_setup_generated_forward_system",
        lambda system, **_kwargs: setup_calls.append(system),
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_forward_runtime_diagnostics",
        lambda _system: {"static_setup_lookup": {"hit": True}},
    )
    progress: list[str] = []

    metadata = forward_controller_module.prime_forward_setup_request(
        ForwardSolverRequest(mesh_dimension=3),
        progress_cb=progress.append,
    )

    assert setup_calls == [fake_system]
    assert metadata["forward_setup_prime"] is True
    assert metadata["n_nodes"] == 11
    assert metadata["n_elements"] == 7
    assert metadata["runtime_diagnostics"]["static_setup_lookup"]["hit"] is True
    assert metadata["forward_timing_schema"] == "eit_app_forward_timing_v1"
    assert "configure.runtime" in metadata["forward_timing_ms"]
    assert "configure.system_object" in metadata["forward_timing_ms"]
    assert metadata["forward_timing_phase_order"].index("configure.runtime") < (
        metadata["forward_timing_phase_order"].index("configure_system")
    )
    assert any("setup prime" in item.lower() for item in progress)


def test_v320_prime_forward_setup_worker_pool_helper_dispatches_command(
    monkeypatch,
    tmp_path,
) -> None:
    import eit_app.backend_worker_pool as pool

    captured: dict[str, object] = {}

    def _fake_run_persistent_backend_worker_request(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            profile=kwargs["profile"],
            cache_home=tmp_path / "xdg-cache",
            launch_mode="current_python",
            pid=123,
            reused_process=True,
            stale_jit_locks_removed=0,
        )

    monkeypatch.setattr(
        pool,
        "run_persistent_backend_worker_request",
        _fake_run_persistent_backend_worker_request,
    )
    request_path = tmp_path / "forward_request.h5"
    request_path.write_text("placeholder", encoding="utf-8")

    meta = pool.prime_persistent_backend_worker_forward_setup(
        repo=tmp_path,
        profile="cuda",
        input_path=request_path,
    )

    assert meta is not None
    assert captured["command"] == "prime_forward_setup"
    assert captured["input_path"] == request_path
    assert str(captured["output_path"]).endswith(".prime.out")


def test_v322_setup_prime_warm_key_tracks_setup_not_inhomogeneity() -> None:
    import eit_app.ui.forward_prewarm as forward_prewarm_module

    setup_config = {
        "mesh_dimension": 3,
        "mesh_refinement": 0.1,
        "n_elec": 16,
        "n_rings": 1,
        "measurement_protocol": "eidors_full_3d",
        "stim_pattern": "{ad}",
        "meas_pattern": "{ad}",
        "forward_backend": "dolfinx",
        "petsc_device": "cuda",
        "background_conductivity": 1.0,
        "noise_level": 0.0,
    }

    def _request(
        *,
        inhomogeneity_size: float,
        mesh_refinement: float = 0.1,
        setup_payload: dict[str, object] | None = None,
        signature: str = "sig-a",
    ) -> ForwardSolverRequest:
        forward_config = dict(setup_config)
        forward_config["mesh_refinement"] = mesh_refinement
        forward_config.update(
            {
                "request_source": "prewarm",
                "simulation_input_signature": signature,
                "simulation_input_signature_payload": (
                    {
                        "schema": "simulation_forward_inputs_v1",
                        "forward_model_config": setup_payload or forward_config,
                        "inhomogeneities": [
                            {"shape": "sphere", "size_x": inhomogeneity_size}
                        ],
                    }
                ),
            }
        )
        return ForwardSolverRequest(
            mesh_dimension=3,
            mesh_refinement=mesh_refinement,
            n_electrodes=16,
            background_conductivity=1.0,
            inhomogeneities=[
                InhomogeneitySpec(shape="sphere", size_x=inhomogeneity_size)
            ],
            noise_level=0.0,
            forward_model_config=forward_config,
        )

    key_a = forward_prewarm_module.backend_forward_setup_warm_key(
        profile="cuda",
        request=_request(inhomogeneity_size=0.1),
        setup_prime=True,
    )
    key_b = forward_prewarm_module.backend_forward_setup_warm_key(
        profile="cuda",
        request=_request(inhomogeneity_size=0.3, signature="sig-b"),
        setup_prime=True,
    )
    key_c = forward_prewarm_module.backend_forward_setup_warm_key(
        profile="cuda",
        request=_request(
            inhomogeneity_size=0.3,
            mesh_refinement=0.08,
            setup_payload={**setup_config, "mesh_refinement": 0.08},
            signature="sig-c",
        ),
        setup_prime=True,
    )

    assert key_a == key_b
    assert key_a != key_c
    assert (
        forward_prewarm_module.backend_forward_setup_warm_key(
            profile="cuda",
            request=_request(inhomogeneity_size=0.1),
            setup_prime=False,
        )
        == "cuda"
    )


def test_v322_setup_prime_warm_key_fallback_ignores_volatile_fields() -> None:
    import eit_app.ui.forward_prewarm as forward_prewarm_module

    stable_config = {
        "mesh_dimension": 3,
        "mesh_refinement": 0.1,
        "n_elec": 16,
        "measurement_protocol": "eidors_full_3d",
        "forward_backend": "dolfinx",
    }

    def _request(**volatile: object) -> ForwardSolverRequest:
        forward_config = dict(stable_config)
        forward_config.update(volatile)
        return ForwardSolverRequest(
            mesh_dimension=3,
            mesh_refinement=0.1,
            n_electrodes=16,
            background_conductivity=1.0,
            inhomogeneities=[InhomogeneitySpec(shape="sphere", size_x=0.1)],
            noise_level=float(volatile.get("noise_level", 0.0) or 0.0),
            forward_model_config=forward_config,
        )

    key_a = forward_prewarm_module.backend_forward_setup_warm_key(
        profile="cuda",
        request=_request(
            background_conductivity=1.0,
            noise_level=0.0,
            request_source="prewarm",
            simulation_input_signature="sig-a",
            inhomogeneities_hash="anomaly-a",
        ),
        setup_prime=True,
    )
    key_b = forward_prewarm_module.backend_forward_setup_warm_key(
        profile="cuda",
        request=_request(
            background_conductivity=2.0,
            noise_level=0.2,
            request_source="manual",
            simulation_input_signature="sig-b",
            inhomogeneities_hash="anomaly-b",
        ),
        setup_prime=True,
    )
    key_c = forward_prewarm_module.backend_forward_setup_warm_key(
        profile="cuda",
        request=_request(measurement_protocol="adjacent"),
        setup_prime=True,
    )

    assert key_a == key_b
    assert key_a != key_c


def test_v610_3d_prewarm_mode_defaults_to_setup(monkeypatch) -> None:
    import eit_app.ui.forward_prewarm as forward_prewarm_module

    monkeypatch.delenv("EIT_APP_FORWARD_PREWARM_3D_MODE", raising=False)
    mode = forward_prewarm_module.sim_forward_prewarm_mode(mesh_dimension=3)

    assert mode == "setup"


def test_v148_3d_prewarm_mode_keeps_explicit_worker(monkeypatch) -> None:
    import eit_app.ui.forward_prewarm as forward_prewarm_module

    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM_3D_MODE", "worker")
    mode = forward_prewarm_module.sim_forward_prewarm_mode(mesh_dimension=3)

    assert mode == "worker"


def test_v329_simulation_backend_warm_report_surfaces_petsc_probe() -> None:
    import eit_app.ui.forward_prewarm as forward_prewarm_module

    report = forward_prewarm_module.simulation_backend_warm_report(
        SimpleNamespace(
            profile="cuda",
            pid=4321,
            rss_bytes=1024,
            rss_limit_bytes=2048,
            primed_runtime=True,
            prime_command="prime_runtime",
            prime_duration_ms=12.5,
            prime_metadata={
                "petsc_cuda_probe": {
                    "petsc_cuda": True,
                    "probe_cache": {"hit": True, "layer": "disk"},
                }
            },
            request_duration_ms=13.0,
            recycled_after_request=False,
            recycle_reason="",
        ),
        profile="cuda",
        warm_key="cuda",
        setup_prime=False,
    )

    assert report["profile"] == "cuda"
    assert report["pid"] == 4321
    assert report["primed_runtime"] is True
    assert report["petsc_cuda_available"] is True
    assert report["petsc_cuda_probe_cache_hit"] is True
    assert report["petsc_cuda_probe_cache_layer"] == "disk"
    assert report["petsc_cuda_probe_status"] == "hit/disk"


def test_v136_real_3d_reconstruction_routes_from_complex_gui_to_real_cuda_worker(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    frame = FrameData(
        real=np.array([1.0, 2.0], dtype=np.float32),
        imag=np.zeros(2, dtype=np.float32),
        timestamp=0.0,
        frame_index=0,
    )
    request = ReconstructionRequest(
        reference_frame=frame,
        target_frame=frame,
        use_part="real",
        mesh_dimension=3,
        metadata={"acceleration_profile": "gpu3d"},
    )

    route = select_reconstruction_backend_route(request)

    assert route.profile == "cuda-amgx"
    assert route.external is True
    assert route.reason.startswith("real_input_uses_real_amgx_petsc_runtime")


def test_v136_complex_reconstruction_stays_on_complex_cuda_profile(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    frame = FrameData(
        real=np.array([1.0, 2.0], dtype=np.float32),
        imag=np.array([0.1, 0.2], dtype=np.float32),
        timestamp=0.0,
        frame_index=0,
    )
    request = ReconstructionRequest(
        reference_frame=frame,
        target_frame=frame,
        use_part="complex",
        mesh_dimension=3,
        metadata={"acceleration_profile": "gpu3d"},
    )

    route = select_reconstruction_backend_route(request)

    assert route.profile == "complex64-cuda"
    assert route.external is False
    assert "target_profile_matches_current_runtime" in route.reason


def test_sm61_complex128_reconstruction_routes_to_cpu_complex_runtime(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda-sm61")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setenv("EIT_APP_GUI_PRECISION", "complex128")
    frame = FrameData(
        real=np.array([1.0, 2.0], dtype=np.float32),
        imag=np.array([0.1, 0.2], dtype=np.float32),
        timestamp=0.0,
        frame_index=0,
    )
    request = ReconstructionRequest(
        reference_frame=frame,
        target_frame=frame,
        use_part="complex",
        mesh_dimension=3,
        metadata={"acceleration_profile": "gpu3d"},
    )

    route = select_reconstruction_backend_route(request)

    assert route.profile == "complex"
    assert route.external is True
    assert "complex128_gpu_unsupported_on_sm61_fallback_cpu" in route.reason


def test_v136_reconstruction_backend_worker_uses_hdf5_protocol_and_profile(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_PERSISTENT", "0")
    captured: dict[str, object] = {}

    def fake_run(cmd, *, cwd, env, text, capture_output, check):
        captured.update(
            {
                "cmd": list(cmd),
                "cwd": cwd,
                "env": dict(env),
                "text": text,
                "capture_output": capture_output,
                "check": check,
            }
        )
        output_path = list(cmd)[-1].split("--output ", 1)[1].strip().split()[0]
        request_path = list(cmd)[-1].split("--input ", 1)[1].split(" --output", 1)[0]
        request = read_reconstruction_request(request_path)
        assert request.use_part == "real"
        assert request.reference_frame.real.tolist() == [1.0, 2.0]
        write_reconstruction_result(
            output_path,
            ReconstructionResult(
                conductivity=np.array([1.0], dtype=np.float32),
                node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
                cell_connectivity=np.array([[0]], dtype=np.int32),
                measured=np.array([0.1, 0.2], dtype=np.float32),
                simulated=np.array([0.1, 0.2], dtype=np.float32),
                metadata={"runtime": "fake"},
            ),
        )
        return SimpleNamespace(returncode=0, stderr="[backend-worker] ok\n", stdout="")

    monkeypatch.setattr(
        "eit_app.controllers.reconstruction_controller.subprocess.run",
        fake_run,
    )
    frame = FrameData(
        real=np.array([1.0, 2.0], dtype=np.float32),
        imag=np.zeros(2, dtype=np.float32),
        timestamp=0.0,
        frame_index=0,
    )

    result = execute_reconstruction_request_in_backend(
        ReconstructionRequest(
            reference_frame=frame,
            target_frame=frame,
            use_part="real",
            mesh_dimension=3,
        ),
        profile="cuda",
        route_reason="real_input_uses_real_petsc_runtime",
        progress_cb=lambda _message: None,
    )

    assert ".#cuda" in captured["cmd"]
    assert captured["env"]["EIT_APP_GUI_RUNTIME_PROFILE"] == "cuda"
    assert captured["env"]["PYEIDORS_ENV_SYNC_CACHE"] == "1"
    assert captured["env"]["XDG_CACHE_HOME"].endswith("/v1/cuda/xdg-cache")
    assert result.metadata["backend_worker_profile"] == "cuda"
    assert result.metadata["backend_worker_process_isolated"] is True
    assert result.metadata["backend_worker_launch_mode"] == "nix_develop"
    assert result.metadata["backend_worker_cache_home"].endswith("/v1/cuda/xdg-cache")


def test_v138_persistent_transport_failure_falls_back_to_one_shot(
    monkeypatch,
    tmp_path,
) -> None:
    from eit_app.backend_worker_pool import BackendWorkerTransportError

    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))

    def fail_persistent(*_args, **_kwargs):
        raise BackendWorkerTransportError("transport failed")

    monkeypatch.setattr(
        "eit_app.backend_worker_pool.run_persistent_backend_worker_request",
        fail_persistent,
    )

    def fake_run(cmd, *, cwd, env, text, capture_output, check):
        output_path = list(cmd)[-1].split("--output ", 1)[1].strip().split()[0]
        write_forward_result(
            output_path,
            ForwardSolverResult(
                boundary_voltages=np.array([1.0], dtype=np.float32),
                ground_truth_conductivity=np.array([1.0], dtype=np.float32),
                node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
                cell_connectivity=np.array([[0]], dtype=np.int32),
                n_elements=1,
                n_measurements=1,
                forward_model_config={},
            ),
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.subprocess.run",
        fake_run,
    )
    progress: list[str] = []

    result = execute_forward_request_in_backend(
        ForwardSolverRequest(mesh_dimension=3, background_conductivity=1.0),
        profile="cuda",
        route_reason="fallback",
        progress_cb=progress.append,
    )

    assert result.forward_model_config["backend_worker_persistent"] is False
    assert any("falling back to one-shot worker" in item for item in progress)


def test_v146_cancelled_forward_backend_transport_error_does_not_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    from eit_app.backend_worker_pool import BackendWorkerTransportError

    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))

    def fail_persistent(*_args, **_kwargs):
        raise BackendWorkerTransportError("cancelled transport")

    def fail_subprocess_run(*_args, **_kwargs):
        raise AssertionError("cancelled backend request must not start one-shot worker")

    monkeypatch.setattr(
        "eit_app.backend_worker_pool.run_persistent_backend_worker_request",
        fail_persistent,
    )
    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.subprocess.run",
        fail_subprocess_run,
    )

    with pytest.raises(InterruptedError):
        execute_forward_request_in_backend(
            ForwardSolverRequest(mesh_dimension=3, background_conductivity=1.0),
            profile="cuda",
            route_reason="cancelled",
            cancelled=lambda: True,
        )


def test_v146_cancelled_reconstruction_backend_transport_error_does_not_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    from eit_app.backend_worker_pool import BackendWorkerTransportError

    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))

    def fail_persistent(*_args, **_kwargs):
        raise BackendWorkerTransportError("cancelled transport")

    def fail_subprocess_run(*_args, **_kwargs):
        raise AssertionError("cancelled backend request must not start one-shot worker")

    monkeypatch.setattr(
        "eit_app.backend_worker_pool.run_persistent_backend_worker_request",
        fail_persistent,
    )
    monkeypatch.setattr(
        "eit_app.controllers.reconstruction_controller.subprocess.run",
        fail_subprocess_run,
    )
    frame = FrameData(
        real=np.array([1.0, 2.0], dtype=np.float32),
        imag=np.zeros(2, dtype=np.float32),
        timestamp=0.0,
        frame_index=0,
    )

    with pytest.raises(InterruptedError):
        execute_reconstruction_request_in_backend(
            ReconstructionRequest(
                reference_frame=frame,
                target_frame=frame,
                use_part="real",
                mesh_dimension=3,
            ),
            profile="cuda",
            route_reason="cancelled",
            cancelled=lambda: True,
        )


def test_v147_persistent_jit_timeout_repairs_cache_and_retries(
    monkeypatch,
    tmp_path,
) -> None:
    import eit_app.backend_worker_pool as pool
    from eit_app.backend_worker_pool import BackendWorkerRequestError

    repo = tmp_path / "repo"
    repo.mkdir()
    cache_home = tmp_path / "cache" / "v1" / "cuda" / "xdg-cache"
    fenics_cache = cache_home / "fenics"
    fenics_cache.mkdir(parents=True)
    stale_source = fenics_cache / "libffcx_forms_retry.c"
    stale_source.write_text("", encoding="utf-8")
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))
    key = (str(repo.resolve()), "cuda")
    attempts = {"count": 0, "stopped": 0}

    class _FakeWorker:
        def __init__(self, *, repo, profile) -> None:
            self.repo = repo
            self.profile = profile

        def run(self, **_kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise BackendWorkerRequestError(
                    "JIT compilation timed out, probably due to a failed previous "
                    f"compile. Try cleaning cache (e.g. remove {stale_source})"
                )
            return pool.WorkerRunMetadata(
                profile="cuda",
                cache_home=cache_home,
                launch_mode="current_python",
                pid=123,
                reused_process=False,
                stale_jit_locks_removed=0,
            )

        def request_stop(self) -> None:
            attempts["stopped"] += 1

        def shutdown(self) -> None:
            attempts["stopped"] += 1

    monkeypatch.setattr(pool, "_PersistentBackendWorker", _FakeWorker)
    with pool._POOL_LOCK:
        pool._POOL.pop(key, None)
    try:
        progress: list[str] = []

        meta = pool.run_persistent_backend_worker_request(
            repo=repo,
            profile="cuda",
            command="forward",
            input_path=tmp_path / "request.h5",
            output_path=tmp_path / "result.h5",
            progress_cb=progress.append,
        )

        assert meta.pid == 123
        assert attempts == {"count": 2, "stopped": 1}
        assert not stale_source.exists()
        assert any("Recovered backend FFCx JIT cache" in item for item in progress)
    finally:
        with pool._POOL_LOCK:
            pool._POOL.pop(key, None)


def test_v147_persistent_solver_error_is_not_retried(monkeypatch, tmp_path) -> None:
    import eit_app.backend_worker_pool as pool
    from eit_app.backend_worker_pool import BackendWorkerRequestError

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))
    key = (str(repo.resolve()), "cuda")
    attempts = {"count": 0}

    class _FakeWorker:
        def __init__(self, *, repo, profile) -> None:
            self.repo = repo
            self.profile = profile

        def run(self, **_kwargs):
            attempts["count"] += 1
            raise BackendWorkerRequestError("real solver error")

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(pool, "_PersistentBackendWorker", _FakeWorker)
    with pool._POOL_LOCK:
        pool._POOL.pop(key, None)
    try:
        with pytest.raises(BackendWorkerRequestError, match="real solver error"):
            pool.run_persistent_backend_worker_request(
                repo=repo,
                profile="cuda",
                command="forward",
                input_path=tmp_path / "request.h5",
                output_path=tmp_path / "result.h5",
            )

        assert attempts == {"count": 1}
    finally:
        with pool._POOL_LOCK:
            pool._POOL.pop(key, None)


def test_v148_persistent_worker_warmup_starts_process_without_request(
    monkeypatch,
    tmp_path,
) -> None:
    import eit_app.backend_worker_pool as pool

    repo = tmp_path / "repo"
    repo.mkdir()
    key = (str(repo.resolve()), "cuda")
    calls: list[str] = []

    class _FakeWorker:
        def __init__(self, *, repo, profile) -> None:
            self.repo = repo
            self.profile = profile

        def warm(self, *, progress_cb=None):
            calls.append("warm")
            if progress_cb is not None:
                progress_cb("started")
            return pool.WorkerRunMetadata(
                profile="cuda",
                cache_home=tmp_path / "cache",
                launch_mode="current_python",
                pid=456,
                reused_process=False,
                stale_jit_locks_removed=0,
            )

        def run(self, **_kwargs):
            raise AssertionError("warmup must not execute a solver request")

    monkeypatch.setattr(pool, "_PersistentBackendWorker", _FakeWorker)
    with pool._POOL_LOCK:
        pool._POOL.pop(key, None)
    try:
        progress: list[str] = []

        meta = pool.warm_persistent_backend_worker(
            repo=repo,
            profile="cuda",
            progress_cb=progress.append,
        )

        assert meta is not None
        assert meta.pid == 456
        assert calls == ["warm"]
        assert progress == ["started"]
    finally:
        with pool._POOL_LOCK:
            pool._POOL.pop(key, None)


def test_v194_backend_worker_serve_prime_runtime_uses_no_solver_io(
    monkeypatch,
    capsys,
) -> None:
    import eit_app.backend_worker as worker

    monkeypatch.setattr(
        worker,
        "_prime_runtime",
        lambda: {"modules": ["pyeidors.forward.eit_forward_model"]},
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "id": "prime-1",
                    "command": "prime_runtime",
                },
                sort_keys=True,
            )
            + "\n"
        ),
    )

    assert worker._serve(SimpleNamespace()) == 0

    messages = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert messages == [
        {
            "id": "prime-1",
            "metadata": {"modules": ["pyeidors.forward.eit_forward_model"]},
            "status": "ok",
            "type": "done",
        }
    ]


def test_v327_backend_worker_prime_runtime_warms_petsc_cuda_probe(monkeypatch) -> None:
    import eit_app.backend_worker as worker
    from pyeidors.forward import complex_support
    from pyeidors.perf import capabilities as perf_caps

    imported: list[str] = []
    monkeypatch.setattr(
        worker.importlib,
        "import_module",
        lambda name: imported.append(str(name)) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        complex_support,
        "runtime_scalar_summary",
        lambda: {"scalar_type": "complex64"},
    )
    monkeypatch.setattr(
        perf_caps,
        "probe_mpi_runtime",
        lambda: {"mpi_available": True, "mpi_size": 1},
    )
    monkeypatch.setattr(
        perf_caps,
        "probe_petsc_cuda_runtime",
        lambda: {"petsc_cuda": True, "probe_cache": {"hit": True}},
    )

    metadata = worker._prime_runtime()

    assert "pyeidors.perf.capabilities" in imported
    assert metadata["petsc_cuda_probe"]["petsc_cuda"] is True
    assert metadata["petsc_cuda_probe"]["probe_cache"]["hit"] is True
    assert metadata["mpi"]["mpi_available"] is True
    assert metadata["scalar"]["scalar_type"] == "complex64"


def test_v195_backend_worker_import_is_lazy_light() -> None:
    code = """
import json
import sys
import eit_app.backend_worker
sentinels = [
    "eit_app.backend_worker_protocol",
    "eit_app.controllers.forward_solver_controller",
    "eit_app.controllers.reconstruction_controller",
    "pyeidors.core_system",
    "pyeidors.forward.eit_forward_model",
]
print(json.dumps({name: name in sys.modules for name in sentinels}, sort_keys=True))
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=True,
    )

    loaded = json.loads(proc.stdout)
    assert loaded == {
        "eit_app.backend_worker_protocol": False,
        "eit_app.controllers.forward_solver_controller": False,
        "eit_app.controllers.reconstruction_controller": False,
        "pyeidors.core_system": False,
        "pyeidors.forward.eit_forward_model": False,
    }


def test_v199_backend_worker_protocol_import_is_lazy_light() -> None:
    code = """
import json
import sys
import eit_app.backend_worker_protocol
sentinels = [
    "eit_app.controllers.forward_solver_controller",
    "eit_app.controllers.reconstruction_controller",
    "pyeidors.core_system",
    "pyeidors.forward.eit_forward_model",
]
print(json.dumps({name: name in sys.modules for name in sentinels}, sort_keys=True))
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=True,
    )

    loaded = json.loads(proc.stdout)
    assert loaded == {
        "eit_app.controllers.forward_solver_controller": False,
        "eit_app.controllers.reconstruction_controller": False,
        "pyeidors.core_system": False,
        "pyeidors.forward.eit_forward_model": False,
    }


def test_v196_backend_worker_protocol_uses_lzf_for_array_payloads(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_HDF5_COMPRESSION", raising=False)
    output = tmp_path / "forward_result.h5"

    write_forward_result(
        output,
        ForwardSolverResult(
            boundary_voltages=np.arange(8, dtype=np.float32),
            ground_truth_conductivity=np.arange(4, dtype=np.float32),
            node_coords=np.arange(12, dtype=np.float32).reshape(4, 3),
            cell_connectivity=np.arange(8, dtype=np.int32).reshape(2, 4),
            n_elements=4,
            n_measurements=8,
            homogeneous_voltages=np.arange(8, dtype=np.float32),
            forward_model_config={},
        ),
    )

    with h5py.File(output, "r") as handle:
        assert handle["node_coords"].compression == "lzf"
        assert handle["node_coords"].attrs["compression"] == "lzf"
        assert handle["node_coords"].shuffle is True
        assert bool(handle["node_coords"].attrs["shuffle"]) is True
        assert handle["cell_connectivity"].compression == "lzf"
        assert handle["cell_connectivity"].shuffle is True
        assert handle["boundary_voltages"].compression == "lzf"
        assert handle["boundary_voltages"].shuffle is True


def test_v196_backend_worker_protocol_compression_can_be_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_HDF5_COMPRESSION", "off")
    output = tmp_path / "reconstruction_result.h5"

    write_reconstruction_result(
        output,
        ReconstructionResult(
            conductivity=np.arange(4, dtype=np.float32),
            node_coords=np.arange(12, dtype=np.float32).reshape(4, 3),
            cell_connectivity=np.arange(8, dtype=np.int32).reshape(2, 4),
            measured=np.arange(8, dtype=np.float32),
            simulated=np.arange(8, dtype=np.float32),
            metadata={},
        ),
    )

    with h5py.File(output, "r") as handle:
        assert handle["conductivity"].compression is None
        assert handle["conductivity"].attrs["compression"] == "none"
        assert handle["conductivity"].shuffle is False
        assert bool(handle["conductivity"].attrs["shuffle"]) is False
        assert handle["node_coords"].compression is None


def test_v333_backend_worker_protocol_shuffle_can_be_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_HDF5_COMPRESSION", raising=False)
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_HDF5_SHUFFLE", "off")
    output = tmp_path / "forward_result.h5"

    write_forward_result(
        output,
        ForwardSolverResult(
            boundary_voltages=np.arange(8, dtype=np.float32),
            ground_truth_conductivity=np.arange(4, dtype=np.float32),
            node_coords=np.arange(12, dtype=np.float32).reshape(4, 3),
            cell_connectivity=np.arange(8, dtype=np.int32).reshape(2, 4),
            n_elements=4,
            n_measurements=8,
            homogeneous_voltages=np.arange(8, dtype=np.float32),
            forward_model_config={},
        ),
    )

    with h5py.File(output, "r") as handle:
        assert handle["node_coords"].compression == "lzf"
        assert handle["node_coords"].shuffle is False
        assert bool(handle["node_coords"].attrs["shuffle"]) is False


def test_v335_backend_worker_protocol_uses_row_major_chunks(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_HDF5_COMPRESSION", raising=False)
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_HDF5_CHUNK_BYTES", "32")
    output = tmp_path / "forward_result.h5"

    write_forward_result(
        output,
        ForwardSolverResult(
            boundary_voltages=np.arange(16, dtype=np.float32),
            ground_truth_conductivity=np.arange(10, dtype=np.float32),
            node_coords=np.arange(30, dtype=np.float32).reshape(10, 3),
            cell_connectivity=np.arange(40, dtype=np.int32).reshape(10, 4),
            n_elements=10,
            n_measurements=16,
            homogeneous_voltages=np.arange(16, dtype=np.float32),
            forward_model_config={},
        ),
    )

    with h5py.File(output, "r") as handle:
        assert handle["boundary_voltages"].chunks == (8,)
        assert handle["node_coords"].chunks == (2, 3)
        assert handle["cell_connectivity"].chunks == (2, 4)
        assert int(handle["node_coords"].attrs["chunk_bytes_target"]) == 32
        np.testing.assert_array_equal(
            handle["node_coords"].attrs["chunk_shape"],
            np.asarray([2, 3], dtype=np.int64),
        )


def test_v336_backend_worker_protocol_reads_arrays_directly() -> None:
    class _FakeDataset:
        shape = (2, 3)
        dtype = np.dtype("float32")

        def __init__(self) -> None:
            self.read_direct_called = False

        def read_direct(self, out: np.ndarray) -> None:
            self.read_direct_called = True
            out[...] = np.arange(6, dtype=np.float32).reshape(2, 3)

    dataset = _FakeDataset()

    loaded = _read_dataset_array(dataset)

    assert dataset.read_direct_called is True
    assert loaded.flags.c_contiguous is True
    np.testing.assert_array_equal(
        loaded,
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )


def test_v334_forward_result_omits_absent_optional_dataset(tmp_path) -> None:
    output = tmp_path / "forward_result.h5"

    write_forward_result(
        output,
        ForwardSolverResult(
            boundary_voltages=np.arange(8, dtype=np.float32),
            ground_truth_conductivity=np.arange(4, dtype=np.float32),
            node_coords=np.arange(12, dtype=np.float32).reshape(4, 3),
            cell_connectivity=np.arange(8, dtype=np.int32).reshape(2, 4),
            n_elements=4,
            n_measurements=8,
            homogeneous_voltages=None,
            forward_model_config={},
        ),
    )

    with h5py.File(output, "r") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        assert metadata["has_homogeneous_voltages"] is False
        assert "homogeneous_voltages" not in handle

    loaded = read_forward_result(output)

    assert loaded.homogeneous_voltages is None
    np.testing.assert_array_equal(
        loaded.boundary_voltages,
        np.arange(8, dtype=np.float32),
    )


def test_v334_reconstruction_result_omits_absent_optional_datasets(tmp_path) -> None:
    output = tmp_path / "reconstruction_result.h5"

    write_reconstruction_result(
        output,
        ReconstructionResult(
            conductivity=np.arange(4, dtype=np.float32),
            node_coords=np.arange(12, dtype=np.float32).reshape(4, 3),
            cell_connectivity=np.arange(8, dtype=np.int32).reshape(2, 4),
            measured=None,
            simulated=None,
            metadata={},
        ),
    )

    with h5py.File(output, "r") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
        assert metadata["has_measured"] is False
        assert metadata["has_simulated"] is False
        assert "measured" not in handle
        assert "simulated" not in handle

    loaded = read_reconstruction_result(output)

    assert loaded.measured is None
    assert loaded.simulated is None
    np.testing.assert_array_equal(loaded.conductivity, np.arange(4, dtype=np.float32))


def test_v194_persistent_worker_warmup_primes_runtime_once(
    monkeypatch,
    tmp_path,
) -> None:
    import eit_app.backend_worker_pool as pool

    worker = pool._PersistentBackendWorker(repo=tmp_path, profile="cuda")
    worker._proc = SimpleNamespace(pid=321, poll=lambda: None)
    worker._cache = SimpleNamespace(
        xdg_cache_home=tmp_path / "cache",
        removed_stale_jit_locks=[],
    )
    worker._launch_mode = "unit"
    sent: list[dict[str, object]] = []

    def _fake_send_payload(*, proc, payload, progress_cb=None):
        assert proc is worker._proc
        sent.append(dict(payload))
        return {
            "id": payload["id"],
            "type": "done",
            "status": "ok",
            "metadata": {"modules": ["pyeidors.core_system"]},
        }

    monkeypatch.delenv("EIT_APP_BACKEND_WORKER_WARM_PRIME", raising=False)
    monkeypatch.setattr(worker, "_send_payload", _fake_send_payload)

    progress: list[str] = []
    first = worker.warm(progress_cb=progress.append)
    second = worker.warm(progress_cb=progress.append)

    assert [item["command"] for item in sent] == ["prime_runtime"]
    assert "input" not in sent[0]
    assert "output" not in sent[0]
    assert first.primed_runtime is True
    assert first.prime_command == "prime_runtime"
    assert first.prime_duration_ms >= 0.0
    assert first.prime_metadata == {"modules": ["pyeidors.core_system"]}
    assert second.primed_runtime is True
    assert second.prime_command == "prime_runtime"
    assert second.prime_duration_ms == 0.0
    assert any("Primed backend worker runtime" in item for item in progress)


def test_v194_persistent_worker_warmup_prime_can_be_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    import eit_app.backend_worker_pool as pool

    worker = pool._PersistentBackendWorker(repo=tmp_path, profile="cuda")
    worker._proc = SimpleNamespace(pid=654, poll=lambda: None)
    worker._cache = SimpleNamespace(
        xdg_cache_home=tmp_path / "cache",
        removed_stale_jit_locks=[],
    )
    worker._launch_mode = "unit"

    def _fail_send_payload(**_kwargs):
        raise AssertionError("disabled warm prime must not send a worker command")

    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_WARM_PRIME", "0")
    monkeypatch.setattr(worker, "_send_payload", _fail_send_payload)

    meta = worker.warm()

    assert meta.primed_runtime is False
    assert meta.prime_command == ""
    assert meta.prime_duration_ms == 0.0


def test_v316_persistent_worker_warmup_recycles_after_rss_budget(
    monkeypatch,
    tmp_path,
) -> None:
    import eit_app.backend_worker_pool as pool

    class _WarmProc:
        pid = 7654

        def __init__(self) -> None:
            self.terminated = False
            self._returncode: int | None = None

        def poll(self) -> int | None:
            return self._returncode

        def terminate(self) -> None:
            self.terminated = True
            self._returncode = -15

        def kill(self) -> None:
            self.terminated = True
            self._returncode = -9

        def wait(self, timeout=None) -> int:
            if self._returncode is None:
                self._returncode = 0
            return self._returncode

    rss_bytes = 3 * 1024 * 1024
    worker = pool._PersistentBackendWorker(repo=tmp_path, profile="cuda")
    proc = _WarmProc()
    worker._proc = proc
    worker._cache = SimpleNamespace(
        xdg_cache_home=tmp_path / "cache",
        removed_stale_jit_locks=[],
    )
    worker._launch_mode = "unit"
    worker._runtime_primed = True

    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_MAX_RSS_MB", "1")
    monkeypatch.setattr(pool, "_process_rss_bytes", lambda _pid: rss_bytes)

    progress: list[str] = []
    meta = worker.warm(progress_cb=progress.append)

    assert meta.rss_bytes == rss_bytes
    assert meta.rss_limit_bytes == 1024 * 1024
    assert meta.recycled_after_request is True
    assert meta.recycle_reason == "rss_budget_exceeded"
    assert proc.terminated is True
    assert worker._proc is None
    assert any("warm RSS exceeded budget" in item for item in progress)


def test_v187_persistent_worker_recycles_after_rss_budget(
    monkeypatch,
    tmp_path,
) -> None:
    import eit_app.backend_worker_pool as pool

    repo = tmp_path / "repo"
    repo.mkdir()
    key = (str(repo.resolve()), "cuda")
    cache_home = tmp_path / "cache"
    cache_home.mkdir()
    rss_bytes = 2 * 1024 * 1024
    instances: list[object] = []

    class _FakeStdin:
        def __init__(self, proc) -> None:
            self._proc = proc

        def write(self, text: str) -> int:
            self._proc.request_id = json.loads(text)["id"]
            return len(text)

        def flush(self) -> None:
            return None

    class _FakeStdout:
        def __init__(self, proc) -> None:
            self._proc = proc
            self._sent = False

        def readline(self) -> str:
            if self._sent:
                return ""
            request_id = self._proc.request_id
            if not request_id:
                return ""
            self._sent = True
            return (
                json.dumps(
                    {
                        "id": request_id,
                        "type": "done",
                        "status": "ok",
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    class _FakeStderr:
        def __iter__(self):
            return iter(())

    class _FakeProc:
        def __init__(self) -> None:
            self.pid = 9000 + len(instances)
            self.request_id = ""
            self.stdin = _FakeStdin(self)
            self.stdout = _FakeStdout(self)
            self.stderr = _FakeStderr()
            self.terminated = False
            self._returncode: int | None = None

        def poll(self) -> int | None:
            return self._returncode

        def terminate(self) -> None:
            self.terminated = True
            self._returncode = -15

        def kill(self) -> None:
            self.terminated = True
            self._returncode = -9

        def wait(self, timeout=None) -> int:
            if self._returncode is None:
                self._returncode = 0
            return self._returncode

    def _fake_popen(*_args, **_kwargs):
        proc = _FakeProc()
        instances.append(proc)
        return proc

    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_MAX_RSS_MB", "1")
    monkeypatch.setattr(
        pool,
        "backend_worker_env",
        lambda *, repo, profile: (
            {},
            SimpleNamespace(
                xdg_cache_home=cache_home,
                removed_stale_jit_locks=[],
            ),
        ),
    )
    monkeypatch.setattr(
        pool,
        "backend_worker_command",
        lambda *, profile, worker_args: (["fake-worker"], "unit"),
    )
    monkeypatch.setattr(pool, "_process_rss_bytes", lambda _pid: rss_bytes)
    monkeypatch.setattr(pool.subprocess, "Popen", _fake_popen)
    with pool._POOL_LOCK:
        pool._POOL.pop(key, None)
    try:
        progress: list[str] = []
        meta = pool.run_persistent_backend_worker_request(
            repo=repo,
            profile="cuda",
            command="forward",
            input_path=tmp_path / "request.h5",
            output_path=tmp_path / "result.h5",
            progress_cb=progress.append,
        )

        assert meta.rss_bytes == rss_bytes
        assert meta.rss_limit_bytes == 1024 * 1024
        assert meta.recycled_after_request is True
        assert meta.recycle_reason == "rss_budget_exceeded"
        assert instances[0].terminated is True
        assert any("RSS exceeded budget" in item for item in progress)

        pool.run_persistent_backend_worker_request(
            repo=repo,
            profile="cuda",
            command="forward",
            input_path=tmp_path / "request2.h5",
            output_path=tmp_path / "result2.h5",
        )
        assert len(instances) == 2
    finally:
        with pool._POOL_LOCK:
            worker = pool._POOL.pop(key, None)
        if worker is not None:
            worker.request_stop()


def test_v137_backend_worker_cache_is_profile_scoped_and_prunes_stale_jit_locks(
    monkeypatch,
    tmp_path,
) -> None:
    cache_root = tmp_path / "backend-cache"
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_STALE_JIT_LOCK_SECONDS", "0")
    repo = tmp_path / "repo"
    repo.mkdir()
    fenics_cache = cache_root / "v1" / "cuda" / "xdg-cache" / "fenics"
    fenics_cache.mkdir(parents=True)
    stale_lock = fenics_cache / "libffcx_forms_stale.c"
    stale_lock.write_text("stale", encoding="utf-8")
    compiled_source = fenics_cache / "libffcx_forms_compiled.c"
    compiled_source.write_text("compiled", encoding="utf-8")
    compiled_source.with_suffix(".so").write_text("module", encoding="utf-8")
    orphan_ready = fenics_cache / "libffcx_forms_orphan.c.cached"
    orphan_ready.write_text("orphan", encoding="utf-8")
    ready_with_module = fenics_cache / "libffcx_forms_ready.c.cached"
    ready_with_module.write_text("ready", encoding="utf-8")
    ready_with_module.with_name(
        "libffcx_forms_ready.cpython-313-x86_64-linux-gnu.so"
    ).write_text("module", encoding="utf-8")

    env, cache = backend_worker_env(repo=repo, profile="cuda")
    env_again, cache_again = backend_worker_env(repo=repo, profile="cuda")

    expected_home = cache_root / "v1" / "cuda" / "xdg-cache"
    assert cache.xdg_cache_home == expected_home
    assert cache_again.xdg_cache_home == expected_home
    assert env["XDG_CACHE_HOME"] == str(expected_home)
    assert env_again["XDG_CACHE_HOME"] == str(expected_home)
    assert env["PYEIDORS_ENV_SYNC_CACHE"] == "1"
    assert env["PYEIDORS_ENV_SYNC_CACHE_TTL_SECONDS"] == "43200"
    assert env["PYEIDORS_PETSC_CUDA_PROBE_CACHE"] == "1"
    assert env["PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR"] == str(
        expected_home / "pyeidors-capabilities"
    )
    assert stale_lock in cache.removed_stale_jit_locks
    assert orphan_ready in cache.removed_stale_jit_locks
    assert not stale_lock.exists()
    assert not orphan_ready.exists()
    assert compiled_source.exists()
    assert compiled_source.with_suffix(".so").exists()
    assert compiled_source.with_suffix(".c.cached").exists()
    assert ready_with_module.exists()
    assert ready_with_module.with_suffix("").exists()


def test_v666_backend_worker_env_marks_cuda_amgx_as_gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))
    repo = tmp_path / "repo"
    repo.mkdir()

    env, cache = backend_worker_env(repo=repo, profile="cuda-amgx")

    assert cache.profile == "cuda-amgx"
    assert env["EIT_APP_GUI_RUNTIME_PROFILE"] == "cuda-amgx"
    assert env["EIT_APP_GUI_PROFILE"] == "gpu"


def test_sm61_backend_worker_env_marks_legacy_cuda_as_gpu(monkeypatch, tmp_path):
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(tmp_path / "cache"))
    repo = tmp_path / "repo"
    repo.mkdir()

    env, cache = backend_worker_env(repo=repo, profile="complex64-cuda-sm61")

    assert cache.profile == "complex64-cuda-sm61"
    assert env["EIT_APP_GUI_RUNTIME_PROFILE"] == "complex64-cuda-sm61"
    assert env["EIT_APP_GUI_PROFILE"] == "gpu"
    assert env["EIT_APP_GUI_PRECISION"] == "complex64"


def test_v590_backend_worker_jit_cleanup_indexes_compiled_modules(tmp_path) -> None:
    import eit_app.backend_worker_runtime as runtime

    fenics_cache = tmp_path / "fenics"
    fenics_cache.mkdir()
    (fenics_cache / "libffcx_forms_plain.so").write_text("module", encoding="utf-8")
    (fenics_cache / "libffcx_forms_tagged.cpython-313-x86_64-linux-gnu.so").write_text(
        "module",
        encoding="utf-8",
    )

    assert runtime._compiled_ffcx_module_stems(fenics_cache) == {
        "libffcx_forms_plain",
        "libffcx_forms_tagged",
    }

    source = inspect.getsource(runtime.cleanup_stale_ffcx_jit_locks)
    assert "compiled_stems = _compiled_ffcx_module_stems(fenics_cache)" in source
    assert "_compiled_ffcx_module_exists(fenics_cache" not in source


def test_v139_inprocess_runtime_uses_project_profile_jit_cache(
    monkeypatch,
    tmp_path,
) -> None:
    cache_root = tmp_path / "backend-cache"
    monkeypatch.setenv("EIT_APP_BACKEND_WORKER_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.delenv("PYEIDORS_PETSC_CUDA_PROBE_CACHE", raising=False)
    monkeypatch.delenv("PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()

    cache = prepare_inprocess_backend_runtime(repo=repo)

    expected = cache_root / "v1" / "complex64-cuda" / "xdg-cache"
    assert cache.xdg_cache_home == expected
    assert cache.xdg_cache_home.exists()
    assert "PYEIDORS_PETSC_CUDA_PROBE_CACHE" not in os.environ
    assert "PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR" not in os.environ


def test_v145_inprocess_forward_uses_profile_lock(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setattr(forward_controller_module, "_repo_root", lambda: tmp_path)
    calls: list[tuple[str, Path, str]] = []

    class _Lock:
        def __init__(self, repo: Path, profile: str) -> None:
            self.repo = repo
            self.profile = profile

        def __enter__(self):
            calls.append(("enter", self.repo, self.profile))
            return self

        def __exit__(self, *_args):
            calls.append(("exit", self.repo, self.profile))

    monkeypatch.setattr(
        "eit_app.backend_worker_runtime.backend_worker_profile_lock",
        lambda repo, profile: _Lock(Path(repo), str(profile)),
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_execute_forward_request_unlocked",
        lambda *_args, **_kwargs: "ok",
    )

    assert execute_forward_request(ForwardSolverRequest()) == "ok"
    assert calls == [
        ("enter", tmp_path, "complex64-cuda"),
        ("exit", tmp_path, "complex64-cuda"),
    ]


def test_v145_backend_child_skips_parent_held_profile_lock(monkeypatch) -> None:
    monkeypatch.setenv("EIT_APP_BACKEND_PROFILE_LOCK_HELD", "1")
    monkeypatch.setattr(
        "eit_app.backend_worker_runtime.backend_worker_profile_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parent-held lock must not be reacquired")
        ),
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_execute_forward_request_unlocked",
        lambda *_args, **_kwargs: "ok",
    )

    assert execute_forward_request(ForwardSolverRequest()) == "ok"


def test_v323_forward_setup_prime_uses_profile_lock(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "cuda")
    monkeypatch.setattr(forward_controller_module, "_repo_root", lambda: tmp_path)
    calls: list[tuple[str, Path, str]] = []

    class _Lock:
        def __init__(self, repo: Path, profile: str) -> None:
            self.repo = repo
            self.profile = profile

        def __enter__(self):
            calls.append(("enter", self.repo, self.profile))
            return self

        def __exit__(self, *_args):
            calls.append(("exit", self.repo, self.profile))

    monkeypatch.setattr(
        "eit_app.backend_worker_runtime.backend_worker_profile_lock",
        lambda repo, profile: _Lock(Path(repo), str(profile)),
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_prime_forward_setup_request_unlocked",
        lambda *_args, **_kwargs: {"ok": True},
    )

    assert forward_controller_module.prime_forward_setup_request(
        ForwardSolverRequest(mesh_dimension=3)
    ) == {"ok": True}
    assert calls == [
        ("enter", tmp_path, "cuda"),
        ("exit", tmp_path, "cuda"),
    ]


def test_v323_forward_setup_prime_skips_parent_held_profile_lock(monkeypatch) -> None:
    monkeypatch.setenv("EIT_APP_BACKEND_PROFILE_LOCK_HELD", "1")
    monkeypatch.setattr(
        "eit_app.backend_worker_runtime.backend_worker_profile_lock",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parent-held lock must not be reacquired")
        ),
    )
    monkeypatch.setattr(
        forward_controller_module,
        "_prime_forward_setup_request_unlocked",
        lambda *_args, **_kwargs: {"ok": True},
    )

    assert forward_controller_module.prime_forward_setup_request(
        ForwardSolverRequest(mesh_dimension=3)
    ) == {"ok": True}
