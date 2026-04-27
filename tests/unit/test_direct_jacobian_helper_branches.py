"""Additional branch coverage for direct Jacobian helper logic."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pyeidors.inverse.jacobian._core as core_module
import pyeidors.inverse.jacobian.base_jacobian as base_module
import pyeidors.inverse.jacobian.direct_jacobian as direct_module
from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator


class _InterpFunction:
    def __init__(self, space):
        self.space = space
        size = 2 if getattr(space, "name", "") == "V" else 4
        self.x = SimpleNamespace(array=np.zeros(size, dtype=float))

    def interpolate(self, expr):
        base = float(expr.grad_value)
        self.x.array[:] = np.array(
            [base, base + 1.0, base + 2.0, base + 3.0], dtype=float
        )


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=float)

    def to(self, *_args, **_kwargs):
        return self

    def unsqueeze(self, axis):
        return _FakeTensor(np.expand_dims(self.array, axis))

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.array, dtype=float)

    def numel(self):
        return int(self.array.size)

    def __getitem__(self, item):
        return _FakeTensor(self.array[item])

    def __mul__(self, other):
        other_array = other.array if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.array * other_array)

    __rmul__ = __mul__


class _FakeTorch:
    float64 = "float64"

    class cuda:
        @staticmethod
        def is_available():
            return True

    @staticmethod
    def from_numpy(array):
        return _FakeTensor(array)

    @staticmethod
    def einsum(pattern, left, right):
        return _FakeTensor(np.einsum(pattern, left.array, right.array, optimize=True))


def _make_calc() -> DirectJacobianCalculator:
    calc = DirectJacobianCalculator.__new__(DirectJacobianCalculator)
    calc.block_tune_mode = "auto"
    calc.block_size = 0
    calc.block_candidates = (32, 64, 128)
    calc._resolved_block_size = None
    calc._block_tune_source = "unset"
    calc._last_assembly_elapsed_only = 0.0
    calc._runtime_device_requested = "auto"
    calc._runtime_device_effective = "cpu"
    calc._runtime_cuda_device = "cuda:0"
    calc._jacobian_backend_requested = "auto"
    calc._jacobian_backend_effective = "cpu"
    calc._jacobian_block_backend = "numpy"
    calc._jacobian_transfer_estimate = 0.0
    calc._jacobian_cuda_threshold_hit = False
    calc._cell_areas_cuda = None
    calc.gdim = 2
    calc.cell_areas = np.array([1.0, 2.0], dtype=float)
    calc.fwd_model = SimpleNamespace(
        n_elec=2,
        forward_solve=lambda sigma, current_patterns=None: (
            [np.array([1.0, 2.0], dtype=float)],
            None,
        ),
        pattern_manager=SimpleNamespace(
            n_meas_total=3,
            n_stim=2,
            n_meas_per_stim=[2, 1],
            meas_matrices=[
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
                np.array([[1.0, -1.0]], dtype=float),
            ],
        ),
        cache_manager=None,
    )
    calc.Q_DG = SimpleNamespace(
        element=SimpleNamespace(
            interpolation_points=np.array([[0.0, 0.0]], dtype=float)
        )
    )
    calc.V = SimpleNamespace(name="V")
    return calc


def test_init_and_runtime_device_configuration_cover_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        DirectJacobianCalculator,
        "_setup_computation",
        lambda self: setattr(self, "cell_areas", np.ones(2, dtype=float)),
    )
    monkeypatch.setattr(
        base_module.fem,
        "Function",
        lambda _space: SimpleNamespace(
            x=SimpleNamespace(array=np.zeros(3, dtype=float))
        ),
    )
    fake_fwd_model = SimpleNamespace(
        V_sigma=SimpleNamespace(), pattern_manager=SimpleNamespace(n_meas_total=5)
    )
    with pytest.raises(ValueError, match="Unsupported block_tune_mode"):
        DirectJacobianCalculator(fake_fwd_model, block_tune_mode="bad")

    calc = DirectJacobianCalculator(
        fake_fwd_model,
        block_candidates=(0, -1),
        runtime_device="cpu",
    )
    assert calc.block_candidates == (64, 128, 256, 512)
    calc.set_runtime_device("cuda", "cuda", torch_device="cuda:2")
    assert calc._runtime_cuda_device == "cuda:2"
    assert calc._jacobian_backend_requested == "cuda"


def test_block_size_calibration_variants_cover_fixed_small_and_cache_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    calc = _make_calc()
    calc._calibrate_block_size_once = lambda *_args, **_kwargs: 77

    calc.block_size = 40
    assert (
        calc._calibrate_block_size(grad_u_all=[1], adjoint_gradients=[1], n_elements=30)
        == 30
    )
    assert calc._block_tune_source == "fixed"

    calc.block_size = 0
    calc.block_tune_mode = "off"
    assert (
        calc._calibrate_block_size(grad_u_all=[1], adjoint_gradients=[1], n_elements=30)
        == 30
    )
    assert calc._block_tune_source == "disabled"

    calc.block_tune_mode = "auto"
    assert (
        calc._calibrate_block_size(
            grad_u_all=[1], adjoint_gradients=[1], n_elements=128
        )
        == 128
    )
    assert calc._block_tune_source == "small-problem"

    class _Lookup:
        hit = True
        layer = "disk"

    class _Cache:
        enabled = True

        def get_or_compute_semantic(self, **kwargs):
            return 96, _Lookup()

    calc.fwd_model.cache_manager = _Cache()
    monkeypatch.setattr(
        direct_module, "model_signature_from_forward_model", lambda _fm: "m"
    )
    monkeypatch.setattr(
        direct_module, "pattern_signature_from_forward_model", lambda _fm: "p"
    )
    monkeypatch.setattr(
        direct_module, "backend_signature_from_forward_model", lambda _fm: "b"
    )
    assert (
        calc._calibrate_block_size(
            grad_u_all=[1], adjoint_gradients=[1, 2], n_elements=300
        )
        == 96
    )
    assert calc._block_tune_source == "disk"

    calc._resolved_block_size = None
    calc._calibrate_block_size = lambda **_kwargs: 111
    assert calc._resolve_block_size([1], [1], 120) == 111
    assert calc._resolve_block_size([1], [1], 80) == 80


def test_calibrate_block_size_once_edge_cases_and_candidates():
    calc = _make_calc()
    calc.block_candidates = (32, 64)

    class _BadAdjoint:
        def __bool__(self):
            return True

        def __getitem__(self, _item):
            return np.ones((2, 2), dtype=float)

    assert calc._calibrate_block_size_once([], [], 200) == 64
    assert (
        calc._calibrate_block_size_once(
            [np.array([1.0, 2.0], dtype=float)],
            [np.array([1.0, 2.0], dtype=float)],
            200,
        )
        == 64
    )
    assert (
        calc._calibrate_block_size_once(
            [np.ones((10, 2), dtype=float)], _BadAdjoint(), 200
        )
        == 64
    )

    calc.block_candidates = ()
    calc.fwd_model.pattern_manager.n_meas_per_stim = [2]
    grad_u_all = [np.ones((300, 2), dtype=float)]
    adjoint_gradients = [np.ones((300, 2), dtype=float), np.ones((300, 2), dtype=float)]
    assert calc._calibrate_block_size_once(grad_u_all, adjoint_gradients, 400) == 256


def test_cuda_threshold_cell_area_cache_and_calculate_cache_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    calc = _make_calc()
    monkeypatch.setattr(direct_module, "torch", None)
    assert calc._get_cell_areas_cuda() is None

    monkeypatch.setattr(direct_module, "torch", _FakeTorch)
    calc.set_runtime_device("auto", "cuda")
    assert calc._wants_cuda_contraction() is True
    assert calc._should_use_cuda_contraction(n_measurements=1, n_elements=10) is False
    assert calc._jacobian_cuda_threshold_hit is False
    assert (
        calc._should_use_cuda_contraction(n_measurements=1000, n_elements=1024) is True
    )
    assert calc._jacobian_cuda_threshold_hit is True

    calc.set_runtime_device("cuda", "cuda")
    assert calc._should_use_cuda_contraction(n_measurements=1, n_elements=10) is True
    cached = calc._get_cell_areas_cuda()
    assert calc._get_cell_areas_cuda() is cached

    calc.block_tuning_info = lambda: {"selected_block_size": 64}
    calc._calculate_efficient = lambda sigma: np.array([[1.0]], dtype=float)
    calc._calculate_traditional = lambda sigma: np.array([[2.0]], dtype=float)
    with pytest.raises(ValueError, match="Unknown method"):
        calc.calculate(SimpleNamespace(), method="bad")

    calc.fwd_model.cache_manager = None
    out_no_cache = calc.calculate(SimpleNamespace(), method="efficient")
    np.testing.assert_allclose(out_no_cache, np.array([[1.0]], dtype=float))
    assert calc._last_block_tune_info == {"selected_block_size": 64}

    class _Lookup:
        hit = False
        layer = "compute"
        artifact = "jacobian"
        key = "jac-key"

    class _Cache:
        enabled = True

        def get_or_compute_semantic(self, **kwargs):
            return kwargs["compute_fn"](), _Lookup()

    calc.fwd_model.cache_manager = _Cache()
    monkeypatch.setattr(
        direct_module,
        "function_get_array",
        lambda _sigma: np.array([1.0, 2.0], dtype=float),
    )
    monkeypatch.setattr(
        direct_module, "model_signature_from_forward_model", lambda _fm: "m"
    )
    monkeypatch.setattr(
        direct_module, "pattern_signature_from_forward_model", lambda _fm: "p"
    )
    monkeypatch.setattr(
        direct_module, "backend_signature_from_forward_model", lambda _fm: "b"
    )
    out_cache = calc.calculate(SimpleNamespace(), method="traditional")
    np.testing.assert_allclose(out_cache, np.array([[2.0]], dtype=float))
    assert calc._last_cache_lookup["key"] == "jac-key"


def test_compute_gradient_patterns_and_calculation_wrappers(
    monkeypatch: pytest.MonkeyPatch,
):
    calc = _make_calc()
    calc.Q_DG = SimpleNamespace(
        element=SimpleNamespace(
            interpolation_points=lambda: np.array([[0.0, 0.0]], dtype=float)
        )
    )
    calc._geometry = SimpleNamespace(V=calc.V, Q_DG=calc.Q_DG, gdim=calc.gdim)
    monkeypatch.setattr(core_module.fem, "Function", _InterpFunction)
    monkeypatch.setattr(
        core_module.fem,
        "Expression",
        lambda grad_value, points: SimpleNamespace(
            grad_value=grad_value, points=points
        ),
    )
    monkeypatch.setattr(
        core_module.ufl, "grad", lambda u_fun: float(np.sum(u_fun.x.array))
    )

    grads = calc._compute_field_gradients(
        [np.array([1.0, 2.0], dtype=float), np.array([3.0, 4.0], dtype=float)]
    )
    np.testing.assert_allclose(
        grads[0], np.array([[3.0, 4.0], [5.0, 6.0]], dtype=float)
    )
    np.testing.assert_allclose(
        grads[1], np.array([[7.0, 8.0], [9.0, 10.0]], dtype=float)
    )

    patterns = calc._measurement_to_current_patterns()
    np.testing.assert_allclose(
        patterns,
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]], dtype=float),
    )

    calls = {"count": 0}

    def _forward_solve(_sigma, current_patterns=None):
        calls["count"] += 1
        if current_patterns is None:
            return [np.array([1.0, 2.0], dtype=float)], None
        return [
            np.array([3.0, 4.0], dtype=float),
            np.array([5.0, 6.0], dtype=float),
        ], None

    calc.fwd_model.forward_solve = _forward_solve
    calc._compute_field_gradients = lambda fields: (
        ["grad-u"] if calls["count"] == 1 else ["grad-adj"]
    )
    calc._assemble_jacobian_efficient = lambda grad_u_all, adjoint_fields: (
        "efficient",
        grad_u_all,
        adjoint_fields,
    )
    calc._assemble_jacobian_traditional = lambda grad_u_all, grad_bu_all: np.array(
        [[7.0, 8.0]], dtype=float
    )
    calc._convert_to_measurement_jacobian = lambda jacobian: jacobian + 1.0

    assert calc._calculate_efficient(SimpleNamespace())[0] == "efficient"
    traditional = calc._calculate_traditional(SimpleNamespace())
    np.testing.assert_allclose(traditional, np.array([[8.0, 9.0]], dtype=float))


def test_assembly_helpers_cover_cuda_and_traditional_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    calc = _make_calc()
    calc.cell_areas = np.array([1.5, 0.5], dtype=float)
    calc._resolve_block_size = lambda grad_u_all, adjoint_gradients, n_elements: 1
    calc._should_use_cuda_contraction = lambda **kwargs: True
    monkeypatch.setattr(direct_module, "torch", _FakeTorch)

    jac = calc._assemble_jacobian_efficient(
        [np.array([[1.0, 2.0], [0.5, 1.0]], dtype=float)],
        [np.array([[0.5, 1.0], [1.0, 1.5]], dtype=float)],
    )
    expected = np.array([[3.75, 1.0]], dtype=float)
    np.testing.assert_allclose(jac, expected)
    assert calc._jacobian_backend_effective == "cuda"
    assert calc._jacobian_block_backend == "torch-cuda"
    assert calc._jacobian_transfer_estimate > 0.0

    calc.fwd_model.pattern_manager.n_stim = 2
    calc.fwd_model.n_elec = 2
    calc.fwd_model.pattern_manager.meas_matrices = [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        np.array([[1.0, -1.0]], dtype=float),
    ]
    electrode_jacobian = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=float,
    )
    meas_jac = calc._convert_to_measurement_jacobian(electrode_jacobian)
    np.testing.assert_allclose(
        meas_jac,
        np.array([[1.0, 2.0], [3.0, 4.0], [-2.0, -2.0]], dtype=float),
    )

    traditional = calc._assemble_jacobian_traditional(
        [np.array([[1.0, 2.0], [0.5, 1.0]], dtype=float)],
        [
            np.array([[0.5, 1.0], [1.0, 1.5]], dtype=float),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
        ],
    )
    np.testing.assert_allclose(
        traditional,
        np.array([[3.75, 1.0], [1.5, 0.5]], dtype=float),
    )
