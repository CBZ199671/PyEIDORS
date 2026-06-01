from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY_SCRIPT = ROOT / "scripts" / "diagnostics" / "complex_eidors_pyeidors_step_compare.py"
M_SCRIPT = ROOT / "scripts" / "diagnostics" / "complex_eidors_pyeidors_step_compare.m"
VIS_SCRIPT = (
    ROOT
    / "scripts"
    / "diagnostics"
    / "render_complex_eidors_pyeidors_visual_compare.py"
)


def test_complex_step_compare_preserves_requested_complex_admittance_defaults() -> None:
    source = PY_SCRIPT.read_text(encoding="utf-8")

    assert "base_sigma: complex = 1.0 + 2.0j" in source
    assert "target_sigma: complex = 2.0 + 3.0j" in source
    assert "contact_impedance: complex = 0.01 + 0.05j" in source
    assert (
        '"base_sigma": np.asarray([[case.base_sigma]], dtype=np.complex128)' in source
    )
    assert (
        '"target_sigma": np.asarray([[case.target_sigma]], dtype=np.complex128)'
        in source
    )
    assert '"truth_elem_data": truth.reshape(-1, 1)' in source


def test_complex_step_compare_uses_shared_mesh_pattern_payload_for_eidors() -> None:
    py_source = PY_SCRIPT.read_text(encoding="utf-8")
    matlab_source = M_SCRIPT.read_text(encoding="utf-8")

    assert 'measurement_protocol: str = "eidors_full_3d"' in py_source
    assert '"stim_matrix": np.asarray(pm.stim_matrix, dtype=float)' in py_source
    assert (
        '"meas_matrix_concat": _stack_measurement_matrices(pm.meas_matrices)'
        in py_source
    )
    assert "fmdl.nodes = double(payload.nodes);" in matlab_source
    assert "fmdl.elems = double(payload.elems);" in matlab_source
    assert (
        "fmdl.stimulation = build_stimulation_from_payload(payload);" in matlab_source
    )
    assert "contact_z = payload.contact_impedance(1);" in matlab_source
    assert "truth_elem_data = payload.truth_elem_data(:);" in matlab_source


def test_complex_step_compare_reports_orientation_and_conjugate_diagnostics() -> None:
    source = PY_SCRIPT.read_text(encoding="utf-8")

    assert '"dv_raw_tmr"' in source
    assert '"dv_raw_rmt"' in source
    assert '"dv_norm_tmr"' in source
    assert '"dv_norm_rmt"' in source
    assert "candidate_is_negative_reference_rel_l2" in source
    assert "candidate_conjugate_rel_l2" in source
    assert "best_complex_scalar_fit" in source


def test_complex_visual_compare_renders_channel_and_phase_diagnostics() -> None:
    source = VIS_SCRIPT.read_text(encoding="utf-8")

    assert 'matplotlib.use("Agg")' in source
    assert '"font.family": "Times New Roman"' in source
    assert 'Channel("abs", "Magnitude |.|", "S/m", "viridis")' in source
    assert (
        'Channel("phase", "Phase angle", "rad", "twilight_shifted", (-np.pi, np.pi))'
        in source
    )
    assert "visual_compare_channels_xz.png" in source
    assert "visual_compare_differences_xz.png" in source
    assert "visual_compare_phase_zoom_xz.png" in source
