# Core runtime and public API invariants

Authority: registered by `docs/spec/registry.json`; IDs remain stable and semantics are copied verbatim from the former monolithic §V.

| id | invariant | source |
|----|-----------|--------|
| V71 | Bucket all-modes noise gradient exp ! reuse V70 `add_noise`; same 7 recon modes as `仿真各情况全量测试`; deterministic seeds; ∀ SNR row records actual SNR, sigma RMSE, artifact, direct delta vs `full_208`; output isolated folder | future `scripts/eit_bucket_all_modes_noise_sweep.py`; V70 |
| V640 | `safe_dot` strict finite contract ! finite inputs that overflow/nonfinite during dot raise `FloatingPointError` from project guard; NumPy `RuntimeWarning` must not escape under warnings-as-errors before result finite check | src/pyeidors/utils/numeric_ops.py; tests/unit/test_coverage_init_utils.py; tests/unit/test_numeric_ops_warning_free.py; B538 |
| V665 | Complex GPU-only route comparison ! every route records explicit true residual `||Ax-b||/||b||`; reports gate residual separately from route-vs-route voltage/solution deltas; native CUDA sparse `3d_gamg` convergence reason ⊥ accepted as correctness proof when dense fallback skipped | scripts/diagnostics/complex_block_real_amgx_probe.py; scripts/diagnostics/complex_route_speed_accuracy_compare.py; tests/unit/test_complex_block_real_amgx_probe.py; tests/unit/test_complex_route_speed_accuracy_compare.py; B564 |
| V684 | Same mesh/order/scalar/σ/z/pattern direct solve classic-vs-Robin parity ! finite `u/U/meas`; relative L2 ≤`2e-5` for float32/complex64, ≤`1e-10` for float64/complex128; `sum(I)` & `sum(U)` residual bounded; default `cem_formulation=classic` byte-compatible API behavior | tests/unit/test_robin_transconductance_cem.py; tests/integration/test_robin_transconductance_cem_integration.py |
| V778 | Package entrypoint surface ! `pyeidors` exposes only §I-declared façade/console surfaces; placeholder `src/pyeidors/main.py` hello entrypoint + coverage-only tests ⊥ shipped. `python -m pyeidors` remains unsupported unless future §I explicitly adds it | pyproject.toml; src/pyeidors/__init__.py; tests/unit/test_public_api_and_launcher_contract.py |
