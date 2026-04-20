---
status: ready
source: map
build_site: brownfield-cavekit
---

# Build Site: Brownfield Cavekit Stabilization

## Objective

Turn the from-code sketch into a reliable brownfield Cavekit layer. This plan
does not rewrite PyEIDORS. It validates the generated kits, adds traceability
from source modules to kit requirements, and documents any gaps that require
future behavior changes.

## Task Registry

| ID | Depth | Depends On | Kits | Summary | Validation |
| --- | --- | --- | --- | --- | --- |
| T-CK-001 | quick | none | overview, validation report | Run pytest collect-only for cited smoke tests and update validation report with collect status. | `pytest --collect-only` on recommended smoke set |
| T-CK-002 | standard | T-CK-001 | all kits | Add concise `CLAUDE.md` source-to-kit mapping files for major source subtrees. | Diff review; mappings reference existing kit IDs |
| T-CORE-001 | standard | T-CK-001 | core-system R1-R4 | Validate core setup/facade behavior against existing tests and update kit evidence. | `pytest tests/unit/test_core_setup_contract.py tests/unit/test_workflow_wrapper_branches.py` |
| T-DATA-001 | standard | T-CK-001 | data-and-units R1-R4 | Validate measurement dataset, frame I/O, drive semantics, and unit checks. | `pytest tests/unit/test_measurement_dataset.py tests/unit/test_frame_io_legacy_compat.py tests/integration/test_unit_scale_invariance_mm_cm_m.py` |
| T-GEOM-001 | standard | T-CK-001 | geometry-electrodes R1-R4 | Validate mesh/electrode/pattern behavior and document any generation gaps. | `pytest tests/unit/test_electrode_position_y_axis.py tests/unit/test_mesh3d_cylinder_generator.py tests/unit/test_measurement_projection_finite_guard.py` |
| T-FWD-001 | thorough | T-GEOM-001, T-DATA-001 | forward-solver R1-R4 | Validate forward solver backend policy, CEM smoke, and PETSc/CUDA probe docs. | `pytest tests/unit/test_forward_model_3d_cem.py tests/unit/test_forward_mat_solve_policy.py`; optional CUDA probe |
| T-INV-001 | thorough | T-FWD-001 | inverse-reconstruction R1-R4 | Validate GN, difference, sparse, and reduced workflow evidence; split gaps by algorithm. | targeted unit tests plus one integration smoke |
| T-CACHE-001 | thorough | T-FWD-001, T-INV-001 | cache-performance R1-R4 | Validate semantic cache behavior, lifecycle docs, and perf policy gates. | cache unit tests; perf-policy tests; optional perf gate |
| T-INTEROP-001 | standard | T-GEOM-001, T-FWD-001 | interop R1-R3 | Validate geometry exchange and GUI interop hub behavior without MATLAB dependency. | `pytest tests/unit/test_interop_geometry_exchange.py tests/unit/test_eit_app_interop_environment.py tests/unit/test_eit_app_interop_hub.py` |
| T-GUI-001 | thorough | T-DATA-001, T-INV-001, T-ENV-001 | workstation-gui R1-R5 | Validate GUI startup, acquisition, database, transport, theme/i18n evidence; note WSLg XCB/Wayland gap. | GUI smoke subset; acquisition/database/transport tests |
| T-ENV-001 | standard | T-CK-001 | environment-cli R1-R4 | Validate Nix/uv launcher docs, env manifest tests, and CLI smoke evidence. | env unit tests; command help smoke |
| T-CK-003 | standard | all domain validation tasks | all kits | Update kits based on validated evidence and mark unsupported criteria as explicit gaps. | `git diff --check`; human review |
| T-CK-004 | quick | T-CK-003 | all kits | Create implementation tracking records summarizing completed validation and remaining gaps. | `context/impl/impl-overview.md` updated |

## Task Details

### T-CK-001: Collect Cited Test Inventory

**Goal:** Confirm that tests cited by `validation-report.md` still collect.

**Actions:**

- Run collect-only for the recommended smoke set.
- Record missing, renamed, skipped, or warning-failing tests.
- Update `context/kits/validation-report.md`.

**Done When:**

- Validation report lists collection result per cited smoke file.
- Any missing test reference becomes a gap, not an assumed pass.

### T-CK-002: Add Source-To-Kit Mapping Files

**Goal:** Let future agents enter source subtrees and discover relevant kits
without loading the whole context tree.

**Actions:**

- Add minimal `CLAUDE.md` files under major subtrees:
  `src/pyeidors`, `src/pyeidors/data`, `src/pyeidors/geometry`,
  `src/pyeidors/forward`, `src/pyeidors/inverse`, `src/pyeidors/cache`,
  `src/pyeidors/interop`, `src/eit_app`, `scripts`.
- Keep each file 3-10 lines.
- Reference exact kit files and requirement ranges.

**Done When:**

- Every major domain kit has at least one source-tree entry point.
- No source `CLAUDE.md` duplicates kit content.

### T-CORE-001: Validate Core System Kit

**Goal:** Confirm `cavekit-core-system.md` describes current public API behavior.

**Actions:**

- Run targeted core tests.
- Inspect failures or skips and decide whether kit, code, or test is stale.
- Update validation report with evidence.

**Done When:**

- Core R1-R4 have covered/partial/gap status.

### T-DATA-001: Validate Data And Units Kit

**Goal:** Confirm measurement data, frame I/O, and unit semantics match current
docs and tests.

**Actions:**

- Run targeted data/unit tests.
- Compare `docs/MEASUREMENT_DATA_SPEC.md` with kit R1-R4.
- Record any missing tests for obsolete amplitude rejection or copy policy.

**Done When:**

- Data R1-R4 have covered/partial/gap status.

### T-GEOM-001: Validate Geometry And Electrodes Kit

**Goal:** Confirm mesh, electrode, and pattern requirements match current tests.

**Actions:**

- Run targeted geometry/electrode tests.
- Check whether pattern ordering and GUI measurement count have explicit tests.
- Update gaps where evidence is indirect.

**Done When:**

- Geometry R1-R4 have covered/partial/gap status.

### T-FWD-001: Validate Forward Solver Kit

**Goal:** Confirm forward CEM behavior and backend policy are captured.

**Actions:**

- Run targeted forward tests.
- If CUDA shell is active, run PETSc CUDA probe; otherwise record as not run.
- Inspect backend-policy tests for SciPy/PETSc fallback coverage.

**Done When:**

- Forward R1-R4 have covered/partial/gap status.
- CUDA criteria are marked covered only if probe is executed.

### T-INV-001: Validate Inverse Reconstruction Kit

**Goal:** Confirm inverse workflow requirements cover strict/fast difference,
absolute GN, sparse Bayesian, Jacobian, regularization, and reduced helpers.

**Actions:**

- Run targeted unit tests for GN, difference, sparse, Jacobian, and reduced
  helpers.
- Run one integration smoke if runtime permits.
- Split any oversized requirement into smaller kit requirements if evidence
  becomes too broad.

**Done When:**

- Inverse R1-R4 have covered/partial/gap status.

### T-CACHE-001: Validate Cache And Performance Kit

**Goal:** Confirm cache semantics, lifecycle, and performance policy are
captured without overstating benchmark coverage.

**Actions:**

- Run cache and perf policy unit tests.
- Review `docs/CACHE_ARCHITECTURE.md` against kit R1-R4.
- Mark long benchmark gates as optional unless executed in current environment.

**Done When:**

- Cache R1-R4 have covered/partial/gap status.

### T-INTEROP-001: Validate Interop Kit

**Goal:** Confirm bridge format and GUI hub behavior are captured without
requiring MATLAB.

**Actions:**

- Run Python-side interop tests.
- Confirm MATLAB/EIDORS-dependent checks are documented as external/manual.
- Update interop kit gaps accordingly.

**Done When:**

- Interop R1-R3 have covered/partial/gap status.

### T-GUI-001: Validate Workstation GUI Kit

**Goal:** Confirm GUI startup, hardware, acquisition, database, visualization,
theme, and i18n behavior are represented.

**Actions:**

- Run GUI smoke subset and non-hardware controller tests.
- Record WSLg XCB/Wayland clarity/stability as a manual or future automated
  smoke gap.
- Avoid physical hardware assumptions unless a device is explicitly available.

**Done When:**

- GUI R1-R5 have covered/partial/gap status.

### T-ENV-001: Validate Environment And CLI Kit

**Goal:** Confirm supported runtime and command-line contracts are represented.

**Actions:**

- Run env manifest/sync unit tests.
- Check CLI `--help` for unified reconstruction and launcher scripts if feasible.
- Do not mutate global shell or Nix config.

**Done When:**

- Env R1-R4 have covered/partial/gap status.

### T-CK-003: Stabilize Kits

**Goal:** Convert validation evidence into stable kit updates.

**Actions:**

- Update acceptance criteria that are too broad or not automatable.
- Mark uncovered criteria as coverage gaps.
- Preserve implementation-agnostic wording.

**Done When:**

- No requirement lacks acceptance criteria.
- Validation report reflects actual test results.

### T-CK-004: Track Implementation State

**Goal:** Record the Map/Make state for future sessions.

**Actions:**

- Update `context/impl/impl-overview.md`.
- Add per-domain tracking files only if validation produces substantial gaps.

**Done When:**

- Future agents can see done, blocked, and not-run validation status.

## Execution Notes

- Use WSL2 Ubuntu and project root `/home/tom/workspace/PyEidors_wsl2`.
- Prefer `uv run` or project launchers only after checking `pyproject.toml`.
- Do not create virtual environments under `~/workspace`.
- Do not touch OneDrive paths.
- Do not rewrite source code unless a task uncovers a concrete failing
  criterion and the user asks to fix it.

## Completion Criteria

- [ ] All tasks are DONE or BLOCKED with reason.
- [ ] `context/kits/validation-report.md` reflects actual verification.
- [ ] Source-to-kit mappings exist for major source domains.
- [ ] `context/impl/impl-overview.md` summarizes status.
- [ ] Human review has approved which current behaviors are intended.
