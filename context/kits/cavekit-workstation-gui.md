---
status: draft
source: from-code
domain: workstation-gui
---

# Cavekit: Workstation GUI

## Scope

This kit covers the EIT Workstation desktop app: startup behavior, runtime
threading, hardware connection, acquisition, recording, simulation, dataset
generation, database browsing, visualization, i18n, themes, and GUI diagnostics.

## Requirements

### R1: GUI startup is stable on supported WSL2 and desktop environments

**Description:** The workstation starts through the official launcher with
runtime preflight checks and Qt platform handling appropriate for WSLg.

**Acceptance Criteria:**
- [ ] Launcher verifies required imports before opening the app.
- [ ] WSLg defaults to XCB unless the user opts into Wayland.
- [ ] GUI startup configures compute thread counts before heavy imports.
- [ ] gmsh is initialized on the main thread to avoid worker signal failures.

**Dependencies:** `cavekit-environment-cli.md`

### R2: Hardware acquisition has simulator and transport boundaries

**Description:** The GUI can operate with simulator, serial, relay, or
Windows-hosted serial transports while keeping blocking I/O off the main thread.

**Acceptance Criteria:**
- [ ] Connection preflight reports target availability before worker connection.
- [ ] Device operations run in worker threads or acquisition process.
- [ ] Simulator workflows support offline tests without physical hardware.
- [ ] Windows serial bridge behavior is tested independently of native Linux
  serial discovery.

**Dependencies:** `cavekit-data-and-units.md`

### R3: Acquisition frames are recorded, indexed, and browsable

**Description:** Acquired frames can be saved to disk, indexed into SQLite, and
selected later as reference/target frames for reconstruction.

**Acceptance Criteria:**
- [ ] Recording writes frame CSV/YAML pairs with metadata.
- [ ] Database backfill runs in a background thread and shuts down cleanly.
- [ ] Database filters cover session names, frequencies, electrode counts, and
  stimulation current ranges where UI supports them.
- [ ] Difference dialog opens modelessly and does not freeze the main window.

**Dependencies:** `cavekit-data-and-units.md`, `cavekit-inverse-reconstruction.md`

### R4: Simulation and dataset generation workflows remain responsive

**Description:** Forward solve, inverse reconstruction, and dataset generation
run through controllers that keep long work off the UI thread and update visual
state deterministically.

**Acceptance Criteria:**
- [ ] Forward and reconstruction controllers emit completion/error signals.
- [ ] Visual widgets show loading or empty states while results are unavailable.
- [ ] 2D and 3D conductivity visualizations render without orphaned VTK windows
  under tested conditions.
- [ ] Dataset generation reports progress and output location.

**Dependencies:** `cavekit-forward-solver.md`, `cavekit-inverse-reconstruction.md`

### R5: UI language, theme, and plotting remain coherent

**Description:** The app persists language, precision, and theme preferences and
retranslates or repolishes visible UI state when preferences change.

**Acceptance Criteria:**
- [ ] English and Chinese translation dictionaries cover visible main-window
  actions and tab labels.
- [ ] Theme changes update custom plot widgets and inline surfaces.
- [ ] Plotting code follows project font policy for English and numeric text.

**Dependencies:** None

## Brownfield Evidence

- Source: `src/eit_app/app.py`
- Source: `src/eit_app/ui/main_window.py`
- Source: `src/eit_app/controllers/`
- Source: `src/eit_app/hardware/`
- Source: `src/eit_app/ui/`
- Source: `scripts/gui/run_eit_app.sh`
- Tests: `tests/unit/test_eit_app_gui_smoke.py`
- Tests: `tests/unit/test_acquisition_controller.py`
- Tests: `tests/unit/test_database_backfill_shutdown.py`
- Tests: `tests/unit/test_eit_app_windows_serial_transport.py`

## Out of Scope

- Core numerical solver correctness; see Forward Solver and Inverse
  Reconstruction.
- MATLAB/EIDORS bridge internals; see Interop.

## Cross-References

- Depends on: `cavekit-environment-cli.md`
- Depends on: `cavekit-data-and-units.md`
- Related: `cavekit-interop.md`

