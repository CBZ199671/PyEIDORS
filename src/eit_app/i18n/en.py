"""English translation dictionary.

Keys follow a dotted scope convention: ``<area>.<component>.<element>``.
Formatting placeholders use :py:meth:`str.format` syntax (e.g. ``{count}``).

This file is a plain Python ``dict`` (not JSON / YAML) on purpose so that:
  * IDEs can autocomplete key references,
  * ``grep`` finds usages directly,
  * static checkers can flag unused keys.
"""

from __future__ import annotations

TRANSLATIONS: dict[str, str] = {
    # ==================================================================
    # Application chrome
    # ==================================================================
    "app.title": "EIT Workstation",

    # ------------------------------------------------------------------
    # Tab labels  (kept as short domain nouns for a tight tab bar)
    # ------------------------------------------------------------------
    "tab.hardware": "Hardware",
    "tab.simulation": "Simulation",
    "tab.dataset": "Dataset",
    "tab.database": "Database",

    # ------------------------------------------------------------------
    # File menu
    # ------------------------------------------------------------------
    "menu.file": "&File",
    "menu.file.settings": "&Settings\u2026",
    "menu.file.exit": "E&xit",

    # ------------------------------------------------------------------
    # Tools menu
    # ------------------------------------------------------------------
    "menu.tools": "&Tools",
    "menu.tools.interop_hub": "EIDORS &Interop Hub\u2026",

    # ------------------------------------------------------------------
    # Language menu
    # ------------------------------------------------------------------
    "menu.language": "&Language",
    "menu.language.zh": "\u4e2d\u6587",
    "menu.language.en": "English",
    "menu.language.tooltip": "Switch between Chinese and English",

    # ==================================================================
    # Hardware tab — Step labels on the left QToolBox
    # ==================================================================
    "hw.step.link": "Step 1 \u00b7 Link",
    "hw.step.setup": "Step 2 \u00b7 Setup",
    "hw.step.acquire": "Step 3 \u00b7 Acquire",

    # ==================================================================
    # Hardware tab — Step 1 Connection panel
    # ==================================================================
    "hw.connection.title": "1. Link & Verify",
    "hw.connection.flow_hint": "Select the transport and verify the device link first.",
    "hw.connection.transport_label": "Transport:",
    "hw.connection.transport.serial": "Serial",
    "hw.connection.transport.relay_4g": "4G Relay",
    "hw.connection.port_label": "Port:",
    "hw.connection.scan_button": "Scan",
    "hw.connection.scan_button_tooltip": "Refresh serial ports",
    "hw.connection.baud_label": "Baud rate:",
    "hw.connection.host_label": "Server host:",
    "hw.connection.port_spin_label": "Server port:",
    "hw.connection.board_id_label": "Board ID:",
    "hw.connection.user_id_label": "User ID:",
    "hw.connection.connect_button": "Connect",
    "hw.connection.connect_button_tooltip": "Connect and verify the device link",
    "hw.connection.disconnect_button": "Disconnect",
    "hw.connection.port_hint.no_ports": "No serial ports detected. The launcher checks both local Linux ports and the Windows COM bridge; verify the USB cable, driver, and device power, then click Scan.",
    "hw.connection.port_hint.still_no_ports": "Still no serial ports detected \u2014 holding off on connecting. Check USB cable, driver, and device power.",
    "hw.connection.port_hint.single_port_bridge": "Auto-selected the only port: {port}. Will use the Windows COM bridge on connect.",
    "hw.connection.port_hint.single_port": "Auto-selected the only port: {port}.",
    "hw.connection.port_hint.multi_port_bridge": "Detected {count} ports, current selection: {port}. Will use the Windows COM bridge on connect.",
    "hw.connection.port_hint.multi_port": "Detected {count} ports \u2014 verify and pick the one matching your hardware.",
    "hw.connection.relay_hint.dynamic": "4G Relay will connect to {host}:{port}. A reachability check runs before connecting.",

    # ==================================================================
    # Hardware tab — Step 2 Control panel
    # ==================================================================
    "hw.control.title": "2. Setup & Diagnostics",
    "hw.control.power_header": "Measurement power",
    "hw.control.power_on_button": "Power ON",
    "hw.control.power_off_button": "Power OFF",
    "hw.control.power_hint": "Power ON/OFF directly drives the board supply. Single Point is functional-test only.",
    "hw.control.layout_header": "Hardware layout",
    "hw.control.rotate_meas_check": "Rotate measurement with drive",
    "hw.control.use_meas_current_check": "Measure drive-related electrodes",
    "hw.control.setup_header": "Measurement setup",
    "hw.control.frequency_label": "Frequency",
    "hw.control.freq_apply_button": "Set",
    "hw.control.stim_amp_label": "Stim amplitude",
    "hw.control.stim_apply_button": "Set",
    "hw.control.voltage_gain_label": "Voltage gain",
    "hw.control.vamp_apply_button": "Set",
    "hw.control.diag_header": "Diagnostics",
    "hw.control.spt_button": "Single Point",
    "hw.control.impedance_button": "Impedance",
    "hw.control.layout_grid.mode": "Mode",
    "hw.control.layout_grid.elec_ring": "Elec/ring",
    "hw.control.layout_grid.rings": "Rings",
    "hw.control.layout_grid.stim_pattern": "Stim pattern",
    "hw.control.layout_grid.meas_pattern": "Meas pattern",
    "hw.control.layout_grid.extra_neighbors": "Extra excluded neighbors",
    "hw.control.cem_grid.radius": "Radius",
    "hw.control.cem_grid.elec_length": "Elec length",
    "hw.control.cem_grid.contact_z": "Contact z",

    # ==================================================================
    # Hardware tab — Step 3 Acquisition panel
    # ==================================================================
    "hw.acquisition.title": "3. Acquire & Record",
    "hw.acquisition.flow_hint": "Prepare the save path and plan, then launch the acquisition run.",
    "hw.acquisition.record_header": "Recording setup",
    "hw.acquisition.save_to_label": "Save to:",
    "hw.acquisition.dir_placeholder": "Output directory\u2026",
    "hw.acquisition.browse_button": "Browse\u2026",
    "hw.acquisition.record_check": "Record to disk",
    "hw.acquisition.plan_header": "Acquisition plan",
    "hw.acquisition.timed_interval_check": "Timed interval",
    "hw.acquisition.interval_label": "Interval:",
    "hw.acquisition.count_label": "Acquisitions:",
    "hw.acquisition.count_continuous": "Continuous",
    "hw.acquisition.freq_step_check": "Step frequency across the run",
    "hw.acquisition.start_freq_label": "Start freq:",
    "hw.acquisition.end_freq_label": "End freq:",
    "hw.acquisition.plan_hint": "Acquisitions = 0 means unlimited continuous capture. A positive number runs exactly that many and stops automatically.",
    "hw.acquisition.action_header": "Acquisition actions",
    "hw.acquisition.start_button": "Start",
    "hw.acquisition.start_button_tooltip": "Start the current acquisition plan",
    "hw.acquisition.single_frame_button": "Single Frame",
    "hw.acquisition.single_frame_button_tooltip": "Acquire exactly one frame",
    "hw.acquisition.stop_button": "Stop",
    "hw.acquisition.stop_button_tooltip": "Stop the current acquisition run",
    "hw.acquisition.frames_acquired_label": "Frames acquired:",
    "hw.acquisition.file_dialog_title": "Select Output Directory",

    # ==================================================================
    # Hardware tab — Session summary footer
    # ==================================================================
    "hw.summary.field.identity": "Identity:",
    "hw.summary.field.transport": "Transport:",
    "hw.summary.field.layout": "Layout:",
    "hw.summary.field.drive": "Drive:",
    "hw.summary.field.record": "Record path:",
    "hw.summary.field.plan": "Plan:",
    "hw.summary.indicator.link": "LINK",
    "hw.summary.indicator.power": "POWER",
    "hw.summary.indicator.record": "RECORD",
    "hw.summary.indicator.acq": "ACQ",
    # Full banner variants (title / detail / action) — one set per
    # connection / power / acquisition / recording state combination.
    "hw.summary.banner.link_down.title": "LINK DOWN",
    "hw.summary.banner.link_down.detail": "No verified device link is active.",
    "hw.summary.banner.link_down.action": "Select a transport and click Connect & Verify.",
    "hw.summary.banner.fault.title": "FAULT",
    "hw.summary.banner.fault.detail": "The link is in an error state and requires operator attention.",
    "hw.summary.banner.fault.action": "Disconnect the link, check transport settings, and verify again.",
    "hw.summary.banner.verifying.title": "VERIFYING LINK",
    "hw.summary.banner.verifying.detail": "The workstation is probing the device and reading its protocol capabilities.",
    "hw.summary.banner.verifying.action": "Wait for link verification to finish.",
    "hw.summary.banner.acquiring.title": "ACQUIRING",
    "hw.summary.banner.acquiring.detail": "Frames are being captured from the active transport.",
    "hw.summary.banner.acquiring.action": "Monitor the live plot and stop acquisition when the run is complete.",
    "hw.summary.banner.acquiring_recording.title": "ACQUIRING + RECORDING",
    "hw.summary.banner.acquiring_recording.detail": "Frames are being captured and written to the active session.",
    "hw.summary.banner.acquiring_recording.action": "Monitor incoming frames or stop acquisition when the run is complete.",
    "hw.summary.banner.ready_simulator.title": "READY FOR ACQUISITION",
    "hw.summary.banner.ready_simulator.detail": "The simulator link is verified and can start generating frames immediately.",
    "hw.summary.banner.ready_simulator.action": "Start continuous or single-frame acquisition.",
    "hw.summary.banner.ready_record_armed.title": "READY + RECORD ARMED",
    "hw.summary.banner.ready_record_armed.detail": "The device link is verified, measurement power is ON, and the next run will be saved.",
    "hw.summary.banner.ready_record_armed.action": "Start acquisition to capture and record the next session.",
    "hw.summary.banner.ready.title": "READY FOR ACQUISITION",
    "hw.summary.banner.ready.detail": "The device link is verified and measurement power is ON.",
    "hw.summary.banner.ready.action": "Start continuous or single-frame acquisition.",
    "hw.summary.banner.link_verified_armed.title": "LINK VERIFIED",
    "hw.summary.banner.link_verified_armed.detail": "The link is verified and recording is armed, but measurement power is not confirmed ON.",
    "hw.summary.banner.link_verified_armed.action": "Turn measurement power ON when the hardware is ready, then start acquisition.",
    "hw.summary.banner.link_verified.title": "LINK VERIFIED",
    "hw.summary.banner.link_verified.detail": "The device link is verified and waiting for measurement power or the next setup change.",
    "hw.summary.banner.link_verified.action": "Turn measurement power ON when the hardware is ready, then start acquisition.",

    # Short-form indicator chips (LINK / POWER / RECORD / ACQ)
    "hw.summary.chip.link.down": "DOWN",
    "hw.summary.chip.link.check": "CHECK",
    "hw.summary.chip.link.ok": "OK",
    "hw.summary.chip.link.fault": "FAULT",
    "hw.summary.chip.link.unk": "UNK",
    "hw.summary.chip.power.unk": "UNK",
    "hw.summary.chip.power.off": "OFF",
    "hw.summary.chip.power.on": "ON",
    "hw.summary.chip.record.off": "OFF",
    "hw.summary.chip.record.arm": "ARM",
    "hw.summary.chip.record.rec": "REC",
    "hw.summary.chip.acq.idle": "IDLE",
    "hw.summary.chip.acq.run": "RUN",
    "hw.summary.chip.acq.sch": "SCH",
    "hw.summary.chip.acq.fin": "FIN",
    "hw.summary.chip.acq.step": "STEP",
    "hw.summary.chip.acq.1fr": "1FR",

    # Legacy state keys kept for the first-run / default indicator states.
    "hw.summary.state.down": "Down",
    "hw.summary.state.unknown": "Unknown",
    "hw.summary.state.off": "Off",
    "hw.summary.state.idle": "Idle",

    # ==================================================================
    # Hardware tab — Right-side Frame browser
    # ==================================================================
    "hw.frame_browser.title": "Recorded Frames",
    "hw.frame_browser.hint": "The first frame of each run is used as reference automatically. Click any frame and then \u2018Set as Reference\u2019 to override it \u2014 the newest acquired frame is always the target.",
    "hw.frame_browser.count_label": "Recorded frames: {count}",
    "hw.frame_browser.column.index": "Index",
    "hw.frame_browser.column.timestamp": "Timestamp",
    "hw.frame_browser.column.file": "File",
    "hw.frame_browser.set_ref_button": "Set as Reference",
    "hw.frame_browser.clear_button": "Clear List",

    # ==================================================================
    # Hardware tab — Live measurement plot
    # ==================================================================
    "hw.live_plot.title": "Live Measurement Channels",
    "hw.live_plot.y_label": "Voltage (V)",
    "hw.live_plot.x_label_dynamic": "Measurement Index (1-{count})",
    "hw.live_plot.curve.real": "Real",
    "hw.live_plot.curve.imag": "Imag",
    "hw.live_plot.empty_overlay": "No live frames yet.\nStart acquisition to display Real and Imag.",

    # ==================================================================
    # Hardware tab — Reconstruction widget
    # ==================================================================
    "hw.reconstruction.title": "Reconstruction",
    "hw.reconstruction.empty_overlay": "No reconstruction yet",
    "hw.reconstruction.error.expect_2d_triangles": "Fast reconstruction view currently expects 2D triangles",

    # ==================================================================
    # Hardware tab — Boundary voltage fit plot
    # ==================================================================
    "hw.boundary.title": "Boundary Voltage Fit",
    "hw.boundary.y_label": "Voltage (V)",
    "hw.boundary.x_label_dynamic": "Boundary Voltage Index (1-{count})",
    "hw.boundary.primary.measured": "Measured",
    "hw.boundary.primary.ground_truth": "Ground Truth",
    "hw.boundary.secondary": "Recon Fit",
    "hw.boundary.empty.hardware": "Measured and reconstruction-fit boundary voltages will appear after reconstruction updates.",
    "hw.boundary.empty.simulation": "Ground-truth and reconstruction-fit boundary voltages will appear after forward or inverse updates.",

    # ==================================================================
    # Shared plot-legend overlay (used across Hardware and Simulation)
    # ==================================================================
    "plot_legend.drag_tooltip": "Drag to reposition the legend",

    # ==================================================================
    # Simulation tab — Step labels and Run Guide footer
    # ==================================================================
    "sim.step.mesh": "Step 1 \u00b7 Mesh & Electrodes",
    "sim.step.inhom": "Step 2 \u00b7 Inhomogeneities",
    "sim.step.forward": "Step 3 \u00b7 Forward Problem",
    "sim.step.inverse": "Step 4 \u00b7 Inverse Problem",

    "sim.runguide.title": "Run Guide",
    "sim.runguide.step1": "Configure the mesh and electrodes, then maintain the inhomogeneity list.",
    "sim.runguide.step2": "Run Forward to inspect boundary voltages and the ground truth image.",
    "sim.runguide.step3": "Run Inverse to view the reconstructed image and error metrics on the right.",
    "sim.runguide.hint": "The central area is reserved for image and curve comparisons.",

    # ==================================================================
    # Simulation tab — Step 1 Mesh & Electrodes
    # ==================================================================
    "sim.mesh.title": "Mesh & Electrodes",
    "sim.mesh.hint": "Configure the simulation mesh and electrode layout.",
    "sim.mesh.dim.2d": "2D",
    "sim.mesh.dim.3d": "3D",
    "sim.mesh.dimension_label": "Dimension:",
    "sim.mesh.size_label": "Mesh size:",
    "sim.mesh.refinement_tooltip": "Smaller values produce finer meshes (more elements)",
    "sim.mesh.electrodes_label": "Electrodes:",
    "sim.mesh.conductivity_label": "Background \u03c3:",

    # ==================================================================
    # Simulation tab — Step 2 Inhomogeneities
    # ==================================================================
    "sim.inhom.title": "Inhomogeneities",
    "sim.inhom.col.shape": "Shape",
    "sim.inhom.col.x": "X",
    "sim.inhom.col.y": "Y",
    "sim.inhom.col.sizex": "Size X",
    "sim.inhom.col.sizey": "Size Y",
    "sim.inhom.col.conductivity": "\u03c3 (S/m)",
    "sim.inhom.add_circle": "+ Circle",
    "sim.inhom.add_ellipse": "+ Ellipse",
    "sim.inhom.add_rectangle": "+ Rectangle",
    "sim.inhom.remove_button": "Remove",

    # ==================================================================
    # Simulation tab — Step 3 Forward Problem
    # ==================================================================
    "sim.forward.title": "Forward Problem",
    "sim.forward.hint": "Compute boundary voltages from the conductivity distribution.",
    "sim.forward.noise_label": "Noise level:",
    "sim.forward.noise_tooltip": "Relative noise level (0 = noiseless)",
    "sim.forward.solve_button": "Solve Forward Problem",
    "sim.forward.status_solving": "Solving\u2026",

    # ==================================================================
    # Simulation tab — Step 4 Inverse Problem
    # ==================================================================
    "sim.inverse.title": "Inverse Problem",
    "sim.inverse.hint": "Reconstruct the conductivity distribution from boundary voltages.",
    "sim.inverse.method_label": "Method:",
    "sim.inverse.alpha_label": "Regularization \u03b1:",
    "sim.inverse.iterations_label": "Max iterations:",
    "sim.inverse.reconstruct_button": "Reconstruct",
    "sim.inverse.save_button": "Save Results",
    "sim.inverse.status_reconstructing": "Reconstructing\u2026",

    # ==================================================================
    # Simulation tab — Right-side Metrics panel
    # ==================================================================
    "sim.metrics.title": "Metrics",
    "sim.metrics.l2_label": "Relative L2 error:",
    "sim.metrics.correlation_label": "Correlation:",
    "sim.metrics.rmse_label": "RMSE:",

    # ==================================================================
    # Simulation tab — Centre results widget
    # ==================================================================
    "sim.results.ground_truth_title": "Ground Truth",
    "sim.results.reconstruction_title": "Reconstruction",

    # ==================================================================
    # Dataset Generator tab — Step labels
    # ==================================================================
    "dataset.step.mesh": "Step 1 \u00b7 Mesh & Electrodes",
    "dataset.step.ranges": "Step 2 \u00b7 Randomization Ranges",
    "dataset.step.run": "Step 3 \u00b7 Output & Run",

    # ==================================================================
    # Dataset Generator tab — Central workspace blocks
    # ==================================================================
    "dataset.hero.title": "Batch Dataset Pipeline",
    "dataset.hero.title_text": "Generate mesh-aware conductivity targets and boundary-voltage pairs with a cleaner, step-by-step workflow.",
    "dataset.hero.hint": "Use the left-side steps to define mesh, randomization ranges, and the batch output target. The summary panel on the right mirrors the active run.",
    "dataset.artifacts.title": "Generated Artifacts",
    "dataset.artifacts.item1": "mesh_info.npz with node coordinates, cell connectivity, and homogeneous voltages",
    "dataset.artifacts.item2": "sample_000000.npz style per-sample conductivity and boundary-voltage pairs",
    "dataset.artifacts.item3": "The configured output directory becomes a self-contained dataset package",
    "dataset.notes.title": "Operating Notes",
    "dataset.notes.item1": "Mesh settings here are independent from the interactive Simulation tab.",
    "dataset.notes.item2": "Shape toggles define the random family pool; if none are checked, circle is used by default.",
    "dataset.notes.item3": "Noise is applied after the forward solve, so voltage perturbations match the configured batch range.",

    # ==================================================================
    # Dataset Generator tab — Step 2 Randomization panel
    # ==================================================================
    "dataset.random.title": "Randomization Ranges",
    "dataset.random.hint": "Choose which shapes to sample and the numeric ranges used to paint synthetic conductivity targets.",
    "dataset.random.header.shapes": "Shape families",
    "dataset.random.header.count": "Target population",
    "dataset.random.header.spatial": "Spatial ranges",
    "dataset.random.header.conductivity": "Conductivity ranges",
    "dataset.random.shape.circle": "Circle",
    "dataset.random.shape.ellipse": "Ellipse",
    "dataset.random.shape.rectangle": "Rectangle",
    "dataset.random.shapes_label": "Shapes:",
    "dataset.random.n_label": "N inhom.:",
    "dataset.random.position_label": "Position:",
    "dataset.random.size_label": "Size:",
    "dataset.random.conductivity_label": "\u03c3 range:",
    "dataset.random.background_label": "Background \u03c3:",
    "dataset.random.noise_label": "Noise level:",

    # ==================================================================
    # Dataset Generator tab — Step 3 Output & Run panel
    # ==================================================================
    "dataset.run.title": "Output & Run",
    "dataset.run.hint": "Choose where the dataset should be written, then start the batch job when the mesh and ranges look right.",
    "dataset.run.samples_label": "Samples:",
    "dataset.run.save_to_label": "Save to:",
    "dataset.run.dir_placeholder": "Output directory\u2026",
    "dataset.run.browse_button": "Browse\u2026",
    "dataset.run.progress_header": "Execution progress",
    "dataset.run.status.ready": "Ready to generate.",
    "dataset.run.status.progress": "Generated {current} / {total} samples.",
    "dataset.run.generate_button": "Generate Dataset",
    "dataset.run.cancel_button": "Cancel",
    "dataset.run.file_dialog_title": "Select Output Directory",

    # ==================================================================
    # Dataset Generator tab — Right-side Summary panel
    # ==================================================================
    "dataset.summary.title": "Generation Summary",
    "dataset.summary.hint": "Review the active batch configuration here before launching the generator.",
    "dataset.summary.progress": "Progress: {current} / {total}",
    "dataset.summary.state.idle": "Idle",
    "dataset.summary.state.generating": "Generating",
    "dataset.summary.state.complete": "Complete",
    "dataset.summary.field.output": "Output:",
    "dataset.summary.field.samples": "Samples:",
    "dataset.summary.field.shapes": "Shapes:",
    "dataset.summary.field.mesh": "Mesh:",
    "dataset.summary.field.electrodes": "Electrodes:",
    "dataset.summary.field.status": "Status:",

    # ==================================================================
    # Database tab — Left filter panel
    # ==================================================================
    "db.filters.title": "FILTERS",
    "db.filters.hint": "Search the archive by name, frequency, or date.",
    "db.filters.name_label": "Name:",
    "db.filters.name_placeholder": "tank, test_for_gui \u2026",
    "db.filters.freq_label": "Frequency (Hz):",
    "db.filters.freq_placeholder": "e.g. 1000",
    "db.filters.date_any": "Any",
    "db.filters.date_from_label": "Date from:",
    "db.filters.date_to_label": "Date to:",
    "db.filters.apply_button": "Apply Filters",
    "db.filters.clear_button": "Clear",
    "db.filters.refresh_button": "Refresh",
    "db.stats.count": "{count} sessions",
    "db.stats.ready": "Ready",
    "db.stats.backfill_progress": "Backfill: {current}/{total}",
    "db.stats.backfill_done": "Backfill complete: {count} sessions imported.",

    # ==================================================================
    # Database tab — Central sessions / frames section
    # ==================================================================
    "db.sessions.title": "SESSIONS",
    "db.sessions.col.id": "ID",
    "db.sessions.col.name": "Name",
    "db.sessions.col.started": "Started",
    "db.sessions.col.n_elec": "N_elec",
    "db.sessions.col.frequency": "Frequency",
    "db.sessions.col.stim": "Stim (uA)",
    "db.sessions.col.gain": "Gain",
    "db.sessions.col.frames": "Frames",
    "db.sessions.open_folder_button": "Open Folder",
    "db.sessions.batch_recon_button": "Batch Reconstruct\u2026",

    "db.frames.title": "FRAMES",
    "db.frames.col.index": "Index",
    "db.frames.col.timestamp": "Timestamp",
    "db.frames.col.file": "File",
    "db.frames.selection_hint": "Select a frame, then click \u2018Set as Reference\u2019 or \u2018Set as Target\u2019.",
    "db.frames.selection_role.reference": "Reference",
    "db.frames.selection_role.target": "Target",
    "db.frames.selection_unset": "{role}: <unset>",
    "db.frames.selection_set": "{role}: #{index}",
    "db.frames.set_ref_button": "Set as Reference",
    "db.frames.set_tgt_button": "Set as Target",
    "db.frames.reconstruct_button": "Reconstruct\u2026",
    "db.frames.clear_button": "Clear",

    # ==================================================================
    # Database tab — Right-side preview panel
    # ==================================================================
    "db.preview.title": "FRAME PREVIEW",
    "db.preview.hint": "Click any frame row to preview its waveform here.",

    # ==================================================================
    # Main window — transient status-bar flash messages
    # ==================================================================
    # Connection / transport
    "main.status.port_not_found_scan": "No serial ports detected. Check USB cable, driver, and device power, then click Scan to retry.",
    "main.status.relay_host_empty": "4G Relay server host is empty. Please fill in a reachable host first.",
    "main.status.verifying.windows_bridge": "Verifying device link via Windows serial bridge {port} at {baud} baud.",
    "main.status.verifying.serial": "Verifying serial link: {port} @ {baud}",
    "main.status.verifying.relay": "Verifying 4G Relay link: {host}:{port}",
    "main.status.verifying.generic": "Verifying device link.",
    "main.status.link_verified": "Link verification complete. Turn on measurement power and start acquisition when ready.",

    # Acquisition + recording
    "main.error.connection_required": "Please verify the device connection first.",
    "main.error.port_release_failed": "Failed to release the control serial port before starting. Retry or reconnect the device.",
    "main.error.acq_count_zero": "Finite or timed acquisition requires Acquisitions > 0.",
    "main.status.single_frame_started": "Single-frame acquisition started. It will stop after capturing 1 frame.",
    "main.status.single_frame_done": "Single-frame acquisition complete.",
    "main.status.continuous_started": "Continuous acquisition started.",
    "main.status.plan_stopped": "Planned acquisition stopped.",
    "main.status.plan_step_done": "Acquisition {current}/{total} complete; next run starts in {interval:.1f}s.",
    "main.status.recording_started": "Recording started: {dir}",
    "main.status.recording_stopped": "Recording stopped; {count} frames saved.",
    "main.status.frames_cleared": "Recorded frame list cleared.",
    "main.status.record_enabled": "Recording enabled; captures will be saved to {dir}.",
    "main.status.record_path_pending": "Recording is already running; the new save path will take effect on the next acquisition.",

    # Reconstruction pre-warm
    "main.status.prewarming": "Pre-warming the realtime reconstruction context\u2026",
    "main.status.prewarm_done": "Realtime reconstruction context pre-warmed; subsequent captures will use the hot-start path.",
    "main.status.prewarm_failed": "Realtime reconstruction pre-warm failed; will retry when needed: {reason}",

    # Frame browser / reference / target
    "main.status.reference_updated": "Reference frame updated: #{index}",
    "main.status.reference_selected": "Reference frame selected: #{index}",
    "main.status.target_selected": "Target frame selected: #{index}",
    "main.status.frame_preview": "Showing waveform data for frame #{index}",

    # Layout + protocol + power + diagnostics
    "main.status.layout_updated": "Hardware layout updated: {points} boundary voltage points.",
    "main.status.protocol_caps": "Protocol capabilities: {version}",
    "main.status.spt_result": "Single-point returned: real={real:.4f} V, imag={imag:.4f} V",
    "main.status.power_on": "Measurement power switched to ON.",
    "main.status.power_off": "Measurement power switched to OFF.",
    "main.status.power_sent": "Measurement power command sent.",
    "main.status.command_sent": "Command sent: {name}",
    "main.status.impedance_done": "Contact impedance measurement complete.",
    "main.status.impedance_result": "Contact impedance: {values}",

    # Plan + frequency sweep
    "main.status.plan_started": "Planned acquisition started: {count} runs.",
    "main.status.plan_sweep_note": "Frequency sweep active: waveforms, boundary voltage, and reconstruction update per step.",
    "main.status.plan_step_start": "Starting run {current}/{total}: {hz} Hz",
    "main.status.plan_complete": "Planned acquisition finished ({count} runs).",

    # Interop hub bridge results
    "main.interop.geometry_generate_failed": "Failed to auto-generate simulation geometry.mat: {error}",
    "main.interop.export_note_hw_real": "Recording export defaults to the real part of the boundary voltage so it matches common EIDORS difference workflows.",
    "main.interop.export_note_hw_no_geom": "Hardware export defaults to a layout template. Import a geometry asset from simulation results or a bridge bundle first if you need real mesh geometry.",
    "main.interop.applied_to_hw": "Bridge config applied to Hardware: {dim} | {n_elec} electrodes/ring | {points} points.",
    "main.interop.applied_to_sim": "Bridge config applied to Simulation: {dim} | {n_elec} electrodes/ring | {points} points.",
    "main.interop.applied_to_dataset": "Bridge config applied to Dataset: {dim} | {n_elec} electrodes/ring | {points} points.",
    "main.interop.no_voltage_data": "This bridge bundle does not contain importable boundary voltage data.",
    "main.interop.voltage_cached": "Boundary voltage asset cached; it can now be used for exports, comparison, or reconstruction smoke tests.",
    "main.interop.no_geometry": "This bridge bundle has no geometry.mat.",
    "main.interop.geometry_cached": "Geometry asset cached; subsequent EIDORS exports can reuse it directly.",
    "main.interop.unknown_target": "Unknown import target: {target}",
    "main.interop.smoke_done": "Interop smoke test complete.",

    # humanize_error_message branches
    "main.hw_error.no_serial_ports": "No serial ports detected. Check USB, driver, and device power, then click Scan again.",
    "main.hw_error.port_access_denied": "Serial port access denied; another process may be using it. Close the occupying process and retry.",
    "main.hw_error.windows_port_invalid": "Serial port cannot be configured. The port is not available in this environment. Pick an auto-detected COM port from the drop-down; do not type /dev/ttyS* manually.",
    "main.hw_error.windows_bridge_port_busy": "Windows serial bridge failed: the COM port is still held by another process. If you just closed the app, wait 1-2 seconds and retry.",
    "main.hw_error.windows_bridge_port_missing": "Windows serial bridge failed: this COM port is not visible. Unplug + replug the device and Scan again.",
    "main.hw_error.windows_bridge_generic": "Windows serial bridge failed to start. Scan again and retry.",
    "main.hw_error.relay_host_empty": "4G Relay server host is empty. Please fill in a reachable host.",
    "main.hw_error.relay_refused": "4G Relay server refused the connection. Verify host/port and confirm the service is running.",
    "main.hw_error.relay_timeout": "4G Relay connection timed out. Check the network, server address, and target device status.",

    # ==================================================================
    # Bottom status bar (persistent chips + FPS / frame counters)
    # ==================================================================
    "status.fps": "FPS: --",
    "status.fps_value": "FPS: {value:.1f}",
    "status.frames": "Frames: 0",
    "status.frames_value": "Frames: {count}",
    "status.mode.hardware": "Mode: Hardware",
    "status.mode.simulation": "Mode: Simulation",
    "status.mode.dataset": "Mode: Dataset",
    "status.mode.database": "Mode: Database",
    "status.mode.other": "Mode: {index}",
    "status.link.connected": "Link: Verified",
    "status.link.connecting": "Link: Connecting",
    "status.link.disconnected": "Link: Down",
    "status.link.error": "Link: Error",
    "status.link.other": "Link: {status}",
    "status.power.on": "Power: ON",
    "status.power.off": "Power: OFF",
    "status.power.unknown": "Power: Unknown",
    "status.power.other": "Power: {status}",
    "status.acq.idle": "Acq: Idle",
    "status.acq.continuous": "Acq: Continuous",
    "status.acq.scheduled": "Acq: Scheduled",
    "status.acq.finite_run": "Acq: Finite Run",
    "status.acq.stepped_run": "Acq: Stepped Run",
    "status.acq.single_shot": "Acq: Single Frame",
    "status.acq.other": "Acq: {mode}",
    "status.record.off": "Record: Off",
    "status.record.armed": "Record: Armed",
    "status.record.recording": "Record: Writing",
    "status.record.other": "Record: {status}",

    # ==================================================================
    # Dialog — Settings
    # ==================================================================
    "dlg.settings.title": "Settings",
    "dlg.settings.recon.title": "Reconstruction",
    "dlg.settings.recon.method_label": "Method:",
    "dlg.settings.recon.alpha_label": "Regularization alpha:",
    "dlg.settings.recon.iter_label": "Max iterations:",
    "dlg.settings.recon.dim_label": "Mesh dimension:",
    "dlg.settings.recon.refine_label": "Mesh refinement:",
    "dlg.settings.recon.part_label": "Use part:",
    "dlg.settings.paths.title": "Data Paths",
    "dlg.settings.paths.output_placeholder": "Default output directory\u2026",
    "dlg.settings.paths.browse_button": "Browse\u2026",
    "dlg.settings.paths.output_label": "Output dir:",

    # ==================================================================
    # Dialog — Difference Reconstruction
    # ==================================================================
    "dlg.difference.title": "Difference Reconstruction",
    "dlg.difference.frame_group": "Frame Selection",
    "dlg.difference.ref_label": "Reference frame:",
    "dlg.difference.tgt_label": "Target frame:",
    "dlg.difference.settings_group": "Settings",
    "dlg.difference.mode_label": "Difference mode:",
    "dlg.difference.orient_label": "Orientation:",
    "dlg.difference.part_label": "Use part:",
    "dlg.difference.warn_same_frame": "Reference and target must be different frames.",

    # ==================================================================
    # Dialog — Single-session Reconstruct
    # ==================================================================
    "dlg.reconstruction.title": "Reconstruct",
    "dlg.reconstruction.heading": "Reconstruct from Recorded Frames",
    "dlg.reconstruction.cancel_button": "Cancel",
    "dlg.reconstruction.run_button": "Run Reconstruction",
    "dlg.reconstruction.selected_frames_group": "SELECTED FRAMES",
    "dlg.reconstruction.ref_label": "Reference:",
    "dlg.reconstruction.tgt_label": "Target:",
    "dlg.reconstruction.algo_params_group": "ALGORITHM && PARAMETERS",
    "dlg.reconstruction.method_label": "Method:",
    "dlg.reconstruction.part_label": "Use part:",
    "dlg.reconstruction.alpha_label": "Regularization \u03b1:",
    "dlg.reconstruction.iter_label": "Max iterations:",
    "dlg.reconstruction.output_group": "OUTPUT (OPTIONAL)",
    "dlg.reconstruction.output_placeholder": "Leave empty to only display the result (not save)",
    "dlg.reconstruction.browse_button": "Browse\u2026",
    "dlg.reconstruction.output_folder_label": "Output folder:",
    "dlg.reconstruction.save_image_check": "Save reconstruction image (PNG)",
    "dlg.reconstruction.save_voltage_check": "Save boundary voltage fit plot (PNG)",
    "dlg.reconstruction.not_selected": "<not selected>",
    "dlg.reconstruction.absolute_no_ref_tip": "Absolute methods do not use a reference frame.",

    # ==================================================================
    # Dialog — Batch Reconstruct
    # ==================================================================
    "dlg.batch.title": "Batch Reconstruct",
    "dlg.batch.heading": "Batch Reconstruction",
    "dlg.batch.close_button": "Close",
    "dlg.batch.open_output_button": "Open Output Folder",
    "dlg.batch.cancel_button": "Cancel Job",
    "dlg.batch.run_button": "Run Batch",
    "dlg.batch.folders_group": "FOLDERS",
    "dlg.batch.input_placeholder": "Folder containing frame CSV files",
    "dlg.batch.browse_button": "Browse\u2026",
    "dlg.batch.input_label": "Input folder:",
    "dlg.batch.output_placeholder": "Folder to write reconstruction images",
    "dlg.batch.output_label": "Output folder:",
    "dlg.batch.algo_params_group": "ALGORITHM && PARAMETERS",
    "dlg.batch.method_label": "Method:",
    "dlg.batch.part_label": "Use part:",
    "dlg.batch.alpha_label": "Regularization \u03b1:",
    "dlg.batch.iter_label": "Max iterations:",
    "dlg.batch.ref_browse_button": "Browse\u2026",
    "dlg.batch.ref_label": "Reference frame:",
    "dlg.batch.outputs_group": "OUTPUTS",
    "dlg.batch.save_image_check": "Save reconstruction image (PNG)",
    "dlg.batch.progress_group": "PROGRESS",
    "dlg.batch.ready": "Ready to run.",
    "dlg.batch.cancelling": "Cancelling\u2026",
    "dlg.batch.progress_default": "{current}/{total}",
    "dlg.batch.error": "\u2715  Error: {message}",
    "dlg.batch.subtitle": "Reconstruct every frame CSV in the input folder. For difference methods, the reference is applied to all targets and is automatically excluded when it sits in the same folder.",
    "dlg.batch.ref_placeholder": "CSV file to use as reference (required for difference methods)",
    "dlg.batch.save_voltage_check": "Save boundary voltage fit plot (PNG)",
    "dlg.batch.file_dialog.input": "Select Input Folder",
    "dlg.batch.file_dialog.output": "Select Output Folder",
    "dlg.batch.file_dialog.ref": "Select Reference Frame CSV",
    "dlg.batch.file_dialog.csv_filter": "CSV files (*.csv)",
    "dlg.batch.finished_ok": "\u2713  Finished \u2014 succeeded: {succeeded}, failed: {failed}",
    "dlg.batch.finished_mixed": "\u26a0  Finished \u2014 succeeded: {succeeded}, failed: {failed}",
    "dlg.batch.finished_fail": "\u2715  Finished \u2014 succeeded: {succeeded}, failed: {failed}",

    # Reconstruction dialog — subtitle copy
    "dlg.reconstruction.subtitle": "Pick an algorithm, set regularization, then run. Difference methods need both reference and target; absolute methods only need a target.",
}
