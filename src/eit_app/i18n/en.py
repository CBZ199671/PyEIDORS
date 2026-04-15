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
}
