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
    "menu.file.open_recordings": "Open &Recordings Folder",
    "menu.file.open_output": "Open &Output Folder",
    "menu.file.exit": "E&xit",
    # ------------------------------------------------------------------
    # Tools menu
    # ------------------------------------------------------------------
    "menu.view": "&View",
    "menu.view.theme_light": "&Light Theme",
    "menu.view.theme_dark": "&Dark Theme",
    # Compute precision moved from View to Tools (changes computation,
    # not what's drawn on screen).  New keys:
    "menu.tools.precision": "Compute &Precision",
    "menu.tools.precision_float32": "Float32 (Fast, AD 7-bit headroom)",
    "menu.tools.precision_float64": "Float64 (High Precision)",
    "main.status.precision_changed": "Compute precision switched to {mode}; effective next acquisition / solve.",
    "menu.tools": "&Tools",
    "menu.tools.interop_hub": "EIDORS &Interop Hub\u2026",
    "menu.tools.difference": "&Difference\u2026",
    "menu.tools.batch_reconstruction": "&Batch Reconstruction\u2026",
    "menu.tools.reconstruction": "&Reconstruction\u2026",
    "main.status.need_frames_for_difference": "Record at least 2 frames on the Hardware tab before using Difference.",
    "main.status.reconstruction_hint": "Switched to Database tab \u2014 select a reference and target frame, then click Reconstruct.",
    "main.status.recon_running": "Running {method}\u2026",
    "main.status.recon_failed": "Reconstruction failed: {error}",
    "main.status.recon_complete": "Reconstruction complete: {method}",
    "main.status.recon_save_ok": "Saved outputs to {folder}",
    "main.status.recon_save_failed": "Save failed: {error}",
    "main.popup.recon_complete.title": "Reconstruction complete",
    "main.popup.recon_complete.text": "Reconstruction saved successfully.",
    "main.popup.recon_complete.informative": "Output folder:\n{folder}",
    "main.popup.recon_complete.open_folder": "Open Folder",
    "main.popup.recon_complete.close": "Close",
    # ==================================================================
    # Loading / busy overlay messages (shared across plots)
    # ==================================================================
    "hw.live_plot.loading_overlay": "Waiting for device frames\u2026",
    "hw.reconstruction.loading_overlay": "Reconstructing\u2026",
    "voltage_plot.loading_overlay": "Computing voltages\u2026",
    "sim.results.ground_truth_loading": "Solving forward problem\u2026",
    "sim.results.reconstruction_loading": "Reconstructing\u2026",
    "sim.results.viewer3d_no_data": "No 3D data yet",
    "sim.results.viewer3d_loading": "Rendering 3D scene\u2026",
    "sim.results.viewer3d_unavailable": "PyVista / VTK is not installed; 3D mesh cannot be displayed.",
    "sim.results.viewer3d_embedded_disabled": "Embedded PyVista / VTK is disabled in this runtime to avoid Qt/OpenGL crashes. Using the safe built-in 3D renderer instead.",
    "sim.results.viewer3d_bad_mesh": "Mesh is not a supported 3D tetra/hex volume grid",
    "sim.results.viewer3d_size_mismatch": "Conductivity length does not match the mesh",
    "sim.results.viewer3d_display": "View",
    "sim.results.viewer3d_display_volume": "Volume mesh rendering",
    "sim.results.viewer3d_display_volume_short": "Vol",
    "sim.results.viewer3d_display_points": "Point-cloud rendering",
    "sim.results.viewer3d_display_points_short": "Points",
    "sim.results.viewer3d_anomaly_mode": "Anom.",
    "sim.results.viewer3d_anomaly_positive": "Highlight only positive deviations above the background median",
    "sim.results.viewer3d_anomaly_positive_short": "+",
    "sim.results.viewer3d_anomaly_negative": "Highlight only negative deviations below the background median",
    "sim.results.viewer3d_anomaly_negative_short": "-",
    "sim.results.viewer3d_anomaly_absolute": "Highlight positive and negative absolute deviations",
    "sim.results.viewer3d_anomaly_absolute_short": "|d|",
    "sim.results.viewer3d_opacity": "Opacity",
    "sim.results.viewer3d_opacity_short": "Alpha",
    "sim.results.viewer3d_highlight": "Highlight the selected anomaly sign",
    "sim.results.viewer3d_highlight_short": "Hi",
    "sim.results.viewer3d_wireframe": "Outline edges",
    "sim.results.viewer3d_wireframe_short": "Edges",
    "sim.results.viewer3d_reset": "Reset view",
    "sim.results.viewer3d_reset_short": "Reset",
    "sim.results.electrodes_toggle_short": "Elec.",
    "sim.results.electrodes_toggle": "Show electrodes",
    # ------------------------------------------------------------------
    # Help menu + About dialog.
    # ------------------------------------------------------------------
    "menu.help": "&Help",
    "menu.help.about": "&About EIT Workstation",
    "about.title": "About EIT Workstation",
    "about.brand_headline": "EIT Workstation \u00b7 Electrical Impedance Tomography",
    "about.version_line": "Version {version} \u00b7 {build}",
    "about.body": "Cross-platform PySide6 desktop app covering the full EIT workflow \u2014 hardware acquisition, simulation, dataset generation, and reconstruction. Bilingual zh / en, with PyVista 3D visualisation and PETSc / dolfinx solvers by default.",
    "about.credit": "",
    "about.close": "Close",
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
    "hw.connection.port_hint.scan_prompt": "Serial ports are scanned on demand to keep the GUI startup fast. Click Scan, or Connect to scan before opening the device.",
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
    "hw.summary.title": "Session Summary",
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
    "hw.equipotential.title": "Equipotential map",
    "hw.equipotential.empty_overlay": "No equipotential data yet",
    "hw.equipotential.no_surface": "Failed to extract 3D mesh surface",
    "hw.equipotential.bad_coords": "Invalid mesh coordinates",
    "hw.equipotential.size_mismatch": "Conductivity / mesh mismatch: \u03c3={sigma}, cells={cells}, nodes={nodes}",
    "hw.equipotential.height_label": "Height scale",
    "hw.equipotential.reset_button": "Reset view",
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
    "sim.step.inhom_2d": "Step 2 \u00b7 Inhomogeneity Areas",
    "sim.step.inhom_3d": "Step 2 \u00b7 Inhomogeneity Volumes",
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
    "sim.mesh.family_label": "3D cell type:",
    "sim.mesh.family.tetra": "Tetra (4-node)",
    "sim.mesh.family.hex": "Hex (8-node, fast GPU)",
    "sim.mesh.size_label": "Mesh size:",
    "sim.mesh.refinement_tooltip": "Smaller values produce finer meshes (more elements)",
    "sim.mesh.radius_label": "Radius:",
    "sim.mesh.radius_tooltip": "Radius of the 2D circular domain or 3D cylinder, in metres (m)",
    "sim.mesh.height_label": "Height:",
    "sim.mesh.height_tooltip": "Height of the 3D cylinder in metres; electrode rings are auto-distributed within this height (15%-85%).",
    "sim.mesh.electrodes_label": "Electrodes / ring:",
    "sim.mesh.rings_label": "Rings / layers:",
    "sim.mesh.electrode_length_label": "2D electrode length:",
    "sim.mesh.electrode_length_tooltip": "Boundary arc length for each 2D electrode, in metres; this is converted into mesh electrode coverage.",
    "sim.mesh.electrode_area_label": "3D electrode area:",
    "sim.mesh.electrode_area_tooltip": "Side-wall patch area for each 3D electrode, in square metres; this is converted into electrode height ratio.",
    "sim.mesh.electrode_layout_label": "3D numbering:",
    "sim.mesh.electrode_layout.ring_major": "Ring-major (EIDORS)",
    "sim.mesh.electrode_layout.zigzag": "Zigzag (legacy)",
    "sim.mesh.conductivity_label": "Background \u03c3:",
    "sim.mesh.patterns_header": "Drive & measurement pattern",
    "sim.mesh.patterns_hint": "Controls how the forward solver builds stim/meas pairs. Inverse reconstruction reuses the same pattern — keep these in sync with your hardware board.",
    "sim.mesh.measurement_protocol_label": "3D protocol (drive -> measure):",
    "sim.mesh.measurement_protocol.eidors_full_3d": "In-layer drive -> all-layer meas (standard 3D)",
    "sim.mesh.measurement_protocol.layer_local_2p5d": "Per-layer 2D -> sliced/interpolated 3D (2.5D)",
    "sim.mesh.measurement_protocol.cross_layer_full": "Cross-layer-only drive -> all-layer + vertical meas",
    "sim.mesh.measurement_protocol.hybrid_full_3d": "In-layer + cross-layer drive -> full 3D meas",
    "sim.mesh.measurement_protocol.custom": "Custom drive/meas matrices",
    "sim.mesh.measurement_protocol_hint.eidors_full_3d": "Drive current rotates within each layer; after each drive, same-layer voltage differences are measured on every layer. This matches the standard EIDORS multi-layer 3D pattern.",
    "sim.mesh.measurement_protocol_hint.layer_local_2p5d": "Each layer uses only its own electrodes for 2D drive and 2D measurement; layer results are then used as slices for 3D display or interpolation. No vertical voltage differences are measured.",
    "sim.mesh.measurement_protocol_hint.cross_layer_full": "Drive current is injected only between matching electrodes on adjacent layers; measurements include same-layer voltage differences plus vertical layer-to-layer voltage differences. This reproduces cross-plane-drive hardware, but usually gives weaker lateral localization for small 3D inclusions than the hybrid protocol.",
    "sim.mesh.measurement_protocol_hint.hybrid_full_3d": "Includes both in-layer drives on each ring and cross-layer drives between adjacent rings; measurements include same-layer and vertical voltage differences. This gives broader 3D coverage for small inclusions, at the cost of more samples and slower solves.",
    "sim.mesh.measurement_protocol_hint.custom": "Provide stim_matrix and meas_matrices manually to reproduce a real wiring scheme, fixed-layer drives, or any special acquisition protocol.",
    "sim.mesh.stim_pattern_label": "Stim pattern:",
    "sim.mesh.meas_pattern_label": "Meas pattern:",
    "sim.mesh.rotate_meas_check": "Rotate measurement with drive",
    "sim.mesh.use_meas_current_check": "Include drive-related electrodes",
    "sim.mesh.extra_neighbors_label": "Extra excluded neighbors:",
    "sim.mesh.custom_pattern_label": "Custom matrices JSON:",
    "sim.mesh.custom_pattern_placeholder": '{"stim_matrix": [[1, -1, 0, 0]], "meas_matrices": [[1, 0, -1, 0]]}',
    "sim.mesh.point_count_hint": "Expected boundary samples: {count}",
    # ==================================================================
    # Simulation tab — Step 2 Inhomogeneities
    # ==================================================================
    "sim.inhom.title": "Inhomogeneities",
    "sim.inhom.title_2d": "Inhomogeneity Areas",
    "sim.inhom.title_3d": "Inhomogeneity Volumes",
    "sim.inhom.col.shape": "Shape",
    # Single-character header labels keep the table readable in the
    # narrow context pane; units moved to a dedicated hint line above
    # the table — see sim.inhom.units_hint.
    "sim.inhom.col.x": "X",
    "sim.inhom.col.y": "Y",
    "sim.inhom.col.z": "Z",
    # X / Y / Z axis sizes — labelled L / W / H so the column reads
    # "length / width / height" left-to-right (X = length, Y = width,
    # Z = height), matching the convention used in the Chinese
    # translation.
    "sim.inhom.col.sizex": "L",
    "sim.inhom.col.sizey": "W",
    "sim.inhom.col.sizez": "H",
    "sim.inhom.col.conductivity": "\u03c3",
    "sim.inhom.units_hint": "Coords / full sizes in metres; \u03c3 in S/m.",
    "sim.inhom.boundary_warning": "Rows {rows} exceed the domain; the forward problem clips them to the in-domain volume and the inverse result may show strong artifacts.",
    "sim.inhom.add_circle": "+ Circle",
    "sim.inhom.add_ellipse": "+ Ellipse",
    "sim.inhom.add_rectangle": "+ Rectangle",
    "sim.inhom.add_sphere": "+ Sphere",
    "sim.inhom.add_ellipsoid": "+ Ellipsoid",
    "sim.inhom.add_box": "+ Box",
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
    "sim.inverse.alpha_tooltip": (
        "Applied by absolute GN and the debug full-GN route; RM/single-step "
        "routes do not use it."
    ),
    "sim.inverse.lambda_eff_locked_label": "\u03bb_eff (locked):",
    "sim.inverse.lambda_eff_locked_tooltip": (
        "Defaults to fixed \u03bb_eff=1e-2 for single-step/RM routes; the "
        "one-step formula uses hp^2 RtR with hp=0.1. To change it, enable "
        "the advanced custom option and rebuild the RM."
    ),
    "sim.inverse.lambda_eff_custom_label": "\u03bb_eff (custom):",
    "sim.inverse.custom_lambda_check": "Advanced: custom \u03bb_eff",
    "sim.inverse.custom_lambda_tooltip": (
        "When enabled, the entered \u03bb_eff builds/loads a separate RM artifact; "
        "the first run is a cold build and will be noticeably slower."
    ),
    "sim.inverse.artifact_weight_label": "Artifact weight:",
    "sim.inverse.artifact_weight_tooltip": (
        "GREIT weighting is stored in the HDF5 artifact; this value is not "
        "used as \u03b1."
    ),
    "sim.inverse.artifact_nf1_tooltip": (
        "GREIT uses EIDORS NF=1 automatic weight search; this value is not "
        "used as \u03b1, and cold builds will be slower."
    ),
    "sim.inverse.greit.group_title": "GREIT Advanced Parameters",
    "sim.inverse.greit.desired_label": "Desired image:",
    "sim.inverse.greit.desired.center": "Center sample",
    "sim.inverse.greit.desired.gauss": "Gauss integration",
    "sim.inverse.greit.desired.adaptive_gauss": "Adaptive integration",
    "sim.inverse.greit.desired.sobol_qmc": "Sobol-QMC",
    "sim.inverse.greit.target_count_label": "Training targets:",
    "sim.inverse.greit.target_count_tooltip": (
        "0 lets the GUI choose the reconstruction grid from the electrode "
        "count; larger counts make cold builds slower."
    ),
    "sim.inverse.greit.target_size_label": "Target radius (R ratio):",
    "sim.inverse.greit.target_size_tooltip": (
        "EIDORS/GREIT target_size semantics: fraction of the tank radius R; "
        "for example 0.20 means 0.2R."
    ),
    "sim.inverse.greit.weight_strategy_label": "Weight strategy:",
    "sim.inverse.greit.weight_strategy.fixed": "Fixed weight",
    "sim.inverse.greit.weight_strategy.eidors_nf1": "EIDORS NF=1 auto search",
    "sim.inverse.greit.weight_strategy_tooltip": (
        "Fixed weight uses the value below directly; EIDORS NF=1 searches the "
        "weight during a cold RM build so the noise figure is close to 1."
    ),
    "sim.inverse.greit.weight_label": "Weight / NF:",
    "sim.inverse.greit.weight_tooltip": (
        "GREIT RM training weight / regularization strength; changing it "
        "selects or builds a different artifact."
    ),
    "sim.inverse.greit.use_cache_check": "Use cached RM",
    "sim.inverse.greit.cache_tooltip": (
        "When enabled, the registry first reuses an exact-signature GREIT RM artifact."
    ),
    "sim.inverse.greit.rebuild_check": "Rebuild RM",
    "sim.inverse.greit.rebuild_tooltip": (
        "Ignore any existing artifact and cold-build the current GREIT RM; "
        "this is noticeably slower."
    ),
    "sim.inverse.greit.cold_build_hint": (
        "Changing desired image, target count, target radius, weight strategy, "
        "or weight/NF changes the GREIT signature. The first run cold-builds "
        "the RM; NF=1 auto search is slower."
    ),
    "sim.inverse.iterations_label": "Max iterations:",
    "sim.inverse.iterations_tooltip": (
        "Used only by the absolute GN route; difference/RM algorithms are "
        "single-step or cached-matrix routes and do not accept this parameter."
    ),
    "sim.inverse.reconstruct_button": "Reconstruct",
    "sim.inverse.save_button": "Save Results",
    "sim.inverse.status_reconstructing": "Reconstructing\u2026",
    "sim.inverse.method.debug_fine_mesh_noser.tooltip": (
        "Debug baseline: cold-builds the fine-mesh dense NOSER context and is "
        "slower/noisier than the v1 RM hot path."
    ),
    "sim.inverse.method.noser_rm.tooltip": (
        "NOSER RM default: cold-build or reuse an HDF5 coarse inverse-model "
        "artifact, then reconstruct with the RM @ dv hot path."
    ),
    "sim.inverse.method.laplace_rm.tooltip": (
        "Laplace RM smooth route: cold-build or reuse an HDF5 graph-Laplacian "
        "artifact, then reconstruct with the RM @ dv hot path."
    ),
    "sim.inverse.method.curvature_rm.tooltip": (
        "Curvature RM smooth route: cold-build or reuse an HDF5 graph-LtL "
        "artifact, then reconstruct with the RM @ dv hot path."
    ),
    "sim.inverse.method.greit.tooltip": (
        "GREIT route: builds or reuses an HDF5 artifact from the current 2D/3D "
        "mesh, electrodes, protocol, and advanced training parameters, then "
        "reconstructs with the RM @ dv hot path."
    ),
    "sim.inverse.method.greit3d_rm.tooltip": (
        "Legacy GREIT route name; the GUI now shows the unified greit method."
    ),
    "sim.inverse.method.absolute_gn.tooltip": (
        "Absolute imaging: estimates absolute conductivity directly from the "
        "target boundary voltages without a reference frame; iterative full-GN "
        "cold path, suitable for small-mesh comparisons."
    ),
    "sim.inverse.method.debug_full_gn.tooltip": (
        "Debug baseline: iterative full GN cold path; useful for comparison, not "
        "the realtime RM route."
    ),
    # ==================================================================
    # Simulation tab — Right-side Metrics panel
    # ==================================================================
    "sim.metrics.title": "Mesh & Metrics",
    "sim.metrics.truth_mesh_label": "Truth mesh:",
    "sim.metrics.recon_mesh_label": "Recon mesh:",
    "sim.metrics.mesh_value": "{nodes} nodes / {elements} elements",
    "sim.metrics.l2_label": "Relative L2 error:",
    "sim.metrics.correlation_label": "Correlation:",
    "sim.metrics.rmse_label": "RMSE:",
    # ==================================================================
    # Simulation tab — Centre results widget
    # ==================================================================
    "sim.results.ground_truth_title": "Ground Truth",
    "sim.results.reconstruction_title": "Reconstruction",
    "sim.results.save_dialog_title": "Save Simulation Results",
    "sim.results.save_dialog_filter": "HDF5 package (*.h5 *.hdf5)",
    "sim.results.save_status": "Saved HDF5 package to {path}",
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
    "dataset.artifacts.item1": "mesh_info.h5 with node coordinates, cell connectivity, and homogeneous voltages",
    "dataset.artifacts.item2": "sample_000000.h5 style per-sample conductivity and boundary-voltage pairs",
    "dataset.artifacts.item3": "The configured output directory becomes a self-contained HDF5 dataset package",
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
    "db.filters.hint": "Search the archive by name, frequency, electrodes, stim amp, or date.",
    "db.filters.name_label": "Name:",
    "db.filters.name_placeholder": "tank, test_for_gui \u2026",
    "db.filters.freq_label": "Frequency (Hz):",
    "db.filters.freq_min_placeholder": "min",
    "db.filters.freq_max_placeholder": "max",
    "db.filters.date_any": "Any",
    "db.filters.date_from_label": "Date from:",
    "db.filters.date_to_label": "Date to:",
    "db.filters.n_elec_label": "Electrodes:",
    "db.filters.n_elec_min_placeholder": "min",
    "db.filters.n_elec_max_placeholder": "max",
    "db.filters.stim_amp_label": "Stim amp (\u00b5A):",
    "db.filters.stim_amp_min_placeholder": "min",
    "db.filters.stim_amp_max_placeholder": "max",
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
    "dlg.reconstruction.lambda_eff_label": "\u03bb_eff:",
    "dlg.reconstruction.custom_lambda_check": "Advanced: rebuild RM with custom \u03bb_eff (slower)",
    "dlg.reconstruction.custom_lambda_tip": "Uses the entered \u03bb_eff to cold-build or load a separate RM artifact.",
    "dlg.reconstruction.lambda_locked_tip": "One-step/RM difference routes use the canonical locked \u03bb_eff=1e-2 by default.",
    "dlg.reconstruction.iter_label": "Max iterations:",
    "dlg.reconstruction.output_group": "OUTPUT (OPTIONAL)",
    "dlg.reconstruction.output_placeholder": "Leave empty to only display the result (not save)",
    "dlg.reconstruction.browse_button": "Browse\u2026",
    "dlg.reconstruction.output_folder_label": "Output folder:",
    "dlg.reconstruction.save_image_check": "Save reconstruction image (PNG)",
    "dlg.reconstruction.save_voltage_check": "Save boundary voltage fit plot (PNG)",
    "dlg.reconstruction.not_selected": "<not selected>",
    "dlg.reconstruction.absolute_no_ref_tip": "Absolute methods do not use a reference frame.",
    "dlg.recon_settings.toggle_show": "Show forward/inverse parameters",
    "dlg.recon_settings.toggle_hide": "Hide forward/inverse parameters",
    "dlg.recon_settings.tab_mesh": "Mesh && Electrodes",
    "dlg.recon_settings.tab_protocol": "Stimulation && Measurement",
    "dlg.recon_settings.tab_solver": "Solver && Runtime",
    "dlg.recon_settings.mesh_dimension": "Domain:",
    "dlg.recon_settings.mesh_refinement": "Mesh size/refinement:",
    "dlg.recon_settings.rm_inverse_mesh": "RM inverse mesh size:",
    "dlg.recon_settings.n_elec": "Electrodes per ring:",
    "dlg.recon_settings.n_rings": "Rings:",
    "dlg.recon_settings.electrode_layout": "Electrode layout:",
    "dlg.recon_settings.radius": "Radius:",
    "dlg.recon_settings.height": "Height:",
    "dlg.recon_settings.geometry_scale": "Geometry scale to m:",
    "dlg.recon_settings.electrode_coverage": "Electrode coverage:",
    "dlg.recon_settings.electrode_length": "Electrode length:",
    "dlg.recon_settings.electrode_area": "Electrode area:",
    "dlg.recon_settings.electrode_height_ratio": "Electrode height ratio:",
    "dlg.recon_settings.stim_pattern": "Stim pattern:",
    "dlg.recon_settings.meas_pattern": "Meas pattern:",
    "dlg.recon_settings.measurement_protocol": "Measurement protocol:",
    "dlg.recon_settings.rotate_meas": "Rotate measurement order",
    "dlg.recon_settings.use_meas_current": "Include measurement-current electrodes",
    "dlg.recon_settings.use_meas_current_next": "Meas-current next:",
    "dlg.recon_settings.stim_direction": "Stim direction:",
    "dlg.recon_settings.meas_direction": "Meas direction:",
    "dlg.recon_settings.stim_first_positive": "First stimulation electrode is positive",
    "dlg.recon_settings.drive_mode": "Drive mode:",
    "dlg.recon_settings.drive_value": "Drive value:",
    "dlg.recon_settings.contact_impedance": "Contact impedance:",
    "dlg.recon_settings.solver_mode": "Inverse solver mode:",
    "dlg.recon_settings.linear_solver": "Linear solver:",
    "dlg.recon_settings.preconditioner": "Preconditioner:",
    "dlg.recon_settings.jacobian_representation": "Jacobian representation:",
    "dlg.recon_settings.forward_solver_preset": "Forward solver preset:",
    "dlg.recon_settings.forward_mat_solve": "Forward mat-solve:",
    "dlg.recon_settings.petsc_device": "PETSc device:",
    "dlg.recon_settings.runtime_device": "Runtime device:",
    "dlg.recon_settings.acceleration_profile": "Acceleration profile:",
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
    "dlg.batch.lambda_eff_label": "\u03bb_eff:",
    "dlg.batch.custom_lambda_check": "Advanced: rebuild RM with custom \u03bb_eff (slower)",
    "dlg.batch.custom_lambda_tip": "Uses the entered \u03bb_eff to cold-build or load a separate RM artifact for the batch.",
    "dlg.batch.lambda_locked_tip": "One-step/RM difference routes use the canonical locked \u03bb_eff=1e-2 by default.",
    "dlg.batch.iter_label": "Max iterations:",
    "dlg.batch.ref_browse_button": "Browse\u2026",
    "dlg.batch.ref_label": "Reference frame:",
    "dlg.batch.outputs_group": "OUTPUTS",
    "dlg.batch.save_image_check": "Save reconstruction image (PNG)",
    "dlg.batch.progress_group": "PROGRESS",
    "dlg.batch.ready": "Ready to run.",
    "dlg.batch.cancelling": "Cancelling\u2026",
    "dlg.batch.progress_default": "{current}/{total}",
    "dlg.batch.progress_with_eta": "{current}/{total}  \u00b7  ETA {eta}",
    "dlg.batch.eta_seconds": "{seconds}s remaining",
    "dlg.batch.eta_minutes": "{minutes}m {seconds}s remaining",
    "dlg.batch.eta_hours": "{hours}h {minutes}m remaining",
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
    # ==================================================================
    # Dialog — Interop Hub (EIDORS ↔ PyEIDORS migration workbench)
    # ==================================================================
    "dlg.interop.title": "Interop Hub",
    "dlg.interop.intro": "A visual, reviewable, reversible workflow for migrating between EIDORS and PyEIDORS.",
    # Tab labels
    "dlg.interop.tabs.import": "Import from EIDORS",
    "dlg.interop.tabs.export": "Export to EIDORS",
    "dlg.interop.tabs.profiles": "Profiles & Paths",
    # Shared — path pick button
    "dlg.interop.path_pick_button": "Pick\u2026",
    # Manual status panel (top of Import tab)
    "dlg.interop.status.title": "Current manual selections",
    "dlg.interop.status.unspecified": "Not set",
    "dlg.interop.status.pending": "Pending",
    "dlg.interop.status.specified": "Set",
    "dlg.interop.status.not_selected": "Not chosen",
    "dlg.interop.status.not_found": "Not found",
    "dlg.interop.status.ready_fmt": "Ready ({suffix})",
    "dlg.interop.status.ready": "Ready",
    "dlg.interop.status.failed": "Failed",
    # Step 1 — Environment
    "dlg.interop.env.title": "Step 1 \u00b7 Environment",
    "dlg.interop.env.hint": "Click \u201cPick\u2026\u201d to set MATLAB and startup.m manually. The unified file picker shows the Linux / WSL / Windows locations reachable from the current environment. Environment profiles can be managed on the Profiles & Paths tab.",
    "dlg.interop.env.matlab_label": "MATLAB:",
    "dlg.interop.env.matlab_placeholder": "Path to matlab executable",
    "dlg.interop.env.pick_matlab_title": "Select MATLAB executable",
    "dlg.interop.env.matlab_filter": "Executable (*.exe *.bin *.sh);;All files (*)",
    "dlg.interop.env.startup_label": "EIDORS startup:",
    "dlg.interop.env.startup_placeholder": "Path to startup.m",
    "dlg.interop.env.pick_startup_title": "Select EIDORS startup.m",
    "dlg.interop.env.startup_filter": "MATLAB script (*.m);;All files (*)",
    "dlg.interop.env.manual_entry": "Current manual input",
    "dlg.interop.env.saved_default_name": "Saved EIDORS Environment",
    # Step 2 — Source
    "dlg.interop.source.title": "Step 2 \u00b7 Source",
    "dlg.interop.source.label": "Source:",
    "dlg.interop.source.placeholder": "Pick an EIDORS .m script, bridge directory, legacy .mat, or bridge JSON",
    "dlg.interop.source.pick_title": "Select EIDORS script, bridge file, or bridge directory",
    "dlg.interop.source.pick_filter": "Supported (*.m *.mat *.json);;MATLAB script (*.m);;MAT file (*.mat);;JSON (*.json);;All files (*)",
    "dlg.interop.source.capture_label": "Capture output:",
    "dlg.interop.source.pick_capture_title": "Select bridge capture output directory",
    "dlg.interop.source.hint": "Three kinds of source are supported: user scripts, existing bridge projects, and legacy geometry .mat files.",
    # Step 3 — Capture & preview actions
    "dlg.interop.actions.title": "Step 3 \u00b7 Capture & preview",
    "dlg.interop.actions.preview_button": "Generate preview",
    "dlg.interop.actions.reload_button": "Reload last result",
    "dlg.interop.actions.no_preview_yet": "No migration preview yet.",
    # Step 4 — Preview & import
    "dlg.interop.preview.title": "Step 4 \u00b7 Preview & import",
    "dlg.interop.preview.waiting": "Waiting for bridge package preview.",
    "dlg.interop.preview.source_col_header": "EIDORS source",
    "dlg.interop.preview.value_col_header": "Value",
    "dlg.interop.preview.mapping_col_header": "PyEIDORS mapping",
    "dlg.interop.preview.warnings_placeholder": "Warnings and unresolved fields will appear here.",
    "dlg.interop.preview.missing_fallback": "Fill in manually, or wrap with a bridge template script.",
    "dlg.interop.preview.overview": "EIDORS \u2192 PyEIDORS mapping preview: {dim}, {n_elec} electrodes/ring, {pts} boundary-voltage points.",
    "dlg.interop.preview.counts": "Recognized: {recognized}  |  Inferred: {inferred}  |  Missing: {missing}",
    "dlg.interop.preview.no_warnings": "No high-risk items requiring manual review.",
    "dlg.interop.preview.done": "Preview complete: {dim}  |  {n_elec} electrodes/ring  |  {pts} boundary-voltage points.",
    "dlg.interop.preview.smoke_placeholder": "Inverse-problem smoke test output will appear here.",
    # Import target combo
    "dlg.interop.import_target.hardware": "Hardware config template",
    "dlg.interop.import_target.simulation": "Simulation config",
    "dlg.interop.import_target.dataset": "Dataset config",
    "dlg.interop.import_target.measurements": "Boundary voltages only",
    "dlg.interop.import_target.geometry": "Geometry assets only",
    "dlg.interop.auto_smoke_check": "Auto-run inverse-problem smoke test after import",
    "dlg.interop.import_button": "Import into PyEIDORS",
    "dlg.interop.smoke_button": "Run smoke validation",
    # Export tab
    "dlg.interop.export.title": "Export to EIDORS",
    "dlg.interop.export.source.simulation": "Current simulation config",
    "dlg.interop.export.source.hardware": "Current hardware layout",
    "dlg.interop.export.source.recording": "Current recording / reconstruction",
    "dlg.interop.export.source_label": "Source:",
    "dlg.interop.export.output_label": "Output dir:",
    "dlg.interop.export.pick_output_title": "Select bridge export directory",
    "dlg.interop.export.hint": "When exporting a bridge project, the currently-set MATLAB / startup.m paths are baked in. If neither is set, only the data and configuration files are exported.",
    "dlg.interop.export.include_label": "Include:",
    "dlg.interop.export.include_geometry": "Geometry",
    "dlg.interop.export.include_data": "Boundary voltages",
    "dlg.interop.export.include_scripts": "Runnable EIDORS script",
    "dlg.interop.export.generate_button": "Generate Bridge Project",
    "dlg.interop.export.log_placeholder": "Export notes, generated paths, and any fallback behavior will be logged here.",
    "dlg.interop.export.success": "[OK] Bridge project generated: {root}",
    "dlg.interop.export.source_tag": "      Source: {source_kind}",
    # Profiles & Paths tab
    "dlg.interop.profiles.group_title": "Saved environments",
    "dlg.interop.profiles.name_label": "Name:",
    "dlg.interop.profiles.matlab_label": "MATLAB:",
    "dlg.interop.profiles.startup_label": "startup.m:",
    "dlg.interop.profiles.script_label": "Last script:",
    "dlg.interop.profiles.output_label": "Last output:",
    "dlg.interop.profiles.save_button": "Save current environment",
    "dlg.interop.profiles.remove_button": "Remove selected",
    "dlg.interop.profiles.note": "Saved here are EIDORS environment profiles only \u2014 your original MATLAB project is never modified.",
    "dlg.interop.profiles.unnamed": "Unnamed EIDORS Environment",
    "dlg.interop.profiles.custom_default": "Custom EIDORS Environment",
    "dlg.interop.profiles.manual_name": "Manual Environment",
    # Status-bar / message-box text
    "dlg.interop.msg.no_source": "Please select an EIDORS script or bridge package source first.",
    "dlg.interop.msg.missing_before_script": "Before running an EIDORS script, please set: {parts}.",
    "dlg.interop.msg.missing_joiner": ", ",
    "dlg.interop.msg.preview_failed": "Preview failed: {error}",
    "dlg.interop.msg.no_bundle": "No bridge package loaded.",
    "dlg.interop.msg.no_callback_import": "This window has no import callback wired up.",
    "dlg.interop.msg.no_callback_smoke": "This window has no smoke-test callback wired up.",
    "dlg.interop.msg.no_callback_export": "This window has no export data provider wired up.",
    "dlg.interop.msg.no_snapshot": "The current source has no exportable context right now.",
    "dlg.interop.msg.import_failed": "Import failed: {error}",
    "dlg.interop.msg.smoke_failed": "Smoke test failed: {error}",
    "dlg.interop.msg.smoke_no_bundle": "No bridge package available for smoke testing.",
    "dlg.interop.msg.export_failed": "Export failed: {error}",
    "dlg.interop.msg.bundle_no_preview": "No bridge package has been loaded yet.",
    "dlg.interop.msg.profile_saved": "Profile saved: {name}",
    "dlg.interop.msg.profile_removed": "Profile removed: {name}",
    # ==================================================================
    # Visual path picker (pick_visual_path)
    # ==================================================================
    "path_picker.sidebar.wsl_home": "WSL home",
    "path_picker.sidebar.wsl_root": "WSL root",
    "path_picker.sidebar.windows_home": "Windows user",
    "path_picker.sidebar.linux_home": "Linux home",
    "path_picker.sidebar.linux_root": "Linux root",
    "path_picker.label.look_in": "Look in:",
    "path_picker.label.file_name": "Name:",
    "path_picker.label.file_type": "Type:",
    "path_picker.label.accept": "Choose",
    "path_picker.label.reject": "Cancel",
    "path_picker.button.choose_current_folder": "Use this folder",
}
