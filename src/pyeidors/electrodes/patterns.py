from __future__ import annotations

import numpy as np

from ..data.structures import PatternConfig
from ..physics.current_drive import (
    build_stim_currents,
    validate_drive_config,
)


class StimMeasPatternManager:
    """Stimulation and measurement pattern manager."""

    def __init__(
        self,
        config: PatternConfig,
        electrode_lengths_m: np.ndarray | None = None,
        mesh_tdim: int | None = None,
    ) -> None:
        self.config = config
        self.n_elec = config.n_elec
        self.n_rings = config.n_rings
        self.tn_elec = self.n_elec * self.n_rings
        self.stim_direction = 1 if self.config.stim_direction.lower() == "ccw" else -1
        self.meas_direction = 1 if self.config.meas_direction.lower() == "ccw" else -1
        self.drive_mode = validate_drive_config(
            drive_mode=self.config.drive_mode,
            drive_value=self.config.drive_value,
            geometry_scale_to_m=self.config.geometry_scale_to_m,
            mesh_tdim=mesh_tdim,
        )
        self.drive_value = float(self.config.drive_value)
        self._electrode_lengths_m = self._resolve_electrode_lengths(electrode_lengths_m)
        self.measurement_protocol = self._normalize_measurement_protocol(
            getattr(self.config, "measurement_protocol", "eidors_full_3d")
        )
        if self.n_rings <= 1 and self.measurement_protocol != "custom":
            self.measurement_protocol = "eidors_full_3d"

        self._full_meas_matrices: list[np.ndarray] = []
        if self.measurement_protocol == "custom":
            self._load_custom_patterns()
        else:
            self._parse_patterns()
            self._generate_patterns()
        self._build_measurement_projection()
        self._compute_measurement_selector()

    @staticmethod
    def _normalize_measurement_protocol(value: object | None) -> str:
        raw = str(value or "eidors_full_3d").strip().lower().replace("-", "_")
        aliases = {
            "2.5d": "layer_local_2p5d",
            "25d": "layer_local_2p5d",
            "layer_local": "layer_local_2p5d",
            "layer_local_2p5d": "layer_local_2p5d",
            "planar": "layer_local_2p5d",
            "ring_local": "layer_local_2p5d",
            "full_3d": "eidors_full_3d",
            "eidors_full": "eidors_full_3d",
            "eidors_full_3d": "eidors_full_3d",
            "true_3d": "eidors_full_3d",
            "cross_layer": "cross_layer_full",
            "cross_layer_full": "cross_layer_full",
            "hybrid": "hybrid_full_3d",
            "hybrid_full": "hybrid_full_3d",
            "hybrid_3d": "hybrid_full_3d",
            "hybrid_full_3d": "hybrid_full_3d",
            "mixed_3d": "hybrid_full_3d",
            "custom": "custom",
            "arbitrary": "custom",
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise ValueError(
                "Unsupported measurement_protocol "
                f"{value!r}; expected layer_local_2p5d, eidors_full_3d, "
                "cross_layer_full, hybrid_full_3d, or custom."
            ) from exc

    def _resolve_electrode_lengths(
        self, electrode_lengths_m: np.ndarray | None
    ) -> np.ndarray | None:
        if electrode_lengths_m is None:
            if self.drive_mode == "line_current_density":
                # Allow pattern construction without mesh in metadata-only paths.
                return np.ones(self.tn_elec, dtype=float)
            return None

        lengths = np.asarray(electrode_lengths_m, dtype=float).reshape(-1)
        if lengths.size == self.n_elec and self.n_rings > 1:
            lengths = np.tile(lengths, self.n_rings)
        if lengths.size != self.tn_elec:
            raise ValueError(
                "electrode_lengths_m size mismatch: "
                f"expected {self.tn_elec}, got {lengths.size}."
            )
        if np.any(lengths <= 0.0):
            bad = int(np.nonzero(lengths <= 0.0)[0][0])
            raise ValueError(
                f"electrode_lengths_m[{bad}] must be positive, got {lengths[bad]!r}."
            )
        return lengths

    def _parse_patterns(self) -> None:
        # Parse stimulation pattern
        if isinstance(self.config.stim_pattern, str):
            if self.config.stim_pattern == "{ad}":
                self.inj_electrodes = [0, 1]
            elif self.config.stim_pattern == "{op}":
                self.inj_electrodes = [0, self.n_elec // 2]
            else:
                raise ValueError(
                    f"Unknown stimulation pattern: {self.config.stim_pattern}"
                )
        else:
            self.inj_electrodes = self.config.stim_pattern

        if len(self.inj_electrodes) == 2:
            if self.config.stim_first_positive:
                self.inj_weights = np.array([1, -1])
            else:
                self.inj_weights = np.array([-1, 1])
        else:
            self.inj_weights = np.array([1])

        # Parse measurement pattern
        if isinstance(self.config.meas_pattern, str):
            if self.config.meas_pattern == "{ad}":
                self.meas_electrodes = [0, 1]
            elif self.config.meas_pattern == "{op}":
                self.meas_electrodes = [0, self.n_elec // 2]
            else:
                raise ValueError(
                    f"Unknown measurement pattern: {self.config.meas_pattern}"
                )
        else:
            self.meas_electrodes = self.config.meas_pattern

        self.meas_weights = (
            np.array([1, -1]) if len(self.meas_electrodes) == 2 else np.array([1])
        )

    def _generate_patterns(self) -> None:
        self.stim_matrix = []
        self.meas_matrices = []
        self._full_meas_matrices = []
        self.meas_start_indices = []
        self.n_meas_total = 0
        self.n_meas_per_stim = []

        for elec, ring, inj_indices in self._iter_stimulation_records():
            # Stimulation vector
            stim_vec = np.zeros(self.tn_elec)

            stim_currents = build_stim_currents(
                drive_mode=self.drive_mode,
                drive_value=self.drive_value,
                inj_indices=inj_indices,
                inj_weights=self.inj_weights,
                electrode_lengths_m=self._electrode_lengths_m,
            )
            for idx, current in zip(inj_indices, stim_currents):
                stim_vec[idx] = current

            # Measurement matrix
            full_meas_mat = self._make_meas_matrix_for_protocol(elec, ring)
            meas_mat = full_meas_mat

            if not self.config.use_meas_current:
                meas_mat = self._filter_measurements(meas_mat, inj_indices)

            if meas_mat.shape[0] > 0:
                self.stim_matrix.append(stim_vec)
                self.meas_matrices.append(meas_mat)
                self._full_meas_matrices.append(full_meas_mat)
                self.meas_start_indices.append(self.n_meas_total)
                self.n_meas_per_stim.append(meas_mat.shape[0])
                self.n_meas_total += meas_mat.shape[0]

        self.stim_matrix = np.array(self.stim_matrix)
        self.n_stim = len(self.stim_matrix)

    def _iter_stimulation_records(self) -> list[tuple[int, int, list[int]]]:
        records: list[tuple[int, int, list[int]]] = []
        if (
            self.measurement_protocol in {"cross_layer_full", "hybrid_full_3d"}
            and self.n_rings > 1
        ):
            if self.measurement_protocol == "hybrid_full_3d":
                for ring in range(self.n_rings):
                    for elec in range(self.n_elec):
                        inj_indices = []
                        for inj_elec in self.inj_electrodes:
                            idx = (
                                inj_elec + self.stim_direction * elec
                            ) % self.n_elec + ring * self.n_elec
                            inj_indices.append(idx)
                        records.append((elec, ring, inj_indices))

            for ring in range(self.n_rings - 1):
                for elec in range(self.n_elec):
                    lower = (
                        self.stim_direction * elec
                    ) % self.n_elec + ring * self.n_elec
                    upper = (self.stim_direction * elec) % self.n_elec + (
                        ring + 1
                    ) * self.n_elec
                    records.append((elec, ring, [lower, upper]))
            return records

        for ring in range(self.n_rings):
            for elec in range(self.n_elec):
                inj_indices = []
                for inj_elec in self.inj_electrodes:
                    idx = (
                        inj_elec + self.stim_direction * elec
                    ) % self.n_elec + ring * self.n_elec
                    inj_indices.append(idx)
                records.append((elec, ring, inj_indices))
        return records

    def _compute_measurement_selector(self) -> None:
        if self.config.use_meas_current:
            full_count = sum(int(mat.shape[0]) for mat in self._full_meas_matrices)
            self.meas_selector = np.ones(full_count, dtype=bool)
            return

        selector = []
        for i in range(self.n_stim):
            full_meas_mat = self._full_meas_matrices[i]
            filtered_meas_mat = self.meas_matrices[i]

            full_set_hash = self._create_meas_hash(full_meas_mat)
            filtered_set_hash = self._create_meas_hash(filtered_meas_mat)

            frame_selector = np.isin(full_set_hash, filtered_set_hash)
            selector.append(frame_selector)

        self.meas_selector = np.concatenate(selector)

    def _load_custom_patterns(self) -> None:
        if (
            self.config.custom_stim_matrix is None
            or self.config.custom_meas_matrices is None
        ):
            raise ValueError(
                "measurement_protocol='custom' requires custom_stim_matrix "
                "and custom_meas_matrices."
            )
        stim_matrix = np.asarray(self.config.custom_stim_matrix, dtype=float)
        if stim_matrix.ndim == 1:
            stim_matrix = stim_matrix.reshape(1, -1)
        if stim_matrix.ndim != 2 or stim_matrix.shape[1] != self.tn_elec:
            raise ValueError(
                "custom_stim_matrix must have shape (n_stim, n_elec*n_rings); "
                f"got {stim_matrix.shape}, expected second dimension {self.tn_elec}."
            )
        meas_matrices = self._coerce_custom_meas_matrices(
            self.config.custom_meas_matrices,
            n_stim=stim_matrix.shape[0],
        )
        if len(meas_matrices) != stim_matrix.shape[0]:
            raise ValueError(
                "custom_meas_matrices count must match custom_stim_matrix rows: "
                f"{len(meas_matrices)} != {stim_matrix.shape[0]}."
            )
        self.stim_matrix = stim_matrix
        self.meas_matrices = []
        self._full_meas_matrices = []
        self.meas_start_indices = []
        self.n_meas_total = 0
        self.n_meas_per_stim = []
        for meas_mat in meas_matrices:
            mat = np.asarray(meas_mat, dtype=float)
            if mat.ndim != 2 or mat.shape[1] != self.tn_elec:
                raise ValueError(
                    "Each custom measurement matrix must have shape "
                    f"(n_meas, {self.tn_elec}); got {mat.shape}."
                )
            self.meas_matrices.append(mat)
            self._full_meas_matrices.append(mat)
            self.meas_start_indices.append(self.n_meas_total)
            self.n_meas_per_stim.append(mat.shape[0])
            self.n_meas_total += mat.shape[0]
        self.n_stim = int(stim_matrix.shape[0])

    @staticmethod
    def _coerce_custom_meas_matrices(
        custom_meas_matrices: object, *, n_stim: int
    ) -> list[np.ndarray]:
        """Normalize custom measurement matrices, including ragged per-stim lists."""
        if isinstance(custom_meas_matrices, (list, tuple)) and custom_meas_matrices:
            candidates = [
                np.asarray(item, dtype=float) for item in custom_meas_matrices
            ]
            if all(candidate.ndim == 2 for candidate in candidates):
                return candidates

        meas_payload = np.asarray(custom_meas_matrices, dtype=float)
        if meas_payload.ndim == 2:
            return [meas_payload.copy() for _ in range(n_stim)]
        if meas_payload.ndim == 3:
            return [np.asarray(item, dtype=float) for item in meas_payload]
        raise ValueError(
            "custom_meas_matrices must be either (n_meas, total_elec), "
            "(n_stim, n_meas, total_elec), or a list of per-stim 2D matrices."
        )

    def _build_measurement_projection(self) -> None:
        """Build a dense projection matrix to avoid per-pattern matmul loops."""
        projection = np.zeros(
            (self.n_meas_total, self.n_stim * self.tn_elec), dtype=float
        )
        for i, (start_idx, meas_mat) in enumerate(
            zip(self.meas_start_indices, self.meas_matrices)
        ):
            n_meas = meas_mat.shape[0]
            row_slice = slice(start_idx, start_idx + n_meas)
            col_start = i * self.tn_elec
            col_slice = slice(col_start, col_start + self.tn_elec)
            projection[row_slice, col_slice] = meas_mat
        self._meas_projection = projection

    def _create_meas_hash(self, meas_mat: np.ndarray) -> np.ndarray:
        if meas_mat.size == 0:
            return np.array([])

        pos_indices = np.argmax(meas_mat > 0, axis=1)
        neg_indices = np.argmax(meas_mat < 0, axis=1)

        pos_mask = np.any(meas_mat > 0, axis=1)
        neg_mask = np.any(meas_mat < 0, axis=1)

        hash_vals = (pos_indices * pos_mask) * 1e7 + (neg_indices * neg_mask)
        return hash_vals

    def _make_meas_matrix(self, elec: int, ring: int) -> np.ndarray:
        return self._make_meas_matrix_for_rings(elec, range(self.n_rings))

    def _make_meas_matrix_for_protocol(self, elec: int, ring: int) -> np.ndarray:
        if self.measurement_protocol == "layer_local_2p5d":
            return self._make_meas_matrix_for_rings(elec, [ring])
        if self.measurement_protocol in {"cross_layer_full", "hybrid_full_3d"}:
            same_layer = self._make_meas_matrix_for_rings(elec, range(self.n_rings))
            cross_layer = self._make_cross_layer_meas_matrix(elec)
            if cross_layer.size == 0:
                return same_layer
            return np.vstack([same_layer, cross_layer])
        return self._make_meas_matrix_for_rings(elec, range(self.n_rings))

    def _make_meas_matrix_for_rings(self, elec: int, rings: object) -> np.ndarray:
        meas_list = []
        offset = self.meas_direction * elec if self.config.rotate_meas else 0

        for ring in rings:
            ring_offset = int(ring) * self.n_elec
            for within_ring in range(self.n_elec):
                meas_vec = np.zeros(self.tn_elec)

                for i, meas_elec in enumerate(self.meas_electrodes):
                    idx = (meas_elec + within_ring + offset) % self.n_elec + ring_offset
                    meas_vec[idx] = self.meas_weights[i]

                meas_list.append(meas_vec)

        return np.array(meas_list)

    def _make_cross_layer_meas_matrix(self, elec: int) -> np.ndarray:
        if self.n_rings <= 1:
            return np.empty((0, self.tn_elec), dtype=float)
        meas_list = []
        offset = self.meas_direction * elec if self.config.rotate_meas else 0
        for ring in range(self.n_rings - 1):
            lower_offset = ring * self.n_elec
            upper_offset = (ring + 1) * self.n_elec
            for within_ring in range(self.n_elec):
                base = (within_ring + offset) % self.n_elec
                meas_vec = np.zeros(self.tn_elec)
                meas_vec[lower_offset + base] = 1.0
                meas_vec[upper_offset + base] = -1.0
                meas_list.append(meas_vec)
        return np.array(meas_list, dtype=float)

    def _filter_measurements(
        self,
        meas_mat: np.ndarray,
        stim_indices: list[int] | None = None,
        *,
        elec: int | None = None,
        ring: int | None = None,
    ) -> np.ndarray:
        if stim_indices is None:
            if elec is None:
                raise TypeError(
                    "_filter_measurements requires stim_indices or elec/ring"
                )
            ring_offset = int(ring or 0) * self.n_elec
            stim_indices = [
                (int(inj_elec) + self.stim_direction * int(elec)) % self.n_elec
                + ring_offset
                for inj_elec in self.inj_electrodes
            ]
        stim_indices = [int(idx) for idx in stim_indices]
        if self.config.use_meas_current_next > 0:
            extended = []
            for idx in stim_indices:
                base = idx % self.n_elec
                ring_base = idx - base
                for offset in range(
                    -self.config.use_meas_current_next,
                    self.config.use_meas_current_next + 1,
                ):
                    extended.append((base + offset) % self.n_elec + ring_base)
            stim_indices = list(set(extended))

        mask = ~np.any(meas_mat[:, stim_indices] != 0, axis=1)
        return meas_mat[mask]

    def get_stim_matrix(self) -> np.ndarray:
        return self.stim_matrix

    def apply_meas_pattern(self, electrode_voltages: np.ndarray) -> np.ndarray:
        voltages = np.asarray(electrode_voltages, dtype=float)
        if voltages.shape != (self.n_stim, self.tn_elec):
            raise ValueError(
                "electrode_voltages shape mismatch: "
                f"expected {(self.n_stim, self.tn_elec)}, got {voltages.shape}"
            )
        if not np.isfinite(voltages).all():
            raise FloatingPointError(
                "electrode_voltages contains non-finite values: "
                f"{self._finite_summary(voltages)}"
            )
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            measured = self._meas_projection @ voltages.reshape(-1)
        if not np.isfinite(measured).all():
            raise FloatingPointError(
                "Measurement projection produced non-finite values: "
                f"{self._finite_summary(measured)}"
            )
        return measured

    @staticmethod
    def _finite_summary(values: np.ndarray) -> str:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return "finite_count=0"
        return (
            f"finite_count={finite.size} "
            f"min={float(finite.min()):.6e} "
            f"max={float(finite.max()):.6e}"
        )
