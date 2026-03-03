import numpy as np
from typing import List, Union
from ..data.structures import PatternConfig


class StimMeasPatternManager:
    """Stimulation and measurement pattern manager."""

    def __init__(self, config: PatternConfig):
        self.config = config
        self.n_elec = config.n_elec
        self.n_rings = config.n_rings
        self.tn_elec = self.n_elec * self.n_rings
        self.stim_direction = 1 if self.config.stim_direction.lower() == 'ccw' else -1
        self.meas_direction = 1 if self.config.meas_direction.lower() == 'ccw' else -1
        
        self._parse_patterns()
        self._generate_patterns()
        self._compute_measurement_selector()
    
    def _parse_patterns(self):
        # Parse stimulation pattern
        if isinstance(self.config.stim_pattern, str):
            if self.config.stim_pattern == '{ad}':
                self.inj_electrodes = [0, 1]
            elif self.config.stim_pattern == '{op}':
                self.inj_electrodes = [0, self.n_elec // 2]
            else:
                raise ValueError(f"Unknown stimulation pattern: {self.config.stim_pattern}")
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
            if self.config.meas_pattern == '{ad}':
                self.meas_electrodes = [0, 1]
            elif self.config.meas_pattern == '{op}':
                self.meas_electrodes = [0, self.n_elec // 2]
            else:
                raise ValueError(f"Unknown measurement pattern: {self.config.meas_pattern}")
        else:
            self.meas_electrodes = self.config.meas_pattern
            
        self.meas_weights = np.array([1, -1]) if len(self.meas_electrodes) == 2 else np.array([1])
    
    def _generate_patterns(self):
        self.stim_matrix = []
        self.meas_matrices = []
        self.meas_start_indices = []
        self.n_meas_total = 0
        self.n_meas_per_stim = []
        
        for ring in range(self.n_rings):
            for elec in range(self.n_elec):
                # Stimulation vector
                stim_vec = np.zeros(self.tn_elec)
                for i, inj_elec in enumerate(self.inj_electrodes):
                    idx = (inj_elec + self.stim_direction * elec) % self.n_elec + ring * self.n_elec
                    stim_vec[idx] = self.config.amplitude * self.inj_weights[i]

                # Measurement matrix
                meas_mat = self._make_meas_matrix(elec, ring)
                
                if not self.config.use_meas_current:
                    meas_mat = self._filter_measurements(meas_mat, elec, ring)
                
                if meas_mat.shape[0] > 0:
                    self.stim_matrix.append(stim_vec)
                    self.meas_matrices.append(meas_mat)
                    self.meas_start_indices.append(self.n_meas_total)
                    self.n_meas_per_stim.append(meas_mat.shape[0])
                    self.n_meas_total += meas_mat.shape[0]
        
        self.stim_matrix = np.array(self.stim_matrix)
        self.n_stim = len(self.stim_matrix)
    
    def _compute_measurement_selector(self):
        if self.config.use_meas_current:
            self.meas_selector = np.ones(self.n_elec * self.n_stim, dtype=bool)
            return

        selector = []
        for i in range(self.n_stim):
            elec = i % self.n_elec
            ring = i // self.n_elec
            
            full_meas_mat = self._make_meas_matrix(elec, ring)
            filtered_meas_mat = self.meas_matrices[i]
            
            full_set_hash = self._create_meas_hash(full_meas_mat)
            filtered_set_hash = self._create_meas_hash(filtered_meas_mat)
            
            frame_selector = np.isin(full_set_hash, filtered_set_hash)
            selector.append(frame_selector)
        
        self.meas_selector = np.concatenate(selector)
    
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
        meas_list = []
        offset = self.meas_direction * elec if self.config.rotate_meas else 0
        
        for meas_idx in range(self.tn_elec):
            meas_vec = np.zeros(self.tn_elec)
            within_ring = meas_idx % self.n_elec
            ring_offset = (meas_idx // self.n_elec) * self.n_elec
            
            for i, meas_elec in enumerate(self.meas_electrodes):
                idx = (meas_elec + within_ring + offset) % self.n_elec + ring_offset
                meas_vec[idx] = self.meas_weights[i]
            
            meas_list.append(meas_vec)
        
        return np.array(meas_list)
    
    def _filter_measurements(self, meas_mat: np.ndarray, elec: int, ring: int) -> np.ndarray:
        stim_indices = []
        for inj_elec in self.inj_electrodes:
            idx = (inj_elec + self.stim_direction * elec) % self.n_elec + ring * self.n_elec
            stim_indices.append(idx)
        
        if self.config.use_meas_current_next > 0:
            extended = []
            for idx in stim_indices:
                base = idx % self.n_elec
                ring_base = idx - base
                for offset in range(-self.config.use_meas_current_next, 
                                  self.config.use_meas_current_next + 1):
                    extended.append((base + offset) % self.n_elec + ring_base)
            stim_indices = list(set(extended))
        
        mask = ~np.any(meas_mat[:, stim_indices] != 0, axis=1)
        return meas_mat[mask]
    
    def get_stim_matrix(self) -> np.ndarray:
        return self.stim_matrix
    
    def apply_meas_pattern(self, electrode_voltages: np.ndarray) -> np.ndarray:
        measurements = np.zeros(self.n_meas_total)
        
        for i, (start_idx, meas_mat) in enumerate(zip(self.meas_start_indices, self.meas_matrices)):
            n_meas = meas_mat.shape[0]
            measurements[start_idx:start_idx + n_meas] = meas_mat @ electrode_voltages[i]
        
        return measurements
