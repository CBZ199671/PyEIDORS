"""EIT Forward Model - Based on Complete Electrode Model."""

import numpy as np
import warnings
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized

# FEniCS imports
from fenics import *
from dolfin import as_backend_type

from ..data.structures import PatternConfig, EITData, EITImage
from ..electrodes.patterns import StimMeasPatternManager


class EITForwardModel:
    """EIT Forward Model - Based on Complete Electrode Model."""

    def __init__(self, n_elec: int, pattern_config: PatternConfig, z: np.ndarray, mesh):
        self.n_elec = n_elec
        self.z = np.array(z)
        self.mesh = mesh

        if self.z.size != self.n_elec:
            raise ValueError(f"Contact impedance length ({self.z.size}) does not match electrode count ({self.n_elec})")

        # Create stimulation/measurement pattern manager
        self.pattern_manager = StimMeasPatternManager(pattern_config)

        # Set up boundary conditions
        self.boundaries_mf = mesh.boundaries_mf
        self.association_table = mesh.association_table
        if self.boundaries_mf is None:
            raise ValueError("Mesh lacks electrode boundary markers, cannot assemble CEM")
        self.ds_electrodes = Measure("ds", domain=mesh, subdomain_data=self.boundaries_mf)

        # Resolve electrode boundary tags from association table and compute electrode lengths
        self.electrode_tags = self._resolve_electrode_tags()
        self.electrode_lengths = self._compute_electrode_lengths()

        # Create function spaces
        self.V = FunctionSpace(mesh, "Lagrange", 1)  # Potential
        self.V_sigma = FunctionSpace(mesh, "DG", 0)  # Conductivity

        # Get degrees of freedom count
        u_sol = Function(self.V)
        self.dofs = u_sol.vector().size()

        # Define variational forms
        self.u = TrialFunction(self.V)
        self.phi = TestFunction(self.V)

        # Assemble system matrix
        self.M = self._assemble_electrode_matrix()

    def _resolve_electrode_tags(self):
        """Extract boundary tags sorted by electrode index from association table."""
        electrode_map = {}

        if isinstance(self.association_table, dict):
            # Prefer "electrode_i" style keys
            for key, val in self.association_table.items():
                try:
                    tag_val = int(val)
                except (TypeError, ValueError):
                    continue
                
                if isinstance(key, str):
                    key_lower = key.lower()
                    if key_lower == "electrodes" and isinstance(val, dict):
                        for idx_str, tag in val.items():
                            try:
                                electrode_map[int(idx_str)] = int(tag)
                            except (TypeError, ValueError):
                                continue
                        continue
                    if key_lower.startswith("electrode"):
                        try:
                            idx = int(key_lower.split('_')[-1])
                            electrode_map[idx] = tag_val
                        except ValueError:
                            continue

        # Fallback for legacy numeric keys (assign electrode indices by sorted tag values)
        if len(electrode_map) < self.n_elec and isinstance(self.association_table, dict):
            candidates = []
            for key, val in self.association_table.items():
                try:
                    tag_val = int(val)
                except (TypeError, ValueError):
                    continue
                if isinstance(key, (int, np.integer)) and key >= 2:
                    candidates.append(tag_val)
            if candidates:
                for idx, tag_val in enumerate(sorted(set(candidates))[: self.n_elec], start=1):
                    electrode_map.setdefault(idx, tag_val)

        missing = [i for i in range(1, self.n_elec + 1) if i not in electrode_map]
        if missing:
            raise ValueError(
                f"Association table missing electrode tags {missing}, cannot assemble CEM"
            )

        return [electrode_map[i] for i in range(1, self.n_elec + 1)]

    def _compute_electrode_lengths(self):
        """Compute boundary measure (length or area) for each electrode."""
        lengths = {}
        for tag in self.electrode_tags:
            length = assemble(1 * self.ds_electrodes(tag))
            lengths[tag] = length
            if np.isclose(length, 0.0):
                warnings.warn(f"Electrode boundary tag {tag} has zero measure, check mesh markers", RuntimeWarning)
        return lengths

    def _assemble_electrode_matrix(self):
        """Assemble electrode-related system matrix."""
        b = 0
        for i, electrode_tag in enumerate(self.electrode_tags):
            b += 1 / self.z[i] * inner(self.u, self.phi) * self.ds_electrodes(electrode_tag)

        B = assemble(b)
        
        row, col, val = as_backend_type(B).mat().getValuesCSR()
        M = csr_matrix((val, col, row))
        
        M.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        M_lil = lil_matrix(M)

        for i, electrode_tag in enumerate(self.electrode_tags):
            c = -1 / self.z[i] * self.phi * self.ds_electrodes(electrode_tag)
            C_i = assemble(c).get_local()
            M_lil[self.dofs + i, :self.dofs] = C_i
            M_lil[:self.dofs, self.dofs + i] = C_i
            
            electrode_len = self.electrode_lengths.get(electrode_tag, 0.0)
            M_lil[self.dofs + i, self.dofs + i] = 1 / self.z[i] * electrode_len
            M_lil[self.dofs + self.n_elec, self.dofs + i] = 1
            M_lil[self.dofs + i, self.dofs + self.n_elec] = 1

        return csr_matrix(M_lil)

    def create_full_matrix(self, sigma: Function):
        """Build complete system matrix including conductivity."""
        a = inner(sigma * grad(self.u), grad(self.phi)) * dx
        A = assemble(a)
        
        row, col, val = as_backend_type(A).mat().getValuesCSR()
        scipy_A = csr_matrix((val, col, row))
        scipy_A.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)

        return scipy_A + self.M

    def forward_solve(self, sigma: Function, current_patterns=None):
        """Forward solve."""
        if current_patterns is None:
            # Use default stimulation pattern
            M_complete = self.create_full_matrix(sigma).tocsc()
            solver = factorized(M_complete)
            
            u_all = []
            U_all = np.zeros((self.pattern_manager.n_stim, self.n_elec))
            
            stim_matrix = self.pattern_manager.stim_matrix
            
            for i in range(self.pattern_manager.n_stim):
                rhs = np.zeros(self.dofs + self.n_elec + 1)
                for j in range(self.n_elec):
                    rhs[self.dofs + j] = stim_matrix[i, j]
                
                sol = solver(rhs)
                u_all.append(sol[:self.dofs])
                U_all[i, :] = sol[self.dofs:-1]
            
            return u_all, U_all
        else:
            # Use specified current patterns
            M_complete = self.create_full_matrix(sigma).tocsc()
            solver = factorized(M_complete)
            
            n_patterns = current_patterns.shape[1]
            u_all = []
            U_all = np.zeros((n_patterns, self.n_elec))
            
            for i in range(n_patterns):
                rhs = np.zeros(self.dofs + self.n_elec + 1)
                for j in range(self.n_elec):
                    rhs[self.dofs + j] = current_patterns[j, i]
                
                sol = solver(rhs)
                u_all.append(sol[:self.dofs])
                U_all[i, :] = sol[self.dofs:-1]
            
            return u_all, U_all

    def fwd_solve(self, img: EITImage):
        """Forward solve interface."""
        sigma = Function(self.V_sigma)
        sigma.vector()[:] = img.get_conductivity()

        u_all, U_all = self.forward_solve(sigma)

        # Apply measurement pattern
        meas = self.pattern_manager.apply_meas_pattern(U_all)

        # Create EIT data object
        data = EITData(
            meas=meas,
            stim_pattern=self.pattern_manager.stim_matrix,
            n_elec=self.n_elec,
            n_stim=self.pattern_manager.n_stim,
            n_meas=self.pattern_manager.n_meas_total,
            type='simulated'
        )
        
        return data, U_all
