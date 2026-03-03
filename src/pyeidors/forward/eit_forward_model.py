"""EIT Forward Model based on the Complete Electrode Model (CEM)."""

from __future__ import annotations

import warnings

import numpy as np
import ufl
from mpi4py import MPI
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import splu
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc

from ..data.structures import EITData, EITImage, EITMesh, PatternConfig
from ..electrodes.patterns import StimMeasPatternManager
from ..femx import create_ds_measure


class EITForwardModel:
    """EIT Forward Model - Complete Electrode Model assembly and solve."""

    def __init__(
        self,
        n_elec: int,
        pattern_config: PatternConfig,
        z: np.ndarray,
        mesh: EITMesh,
        linear_backend: str = "scipy",
    ):
        self.n_elec = n_elec
        self.z = np.asarray(z)
        if not isinstance(mesh, EITMesh):
            raise TypeError("EITForwardModel expects an EITMesh instance")
        self.eit_mesh = mesh
        self.mesh = mesh.mesh

        if self.mesh.comm.size != 1:
            raise RuntimeError(
                "PyEIDORS phase-2 migration currently supports MPI size=1 only. "
                "Use single-rank execution in this stage."
            )

        if self.z.size != self.n_elec:
            raise ValueError(
                f"Contact impedance length ({self.z.size}) does not match electrode count ({self.n_elec})"
            )

        self.pattern_manager = StimMeasPatternManager(pattern_config)
        self.linear_backend = linear_backend.lower()

        self.facet_tags = mesh.facet_tags
        self.association_table = mesh.association_table
        if self.facet_tags is None:
            raise ValueError("EITMesh lacks electrode facet tags, cannot assemble CEM")

        self.ds_electrodes = create_ds_measure(self.mesh, self.facet_tags)

        self.electrode_tags = self._resolve_electrode_tags()
        self.electrode_lengths = self._compute_electrode_lengths()

        self.V = fem.functionspace(self.mesh, ("Lagrange", 1))
        self.V_sigma = fem.functionspace(self.mesh, ("DG", 0))

        dofmap = self.V.dofmap.index_map
        self.dofs = int(dofmap.size_local * self.V.dofmap.index_map_bs)

        self.u = ufl.TrialFunction(self.V)
        self.phi = ufl.TestFunction(self.V)

        self.M = self._assemble_electrode_matrix()

    def _resolve_pattern_matrix(self, current_patterns=None) -> np.ndarray:
        """Return stimulation matrix with shape ``(n_patterns, n_elec)``."""
        if current_patterns is None:
            matrix = np.asarray(self.pattern_manager.stim_matrix, dtype=float)
        else:
            matrix = np.asarray(current_patterns, dtype=float)
            if matrix.ndim != 2:
                raise ValueError("current_patterns must be a 2D array")
            if matrix.shape[1] == self.n_elec:
                pass
            elif matrix.shape[0] == self.n_elec:
                matrix = matrix.T
            else:
                raise ValueError(
                    "current_patterns shape mismatch. Expected (n_patterns, n_elec) "
                    f"or (n_elec, n_patterns), got {matrix.shape}"
                )

        if matrix.shape[1] != self.n_elec:
            raise ValueError(
                f"Pattern width mismatch: expected {self.n_elec}, got {matrix.shape[1]}"
            )
        return matrix

    def _resolve_electrode_tags(self):
        """Extract boundary tags sorted by electrode index from association table."""

        electrode_map = {}

        if isinstance(self.association_table, dict):
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
                        idx_str = key_lower.split("_")[-1]
                        if idx_str.isdigit():
                            electrode_map[int(idx_str)] = tag_val

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
        """Compute boundary measure (length in 2D) for each electrode."""

        lengths = {}
        one = fem.Constant(self.mesh, 1.0)
        for tag in self.electrode_tags:
            length_local = fem.assemble_scalar(fem.form(one * self.ds_electrodes(tag)))
            length = self.mesh.comm.allreduce(length_local, op=MPI.SUM)
            lengths[tag] = float(length)
            if np.isclose(length, 0.0):
                warnings.warn(
                    f"Electrode boundary tag {tag} has zero measure, check mesh markers",
                    RuntimeWarning,
                )
        return lengths

    @staticmethod
    def _petsc_to_csr(mat) -> csr_matrix:
        indptr, indices, values = mat.getValuesCSR()
        shape = mat.getSize()
        return csr_matrix((values, indices, indptr), shape=shape)

    def _assemble_electrode_matrix(self):
        b_form = 0
        for i, electrode_tag in enumerate(self.electrode_tags):
            b_form += (1.0 / self.z[i]) * ufl.inner(self.u, self.phi) * self.ds_electrodes(electrode_tag)

        B = fem_petsc.assemble_matrix(fem.form(b_form))
        B.assemble()
        M = self._petsc_to_csr(B)

        M.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        M_lil = lil_matrix(M)

        for i, electrode_tag in enumerate(self.electrode_tags):
            c_form = (-1.0 / self.z[i]) * self.phi * self.ds_electrodes(electrode_tag)
            C_vec = fem_petsc.assemble_vector(fem.form(c_form))
            C_vec.assemble()
            C_i = np.asarray(C_vec.array, dtype=float)

            M_lil[self.dofs + i, : self.dofs] = C_i
            M_lil[: self.dofs, self.dofs + i] = C_i

            electrode_len = self.electrode_lengths.get(electrode_tag, 0.0)
            M_lil[self.dofs + i, self.dofs + i] = (1.0 / self.z[i]) * electrode_len
            M_lil[self.dofs + self.n_elec, self.dofs + i] = 1.0
            M_lil[self.dofs + i, self.dofs + self.n_elec] = 1.0

        return csr_matrix(M_lil)

    def create_full_matrix(self, sigma: fem.Function):
        """Build complete system matrix including conductivity term."""

        a_form = ufl.inner(sigma * ufl.grad(self.u), ufl.grad(self.phi)) * ufl.dx
        A = fem_petsc.assemble_matrix(fem.form(a_form))
        A.assemble()
        scipy_A = self._petsc_to_csr(A)
        scipy_A.resize(self.dofs + self.n_elec + 1, self.dofs + self.n_elec + 1)
        return scipy_A + self.M

    def forward_solve(self, sigma: fem.Function, current_patterns=None):
        """Forward solve for given conductivity and stimulation patterns."""
        if self.linear_backend == "petsc":
            raise NotImplementedError(
                "linear_backend='petsc' is reserved for future acceleration. "
                "Use linear_backend='scipy' in the current phase."
            )
        if self.linear_backend != "scipy":
            raise ValueError(f"Unsupported linear_backend: {self.linear_backend}")

        pattern_matrix = self._resolve_pattern_matrix(current_patterns)
        n_patterns = pattern_matrix.shape[0]
        system_matrix = self.create_full_matrix(sigma).tocsc()
        lu = splu(system_matrix)

        rhs_matrix = np.zeros((self.dofs + self.n_elec + 1, n_patterns), dtype=float)
        rhs_matrix[self.dofs : self.dofs + self.n_elec, :] = pattern_matrix.T
        sol_matrix = lu.solve(rhs_matrix)

        potential_block = np.asarray(sol_matrix[: self.dofs, :], dtype=float)
        electrode_block = np.asarray(
            sol_matrix[self.dofs : self.dofs + self.n_elec, :].T,
            dtype=float,
        )
        u_all = [potential_block[:, i].copy() for i in range(n_patterns)]
        return u_all, electrode_block

    def fwd_solve(self, img: EITImage):
        """Forward solve interface for ``EITImage``."""

        sigma = fem.Function(self.V_sigma)
        sigma.x.array[:] = img.get_conductivity()

        u_all, U_all = self.forward_solve(sigma)
        meas = self.pattern_manager.apply_meas_pattern(U_all)

        data = EITData(
            meas=meas,
            stim_pattern=self.pattern_manager.stim_matrix,
            n_elec=self.n_elec,
            n_stim=self.pattern_manager.n_stim,
            n_meas=self.pattern_manager.n_meas_total,
            type="simulated",
        )

        return data, U_all
