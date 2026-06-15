"""Optimized EIT mesh generation using Gmsh + DOLFINx native mesh import."""

from __future__ import annotations

import importlib.util as _import_util
import logging
import tempfile
import time
from configparser import ConfigParser
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..data.structures import EITMesh
from ..electrodes.layout import ELECTRODE_LAYOUT_RING_MAJOR, normalize_electrode_layout
from ..forward.complex_support import petsc_scalar_dtype
from ..perf.policy import (
    DEFAULT_3D_GENERATOR_REVISION,
    DEFAULT_3D_GEOMETRY_VERSION,
    DEFAULT_MESH_FAMILY,
    normalize_mesh_family,
)
from ..runtime_paths import resolve_pyeidors_mesh_dir
from ..utils.numeric_ops import all_finite_values, real_array_if_zero_imaginary
from ._helpers import (
    add_named_physical_group,
    association_from_mesh_data,  # noqa: F401  re-exported for in-tree callers
    assert_unique_physical_group_ownership,
    build_mesh_cache_name,
    build_mesh_cache_name_3d,
    format_float_compact,
    infer_generator_revision,
    infer_geometry_version,
    infer_mesh_family_from_mesh,
    validate_mesh_data_tags,
    write_association_table,
)
from .mesh_converter import MeshConverter
from .dolfinx_mesh_cache import (
    dolfinx_cache_metadata_path_for_mesh,
    load_dolfinx_mesh_cache,
    write_dolfinx_mesh_cache,
    xdmf_cache_path_for_mesh,
    xdmf_h5_path_for_mesh,
)
from .mesh3d_generator import (
    DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
    STRUCTURED_SIDECAR_VERSION,
    create_cylinder_3d_eit_mesh,
    load_structured_sidecar,
    normalize_electrode_level_fractions,
    structured_sidecar_path_for_mesh,
)
from .process_mesh_cache import (
    build_process_mesh_cache_key,
    get_process_cached_mesh,
    put_process_cached_mesh,
)
from ._runtime import mpi_comm_world, mpi_sum_op

logger = logging.getLogger(__name__)

build_eit_mesh = None
estimate_radius = None
fem = None
ufl = None


def _active_geometry_dtype() -> np.dtype:
    """Return DOLFINx's active real geometry dtype for this runtime."""
    try:
        import dolfinx

        return np.dtype(getattr(dolfinx, "default_real_type", np.float64))
    except Exception:
        return np.dtype(np.float64)


def _fem_unit_constant(domain) -> Any:
    return fem.Constant(domain, np.asarray(1.0, dtype=petsc_scalar_dtype())[()])


def _real_scalar(value: Any, *, name: str) -> float:
    return float(real_array_if_zero_imaginary(value, name=name).reshape(()))


def _normalize_geometry_dtype(dtype: Any | None) -> np.dtype:
    if dtype is None:
        return _active_geometry_dtype()
    return np.dtype(dtype)


def _model_to_mesh_with_dtype(model, *, gdim: int, geometry_dtype: Any | None):
    kwargs = {"rank": 0, "gdim": int(gdim)}
    if geometry_dtype is not None:
        kwargs["dtype"] = np.dtype(geometry_dtype).type
    try:
        return gmshio.model_to_mesh(model, mpi_comm_world(), **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        return gmshio.model_to_mesh(model, mpi_comm_world(), **kwargs)


def _ensure_femx() -> None:
    """Defer pyeidors.femx + dolfinx.fem + ufl (transitive dolfinx.io.gmsh)."""
    global build_eit_mesh, estimate_radius, fem, ufl
    if all(x is not None for x in (build_eit_mesh, estimate_radius, fem, ufl)):
        return
    from ..femx import build_eit_mesh as _build, estimate_radius as _estimate
    from dolfinx import fem as _fem
    import ufl as _ufl

    if build_eit_mesh is None:
        build_eit_mesh = _build
    if estimate_radius is None:
        estimate_radius = _estimate
    if fem is None:
        fem = _fem
    if ufl is None:
        ufl = _ufl


GMSH_AVAILABLE: bool = _import_util.find_spec("gmsh") is not None
gmsh = None  # populated lazily by _ensure_gmsh on first generation/parse path
gmshio = None
_GMSH_LOADED = False


def _ensure_gmsh() -> bool:
    """Import gmsh + dolfinx.io.gmsh on demand. Return False iff unavailable."""
    global gmsh, gmshio, _GMSH_LOADED
    if not GMSH_AVAILABLE:
        return False
    if _GMSH_LOADED:
        return gmsh is not None and gmshio is not None
    if gmsh is not None and gmshio is not None:
        _GMSH_LOADED = True
        return True
    try:
        import gmsh as _gmsh_mod
        from dolfinx.io import gmsh as _gmshio_mod
    except (ImportError, OSError):
        return False

    gmsh = _gmsh_mod
    gmshio = _gmshio_mod
    _GMSH_LOADED = True
    return True


@dataclass
class ElectrodePosition:
    """Electrode position configuration."""

    L: int
    coverage: float = 0.5
    rotation: float = 0.0
    anticlockwise: bool = True

    def __post_init__(self):
        if not isinstance(self.L, int) or self.L <= 0:
            raise ValueError("Number of electrodes must be a positive integer")
        if not 0 < self.coverage <= 1:
            raise ValueError("Coverage must be in range (0, 1]")

    @property
    def positions(self) -> List[Tuple[float, float]]:
        electrode_size = 2 * pi / self.L * self.coverage
        gap_size = 2 * pi / self.L * (1 - self.coverage)

        first_electrode_center = pi / 2 + self.rotation
        first_electrode_start = first_electrode_center - electrode_size / 2

        positions: List[Tuple[float, float]] = []
        for i in range(self.L):
            total_space = electrode_size + gap_size
            start = first_electrode_start + i * total_space
            end = start + electrode_size
            positions.append((start, end))

        if not self.anticlockwise:
            positions[1:] = positions[1:][::-1]

        return positions


@dataclass
class OptimizedMeshConfig:
    radius: float = 1.0
    refinement: int = 8
    electrode_vertices: int = 6
    gap_vertices: int = 1

    @property
    def mesh_size(self) -> float:
        return self.radius / (self.refinement * 2)


class OptimizedMeshGenerator:
    """Optimized mesh generator with stable physical tagging."""

    def __init__(
        self,
        config: OptimizedMeshConfig,
        electrodes: ElectrodePosition,
        *,
        geometry_dtype: Any | None = None,
    ):
        self.config = config
        self.electrodes = electrodes
        self.geometry_dtype = _normalize_geometry_dtype(geometry_dtype)
        self.mesh_data: Dict[str, object] = {}

    def generate(
        self, output_dir: Optional[Path] = None, mesh_name: Optional[str] = None
    ) -> EITMesh:
        if not _ensure_gmsh():
            raise ImportError("gmsh Python bindings are required to generate meshes.")

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if mesh_name is None:
            mesh_name = f"mesh_{int(time.time() * 1e6) % 1000000}"

        msh_path = output_dir / f"{mesh_name}.msh"
        association_path = output_dir / f"{mesh_name}_association_table.ini"

        initialized_here = False
        if not gmsh.isInitialized():
            gmsh.initialize()
            initialized_here = True
        gmsh.clear()
        gmsh.model.add(mesh_name)
        try:
            self._create_geometry()
            self._set_physical_groups()
            self._generate_mesh()
            gmsh.write(str(msh_path))
            self._extract_electrode_vertices()

            assert_unique_physical_group_ownership(gmsh.model)
            mesh_data = _model_to_mesh_with_dtype(
                gmsh.model,
                gdim=2,
                geometry_dtype=self.geometry_dtype,
            )
        finally:
            if initialized_here:
                gmsh.finalize()
            else:
                gmsh.clear()

        electrode_names = [
            f"electrode_{idx}" for idx in range(1, self.electrodes.L + 1)
        ]
        facet_names = [*electrode_names, "gaps"]
        association_table = validate_mesh_data_tags(
            mesh_data,
            gdim=2,
            required_names=["domain", *facet_names],
            required_facet_names=facet_names,
        )
        write_association_table(association_path, association_table)
        write_dolfinx_mesh_cache(
            mesh_data,
            source_msh_file=msh_path,
            association_table=association_table,
            gdim=2,
        )

        _ensure_femx()

        electrode_vertices = [
            np.asarray(v, dtype=float)
            for v in self.mesh_data.get("electrode_vertices", [])
        ]
        mesh = build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            radius=self.config.radius,
            mesh_file=str(msh_path),
            electrode_vertices=electrode_vertices,
        )
        return mesh

    def _create_geometry(self):
        positions = self.electrodes.positions
        n_in = self.config.electrode_vertices
        n_out = self.config.gap_vertices
        r = self.config.radius

        boundary_points = []
        electrode_ranges = []

        for i, (start, end) in enumerate(positions):
            start_idx = len(boundary_points)

            for theta in np.linspace(start, end, n_in):
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)

            electrode_ranges.append((start_idx, len(boundary_points) - 1))

            if i < len(positions) - 1:
                gap_start = end
                gap_end = positions[i + 1][0]
            else:
                gap_start = end
                gap_end = positions[0][0] + 2 * pi

            gap_points = np.linspace(gap_start, gap_end, n_out + 2)[1:-1]
            for theta in gap_points:
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)

        lines = []
        for i in range(len(boundary_points)):
            next_i = (i + 1) % len(boundary_points)
            line = gmsh.model.occ.addLine(boundary_points[i], boundary_points[next_i])
            lines.append(line)

        loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([loop])

        mesh_size_center = 0.095 * r
        cp_distance = 0.1 * r
        center_points = [
            gmsh.model.occ.addPoint(x, y, 0.0, meshSize=mesh_size_center)
            for x, y in [
                (-cp_distance, cp_distance),
                (cp_distance, cp_distance),
                (-cp_distance, -cp_distance),
                (cp_distance, -cp_distance),
            ]
        ]

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.embed(0, center_points, 2, surface)

        self.mesh_data["boundary_points"] = boundary_points
        self.mesh_data["electrode_ranges"] = electrode_ranges
        self.mesh_data["lines"] = lines
        self.mesh_data["surface"] = surface

    def _set_physical_groups(self):
        surface = self.mesh_data["surface"]
        lines = self.mesh_data["lines"]
        electrode_ranges = self.mesh_data["electrode_ranges"]

        add_named_physical_group(gmsh.model, 2, [surface], 1, "domain")

        electrode_lines = []
        for i, (start, end) in enumerate(electrode_ranges):
            lines_for_electrode = []
            for j in range(start, end):
                line_idx = j % len(lines)
                lines_for_electrode.append(lines[line_idx])

            if lines_for_electrode:
                add_named_physical_group(
                    gmsh.model,
                    1,
                    lines_for_electrode,
                    i + 2,
                    f"electrode_{i + 1}",
                )
                electrode_lines.extend(lines_for_electrode)

        gap_lines = [line for line in lines if line not in electrode_lines]
        if gap_lines:
            add_named_physical_group(
                gmsh.model,
                1,
                gap_lines,
                self.electrodes.L + 2,
                "gaps",
            )

    def _generate_mesh(self):
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.config.mesh_size)
        gmsh.model.mesh.generate(2)

    def _extract_electrode_vertices(self):
        positions = self.electrodes.positions
        r = self.config.radius
        n_in = self.config.electrode_vertices

        electrode_vertices = []
        for start, end in positions:
            vertices = []
            for theta in np.linspace(start, end, n_in):
                vertices.append([r * cos(theta), r * sin(theta)])
            electrode_vertices.append(vertices)

        self.mesh_data["electrode_vertices"] = electrode_vertices


class OptimizedMeshConverter(MeshConverter):
    """``MeshConverter`` variant that records ``radius=estimate_radius(...)``.

    Behavior identical to the canonical :class:`MeshConverter` apart from
    populating :attr:`EITMesh.radius` via :func:`pyeidors.femx.estimate_radius`.
    Kept as an explicit subclass so existing callers (tests, scripts,
    benchmarks) referencing this name keep working without alias warnings.
    T78 Path C consolidates the two converter implementations onto the
    canonical ``MeshConverter`` body via the ``radius_provider`` hook.
    """

    def __init__(self, mesh_file: str, output_dir: str, gdim: int = 2):
        _ensure_femx()
        super().__init__(
            mesh_file,
            output_dir,
            gdim=gdim,
            radius_provider=estimate_radius,
        )


# Convenience functions


def create_eit_mesh(
    n_elec: int = 16,
    radius: float = 1.0,
    refinement: int = 6,
    electrode_coverage: float = 0.5,
    output_dir: str = None,
    mesh_name: Optional[str] = None,
    geometry_dtype: Any | None = None,
):
    mesh_config = OptimizedMeshConfig(
        radius=radius,
        refinement=refinement,
        electrode_vertices=6,
        gap_vertices=1,
    )

    electrode_config = ElectrodePosition(
        L=n_elec,
        coverage=electrode_coverage,
        rotation=0.0,
        anticlockwise=True,
    )

    generator_kwargs = {}
    if geometry_dtype is not None:
        generator_kwargs["geometry_dtype"] = geometry_dtype
    generator = OptimizedMeshGenerator(
        mesh_config, electrode_config, **generator_kwargs
    )
    return generator.generate(
        output_dir=Path(output_dir) if output_dir else None, mesh_name=mesh_name
    )


# T78 Path C: cache-name helpers consolidated into ``geometry/_helpers.py``.
# These module-level aliases preserve the historical private symbols for
# any in-tree ``import`` users while the canonical bodies live in
# :mod:`pyeidors.geometry._helpers`.

_format_float = format_float_compact
_build_cache_name = build_mesh_cache_name
_build_cache_name_3d = build_mesh_cache_name_3d


def _facet_tags_cover_electrodes(
    mesh: EITMesh,
    *,
    electrode_keys: list[str],
    association: dict[str, int],
) -> bool | None:
    """Cheap 3D CEM tag-completeness check that avoids FEM/JIT assembly."""
    values = getattr(getattr(mesh, "facet_tags", None), "values", None)
    if values is None:
        return None
    try:
        tag_values = np.asarray(values, dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if tag_values.size == 0:
        return False

    for key in electrode_keys:
        tag = int(association[key])
        local_count = int(np.count_nonzero(tag_values == tag))
        try:
            total_count = int(mesh.comm.allreduce(local_count, op=mpi_sum_op()))
        except Exception:
            total_count = local_count
        if total_count <= 0:
            return False
    return True


def _cached_3d_cem_mesh_is_complete(mesh: EITMesh, *, n_elec: int) -> bool:
    if int(mesh.topology.dim) != 3:
        return True
    association = dict(getattr(mesh, "association_table", {}) or {})
    if not association or "domain" not in association or "gaps" not in association:
        return False
    electrode_keys = [f"electrode_{idx}" for idx in range(1, int(n_elec) + 1)]
    if any(key not in association for key in electrode_keys):
        return False
    if mesh.facet_tags is None:
        return False
    facet_coverage = _facet_tags_cover_electrodes(
        mesh,
        electrode_keys=electrode_keys,
        association=association,
    )
    if facet_coverage is False:
        return False
    if facet_coverage is None:
        try:
            _ensure_femx()
            ds = ufl.Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
            one = _fem_unit_constant(mesh.mesh)
            measures = []
            for key in electrode_keys:
                value_local = fem.assemble_scalar(
                    fem.form(one * ds(int(association[key])))
                )
                value = mesh.comm.allreduce(value_local, op=mpi_sum_op())
                measures.append(_real_scalar(value, name=f"electrode {key} measure"))
        except Exception as exc:
            logger.warning(
                "Skipping cached 3D mesh %s due to CEM validation failure: %s",
                mesh.mesh_file,
                exc,
            )
            return False
        arr = np.asarray(measures, dtype=float)
        if not bool(
            arr.size == int(n_elec)
            and all_finite_values(arr)
            and float(np.min(arr)) > 0.0
        ):
            return False
    mesh_family = str(getattr(mesh, "mesh_family", None) or "").strip().lower()
    geometry_version = (
        str(getattr(mesh, "geometry_version", None) or "").strip().lower()
    )
    generator_revision = (
        str(getattr(mesh, "generator_revision", None) or "").strip().lower()
    )
    if (
        mesh_family == "hex"
        and geometry_version == "geomv2"
        and generator_revision == DEFAULT_3D_GENERATOR_REVISION
    ):
        mesh_file = getattr(mesh, "mesh_file", None)
        sidecar_file = getattr(mesh, "structured_sidecar_file", None)
        if not mesh_file and not sidecar_file:
            return False
        sidecar_path = (
            Path(sidecar_file)
            if sidecar_file
            else structured_sidecar_path_for_mesh(mesh_file)
        )
        if not sidecar_path.exists():
            logger.warning(
                "Skipping cached mesh %s because structured sidecar is missing",
                mesh.mesh_file,
            )
            return False
        try:
            load_structured_sidecar(sidecar_path)
        except Exception as exc:
            logger.warning(
                "Skipping cached mesh %s because structured sidecar validation failed: %s",
                mesh.mesh_file,
                exc,
            )
            return False
    return True


def _load_cached_mesh(
    mesh_dir: Path,
    mesh_name: str,
    *,
    gdim: int = 2,
    n_elec: int = 16,
    geometry_dtype: Any | None = None,
):
    msh_file = mesh_dir / f"{mesh_name}.msh"
    association_file = mesh_dir / f"{mesh_name}_association_table.ini"

    cache_data = load_dolfinx_mesh_cache(
        msh_file,
        gdim=int(gdim),
        expected_geometry_dtype=geometry_dtype,
    )
    if cache_data is not None:
        _ensure_femx()
        metadata = cache_data.metadata
        sidecar_file = metadata.get("structured_sidecar_file")
        mesh_file = cache_data.source_msh_file or cache_data.xdmf_file
        mesh = build_eit_mesh(
            cache_data.mesh,
            facet_tags=cache_data.facet_tags,
            cell_tags=cache_data.cell_tags,
            association_table=cache_data.association_table,
            physical_groups=cache_data.physical_groups,
            radius=estimate_radius(cache_data.mesh),
            mesh_file=mesh_file,
            mesh_family=metadata.get("mesh_family"),
            geometry_version=metadata.get("geometry_version")
            or infer_geometry_version(mesh_name),
            generator_revision=metadata.get("generator_revision")
            or infer_generator_revision(mesh_name),
            structured_sidecar_file=sidecar_file,
            structured_sidecar_version=metadata.get("structured_sidecar_version"),
        )
        if not getattr(mesh, "mesh_family", None):
            mesh.mesh_family = infer_mesh_family_from_mesh(mesh)
        if int(gdim) == 3 and not _cached_3d_cem_mesh_is_complete(
            mesh, n_elec=int(n_elec)
        ):
            logger.warning(
                "Skipping cached mesh %s because 3D CEM tags/measures are incomplete",
                mesh_name,
            )
            return None
        return mesh

    if not msh_file.exists():
        return None

    if not _ensure_gmsh():
        return None

    try:
        mesh_data = gmshio.read_from_msh(
            str(msh_file),
            mpi_comm_world(),
            rank=0,
            gdim=int(gdim),
        )
    except Exception as exc:
        logger.warning(
            "Skipping cached mesh %s due to gdim=%d load failure: %s",
            msh_file,
            int(gdim),
            exc,
        )
        return None

    association_table = validate_mesh_data_tags(mesh_data, gdim=int(gdim))
    if not association_table and association_file.exists():
        association = ConfigParser()
        association.read(association_file)
        if "ASSOCIATION TABLE" in association:
            section = association["ASSOCIATION TABLE"]
            association_table = {key: int(value) for key, value in section.items()}
        else:
            association_table = {}

    sidecar_path = structured_sidecar_path_for_mesh(msh_file)
    geometry_version = infer_geometry_version(mesh_name)
    generator_revision = infer_generator_revision(mesh_name)
    if sidecar_path.exists():
        try:
            sidecar = load_structured_sidecar(sidecar_path)
            geometry_version = (
                str(sidecar.get("geometry_version", geometry_version)).strip().lower()
                or geometry_version
            )
            generator_revision = (
                str(sidecar.get("generator_revision", generator_revision))
                .strip()
                .lower()
                or generator_revision
            )
        except Exception:
            pass

    sidecar_exists = sidecar_path.exists()
    structured_sidecar_file = str(sidecar_path) if sidecar_exists else None
    structured_sidecar_version = STRUCTURED_SIDECAR_VERSION if sidecar_exists else None
    write_dolfinx_mesh_cache(
        mesh_data,
        source_msh_file=msh_file,
        association_table=association_table,
        gdim=int(gdim),
        geometry_version=geometry_version,
        generator_revision=generator_revision,
        structured_sidecar_file=structured_sidecar_file,
        structured_sidecar_version=structured_sidecar_version,
    )
    _ensure_femx()
    mesh = build_eit_mesh(
        mesh_data.mesh,
        facet_tags=mesh_data.facet_tags,
        cell_tags=mesh_data.cell_tags,
        association_table=association_table,
        physical_groups=mesh_data.physical_groups,
        radius=estimate_radius(mesh_data.mesh),
        mesh_file=str(msh_file),
        geometry_version=geometry_version,
        generator_revision=generator_revision,
        structured_sidecar_file=structured_sidecar_file,
        structured_sidecar_version=structured_sidecar_version,
    )
    mesh.mesh_family = infer_mesh_family_from_mesh(mesh)
    if int(gdim) == 3 and not _cached_3d_cem_mesh_is_complete(mesh, n_elec=int(n_elec)):
        logger.warning(
            "Skipping cached mesh %s because 3D CEM tags/measures are incomplete",
            mesh_name,
        )
        return None
    return mesh


def load_or_create_mesh(
    mesh_dir: str = "eit_meshes",
    mesh_name: str = None,
    n_elec: int = 16,
    dimension: int = 2,
    **kwargs,
):
    mesh_dir_path = resolve_pyeidors_mesh_dir(mesh_dir)
    mesh_dir_path.mkdir(parents=True, exist_ok=True)

    params = dict(kwargs)
    radius = params.pop("radius", 1.0)
    refinement = params.pop("refinement", 6)
    electrode_coverage = params.pop("electrode_coverage", 0.5)
    height = params.pop("height", 1.0)
    electrode_height_ratio = params.pop("electrode_height_ratio", 0.2)
    electrode_layout = normalize_electrode_layout(
        params.pop("electrode_layout", ELECTRODE_LAYOUT_RING_MAJOR)
    )
    electrode_level_fractions = normalize_electrode_level_fractions(
        params.pop("electrode_level_fractions", DEFAULT_ZIGZAG_LEVEL_FRACTIONS),
        default=DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
    )
    z_center = params.pop("z_center", 0.0)
    mesh_family = normalize_mesh_family(
        params.pop("mesh_family", DEFAULT_MESH_FAMILY),
        default=DEFAULT_MESH_FAMILY,
    )
    geometry_version = (
        str(params.pop("geometry_version", DEFAULT_3D_GEOMETRY_VERSION)).strip().lower()
        or DEFAULT_3D_GEOMETRY_VERSION
    )
    generator_revision = (
        str(params.pop("generator_revision", DEFAULT_3D_GENERATOR_REVISION))
        .strip()
        .lower()
        or DEFAULT_3D_GENERATOR_REVISION
    )
    geometry_dtype = _normalize_geometry_dtype(params.pop("geometry_dtype", None))
    gdim = int(dimension)
    if gdim not in {2, 3}:
        raise ValueError(f"dimension must be 2 or 3, got {dimension!r}")

    if mesh_name:
        cache_name = mesh_name
    elif gdim == 2:
        cache_name = _build_cache_name(
            n_elec,
            radius,
            refinement,
            electrode_coverage,
            geometry_dtype=geometry_dtype,
        )
    else:
        cache_name = _build_cache_name_3d(
            n_elec=n_elec,
            radius=radius,
            height=height,
            refinement=refinement,
            electrode_coverage=electrode_coverage,
            electrode_height_ratio=electrode_height_ratio,
            electrode_level_fractions=electrode_level_fractions,
            z_center=z_center,
            mesh_family=mesh_family,
            geometry_version=geometry_version,
            generator_revision=generator_revision,
            electrode_layout=electrode_layout,
            geometry_dtype=geometry_dtype,
        )

    process_mesh_key: str | None = None
    msh_file = mesh_dir_path / f"{cache_name}.msh"
    xdmf_file = xdmf_cache_path_for_mesh(msh_file)
    if msh_file.exists() or xdmf_file.exists():
        association_file = mesh_dir_path / f"{cache_name}_association_table.ini"
        sidecar_file = structured_sidecar_path_for_mesh(msh_file)
        metadata_file = dolfinx_cache_metadata_path_for_mesh(msh_file)
        process_mesh_file = xdmf_file if xdmf_file.exists() else msh_file
        extra_files = []
        h5_file = xdmf_h5_path_for_mesh(msh_file)
        if h5_file.exists():
            extra_files.append(h5_file)
        if msh_file.exists() and process_mesh_file != msh_file:
            extra_files.append(msh_file)
        process_mesh_key = build_process_mesh_cache_key(
            mesh_file=process_mesh_file,
            association_file=(
                metadata_file
                if metadata_file.exists()
                else association_file
                if association_file.exists()
                else None
            ),
            sidecar_file=sidecar_file if sidecar_file.exists() else None,
            extra_files=extra_files,
            gdim=gdim,
            n_elec=n_elec,
            mesh_name=cache_name,
            geometry_dtype=geometry_dtype,
        )
        process_mesh = get_process_cached_mesh(process_mesh_key)
        if process_mesh is not None:
            logger.info("Loaded process-cached mesh: %s", cache_name)
            setattr(process_mesh, "_pyeidors_mesh_cache_hit", True)
            setattr(process_mesh, "_pyeidors_mesh_cache_layer", "process")
            setattr(process_mesh, "_pyeidors_mesh_cache_name", cache_name)
            return process_mesh

    cached_mesh = _load_cached_mesh(
        mesh_dir_path,
        cache_name,
        gdim=gdim,
        n_elec=n_elec,
        geometry_dtype=geometry_dtype,
    )
    if cached_mesh is not None:
        logger.info("Loaded cached mesh: %s", cache_name)
        metadata_file = dolfinx_cache_metadata_path_for_mesh(msh_file)
        association_file = mesh_dir_path / f"{cache_name}_association_table.ini"
        sidecar_file = structured_sidecar_path_for_mesh(msh_file)
        process_mesh_file = xdmf_file if xdmf_file.exists() else msh_file
        extra_files = []
        h5_file = xdmf_h5_path_for_mesh(msh_file)
        if h5_file.exists():
            extra_files.append(h5_file)
        if msh_file.exists() and process_mesh_file != msh_file:
            extra_files.append(msh_file)
        process_mesh_key = build_process_mesh_cache_key(
            mesh_file=process_mesh_file,
            association_file=(
                metadata_file
                if metadata_file.exists()
                else association_file
                if association_file.exists()
                else None
            ),
            sidecar_file=sidecar_file if sidecar_file.exists() else None,
            extra_files=extra_files,
            gdim=gdim,
            n_elec=n_elec,
            mesh_name=cache_name,
            geometry_dtype=geometry_dtype,
        )
        put_process_cached_mesh(process_mesh_key, cached_mesh)
        setattr(cached_mesh, "_pyeidors_mesh_cache_hit", True)
        setattr(cached_mesh, "_pyeidors_mesh_cache_layer", "disk")
        setattr(cached_mesh, "_pyeidors_mesh_cache_name", cache_name)
        return cached_mesh

    logger.info("Cached mesh not found, generating: %s", cache_name)
    if params:
        logger.debug("Unused mesh parameters: %s", params)

    if gdim == 2:
        created_mesh = create_eit_mesh(
            n_elec=n_elec,
            radius=radius,
            refinement=refinement,
            electrode_coverage=electrode_coverage,
            output_dir=str(mesh_dir_path),
            mesh_name=cache_name,
            geometry_dtype=geometry_dtype,
        )
    else:
        created_mesh = create_cylinder_3d_eit_mesh(
            n_elec=n_elec,
            radius=radius,
            height=height,
            refinement=refinement,
            electrode_coverage=electrode_coverage,
            electrode_height_ratio=electrode_height_ratio,
            electrode_level_fractions=electrode_level_fractions,
            z_center=z_center,
            output_dir=str(mesh_dir_path),
            mesh_name=cache_name,
            mesh_family=mesh_family,
            geometry_version=geometry_version,
            generator_revision=generator_revision,
            electrode_layout=electrode_layout,
        )

    created_mesh_file = getattr(created_mesh, "mesh_file", None)
    if created_mesh_file:
        association_file = mesh_dir_path / f"{cache_name}_association_table.ini"
        sidecar_file = structured_sidecar_path_for_mesh(created_mesh_file)
        metadata_file = dolfinx_cache_metadata_path_for_mesh(created_mesh_file)
        xdmf_file = xdmf_cache_path_for_mesh(created_mesh_file)
        process_mesh_file = xdmf_file if xdmf_file.exists() else created_mesh_file
        extra_files = []
        h5_file = xdmf_h5_path_for_mesh(created_mesh_file)
        if h5_file.exists():
            extra_files.append(h5_file)
        if xdmf_file.exists():
            extra_files.append(created_mesh_file)
        process_mesh_key = build_process_mesh_cache_key(
            mesh_file=process_mesh_file,
            association_file=(
                metadata_file
                if metadata_file.exists()
                else association_file
                if association_file.exists()
                else None
            ),
            sidecar_file=sidecar_file if sidecar_file.exists() else None,
            extra_files=extra_files,
            gdim=gdim,
            n_elec=n_elec,
            mesh_name=cache_name,
            geometry_dtype=geometry_dtype,
        )
        put_process_cached_mesh(process_mesh_key, created_mesh)
    setattr(created_mesh, "_pyeidors_mesh_cache_hit", False)
    setattr(created_mesh, "_pyeidors_mesh_cache_layer", "generated")
    setattr(created_mesh, "_pyeidors_mesh_cache_name", cache_name)
    return created_mesh
