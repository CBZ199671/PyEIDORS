"""Electrode layout helpers shared by mesh and pattern adapters."""

from __future__ import annotations


ELECTRODE_LAYOUT_RING_MAJOR = "ring_major"
ELECTRODE_LAYOUT_ZIGZAG = "zigzag"


def normalize_electrode_layout(value: object | None) -> str:
    """Normalize supported 3D electrode-numbering conventions.

    ``ring_major`` follows the EIDORS multi-ring convention:
    layer/ring 0 owns electrodes 1..N, layer/ring 1 owns N+1..2N, etc.
    ``zigzag`` is kept as a legacy compatibility layout for older PyEIDORS
    cylinder meshes that distributed a single circular sequence across z-levels.
    """
    raw = str(value or ELECTRODE_LAYOUT_RING_MAJOR).strip().lower().replace("-", "_")
    aliases = {
        "ring": ELECTRODE_LAYOUT_RING_MAJOR,
        "ring_major": ELECTRODE_LAYOUT_RING_MAJOR,
        "planar": ELECTRODE_LAYOUT_RING_MAJOR,
        "planar_ring_major": ELECTRODE_LAYOUT_RING_MAJOR,
        "eidors": ELECTRODE_LAYOUT_RING_MAJOR,
        "eidors_ring_major": ELECTRODE_LAYOUT_RING_MAJOR,
        "zigzag": ELECTRODE_LAYOUT_ZIGZAG,
        "zig_zag": ELECTRODE_LAYOUT_ZIGZAG,
        "legacy": ELECTRODE_LAYOUT_ZIGZAG,
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            "Unsupported electrode_layout "
            f"{value!r}; expected 'ring_major' or 'zigzag'."
        ) from exc


def effective_pattern_layout_for_3d_mesh(
    *,
    mesh_tdim: int | None,
    n_elec: int,
    n_rings: int,
    electrode_layout: object | None = ELECTRODE_LAYOUT_RING_MAJOR,
) -> tuple[int, int]:
    """Return the PatternConfig layout matching the mesh numbering.

    For the EIDORS-compatible ``ring_major`` layout we keep the user's
    per-ring x ring-count shape.  For legacy ``zigzag`` 3D meshes we preserve
    the historical adapter: collapse N electrodes/ring x R rings into a single
    N*R circular sequence so pattern columns match physical electrode tags.
    """
    electrodes = max(int(n_elec), 1)
    rings = max(int(n_rings), 1)
    layout = normalize_electrode_layout(electrode_layout)
    if (
        mesh_tdim is not None
        and int(mesh_tdim) == 3
        and rings > 1
        and layout == ELECTRODE_LAYOUT_ZIGZAG
    ):
        return electrodes * rings, 1
    return electrodes, rings


def effective_pattern_layout_for_zigzag_3d_mesh(
    *,
    mesh_tdim: int | None,
    n_elec: int,
    n_rings: int,
    electrode_layout: object | None = ELECTRODE_LAYOUT_ZIGZAG,
) -> tuple[int, int]:
    """Backward-compatible wrapper for existing callers.

    New code should call :func:`effective_pattern_layout_for_3d_mesh` and pass
    the explicit ``electrode_layout``.  This wrapper defaults to the old zigzag
    behavior to avoid silently changing legacy scripts.
    """
    return effective_pattern_layout_for_3d_mesh(
        mesh_tdim=mesh_tdim,
        n_elec=n_elec,
        n_rings=n_rings,
        electrode_layout=electrode_layout,
    )
