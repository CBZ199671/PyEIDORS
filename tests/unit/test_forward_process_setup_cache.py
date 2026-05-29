"""Content-addressed process-local forward-setup cache key (G2)."""

from __future__ import annotations

import hashlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.data.structures import PatternConfig
from pyeidors.forward.eit_forward_model import _hash_mesh_content, _hash_scalar_array
from pyeidors.forward.process_setup_cache import build_process_forward_setup_key


def _default_key_kwargs(**overrides):
    payload = {
        "mesh_file": "/tmp/mesh.xdmf",
        "n_elec": 16,
        "z": np.full(16, 0.01, dtype=float),
        "pattern_config": PatternConfig(n_elec=16),
        "mesh_content_hash": None,
    }
    payload.update(overrides)
    return payload


def test_build_key_requires_mesh_identifier():
    with pytest.raises(ValueError, match="mesh_file or mesh_content_hash"):
        build_process_forward_setup_key(
            **_default_key_kwargs(mesh_file=None, mesh_content_hash=None)
        )


def test_build_key_accepts_mesh_file_only():
    key = build_process_forward_setup_key(**_default_key_kwargs())
    assert isinstance(key, str) and len(key) == 64


def test_build_key_accepts_content_hash_only():
    key = build_process_forward_setup_key(
        **_default_key_kwargs(mesh_file=None, mesh_content_hash="abc123")
    )
    assert isinstance(key, str) and len(key) == 64


def test_build_key_changes_with_content_hash():
    a = build_process_forward_setup_key(
        **_default_key_kwargs(mesh_file=None, mesh_content_hash="aaa")
    )
    b = build_process_forward_setup_key(
        **_default_key_kwargs(mesh_file=None, mesh_content_hash="bbb")
    )
    assert a != b


def test_build_key_changes_with_potential_order():
    p1 = build_process_forward_setup_key(**_default_key_kwargs(potential_order=1))
    p2 = build_process_forward_setup_key(**_default_key_kwargs(potential_order=2))

    assert p1 != p2


def test_build_key_stable_for_same_inputs():
    a = build_process_forward_setup_key(**_default_key_kwargs())
    b = build_process_forward_setup_key(**_default_key_kwargs())
    assert a == b


def test_hash_mesh_content_returns_empty_for_missing_geometry():
    assert _hash_mesh_content(SimpleNamespace()) == ""
    assert _hash_mesh_content(SimpleNamespace(geometry=None, topology=None)) == ""


def test_hash_mesh_content_detects_coordinate_change():
    class _FakeTopology:
        def __init__(self, cells):
            self.dim = 2
            self._connectivity = SimpleNamespace(
                array=np.asarray(cells, dtype=np.int64)
            )

        def create_connectivity(self, _tdim, _vdim):
            return None

        def connectivity(self, _tdim, _vdim):
            return self._connectivity

    class _FakeMesh:
        def __init__(self, coords, cells):
            self.geometry = SimpleNamespace(x=np.asarray(coords, dtype=np.float64))
            self.topology = _FakeTopology(cells)

    mesh_a = _FakeMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    mesh_b = _FakeMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], [[0, 1, 2]])

    hash_a = _hash_mesh_content(mesh_a)
    hash_b = _hash_mesh_content(mesh_b)

    assert hash_a and hash_b
    assert hash_a != hash_b


def test_hash_mesh_content_detects_connectivity_change():
    class _FakeTopology:
        def __init__(self, cells):
            self.dim = 2
            self._cells = np.asarray(cells, dtype=np.int64)

        def create_connectivity(self, _tdim, _vdim):
            return None

        def connectivity(self, _tdim, _vdim):
            return SimpleNamespace(array=self._cells)

    class _FakeMesh:
        def __init__(self, coords, cells):
            self.geometry = SimpleNamespace(x=np.asarray(coords, dtype=np.float64))
            self.topology = _FakeTopology(cells)

    coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    mesh_a = _FakeMesh(coords, [[0, 1, 2], [1, 2, 3]])
    mesh_b = _FakeMesh(coords, [[0, 1, 3], [1, 2, 3]])

    assert _hash_mesh_content(mesh_a) != _hash_mesh_content(mesh_b)


def test_v258_forward_hash_helpers_stream_array_payloads() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    array = np.ascontiguousarray(values, dtype=np.float64)
    expected = hashlib.sha256(
        f"{array.dtype}:{array.shape}:".encode("utf-8") + array.tobytes()
    ).hexdigest()

    assert _hash_scalar_array(values, np.dtype(np.float64)) == expected
    assert ".tobytes(" not in inspect.getsource(_hash_scalar_array)
    assert ".tobytes(" not in inspect.getsource(_hash_mesh_content)
    assert "np.ascontiguousarray(np.asarray" not in inspect.getsource(
        _hash_mesh_content
    )
    assert "update_digest_with_array_payload" in inspect.getsource(_hash_scalar_array)
    assert "update_digest_with_array_payload" in inspect.getsource(_hash_mesh_content)
