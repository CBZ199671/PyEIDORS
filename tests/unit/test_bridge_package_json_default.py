"""T89: bridge_package._json_default delegates to pyeidors.io._json.json_ready."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eit_app.interop import bridge_package
from pyeidors.io._json import json_ready


def test_json_default_delegates_path_and_ndarray_to_json_ready() -> None:
    assert bridge_package._json_default(Path("a/b.json")) == "a/b.json"
    assert bridge_package._json_default(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


def test_json_default_widens_numpy_scalars_via_json_ready() -> None:
    assert bridge_package._json_default(np.float64(1.25)) == 1.25
    assert bridge_package._json_default(np.int32(7)) == 7


def test_json_default_raises_typeerror_for_unknown_objects() -> None:
    class Opaque:
        pass

    with pytest.raises(
        TypeError, match="Object of type Opaque is not JSON serializable"
    ):
        bridge_package._json_default(Opaque())


def test_json_default_dump_byte_parity_against_json_ready() -> None:
    payload = {
        "manifest_path": Path("bundle/manifest.json"),
        "frames": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "frame_count": np.int64(2),
    }
    via_default = json.dumps(
        payload, ensure_ascii=False, indent=2, default=bridge_package._json_default
    )
    via_helper = json.dumps(json_ready(payload), ensure_ascii=False, indent=2)
    assert via_default == via_helper
