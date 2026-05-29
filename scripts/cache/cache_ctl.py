#!/usr/bin/env python3
"""Compatibility wrapper for the installable ``eit-cache`` command."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.cache.cli import main


if __name__ == "__main__":
    main()
