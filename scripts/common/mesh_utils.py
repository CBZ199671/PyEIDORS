"""Compatibility shim: implementation lives in :mod:`pyeidors.realtime.mesh_utils`."""

import sys

from pyeidors.realtime import mesh_utils as _impl

sys.modules[__name__] = _impl
