"""Compatibility shim: implementation lives in :mod:`pyeidors.realtime.array_metrics`."""

import sys

from pyeidors.realtime import array_metrics as _impl

sys.modules[__name__] = _impl
