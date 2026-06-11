"""Compatibility shim: implementation lives in :mod:`pyeidors.realtime.hdf5_outputs`."""

import sys

from pyeidors.realtime import hdf5_outputs as _impl

sys.modules[__name__] = _impl
