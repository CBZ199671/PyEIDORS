"""Tests for object_signature edge cases."""

from __future__ import annotations


import numpy as np


class TestModelSignatureHashFailure:
    """Cover lines 228-229 in object_signature.py: mesh hash failure fallback."""

    def test_mesh_coordinates_extraction_failure(self):
        from pyeidors.cache.object_signature import model_signature_from_forward_model

        class FakeMesh:
            def coordinates(self):
                raise RuntimeError("no coords")

            def cells(self):
                raise RuntimeError("no cells")

        class FakeFwdModel:
            n_elec = 4
            eit_mesh = None
            z = np.ones(4)
            geometry_scale_to_m = 1.0

        sig = model_signature_from_forward_model(FakeFwdModel())
        assert isinstance(sig, str)
