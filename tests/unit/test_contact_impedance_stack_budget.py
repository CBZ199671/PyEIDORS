from __future__ import annotations

import inspect
import os

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from eit_app.controllers import forward_solver_controller as fwd_controller  # noqa: E402
from eit_app.controllers import reconstruction_controller as recon_controller  # noqa: E402


def test_v307_contact_impedance_vectors_direct_fill_repeated_inputs() -> None:
    forward = fwd_controller._contact_impedance_vector(
        np.array([0.01, 0.02], dtype=np.float32),
        total_electrodes=6,
    )
    recon = recon_controller._contact_impedance_vector_from_meta(
        {"contact_impedance": [0.01 + 0.001j, 0.02 + 0.002j]},
        total_electrodes=6,
    )

    np.testing.assert_allclose(
        forward,
        np.array([0.01, 0.02, 0.01, 0.02, 0.01, 0.02], dtype=np.float64),
    )
    np.testing.assert_allclose(
        recon,
        np.array(
            [
                0.01 + 0.001j,
                0.02 + 0.002j,
                0.01 + 0.001j,
                0.02 + 0.002j,
                0.01 + 0.001j,
                0.02 + 0.002j,
            ],
            dtype=np.complex128,
        ),
    )
    assert forward.dtype == np.float64
    assert recon.dtype == np.complex128
    assert "np.tile" not in inspect.getsource(fwd_controller._contact_impedance_vector)
    assert "np.tile" not in inspect.getsource(
        recon_controller._contact_impedance_vector_from_meta
    )
