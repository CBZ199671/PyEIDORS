"""Device selection helpers for Gauss-Newton solver."""

from __future__ import annotations

import torch


def resolve_torch_device(requested: str, *, verbose: bool) -> torch.device:
    """Resolve compute device with stable CPU-first fallback."""
    normalized = requested.lower()
    if normalized.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested)
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        return device

    if normalized.startswith("mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("Using Apple MPS device")
        return device

    if verbose:
        print("Using CPU for computation")
    return torch.device("cpu")
