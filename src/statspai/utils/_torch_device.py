"""Torch device resolution for StatsPAI's optional neural backends.

Centralises the ``device = torch.device('cpu')`` pattern that appeared
across [deepiv/](../deepiv/), [neural_causal/](../neural_causal/) and
[dose_response/](../dose_response/). Default behaviour is preserved
bit-for-bit (CPU); GPU / MPS dispatch is **opt-in** via the
``STATSPAI_TORCH_DEVICE`` environment variable so existing pinned
numerical tests never silently change.

Resolution rules
----------------
``prefer`` argument > ``STATSPAI_TORCH_DEVICE`` env var > ``"cpu"``.

Accepted values:
- ``"cpu"`` — explicit CPU.
- ``"cuda"`` / ``"cuda:N"`` — explicit CUDA. Raises ``RuntimeError`` if
  CUDA is not available so failures are loud (no silent CPU fallback
  for an explicit request).
- ``"mps"`` — Apple Silicon Metal backend. Raises if unavailable.
- ``"auto"`` — pick CUDA → MPS → CPU in that order, never raises.

The resolver is import-safe when ``torch`` is missing: it only imports
torch when actually called (so module import in pure-numpy paths stays
free).
"""

from __future__ import annotations

import os
from typing import Any, Optional

_ENV_VAR = "STATSPAI_TORCH_DEVICE"


def resolve_torch_device(prefer: Optional[str] = None) -> Any:
    """Return a ``torch.device`` honouring StatsPAI's opt-in policy.

    Parameters
    ----------
    prefer : str, optional
        Per-call override. ``None`` falls back to the
        ``STATSPAI_TORCH_DEVICE`` env var, then to ``"cpu"``.

    Returns
    -------
    torch.device

    Raises
    ------
    ImportError
        If ``torch`` is not installed.
    RuntimeError
        If an explicit ``"cuda"``/``"mps"`` is requested but unavailable.
    """
    import torch  # local import: keeps non-neural paths torch-free

    raw = prefer if prefer is not None else os.environ.get(_ENV_VAR, "cpu")
    spec = raw.strip().lower() if raw is not None else ""
    if not spec:
        # Empty string (e.g. ``STATSPAI_TORCH_DEVICE=``) → treat as unset.
        spec = "cpu"

    if spec == "cpu":
        return torch.device("cpu")

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available(torch):
            return torch.device("mps")
        return torch.device("cpu")

    if spec == "cuda" or spec.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{_ENV_VAR}={raw!r} requested CUDA but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or set STATSPAI_TORCH_DEVICE=cpu."
            )
        device = torch.device(spec)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"{_ENV_VAR}={raw!r} requested CUDA device {device.index}, "
                f"but only {torch.cuda.device_count()} device(s) are available."
            )
        return device

    if spec == "mps":
        if not _mps_available(torch):
            raise RuntimeError(
                f"{_ENV_VAR}={raw!r} requested MPS but torch.backends.mps is unavailable. "
                "Requires Apple Silicon + macOS 12.3+ with PyTorch>=1.12."
            )
        return torch.device("mps")

    # Pass anything else through to torch (lets future devices like 'xpu' work).
    return torch.device(spec)


def torch_device_info() -> str:
    """One-line diagnostic mirroring :func:`statspai.fast.jax_device_info`.

    Safe to call even when torch is missing — returns a clear status
    string instead of raising. Useful for CLI ``sp doctor``-style health
    checks.
    """
    try:
        import torch
    except ImportError:
        return "torch: not installed"

    parts = [f"torch {torch.__version__}"]
    if torch.cuda.is_available():
        parts.append(f"cuda available ({torch.cuda.device_count()} device(s))")
    else:
        parts.append("cuda unavailable")
    if _mps_available(torch):
        parts.append("mps available")
    env = os.environ.get(_ENV_VAR)
    parts.append(
        f"STATSPAI_TORCH_DEVICE={env!r}"
        if env
        else "STATSPAI_TORCH_DEVICE unset (default cpu)"
    )
    try:
        resolved = resolve_torch_device()
    except RuntimeError as exc:
        parts.append(f"resolved=ERROR ({exc})")
    else:
        parts.append(f"resolved={resolved}")
    return " | ".join(parts)


def _mps_available(torch_mod: Any) -> bool:
    """torch.backends.mps was added in 1.12; guard for older builds."""
    backends = getattr(torch_mod, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is None:
        return False
    is_available = getattr(mps, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


__all__ = ["resolve_torch_device", "torch_device_info"]
