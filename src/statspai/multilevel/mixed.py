"""Back-compat shim — the real implementation now lives in ``lmm.py``.

Older code and external notebooks import from
``statspai.multilevel.mixed``; re-export the symbols they need so that
path keeps working.
"""

from .lmm import MixedResult, mixed  # noqa: F401

__all__ = ["mixed", "MixedResult"]
