"""
Inverse probability of censoring / treatment weighting primitives.
"""
from .ipcw import ipcw, IPCWResult

__all__ = ["ipcw", "IPCWResult"]
