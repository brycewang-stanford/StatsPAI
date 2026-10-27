"""Dynamic discrete choice: Rust's NFXP, Hotz-Miller CCP and Aguirregabiria-Mira NPL.

Exposed lazily as ``sp.ddc`` and ``sp.dgp_ddc``.
"""

from ._result import DDCResult
from .estimate import ddc
from .simulate import dgp_ddc

__all__ = ["ddc", "dgp_ddc", "DDCResult"]
