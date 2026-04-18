"""Stochastic frontier analysis (SFA).

Cross-sectional estimators: :func:`frontier` (half-normal / exponential /
truncated-normal; supports heteroskedastic ``sigma_u`` & ``sigma_v`` plus
inefficiency determinants ``emean``).

Panel estimators: :func:`xtfrontier` with ``model`` in
``{'ti', 'tvd', 'bc95'}`` (Pitt-Lee 1981, Battese-Coelli 1992,
Battese-Coelli 1995).

Helpers: :func:`te_summary`.
"""

from .sfa import frontier, FrontierResult
from .panel import xtfrontier
from .te_tools import te_summary, te_rank

__all__ = [
    "frontier",
    "xtfrontier",
    "FrontierResult",
    "te_summary",
    "te_rank",
]
