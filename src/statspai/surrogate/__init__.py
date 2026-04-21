"""
Long-term effects via surrogate indices (``sp.surrogate``).

Industrial A/B tests can only run for weeks, but the quantities of interest
(LTV, retention, clinical outcomes) are months or years downstream. The
surrogate-index framework lets you extrapolate short-term surrogates to
long-term outcomes by combining an experimental sample (with the surrogate)
and an observational sample (with both surrogate and long-term outcome).

Estimators
----------
- :func:`surrogate_index` — Athey, Chetty, Imbens, Pollmann & Taubinsky (2019).
  Classical single-wave surrogate index.
- :func:`long_term_from_short` — Ghassami, Yang, Shpitser, Tchetgen Tchetgen
  (arXiv:2311.08527, 2024). Long-term effect of long-term treatments from
  short-term experiments.
- :func:`proximal_surrogate_index` — Imbens, Kallus, Mao (arXiv:2601.17712,
  2026). Proximal identification when unobserved confounders link surrogate
  and long-term outcome.

References
----------
Athey, S., Chetty, R., Imbens, G., Pollmann, M., & Taubinsky, D. (2019).
"The Surrogate Index: Combining Short-Term Proxies to Estimate Long-Term
Treatment Effects More Rapidly and Precisely." NBER WP 26463.

Imbens, G., Kallus, N., Mao, X. (2026).
"The Proximal Surrogate Index: Long-Term Treatment Effects under
Unobserved Confounding." arXiv:2601.17712.
"""

from .index import (
    surrogate_index,
    long_term_from_short,
    proximal_surrogate_index,
    SurrogateResult,
)

__all__ = [
    "surrogate_index",
    "long_term_from_short",
    "proximal_surrogate_index",
    "SurrogateResult",
]
