"""
Long-term effects via surrogate indices (``sp.surrogate``).

Industrial A/B tests can only run for weeks, but the quantities of interest
(LTV, retention, clinical outcomes) are months or years downstream. The
surrogate-index framework lets you extrapolate short-term surrogates to
long-term outcomes by combining an experimental sample (with the surrogate)
and an observational sample (with both surrogate and long-term outcome).

Estimators
----------
- :func:`surrogate_index` — Athey, Chetty, Imbens & Kang (2019).
  Classical single-wave surrogate index.
- :func:`long_term_from_short` — Tran, Bibaut & Kallus (arXiv:2311.08527,
  2023). Long-term effect of long-term treatments from short-term
  experiments.
- :func:`proximal_surrogate_index` — Imbens, Kallus, Mao & Wang (2025, JRSS-B
  87(2); arXiv:2202.07234). Proximal identification when unobserved
  confounders link surrogate and long-term outcome.

References
----------
Athey, S., Chetty, R., Imbens, G. W., & Kang, H. (2019).
"The Surrogate Index: Combining Short-Term Proxies to Estimate Long-Term
Treatment Effects More Rapidly and Precisely." NBER Working Paper 26463. [@athey2019surrogate]

Imbens, G., Kallus, N., Mao, X., & Wang, Y. (2025).
"Long-term Causal Inference Under Persistent Confounding via Data
Combination." Journal of the Royal Statistical Society Series B,
87(2), 362-388. arXiv:2202.07234. [@imbens2025long]
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
