"""v1.6 MR Frontier: four post-2020 MR estimators in one coherent family.

- :func:`mr_lap` — sample-overlap-corrected IVW (Burgess, Davies &
  Thompson 2016; Mounier & Kutalik 2023).
- :func:`mr_clust` — clustered causal effects via finite Gaussian
  mixture on Wald ratios (Foley et al. 2021).
- :func:`grapple` — profile-likelihood MR with weak-instrument and
  pleiotropy robustness (Wang et al. 2021, single-exposure variant).
- :func:`mr_cml` — constrained ML with L0-sparse pleiotropy, MR-cML-BIC
  (Xue, Shen & Pan 2021).

Design note: CAUSE (Morrison et al. 2020) was originally scoped for
v1.6 but its full variational-Bayes implementation is ~5000 LOC in the
R reference and cannot be reference-parity validated in-cycle.
MR-cML covers the same use-case (correlated + uncorrelated pleiotropy)
with a tractable constrained-ML formulation and ships instead.
"""

from .lap import MRLapResult, mr_lap
from .clust import MRClustResult, mr_clust
from .grapple import GrappleResult, grapple
from .cml import MRcMLResult, mr_cml
from .raps import MRRapsResult, mr_raps

__all__ = [
    "mr_lap", "mr_clust", "grapple", "mr_cml", "mr_raps",
    "MRLapResult", "MRClustResult", "GrappleResult",
    "MRcMLResult", "MRRapsResult",
]
