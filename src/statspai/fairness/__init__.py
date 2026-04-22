"""
Counterfactual fairness & algorithmic-bias diagnostics (``sp.fairness``).

Treats fairness as a causal-inference problem rather than a pure
statistics one: an algorithm is counterfactually fair if the prediction
would be the same had the protected attribute been different, holding
the non-protected causal ancestors fixed.

Core tools
----------
- :func:`counterfactual_fairness` — Kusner, Loftus, Russell, Silva (2018)
  Level-2/3 predictor evaluation on a user-supplied SCM.
- :func:`orthogonal_to_bias` — Chen & Zhu (arXiv:2403.17852v3, 2024)
  Data pre-processing that removes the part of non-protected features
  correlated with the protected attribute, via residualization.
- :func:`demographic_parity` — P(Y_hat=1 | A=a) — A.
- :func:`equalized_odds` — P(Y_hat=1 | Y=y, A=a) — A.
- :func:`fairness_audit` — one-shot dashboard combining the metrics above.

References
----------
Kusner, M. J., Loftus, J., Russell, C., & Silva, R. (2018).
Counterfactual fairness. NeurIPS.

Hardt, M., Price, E., & Srebro, N. (2016).
Equality of opportunity in supervised learning. NeurIPS.

Chen, S., & Zhu, S. (2024).
Counterfactual Fairness Through Orthogonal to Bias. arXiv:2403.17852v3.
"""

from .core import (
    counterfactual_fairness,
    orthogonal_to_bias,
    demographic_parity,
    equalized_odds,
    fairness_audit,
    FairnessResult,
    FairnessAudit,
)
from .evidence_test import (
    evidence_without_injustice,
    EvidenceWithoutInjusticeResult,
)

__all__ = [
    "counterfactual_fairness",
    "orthogonal_to_bias",
    "demographic_parity",
    "equalized_odds",
    "fairness_audit",
    "FairnessResult",
    "FairnessAudit",
    "evidence_without_injustice",
    "EvidenceWithoutInjusticeResult",
]
