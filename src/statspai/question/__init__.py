"""
Estimand-first causal question DSL (``sp.causal_question``).

The article emphasizes "causal question precedes statistical model" as
the common foundation of all three causal-inference schools:
econometrics' *identification*, epidemiology's *target trial protocol*,
and ML's *estimand-aware learning*.

This module lets a user declare a causal question in one place, then
automatically:

  1. Identify the appropriate research design (IV / DiD / RD / backdoor).
  2. Suggest the right StatsPAI estimator.
  3. Run the analysis and attach diagnostics + sensitivity.
  4. Produce a reproducible Methods paragraph.

>>> import statspai as sp
>>> q = sp.causal_question(
...     treatment="minimum_wage_hike",
...     outcome="employment",
...     estimand="ATT",
...     design="policy_shock",
...     data=df,
...     time_structure="panel",
...     covariates=["industry", "skill"],
... )
>>> q.identify()
>>> r = q.estimate()
>>> q.report()
"""

from .question import (
    CausalQuestion, causal_question,
    IdentificationPlan, EstimationResult,
)
from .preregister import preregister, load_preregister

__all__ = [
    "CausalQuestion", "causal_question",
    "IdentificationPlan", "EstimationResult",
    "preregister", "load_preregister",
]
