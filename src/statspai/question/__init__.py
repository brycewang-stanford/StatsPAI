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
>>> df = sp.dgp_did(n_units=100, n_periods=10, seed=0)
>>> df["first_treat"] = df["first_treat"].fillna(0).astype(int)
>>> q = sp.causal_question(
...     treatment="first_treat",   # first treatment period, 0 = never
...     outcome="y",
...     estimand="ATT",
...     design="did",
...     data=df,
...     time_structure="panel",
...     time="time",
...     id="unit",
... )
>>> q.identify().estimator
'did'
>>> r = q.estimate()
>>> r.estimand
'ATT'
>>> print(q.report().splitlines()[0])
## Causal Question
"""

from .preregister import load_preregister, preregister
from .question import (
    CausalQuestion,
    EstimationResult,
    IdentificationPlan,
    causal_question,
)

__all__ = [
    "CausalQuestion",
    "causal_question",
    "IdentificationPlan",
    "EstimationResult",
    "preregister",
    "load_preregister",
]
