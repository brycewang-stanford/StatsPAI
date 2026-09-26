"""
Target Trial Emulation (``sp.target_trial``).

JAMA 2022 framework — the unifying language for causal inference from
observational data. Use to formalize the target trial before analysis,
then delegate estimation to ``sp.msm`` / ``sp.tmle`` / ``sp.ltmle``.

Quick start
-----------
>>> import statspai as sp
>>> proto = sp.target_trial.protocol(
...     eligibility="age >= 50 and diabetic == 1",
...     treatment_strategies=["statin at t0", "no statin"],
...     assignment="observational emulation",
...     time_zero="date of diabetes diagnosis",
...     followup_end="min(death, loss, 5y)",
...     outcome="incident MI",
...     causal_contrast="per-protocol",
...     analysis_plan="clone-censor-weight + pooled logistic + IPCW",
...     baseline_covariates=["age", "sex", "bmi", "ldl"],
... )
>>> print(proto.summary())  # doctest: +ELLIPSIS
Target Trial Protocol
========================================
1. Eligibility: age >= 50 and diabetic == 1
2. Treatment strategies: statin at t0, no statin
...
"""

from .ccw import CloneCensorWeightResult, clone_censor_weight
from .diagnostics import ImmortalTimeDiagnostic, immortal_time_check
from .emulate import TargetTrialResult, emulate
from .protocol import TargetTrialProtocol, protocol
from .report import TARGET_ITEMS, target_checklist, to_paper

__all__ = [
    "TargetTrialProtocol",
    "protocol",
    "emulate",
    "TargetTrialResult",
    "clone_censor_weight",
    "CloneCensorWeightResult",
    "immortal_time_check",
    "ImmortalTimeDiagnostic",
    "to_paper",
    "target_checklist",
    "TARGET_ITEMS",
]
