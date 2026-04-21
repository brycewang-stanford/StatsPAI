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
>>> print(proto.summary())
"""

from .protocol import TargetTrialProtocol, protocol
from .emulate import emulate, TargetTrialResult
from .ccw import clone_censor_weight, CloneCensorWeightResult
from .diagnostics import immortal_time_check, ImmortalTimeDiagnostic
from .report import to_paper, target_checklist, TARGET_ITEMS

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
