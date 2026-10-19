"""Survival and duration analysis models."""

from .aft import AFTResult, aft
from .causal_forest import (
    CausalSurvivalForestResult,
    causal_survival,
    causal_survival_forest,
)
from .competing_risks import CumIncResult, FineGrayResult, cuminc, finegray
from .frailty import FrailtyResult, cox_frailty
from .models import CoxResult, KMResult, cox, kaplan_meier, logrank_test, survreg

# Imported on first use: the Cox / Kaplan-Meier path (and the parity traces
# that hash every file it executes) does not load them.
_LAZY = {
    "rmst": ".restricted_mean",
    "RMSTResult": ".restricted_mean",
    "stteffects": ".teffects",
}


def __getattr__(name: str):  # noqa: ANN202
    if name in _LAZY:
        import importlib

        value = getattr(importlib.import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "cox",
    "kaplan_meier",
    "survreg",
    "CoxResult",
    "KMResult",
    "logrank_test",
    "cox_frailty",
    "FrailtyResult",
    "aft",
    "AFTResult",
    "causal_survival_forest",
    "causal_survival",
    "CausalSurvivalForestResult",
    "cuminc",
    "finegray",
    "CumIncResult",
    "FineGrayResult",
    "rmst",
    "RMSTResult",
    "stteffects",
]
