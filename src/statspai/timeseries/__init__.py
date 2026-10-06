"""
Time series methods: models for causal inference contexts and forecasting.

Provides VAR (vector autoregression), structural break tests,
Granger causality, unit-root tests and cointegration analysis; and a
forecasting toolkit -- ARIMA, exponential smoothing (ETS), benchmark
methods, STL decomposition, forecast accuracy, rolling-origin
cross-validation and hierarchical reconciliation.
"""

from typing import Any

from .ardl import ARDLResult, ardl
from .arima import ARIMAResult, arima
from .bagging import BaggedForecastResult, bagged_forecast, bootstrap_series
from .bds import bds
from .bvar import BVARResult, bvar
from .chow import chow_test
from .cointegration import CointegrationResult, engle_granger, johansen
from .corrgram import corrgram
from .dlm import DLMResult, dlm
from .forecast_accuracy import TSCVResult, forecast_accuracy, tscv
from .garch import GARCHResult, garch
from .its import ITSResult, its
from .local_projections import LocalProjectionsResult, local_projections
from .reconcile import Hierarchy, ReconcileResult, hierarchy, reconcile
from .simple_forecast import SimpleForecastResult, simple_forecast
from .stl import DecompositionResult, classical_decompose, stl
from .structural_break import StructuralBreakResult, cusum_test, structural_break
from .svar import SVARResult, svar
from .ts_features import ts_features
from .ts_tools import (
    boxcox_lambda,
    fourier_terms,
    ljungbox,
    ndiffs,
    nsdiffs,
    seasonal_dummies,
)
from .unit_root import UnitRootResult, unitroot
from .var import VARResult, granger_causality, irf, var
from .var_diagnostics import varsoc
from .vecm import VECResult, vec

__all__ = [
    "var",
    "VARResult",
    "svar",
    "SVARResult",
    "granger_causality",
    "irf",
    "structural_break",
    "StructuralBreakResult",
    "cusum_test",
    "chow_test",
    "bds",
    "engle_granger",
    "johansen",
    "CointegrationResult",
    "local_projections",
    "LocalProjectionsResult",
    "garch",
    "GARCHResult",
    "arima",
    "ARIMAResult",
    "bvar",
    "dlm",
    "DLMResult",
    "BVARResult",
    "its",
    "ITSResult",
    "unitroot",
    "UnitRootResult",
    "ardl",
    "ARDLResult",
    "corrgram",
    "varsoc",
    "vec",
    "VECResult",
    "ets",
    "ETSResult",
    "simple_forecast",
    "SimpleForecastResult",
    "forecast_accuracy",
    "tscv",
    "TSCVResult",
    "stl",
    "classical_decompose",
    "DecompositionResult",
    "ljungbox",
    "ndiffs",
    "nsdiffs",
    "boxcox_lambda",
    "fourier_terms",
    "hierarchy",
    "Hierarchy",
    "reconcile",
    "ReconcileResult",
    "ts_features",
    "bootstrap_series",
    "bagged_forecast",
    "BaggedForecastResult",
    "seasonal_dummies",
]


def __getattr__(name: str) -> Any:
    # ``ets`` compiles its recursions with numba; importing it on first use
    # keeps ``import statspai`` free of that import.
    if name in ("ets", "ETSResult"):
        from . import _ets

        return getattr(_ets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
