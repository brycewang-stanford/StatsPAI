"""ProductionResult — unified output container for proxy-variable estimators."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...core.results import EconometricResults


_METHOD_REFS: Dict[str, str] = {
    "op":  "Olley & Pakes (1996, Econometrica)",
    "lp":  "Levinsohn & Petrin (2003, Review of Economic Studies)",
    "acf": "Ackerberg, Caves & Frazer (2015, Econometrica)",
    "wrdg": "Wooldridge (2009, Economics Letters)",
}


class ProductionResult(EconometricResults):
    """Result object for production function estimation.

    Inherits ``params`` / ``std_errors`` / ``summary`` / ``to_dict``
    from :class:`EconometricResults` and adds production-function-specific
    payload:

    * ``coef`` — input elasticities keyed by input name
      (e.g. ``{"l": 0.62, "k": 0.31}``)
    * ``tfp`` — firm-time TFP estimates ``omega_it`` (in logs); same length
      as the post-stage-2 working sample
    * ``residuals`` — i.i.d. shock ``eta_it`` from stage 1
    * ``productivity_process`` — ``{"rho": float, "sigma": float}`` from the
      AR fit on omega
    * ``markup`` — placeholder; populated by :func:`statspai.markup`

    Use ``.summary()`` for a Stata-style table or ``.coef`` for the raw dict.
    """

    def __init__(
        self,
        method: str,
        params: pd.Series,
        std_errors: pd.Series,
        coef: Dict[str, float],
        tfp: np.ndarray,
        residuals: np.ndarray,
        productivity_process: Dict[str, float],
        sample: pd.DataFrame,
        diagnostics: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        cov: Optional[np.ndarray] = None,
    ):
        info = {
            "model_type": "Production function",
            "method": method,
            "reference": _METHOD_REFS.get(method, ""),
        }
        if model_info:
            info.update(model_info)
        super().__init__(
            params=params,
            std_errors=std_errors,
            model_info=info,
            data_info={"n_obs": int(len(tfp)), "n_firms": int(sample["__panel_id__"].nunique())},
            diagnostics=diagnostics or {},
        )
        self.method = method
        self.coef = coef
        self.tfp = np.asarray(tfp)
        self.residuals = np.asarray(residuals)
        self.productivity_process = productivity_process
        self.sample = sample
        self.cov = cov
        self.markup: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Public surface
    # ------------------------------------------------------------------ #

    def cite(self) -> str:
        """Return the canonical reference string for ``self.method``."""
        return _METHOD_REFS.get(self.method, "")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        coefs = ", ".join(f"{k}={v:.4f}" for k, v in self.coef.items())
        return (
            f"ProductionResult(method={self.method!r}, "
            f"n_obs={self.data_info.get('n_obs')}, "
            f"n_firms={self.data_info.get('n_firms')}, {coefs})"
        )
