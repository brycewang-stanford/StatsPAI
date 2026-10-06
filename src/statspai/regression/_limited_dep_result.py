"""Result wrapper for limited-dependent-variable regression models.

``LimitedDepResult`` is a thin :class:`~statspai.core.results.CausalResult`
subclass used by the limited-dependent-variable family (``tobit``,
``heckman``, ...). These are *regression* models: Stata (``tobit``,
``heckman``) and R (``AER::tobit``, ``sampleSelection::heckit``) report the
entire coefficient table, not a single treatment effect.

The base ``CausalResult.params`` only surfaces the headline estimand, which
is misleading for a multi-regressor model — an agent that calls
``result.params`` on a Tobit fit should see every coefficient, exactly as
``sp.regress(...).params`` does. This subclass exposes the FULL coefficient
vector through the standard ``params`` / ``std_errors`` / ``tvalues`` /
``pvalues`` accessors by reading the ``detail`` table the estimator already
populates (columns ``variable``, ``coefficient``, ``se``, ``z``,
``pvalue``).

``estimate`` / ``se`` / ``ci`` / ``pvalue`` are left untouched and still
carry the headline first-regressor effect, so single-effect consumers and
the existing test contract (``result.estimate``, ``result.detail``,
``result.model_info``) are unaffected. ``isinstance(result, CausalResult)``
remains ``True``.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..core.results import CausalResult


class LimitedDepResult(CausalResult):
    """``CausalResult`` whose params accessors expose the full coef table.

    The full coefficient vector is derived from ``self.detail`` (the table
    the estimator already builds). When ``detail`` is absent or does not
    look like a coefficient table the accessors fall back to the base
    single-estimand behaviour, so the subclass is always safe to use.
    """

    _COEF_COL = "coefficient"
    # Design and estimate kept by the estimator for post-estimation tests
    # (sp.cmtest); None when the fit cannot be rebuilt from them.
    _cm_design: Optional[Dict[str, Any]] = None
    # Unweighted per-observation log-likelihood, when the fit has one.
    _llobs: Optional[Any] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _coef_table(self) -> Optional[pd.DataFrame]:
        d = self.detail
        if d is None:
            return None
        cols = getattr(d, "columns", [])
        if "variable" not in cols or self._COEF_COL not in cols:
            return None
        return d.set_index("variable")

    def _column_series(self, column: str) -> Optional[pd.Series]:
        d = self._coef_table()
        if d is None or column not in d.columns:
            return None
        s = pd.to_numeric(d[column], errors="coerce").astype(float)
        s.index = list(d.index)
        s.index.name = None
        return s

    # ------------------------------------------------------------------
    # Full-vector accessors (override the single-estimand base properties)
    # ------------------------------------------------------------------
    @property
    def params(self) -> pd.Series:
        s = self._column_series(self._COEF_COL)
        if s is not None:
            return s
        return super().params

    @property
    def std_errors(self) -> pd.Series:
        s = self._column_series("se")
        if s is not None:
            return s
        return super().std_errors

    @property
    def tvalues(self) -> pd.Series:
        s = self._column_series("z")
        if s is not None:
            return s
        params = self.params
        ses = self.std_errors
        if len(params) > 1 and params.index.equals(ses.index):
            with np.errstate(divide="ignore", invalid="ignore"):
                return params / ses
        return super().tvalues

    @property
    def pvalues(self) -> pd.Series:
        s = self._column_series("pvalue")
        if s is not None:
            return s
        return super().pvalues

    # ------------------------------------------------------------------
    # Summary: a second equation, when the estimator reports one
    # ------------------------------------------------------------------
    def summary(self, *args: Any, **kwargs: Any) -> Any:
        """The base summary, with the selection equation of a two-step
        Heckman fit (``model_info['selection_equation']``) printed below
        the outcome equation, as Stata prints the ``select`` block."""
        text = super().summary(*args, **kwargs)
        table = (self.model_info or {}).get("selection_equation")
        if not isinstance(table, pd.DataFrame) or not isinstance(text, str):
            return text
        rule = "-" * 78
        block = "\n".join(
            [
                rule,
                "  Selection Equation (probit)",
                rule,
                table.to_string(index=False, float_format="%.4f"),
                "",
            ]
        )
        marker = f"{rule}\n  Observations:"
        body = str(text)
        if marker not in body:
            return text
        return type(text)(body.replace(marker, f"{block}\n{marker}", 1))

    # ------------------------------------------------------------------
    # Convenience: full coefficient table as a tidy DataFrame
    # ------------------------------------------------------------------
    def coef_table(self) -> pd.DataFrame:
        """Return the full coefficient table (coef / se / z / p / CI).

        Mirrors the Stata / R coefficient block. Falls back to a one-row
        table built from the headline estimate when no ``detail`` table is
        available.
        """
        d = self._coef_table()
        if d is None:
            return pd.DataFrame(
                {
                    "coefficient": [self.estimate],
                    "se": [self.se],
                    "pvalue": [self.pvalue],
                },
                index=[self.estimand],
            )
        out = pd.DataFrame(index=list(d.index))
        out.index.name = "variable"
        out["coefficient"] = self.params.values
        out["se"] = self.std_errors.values
        out["z"] = self.tvalues.values
        out["pvalue"] = self.pvalues.values
        return out
