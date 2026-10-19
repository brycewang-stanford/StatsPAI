"""Result object of :func:`statspai.mi_estimate` (Rubin's rules)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .._dict_result import DictResult


class MIEstimateResult(DictResult):
    """Rubin-pooled estimates from :func:`mi_estimate`.

    A ``dict`` with the keys of :meth:`MICEResult.combine` (``r["params"]``
    is still the array). The attributes ``params`` / ``std_errors`` /
    ``tvalues`` / ``pvalues`` / ``conf_int_lower`` / ``conf_int_upper`` are
    the same numbers as labelled Series, so ``sp.regtable(r)`` and
    ``r.tidy()`` work; ``df_resid`` is the smallest Barnard-Rubin df.
    """

    _title = "Multiple-imputation estimates (Rubin's rules)"
    _citation_keys = ("rubin1987multiple", "barnard1999small")

    def _series(self, key: str) -> pd.Series:
        names = self.get("var_names") or list(range(len(self["params"])))
        return pd.Series(np.asarray(self[key], dtype=float), index=names)

    @property
    def params(self) -> pd.Series:
        return self._series("params")

    @property
    def std_errors(self) -> pd.Series:
        return self._series("se")

    @property
    def tvalues(self) -> pd.Series:
        return self._series("tvalues")

    @property
    def pvalues(self) -> pd.Series:
        return self._series("pvalues")

    @property
    def conf_int_lower(self) -> pd.Series:
        return self._series("ci_lower")

    @property
    def conf_int_upper(self) -> pd.Series:
        return self._series("ci_upper")

    @property
    def data_info(self) -> Dict[str, Any]:
        return {"nobs": self.get("n_obs"), "df_resid": self.df_resid}

    @property
    def df_resid(self) -> Optional[float]:
        dfs = np.asarray(self.get("df", []), dtype=float)
        return float(np.min(dfs)) if dfs.size else None

    def summary(self) -> str:
        table = pd.DataFrame(
            {
                "coef": self.params,
                "se": self.std_errors,
                "t": self.tvalues,
                "p": self.pvalues,
                "ci_lower": self.conf_int_lower,
                "ci_upper": self.conf_int_upper,
                "fmi": self._series("fmi"),
            }
        )
        return "\n".join(
            [
                self._title,
                f"Imputations: {self['n_imputations']}",
                table.to_string(float_format=lambda v: f"{v:.4f}"),
            ]
        )
