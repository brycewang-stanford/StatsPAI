"""Result object of ``sp.fmm``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin

_PREDICT_TYPES = ("mu", "marginal", "classpr", "classposteriorpr")


@dataclass
class FMMResult(ResultProtocolMixin):
    """Output of :func:`statspai.fmm` (finite mixture of regressions).

    Attributes
    ----------
    params, std_errors : pd.Series
        Every free parameter, named as Stata's ``e(b)`` names them, with the
        intercept written ``_cons``: ``"2.Class:z"`` (class membership,
        class 1 is the base), ``"y:2.Class#x1"`` (outcome equation of class
        2) and ``"/var(e.y)#2.Class"`` (Gaussian error variance).
    vcov : pd.DataFrame
        Covariance of ``params`` (OIM, robust or cluster).
    class_probs : pd.DataFrame
        Marginal class probabilities (Stata ``estat lcprob``): the sample
        average of each observation's prior class probability, with
        delta-method standard errors and a logit-scale interval.
    class_means : pd.DataFrame
        Class-specific marginal means of the outcome (``estat lcmean``).
    posterior : pd.DataFrame
        Posterior class probabilities of each estimation-sample row
        (``predict, classposteriorpr``), indexed like ``data``.
    prior : pd.DataFrame
        Prior class probabilities ``pi_k(z_i)`` (``predict, classpr``).
    fitted : pd.DataFrame
        Class-specific predicted means (``predict, mu``).
    loglik, aic, bic : float
        Log-likelihood at the maximum; AIC and BIC with ``N`` the number of
        observations (the sum of frequency weights), as ``estat ic``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=500)
    >>> c = rng.uniform(size=500) < 0.5
    >>> y = np.where(c, 5 + x, -x) + 0.5 * rng.normal(size=500)
    >>> res = sp.fmm("y ~ x", pd.DataFrame({"y": y, "x": x}), k=2, seed=0)
    >>> type(res).__name__
    'FMMResult'
    >>> list(res.coef_table(1).index)
    ['_cons', 'x', 'var(e.y)']
    >>> res.posterior.shape
    (500, 2)
    """

    formula: str
    depvar: str
    family: str
    k: int
    params: pd.Series
    std_errors: pd.Series
    vcov: pd.DataFrame
    class_probs: pd.DataFrame
    class_means: pd.DataFrame
    class_tables: Dict[int, pd.DataFrame]
    membership_table: pd.DataFrame
    posterior: pd.DataFrame
    prior: pd.DataFrame
    fitted: pd.DataFrame
    loglik: float
    aic: float
    bic: float
    n_obs: int
    n_ic: float
    df_model: int
    vce: str
    converged: bool
    iterations: int
    alpha: float
    class_order: str
    start_logliks: List[float]
    model_info: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    _predictor: Any = field(default=None, repr=False)

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "mclachlan2000finite",
        "dempster1977maximum",
    )

    # ------------------------------------------------------------------
    @property
    def method(self) -> str:
        return f"Finite mixture model ({self.k} classes, {self.family})"

    @property
    def se(self) -> pd.Series:
        return self.std_errors

    @property
    def pvalues(self) -> pd.Series:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.params / self.std_errors
        p = 2.0 * stats.norm.sf(np.abs(z))
        out = pd.Series(p, index=self.params.index)
        out[self._is_variance()] = np.nan
        return out

    def _is_variance(self) -> np.ndarray:
        return np.array([n.startswith("/var(") for n in self.params.index])

    # ------------------------------------------------------------------
    def tidy(self, conf_level: Optional[float] = None) -> pd.DataFrame:
        """One row per parameter: ``equation`` (``class`` / ``outcome`` /
        ``variance``), ``class``, ``term``, estimate, standard error, z,
        p-value and interval. Variance intervals are on the log scale and
        carry no test, as Stata prints them."""
        level = 1.0 - self.alpha if conf_level is None else float(conf_level)
        crit = float(stats.norm.ppf(0.5 + level / 2.0))
        rows = []
        for name in self.params.index:
            b = float(self.params[name])
            s = float(self.std_errors[name])
            eq, cls, term = _split_name(name)
            if eq == "variance":
                half = crit * s / b if b > 0 else np.nan
                lo, hi = b * np.exp(-half), b * np.exp(half)
                z = p = np.nan
            else:
                lo, hi = b - crit * s, b + crit * s
                z = b / s if s > 0 else np.nan
                p = float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
            rows.append(
                {
                    "equation": eq,
                    "class": cls,
                    "term": term,
                    "estimate": b,
                    "std_error": s,
                    "statistic": z,
                    "p_value": p,
                    "conf_low": lo,
                    "conf_high": hi,
                }
            )
        return pd.DataFrame(rows)

    def coef_table(self, cls: int) -> pd.DataFrame:
        """Outcome equation (and variance) of class ``cls`` (1-based)."""
        if cls not in self.class_tables:
            raise KeyError(
                f"class must be one of {sorted(self.class_tables)}; got {cls!r}"
            )
        return self.class_tables[cls].copy()

    def predict(
        self,
        newdata: Optional[pd.DataFrame] = None,
        type: str = "mu",
    ) -> pd.DataFrame:
        """Predictions, as Stata's ``predict`` after ``fmm``.

        Parameters
        ----------
        newdata : pd.DataFrame, optional
            Rows to predict for; the estimation sample when omitted. It
            needs the outcome column only for ``type='classposteriorpr'``.
        type : {"mu", "marginal", "classpr", "classposteriorpr"}
            ``"mu"``: class-specific means, one column per class.
            ``"marginal"``: the mixture mean ``sum_k pi_k mu_k``.
            ``"classpr"``: prior class probabilities. ``"classposteriorpr"``:
            posterior class probabilities given the observed outcome.
        """
        key = str(type).lower()
        if key not in _PREDICT_TYPES:
            from ..exceptions import MethodIncompatibility

            raise MethodIncompatibility(
                f"fmm.predict: type must be one of {list(_PREDICT_TYPES)}; "
                f"got {type!r}.",
                recovery_hint="Use type='mu', 'marginal', 'classpr' or "
                "'classposteriorpr'.",
            )
        if newdata is None:
            pieces = {
                "mu": self.fitted,
                "classpr": self.prior,
                "classposteriorpr": self.posterior,
            }
            if key == "marginal":
                m = (self.prior.to_numpy() * self.fitted.to_numpy()).sum(axis=1)
                return pd.DataFrame({"marginal": m}, index=self.fitted.index)
            return pieces[key].copy()
        return self._predictor(newdata, key)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        rule = "-" * 78
        pct = int(round(100 * (1 - self.alpha)))
        se_label = {"oim": "OIM", "robust": "Robust", "cluster": "Robust"}[self.vce]
        head = [
            f"Finite mixture model ({self.family})",
            "=" * 78,
            f"  Classes: {self.k}{'':<16}Number of obs = {self.n_obs}",
            f"  Log likelihood = {self.loglik:.4f}{'':<6}AIC = {self.aic:.3f}"
            f"   BIC = {self.bic:.3f}",
            f"  Std. err.: {se_label}"
            + (
                f" (adjusted for {self.model_info.get('n_clusters')} clusters)"
                if self.vce == "cluster"
                else ""
            ),
            f"  Classes ordered by {self._order_text()}",
        ]

        def block(title: str, table: pd.DataFrame) -> List[str]:
            t = table.copy()
            fmt = t.to_string(float_format=lambda v: f"{v: .6g}")
            return [rule, f"  {title}", rule, fmt]

        lines = list(head)
        if not self.membership_table.empty:
            lines += block(
                "Class membership (multinomial logit, class 1 = base)",
                self.membership_table,
            )
        for cls, table in self.class_tables.items():
            lines += block(f"Class {cls}: {self.depvar} ({self.family})", table)
        lines += block(
            f"Marginal class probabilities (delta method, {pct}% CI on logit scale)",
            self.class_probs,
        )
        lines += block("Class-specific marginal means", self.class_means)
        lines.append(rule)
        return "\n".join(lines)

    def _order_text(self) -> str:
        if self.class_order == "share":
            return "descending marginal class probability"
        if self.class_order == "none":
            return "the labels of the best start (no reordering)"
        return "ascending class-specific mean of the outcome"

    def __repr__(self) -> str:
        shares = ", ".join(f"{v:.3f}" for v in self.class_probs["estimate"])
        return (
            f"<FMMResult {self.family}, k={self.k}, n={self.n_obs}, "
            f"ll={self.loglik:.4f}, class shares=[{shares}]>"
        )

    def __str__(self) -> str:
        return self.summary()


def _split_name(name: str) -> Tuple[str, Optional[int], str]:
    """``(equation, class, term)`` from a parameter name."""
    if name.startswith("/var("):
        cls = int(name.split("#")[1].split(".")[0])
        return "variance", cls, name.split("#")[0].lstrip("/")
    head, term = name.split(":", 1)
    if head.endswith(".Class"):
        return "class", int(head.split(".")[0]), term
    cls_str, term = term.split("#", 1)
    return "outcome", int(cls_str.split(".")[0]), term
