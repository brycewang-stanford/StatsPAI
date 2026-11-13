"""Meta-regression and trim-and-fill for summary-data meta-analysis.

``meta_regression`` fits the mixed-effects model ``y_i = x_i' beta + u_i +
e_i`` (study effects on study-level moderators) with the between-study
variance estimated by DerSimonian-Laird, REML, ML or Paule-Mandel; it is R
``metafor::rma(yi, sei = , mods = )`` and Stata ``meta regress``.

``trim_and_fill`` is the Duval-Tweedie rank-based estimate of the number of
studies missing from one side of the funnel plot, with the pooled effect
re-estimated after imputing them; it is R ``metafor::trimfill``.

Both are pinned to ``metafor`` on the same data in
``tests/reference_parity/test_gaps_meta_parity.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _meta_core as _core
from .basic_tests import _check_alpha
from .meta_analysis import MetaAnalysisResult, meta_analysis

__all__ = [
    "MetaRegressionResult",
    "TrimFillResult",
    "meta_regression",
    "trim_and_fill",
]

_TAU2_LABEL = {
    "DL": "DerSimonian-Laird",
    "REML": "REML",
    "ML": "maximum likelihood",
    "PM": "Paule-Mandel",
    "fixed": "none (fixed effects)",
}


def _study_inputs(effects: Any, se: Any, who: str) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(effects, dtype=float)
    s = np.asarray(se, dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise MethodIncompatibility(
            f"{who}: effects and se must be 1-D and of equal length, got "
            f"shapes {y.shape} and {s.shape}.",
            recovery_hint="Pass one effect size and one standard error per study.",
        )
    bad = ~(np.isfinite(y) & np.isfinite(s))
    if bad.any():
        raise MethodIncompatibility(
            f"{who}: effects and standard errors must be finite; "
            f"{int(bad.sum())} study(ies) have a missing or infinite value.",
            recovery_hint="Drop those studies first.",
        )
    if np.any(s <= 0):
        raise MethodIncompatibility(
            f"{who}: all standard errors must be positive.",
            recovery_hint="Drop studies with a zero or negative standard error.",
        )
    return y, s


# ---------------------------------------------------------------------------
# Meta-regression
# ---------------------------------------------------------------------------


@dataclass
class MetaRegressionResult(ResultProtocolMixin):
    """Result of :func:`meta_regression`.

    Attributes
    ----------
    params, std_errors, statistics, p_values : pandas.Series
        Coefficients (intercept first), their standard errors, the z (or,
        with ``hksj=True``, t) statistics and two-sided p-values.
    conf_int : pandas.DataFrame
        Columns ``lower`` and ``upper``.
    vcov : pandas.DataFrame
        Covariance matrix of the coefficients.
    tau2 : float
        Residual between-study variance.
    i2, h2 : float
        Residual heterogeneity as a share of the unaccounted variability,
        ``tau2 / (tau2 + v_typical)``, and ``1 + tau2 / v_typical``.
    r2 : float
        Share of the between-study variance of the intercept-only model
        that the moderators account for, ``(tau2_null - tau2) / tau2_null``
        truncated at 0; ``nan`` when ``tau2_null`` is 0.
    tau2_null : float
        Between-study variance of the model without moderators.
    qe, qe_df, qe_pvalue : float
        Test of residual heterogeneity.
    qm, qm_df, qm_pvalue : float, tuple, float
        Omnibus test of the moderators: chi-square with ``qm_df`` degrees of
        freedom, or with ``hksj=True`` an F statistic with ``qm_df = (m,
        k - p)``.
    method : str
        Estimator of ``tau2``.
    hksj : bool
        Knapp-Hartung inference.
    k : int
        Number of studies.
    """

    _citation_keys = ("dersimonian1986meta", "higgins2002quantifying")

    params: pd.Series
    std_errors: pd.Series
    statistics: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame
    vcov: pd.DataFrame
    tau2: float
    i2: float
    h2: float
    r2: float
    tau2_null: float
    qe: float
    qe_df: int
    qe_pvalue: float
    qm: float
    qm_df: Any
    qm_pvalue: float
    method: str
    hksj: bool
    k: int
    alpha: float = 0.05
    fitted: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    residuals: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    intercept: bool = True

    def tidy(self) -> pd.DataFrame:
        """Coefficient table as a DataFrame."""
        stat = "t" if self.hksj else "z"
        return pd.DataFrame(
            {
                "term": list(self.params.index),
                "estimate": self.params.to_numpy(),
                "std_error": self.std_errors.to_numpy(),
                stat: self.statistics.to_numpy(),
                "p_value": self.p_values.to_numpy(),
                "conf_low": self.conf_int["lower"].to_numpy(),
                "conf_high": self.conf_int["upper"].to_numpy(),
            }
        )

    def predict(self, moderators: Any = None) -> pd.DataFrame:
        """Predicted average effect at moderator values, with its interval.

        Parameters
        ----------
        moderators : array-like, mapping or DataFrame, optional
            Moderator values, in the order (or with the names) of the fit.
            Default: the fitted values of the studies.

        Returns
        -------
        pandas.DataFrame
            ``prediction``, ``se``, ``conf_low``, ``conf_high``.
        """
        names = [n for n in self.params.index if n != "intercept"]
        if moderators is None:
            raise MethodIncompatibility(
                "MetaRegressionResult.predict: pass the moderator values to "
                "predict at.",
                recovery_hint=(
                    "result.predict({'x': [0, 1]}); the fitted values of the "
                    "studies are in result.fitted."
                ),
            )
        Z, _names = _moderator_matrix(moderators, None, names=names, who="predict")
        if Z.shape[1] != len(names):
            raise MethodIncompatibility(
                f"predict: the fit has {len(names)} moderator(s) {names!r}, got "
                f"{Z.shape[1]} column(s).",
                recovery_hint="Pass one column per moderator of the fit.",
            )
        X = np.column_stack([np.ones(len(Z)), Z]) if self.intercept else Z
        pred = X @ self.params.to_numpy()
        se = np.sqrt(np.einsum("ij,jk,ik->i", X, self.vcov.to_numpy(), X))
        crit = self._crit()
        return pd.DataFrame(
            {
                "prediction": pred,
                "se": se,
                "conf_low": pred - crit * se,
                "conf_high": pred + crit * se,
            }
        )

    def _crit(self) -> float:
        if self.hksj:
            return float(stats.t.ppf(1 - self.alpha / 2, df=self.qe_df))
        return float(stats.norm.ppf(1 - self.alpha / 2))

    def summary(self) -> str:
        """Plain-text report."""
        stat = "t" if self.hksj else "z"
        out = ["=" * 76, "Meta-regression (summary data)", "=" * 76]
        out.append(f"Studies (k)          : {self.k}")
        out.append(f"tau^2 estimator      : {_TAU2_LABEL[self.method]}")
        if self.hksj:
            out.append(
                f"Inference            : Knapp-Hartung adjustment, t({self.qe_df})"
            )
        out.append("-" * 76)
        out.append(
            f"{'':<18}{'coef':>11}{'std err':>11}{stat:>9}{'P>|' + stat + '|':>9}"
            f"{'[' + format(1 - self.alpha, '.0%') + ' conf. int.]':>18}"
        )
        for name in self.params.index:
            lo, hi = self.conf_int.loc[name]
            out.append(
                f"{str(name)[:17]:<18}{self.params[name]:>11.4f}"
                f"{self.std_errors[name]:>11.4f}{self.statistics[name]:>9.3f}"
                f"{self.p_values[name]:>9.4f}{lo:>9.4f}{hi:>9.4f}"
            )
        out.append("-" * 76)
        out.append(
            f"Residual heterogeneity: tau^2 = {self.tau2:.4f}, "
            f"I^2 = {100 * self.i2:.1f}%, H^2 = {self.h2:.4f}"
        )
        if np.isfinite(self.r2):
            out.append(
                f"Heterogeneity accounted for by the moderators: R^2 = "
                f"{100 * self.r2:.1f}%"
            )
        out.append(
            f"Test of residual heterogeneity: QE({self.qe_df}) = {self.qe:.4f}, "
            f"p = {self.qe_pvalue:.4g}"
        )
        if np.isfinite(self.qm):
            if self.hksj:
                d1, d2 = self.qm_df
                out.append(
                    f"Test of moderators: F({d1}, {d2}) = {self.qm:.4f}, "
                    f"p = {self.qm_pvalue:.4g}"
                )
            else:
                out.append(
                    f"Test of moderators: QM({self.qm_df}) = {self.qm:.4f}, "
                    f"p = {self.qm_pvalue:.4g}"
                )
        out.append("=" * 76)
        return "\n".join(out)

    def __repr__(self) -> str:
        return (
            f"<MetaRegressionResult: k={self.k}, {self.method}, "
            f"{len(self.params)} coefficient(s), tau2={self.tau2:.4f}>"
        )


def _moderator_matrix(
    moderators: Any, k: Optional[int], *, names: Optional[Sequence[str]], who: str
) -> Tuple[np.ndarray, List[str]]:
    """Numeric ``(k, m)`` moderator matrix and the moderator names."""
    if isinstance(moderators, pd.DataFrame):
        frame = moderators
    elif isinstance(moderators, pd.Series):
        frame = moderators.to_frame(moderators.name or "x1")
    elif isinstance(moderators, Mapping):
        frame = pd.DataFrame({key: np.asarray(val) for key, val in moderators.items()})
    else:
        arr = np.asarray(moderators)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise MethodIncompatibility(
                f"{who}: moderators must be 1-D or 2-D, got {arr.ndim} dimensions.",
                recovery_hint="Pass one column per moderator, one row per study.",
            )
        cols = list(names) if names is not None and len(names) == arr.shape[1] else None
        frame = pd.DataFrame(
            arr, columns=cols or [f"x{j + 1}" for j in range(arr.shape[1])]
        )
    if names is not None and isinstance(moderators, (pd.DataFrame, Mapping)):
        missing = [n for n in names if n not in frame.columns]
        if missing:
            raise MethodIncompatibility(
                f"{who}: moderator(s) {missing!r} are not in what was passed "
                f"({list(frame.columns)!r}).",
                recovery_hint="Use the moderator names of the fit.",
            )
        frame = frame[list(names)]
    non_numeric = [
        str(c) for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])
    ]
    if non_numeric:
        raise MethodIncompatibility(
            f"{who}: moderator(s) {non_numeric!r} are not numeric.",
            recovery_hint=(
                "Code a categorical moderator as dummy columns, for example "
                "pd.get_dummies(col, drop_first=True, dtype=float)."
            ),
        )
    Z = frame.to_numpy(dtype=float)
    if k is not None and Z.shape[0] != k:
        raise MethodIncompatibility(
            f"{who}: {Z.shape[0]} rows of moderators for {k} studies.",
            recovery_hint="Pass one row of moderators per study.",
        )
    if not np.all(np.isfinite(Z)):
        raise MethodIncompatibility(
            f"{who}: moderators must be finite; "
            f"{int((~np.isfinite(Z)).any(axis=1).sum())} study(ies) have a "
            "missing or infinite value.",
            recovery_hint="Drop those studies, or impute the moderator.",
        )
    return Z, [str(c) for c in frame.columns]


def meta_regression(
    effects: Sequence[float],
    se: Sequence[float],
    moderators: Any,
    *,
    method: str = "REML",
    hksj: bool = False,
    intercept: bool = True,
    names: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
) -> MetaRegressionResult:
    """Meta-regression of study effects on study-level moderators.

    Fits the mixed-effects model ``y_i = x_i' beta + u_i + e_i`` with
    ``e_i ~ N(0, se_i^2)`` and ``u_i ~ N(0, tau2)`` by weighted least
    squares at an estimate of the residual between-study variance. R
    ``metafor::rma(yi, sei = , mods = )``, Stata ``meta regress``.

    Parameters
    ----------
    effects : sequence of float
        Per-study effect sizes (log odds ratios, mean differences, ...).
    se : sequence of float
        Per-study standard errors (positive).
    moderators : array-like, mapping or pandas.DataFrame
        Study-level covariates, one row per study: a 1-D array for a single
        moderator, a 2-D array, a ``{name: values}`` mapping or a DataFrame.
        All columns must be numeric; code categories as dummies.
    method : {"REML", "DL", "ML", "PM", "fixed"}, default "REML"
        Estimator of the residual between-study variance: restricted
        maximum likelihood (the default of ``metafor`` and Stata),
        DerSimonian-Laird, maximum likelihood, Paule-Mandel, or none
        (``"fixed"``, a fixed-effects meta-regression).
    hksj : bool, default False
        Knapp-Hartung adjustment (``metafor`` ``test = "knha"``): the
        covariance matrix is multiplied by the weighted residual mean
        square, coefficients are tested with ``t(k - p)`` and the
        moderators jointly with an F statistic.
    intercept : bool, default True
        Include an intercept.
    names : sequence of str, optional
        Moderator names when ``moderators`` is an array.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    MetaRegressionResult
        Coefficient table (``.tidy()``), ``tau2``, ``i2``, ``r2``, the
        residual-heterogeneity test ``qe`` and the omnibus moderator test
        ``qm``, ``.predict()`` and ``.summary()``.

    Notes
    -----
    ``i2`` and ``h2`` are ``metafor``'s: with the typical within-study
    variance ``v = (k - p) / tr(P)`` at ``tau2 = 0``, ``i2 = tau2 / (tau2 +
    v)`` as a fraction (``metafor`` prints it times 100) and ``h2 = 1 +
    tau2 / v``. ``r2`` compares ``tau2`` with that of the intercept-only
    model under the same estimator. With ``method="fixed"`` they are the
    Q-based ones, ``(QE - df) / QE``, ``QE / df`` and one minus the ratio
    of ``QE / df`` to its intercept-only value. The iterative estimators
    are run to machine precision; ``metafor`` and Stata stop earlier by
    default (relative 1e-5 to 1e-6), so compare with their tolerances
    tightened.

    Without an intercept the residual heterogeneity statistic is the
    weighted residual sum of squares of the regression through the origin,
    as in Stata ``meta regress, noconstant``. ``metafor`` 5.0.1 centres the
    effects at their unweighted mean before computing ``QE`` and ``tau2``,
    which changes both when the model has no intercept; its coefficients
    at a given ``tau2`` are the same.

    Examples
    --------
    >>> import statspai as sp
    >>> y = [-0.89, -1.59, -1.35, -1.44, -0.22, -0.79, -1.62, 0.01, -0.47]
    >>> s = [0.57, 0.44, 0.64, 0.14, 0.23, 0.08, 0.47, 0.06, 0.24]
    >>> lat = [44, 55, 42, 52, 13, 44, 19, 13, 27]
    >>> r = sp.meta_regression(y, s, {"latitude": lat}, method="DL")
    >>> round(float(r.params["latitude"]), 4)
    -0.0286
    >>> bool(r.qm_pvalue < 0.05)
    True
    """
    who = "meta_regression"
    _check_alpha(alpha, who)
    y, s = _study_inputs(effects, se, who)
    if method not in _core.TAU2_METHODS:
        raise MethodIncompatibility(
            f"{who}: method must be one of {list(_core.TAU2_METHODS)}, got "
            f"{method!r}.",
            recovery_hint="Use 'REML', 'DL', 'ML', 'PM' or 'fixed'.",
        )
    if hksj and method == "fixed":
        raise MethodIncompatibility(
            f"{who}: the Knapp-Hartung adjustment (hksj=True) belongs to the "
            "random-effects model.",
            recovery_hint="Use method='REML', 'DL', 'ML' or 'PM' with hksj=True.",
        )
    k = len(y)
    Z, mod_names = _moderator_matrix(moderators, k, names=names, who=who)
    if names is not None:
        if len(names) != Z.shape[1]:
            raise MethodIncompatibility(
                f"{who}: {len(names)} names for {Z.shape[1]} moderators.",
                recovery_hint="Pass one name per moderator column.",
            )
        mod_names = [str(n) for n in names]
    m = Z.shape[1]
    if m == 0:
        raise MethodIncompatibility(
            f"{who}: no moderators were passed.",
            recovery_hint="Use sp.meta_analysis for the model without moderators.",
        )
    X = np.column_stack([np.ones(k), Z]) if intercept else Z
    col_names = (["intercept"] if intercept else []) + mod_names
    p = X.shape[1]
    scale = np.sqrt((X**2).sum(axis=0))
    scale[scale == 0] = 1.0
    if np.linalg.matrix_rank(X / scale) < p:
        raise MethodIncompatibility(
            f"{who}: the moderators {mod_names!r} are collinear"
            + (" with each other or with the intercept." if intercept else "."),
            recovery_hint=(
                "Drop a redundant moderator (a constant column, or the last "
                "level of a set of dummies)."
            ),
        )
    if k <= p:
        raise DataInsufficient(
            f"{who}: {k} studies cannot identify {p} coefficients and a "
            "residual variance.",
            recovery_hint="Use fewer moderators or more studies (k > p).",
        )
    v = s**2
    tau2 = _core.estimate_tau2(y, v, X, method)
    beta, cov = _core.wls(y, v, X, tau2)
    df_resid = k - p
    P0 = _core.projection(v, X, 0.0)
    qe = float(y @ P0 @ y)
    qe_p = float(stats.chi2.sf(qe, df_resid))
    if hksj:
        s2w = float(y @ _core.projection(v, X, tau2) @ y) / df_resid
        cov = cov * s2w
    se_b = np.sqrt(np.diag(cov))
    stat = beta / se_b
    if hksj:
        pvals = 2 * stats.t.sf(np.abs(stat), df=df_resid)
        crit = float(stats.t.ppf(1 - alpha / 2, df=df_resid))
    else:
        pvals = 2 * stats.norm.sf(np.abs(stat))
        crit = float(stats.norm.ppf(1 - alpha / 2))
    # Omnibus test of the moderators (every coefficient when there is no
    # intercept).
    idx = np.arange(1, p) if intercept else np.arange(p)
    b_m = beta[idx]
    wald = float(b_m @ np.linalg.solve(cov[np.ix_(idx, idx)], b_m))
    qm_df: Any
    if hksj:
        qm = wald / len(idx)
        qm_df = (int(len(idx)), int(df_resid))
        qm_p = float(stats.f.sf(qm, len(idx), df_resid))
    else:
        qm = wald
        qm_df = int(len(idx))
        qm_p = float(stats.chi2.sf(qm, len(idx)))
    r2 = float("nan")
    tau2_null = float("nan")
    if method == "fixed":
        # No tau2 to work with: the Q-based versions (Higgins and Thompson).
        i2 = max(0.0, (qe - df_resid) / qe) if qe > 0 else 0.0
        h2 = qe / df_resid
        if intercept:
            w0 = 1.0 / v
            q0 = float(np.sum(w0 * (y - np.sum(w0 * y) / np.sum(w0)) ** 2))
            if q0 > 0:
                r2 = max(0.0, 1.0 - h2 / (q0 / (k - 1)))
    else:
        v_typ = _core.typical_variance(v, X)
        i2 = tau2 / (tau2 + v_typ)
        h2 = 1.0 + tau2 / v_typ
        if intercept:
            tau2_null = _core.estimate_tau2(y, v, np.ones((k, 1)), method)
            if tau2_null > 0:
                r2 = max(0.0, (tau2_null - tau2) / tau2_null)
    fitted = X @ beta
    return MetaRegressionResult(
        params=pd.Series(beta, index=col_names),
        std_errors=pd.Series(se_b, index=col_names),
        statistics=pd.Series(stat, index=col_names),
        p_values=pd.Series(pvals, index=col_names),
        conf_int=pd.DataFrame(
            {"lower": beta - crit * se_b, "upper": beta + crit * se_b},
            index=col_names,
        ),
        vcov=pd.DataFrame(cov, index=col_names, columns=col_names),
        tau2=float(tau2),
        i2=float(i2),
        h2=float(h2),
        r2=r2,
        tau2_null=tau2_null,
        qe=qe,
        qe_df=int(df_resid),
        qe_pvalue=qe_p,
        qm=float(qm),
        qm_df=qm_df,
        qm_pvalue=qm_p,
        method=method,
        hksj=bool(hksj),
        k=int(k),
        alpha=float(alpha),
        fitted=fitted,
        residuals=y - fitted,
        intercept=bool(intercept),
    )


# ---------------------------------------------------------------------------
# Trim and fill
# ---------------------------------------------------------------------------


@dataclass
class TrimFillResult(ResultProtocolMixin):
    """Result of :func:`trim_and_fill`.

    Attributes
    ----------
    k0 : int
        Estimated number of studies missing from ``side``.
    side : str
        ``"left"`` or ``"right"``: the side of the funnel plot the imputed
        studies are added to.
    estimate, se, ci, p_value : float, float, tuple, float
        Pooled effect after imputing the ``k0`` studies.
    tau2 : float
        Between-study variance of the augmented set.
    original : MetaAnalysisResult
        Fit to the observed studies.
    filled : MetaAnalysisResult
        Fit to the observed and imputed studies.
    filled_effects, filled_se : numpy.ndarray
        The imputed studies.
    estimator : str
        ``"L0"``, ``"R0"`` or ``"Q0"``.
    k0_raw : float
        The estimator's value before rounding, at the last iteration.
    n_iter : int
        Iterations until ``k0`` stopped changing.
    """

    k0: int
    side: str
    estimate: float
    se: float
    ci: tuple
    p_value: float
    tau2: float
    original: MetaAnalysisResult
    filled: MetaAnalysisResult
    filled_effects: np.ndarray
    filled_se: np.ndarray
    estimator: str = "L0"
    k0_raw: float = float("nan")
    n_iter: int = 0
    alpha: float = 0.05

    def summary(self) -> str:
        """Plain-text report."""
        o, f = self.original, self.filled
        out = ["=" * 72, "Trim and fill (Duval-Tweedie)", "=" * 72]
        out.append(f"Observed studies     : {len(o.effects)}")
        out.append(
            f"Imputed studies (k0) : {self.k0} on the {self.side} side "
            f"(estimator {self.estimator})"
        )
        out.append("-" * 72)
        out.append(
            f"Observed pooled      : {o.estimate:.4f}  "
            f"[{o.ci[0]:.4f}, {o.ci[1]:.4f}]"
        )
        out.append(
            f"Filled pooled        : {f.estimate:.4f}  "
            f"[{f.ci[0]:.4f}, {f.ci[1]:.4f}]   p = {f.p_value:.4g}"
        )
        out.append(f"tau^2                : {o.tau2:.4f} -> {f.tau2:.4f}")
        out.append("=" * 72)
        out.append(
            "The imputation assumes the asymmetry comes from suppressed "
            "studies; read the\nfilled estimate as a sensitivity analysis, "
            "not as a corrected effect."
        )
        return "\n".join(out)

    def __repr__(self) -> str:
        return (
            f"<TrimFillResult: k0={self.k0} ({self.side}), "
            f"filled={self.estimate:.4f}, observed={self.original.estimate:.4f}>"
        )

    def funnel_plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Funnel plot with the imputed studies as hollow markers."""
        ax = self.original.funnel_plot(ax=ax, **kwargs)
        if self.k0:
            ax.scatter(
                self.filled_effects,
                self.filled_se,
                facecolors="none",
                edgecolors="#333",
                s=25,
            )
        ax.axvline(self.estimate, color="#c1272d", lw=1.0, linestyle="--")
        return ax


def _egger_slope_sign(y: np.ndarray, s: np.ndarray, method: str) -> float:
    """Slope of the effect on its standard error in a meta-regression."""
    fit = meta_regression(y, s, s, method=method)
    return float(fit.params.iloc[1])


def trim_and_fill(
    effects: Union[Sequence[float], MetaAnalysisResult],
    se: Optional[Sequence[float]] = None,
    *,
    side: Optional[str] = None,
    estimator: str = "L0",
    method: str = "DL",
    hksj: bool = False,
    labels: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    maxiter: int = 100,
) -> TrimFillResult:
    """Duval-Tweedie trim-and-fill adjustment for funnel-plot asymmetry.

    Estimates the number of studies missing from one side of the funnel
    plot from the ranks of the centred effects, imputes them as mirror
    images of the most extreme studies on the other side, and re-estimates
    the pooled effect. R ``metafor::trimfill``.

    Parameters
    ----------
    effects : sequence of float or MetaAnalysisResult
        Per-study effect sizes, or a fitted ``sp.meta_analysis`` result
        (then ``se``, ``method``, ``hksj``, ``labels`` and ``alpha`` are
        taken from it).
    se : sequence of float, optional
        Per-study standard errors. Required with raw effects.
    side : {"left", "right"}, optional
        Side on which studies are assumed missing. Default: chosen from the
        sign of the slope of the effect on its standard error in a
        meta-regression (Egger-type test); a positive slope means small
        studies report larger effects, so the missing ones are on the left.
    estimator : {"L0", "R0", "Q0"}, default "L0"
        Rank-based estimator of the number of missing studies.
    method : {"DL", "REML", "ML", "PM", "fixed"}, default "DL"
        Model for the pooled effect at every step, as in
        ``sp.meta_analysis``.
    hksj : bool, default False
        Knapp-Hartung inference for the reported pooled effects.
    labels : sequence of str, optional
        Study labels.
    alpha : float, default 0.05
        One minus the confidence level.
    maxiter : int, default 100
        Maximum number of trim iterations.

    Returns
    -------
    TrimFillResult
        ``k0``, ``side``, the filled pooled ``estimate`` with ``se``, ``ci``
        and ``p_value``, the ``original`` and ``filled`` fits and the
        imputed studies.

    Notes
    -----
    With the studies ordered by effect and the ``k0`` most extreme ones on
    the long side trimmed, the model is refitted, all effects are centred
    at its estimate, and with ``T`` the sum of the ranks of the absolute
    centred effects that are positive,
    ``L0 = (4 T - k (k + 1)) / (2 k - 1)``,
    ``Q0 = k - 1/2 - sqrt(2 k^2 - 4 T + 1/4)``, and ``R0`` is the length
    of the run of largest ranks that are positive, minus one. The value is
    rounded (halves to even, as R does) and floored at zero, and the step
    repeats until it stops changing.

    The method assumes that asymmetry is caused by the suppression of the
    most extreme studies on one side. Heterogeneity and other small-study
    effects produce asymmetry too, so the filled estimate is a sensitivity
    analysis.

    Examples
    --------
    >>> import statspai as sp
    >>> y = [0.10, 0.22, 0.31, 0.38, 0.45, 0.62, 0.75, 0.95, 1.20]
    >>> s = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.24, 0.30, 0.38]
    >>> tf = sp.trim_and_fill(y, s)
    >>> tf.side, tf.k0
    ('left', 4)
    >>> bool(tf.estimate < tf.original.estimate)
    True
    """
    who = "trim_and_fill"
    if isinstance(effects, MetaAnalysisResult):
        if se is not None:
            raise MethodIncompatibility(
                f"{who}: se= is taken from the meta-analysis result; do not "
                "pass it again.",
                recovery_hint="sp.trim_and_fill(result) or "
                "sp.trim_and_fill(effects, se).",
            )
        res0 = effects
        effects, se = res0.effects, res0.se_studies
        method, hksj, alpha = res0.method, res0.hksj, res0.alpha
        labels = list(res0.labels)
    if se is None:
        raise MethodIncompatibility(
            f"{who}: pass the standard errors of the studies as se=.",
            recovery_hint="sp.trim_and_fill(effects, se)",
        )
    _check_alpha(alpha, who)
    y, s = _study_inputs(effects, se, who)
    if estimator not in ("L0", "R0", "Q0"):
        raise MethodIncompatibility(
            f"{who}: estimator must be 'L0', 'R0' or 'Q0', got {estimator!r}.",
            recovery_hint="Use estimator='L0' (the default) or 'R0'.",
        )
    if side not in (None, "left", "right"):
        raise MethodIncompatibility(
            f"{who}: side must be 'left', 'right' or None, got {side!r}.",
            recovery_hint="Leave side=None to let the asymmetry choose it.",
        )
    if method not in _core.TAU2_METHODS:
        raise MethodIncompatibility(
            f"{who}: method must be one of {list(_core.TAU2_METHODS)}, got "
            f"{method!r}.",
            recovery_hint="Use 'DL', 'REML', 'ML', 'PM' or 'fixed'.",
        )
    k = len(y)
    if k < 3:
        raise DataInsufficient(
            f"{who}: needs at least 3 studies, got {k}.",
            recovery_hint="Trim and fill is not informative with so few studies.",
        )
    if labels is None:
        labels = [f"Study {i + 1}" for i in range(k)]
    labels = list(labels)
    if len(labels) != k:
        raise MethodIncompatibility(
            f"{who}: {len(labels)} labels for {k} studies.",
            recovery_hint="Pass one label per study, or labels=None.",
        )
    original = meta_analysis(y, s, method=method, labels=labels, alpha=alpha, hksj=hksj)
    if side is None:
        if np.ptp(s) <= 1e-12 * float(np.max(s)):
            raise MethodIncompatibility(
                f"{who}: all standard errors are equal, so the asymmetry "
                "regression cannot choose a side.",
                recovery_hint="Pass side='left' or side='right'.",
            )
        side = "left" if _egger_slope_sign(y, s, method) > 0 else "right"
    # Work with the missing studies on the left: flip the sign otherwise.
    sign = 1.0 if side == "left" else -1.0
    order = np.argsort(sign * y, kind="stable")
    yo, so = (sign * y)[order], s[order]

    def pooled(yy: np.ndarray, ss: np.ndarray) -> float:
        if len(yy) < 2:
            return float(yy[0])
        return float(meta_analysis(yy, ss, method=method).estimate)

    k0 = 0
    k0_raw = 0.0
    n_iter = 0
    for n_iter in range(1, int(maxiter) + 1):
        centre = pooled(yo[: k - k0], so[: k - k0])
        yc = yo - centre
        ranks = stats.rankdata(np.abs(yc), method="ordinal")
        signed = np.sign(yc) * ranks
        if estimator == "R0":
            neg = signed[signed < 0]
            k0_raw = float(k - (np.max(-neg) if neg.size else 0.0) - 1)
        else:
            t_n = float(np.sum(signed[signed > 0]))
            if estimator == "L0":
                k0_raw = (4.0 * t_n - k * (k + 1)) / (2.0 * k - 1.0)
            else:
                inside = 2.0 * k**2 - 4.0 * t_n + 0.25
                if inside < 0:
                    raise MethodIncompatibility(
                        f"{who}: the Q0 estimator is undefined here (every "
                        "centred effect lies on one side of the pooled "
                        f"estimate with side={side!r}).",
                        recovery_hint=(
                            "Use estimator='L0' or 'R0', or the other side."
                        ),
                    )
                k0_raw = k - 0.5 - float(np.sqrt(inside))
        new = int(max(0.0, np.round(k0_raw)))
        new = min(new, k - 2)  # keep at least two studies to centre on
        if new == k0:
            break
        k0 = new
    centre = pooled(yo[: k - k0], so[: k - k0])
    fill_y = sign * (2.0 * centre - yo[k - k0 :]) if k0 else np.empty(0)
    fill_s = so[k - k0 :] if k0 else np.empty(0)
    y_all = np.concatenate([y, fill_y])
    s_all = np.concatenate([s, fill_s])
    lab_all = labels + [f"Filled {i + 1}" for i in range(k0)]
    filled = meta_analysis(
        y_all, s_all, method=method, labels=lab_all, alpha=alpha, hksj=hksj
    )
    return TrimFillResult(
        k0=int(k0),
        side=str(side),
        estimate=float(filled.estimate),
        se=float(filled.se),
        ci=tuple(filled.ci),
        p_value=float(filled.p_value),
        tau2=float(filled.tau2),
        original=original,
        filled=filled,
        filled_effects=fill_y,
        filled_se=fill_s,
        estimator=estimator,
        k0_raw=float(k0_raw),
        n_iter=int(n_iter),
        alpha=float(alpha),
    )
