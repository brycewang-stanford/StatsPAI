"""
Survival and duration analysis models for StatsPAI.

Implements:
- Cox Proportional Hazards (partial likelihood, Efron/Breslow ties)
- Kaplan-Meier estimator (non-parametric survival function)
- Log-Rank test (group comparison)
- Parametric survival / AFT models (Weibull, exponential, log-normal, log-logistic)
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..core.results import EconometricResults

# ---------------------------------------------------------------------------
# Formula parser (local)
# ---------------------------------------------------------------------------


def _parse_formula(formula: str) -> Tuple[str, List[str]]:
    """Parse 'y ~ x1 + x2' into (y_name, [x1, x2])."""
    parts = formula.split("~")
    if len(parts) != 2:
        raise ValueError(f"Invalid formula: {formula}")
    y = parts[0].strip()
    x = [v.strip() for v in parts[1].split("+") if v.strip()]
    return y, x


# ---------------------------------------------------------------------------
# Robust / clustered variance helpers
# ---------------------------------------------------------------------------


def _sandwich_variance(X: Any, hessian_inv: Any, score_i: Any) -> Any:
    """HC0 sandwich variance: H^{-1} (sum s_i s_i') H^{-1}.

    Delegates to the canonical ``core._vcov.sandwich_vcov`` (CLAUDE.md §4).
    """
    from ..core._vcov import sandwich_vcov

    return sandwich_vcov(hessian_inv, score_i, correction="none")


def _cluster_variance(
    X: Any,
    hessian_inv: Any,
    score_i: Any,
    clusters: Any,
) -> Any:
    """Clustered sandwich variance (correction G/(G-1) = 'cgm').

    Byte-identical to the prior hand-rolled sandwich for G >= 2.
    """
    from ..core._vcov import sandwich_vcov

    return sandwich_vcov(hessian_inv, score_i, clusters=clusters, correction="cgm")


# ===================================================================
# CoxResult — extends EconometricResults
# ===================================================================


class CoxResult(EconometricResults):
    """
    Result from Cox proportional hazards estimation.

    Extends ``EconometricResults`` with survival-specific methods:
    ``.plot()``, ``.ph_test()``, ``.baseline_hazard()``, ``.concordance``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> x = rng.normal(size=n)
    >>> time = rng.exponential(scale=np.exp(-0.5 * x))
    >>> event = (rng.random(n) < 0.8).astype(int)
    >>> df = pd.DataFrame({"time": time, "status": event, "x": x})
    >>> res = sp.cox(data=df, duration="time", event="status", x=["x"])
    >>> type(res).__name__
    'CoxResult'
    >>> list(res.params.index)
    ['x']
    >>> bool(isinstance(float(res.concordance), float))
    True
    >>> list(res.baseline_hazard().columns)
    ['time', 'baseline_cumhaz', 'baseline_survival']
    >>> isinstance(res.summary(), str)
    True
    """

    def __init__(
        self,
        params: Any,
        std_errors: Any,
        model_info: Any,
        data_info: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        # Cox-specific internals
        _X: Any = None,
        _durations: Any = None,
        _events: Any = None,
        _baseline_hazard_df: Optional[pd.DataFrame] = None,
        _concordance: Optional[float] = None,
        _schoenfeld_resid: Any = None,
        _strata: Any = None,
        _hazard_ratios: Any = None,
        _hr_ci: Any = None,
    ) -> None:
        super().__init__(params, std_errors, model_info, data_info, diagnostics)
        self._X = _X
        self._durations = _durations
        self._events = _events
        self._baseline_hazard_df = _baseline_hazard_df
        self._concordance_val = _concordance
        self._schoenfeld_resid = _schoenfeld_resid
        self._strata = _strata
        self._hazard_ratios = _hazard_ratios
        self._hr_ci = _hr_ci

    # -- concordance property -----------------------------------------------
    @property
    def concordance(self) -> float:
        """Harrell's C-statistic (concordance index)."""
        if self._concordance_val is None:
            raise RuntimeError("Concordance statistic is not available.")
        return float(self._concordance_val)

    # -- baseline hazard ----------------------------------------------------
    def baseline_hazard(self) -> pd.DataFrame:
        """
        Baseline cumulative hazard (Breslow estimator).

        Returns
        -------
        pd.DataFrame
            Columns ``time``, ``baseline_cumhaz``, ``baseline_survival``.
        """
        if self._baseline_hazard_df is None:
            raise RuntimeError("Baseline hazard is not available.")
        return self._baseline_hazard_df.copy()

    # -- PH test (Schoenfeld residuals) ------------------------------------
    def ph_test(self) -> pd.DataFrame:
        """
        Test the proportional hazards assumption via Schoenfeld residuals.

        Computes the correlation of scaled Schoenfeld residuals with time
        for each covariate and reports a chi-squared test.

        Returns
        -------
        pd.DataFrame
            Columns: ``variable``, ``rho``, ``chi2``, ``p_value``.
        """
        if self._schoenfeld_resid is None:
            raise RuntimeError("Schoenfeld residuals not stored (strata model).")

        resid = self._schoenfeld_resid  # (n_events, p)
        times = self._durations[self._events == 1]
        # sort by time
        order = np.argsort(times)
        times = times[order]
        resid = resid[order]

        n = resid.shape[0]
        names = list(self.params.index)
        rows = []
        for j, var in enumerate(names):
            col = resid[:, j]
            if np.std(col) < 1e-15 or n < 3:
                rho, chi2, pv = 0.0, 0.0, 1.0
            else:
                rho, _ = stats.spearmanr(times, col)
                if np.isnan(rho):
                    rho, chi2, pv = 0.0, 0.0, 1.0
                else:
                    chi2 = n * rho**2
                    pv = stats.chi2.sf(chi2, df=1)
            rows.append({"variable": var, "rho": rho, "chi2": chi2, "p_value": pv})
        return pd.DataFrame(rows)

    # -- plot ---------------------------------------------------------------
    def plot(self, kind: str = "survival", ax: Any = None, **kwargs: Any) -> Any:
        """
        Plot survival-related curves.

        Parameters
        ----------
        kind : str
            ``'survival'`` — baseline survival curve,
            ``'hazard'`` — baseline cumulative hazard,
            ``'hr'`` — hazard ratio forest plot.
        ax : matplotlib.Axes, optional
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 4)))

        bh = self._baseline_hazard_df
        if bh is None:
            raise RuntimeError("Baseline hazard is not available for plotting.")

        if kind == "survival":
            ax.step(bh["time"], bh["baseline_survival"], where="post", **kwargs)
            ax.set_xlabel("Time")
            ax.set_ylabel("Baseline survival probability")
            ax.set_title("Cox baseline survival curve")
            ax.set_ylim(0, 1.05)
        elif kind == "hazard":
            ax.step(bh["time"], bh["baseline_cumhaz"], where="post", **kwargs)
            ax.set_xlabel("Time")
            ax.set_ylabel("Cumulative hazard")
            ax.set_title("Cox baseline cumulative hazard")
        elif kind == "hr":
            hr = self._hazard_ratios
            if hr is None or self._hr_ci is None:
                raise RuntimeError("Hazard-ratio intervals are not available.")
            lo, hi = self._hr_ci[:, 0], self._hr_ci[:, 1]
            names = list(self.params.index)
            y = np.arange(len(names))
            ax.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", capsize=4)
            ax.axvline(1, ls="--", color="grey", lw=0.8)
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.set_xlabel("Hazard Ratio")
            ax.set_title("Hazard Ratios (95% CI)")
        else:
            raise ValueError(
                f"Unknown kind={kind!r}. Use 'survival', 'hazard', or 'hr'."
            )

        plt.tight_layout()
        return ax

    # -- repr ---------------------------------------------------------------
    def __repr__(self) -> str:
        n_obs = self.data_info.get("nobs", "?")
        n_events = self.data_info.get("n_events", "?")
        return (
            f"<CoxResult: {len(self.params)} covariates, "
            f"{n_obs} obs, {n_events} events, C={self._concordance_val:.4f}>"
        )


# ===================================================================
# KMResult — Kaplan-Meier result
# ===================================================================


class KMResult(ResultProtocolMixin):
    """
    Kaplan-Meier survival analysis result.

    Attributes
    ----------
    survival_table : pd.DataFrame
        Life table with ``time``, ``n_risk``, ``n_event``, ``n_censor``,
        ``survival``, ``std_err``, ``ci_lower``, ``ci_upper``.
    median_survival : float or dict
        Median survival time (per group if groups present).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> df = pd.DataFrame({
    ...     "time": rng.exponential(10, n).round(2),
    ...     "status": (rng.random(n) < 0.7).astype(int),
    ... })
    >>> km = sp.kaplan_meier(data=df, duration="time", event="status")
    >>> type(km).__name__
    'KMResult'
    >>> "survival" in km.survival_table.columns
    True
    >>> bool(0.0 <= km.survival_table["survival"].iloc[-1] <= 1.0)
    True
    >>> isinstance(km.summary(), str)
    True
    """

    def __init__(self, tables: Dict[str, pd.DataFrame], alpha: float = 0.05):
        self._tables = tables  # group_name -> DataFrame
        self._alpha = alpha

    @property
    def survival_table(self) -> pd.DataFrame:
        """Life table (combined if multiple groups)."""
        if len(self._tables) == 1:
            return list(self._tables.values())[0]
        parts = []
        for g, df in self._tables.items():
            tmp = df.copy()
            tmp.insert(0, "group", g)
            parts.append(tmp)
        return pd.concat(parts, ignore_index=True)

    @property
    def median_survival(self) -> Union[float, Dict[str, float]]:
        """Median survival time (scalar or dict by group)."""
        result = {}
        for g, df in self._tables.items():
            below = df.loc[df["survival"] <= 0.5]
            result[g] = float(below["time"].iloc[0]) if len(below) > 0 else np.nan
        if len(result) == 1:
            return list(result.values())[0]
        return result

    # -- summary -----------------------------------------------------------
    def summary(self) -> str:
        """Formatted summary of the Kaplan-Meier analysis."""
        out = []
        out.append("=" * 72)
        out.append("Kaplan-Meier Survival Estimates")
        out.append("=" * 72)
        for g, df in self._tables.items():
            n = int(df["n_risk"].iloc[0]) if len(df) > 0 else 0
            n_events = int(df["n_event"].sum())
            med_val = self.median_survival
            if isinstance(med_val, dict):
                med = med_val.get(g, np.nan)
            else:
                med = med_val
            label = f"Group: {g}" if len(self._tables) > 1 else "Overall"
            out.append(f"\n{label}")
            out.append(f"  N at risk (start): {n}")
            out.append(f"  Number of events : {n_events}")
            out.append(
                f"  Median survival  : {med:.4f}"
                if not np.isnan(med)
                else "  Median survival  : not reached"
            )
        out.append("\n" + "=" * 72)
        return "\n".join(out)

    # -- plot ---------------------------------------------------------------
    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """
        Plot Kaplan-Meier survival curves with confidence bands.

        Parameters
        ----------
        ax : matplotlib.Axes, optional
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 4)))

        for g, df in self._tables.items():
            label = g if len(self._tables) > 1 else "KM estimate"
            ax.step(df["time"], df["survival"], where="post", label=label)
            ax.fill_between(
                df["time"],
                df["ci_lower"],
                df["ci_upper"],
                step="post",
                alpha=0.15,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")
        ax.set_title("Kaplan-Meier survival curve")
        ax.set_ylim(0, 1.05)
        if len(self._tables) > 1:
            ax.legend()
        plt.tight_layout()
        return ax

    def __repr__(self) -> str:
        groups = list(self._tables.keys())
        if len(groups) == 1:
            n_ev = int(list(self._tables.values())[0]["n_event"].sum())
            return f"<KMResult: {n_ev} events>"
        return f"<KMResult: {len(groups)} groups>"


# ===================================================================
# Internal: Kaplan-Meier computation
# ===================================================================


def _km_table(
    durations: Any, events: Any, alpha: float = 0.05, conf_type: str = "plain"
) -> pd.DataFrame:
    """Build a KM life table from raw duration/event arrays."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=float)
    n_total = len(durations)

    unique_times = np.sort(np.unique(durations[events == 1]))
    rows = []
    survival = 1.0
    var_sum = 0.0
    z = stats.norm.ppf(1 - alpha / 2)

    for t in unique_times:
        n_risk = np.sum(durations >= t)
        n_event = np.sum((durations == t) & (events == 1))
        n_censor = np.sum((durations == t) & (events == 0))

        if n_risk > 0:
            survival *= 1 - n_event / n_risk
            # Greenwood variance
            if n_risk > n_event:
                var_sum += n_event / (n_risk * (n_risk - n_event))

        se = survival * np.sqrt(var_sum) if var_sum >= 0 else 0.0
        if conf_type == "plain" or var_sum <= 0 or not 0.0 < survival < 1.0:
            ci_lo = max(0.0, survival - z * se)
            ci_hi = min(1.0, survival + z * se)
            if conf_type != "plain" and survival <= 0.0:
                ci_lo = ci_hi = np.nan  # undefined on the log scales
        elif conf_type == "log":
            # symmetric for log S(t): R survival::survfit's default
            half = z * np.sqrt(var_sum)
            ci_lo = survival * np.exp(-half)
            ci_hi = min(1.0, survival * np.exp(half))
        else:
            # symmetric for log(-log S(t)): Stata sts list's default
            half = z * np.sqrt(var_sum) / abs(np.log(survival))
            ci_lo = survival ** np.exp(half)
            ci_hi = survival ** np.exp(-half)

        rows.append(
            {
                "time": t,
                "n_risk": int(n_risk),
                "n_event": int(n_event),
                "n_censor": int(n_censor),
                "survival": survival,
                "std_err": se,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            }
        )

    # Prepend time=0 row
    t0 = {
        "time": 0.0,
        "n_risk": n_total,
        "n_event": 0,
        "n_censor": 0,
        "survival": 1.0,
        "std_err": 0.0,
        "ci_lower": 1.0,
        "ci_upper": 1.0,
    }
    return pd.DataFrame([t0] + rows)


# ===================================================================
# kaplan_meier()
# ===================================================================


def kaplan_meier(
    data: pd.DataFrame,
    duration: str,
    event: str,
    group: Optional[str] = None,
    alpha: float = 0.05,
    conf_type: str = "plain",
) -> KMResult:
    """
    Kaplan-Meier non-parametric survival function estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    duration : str
        Column name for duration / follow-up time.
    event : str
        Column name for event indicator (1 = event, 0 = censored).
    group : str, optional
        Column name for group variable (stratification).
    alpha : float
        Significance level for confidence intervals (Greenwood formula).
    conf_type : {"plain", "log", "log-log"}, default "plain"
        Scale on which the interval is symmetric. ``"plain"`` is ``S(t)
        +/- z se`` truncated to [0, 1]; ``"log"`` is symmetric for ``log
        S(t)`` (the default of R ``survival::survfit``); ``"log-log"`` for
        ``log(-log S(t))`` (the default of Stata ``sts list``), which
        always stays inside (0, 1) and has the best coverage in the
        tails. The estimate and its standard error are the same under all
        three.

    Returns
    -------
    KMResult
        Object with ``.survival_table``, ``.median_survival``, ``.plot()``,
        ``.summary()``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 150
    >>> df = pd.DataFrame({
    ...     "time": rng.exponential(10, n).round(2),
    ...     "status": (rng.random(n) < 0.7).astype(int),
    ... })
    >>> km = sp.kaplan_meier(data=df, duration="time", event="status")
    >>> bool("survival" in km.survival_table.columns)
    True
    >>> bool(isinstance(km.median_survival, float))
    True
    """
    conf_type = str(conf_type).lower().replace("loglog", "log-log")
    if conf_type not in ("plain", "log", "log-log"):
        raise ValueError(
            f"conf_type must be 'plain', 'log' or 'log-log', got {conf_type!r}"
        )
    data = data.dropna(subset=[duration, event])

    if group is None:
        tables = {
            "all": _km_table(
                data[duration].values, data[event].values, alpha, conf_type
            )
        }
    else:
        tables = {}
        for g_val, gdf in data.groupby(group):
            tables[str(g_val)] = _km_table(
                gdf[duration].values, gdf[event].values, alpha, conf_type
            )

    _result = KMResult(tables, alpha=alpha)
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.survival.kaplan_meier",
            params={
                "duration": duration,
                "event": event,
                "group": group,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ===================================================================
# logrank_test()
# ===================================================================


def logrank_test(
    data: pd.DataFrame,
    duration: str,
    event: str,
    group: str,
) -> dict:
    """
    Log-rank test for equality of survival distributions across groups.

    Parameters
    ----------
    data : pd.DataFrame
    duration, event, group : str
        Column names.

    Returns
    -------
    dict
        Keys: ``test_statistic``, ``p_value``, ``df``, ``n_groups``,
        ``expected_events`` (per group), ``observed_events`` (per group).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 150
    >>> treatment = rng.integers(0, 2, n)
    >>> time = rng.exponential(scale=np.exp(-0.5 * treatment))
    >>> status = (rng.random(n) < 0.8).astype(int)
    >>> df = pd.DataFrame({"time": time, "status": status,
    ...                    "treatment": treatment})
    >>> res = sp.logrank_test(data=df, duration="time", event="status",
    ...                       group="treatment")
    >>> bool("p_value" in res and "test_statistic" in res)
    True

    References
    ----------
    mantel1959statistical
    """
    data = data.dropna(subset=[duration, event, group])
    groups = data[group].unique()
    K = len(groups)
    if K < 2:
        raise ValueError("Need at least 2 groups for the log-rank test.")

    T = data[duration].values
    E = data[event].values.astype(float)
    G = data[group].values

    event_times = np.sort(np.unique(T[E == 1]))

    observed = {g: 0.0 for g in groups}
    expected = {g: 0.0 for g in groups}
    var_mat = np.zeros((K - 1, K - 1))

    for t in event_times:
        at_risk_total = np.sum(T >= t)
        events_total = np.sum((T == t) & (E == 1))

        if at_risk_total == 0:
            continue

        for g in groups:
            mask_g = G == g
            n_g = np.sum(mask_g & (T >= t))
            d_g = np.sum(mask_g & (T == t) & (E == 1))
            observed[g] += d_g
            e_g = n_g * events_total / at_risk_total
            expected[g] += e_g

        # Variance (hypergeometric)
        if at_risk_total > 1:
            for i in range(K - 1):
                gi = groups[i]
                ni = np.sum((G == gi) & (T >= t))
                for j in range(i, K - 1):
                    gj = groups[j]
                    nj = np.sum((G == gj) & (T >= t))
                    if i == j:
                        v = (
                            ni
                            * (at_risk_total - ni)
                            * events_total
                            * (at_risk_total - events_total)
                        ) / (at_risk_total**2 * (at_risk_total - 1))
                    else:
                        v = -(
                            ni * nj * events_total * (at_risk_total - events_total)
                        ) / (at_risk_total**2 * (at_risk_total - 1))
                    var_mat[i, j] += v
                    if i != j:
                        var_mat[j, i] += v

    # Test statistic
    O_E = np.array([observed[groups[i]] - expected[groups[i]] for i in range(K - 1)])
    try:
        chi2 = float(O_E @ np.linalg.solve(var_mat, O_E))
    except np.linalg.LinAlgError:
        chi2 = float(O_E @ np.linalg.lstsq(var_mat, O_E, rcond=None)[0])

    df = K - 1
    p_value = stats.chi2.sf(chi2, df=df)

    return {
        "test_statistic": chi2,
        "p_value": p_value,
        "df": df,
        "n_groups": K,
        "observed_events": {str(g): float(observed[g]) for g in groups},
        "expected_events": {str(g): float(expected[g]) for g in groups},
    }


# ===================================================================
# Internal: Cox partial likelihood helpers
# ===================================================================


def _cox_neg_logpl_efron(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    strata_arr: Optional[np.ndarray] = None,
    breslow: bool = False,
) -> float:
    """Negative log partial likelihood (Efron, or Breslow, for ties)."""
    n, p = X.shape
    xb = X @ beta
    nll = 0.0

    if strata_arr is None:
        strata_arr = np.zeros(n, dtype=int)

    for s in np.unique(strata_arr):
        mask = strata_arr == s
        Ts, Es, xbs = T[mask], E[mask], xb[mask]
        order = np.argsort(-Ts)  # descending
        Ts, Es, xbs = Ts[order], Es[order], xbs[order]

        event_times = np.unique(Ts[Es == 1])

        for t in event_times:
            at_risk = Ts >= t
            events_at_t = (Ts == t) & (Es == 1)
            d = events_at_t.sum()

            risk_sum = np.exp(xbs[at_risk]).sum()
            event_exp = np.exp(xbs[events_at_t])
            event_xb_sum = xbs[events_at_t].sum()

            nll -= event_xb_sum
            event_exp_sum = event_exp.sum()
            for ell in range(d):
                c = 0.0 if breslow else ell / d
                nll += np.log(risk_sum - c * event_exp_sum)

    return float(nll)


def _cox_score_hessian_efron(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    strata_arr: Optional[np.ndarray] = None,
    breslow: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score vector and Hessian of Cox log partial likelihood (Efron or
    Breslow ties)."""
    n, p = X.shape
    xb = X @ beta
    score = np.zeros(p)
    hessian = np.zeros((p, p))

    if strata_arr is None:
        strata_arr = np.zeros(n, dtype=int)

    for s in np.unique(strata_arr):
        mask = strata_arr == s
        Ts, Es, xbs, Xs = T[mask], E[mask], xb[mask], X[mask]
        order = np.argsort(-Ts)
        Ts, Es, xbs, Xs = Ts[order], Es[order], xbs[order], Xs[order]

        event_times = np.unique(Ts[Es == 1])

        for t in event_times:
            at_risk = Ts >= t
            events_at_t = (Ts == t) & (Es == 1)
            d = events_at_t.sum()

            w_r = np.exp(xbs[at_risk])
            S0 = w_r.sum()
            S1 = (Xs[at_risk] * w_r[:, None]).sum(axis=0)
            S2 = (Xs[at_risk].T * w_r[None, :]) @ Xs[at_risk]

            w_d = np.exp(xbs[events_at_t])
            D0 = w_d.sum()
            D1 = (Xs[events_at_t] * w_d[:, None]).sum(axis=0)
            D2 = (Xs[events_at_t].T * w_d[None, :]) @ Xs[events_at_t]

            score += Xs[events_at_t].sum(axis=0)

            for ell in range(d):
                c = 0.0 if breslow else ell / d
                denom = S0 - c * D0
                if denom <= 0:
                    continue
                weighted_x = (S1 - c * D1) / denom
                score -= weighted_x
                hessian -= (S2 - c * D2) / denom - np.outer(weighted_x, weighted_x)

    return score, hessian


def _cox_score_individual(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    strata_arr: Optional[np.ndarray] = None,
    breslow: bool = False,
) -> np.ndarray:
    """Per-observation score residuals, the ingredients of the sandwich.

    Counting-process decomposition of the partial-likelihood score: at an
    event time ``t`` with ``d`` tied deaths, risk set ``R`` and risk scores
    ``r_k = exp(x_k'b)``, the Efron likelihood has ``d`` terms, the
    ``l``-th with the deaths down-weighted by ``1 - l/d``::

        S0_l = sum_R r_k - (l/d) sum_D r_k,    xbar_l = S1_l / S0_l

    A subject at risk contributes ``- r_k w_kl (x_k - xbar_l) / S0_l`` to
    each term, with ``w_kl = 1 - l/d`` for a death at ``t`` and 1
    otherwise; a death adds ``(x_k - xbar_l) / d``. The residuals sum to
    the score, which is zero at the estimate, and are the rows whose
    outer products the robust variance of Lin and Wei [@lin1989robust]
    sums, as R ``survival::coxph(robust = TRUE)`` computes it. ``breslow``
    drops the down-weighting (one term, counted ``d`` times).

    Before 1.39 the Breslow residuals were used under either tie rule, so
    the robust and clustered standard errors of the default
    ``ties='efron'`` fit were off wherever event times were tied (2% on
    daily relapse data); with no ties the two coincide.
    """
    n, p = X.shape
    xb = X @ beta
    scores = np.zeros((n, p))

    if strata_arr is None:
        strata_arr = np.zeros(n, dtype=int)

    for s in pd.unique(strata_arr):
        idx_s = np.where(strata_arr == s)[0]
        Ts = T[idx_s]
        Es = E[idx_s]
        rs = np.exp(xb[idx_s])
        Xs = X[idx_s]

        for t in np.unique(Ts[Es == 1]):
            at_risk = np.where(Ts >= t)[0]
            dead = np.where((Ts == t) & (Es == 1))[0]
            d = len(dead)
            r_risk = rs[at_risk]
            X_risk = Xs[at_risk]
            S0 = r_risk.sum()
            S1 = X_risk.T @ r_risk
            D0 = rs[dead].sum()
            D1 = Xs[dead].T @ rs[dead]
            is_dead = np.isin(at_risk, dead)
            for ell in range(d):
                c = 0.0 if breslow else ell / d
                denom = S0 - c * D0
                xbar = (S1 - c * D1) / denom
                w = np.where(is_dead, 1.0 - c, 1.0) * r_risk / denom
                scores[idx_s[at_risk]] -= w[:, None] * (X_risk - xbar)
                scores[idx_s[dead]] += (Xs[dead] - xbar) / d

    return scores


def _breslow_baseline_hazard(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    strata_arr: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Breslow estimator of baseline cumulative hazard."""
    n = X.shape[0]
    xb = X @ beta

    if strata_arr is None:
        strata_arr = np.zeros(n, dtype=int)

    rows = []
    for s in np.unique(strata_arr):
        mask = strata_arr == s
        Ts, Es, xbs = T[mask], E[mask], xb[mask]
        event_times = np.sort(np.unique(Ts[Es == 1]))
        cumhaz = 0.0
        for t in event_times:
            at_risk = Ts >= t
            d = ((Ts == t) & (Es == 1)).sum()
            risk_sum = np.exp(xbs[at_risk]).sum()
            cumhaz += d / risk_sum
            rows.append(
                {
                    "time": t,
                    "baseline_cumhaz": cumhaz,
                    "baseline_survival": np.exp(-cumhaz),
                }
            )

    # Prepend t=0
    df = pd.DataFrame(
        [{"time": 0.0, "baseline_cumhaz": 0.0, "baseline_survival": 1.0}] + rows
    )
    return df


def _concordance_index(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
) -> float:
    """Harrell's C-statistic for Cox model."""
    risk_scores = X @ beta
    concordant = 0
    discordant = 0
    tied_risk = 0

    # Only consider pairs where at least one had the event
    event_idx = np.where(E == 1)[0]
    for i in event_idx:
        # Compare with subjects who survived longer
        later = (T > T[i]) | ((T == T[i]) & (E == 0))
        for j in np.where(later)[0]:
            if risk_scores[i] > risk_scores[j]:
                concordant += 1
            elif risk_scores[i] < risk_scores[j]:
                discordant += 1
            else:
                tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


def _schoenfeld_residuals(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    strata_arr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Schoenfeld residuals for PH test."""
    n, p = X.shape
    xb = X @ beta

    if strata_arr is None:
        strata_arr = np.zeros(n, dtype=int)

    resid_list = []

    for s in np.unique(strata_arr):
        mask = strata_arr == s
        Ts, Es, xbs, Xs = T[mask], E[mask], xb[mask], X[mask]
        order = np.argsort(Ts)
        Ts, Es, xbs, Xs = Ts[order], Es[order], xbs[order], Xs[order]

        event_times = np.sort(np.unique(Ts[Es == 1]))

        for t in event_times:
            at_risk = Ts >= t
            events_at_t = (Ts == t) & (Es == 1)

            w = np.exp(xbs[at_risk])
            w /= w.sum()
            expected_x = (Xs[at_risk] * w[:, None]).sum(axis=0)

            # One residual per event at this time
            for idx_e in np.where(events_at_t)[0]:
                resid_list.append(Xs[idx_e] - expected_x)

    return np.array(resid_list) if resid_list else np.empty((0, p))


# ===================================================================
# cox()
# ===================================================================


@accepts_aliases(vce="robust")
def cox(
    formula: Optional[str] = None,
    data: pd.DataFrame = None,
    duration: Optional[str] = None,
    event: Optional[str] = None,
    x: Optional[List[str]] = None,
    ties: str = "efron",
    strata: Optional[str] = None,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    hazard_ratio: bool = True,
    alpha: float = 0.05,
) -> CoxResult:
    """
    Cox Proportional Hazards model via partial likelihood.

    Parameters
    ----------
    formula : str, optional
        Formula of the form ``'duration ~ x1 + x2'``.  If given,
        ``duration`` is inferred from the LHS.
    data : pd.DataFrame
        Input data.
    duration : str, optional
        Column name for follow-up time (overrides formula LHS).
    event : str
        Column name for event indicator (1 = event, 0 = censored).
    x : list of str, optional
        Covariate column names (overrides formula RHS).
    ties : str, default ``'efron'``
        Tie-handling method: ``'efron'`` or ``'breslow'``.
    strata : str, optional
        Column name for stratification variable.
    robust : str, default ``'nonrobust'``
        ``'hc0'`` for sandwich SE.
    cluster : str, optional
        Column name for cluster-robust SE.
    hazard_ratio : bool, default True
        If True, report hazard ratios in the summary alongside coefficients.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    CoxResult
        Result object extending ``EconometricResults`` with
        ``.concordance``, ``.baseline_hazard()``, ``.ph_test()``, ``.plot()``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 150
    >>> age = rng.normal(50, 10, n)
    >>> treatment = rng.integers(0, 2, n).astype(float)
    >>> time = rng.exponential(
    ...     scale=np.exp(-0.03 * (age - 50) - 0.5 * treatment))
    >>> status = (rng.random(n) < 0.8).astype(int)
    >>> df = pd.DataFrame({"time": time, "status": status,
    ...                    "age": age, "treatment": treatment})
    >>> res = sp.cox(formula="time ~ age + treatment", data=df,
    ...              event="status")
    >>> bool(list(res.params.index) == ["age", "treatment"])
    True
    >>> bool(isinstance(res.summary(), str))
    True
    """
    # ---- Parse inputs -------------------------------------------------
    if formula is not None:
        dur_name, x_names = _parse_formula(formula)
        if x is None and any(name not in data for name in x_names):
            # C(site), a*b, I(x^2): built as columns, the way sp.regress
            # and sp.qreg read the same right-hand side
            from ..core.utils import formula_to_columns

            data, dur_name, x_names = formula_to_columns(formula, data)
        if duration is None:
            duration = dur_name
        if x is None:
            x = x_names
    if duration is None or event is None or x is None:
        raise ValueError("Provide (formula + event) or (duration, event, x).")

    cols_needed = [duration, event] + x
    if strata is not None:
        cols_needed.append(strata)
    if cluster is not None:
        cols_needed.append(cluster)
    data = data.dropna(subset=cols_needed).copy()

    T = data[duration].values.astype(float)
    E = data[event].values.astype(float)
    X = data[x].values.astype(float)
    n, p = X.shape

    strata_arr = data[strata].values if strata else None
    cluster_arr = data[cluster].values if cluster else None

    # ---- Optimization (Newton-Raphson) --------------------------------
    beta0 = np.zeros(p)

    if ties not in ("efron", "breslow"):
        raise ValueError(f"ties must be 'efron' or 'breslow', got {ties!r}")

    breslow = ties == "breslow"

    def neg_logpl(b: np.ndarray) -> float:
        return _cox_neg_logpl_efron(b, X, T, E, strata_arr, breslow=breslow)

    # Newton-Raphson with fallback to L-BFGS-B
    beta = beta0.copy()
    converged = False
    for iteration in range(50):
        score, hessian = _cox_score_hessian_efron(
            beta, X, T, E, strata_arr, breslow=breslow
        )
        neg_H = -hessian
        # Check if Hessian is positive definite
        try:
            eigvals = np.linalg.eigvalsh(neg_H)
            if np.min(eigvals) > 1e-10:
                step = np.linalg.solve(neg_H, score)
            else:
                # Regularize
                reg = max(0, -np.min(eigvals) + 1e-6) * np.eye(p)
                step = np.linalg.solve(neg_H + reg, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(neg_H, score, rcond=None)[0]

        # Step-halving line search
        step_size = 1.0
        current_nll = neg_logpl(beta)
        for _ in range(20):
            candidate = beta + step_size * step
            if neg_logpl(candidate) < current_nll - 1e-10:
                break
            step_size *= 0.5
        beta = beta + step_size * step

        if np.max(np.abs(step_size * step)) < 1e-9:
            converged = True
            break

    if not converged:
        # Fallback to scipy optimizer
        result = optimize.minimize(neg_logpl, beta, method="L-BFGS-B")
        beta = result.x

    # ---- Variance estimation ------------------------------------------
    _, hessian = _cox_score_hessian_efron(beta, X, T, E, strata_arr, breslow=breslow)
    neg_H = -hessian
    try:
        info_inv = np.linalg.inv(neg_H)
    except np.linalg.LinAlgError:
        info_inv = np.linalg.pinv(neg_H)

    if cluster is not None:
        score_i = _cox_score_individual(beta, X, T, E, strata_arr, breslow=breslow)
        var_beta = _cluster_variance(X, info_inv, score_i, cluster_arr)
    elif robust in ("hc0", "HC0", "robust"):
        score_i = _cox_score_individual(beta, X, T, E, strata_arr, breslow=breslow)
        var_beta = _sandwich_variance(X, info_inv, score_i)
    else:
        var_beta = info_inv

    se = np.sqrt(np.diag(var_beta))

    # ---- Derived quantities -------------------------------------------
    loglik = -neg_logpl(beta)
    loglik0 = -neg_logpl(np.zeros(p))
    bh_df = _breslow_baseline_hazard(beta, X, T, E, strata_arr)
    c_index = _concordance_index(beta, X, T, E)
    schoenfeld = _schoenfeld_residuals(beta, X, T, E, strata_arr)

    # Hazard ratios
    hr = np.exp(beta)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    hr_ci = np.column_stack([np.exp(beta - z_crit * se), np.exp(beta + z_crit * se)])

    # ---- Build result -------------------------------------------------
    params_s = pd.Series(beta, index=x, name="coef")
    se_s = pd.Series(se, index=x, name="se")

    model_info: Dict[str, Any] = {
        "model_type": "Cox Proportional Hazards",
        "method": f"Partial likelihood ({ties})",
        "ties": ties,
        "robust": robust if cluster is None else f"cluster({cluster})",
    }
    if strata:
        model_info["strata"] = strata

    data_info = {
        "nobs": n,
        "n_events": int(E.sum()),
        "dependent_var": duration,
        "event_var": event,
        "df_resid": np.inf,  # use normal distribution for Cox
    }

    diagnostics = {
        "Log-likelihood": loglik,
        "Log-likelihood (null)": loglik0,
        "LR chi2": 2 * (loglik - loglik0),
        "LR chi2 p-value": stats.chi2.sf(2 * (loglik - loglik0), df=p),
        "Concordance (C)": c_index,
        "AIC": -2 * loglik + 2 * p,
        "BIC": -2 * loglik + np.log(E.sum()) * p,
    }

    model_info["alpha"] = alpha
    _result = CoxResult(
        params=params_s,
        std_errors=se_s,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
        _X=X,
        _durations=T,
        _events=E,
        _baseline_hazard_df=bh_df,
        _concordance=c_index,
        _schoenfeld_resid=schoenfeld,
        _strata=strata_arr,
        _hazard_ratios=hr,
        _hr_ci=hr_ci,
    )
    # Store the global proportional-hazards test (from the Schoenfeld residuals
    # computed above) so result.violations() can flag a PH violation without
    # recomputing it — best-effort: strata models don't expose the residuals.
    try:
        _ph = _result.ph_test()
        _imin = int(np.asarray(_ph["p_value"].values).argmin())
        _result.model_info["ph_test"] = {
            "min_pvalue": float(_ph["p_value"].iloc[_imin]),
            "worst_variable": str(_ph["variable"].iloc[_imin]),
        }
    except (RuntimeError, ValueError, KeyError, IndexError):
        pass
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.survival.cox",
            params={
                "formula": formula,
                "duration": duration,
                "event": event,
                "x": list(x) if x else None,
                "ties": ties,
                "strata": strata,
                "robust": robust,
                "cluster": cluster,
                "hazard_ratio": hazard_ratio,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ===================================================================
# Parametric survival / AFT: survreg()
# ===================================================================


def _weibull_loglik(
    params: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
) -> float:
    """Log-likelihood for Weibull AFT: T ~ Weibull(lambda=exp(-Xb/sigma), k=1/sigma)."""
    p = X.shape[1]
    beta = params[:p]
    log_sigma = params[p]
    sigma = np.exp(log_sigma)

    mu = X @ beta
    z = (np.log(T + 1e-15) - mu) / sigma

    # log f(t) = -log(sigma) - log(t) + z - exp(z)  [standard extreme value]
    # log S(t) = -exp(z)
    ll = E * (-log_sigma - np.log(T + 1e-15) + z - np.exp(z)) + (1 - E) * (-np.exp(z))
    return float(-ll.sum())


def _lognormal_loglik(
    params: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
) -> float:
    """Log-likelihood for log-normal AFT."""
    p = X.shape[1]
    beta = params[:p]
    log_sigma = params[p]
    sigma = np.exp(log_sigma)

    mu = X @ beta
    z = (np.log(T + 1e-15) - mu) / sigma

    ll = E * (stats.norm.logpdf(z) - log_sigma - np.log(T + 1e-15)) + (
        1 - E
    ) * stats.norm.logsf(z)
    return float(-ll.sum())


def _loglogistic_loglik(
    params: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
) -> float:
    """Log-likelihood for log-logistic AFT."""
    p = X.shape[1]
    beta = params[:p]
    log_sigma = params[p]
    sigma = np.exp(log_sigma)

    mu = X @ beta
    z = (np.log(T + 1e-15) - mu) / sigma

    # f(t) = [exp(z) / (sigma * t * (1+exp(z))^2)]
    # S(t) = 1 / (1+exp(z))
    ll = E * (z - log_sigma - np.log(T + 1e-15) - 2 * np.log(1 + np.exp(z))) + (
        1 - E
    ) * (-np.log(1 + np.exp(z)))
    return float(-ll.sum())


def _softplus(z: np.ndarray) -> np.ndarray:
    """``log(1 + exp(z))`` without overflow; complex-step safe."""
    pos = np.real(z) > 0
    return np.asarray(np.where(pos, z, 0.0) + np.log1p(np.exp(np.where(pos, -z, z))))


def _aft_log_density_survivor(z: np.ndarray, family: str) -> Tuple[Any, Any]:
    """Log density and log survivor function of the standardised log time.

    ``z = (log t - x'b) / sigma``. The density is that of ``z``; the caller
    subtracts ``log(sigma) + log(t)`` to get the density of ``t``.
    """
    from scipy import special

    if family == "weibull":  # standard extreme value
        return z - np.exp(z), -np.exp(z)
    if family == "lognormal":
        return -0.5 * np.log(2.0 * np.pi) - 0.5 * z * z, special.log_ndtr(-z)
    sp_z = _softplus(z)  # log-logistic
    return z - 2.0 * sp_z, -sp_z


@accepts_aliases(vce="robust")
def survreg(
    formula: Optional[str] = None,
    data: pd.DataFrame = None,
    duration: Optional[str] = None,
    event: Optional[str] = None,
    x: Optional[List[str]] = None,
    dist: str = "weibull",
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    frailty: Optional[str] = None,
) -> EconometricResults:
    """
    Parametric survival model (AFT parameterization).

    The data may also come first: ``sp.survreg(df, duration=..., event=...,
    x=[...])`` and ``sp.survreg(df, "t ~ x1 + x2", event=...)``.

    Parameters
    ----------
    formula : str, optional
        Formula ``'duration ~ x1 + x2'``.
    data : pd.DataFrame
    duration : str, optional
        Follow-up time column (or formula LHS).
    event : str
        Event indicator column.
    x : list of str, optional
        Covariate columns (or formula RHS).
    dist : str, default ``'weibull'``
        Distribution: ``'weibull'``, ``'exponential'``, ``'lognormal'``,
        ``'loglogistic'``.
    robust : str, default ``'nonrobust'``
        ``'nonrobust'`` (observed information), ``'robust'`` (sandwich,
        Stata's ``vce(robust)``) or ``'cluster'``. ``vce=`` is an alias.
    cluster : str, optional
        Cluster column; implies the cluster sandwich.
    alpha : float, default 0.05
    frailty : {None, 'gamma'}, optional
        Unobserved heterogeneity. ``'gamma'`` multiplies each
        observation's hazard by a gamma variable with mean one and
        variance ``theta``, as Stata's ``streg, frailty(gamma)``. The
        survivor function becomes ``[1 - theta log S(t)]^(-1/theta)``.
        ``log(theta)`` is reported with the other parameters, and
        ``model_info`` carries ``theta`` and the likelihood-ratio test of
        ``theta = 0``, whose p-value is halved because the null is on the
        boundary. Ignoring heterogeneity biases a Weibull fit towards
        negative duration dependence.

    Returns
    -------
    EconometricResults
        Fitted parametric survival model.  Parameters include covariates
        and ``log(sigma)`` (scale). ``diagnostics['Log-likelihood']`` is
        the likelihood of the durations, as R ``survival::survreg``
        reports it; ``model_info['ll_log_time']`` is the likelihood of the
        log durations, which is what Stata's ``streg`` prints (the two
        differ by the sum of ``log(t)`` over the events).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 150
    >>> age = rng.normal(50, 10, n)
    >>> treatment = rng.integers(0, 2, n).astype(float)
    >>> time = rng.exponential(
    ...     scale=np.exp(0.02 * (age - 50) + 0.5 * treatment))
    >>> status = (rng.random(n) < 0.8).astype(int)
    >>> df = pd.DataFrame({"time": time, "status": status,
    ...                    "age": age, "treatment": treatment})
    >>> res = sp.survreg("time ~ age + treatment", data=df,
    ...                  event="status", dist="weibull")
    >>> bool("log(sigma)" in list(res.params.index))
    True
    """
    # ---- Parse inputs -------------------------------------------------
    from ..core._vcov import ml_vcov
    from ..core._vcov_spec import parse_se_request
    from ..exceptions import ConvergenceWarning, MethodIncompatibility
    from ..regression._optim_helpers import (
        inverse_information,
        ml_newton_polish,
        se_from_vcov,
    )

    if isinstance(formula, pd.DataFrame):
        # data first, as in sp.tobit and sp.qreg
        formula, data = (data if isinstance(data, str) else None), formula
    if formula is not None:
        dur_name, x_names = _parse_formula(formula)
        if duration is None:
            duration = dur_name
        if x is None:
            x = x_names
    if duration is None or event is None or x is None:
        raise ValueError("Provide (formula + event) or (duration, event, x).")
    se_req = parse_se_request(
        robust,
        cluster,
        function="survreg",
        supported=("nonrobust", "robust", "cluster"),
    )
    se_kind, cluster = se_req.kind, se_req.cluster
    frailty_kind = None if frailty is None else str(frailty).lower()
    if frailty_kind not in (None, "gamma"):
        raise MethodIncompatibility(
            f"survreg: frailty={frailty!r} is not available; use 'gamma' or None."
        )

    cols_needed = [duration, event] + list(x)
    if isinstance(cluster, str):
        cols_needed.append(cluster)
    data = data.dropna(subset=cols_needed).copy()

    T = data[duration].values.astype(float)
    E = data[event].values.astype(float)
    if np.any(T <= 0):
        raise MethodIncompatibility(
            f"survreg: {int(np.sum(T <= 0))} durations are zero or negative; "
            "the model is for the log of a positive duration. Drop those "
            "rows or shift the time origin.",
            diagnostics={"n_nonpositive": int(np.sum(T <= 0))},
        )
    # Design matrix with intercept
    X = np.column_stack([np.ones(len(T)), data[x].values.astype(float)])
    n, p = X.shape  # p includes intercept
    param_names = ["_cons"] + list(x)

    # ---- Select distribution -----------------------------------------
    dist_lower = dist.lower()
    if dist_lower not in ("weibull", "exponential", "lognormal", "loglogistic"):
        raise ValueError(
            f"dist must be weibull/exponential/lognormal/loglogistic, got {dist!r}"
        )
    family = "weibull" if dist_lower == "exponential" else dist_lower
    has_scale = dist_lower != "exponential"
    log_t = np.log(T)
    events = E > 0

    def obs_loglik(theta: np.ndarray, with_frailty: bool) -> np.ndarray:
        """Per-observation log-likelihood of t; complex-step safe."""
        theta = np.asarray(theta)
        ln_s = theta[p] if has_scale else 0.0
        z = (log_t - X @ theta[:p]) / np.exp(ln_s)
        log_f, log_S = _aft_log_density_survivor(z, family)
        log_f = log_f - ln_s - log_t
        if with_frailty:
            th = np.exp(theta[-1])
            g = np.log(1.0 - th * log_S)
            # f_theta = h * [1 - theta log S]^(-1/theta - 1), h the hazard.
            log_f = (log_f - log_S) - (1.0 / th + 1.0) * g
            log_S = -g / th
        return np.asarray(np.where(events, log_f, log_S))

    def fit(with_frailty: bool, start: np.ndarray) -> Tuple[Any, ...]:
        def fn(t: np.ndarray) -> np.ndarray:
            return obs_loglik(t, with_frailty)

        def neg(t: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                v = float(np.sum(np.real(fn(t))))
            return -v if np.isfinite(v) else 1e300

        res = optimize.minimize(neg, start, method="BFGS", options={"gtol": 1e-7})
        est, scores, H, _ = ml_newton_polish(fn, np.asarray(res.x, dtype=float))
        return est, scores, H, float(np.sum(np.real(fn(est))))

    k_base = p + (1 if has_scale else 0)
    start = np.zeros(k_base)
    start[0] = float(np.mean(log_t))
    est0, scores0, H0, ll0 = fit(False, start)

    frailty_info: Dict[str, Any] = {}
    est, scores, H, loglik_val = est0, scores0, H0, ll0
    at_boundary = False
    if frailty_kind == "gamma":
        est1, scores1, H1, ll1 = fit(True, np.append(est0, np.log(0.5)))
        theta_hat = float(np.exp(est1[-1]))
        # The plain model is the limit theta -> 0. When the fit runs to that
        # boundary the likelihood is flat in log(theta) and nothing about
        # theta is identified beyond "zero".
        at_boundary = theta_hat < 1e-5 or ll1 <= ll0 + 1e-8
        if at_boundary:
            import warnings

            warnings.warn(
                "survreg: the frailty variance is estimated at its boundary "
                "(theta = 0); the fit is the model without frailty and "
                "log(theta) has no standard error.",
                ConvergenceWarning,
                stacklevel=2,
            )
            theta_hat, lr = 0.0, 0.0
        else:
            est, scores, H, loglik_val = est1, scores1, H1, ll1
            lr = 2.0 * (ll1 - ll0)
        frailty_info = {
            "frailty": "gamma",
            "theta": theta_hat,
            "theta_at_boundary": bool(at_boundary),
            "lr_theta_chi2": float(lr),
            # H0 is on the boundary: chibar2(01), half the chi2(1) tail.
            "lr_theta_pvalue": (1.0 if lr <= 0 else float(0.5 * stats.chi2.sf(lr, 1))),
            "ll_no_frailty": float(ll0),
        }

    # ---- Standard errors ---------------------------------------------
    clusters = data[cluster].to_numpy() if se_kind == "cluster" else None
    var_mat = ml_vcov(
        inverse_information(H),
        scores if se_kind != "nonrobust" else None,
        kind=se_kind,
        clusters=clusters,
    )
    se_all = se_from_vcov(var_mat)
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))

    beta_hat = est[:p]
    log_sigma_hat = float(est[p]) if has_scale else 0.0
    all_param_names = list(param_names)
    all_params = list(beta_hat)
    all_se = list(se_all[:p])
    if has_scale:
        all_param_names.append("log(sigma)")
        all_params.append(log_sigma_hat)
        all_se.append(se_all[p])
    if frailty_kind == "gamma":
        all_param_names.append("log(theta)")
        if at_boundary:
            all_params.append(-np.inf)
            all_se.append(np.nan)
        else:
            all_params.append(float(est[-1]))
            all_se.append(se_all[-1])
            # Delta method for theta itself.
            frailty_info["theta_se"] = float(se_all[-1] * frailty_info["theta"])

    params_s = pd.Series(all_params, index=all_param_names, name="coef")
    se_s = pd.Series(all_se, index=all_param_names, name="se")

    model_info: Dict[str, Any] = {
        "model_type": f"Parametric Survival ({dist})",
        "method": "AFT — Maximum Likelihood",
        "distribution": dist,
        "robust": se_kind if se_kind != "cluster" else f"cluster({cluster})",
        "vce": se_kind,
        "cluster": cluster if se_kind == "cluster" else None,
        "ll": loglik_val,
        "ll_log_time": float(loglik_val + np.sum(log_t[events])),
        "gradient_norm": grad_norm,
        "converged": bool(grad_norm < 1e-5 * max(1.0, n**0.5)),
        **frailty_info,
    }
    if se_kind == "cluster" and clusters is not None:
        model_info["n_clusters"] = int(pd.unique(clusters).size)

    n_params = int(np.sum(np.isfinite(all_params)))
    data_info_dict = {
        "nobs": n,
        "n_events": int(E.sum()),
        "dependent_var": duration,
        "event_var": event,
        "df_resid": np.inf,
        "var_cov": var_mat,
        "llobs": np.real(obs_loglik(est, len(est) > k_base)),
        "n_params": n_params,
    }

    diagnostics = {
        "Log-likelihood": loglik_val,
        "AIC": -2 * loglik_val + 2 * n_params,
        "BIC": -2 * loglik_val + np.log(n) * n_params,
        "sigma": np.exp(log_sigma_hat),
    }
    if dist_lower == "weibull":
        diagnostics["shape (1/sigma)"] = 1.0 / np.exp(log_sigma_hat)
    if frailty_info:
        diagnostics["theta (frailty variance)"] = frailty_info["theta"]
        diagnostics["LR test of theta = 0 (chibar2)"] = frailty_info["lr_theta_chi2"]
        diagnostics["Prob >= chibar2"] = frailty_info["lr_theta_pvalue"]

    model_info["alpha"] = alpha
    return EconometricResults(
        params=params_s,
        std_errors=se_s,
        model_info=model_info,
        data_info=data_info_dict,
        diagnostics=diagnostics,
    )
