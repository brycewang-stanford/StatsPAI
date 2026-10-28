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
from ..core._call_record import records_call
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._cox_core import CoxKernel, harrell_c, km_left_survival

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
    def ph_test(self, transform: Any = "km", method: str = "score") -> pd.DataFrame:
        """
        Test the proportional hazards assumption (Grambsch and Therneau).

        For each covariate the model is embedded in one whose coefficient
        varies with time, ``beta_j(t) = beta_j + theta_j * g(t)``, and
        ``theta_j = 0`` is tested by a score test at the fitted model; the
        ``GLOBAL`` row tests all the ``theta_j`` jointly. No larger model
        is fitted. This is the test of ``survival::cox.zph`` in R (version
        3.0 and later), and the statistics agree with it under either tie
        rule and with strata.

        Parameters
        ----------
        transform : {'km', 'rank', 'identity', 'log'} or callable
            The function of time ``g``. ``'km'`` (default, as in
            ``cox.zph``) is one minus the left-continuous Kaplan-Meier
            estimate of the pooled sample, which spreads the event times
            evenly and keeps a few late events from dominating;
            ``'identity'`` is time itself (the default of Stata
            ``estat phtest``); ``'rank'`` is the rank of the time among all
            observations (among the failures under ``method='approx'``, as
            in Stata); ``'log'`` needs positive times. A callable is
            applied to the array of event times.
        method : {'score', 'approx'}, default 'score'
            ``'score'`` is the score test described above. ``'approx'`` is
            the approximation of Grambsch and Therneau (1994) that Stata
            ``estat phtest`` and versions of ``survival`` before 3.0 report:
            it replaces the information matrix at each event time by the
            average over event times, which turns the test into a
            regression of the scaled Schoenfeld residuals on ``g(t)``. With
            ``transform='identity'`` it reproduces ``estat phtest``, and
            with the other three its ``km``, ``rank`` and ``log`` options.
            Unweighted right-censored fits only.

        Returns
        -------
        pd.DataFrame
            One row per covariate and a final ``GLOBAL`` row, with columns
            ``variable``, ``rho`` (correlation of the scaled Schoenfeld
            residuals with ``g(t)``; missing for ``GLOBAL``), ``chi2``,
            ``df`` and ``p_value``.

        Notes
        -----
        Weighted fits (``weights=``) and counting-process fits (``start=``)
        get the score test of ``cox.zph`` for such fits: the case weights
        enter the score and the information as they enter the partial
        likelihood, the information is the model-based one whatever
        variance the fit reports, and ``g`` ignores the weights (``'km'``
        is the unweighted Kaplan-Meier of the ``(start, stop]`` data,
        ``'rank'`` ranks the stop times of all rows). Fits with
        ``design=`` are refused.

        The two methods differ most when the covariate distribution of the
        risk set changes over time, which is when the average information
        matrix is a poor stand-in for each event time's own.

        Before 1.39 this method returned ``n * rho^2`` with ``rho`` the
        Spearman correlation between time and the unscaled residuals. That
        number is not the Grambsch-Therneau statistic and was several times
        too small in places.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> import statspai as sp
        >>> rng = np.random.default_rng(0)
        >>> n = 400
        >>> x = rng.binomial(1, 0.5, n)
        >>> t = rng.exponential(1.0, n) ** np.where(x == 1, 2.0, 1.0)
        >>> df = pd.DataFrame({"t": t, "d": 1, "x": x})
        >>> fit = sp.cox(data=df, duration="t", event="d", x=["x"])
        >>> bool(fit.ph_test().loc[0, "p_value"] < 0.01)
        True

        References
        ----------
        [@grambsch1994proportional]
        """
        if self.model_info.get("tvc") is not None:
            # R cox.zph refuses tt() terms for the same reason: the model
            # already has coefficients that vary with time, and its rows
            # are the split episodes, not the observations.
            raise MethodIncompatibility(
                "ph_test: this fit has time-varying coefficients (tvc=); the "
                "Grambsch-Therneau test is defined for the proportional model.",
                recovery_hint="Test the tvc coefficients themselves (their "
                "z statistics test proportionality against g(t)), or refit "
                "without tvc= and call ph_test().",
            )
        if self.model_info.get("survey_design") is not None:
            raise MethodIncompatibility(
                "ph_test: this fit used design= (a complex survey design); "
                "the score test here uses the model-based information and "
                "would ignore the strata and PSUs.",
                recovery_hint="Refit with weights=<design weight column> for "
                "a weighted (non-design) proportional-hazards test.",
            )
        w_arr = getattr(self, "_weights", None)
        start_arr = getattr(self, "_start", None)
        beta = np.asarray(self.params.values, dtype=float)
        X = np.asarray(self._X, dtype=float)
        T = np.asarray(self._durations, dtype=float)
        E = np.asarray(self._events)
        strata = self._strata
        if strata is None:
            strata = np.zeros(len(T), dtype=int)
        breslow = str(self.model_info.get("ties", "efron")).lower() == "breslow"
        names = list(self.params.index)
        p = len(names)
        method = str(method).lower()
        if method not in ("score", "approx"):
            raise MethodIncompatibility(
                f"ph_test: method must be 'score' or 'approx', got {method!r}",
                recovery_hint="Use method='score' (cox.zph) or 'approx' (Stata).",
            )
        if method == "approx" and (w_arr is not None or start_arr is not None):
            raise MethodIncompatibility(
                "ph_test(method='approx') is the Grambsch-Therneau "
                "approximation for unweighted right-censored fits; this fit "
                "used weights= or start= (counting-process data).",
                recovery_hint="Use method='score' (R survival::cox.zph), "
                "which handles case weights and (start, stop] data.",
            )

        kern = CoxKernel(
            X,
            T,
            E,
            np.asarray(strata),
            breslow=breslow,
            weights=w_arr,
            start=start_arr,
        )
        ev, times, U_k, resid_group, resid = kern.event_time_scores(beta)
        if len(times) < 2:
            raise DataInsufficient(
                "ph_test needs at least two distinct event times.",
            )
        ev_only = method == "approx"
        g_k = _ph_time_transform(transform, T, E, times, ev_only, start=start_arr)
        if not np.all(np.isfinite(g_k)):
            raise DataInsufficient(
                f"ph_test: transform={transform!r} is not finite at every "
                "event time (the log needs positive times).",
                recovery_hint="Use transform='km' or 'rank'.",
            )
        # g at each death is g at its event time
        g_group = np.zeros(kern.rs.n_groups)
        g_group[ev] = g_k
        g_event = g_group[resid_group]

        # The information of each event time enters only through three
        # weighted sums, which need no per-time matrix.
        is_event = np.zeros(kern.rs.n_groups)
        is_event[ev] = 1.0
        U = np.einsum("k,kj->j", g_k, U_k)
        I_bb = kern.information(beta, is_event)
        I_bg = kern.information(beta, g_group)
        I_gg = kern.information(beta, g_group**2)
        try:
            I_bb_inv = np.linalg.inv(I_bb)
        except np.linalg.LinAlgError:
            I_bb_inv = np.linalg.pinv(I_bb)
        V_theta = I_gg - I_bg @ I_bb_inv @ I_bg

        # Scaled Schoenfeld residuals: the descriptive correlation, and the
        # whole of the 1994 approximation.
        n_dead = resid.shape[0]
        scaled = resid @ I_bb_inv * n_dead
        g_c = g_event - g_event.mean()
        ss_g = float(g_c @ g_c)
        approx_num = g_c @ scaled  # (p,)
        approx_raw = g_c @ resid

        rows = []
        for j, var in enumerate(names):
            if method == "score":
                v_j = float(V_theta[j, j])
                chi2 = float(U[j] ** 2 / v_j) if v_j > 0 else float("nan")
            else:
                chi2 = float(approx_num[j] ** 2 / (n_dead * I_bb_inv[j, j] * ss_g))
            col = scaled[:, j]
            if np.std(col) > 0 and np.std(g_event) > 0:
                rho = float(np.corrcoef(g_event, col)[0, 1])
            else:
                rho = float("nan")
            rows.append(
                {
                    "variable": var,
                    "rho": rho,
                    "chi2": chi2,
                    "df": 1,
                    "p_value": float(stats.chi2.sf(chi2, df=1)),
                }
            )
        if method == "approx":
            chi2_g = float(approx_raw @ I_bb_inv @ approx_raw * n_dead / ss_g)
        else:
            try:
                chi2_g = float(U @ np.linalg.solve(V_theta, U))
            except np.linalg.LinAlgError:
                chi2_g = float(U @ np.linalg.pinv(V_theta) @ U)
        rows.append(
            {
                "variable": "GLOBAL",
                "rho": float("nan"),
                "chi2": chi2_g,
                "df": p,
                "p_value": float(stats.chi2.sf(chi2_g, df=p)),
            }
        )
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
        **kwargs
            ``type=`` is accepted as an alias of ``kind`` (the spelling
            other result classes use); the rest go to matplotlib.
        """
        import matplotlib.pyplot as plt

        if "type" in kwargs:
            # ``type=`` used to fall into **kwargs and be handed to
            # matplotlib, so an unknown value silently drew the survival
            # curve.
            kind = kwargs.pop("type")
            if kind == "auto":
                kind = "survival"
        if kind not in ("survival", "hazard", "hr"):
            raise MethodIncompatibility(
                f"Unknown kind={kind!r}. Use 'survival', 'hazard', or 'hr'.",
                recovery_hint="Pass kind='survival', 'hazard' or 'hr'.",
                diagnostics={"kind": kind},
            )

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
    >>> km = sp.kaplan_meier(data=df, duration="time", event="status",
    ...                      conf_type="log-log")
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
        """Median survival time (scalar or dict by group).

        The first time at which ``S(t) <= 0.5``, as Stata ``stci`` reports
        it. When ``S(t)`` equals 0.5 exactly on a step, R ``survfit``
        reports the midpoint between that time and the next event time
        instead (deaths at 1, 2, 3, 4: Stata 2, R 2.5).
        """
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

    z = stats.norm.ppf(1 - alpha / 2)

    # One pass over the distinct times: the risk set at a time is everyone
    # not yet counted out, so the counts are running totals.
    all_times, inv, counts = np.unique(
        durations, return_inverse=True, return_counts=True
    )
    dead_at = np.bincount(inv, weights=(events == 1), minlength=len(all_times))
    cens_at = np.bincount(inv, weights=(events == 0), minlength=len(all_times))
    risk_at = n_total - (np.cumsum(counts) - counts)
    keep = dead_at > 0
    unique_times = all_times[keep]
    n_risk = risk_at[keep].astype(float)
    n_event = dead_at[keep]
    n_censor = cens_at[keep]

    survival = np.cumprod(1 - n_event / n_risk)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Greenwood variance; a time that empties the risk set adds nothing
        inc = np.where(n_risk > n_event, n_event / (n_risk * (n_risk - n_event)), 0.0)
    var_sum = np.cumsum(inc)
    se = survival * np.sqrt(var_sum)

    plain = (
        (conf_type == "plain") | (var_sum <= 0) | ~((survival > 0.0) & (survival < 1.0))
    )
    ci_lo = np.maximum(0.0, survival - z * se)
    ci_hi = np.minimum(1.0, survival + z * se)
    if conf_type != "plain":
        undefined = plain & (survival <= 0.0)  # undefined on the log scales
        ci_lo[undefined] = np.nan
        ci_hi[undefined] = np.nan
        ok = ~plain
        with np.errstate(divide="ignore", invalid="ignore"):
            if conf_type == "log":
                # symmetric for log S(t): R survival::survfit's default
                half = z * np.sqrt(var_sum)
                lo = survival * np.exp(-half)
                hi = np.minimum(1.0, survival * np.exp(half))
            else:
                # symmetric for log(-log S(t)): Stata sts list's default
                half = z * np.sqrt(var_sum) / np.abs(np.log(survival))
                lo = survival ** np.exp(half)
                hi = survival ** np.exp(-half)
        ci_lo[ok] = lo[ok]
        ci_hi[ok] = hi[ok]

    body = pd.DataFrame(
        {
            "time": unique_times,
            "n_risk": n_risk.astype(int),
            "n_event": n_event.astype(int),
            "n_censor": n_censor.astype(int),
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
    if not len(body):
        return pd.DataFrame([t0])
    return pd.concat([pd.DataFrame([t0]), body], ignore_index=True)


# ===================================================================
# kaplan_meier()
# ===================================================================


def kaplan_meier(
    data: pd.DataFrame,
    duration: str,
    event: str,
    group: Optional[str] = None,
    alpha: float = 0.05,
    conf_type: Optional[str] = "log-log",
    weights: Optional[str] = None,
    start: Optional[str] = None,
    cluster: Optional[str] = None,
    se_method: Optional[str] = None,
    design: Optional[Any] = None,
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
    conf_type : {"log-log", "log", "plain"}, default "log-log"
        Scale on which the interval is symmetric. ``"plain"`` is ``S(t)
        +/- z se`` truncated to [0, 1]; ``"log"`` is symmetric for ``log
        S(t)`` (the default of R ``survival::survfit``); ``"log-log"`` for
        ``log(-log S(t))`` (the default of Stata ``sts list``), which
        always stays inside (0, 1) and has the best coverage in the
        tails. The estimate and its standard error are the same under all
        three.

        .. versionchanged:: 1.40.0
            The default is ``"log-log"`` (it was ``"plain"``): the plain
            interval is the one of the three that leaves [0, 1] and has to
            be truncated, and it is neither reference package's default.
            Pass ``conf_type="plain"`` for the interval of 1.39 and
            earlier.
    weights : str, optional
        Column of non-negative case weights (inverse-probability-of-
        treatment weights give the IPTW-adjusted curve of each ``group``).
        Deaths and the number at risk are weighted sums, as in R
        ``survfit(weights=)``; ``n_risk`` / ``n_event`` / ``n_censor`` are
        then weighted and reported as floats.
    start : str, optional
        Column with each row's entry time: counting-process data ``(start,
        duration]`` (delayed entry), as R ``Surv(start, stop, event)``. A
        row is at risk at ``t`` iff ``start < t <= duration``.
    cluster : str, optional
        Column whose levels are independent units (subject id with several
        rows); the robust variance sums the influence within each.
    se_method : {'greenwood', 'ij', 'xie_liu'}, optional
        ``'greenwood'`` is Greenwood's formula on the (weighted) counts;
        ``'ij'`` the infinitesimal jackknife (robust) variance of R
        ``survfit(robust=TRUE)``, which treats the weights as fixed
        sampling weights; ``'xie_liu'`` Greenwood's formula with the
        effective number at risk ``(sum w)^2 / sum w^2`` in the denominator,
        the variance R ``adjustedCurves`` reports for IPTW Kaplan-Meier
        curves (``adjustedsurv(method = "iptw_km")``). Not given: R
        survfit's default, ``'ij'`` with ``cluster=`` or non-integer weights
        and ``'greenwood'`` otherwise.
    design : SurveyDesign, optional
        A complex survey design from :func:`sp.svydesign` declared on
        ``data`` (same rows). The curve is weighted by the design weights
        and ``std_err`` is the survey linearisation of the influence of
        each row on ``S(t)`` over strata, PSUs and the finite population
        correction (R ``survey::svyrecvar`` applied to the weighted
        infinitesimal-jackknife influence that ``survival``'s
        ``resid(survfit, type = "surv")`` reports), with intervals on
        the t distribution with the design degrees of freedom. R
        ``svykm(se = TRUE)`` is not this estimator: its curve differs
        from the product-limit one. Each
        ``group``'s curve is a domain: its standard errors keep every PSU
        of the design. Not with ``weights=``, ``cluster=`` or
        ``se_method=``.

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
    >>> km = sp.kaplan_meier(data=df, duration="time", event="status",
    ...                      conf_type="log-log")
    >>> bool("survival" in km.survival_table.columns)
    True
    >>> bool(isinstance(km.median_survival, float))
    True
    """
    if conf_type is None:  # explicit None: the default
        conf_type = "log-log"
    conf_type = str(conf_type).lower().replace("loglog", "log-log")
    if conf_type not in ("plain", "log", "log-log"):
        raise ValueError(
            f"conf_type must be 'plain', 'log' or 'log-log', got {conf_type!r}"
        )
    svy: Any = None
    if design is not None:
        from ..survey.linearization import check_design

        svy = check_design(
            design,
            data,
            function="kaplan_meier",
            weights=weights,
            cluster=cluster,
            allow_calibrated=False,
        )
        if se_method is not None:
            raise MethodIncompatibility(
                "kaplan_meier: design= gives the survey linearisation "
                "variance; se_method= cannot be combined with it.",
                recovery_hint="Drop se_method=.",
            )
        data = data.assign(__svy_row__=np.arange(len(data)))
    extra = [c for c in (weights, start, cluster) if c is not None]
    from ._validate import check_duration_event, warn_missing_weights

    warn_missing_weights(data, weights, "kaplan_meier")
    data = data.dropna(subset=[duration, event] + extra)
    check_duration_event(data[duration], data[event], "kaplan_meier")

    from ._km import km_table_weighted, resolve_km_se

    w_all = data[weights].values.astype(float) if weights is not None else None
    if w_all is not None and (not np.all(np.isfinite(w_all)) or (w_all < 0).any()):
        raise MethodIncompatibility(
            "kaplan_meier: weights must be finite and non-negative.",
            recovery_hint="Check the weight column for negative or infinite values.",
        )
    if start is not None and not np.all(
        data[start].values.astype(float) < data[duration].values.astype(float)
    ):
        raise DataInsufficient(
            "kaplan_meier: start must be strictly less than the duration on "
            "every row.",
            recovery_hint="Drop or fix the zero-length intervals.",
        )
    se_kind = resolve_km_se(se_method, w_all, cluster)
    plain_path = weights is None and start is None and se_kind == "greenwood"
    if svy is not None:
        se_kind = "survey_linearization"

    def _table(gdf: pd.DataFrame) -> pd.DataFrame:
        if svy is not None:
            from ._km import km_table_design

            return km_table_design(
                gdf[duration].values.astype(float),
                gdf[event].values,
                gdf[start].values.astype(float) if start is not None else None,
                svy,
                gdf["__svy_row__"].to_numpy(),
                alpha,
                conf_type,
            )
        if plain_path:
            return _km_table(gdf[duration].values, gdf[event].values, alpha, conf_type)
        return km_table_weighted(
            gdf[duration].values.astype(float),
            gdf[event].values,
            gdf[weights].values.astype(float) if weights is not None else None,
            gdf[start].values.astype(float) if start is not None else None,
            gdf[cluster].values if cluster is not None else None,
            se_kind,
            alpha,
            conf_type,
        )

    if group is None:
        tables = {"all": _table(data)}
    else:
        tables = {}
        for g_val, gdf in data.groupby(group):
            tables[str(g_val)] = _table(gdf)

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
                "weights": weights,
                "start": start,
                "cluster": cluster,
                "se_method": se_kind,
                "design": None if design is None else repr(design),
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
    from ._validate import check_duration_event

    check_duration_event(data[duration], data[event], "logrank_test")
    groups = data[group].unique()
    K = len(groups)
    if K < 2:
        raise ValueError("Need at least 2 groups for the log-rank test.")

    T = data[duration].values
    E = data[event].values.astype(float)
    G = data[group].values

    # Counts by (distinct time, group); the number at risk is everyone not
    # yet counted out, a running total down the time axis.
    times, t_idx = np.unique(T, return_inverse=True)
    g_idx = pd.Series(G).map({g: k for k, g in enumerate(groups)}).to_numpy()
    n_t = len(times)
    flat = t_idx * K + g_idx
    count = np.bincount(flat, minlength=n_t * K).reshape(n_t, K).astype(float)
    dead = np.bincount(flat, weights=(E == 1), minlength=n_t * K).reshape(n_t, K)
    risk = count[::-1].cumsum(axis=0)[::-1]
    d_tot = dead.sum(axis=1)
    keep = d_tot > 0
    risk, dead, d_tot = risk[keep], dead[keep], d_tot[keep]
    n_tot = risk.sum(axis=1)

    obs_k = dead.sum(axis=0)
    exp_k = (risk * (d_tot / n_tot)[:, None]).sum(axis=0)
    observed = {g: float(obs_k[k]) for k, g in enumerate(groups)}
    expected = {g: float(exp_k[k]) for k, g in enumerate(groups)}

    # Variance (hypergeometric); a risk set of one contributes nothing
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(
            n_tot > 1, d_tot * (n_tot - d_tot) / (n_tot**2 * (n_tot - 1)), 0.0
        )
    r = risk[:, : K - 1]
    var_mat = (
        np.diag((r * (factor * n_tot)[:, None]).sum(axis=0))
        - (r * factor[:, None]).T @ r
    )

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
    return CoxKernel(X, T, E, strata_arr, breslow=breslow).neg_loglik(beta)


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
    return CoxKernel(X, T, E, strata_arr, breslow=breslow).score_hessian(beta)


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
    return CoxKernel(X, T, E, strata_arr, breslow=breslow).score_residuals(beta)


def _breslow_baseline_hazard(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    strata_arr: Optional[np.ndarray] = None,
    _kernel: Optional[CoxKernel] = None,
) -> pd.DataFrame:
    """Breslow estimator of baseline cumulative hazard."""
    kern = _kernel if _kernel is not None else CoxKernel(X, T, E, strata_arr)
    code, time, inc = kern.breslow_increments(beta)
    # cumulate within stratum: the running total less its value at the
    # start of the stratum
    total = np.cumsum(inc)
    first = np.ones(len(code), dtype=bool)
    first[1:] = code[1:] != code[:-1]
    offset = np.where(first, total - inc, 0.0)
    cumhaz = total - np.maximum.accumulate(offset)
    return pd.DataFrame(
        {
            "time": np.concatenate([[0.0], time]),
            "baseline_cumhaz": np.concatenate([[0.0], cumhaz]),
            "baseline_survival": np.concatenate([[1.0], np.exp(-cumhaz)]),
        }
    )


def _concordance_index(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
) -> float:
    """Harrell's C-statistic for Cox model."""
    return harrell_c(X @ beta, T, E)


def _ph_time_transform(
    transform: Any,
    T: np.ndarray,
    E: np.ndarray,
    at: np.ndarray,
    rank_events_only: bool = False,
    start: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``g(t)`` of the proportional-hazards test, evaluated at ``at``.

    Case weights do not enter ``g`` (R ``cox.zph`` uses the unweighted
    Kaplan-Meier and ranks, checked on its outputs); ``start`` makes the
    Kaplan-Meier one of counting-process data."""
    at = np.asarray(at, dtype=float)
    if callable(transform):
        return np.asarray(transform(at), dtype=float)
    name = str(transform).lower()
    if name == "identity":
        return at
    if name == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(np.log(at), dtype=float)
    if name == "rank":
        # cox.zph ranks the time among all observations; estat phtest
        # among the failures.
        pool = T[E == 1] if rank_events_only else T
        ranks = stats.rankdata(pool)
        lookup = dict(zip(pool.tolist(), ranks.tolist()))
        return np.asarray([lookup[t] for t in at.tolist()], dtype=float)
    if name == "km":
        # 1 - S(t-): the Kaplan-Meier estimate of the pooled sample just
        # before each time.
        times, left = km_left_survival(T, E, start)
        return np.asarray(1.0 - left[np.searchsorted(times, at)], dtype=float)
    raise MethodIncompatibility(
        "ph_test: transform must be 'km', 'rank', 'identity', 'log' or a "
        f"callable, got {transform!r}",
        recovery_hint="Use transform='km' (the default of cox.zph).",
    )


def _schoenfeld_residuals(
    beta: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    strata_arr: Optional[np.ndarray] = None,
    _kernel: Optional[CoxKernel] = None,
) -> np.ndarray:
    """Schoenfeld residuals for PH test."""
    p = X.shape[1]
    kern = _kernel if _kernel is not None else CoxKernel(X, T, E, strata_arr)
    rs = kern.rs
    if not rs.dead.any():
        return np.empty((0, p))
    expected = kern.breslow_expected_x(beta)
    # stratum, ascending time, data order among deaths tied on a time
    rows = np.flatnonzero(rs.dead)
    rows = rows[np.lexsort((rs.order[rows], rs.T[rows], rs.codes[rows]))]
    return np.asarray(kern.X_raw[rows] - expected[rs.group[rows]])


# ===================================================================
# cox()
# ===================================================================


@records_call("cox")
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
    weights: Optional[str] = None,
    start: Optional[str] = None,
    ssc: Optional[str] = None,
    weights_as: Optional[str] = None,
    design: Optional[Any] = None,
    tvc: Optional[Union[str, List[str]]] = None,
    texp: Optional[Any] = None,
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
        ``'hc0'`` (or ``'robust'`` / ``True``) for the sandwich SE of R
        ``coxph(robust = TRUE)``; with ``ssc='stata'`` it carries Stata's
        ``N/(N-1)``. Other values raise (``'cluster'`` only together with
        ``cluster=``).
    cluster : str, optional
        Column name for cluster-robust SE.
    hazard_ratio : bool, default True
        If True, report hazard ratios in the summary alongside coefficients.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    weights : str, optional
        Column of non-negative case weights (inverse-probability-of-
        treatment weights, survey weights, frequency counts). Weighted
        partial likelihood as in R ``coxph(weights=)``: risk sets sum
        ``w_k exp(x_k'b)`` and each Efron term of an event time carries the
        mean weight of its deaths. The model-based variance reads the
        weights as frequencies; for sampling / IPT weights pass
        ``weights_as='sampling'`` (or ``robust='hc0'`` / ``cluster=``),
        which is what R reports by default for non-integer weights and
        Stata ``stcox [pw=]`` always does.
    start : str, optional
        Column with the start of each row's interval: counting-process data
        ``(start, duration]`` as in R ``Surv(start, stop, event)`` and Stata
        ``stset stop, time0(start)`` / ``enter()``. A row is at risk at
        ``t`` iff ``start < t <= duration``, which covers delayed entry
        (left truncation) and time-varying covariates split into intervals.
        With several rows per subject pass ``cluster=<subject id>`` for the
        robust variance by subject (``coxph(cluster=id)``; Stata clusters on
        the ``stset`` id automatically).
    ssc : {'r', 'stata'}, optional
        Small-sample factor of the robust and clustered variances. ``'r'``
        applies none, as R ``coxph(robust=TRUE)`` / ``coxph(cluster=)``.
        ``'stata'`` multiplies the robust variance by ``N/(N-1)`` and the
        clustered one by ``G/(G-1)``, as Stata ``stcox, vce(robust)`` /
        ``vce(cluster)`` (``N`` counts observations, not weights). Not
        given: the robust variance has no factor and the clustered one
        ``G/(G-1)`` (the behaviour before this option existed). The
        model-based variance and the coefficients are unaffected.
    weights_as : {'frequency', 'sampling'}, optional
        How to read ``weights=``. ``'frequency'`` keeps the model-based
        variance; ``'sampling'`` (Stata ``[pw=]``) asks for the robust one
        when neither ``robust=`` nor ``cluster=`` chose a variance.
    design : SurveyDesign, optional
        A complex survey design from :func:`sp.svydesign` declared on
        ``data`` (same rows). The partial likelihood is weighted by the
        design weights and the covariance is ``I^-1 M I^-1`` with ``M`` the
        survey linearisation covariance of the weighted score residuals
        over strata, PSUs and the finite population correction -- R
        ``survey::svycoxph`` (Binder 1992). Inference uses t with the
        design degrees of freedom. With counting-process rows (``start=``)
        the subject's rows must lie in one PSU. Not with ``weights=``,
        ``cluster=``, ``robust=`` or ``ssc=``.
        A replicate-weight design (:func:`sp.svrepdesign`) is accepted
        too: the estimator is refitted with every replicate's weights
        and the replicate variance replaces the linearisation (R
        ``svrepdesign`` / Stata ``svy brr`` / ``svy jackknife``).
    tvc : str or list of str, optional
        Covariates whose log hazard ratio varies with time, as Stata
        ``stcox, tvc(varlist)``: each adds the regressor ``z * g(t)``,
        reported as ``tvc:z`` (Stata's ``tvc`` equation; the other
        coefficients keep their names). A ``tvc`` variable need not be among
        ``x``. Estimated exactly by splitting every row at the failure times
        of its stratum and fitting the counting-process likelihood, as
        ``stcox`` and R ``coxph(tt=)`` do; the robust variance treats each
        input row as the unit. ``ph_test()`` is refused on such fits.
    texp : str or callable, optional
        The function ``g`` of analysis time multiplying the ``tvc``
        covariates, Stata ``texp()``; default ``'_t'``. A string is a Stata
        expression in ``_t`` (``'ln(_t)'``, ``'_t^2'``, ``'_t > 5'``,
        ``'min(_t, 10)'``); a callable maps an array of times to an array.
        It must be finite at every failure time.

    Returns
    -------
    CoxResult
        Result object extending ``EconometricResults`` with
        ``.concordance``, ``.baseline_hazard()``, ``.ph_test()``, ``.plot()``.

    Notes
    -----
    With ``start=`` or ``weights=`` the concordance is computed over the
    risk sets of the fit (pairs in which one row fails while the other is
    at risk, weighted by the product of their weights), as R
    ``concordance()`` does for counting-process and weighted fits; Harrell's
    pairwise C of right-censored data is the special case without either.

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
    if texp is not None and tvc is None:
        raise MethodIncompatibility(
            "sp.cox: texp= requires tvc= (Stata: texp() only with tvc()).",
            recovery_hint="Pass tvc=<covariates> together with texp=, or drop texp=.",
        )
    if tvc is not None and design is not None:
        raise MethodIncompatibility(
            "sp.cox: tvc= cannot be combined with design=.",
            recovery_hint="Fit the time-varying coefficients with weights=<design "
            "weight> and cluster=<PSU> instead, or split the data with start= and "
            "build the z*g(t) column yourself.",
        )
    if design is not None and getattr(design, "is_replicate_design", False):
        _args = dict(locals())
        from ..survey._replicate_refit import replicate_refit

        return replicate_refit(cox, _args, design, function="sp.cox")
    svy: Any = None
    if design is not None:
        from ..survey._regression_design import refuse_design_options
        from ..survey.linearization import check_design

        refuse_design_options("cox", robust=robust, ssc=ssc, weights_as=weights_as)
        svy = check_design(
            design, data, function="cox", weights=weights, cluster=cluster
        )
        data = data.assign(__svy_row__=np.arange(len(data)), __svy_weight__=svy.weights)
        weights = "__svy_weight__"
        weights_as = "sampling"

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
    if weights is not None:
        cols_needed.append(weights)
    if start is not None:
        cols_needed.append(start)
    tvc_cols: Optional[List[str]] = None
    if tvc is not None:
        from ._cox_tvc import resolve_tvc

        tvc_cols = resolve_tvc(tvc, data)
        cols_needed += [c for c in tvc_cols if c not in cols_needed]
    from ._validate import check_duration_event, warn_missing_weights

    warn_missing_weights(data, weights, "sp.cox")
    data = data.dropna(subset=cols_needed).copy()
    check_duration_event(data[duration], data[event], "sp.cox")

    from ..core._agent_summary import resolve_weights_as
    from ._cox_inputs import check_cox_inputs, resolve_cox_ssc

    weights_as = resolve_weights_as(
        weights_as,
        has_weights=weights is not None,
        native="frequency",
        entry="sp.cox",
    )
    ssc = resolve_cox_ssc(ssc)
    if robust is True:
        robust = "hc0"  # R coxph(robust = TRUE)
    elif robust is False:
        robust = "nonrobust"
    if robust is not None and not (
        str(robust).lower() in ("nonrobust", "hc0", "robust")
        or (str(robust).lower() == "cluster" and cluster is not None)
    ):
        # Until 1.40.2 any other value ('hc1', 'HC3', 'cluster' without
        # cluster=) silently gave the model-based variance.
        raise MethodIncompatibility(
            f"sp.cox: robust={robust!r} is not available.",
            recovery_hint=(
                "Use robust='hc0' (R coxph(robust=TRUE)); add ssc='stata' for "
                "Stata's stcox, vce(robust) (the HC1-type N/(N-1)); pass "
                "cluster=<column> for the clustered variance."
            ),
            diagnostics={"robust": robust},
        )
    if weights_as == "sampling" and cluster is None and robust in ("nonrobust", None):
        robust = "hc0"

    T = data[duration].values.astype(float)
    E = data[event].values.astype(float)
    X = data[x].values.astype(float)
    n, p = X.shape
    w_arr = data[weights].values.astype(float) if weights is not None else None
    start_arr = data[start].values.astype(float) if start is not None else None
    w_arr, keep = check_cox_inputs(
        T,
        E,
        w_arr,
        start_arr,
        robust=robust,
        cluster=cluster,
        weights_as=weights_as,
    )
    if keep is not None:
        # zero weights carry no information: drop them (coxph does too)
        data = data.iloc[np.flatnonzero(keep)]
        T, E, X = T[keep], E[keep], X[keep]
        w_arr = w_arr[keep]
        if start_arr is not None:
            start_arr = start_arr[keep]
        n = len(T)

    strata_arr = data[strata].values if strata else None
    cluster_arr = data[cluster].values if cluster else None

    n_input = n
    tvc_origin: Optional[np.ndarray] = None
    texp_label: Optional[str] = None
    if tvc_cols is not None:
        from ._cox_tvc import resolve_texp, split_at_failures, tvc_labels

        g_fun, texp_label = resolve_texp(texp)
        split = split_at_failures(T, E, start_arr, strata_arr, g_fun, texp_label)
        tvc_origin = split.origin
        Z = data[tvc_cols].values.astype(float)[tvc_origin] * split.g[:, None]
        X = np.column_stack([X[tvc_origin], Z])
        x = list(x) + tvc_labels(tvc_cols)
        T, E, start_arr = split.stop, split.event, split.start
        if w_arr is not None:
            w_arr = w_arr[tvc_origin]
        if strata_arr is not None:
            strata_arr = strata_arr[tvc_origin]
        if cluster_arr is not None:
            cluster_arr = cluster_arr[tvc_origin]
        n, p = X.shape

    # ---- Optimization (Newton-Raphson) --------------------------------
    beta0 = np.zeros(p)

    if ties not in ("efron", "breslow"):
        raise ValueError(f"ties must be 'efron' or 'breslow', got {ties!r}")

    breslow = ties == "breslow"

    # The risk sets are sorted once and shared by every evaluation below.
    kern = CoxKernel(
        X, T, E, strata_arr, breslow=breslow, weights=w_arr, start=start_arr
    )
    neg_logpl = kern.neg_loglik

    # Newton-Raphson with fallback to L-BFGS-B
    beta = beta0.copy()
    converged = False
    for iteration in range(50):
        score, hessian = kern.score_hessian(beta)
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

        # Converged when the Newton step itself is below 1e-9; that step is
        # not taken, which is where R coxph stops too (the R parity of the
        # weighted / counting-process fits is to ~1e-15 on this rule).
        if np.max(np.abs(step)) < 1e-9:
            converged = True
            break

        # Step-halving line search. A larger step may still leave the
        # likelihood flat to rounding: near the optimum a Newton step of
        # 1e-7 changes it by ~1e-12. Requiring a decrease of 1e-10 used to
        # exhaust the search on such a step, and the 2^-20-damped step then
        # passed the convergence test, so the fit stopped one Newton step
        # short (up to ~1e-6 on the coefficients, depending on where the
        # iterates fell).
        step_size = 1.0
        current_nll = neg_logpl(beta)
        slack = 1e-12 * max(1.0, abs(current_nll))
        for _ in range(20):
            candidate = beta + step_size * step
            if neg_logpl(candidate) <= current_nll + slack:
                break
            step_size *= 0.5
        beta = beta + step_size * step

    if not converged:
        # Fallback to scipy optimizer
        result = optimize.minimize(neg_logpl, beta, method="L-BFGS-B")
        beta = result.x

    # Monotone partial likelihood: a covariate that separates the failures
    # (the Cox analogue of separation in logit) sends its coefficient to
    # +/- infinity while the likelihood flattens. Newton then either fails
    # to converge or "converges" through the regularised step once the
    # information has vanished. Until 1.40.2 both were silent; R coxph
    # warns "coefficient may be infinite". The check compares each
    # coefficient's information at the estimate with its information at
    # zero (scale-free); a collapse by eight orders of magnitude is not
    # reached by an interior maximum.
    convergence_note: Optional[str] = None
    _, _hess_hat = kern.score_hessian(beta)
    _score_zero, _hess_zero = kern.score_hessian(np.zeros(p))
    _info_hat = -np.diag(_hess_hat)
    _info_zero = -np.diag(_hess_zero)
    runaway = [
        x[j]
        for j in range(p)
        if _info_zero[j] > 0 and _info_hat[j] < 1e-8 * _info_zero[j]
    ]
    if runaway or not converged:
        convergence_note = (
            (
                f"the partial likelihood is monotone in {runaway}: the "
                "coefficient may be infinite (a covariate separates the "
                "failures). Estimates and standard errors are not "
                "meaningful for it."
            )
            if runaway
            else (
                "Newton-Raphson did not converge in 50 iterations; the "
                "estimates come from an L-BFGS-B fallback."
            )
        )
        import warnings

        from ..exceptions import ConvergenceWarning

        warnings.warn("sp.cox: " + convergence_note, ConvergenceWarning, stacklevel=2)

    # ---- Variance estimation ------------------------------------------
    hessian = _hess_hat
    neg_H = -hessian
    try:
        info_inv = np.linalg.inv(neg_H)
    except np.linalg.LinAlgError:
        info_inv = np.linalg.pinv(neg_H)

    svy_df: Optional[float] = None
    if svy is not None:
        from ..survey.linearization import design_df, design_vcov

        assert w_arr is not None
        svy_rows = data["__svy_row__"].to_numpy()
        score_i = kern.score_residuals(beta) * w_arr[:, None]
        var_beta = info_inv @ design_vcov(score_i, svy, svy_rows) @ info_inv
        svy_df = design_df(svy, svy_rows)
    elif cluster is not None or str(robust).lower() in ("hc0", "robust"):
        from ._cox_inputs import cox_robust_vcov

        score_i = kern.score_residuals(beta)
        if w_arr is not None:
            score_i = score_i * w_arr[:, None]
        if tvc_origin is not None and cluster_arr is None:
            # the unit of the robust variance is the input row, not the
            # episode: sum each row's pieces (N in N/(N-1) counts rows)
            by_row = np.zeros((n_input, score_i.shape[1]))
            np.add.at(by_row, tvc_origin, score_i)
            score_i = by_row
        var_beta = cox_robust_vcov(info_inv, score_i, cluster_arr, ssc)
    else:
        var_beta = info_inv

    se = np.sqrt(np.diag(var_beta))

    # ---- Derived quantities -------------------------------------------
    loglik = -neg_logpl(beta)
    loglik0 = -neg_logpl(np.zeros(p))
    bh_df = _breslow_baseline_hazard(beta, X, T, E, strata_arr, _kernel=kern)
    if w_arr is None and start_arr is None:
        c_index = _concordance_index(beta, X, T, E)
    else:
        from ._cox_inputs import risk_set_concordance

        c_index = risk_set_concordance(X @ beta, T, E, start_arr, w_arr, strata_arr)
    schoenfeld = _schoenfeld_residuals(beta, X, T, E, strata_arr, _kernel=kern)

    # Hazard ratios
    hr = np.exp(beta)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    if svy_df is not None:
        z_crit = stats.t.ppf(1 - alpha / 2, svy_df)
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
    if weights is not None:
        model_info["weights"] = weights
        model_info["weights_as"] = weights_as or "frequency"
    if start is not None:
        model_info["start"] = start
    if ssc is not None:
        model_info["ssc"] = ssc
    if tvc_cols is not None:
        model_info["tvc"] = list(tvc_cols)
        model_info["texp"] = texp_label
    if svy is not None:
        from ..survey.linearization import design_info

        assert svy_df is not None
        model_info["robust"] = "survey"
        model_info["weights"] = None
        model_info["weights_as"] = None
        model_info["vcov_type"] = "Survey linearisation (R svycoxph)"
        model_info["survey_design"] = design_info(svy, svy_df)

    data_info = {
        "nobs": n_input,
        "n_events": int(E.sum()),
        "dependent_var": duration,
        "event_var": event,
        "df_resid": np.inf,  # use normal distribution for Cox
        "n_clusters": (
            int(pd.Series(cluster_arr).nunique()) if cluster is not None else None
        ),
    }
    if svy_df is not None:
        data_info["df_inference"] = svy_df
    if tvc_origin is not None:
        data_info["n_episodes"] = n

    # The other two tests of beta = 0 that survival::coxph prints. The Wald
    # statistic uses the reported covariance (robust when one was asked
    # for); the score statistic is evaluated at zero and, without
    # covariates beyond a group indicator, is the log-rank test.
    try:
        wald = float(beta @ np.linalg.solve(var_beta, beta))
    except np.linalg.LinAlgError:
        wald = float("nan")
    score0, hess0 = _score_zero, _hess_zero
    try:
        score_stat = float(score0 @ np.linalg.solve(-hess0, score0))
    except np.linalg.LinAlgError:
        score_stat = float("nan")

    diagnostics = {
        "Log-likelihood": loglik,
        "Log-likelihood (null)": loglik0,
        "LR chi2": 2 * (loglik - loglik0),
        "LR chi2 p-value": stats.chi2.sf(2 * (loglik - loglik0), df=p),
        "Wald chi2": wald,
        "Wald chi2 p-value": float(stats.chi2.sf(wald, df=p)),
        "Score chi2": score_stat,
        "Score chi2 p-value": float(stats.chi2.sf(score_stat, df=p)),
        "Concordance (C)": c_index,
        "AIC": -2 * loglik + 2 * p,
        "BIC": -2 * loglik + np.log(E.sum()) * p,
    }
    if w_arr is not None:
        # the model-based Wald / score / LR tests assume frequency weights
        diagnostics["Sum of weights"] = float(w_arr.sum())
    if convergence_note is not None:
        diagnostics["Converged"] = False
        model_info["convergence_warning"] = convergence_note

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
    _result._weights = w_arr
    _result._start = start_arr
    # Store the global proportional-hazards test (from the Schoenfeld residuals
    # computed above) so result.violations() can flag a PH violation without
    # recomputing it — best-effort: strata models don't expose the residuals.
    try:
        _ph = _result.ph_test()
        _ph = _ph[_ph["variable"] != "GLOBAL"].reset_index(drop=True)
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
                "weights": weights,
                "start": start,
                "ssc": ssc,
                "weights_as": weights_as,
                "design": None if design is None else repr(design),
                "tvc": tvc_cols,
                "texp": texp_label,
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


@records_call("survreg")
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
        differ by the sum of ``log(t)`` over the events), and what
        :func:`sp.aft` reports for the same fit. ``model_info
        ['ll_scale']`` names the convention of ``'Log-likelihood'``.

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
    if np.any(T <= 0):
        raise MethodIncompatibility(
            f"survreg: {int(np.sum(T <= 0))} durations are zero or negative; "
            "the model is for the log of a positive duration. Drop those "
            "rows or shift the time origin.",
            diagnostics={"n_nonpositive": int(np.sum(T <= 0))},
        )
    from ._validate import check_duration_event

    # 0/1 only, as sp.cox / sp.kaplan_meier: until 1.40.2 any positive code
    # was read as an event (Stata's rule, the opposite of R's 1/2 coding)
    check_duration_event(data[duration], data[event], "survreg")
    E = data[event].values.astype(float)
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
        "ll_scale": "duration t (R survreg); ll_log_time is Stata streg's",
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
