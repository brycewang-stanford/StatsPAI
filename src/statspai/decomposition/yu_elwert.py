"""
Yu-Elwert (2025) nonparametric causal decomposition of group disparities.

Decomposes the disparity ``D = E[Y | R=1] - E[Y | R=0]`` between an
advantaged group (``R=1``) and a reference group (``R=0``) into four
mechanisms operating through a binary treatment ``T``:

.. math::

    D = \\underbrace{(E[Y(0)|R{=}1] - E[Y(0)|R{=}0])}_{\\text{baseline}}
      + \\underbrace{E_0[\\tau](E[T|R{=}1] - E[T|R{=}0])}_{\\text{prevalence}}
      + \\underbrace{E[T|R{=}1](E_1[\\tau] - E_0[\\tau])}_{\\text{effect}}
      + \\underbrace{\\operatorname{Cov}_1(T, \\tau) - \\operatorname{Cov}_0(T, \\tau)}_{\\text{selection}}

where ``τ_i = Y_i(1) - Y_i(0)`` is the individual treatment effect and
``E_r[·]`` denotes expectation conditional on ``R=r``. Identification
requires *only* conditional ignorability of ``T`` given ``(R, X)``
(no assumption that ``R`` itself is exogenous), which sets this
decomposition apart from causal-mediation approaches.

The selection component is the key novel piece: it captures *whether
the right people end up treated* — i.e. group-specific covariance
between assignment and individual-level effect heterogeneity.

The default estimator is plug-in:

* within-group, within-treatment OLS for ``m_rt(X) = E[Y|R=r,T=t,X]``;
* within-group logit for ``p_r(X) = P(T=1|R=r,X)``;
* per-component plug-in expectations.

Standard errors come from the non-parametric (cluster-aware)
bootstrap. A doubly-robust ``method="efficient"`` mode swaps the
plug-in moments for augmented (DR) moments evaluated on the same
nuisance fits; users wanting flexible ML nuisances can wrap their own
fits and call the underlying ``_components_from_nuisance`` directly.

References
----------
Yu, A. & Elwert, F. (2025). Nonparametric causal decomposition of
group disparities. *Annals of Applied Statistics*, 19(1), 821-845.
doi:10.1214/24-AOAS1990. R implementation: ``cdgd``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._common import (
    add_constant,
    bootstrap_ci,
    bootstrap_stat,
    logit_fit,
    logit_predict,
    wls,
)
from ._results import DecompResultMixin

# ════════════════════════════════════════════════════════════════════════
# Result class
# ════════════════════════════════════════════════════════════════════════


@dataclass
class YuElwertResult(DecompResultMixin):
    """Yu-Elwert (2025) causal decomposition of a group disparity.

    Attributes
    ----------
    disparity : float
        Observed gap ``E[Y|R=1] - E[Y|R=0]``.
    baseline : float
        Counterfactual disparity if no one were treated.
    prevalence : float
        Contribution of differential treatment uptake (group A vs. B),
        scaled by the reference group's average treatment effect.
    effect : float
        Contribution of group heterogeneity in average treatment effects,
        scaled by the advantaged group's treatment prevalence.
    selection : float
        Group-specific covariance between treatment assignment and
        individual-level effect heterogeneity — the signature mechanism
        of Yu-Elwert.
    se : dict[str, float] | None
        Standard errors keyed by component (``disparity``, ``baseline``,
        ``prevalence``, ``effect``, ``selection``).
    ci : dict[str, (float, float)] | None
        Two-sided 95% confidence intervals (matching ``alpha`` argument).
    detailed : pandas.DataFrame
        Tidy table of (component, value, se, ci_low, ci_high).
    nuisance : dict
        Diagnostic snapshot — group sizes, fitted per-cell means,
        plus ``fallback_cell_count`` and ``bootstrap_failure_count``
        when applicable so the user can audit a degenerate run.
    method : str
        ``"plugin"`` or ``"efficient"``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> r = rng.integers(0, 2, size=n)
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> p = 1 / (1 + np.exp(-(0.5 * r + 0.3 * x1)))
    >>> t = (rng.uniform(size=n) < p).astype(int)
    >>> y = (1.0 + 0.5 * x1 + 0.3 * x2 + (0.8 + 0.4 * r) * t
    ...      + rng.normal(size=n))
    >>> df = pd.DataFrame({"y": y, "t": t, "r": r, "x1": x1, "x2": x2})
    >>> res = sp.yu_elwert_decompose(
    ...     df, y="y", treatment="t", group="r", x=["x1", "x2"],
    ...     inference="none",
    ... )
    >>> type(res).__name__
    'YuElwertResult'
    >>> res.method
    'plugin'
    """

    method_name: ClassVar[str] = "Yu-Elwert (2025) Causal Decomposition"
    bib_keys: ClassVar[Tuple[str, ...]] = ("yu2025nonparametric",)

    disparity: float = 0.0
    baseline: float = 0.0
    prevalence: float = 0.0
    effect: float = 0.0
    selection: float = 0.0
    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None
    detailed: pd.DataFrame = field(default_factory=pd.DataFrame)
    nuisance: Dict[str, Any] = field(default_factory=dict)
    method: str = "plugin"

    # ── Pretty text summary ──────────────────────────────────────────

    def summary(self) -> str:
        from ._common import sig_stars

        def fmt(name: str, val: float, se_key: str) -> str:
            line = f"  {name:<22s}{val:>10.4f}"
            if self.se and se_key in self.se and self.se[se_key] > 0:
                se = self.se[se_key]
                z = abs(val) / se if se > 0 else 0.0
                from scipy.stats import norm

                pval = 2 * (1 - norm.cdf(z))
                line += f"   SE={se:.4f}{sig_stars(pval):<3s}"
                if self.ci and se_key in self.ci:
                    lo, hi = self.ci[se_key]
                    line += f"   95% CI [{lo: .4f}, {hi: .4f}]"
            return line

        residual = (
            self.disparity
            - self.baseline
            - self.prevalence
            - self.effect
            - self.selection
        )
        lines = [
            "━" * 70,
            "  Yu-Elwert (2025) Nonparametric Causal Decomposition",
            "━" * 70,
            f"  Method: {self.method}    "
            f"N_a = {self.nuisance.get('n_a', '?')}    "
            f"N_b = {self.nuisance.get('n_b', '?')}",
            "",
            fmt("Observed disparity:", self.disparity, "disparity"),
            "  " + "─" * 66,
            fmt("Baseline:", self.baseline, "baseline"),
            fmt("Prevalence:", self.prevalence, "prevalence"),
            fmt("Effect:", self.effect, "effect"),
            fmt("Selection:", self.selection, "selection"),
            "  " + "─" * 66,
            f"  {'Implied residual:':<22s}{residual:>10.4f}"
            f"   (= disparity − Σ components)",
            "━" * 70,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    # ── Plot delegate ────────────────────────────────────────────────

    def plot(self, **kwargs: Any) -> Any:
        from .plots import yu_elwert_mechanisms_plot

        return yu_elwert_mechanisms_plot(self, **kwargs)

    # ── confint override (the mixin reads from `overall`, but
    #    YuElwertResult uses flat scalar fields plus an `se` dict.) ──

    def confint(  # type: ignore[override]
        self, alpha: float = 0.05, which: str = "overall"
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        if which == "detailed":
            return super().confint(alpha=alpha, which="detailed")
        if not self.se:
            return None
        from scipy.stats import norm

        z = float(norm.ppf(1 - alpha / 2))
        out: Dict[str, Tuple[float, float]] = {}
        for key in ("disparity", "baseline", "prevalence", "effect", "selection"):
            v = getattr(self, key)
            s = self.se.get(key, 0.0)
            if s and s > 0:
                out[key] = (v - z * s, v + z * s)
        return out or None

    # ── LaTeX ────────────────────────────────────────────────────────

    def to_latex(self) -> str:
        rows = [
            ("Disparity", self.disparity, "disparity"),
            ("Baseline", self.baseline, "baseline"),
            ("Prevalence", self.prevalence, "prevalence"),
            ("Effect", self.effect, "effect"),
            ("Selection", self.selection, "selection"),
        ]
        out = [
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Component & Estimate & Std.\ Error \\",
            r"\midrule",
        ]
        for label, val, key in rows:
            se = (self.se or {}).get(key, None)
            se_str = f"{se:.4f}" if se is not None else ""
            out.append(f"{label} & {val:.4f} & {se_str} \\\\")
        out += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(out)

    def _repr_html_(self) -> str:  # pragma: no cover - cosmetic
        rows = "".join(
            f"<tr><td>{label}</td><td>{val:.4f}</td></tr>"
            for label, val in [
                ("Disparity", self.disparity),
                ("Baseline", self.baseline),
                ("Prevalence", self.prevalence),
                ("Effect", self.effect),
                ("Selection", self.selection),
            ]
        )
        return (
            "<div style='font-family: monospace;'>"
            "<h3>Yu-Elwert (2025) Causal Decomposition</h3>"
            f"<table>{rows}</table></div>"
        )


# ════════════════════════════════════════════════════════════════════════
# Internal: plug-in nuisances
# ════════════════════════════════════════════════════════════════════════


def _fit_within_cell_outcome(
    y: np.ndarray,
    X: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], int]:
    """Fit four within-cell OLS regressions m_rt(X).

    Returns the coefficient dict plus the number of cells that had to
    use a constant fallback (so the caller can audit degeneracy).
    """
    out: Dict[Tuple[int, int], np.ndarray] = {}
    fallback_count = 0
    k = X.shape[1]
    for r_val in (0, 1):
        for t_val in (0, 1):
            mask = (r == r_val) & (t == t_val)
            if mask.sum() < k + 1:
                grp_mask = r == r_val
                fallback = np.zeros(k)
                fallback[0] = (
                    float(y[mask].mean())
                    if mask.any()
                    else float(y[grp_mask].mean()) if grp_mask.any() else 0.0
                )
                out[(r_val, t_val)] = fallback
                fallback_count += 1
                continue
            beta, _, _ = wls(y[mask], X[mask])
            out[(r_val, t_val)] = beta
    return out, fallback_count


def _fit_within_group_propensity(
    t: np.ndarray,
    X: np.ndarray,
    r: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Fit p_r(X) = P(T=1|R=r,X) via within-group logit."""
    out: Dict[int, np.ndarray] = {}
    for r_val in (0, 1):
        mask = r == r_val
        if mask.sum() < X.shape[1] + 1:
            out[r_val] = np.zeros(X.shape[1])
            out[r_val][0] = float(t[mask].mean()) if mask.any() else 0.5
            continue
        beta, _ = logit_fit(t[mask], X[mask], warn_on_nonconvergence=False)
        out[r_val] = beta
    return out


def _components_from_nuisance(
    y: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    X: np.ndarray,
    m_coef: Dict[Tuple[int, int], np.ndarray],
    p_coef: Dict[int, np.ndarray],
    *,
    efficient: bool = False,
    trim: float = 0.005,
) -> Dict[str, float]:
    """Compute the four Yu-Elwert components on a (sub-)sample.

    With ``efficient=True`` the per-component statistics use augmented
    (doubly-robust) terms; with ``efficient=False`` (default) we fall
    back on the plug-in formulas in the paper's Section 4.1.
    """
    grp_a = r == 1
    grp_b = r == 0

    def m(rv: int, tv: int) -> np.ndarray:
        return np.asarray(X @ m_coef[(rv, tv)])

    def p(rv: int) -> np.ndarray:
        return logit_predict(p_coef[rv], X)

    # Within-group treatment prevalences and average outcomes
    eD_a = float(t[grp_a].mean())
    eD_b = float(t[grp_b].mean())
    eY_a = float(y[grp_a].mean())
    eY_b = float(y[grp_b].mean())

    m_a0_all = m(1, 0)
    m_a1_all = m(1, 1)
    m_b0_all = m(0, 0)
    m_b1_all = m(0, 1)

    if efficient:
        # Augmented (doubly-robust) outcome surfaces. Compute psi_r only
        # on the relevant subgroup so that the t / p_r and (1-t)/(1-p_r)
        # terms are evaluated on the population whose propensity p_r was
        # actually fit on (avoids a foot-gun if anyone reuses tau_*_per_obs
        # outside its own subgroup).
        p_a_in = np.clip(p(1)[grp_a], trim, 1 - trim)
        p_b_in = np.clip(p(0)[grp_b], trim, 1 - trim)
        y_a, t_a = y[grp_a], t[grp_a]
        y_b, t_b = y[grp_b], t[grp_b]
        m_a0_in, m_a1_in = m_a0_all[grp_a], m_a1_all[grp_a]
        m_b0_in, m_b1_in = m_b0_all[grp_b], m_b1_all[grp_b]

        ey0_a = float(np.mean(m_a0_in + (1 - t_a) / (1 - p_a_in) * (y_a - m_a0_in)))
        ey0_b = float(np.mean(m_b0_in + (1 - t_b) / (1 - p_b_in) * (y_b - m_b0_in)))
        psi_a = (
            m_a1_in
            - m_a0_in
            + t_a / p_a_in * (y_a - m_a1_in)
            - (1 - t_a) / (1 - p_a_in) * (y_a - m_a0_in)
        )
        psi_b = (
            m_b1_in
            - m_b0_in
            + t_b / p_b_in * (y_b - m_b1_in)
            - (1 - t_b) / (1 - p_b_in) * (y_b - m_b0_in)
        )
        # Place each subgroup's psi back into a same-length-as-y array,
        # leaving the complementary positions as NaN (they are never
        # touched by downstream code; this just makes the contract clear).
        tau_a_per_obs = np.full_like(y, np.nan, dtype=float)
        tau_a_per_obs[grp_a] = psi_a
        tau_b_per_obs = np.full_like(y, np.nan, dtype=float)
        tau_b_per_obs[grp_b] = psi_b
        e_tau_a = float(psi_a.mean())
        e_tau_b = float(psi_b.mean())
    else:
        ey0_a = float(m_a0_all[grp_a].mean())
        ey0_b = float(m_b0_all[grp_b].mean())
        tau_a_per_obs = m_a1_all - m_a0_all
        tau_b_per_obs = m_b1_all - m_b0_all
        e_tau_a = float(tau_a_per_obs[grp_a].mean())
        e_tau_b = float(tau_b_per_obs[grp_b].mean())

    baseline = ey0_a - ey0_b
    prevalence = e_tau_b * (eD_a - eD_b)
    effect = eD_a * (e_tau_a - e_tau_b)

    # Selection: Cov_a(T, τ_a) − Cov_b(T, τ_b)
    cov_a = float(np.mean(t[grp_a] * tau_a_per_obs[grp_a]) - eD_a * e_tau_a)
    cov_b = float(np.mean(t[grp_b] * tau_b_per_obs[grp_b]) - eD_b * e_tau_b)
    selection = cov_a - cov_b

    disparity = eY_a - eY_b
    return dict(
        disparity=disparity,
        baseline=baseline,
        prevalence=prevalence,
        effect=effect,
        selection=selection,
    )


# ════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════


def yu_elwert_decompose(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    group: str,
    x: Sequence[str],
    *,
    method: str = "plugin",
    inference: str = "bootstrap",
    n_boot: int = 499,
    alpha: float = 0.05,
    trim: float = 0.005,
    cluster: Optional[str] = None,
    seed: Optional[int] = 12345,
) -> YuElwertResult:
    """Nonparametric causal decomposition of a group disparity.

    Parameters
    ----------
    data : DataFrame
        Long-format panel with one row per observation.
    y : str
        Name of the (continuous) outcome column.
    treatment : str
        Binary treatment indicator (0/1).
    group : str
        Binary group indicator (0/1) — ``1`` = advantaged / index group.
    x : sequence of str
        Adjustment covariates (used to identify within-group CATEs).
    method : {"plugin", "efficient"}
        ``"plugin"`` uses within-cell OLS for outcomes and within-group
        logit for the propensity and computes plug-in expectations
        (Yu-Elwert 2025, Section 4.1). ``"efficient"`` augments each
        moment with the doubly-robust correction term — recommended
        when nuisance functions might be misspecified.
    inference : {"bootstrap", "none"}
        ``"bootstrap"`` returns SEs and percentile CIs from the
        non-parametric (cluster-aware) bootstrap. ``"none"`` skips
        inference.
    n_boot : int
    alpha : float
        Two-sided coverage level.
    trim : float
        Lower/upper clip for fitted propensities (only used in
        ``method="efficient"``).
    cluster : str or None
        Column name to use for cluster bootstrap.
    seed : int or None

    Returns
    -------
    YuElwertResult

    Notes
    -----
    Identification requires conditional ignorability of treatment given
    ``(R, X)`` (no unmeasured confounders within group). The framework
    does *not* require ``R`` itself to be unconfounded, distinguishing
    it from causal-mediation approaches.

    The "selection" component is zero whenever individuals are randomly
    assigned to treatment (no selection on individual gain) or whenever
    the CATE is constant within group (no heterogeneity to select on).
    A non-zero selection term — particularly one of opposite sign in
    the two groups — flags that targeting differs systematically across
    groups, often the lever a designer can pull.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> r = rng.integers(0, 2, size=n)
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> p = 1 / (1 + np.exp(-(0.5 * r + 0.3 * x1)))
    >>> t = (rng.uniform(size=n) < p).astype(int)
    >>> y = (1.0 + 0.5 * x1 + 0.3 * x2 + (0.8 + 0.4 * r) * t
    ...      + rng.normal(size=n))
    >>> df = pd.DataFrame({"y": y, "t": t, "r": r, "x1": x1, "x2": x2})
    >>> res = sp.decompose(
    ...     "yu_elwert", data=df, y="y", treatment="t", group="r",
    ...     x=["x1", "x2"], inference="none",
    ... )
    >>> type(res).__name__
    'YuElwertResult'
    >>> res.method
    'plugin'

    References
    ----------
    yu2025nonparametric
    """
    if method not in ("plugin", "efficient"):
        raise ValueError(f"method must be 'plugin' or 'efficient', got {method!r}")
    if inference not in ("bootstrap", "none"):
        raise ValueError("inference must be 'bootstrap' or 'none'")

    cols = [y, treatment, group] + list(x) + ([cluster] if cluster else [])
    df = data[cols].dropna().copy()
    if df.empty:
        raise ValueError("No complete observations after dropping NA.")

    y_arr = df[y].to_numpy(dtype=float)
    t_arr = df[treatment].to_numpy(dtype=float)
    r_arr = df[group].to_numpy(dtype=float)
    if not np.all(np.isin(t_arr, (0.0, 1.0))):
        raise ValueError(f"treatment {treatment!r} must be binary 0/1")
    if not np.all(np.isin(r_arr, (0.0, 1.0))):
        raise ValueError(f"group {group!r} must be binary 0/1")
    X = add_constant(df[list(x)].to_numpy(dtype=float))
    cluster_arr = df[cluster].to_numpy() if cluster else None

    fallback_total = [0]  # mutable counter shared across closures

    def _point(idx: Optional[np.ndarray] = None) -> Dict[str, float]:
        if idx is None:
            y_, t_, r_, X_ = y_arr, t_arr, r_arr, X
        else:
            y_, t_, r_, X_ = y_arr[idx], t_arr[idx], r_arr[idx], X[idx]
        m_coef, fb = _fit_within_cell_outcome(y_, X_, t_, r_)
        fallback_total[0] += fb
        p_coef = _fit_within_group_propensity(t_, X_, r_)
        return _components_from_nuisance(
            y_,
            t_,
            r_,
            X_,
            m_coef,
            p_coef,
            efficient=(method == "efficient"),
            trim=trim,
        )

    point = _point(None)
    n_a = int((r_arr == 1).sum())
    n_b = int((r_arr == 0).sum())
    point_fallback = fallback_total[0]
    fallback_total[0] = 0  # reset for bootstrap accounting

    se: Optional[Dict[str, float]] = None
    ci: Optional[Dict[str, Tuple[float, float]]] = None
    boot_failures = 0
    if inference == "bootstrap":
        keys = ("disparity", "baseline", "prevalence", "effect", "selection")

        # Surface failures explicitly: bootstrap_stat already counts
        # exceptions and emits a RuntimeWarning if > 5%, so just let
        # _point exceptions propagate up to it.
        def stat_fn(idx: np.ndarray) -> np.ndarray:
            pt = _point(idx)
            return np.array([pt[k] for k in keys])

        rng = np.random.default_rng(seed)
        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            boot = bootstrap_stat(
                stat_fn,
                n=len(y_arr),
                n_boot=n_boot,
                rng=rng,
                clusters=cluster_arr,
            )
        # Re-emit the warning *and* extract the failure count if present.
        for w in caught:
            _warnings.warn(w.message, w.category, stacklevel=2)
            msg = str(w.message)
            if "bootstrap replications failed" in msg:
                # Pattern: "{n}/{n_boot} bootstrap replications failed"
                try:
                    boot_failures = int(msg.split("/")[0].split()[-1])
                except (ValueError, IndexError):  # pragma: no cover
                    pass
        if len(boot) == 0:
            raise RuntimeError("All bootstrap replications failed.")  # pragma: no cover
        point_arr = np.array([point[k] for k in keys])
        ses, los, his = bootstrap_ci(
            boot,
            point_arr,
            alpha=alpha,
            method="percentile",
        )
        se = {k: float(ses[i]) for i, k in enumerate(keys)}
        ci = {k: (float(los[i]), float(his[i])) for i, k in enumerate(keys)}

    detailed = pd.DataFrame(
        [
            dict(
                component=k.title(),
                value=point[k],
                se=(se or {}).get(k, np.nan),
                ci_low=(ci or {}).get(k, (np.nan, np.nan))[0],
                ci_high=(ci or {}).get(k, (np.nan, np.nan))[1],
            )
            for k in ("disparity", "baseline", "prevalence", "effect", "selection")
        ]
    )

    nuisance = dict(
        n_a=n_a,
        n_b=n_b,
        n_treated_a=int(((r_arr == 1) & (t_arr == 1)).sum()),
        n_treated_b=int(((r_arr == 0) & (t_arr == 1)).sum()),
        e_t_a=float(t_arr[r_arr == 1].mean()) if n_a else float("nan"),
        e_t_b=float(t_arr[r_arr == 0].mean()) if n_b else float("nan"),
        fallback_cell_count=point_fallback,
        bootstrap_failure_count=boot_failures,
    )

    return YuElwertResult(
        disparity=point["disparity"],
        baseline=point["baseline"],
        prevalence=point["prevalence"],
        effect=point["effect"],
        selection=point["selection"],
        se=se,
        ci=ci,
        detailed=detailed,
        nuisance=nuisance,
        method=method,
    )
