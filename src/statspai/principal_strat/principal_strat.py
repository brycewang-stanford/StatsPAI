"""
Principal Stratification estimators.

Setting
-------
Treatment :math:`D \\in \\{0, 1\\}`, post-treatment variable
:math:`S \\in \\{0, 1\\}` (often compliance or survival), outcome
:math:`Y`. Principal strata classify units by the pair
:math:`(S(0), S(1))`:

* **always-taker / always-survivor**: (1, 1)
* **complier / harmed**: (0, 1)
* **defier / helped**: (1, 0)   — usually ruled out by monotonicity
* **never-taker / dead-under-both**: (0, 0)

Under Angrist-Imbens-Rubin (AIR) monotonicity (:math:`S(1) \\ge
S(0)` a.s., no defiers), the three remaining strata have
**observable** mixture decompositions that yield sharp bounds
or point estimates on stratum-specific causal effects
(principal causal effects, PCEs).

Two methods are supported here:

1. **Monotonicity bounds / Wald LATE**. Uses only D, S, Y to identify
   stratum proportions and the complier PCE (= LATE). Zhang-Rubin
   (2003) sharp bounds for always-survivor SACE. No covariates needed.
2. **Principal score weighting** (Jo & Stuart 2009; Ding & Lu 2017).
   Estimates :math:`e_s(X) = P(\\text{stratum} | X)` using the
   observable-stratum logistic assignments and integrates to get
   stratum-specific ATEs. Relies on *principal ignorability*:
   :math:`Y(d) \\perp \\text{stratum} | X` within D=d.

References
----------
Frangakis, C.E. and Rubin, D.B. (2002). "Principal Stratification in
Causal Inference." *Biometrics*, 58(1), 21-29.

Zhang, J.L. and Rubin, D.B. (2003). "Estimation of Causal Effects via
Principal Stratification When Some Outcomes Are Truncated by 'Death'."
*Journal of Educational and Behavioral Statistics*, 28(4), 353-368.

Angrist, J.D., Imbens, G.W. and Rubin, D.B. (1996). "Identification of
causal effects using instrumental variables." *JASA*, 91(434), 444-455.

Ding, P. and Lu, J. (2017). "Principal stratification analysis using
principal scores." *JRSS-B*, 79(3), 757-777.

Jo, B. and Stuart, E.A. (2009). "On the use of propensity scores in
principal causal effect estimation." *Statistics in Medicine*, 28(23),
2857-2875.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


@dataclass
class PrincipalStratResult:
    """
    Principal stratification result.

    Attributes
    ----------
    method : str
        'monotonicity' or 'principal_score'.
    strata_proportions : dict
        Estimated proportion in each stratum.
    effects : pd.DataFrame
        Point estimate / SE / CI for each stratum-specific causal effect.
    bounds : pd.DataFrame or None
        For 'monotonicity' method, sharp Zhang-Rubin bounds on SACE.
    n_obs : int
    alpha : float
    model_info : dict
    """
    method: str
    strata_proportions: Dict[str, float]
    effects: pd.DataFrame
    bounds: Optional[pd.DataFrame]
    n_obs: int
    alpha: float
    model_info: Dict[str, Any]

    def summary(self) -> str:
        lines = [
            "=" * 72,
            f"Principal Stratification ({self.method})",
            "=" * 72,
            f"N = {self.n_obs}    alpha = {self.alpha}",
            "",
            "Stratum proportions:",
        ]
        for s, p in self.strata_proportions.items():
            lines.append(f"  {s:<25s}  {p:>7.4f}")
        lines += ["", "Principal causal effects:"]
        lines.append(self.effects.to_string(index=False, float_format='%.4f'))
        if self.bounds is not None:
            lines += ["", "Zhang-Rubin sharp bounds:"]
            lines.append(self.bounds.to_string(index=False, float_format='%.4f'))
        lines.append("=" * 72)
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()


def principal_strat(
    data: pd.DataFrame,
    y: str,
    treat: str,
    strata: str,
    covariates: Optional[List[str]] = None,
    method: str = 'monotonicity',
    instrument: Optional[str] = None,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: Optional[int] = None,
) -> PrincipalStratResult:
    """
    Principal stratification estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment assignment / instrument (0/1). For the IV
        interpretation, pass the **instrument** (encouragement), and
        ``strata`` = the actual take-up variable.
    strata : str
        Binary post-treatment variable (0/1) defining the principal
        strata (e.g. compliance, survival, employment).
    covariates : list of str, optional
        Baseline covariates. Required for ``method='principal_score'``.
    method : {'monotonicity', 'principal_score'}, default 'monotonicity'
        Identification strategy.
    instrument : str, optional
        If supplied, treat ``treat`` as the actual treatment and
        ``instrument`` as a randomized encouragement. Enables the
        AIR/Wald LATE estimator. **Not yet implemented in this minimal
        release** — raise a clear NotImplementedError if passed.
    alpha : float, default 0.05
    n_boot : int, default 500
        Bootstrap replications for SE/CI.
    seed : int, optional

    Returns
    -------
    PrincipalStratResult
    """
    if method not in ('monotonicity', 'principal_score'):
        raise ValueError(
            f"method must be 'monotonicity' or 'principal_score', got '{method}'"
        )
    if instrument is not None:
        raise NotImplementedError(
            "Explicit instrument + treatment two-layer setup not yet "
            "supported in principal_strat(). For LATE via encouragement "
            "design, use sp.iv or sp.dml(model='iivm')."
        )

    covariates = list(covariates or [])
    cols = [y, treat, strata] + covariates
    df = data[cols].dropna().reset_index(drop=True)
    n = len(df)

    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    S = df[strata].values.astype(float)
    X = df[covariates].values.astype(float) if covariates else None

    if not set(np.unique(D)).issubset({0, 1}):
        raise ValueError("treat must be binary (0/1).")
    if not set(np.unique(S)).issubset({0, 1}):
        raise ValueError("strata must be binary (0/1).")

    if method == 'monotonicity':
        return _fit_monotonicity(Y, D, S, n, alpha, n_boot, seed)
    return _fit_principal_score(Y, D, S, X, covariates, n, alpha, n_boot, seed)


def _fit_monotonicity(Y, D, S, n, alpha, n_boot, seed):
    """
    Monotonicity / AIR decomposition, S(1) >= S(0).

    Observable mixtures:
      P(S=1 | D=1) = π_always + π_complier
      P(S=1 | D=0) = π_always
      P(S=0 | D=0) = π_never + π_complier
      P(S=0 | D=1) = π_never

    → π_complier = P(S=1|D=1) - P(S=1|D=0)    (monotonicity requires ≥ 0)
      π_always   = P(S=1|D=0)
      π_never    = P(S=0|D=1)

    Complier PCE (LATE):
      τ_C = [E(Y|D=1,S=1) · P(S=1|D=1) - E(Y|D=0,S=1) · P(S=1|D=0)] / π_C

    Always-taker PCE on Y(1) - Y(0): under monotonicity + exclusion
    restriction the always-taker outcome is observable only in the
    (D=1, S=1) and (D=0, S=1) arms — specifically,
        E[Y(0) | always] = E[Y | D=0, S=1]
        E[Y(1) | always] is a mixture among (D=1, S=1) which contains
        always-takers + compliers.

    We report Zhang-Rubin (2003) sharp bounds for the always-survivor
    SACE = E[Y(1) - Y(0) | S(0)=S(1)=1].
    """
    def _point(Y_, D_, S_):
        # Cell probabilities and conditional means
        p_s1_d1 = float(np.mean(S_[D_ == 1])) if np.any(D_ == 1) else 0.0
        p_s1_d0 = float(np.mean(S_[D_ == 0])) if np.any(D_ == 0) else 0.0
        pi_complier = max(p_s1_d1 - p_s1_d0, 0.0)
        pi_always = p_s1_d0
        pi_never = 1 - p_s1_d1

        # Conditional means in the (D, S) cells
        def _safe_mean(mask, fallback=0.0):
            return float(np.mean(Y_[mask])) if np.any(mask) else fallback

        mu_11 = _safe_mean((D_ == 1) & (S_ == 1))
        mu_01 = _safe_mean((D_ == 0) & (S_ == 1))
        mu_10 = _safe_mean((D_ == 1) & (S_ == 0))
        mu_00 = _safe_mean((D_ == 0) & (S_ == 0))

        # Complier LATE (Wald-like on S=1 arm)
        if pi_complier > 1e-8:
            tau_c = (mu_11 * p_s1_d1 - mu_01 * p_s1_d0) / pi_complier
        else:
            tau_c = np.nan

        # Always-taker: Y(1) | always is the fraction of (D=1, S=1) that
        # is always-takers. Under monotonicity the (D=1, S=1) cell is a
        # mixture of compliers (fraction pi_complier/p_s1_d1) and always
        # (fraction pi_always/p_s1_d1). Without further assumptions,
        # point identification of E[Y(1)|always] needs principal
        # ignorability — bounds only for this method.

        # Zhang-Rubin sharp bounds on SACE.
        # The never-survivor ratio among (D=1, S=1) is bounded above by
        #   q = pi_always / p_s1_d1   (share of always-takers)
        # and we extract the worst/best q-slice of (D=1, S=1) outcomes.
        sace_lo, sace_hi = np.nan, np.nan
        if np.any((D_ == 1) & (S_ == 1)):
            y_11 = Y_[(D_ == 1) & (S_ == 1)]
            n_11 = len(y_11)
            if p_s1_d1 > 1e-8:
                q = pi_always / p_s1_d1
                q = float(np.clip(q, 0.0, 1.0))
                k = max(int(round(q * n_11)), 1)
                y_sorted = np.sort(y_11)
                # Lower bound: take the k smallest (pessimistic for always)
                lb_mu1 = float(np.mean(y_sorted[:k]))
                # Upper bound: take the k largest
                ub_mu1 = float(np.mean(y_sorted[-k:]))
                # Control-arm always-takers are directly identified:
                if pi_always > 1e-8:
                    sace_lo = lb_mu1 - mu_01
                    sace_hi = ub_mu1 - mu_01

        return {
            'pi_complier': pi_complier,
            'pi_always': pi_always,
            'pi_never': pi_never,
            'tau_c': tau_c,
            'mu_11': mu_11, 'mu_01': mu_01,
            'mu_10': mu_10, 'mu_00': mu_00,
            'sace_lo': sace_lo, 'sace_hi': sace_hi,
        }

    point = _point(Y, D, S)

    # Bootstrap inference for the LATE (complier PCE) and bounds endpoints
    rng = np.random.default_rng(seed)
    boot_tau = np.full(n_boot, np.nan)
    boot_lo = np.full(n_boot, np.nan)
    boot_hi = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            bp = _point(Y[idx], D[idx], S[idx])
            boot_tau[b] = bp['tau_c']
            boot_lo[b] = bp['sace_lo']
            boot_hi[b] = bp['sace_hi']
        except Exception:
            pass  # leave NaN — low volume of failures in practice

    def _ci(boot_arr, ppoint):
        valid = ~np.isnan(boot_arr)
        if valid.sum() < 2:
            return float('nan'), (float('nan'), float('nan')), float('nan')
        se = float(np.nanstd(boot_arr, ddof=1))
        lo = float(np.nanpercentile(boot_arr, 100 * alpha / 2))
        hi = float(np.nanpercentile(boot_arr, 100 * (1 - alpha / 2)))
        if np.isnan(ppoint) or se == 0:
            pv = float('nan')
        else:
            z = ppoint / se
            pv = float(2 * (1 - stats.norm.cdf(abs(z))))
        return se, (lo, hi), pv

    se_tc, ci_tc, pv_tc = _ci(boot_tau, point['tau_c'])
    se_lo, ci_lo, _ = _ci(boot_lo, point['sace_lo'])
    se_hi, ci_hi, _ = _ci(boot_hi, point['sace_hi'])

    effects = pd.DataFrame([{
        'stratum': 'Complier (LATE)',
        'estimate': point['tau_c'],
        'se': se_tc,
        'ci_lower': ci_tc[0],
        'ci_upper': ci_tc[1],
        'pvalue': pv_tc,
    }])

    bounds = pd.DataFrame([
        {'quantity': 'SACE lower bound (always-survivor)',
         'estimate': point['sace_lo'], 'se': se_lo,
         'ci_lower': ci_lo[0], 'ci_upper': ci_lo[1]},
        {'quantity': 'SACE upper bound (always-survivor)',
         'estimate': point['sace_hi'], 'se': se_hi,
         'ci_lower': ci_hi[0], 'ci_upper': ci_hi[1]},
    ])

    return PrincipalStratResult(
        method='monotonicity',
        strata_proportions={
            'always-taker / always-survivor': point['pi_always'],
            'complier': point['pi_complier'],
            'never-taker / never-survivor': point['pi_never'],
        },
        effects=effects,
        bounds=bounds,
        n_obs=n,
        alpha=alpha,
        model_info={
            'estimator': 'Monotonicity + Zhang-Rubin bounds',
            'n_boot': n_boot,
            **{k: v for k, v in point.items() if k.startswith('mu_')},
        },
    )


def _fit_principal_score(Y, D, S, X, covariates, n, alpha, n_boot, seed):
    """
    Principal-score weighting (Ding & Lu 2017 style).

    Under principal ignorability (PI) and monotonicity, the stratum
    membership e_s(X) = P(stratum | X) can be recovered from observable
    cell models:

      p11(X) = P(S=1 | D=1, X)
      p10(X) = P(S=1 | D=0, X)

      e_always(X)   = p10(X)
      e_complier(X) = p11(X) - p10(X)
      e_never(X)    = 1 - p11(X)

    The stratum-specific ATE is then estimated by weighting / slicing
    the observed cells. For the complier:
        τ_C = E[ W1 * Y | D=1, S=1 ] - E[ W0 * Y | D=0, S=0 ]
    with appropriate IPW weights built from e_s(X).

    We report complier, always-taker (Y(1) - Y(0)), never-taker PCE.
    Principal ignorability is a strong assumption — results should be
    paired with a sensitivity analysis (not yet shipped here).
    """
    if X is None or X.size == 0:
        raise ValueError(
            "method='principal_score' requires at least one covariate."
        )

    import statsmodels.api as sm

    def _fit_cell_probs(Y_, D_, S_, X_):
        # p11(X) = P(S=1 | D=1, X)
        mask1 = D_ == 1
        mask0 = D_ == 0
        p11_fit = _logit_safe(S_[mask1], X_[mask1])
        p10_fit = _logit_safe(S_[mask0], X_[mask0])
        p11 = _logit_predict(p11_fit, X_, fallback=float(np.mean(S_[mask1])) if mask1.any() else 0.5)
        p10 = _logit_predict(p10_fit, X_, fallback=float(np.mean(S_[mask0])) if mask0.any() else 0.5)
        # Enforce monotonicity e_complier ≥ 0 by clipping
        e_always = np.clip(p10, 1e-4, 1 - 1e-4)
        e_complier = np.clip(p11 - p10, 1e-4, 1 - 1e-4)
        e_never = np.clip(1 - p11, 1e-4, 1 - 1e-4)
        # Normalize to sum to 1 (can drift from clipping)
        tot = e_always + e_complier + e_never
        e_always /= tot
        e_complier /= tot
        e_never /= tot
        return e_always, e_complier, e_never

    def _point(Y_, D_, S_, X_):
        e_a, e_c, e_n = _fit_cell_probs(Y_, D_, S_, X_)

        # For the complier PCE under PI:
        #   τ_C = E[Y(1) - Y(0) | complier]
        # Y(1) | complier is identified from the mixture in (D=1, S=1)
        # weighted by e_c / (e_a + e_c):
        w1_c = e_c / np.clip(e_a + e_c, 1e-8, None)  # P(complier | D=1, S=1, X)
        w0_c = e_c / np.clip(e_c + e_n, 1e-8, None)  # P(complier | D=0, S=0, X)

        mask_11 = (D_ == 1) & (S_ == 1)
        mask_00 = (D_ == 0) & (S_ == 0)

        def _weighted_mean(y_arr, w_arr, mask):
            if not np.any(mask):
                return float('nan')
            y_sel = y_arr[mask]
            w_sel = w_arr[mask]
            tot = float(np.sum(w_sel))
            if tot <= 0:
                return float('nan')
            return float(np.sum(y_sel * w_sel) / tot)

        mu1_c = _weighted_mean(Y_, w1_c, mask_11)
        mu0_c = _weighted_mean(Y_, w0_c, mask_00)
        tau_c = mu1_c - mu0_c if not (np.isnan(mu1_c) or np.isnan(mu0_c)) else np.nan

        # Always-taker: only (D=1, S=1) and (D=0, S=1) cells have always-takers.
        w1_a = e_a / np.clip(e_a + e_c, 1e-8, None)  # P(always | D=1, S=1, X)
        mu1_a = _weighted_mean(Y_, w1_a, mask_11)
        # Y(0) | always identified directly from (D=0, S=1):
        mask_01 = (D_ == 0) & (S_ == 1)
        mu0_a = float(np.mean(Y_[mask_01])) if np.any(mask_01) else float('nan')
        tau_a = mu1_a - mu0_a if not (np.isnan(mu1_a) or np.isnan(mu0_a)) else np.nan

        # Never-taker: Y(1) | never from (D=1, S=0).
        mask_10 = (D_ == 1) & (S_ == 0)
        mu1_n = float(np.mean(Y_[mask_10])) if np.any(mask_10) else float('nan')
        w0_n = e_n / np.clip(e_c + e_n, 1e-8, None)  # P(never | D=0, S=0, X)
        mu0_n = _weighted_mean(Y_, w0_n, mask_00)
        tau_n = mu1_n - mu0_n if not (np.isnan(mu1_n) or np.isnan(mu0_n)) else np.nan

        return {
            'tau_c': tau_c, 'tau_a': tau_a, 'tau_n': tau_n,
            'pi_always': float(np.mean(e_a)),
            'pi_complier': float(np.mean(e_c)),
            'pi_never': float(np.mean(e_n)),
        }

    point = _point(Y, D, S, X)

    rng = np.random.default_rng(seed)
    boot = {k: np.full(n_boot, np.nan) for k in ('tau_c', 'tau_a', 'tau_n')}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            bp = _point(Y[idx], D[idx], S[idx], X[idx])
            for k in boot:
                boot[k][b] = bp[k]
        except Exception:
            pass

    def _ci(arr, ppoint):
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            return float('nan'), (float('nan'), float('nan')), float('nan')
        se = float(np.nanstd(arr, ddof=1))
        lo = float(np.nanpercentile(arr, 100 * alpha / 2))
        hi = float(np.nanpercentile(arr, 100 * (1 - alpha / 2)))
        pv = (
            float(2 * (1 - stats.norm.cdf(abs(ppoint / se))))
            if se > 0 and not np.isnan(ppoint) else float('nan')
        )
        return se, (lo, hi), pv

    rows = []
    for label, key in [
        ('Complier PCE', 'tau_c'),
        ('Always-taker PCE', 'tau_a'),
        ('Never-taker PCE', 'tau_n'),
    ]:
        se, ci, pv = _ci(boot[key], point[key])
        rows.append({
            'stratum': label,
            'estimate': point[key],
            'se': se,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'pvalue': pv,
        })

    effects = pd.DataFrame(rows)

    return PrincipalStratResult(
        method='principal_score',
        strata_proportions={
            'always-taker': point['pi_always'],
            'complier': point['pi_complier'],
            'never-taker': point['pi_never'],
        },
        effects=effects,
        bounds=None,
        n_obs=n,
        alpha=alpha,
        model_info={
            'estimator': 'Principal score weighting (Ding & Lu 2017)',
            'n_boot': n_boot,
            'covariates': covariates,
            'assumption': 'principal ignorability + monotonicity',
        },
    )


def survivor_average_causal_effect(
    data: pd.DataFrame,
    y: str,
    treat: str,
    survival: str,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Zhang-Rubin (2003) sharp bounds on the Survivor Average Causal Effect.

    Returns a :class:`CausalResult` with ``estimate`` set to the midpoint
    of the SACE bounds and the endpoints stored in ``model_info``.
    """
    ps = principal_strat(
        data=data, y=y, treat=treat, strata=survival,
        method='monotonicity', alpha=alpha, n_boot=n_boot, seed=seed,
    )
    lo = float(ps.bounds.loc[0, 'estimate'])
    hi = float(ps.bounds.loc[1, 'estimate'])
    midpoint = (lo + hi) / 2
    # Confidence-bound union of the two endpoints (Imbens & Manski style, simplified)
    ci_lo = float(ps.bounds.loc[0, 'ci_lower'])
    ci_hi = float(ps.bounds.loc[1, 'ci_upper'])
    width_se = max((hi - lo) / 2, 0.0)

    model_info = {
        'estimator': 'Zhang-Rubin SACE bounds',
        'sace_lower': lo,
        'sace_upper': hi,
        'bounds_width': hi - lo,
        **ps.model_info,
    }
    return CausalResult(
        method='SACE (Zhang-Rubin sharp bounds)',
        estimand='SACE',
        estimate=midpoint,
        se=width_se,  # half-width as a rough uncertainty surrogate
        pvalue=float('nan'),  # partial identification — point-null p-value not defined
        ci=(ci_lo, ci_hi),
        alpha=alpha,
        n_obs=ps.n_obs,
        model_info=model_info,
        _citation_key='principal_strat',
    )


# ---------------------------------------------------------------------
# Small helpers: safe logit fit/predict for principal score method
# ---------------------------------------------------------------------


def _logit_safe(y, X):
    try:
        import statsmodels.api as sm
        design = sm.add_constant(X, has_constant='add')
        fit = sm.Logit(y, design).fit(
            disp=0, maxiter=200, warn_convergence=False
        )
        return fit
    except Exception:
        return None


def _logit_predict(fit, X, fallback):
    if fit is None:
        return np.full(X.shape[0], fallback)
    import statsmodels.api as sm
    design = sm.add_constant(X, has_constant='add')
    return np.clip(fit.predict(design), 1e-6, 1 - 1e-6)


CausalResult._CITATIONS['principal_strat'] = (
    "@article{frangakis2002principal,\n"
    "  title={Principal Stratification in Causal Inference},\n"
    "  author={Frangakis, Constantine E. and Rubin, Donald B.},\n"
    "  journal={Biometrics},\n"
    "  volume={58},\n"
    "  number={1},\n"
    "  pages={21--29},\n"
    "  year={2002}\n"
    "}"
)
