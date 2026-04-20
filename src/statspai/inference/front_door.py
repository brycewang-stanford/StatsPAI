"""
Front-door adjustment estimator (Pearl 1995, 2009).

Identifies ``E[Y | do(D=d)]`` when the back-door criterion is blocked by
unobserved confounding, but there exists a mediator :math:`M` that
fully transmits the effect of :math:`D` on :math:`Y`, and no
unobserved confounder affects :math:`M` directly (given :math:`D`).

Causal DAG assumed::

    U ──┬──► D ──► M ──► Y
        │             ▲
        └─────────────┘        (U unobserved)

Front-door formula::

    E[Y | do(D=d)]
        = Σ_m P(M=m | D=d) · Σ_{d'} P(D=d') · E[Y | D=d', M=m]

For a binary :math:`D` and (binary or continuous) :math:`M`, the ATE
can be written as::

    ATE = Σ_m [P(M=m|D=1) - P(M=m|D=0)] · μ(m)

where :math:`μ(m) = Σ_{d'} P(D=d') · E[Y | D=d', M=m]` is the
marginalized outcome regression.

This implementation supports:

* Binary :math:`D`.
* Binary :math:`M` — closed-form sums over {0,1}.
* Continuous :math:`M` — Monte Carlo integration via parametric
  conditional-mediator model (Gaussian or sampling from the empirical
  mediator-residual distribution).

Inference via nonparametric bootstrap.

References
----------
Pearl, J. (1995). "Causal diagrams for empirical research." *Biometrika*,
82(4), 669-688.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
(2nd ed.). Cambridge University Press. §3.3.

Fulcher, I.R., Shpitser, I., Marealle, S., Tchetgen Tchetgen, E.J.
(2020). "Robust inference on population indirect causal effects:
the generalized front-door criterion." *JRSS-B*, 82(1), 199-214.
"""

import warnings
from typing import Optional, List, Any
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def front_door(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[List[str]] = None,
    mediator_type: str = 'auto',
    integrate_by: str = 'marginal',
    n_boot: int = 500,
    n_mc: int = 200,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Front-door ATE via Pearl's front-door formula.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment (0/1).
    mediator : str
        Mediator that fully transmits D's effect on Y.
    covariates : list of str, optional
        Pre-treatment covariates to condition on (entered additively in
        the outcome and mediator models). **Covariates must be pre-D**;
        including post-treatment controls will re-open the back-door.
    mediator_type : {'auto', 'binary', 'continuous'}, default 'auto'
        How to model the mediator. ``'auto'`` detects binary if M takes
        only {0,1} values, else continuous.
    integrate_by : {'marginal', 'conditional'}, default 'marginal'
        How to integrate over the M-distribution when computing the
        front-door ATE (continuous M only — ignored for binary M).

        - ``'marginal'``: follows Pearl's (1995) aggregate formulation.
          For each unit's X_base row, Monte Carlo samples of M are drawn
          from the population marginal M|D=d distribution (covariate
          rows are re-sampled inside the MC). Appropriate when the
          causal estimand is population-averaged and you want variance
          that reflects the full population of baseline-covariate rows.
        - ``'conditional'``: follows Fulcher et al. (2020) generalized
          front-door. For each unit i, M-samples are drawn from the
          unit-specific conditional M|D=d,X_i distribution. Stricter
          identification (requires the mediator model to be correct
          conditional on X_i) but gives per-unit ATE contributions.

        For binary M and for problems with no covariates the two
        formulations coincide.
    n_boot : int, default 500
        Nonparametric bootstrap replications.
    n_mc : int, default 200
        Monte Carlo samples per observation for the continuous-M case.
    alpha : float, default 0.05
        Significance level.
    seed : int, optional
        Random seed.

    Returns
    -------
    CausalResult
        ``estimate`` is the front-door ATE, E[Y|do(D=1)] - E[Y|do(D=0)].

    Notes
    -----
    Identification requires:

    1. No direct effect of D on Y (all effect routed through M).
    2. No unobserved confounder of the M-Y relationship.
    3. Positivity on M | D.

    Violations of (1) or (2) are **not** detected from data alone — the
    front-door criterion is a DAG assumption. Use :func:`sp.dag` with
    ``front_door_adjustment_sets()`` to verify identifiability from your
    assumed DAG before calling this estimator.

    References
    ----------
    Pearl, J. (1995). Causal diagrams for empirical research. *Biometrika*.
    Fulcher et al. (2020). Robust inference on population indirect causal
    effects: the generalized front-door criterion. *JRSS-B*.
    """
    if integrate_by not in ('marginal', 'conditional'):
        raise ValueError(
            f"integrate_by must be 'marginal' or 'conditional'; "
            f"got '{integrate_by}'"
        )
    covariates = list(covariates or [])
    df = data[[y, treat, mediator] + covariates].dropna().reset_index(drop=True)
    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    M = df[mediator].values.astype(float)
    X = df[covariates].values.astype(float) if covariates else None
    n = len(Y)

    if not set(np.unique(D)).issubset({0, 1}):
        raise ValueError(
            "Front-door estimator currently supports binary D (0/1); "
            f"got values {sorted(set(np.unique(D)))[:5]}"
            f"{'...' if len(set(np.unique(D))) > 5 else ''}. "
            "Continuous-D front-door is not shipped in this release — "
            "see docs/ROADMAP.md. If the front-door criterion is not "
            "required (i.e. no unmeasured confounder blocking the "
            "back-door), use sp.g_computation(..., estimand='dose_response', "
            "treat_values=[d1, d2, ...]) for a dose-response curve under "
            "the standard unconfoundedness assumption."
        )

    if mediator_type == 'auto':
        mediator_type = 'binary' if set(np.unique(M)).issubset({0, 1}) else 'continuous'
    if mediator_type not in ('binary', 'continuous'):
        raise ValueError(
            f"mediator_type must be 'binary' or 'continuous'; got '{mediator_type}'"
        )

    rng = np.random.default_rng(seed)

    def _point(Y_, D_, M_, X_):
        return _front_door_ate(
            Y_, D_, M_, X_, mediator_type, n_mc, rng, integrate_by
        )

    point = _point(Y, D, M, X)

    # Bootstrap — failures leave NaN, we track and warn rather than
    # silently replace with the point estimate (which would shrink SE).
    boot = np.full(n_boot, np.nan)
    n_failed = 0
    first_err: Optional[str] = None
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot[b] = _point(
                Y[idx], D[idx], M[idx], X[idx] if X is not None else None
            )
        except Exception as e:
            n_failed += 1
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"

    n_success = n_boot - n_failed
    if n_success < 2:
        raise RuntimeError(
            f"Front-door bootstrap failed on {n_failed}/{n_boot} replications "
            f"(only {n_success} succeeded; need ≥2 for SE). "
            f"First error: {first_err}."
        )
    if n_failed > 0:
        frac = n_failed / n_boot
        warnings.warn(
            f"Front-door: {n_failed}/{n_boot} bootstrap replications failed "
            f"({frac:.1%}). SE/CI computed over {n_success} successes. "
            f"First error: {first_err}.",
            RuntimeWarning,
            stacklevel=2,
        )

    se = float(np.nanstd(boot, ddof=1))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (float(point - z_crit * se), float(point + z_crit * se))
    z = point / se if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    # When there are no covariates (or mediator is binary), the
    # marginal and conditional formulas coincide — record both the
    # user's request and what actually ran so the audit trail is clear.
    if mediator_type == 'binary':
        effective = 'n/a (binary mediator — closed-form sum)'
    elif X is None:
        effective = 'conditional (marginal ≡ conditional when X is empty)'
    else:
        effective = integrate_by

    model_info = {
        'estimator': 'Front-door adjustment (Pearl 1995)',
        'mediator': mediator,
        'mediator_type': mediator_type,
        'integrate_by': integrate_by,
        'integrate_by_effective': effective,
        'n_boot': n_boot,
        'n_boot_failed': n_failed,
        'n_boot_success': n_success,
        'n_treated': int((D == 1).sum()),
        'n_control': int((D == 0).sum()),
        'covariates': covariates,
    }
    if n_failed > 0:
        model_info['first_bootstrap_error'] = first_err
    if mediator_type == 'continuous':
        model_info['n_mc'] = n_mc

    return CausalResult(
        method='Front-door adjustment',
        estimand='ATE',
        estimate=float(point),
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key='front_door',
    )


def _ols_fit(y, X):
    """Minimal OLS with intercept; returns (beta, residuals)."""
    design = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    return beta, resid, design


def _ols_predict(beta, X):
    design = np.column_stack([np.ones(X.shape[0]), X])
    return design @ beta


def _logit_fit(y, X):
    """Logistic regression — falls back to empirical mean on singular design."""
    try:
        import statsmodels.api as sm
        design = sm.add_constant(X, has_constant='add')
        fit = sm.Logit(y, design).fit(disp=0, maxiter=200, warn_convergence=False)
        return fit
    except Exception:
        return None


def _logit_predict(fit, X, fallback):
    if fit is None:
        return np.full(X.shape[0], fallback)
    import statsmodels.api as sm
    design = sm.add_constant(X, has_constant='add')
    return np.clip(fit.predict(design), 1e-6, 1 - 1e-6)


def _front_door_ate(Y, D, M, X, mediator_type, n_mc, rng, integrate_by='marginal'):
    """
    Compute front-door ATE on a single (bootstrap or original) sample.
    """
    n = len(Y)

    # P(D=d') — marginal treatment distribution (over full sample)
    p_d1 = float(np.mean(D))
    p_d0 = 1.0 - p_d1

    # Build feature matrices for outcome and mediator regressions.
    if X is None:
        feat_ym = M.reshape(-1, 1)            # outcome regression features: M
        feat_d_only = np.zeros((n, 0))        # no X
    else:
        feat_ym = np.column_stack([M, X])     # outcome regression: M, X
        feat_d_only = X                       # mediator/treatment regression: X

    # Outcome model: Y ~ M (+ X) separately for D=0 and D=1, so we can
    # evaluate E[Y|D=d', M=m, X] at each (d', m, x).
    mask1 = D == 1
    mask0 = D == 0
    if mask1.sum() < 2 or mask0.sum() < 2:
        raise RuntimeError("Insufficient support on D=0 or D=1 for outcome regression.")

    beta_y1, _, _ = _ols_fit(Y[mask1], feat_ym[mask1])
    beta_y0, _, _ = _ols_fit(Y[mask0], feat_ym[mask0])

    # Helper: E[Y | D=d', M=m_grid, X=x_row] for each obs
    def mu_dprime(d_prime_beta, m_vec, X_):
        if X_ is None:
            feat = m_vec.reshape(-1, 1)
        else:
            feat = np.column_stack([m_vec, X_])
        return _ols_predict(d_prime_beta, feat)

    # ------------------------------------------------------------------
    # Mediator distribution P(M | D=d, X) — either Bernoulli (binary M)
    # or Gaussian (continuous M).
    # ------------------------------------------------------------------

    if mediator_type == 'binary':
        # For each observation i, compute ATE contribution in closed form:
        #   ATE = E_x[ [P(M=1|D=1,x) - P(M=1|D=0,x)] * (μ(1) - μ(0)) ]
        # where μ(m) = p_d0 * E[Y|D=0,M=m,x] + p_d1 * E[Y|D=1,M=m,x].
        if X is None:
            # Pool probabilities
            p_m_given_d1 = float(np.mean(M[mask1]))
            p_m_given_d0 = float(np.mean(M[mask0]))
            p_m_d1 = np.full(n, p_m_given_d1)
            p_m_d0 = np.full(n, p_m_given_d0)
        else:
            fit1 = _logit_fit(M[mask1], X[mask1])
            fit0 = _logit_fit(M[mask0], X[mask0])
            p_m_d1 = _logit_predict(fit1, X, float(np.mean(M[mask1])))
            p_m_d0 = _logit_predict(fit0, X, float(np.mean(M[mask0])))

        # μ(m=1) and μ(m=0) at each x
        one_vec = np.ones(n)
        zero_vec = np.zeros(n)
        mu_Y_given_d0_m1 = mu_dprime(beta_y0, one_vec, X)
        mu_Y_given_d1_m1 = mu_dprime(beta_y1, one_vec, X)
        mu_Y_given_d0_m0 = mu_dprime(beta_y0, zero_vec, X)
        mu_Y_given_d1_m0 = mu_dprime(beta_y1, zero_vec, X)

        mu_m1 = p_d0 * mu_Y_given_d0_m1 + p_d1 * mu_Y_given_d1_m1
        mu_m0 = p_d0 * mu_Y_given_d0_m0 + p_d1 * mu_Y_given_d1_m0

        ate = float(np.mean((p_m_d1 - p_m_d0) * (mu_m1 - mu_m0)))
        return ate

    # ------------------------------------------------------------------
    # Continuous mediator: Gaussian conditional model
    #     M | D=d, X ~ Normal(α_d + β_d' X, σ_d^2)
    # Monte Carlo integrate over M | D=d, X for each observation.
    # ------------------------------------------------------------------

    if X is None:
        feat_m = np.zeros((n, 0))
    else:
        feat_m = X

    beta_m1, resid_m1, _ = _ols_fit(M[mask1], feat_m[mask1])
    beta_m0, resid_m0, _ = _ols_fit(M[mask0], feat_m[mask0])

    # Gaussian sigmas
    sigma_m1 = float(np.std(resid_m1, ddof=max(1, feat_m.shape[1] + 1)))
    sigma_m0 = float(np.std(resid_m0, ddof=max(1, feat_m.shape[1] + 1)))
    sigma_m1 = max(sigma_m1, 1e-6)
    sigma_m0 = max(sigma_m0, 1e-6)

    # Means E[M|D=d, X] for every observation
    mean_m1 = _ols_predict(beta_m1, feat_m)
    mean_m0 = _ols_predict(beta_m0, feat_m)

    # Monte-Carlo integrate E[Y|do(D=d)] = E_x E_{m|D=d,(·)} [ Σ_{d'} P(d') E[Y|d',m,x] ]
    # Two variants:
    #   'conditional'  — per-unit: m ~ N(μ_d(x_i), σ_d^2), X held at x_i.
    #                    Matches Fulcher et al. (2020) generalized front-door.
    #   'marginal'     — Pearl (1995) aggregate: for each MC draw, both X and
    #                    M are re-sampled from the population, so the outer
    #                    expectation integrates over baseline covariates via MC.
    if integrate_by == 'conditional' or X is None:
        # Per-observation MC: m draw uses unit's own (μ_d, σ_d); X stays at x_i
        X_rep = None if X is None else np.repeat(X, n_mc, axis=0)
        mean_m1_rep = np.repeat(mean_m1, n_mc)
        mean_m0_rep = np.repeat(mean_m0, n_mc)
        eps = rng.standard_normal(n * n_mc)
        m_samples_d1 = mean_m1_rep + sigma_m1 * eps
        m_samples_d0 = mean_m0_rep + sigma_m0 * eps

        mu0_under_d1 = mu_dprime(beta_y0, m_samples_d1, X_rep)
        mu1_under_d1 = mu_dprime(beta_y1, m_samples_d1, X_rep)
        mu0_under_d0 = mu_dprime(beta_y0, m_samples_d0, X_rep)
        mu1_under_d0 = mu_dprime(beta_y1, m_samples_d0, X_rep)

        inner_d1 = p_d0 * mu0_under_d1 + p_d1 * mu1_under_d1
        inner_d0 = p_d0 * mu0_under_d0 + p_d1 * mu1_under_d0

        inner_d1 = inner_d1.reshape(n, n_mc).mean(axis=1)
        inner_d0 = inner_d0.reshape(n, n_mc).mean(axis=1)
        return float(np.mean(inner_d1 - inner_d0))

    # 'marginal' — Pearl aggregate formulation. One global pool of
    # (x, m)-pairs drawn from the population-marginal distributions.
    sample_idx = rng.integers(0, n, size=n_mc)
    X_pool = X[sample_idx]
    mean_m1_pool = mean_m1[sample_idx]
    mean_m0_pool = mean_m0[sample_idx]
    eps = rng.standard_normal(n_mc)
    m_pool_d1 = mean_m1_pool + sigma_m1 * eps
    m_pool_d0 = mean_m0_pool + sigma_m0 * eps

    # E[Y | D=d', M=m, X=x] evaluated at each pooled (x, m)
    mu0_d1_pool = mu_dprime(beta_y0, m_pool_d1, X_pool)
    mu1_d1_pool = mu_dprime(beta_y1, m_pool_d1, X_pool)
    mu0_d0_pool = mu_dprime(beta_y0, m_pool_d0, X_pool)
    mu1_d0_pool = mu_dprime(beta_y1, m_pool_d0, X_pool)

    e_y_d1 = float(np.mean(p_d0 * mu0_d1_pool + p_d1 * mu1_d1_pool))
    e_y_d0 = float(np.mean(p_d0 * mu0_d0_pool + p_d1 * mu1_d0_pool))
    return e_y_d1 - e_y_d0


# Citation
CausalResult._CITATIONS['front_door'] = (
    "@article{pearl1995causal,\n"
    "  title={Causal Diagrams for Empirical Research},\n"
    "  author={Pearl, Judea},\n"
    "  journal={Biometrika},\n"
    "  volume={82},\n"
    "  number={4},\n"
    "  pages={669--688},\n"
    "  year={1995}\n"
    "}"
)
