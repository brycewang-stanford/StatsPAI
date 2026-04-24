"""
Heckman (1979) two-step selection model.

Corrects for sample selection bias when the outcome is only observed
for a non-random subset of the population.

    Selection:  D_i* = Z_i'γ + u_i,   D_i = 1(D_i* > 0)
    Outcome:    Y_i  = X_i'β + ε_i,   observed only if D_i = 1

Estimation proceeds in two steps:
1. Probit of D on Z → compute inverse Mills ratio λ̂(Z'γ̂)
2. OLS of Y on X and λ̂ → consistent β̂ and σ̂_εu (selection effect)

References
----------
Heckman, J.J. (1979).
"Sample Selection Bias as a Specification Error."
*Econometrica*, 47(1), 153-161. [@heckman1979sample]
"""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def heckman(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    select: str,
    z: List[str],
    alpha: float = 0.05,
) -> CausalResult:
    """
    Heckman (1979) two-step selection model.

    Equivalent to Stata's ``heckman y x, select(z)``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable (observed only when ``select=1``).
    x : list of str
        Regressors in the outcome equation.
    select : str
        Binary selection indicator (1 = observed, 0 = not observed).
    z : list of str
        Variables in the selection equation (should include exclusion
        restrictions — variables in z but not in x).
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        Outcome equation coefficients, with inverse Mills ratio (lambda)
        and selection diagnostics.

    Examples
    --------
    >>> result = sp.heckman(df, y='wage', x=['education', 'experience'],
    ...                     select='employed', z=['education', 'children', 'spouse_income'])
    >>> print(result.summary())

    Notes
    -----
    The inverse Mills ratio λ(·) = φ(·)/Φ(·) captures the selection
    correction. If the coefficient on λ (called ``rho * sigma``) is
    statistically significant, selection bias is present.

    **Exclusion restriction**: At least one variable should be in ``z``
    but not in ``x`` (affects selection but not outcome directly).
    Without this, identification relies on functional form only.

    See Heckman (1979, *Econometrica*), Section 2.
    """
    df = data.copy()

    for col in [y, select] + x + z:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")

    D = df[select].values.astype(float)
    n_total = len(df)

    # --- Step 1: Probit selection equation ---
    Z = np.column_stack([np.ones(n_total)] +
                        [df[v].values.astype(float) for v in z])
    valid_s = np.all(np.isfinite(Z), axis=1) & np.isfinite(D)
    D_v = D[valid_s]
    Z_v = Z[valid_s]

    # Probit via MLE (Newton-Raphson); return γ̂ AND V̂_γ for the
    # Heckman (1979) two-step SE correction below.
    gamma, V_gamma = _probit_fit(D_v, Z_v)
    Z_gamma = Z @ gamma  # for all observations (need IMR for selected)

    # Inverse Mills ratio: λ(Z'γ) = φ(Z'γ) / Φ(Z'γ)
    phi = stats.norm.pdf(Z_gamma)
    Phi = stats.norm.cdf(Z_gamma)
    Phi = np.clip(Phi, 1e-10, 1 - 1e-10)
    imr = phi / Phi

    # --- Step 2: OLS outcome equation with IMR ---
    selected = D == 1
    df_sel = df[selected].copy()
    imr_sel = imr[selected]
    Z_gamma_sel = Z_gamma[selected]
    Z_selected = Z[selected]
    n_sel = selected.sum()

    Y_sel = df_sel[y].values.astype(float)
    X_sel = np.column_stack([np.ones(n_sel)] +
                            [df_sel[v].values.astype(float) for v in x] +
                            [imr_sel])

    valid_o = np.all(np.isfinite(X_sel), axis=1) & np.isfinite(Y_sel)
    Y_v = Y_sel[valid_o]
    X_v = X_sel[valid_o]
    Z_v = Z_selected[valid_o]
    Zg_v = Z_gamma_sel[valid_o]
    imr_v = imr_sel[valid_o]
    n_eff = len(Y_v)
    k = X_v.shape[1]

    beta = np.linalg.lstsq(X_v, Y_v, rcond=None)[0]
    resid = Y_v - X_v @ beta

    # --- Heckman (1979) two-step analytical variance ---
    #
    # The second-stage regressor λ̂ is a "generated regressor", so a
    # naive OLS / HC-style sandwich under-states true uncertainty by
    # ignoring the first-stage estimation error in γ̂ and misses the
    # heteroskedasticity Var(y|X, D=1) = σ²(1 − ρ² δ_i).
    #
    # Correct variance (Greene 2003 eq. 22-22; Heckman 1979 eq. 19;
    # Wooldridge 2010 §19.6):
    #
    #   V(β̂) = σ̂² (X*'X*)^{-1} [ X*'(I − ρ̂² D_δ) X* + ρ̂² F V̂_γ F' ]
    #          × (X*'X*)^{-1}
    #
    # with
    #   δ_i     = λ̂_i (λ̂_i + Z_iγ̂)        ≥ 0 (Mills' ratio inequality)
    #   D_δ     = diag(δ_i)
    #   F       = X*' D_δ Z                  (k × q)
    #   σ̂²     = u'u/n + β̂_λ² · mean(δ_i)  (Greene 22-21)
    #   ρ̂²     = β̂_λ² / σ̂²
    #
    # For 2SLS / k-class this is a different structural bug family
    # (fixed in v1.6.4/1.6.5); for Heckman the generated-regressor
    # correction is the canonical fix.
    delta_v = imr_v * (imr_v + Zg_v)
    beta_lambda = float(beta[-1])
    rss = float(np.sum(resid ** 2))
    sigma2 = rss / n_eff + beta_lambda ** 2 * float(np.mean(delta_v))
    rho2 = beta_lambda ** 2 / sigma2 if sigma2 > 0 else 0.0

    XtX_inv = np.linalg.pinv(X_v.T @ X_v)
    # Heteroskedastic contribution  X*'(I − ρ̂² D_δ) X*
    het_w = 1.0 - rho2 * delta_v
    M_het = X_v.T @ (het_w[:, None] * X_v)
    # Generated-regressor contribution  ρ̂² F V̂_γ F'  with F = X*' D_δ Z
    F = X_v.T @ (delta_v[:, None] * Z_v)
    Q = rho2 * (F @ V_gamma @ F.T)

    vcov = sigma2 * XtX_inv @ (M_het + Q) @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    # Variable names
    var_names = ['const'] + x + ['lambda (IMR)']
    z_stats = beta / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))

    detail = pd.DataFrame({
        'variable': var_names,
        'coefficient': beta,
        'se': se,
        'z': z_stats,
        'pvalue': pvals,
    })

    # Lambda coefficient = rho * sigma (selection correction)
    lambda_coef = float(beta[-1])
    lambda_se = float(se[-1])
    lambda_z = lambda_coef / lambda_se if lambda_se > 0 else 0
    lambda_p = float(2 * (1 - stats.norm.cdf(abs(lambda_z))))

    # Main estimate: first regressor's coefficient (after constant)
    main_coef = float(beta[1])
    main_se = float(se[1])
    z_crit = stats.norm.ppf(1 - alpha / 2)
    main_z = main_coef / main_se
    main_p = float(2 * (1 - stats.norm.cdf(abs(main_z))))
    ci = (main_coef - z_crit * main_se, main_coef + z_crit * main_se)

    model_info = {
        'method': 'Heckman Two-Step',
        'n_total': n_total,
        'n_selected': int(n_sel),
        'n_censored': int(n_total - n_sel),
        'selection_vars': z,
        'lambda_coef': lambda_coef,
        'lambda_se': lambda_se,
        'lambda_pvalue': lambda_p,
        'selection_bias': 'Yes (lambda significant)' if lambda_p < 0.05
                          else 'No evidence (lambda not significant)',
        'sigma': float(np.sqrt(sigma2)),
        'rho': float(lambda_coef / np.sqrt(sigma2)) if sigma2 > 0 else np.nan,
    }

    return CausalResult(
        method='Heckman (1979) Selection Model',
        estimand=f'beta_{x[0]}',
        estimate=main_coef,
        se=main_se,
        pvalue=main_p,
        ci=ci,
        alpha=alpha,
        n_obs=n_eff,
        detail=detail,
        model_info=model_info,
        _citation_key='heckman',
    )


def _probit_fit(D, Z, max_iter=50):
    """Probit MLE via Newton–Raphson.

    Returns
    -------
    gamma : (q,) ndarray
        Maximum-likelihood estimate of the probit coefficient vector.
    V_gamma : (q, q) ndarray
        Asymptotic variance-covariance matrix of γ̂, computed as
        ``(Z' diag(w) Z)^{-1}`` with expected-information weights
        ``w_i = φ(Z_iγ̂)² / [Φ(Z_iγ̂)(1 − Φ(Z_iγ̂))]``. Required by
        the Heckman (1979) two-step SE correction.
    """
    n, k = Z.shape
    gamma = np.zeros(k)
    H = None
    w = None

    for _ in range(max_iter):
        Zg = Z @ gamma
        Phi = np.clip(stats.norm.cdf(Zg), 1e-10, 1 - 1e-10)
        phi = stats.norm.pdf(Zg)

        # Score and Hessian (broadcast avoids allocating diag(w) n×n).
        w = phi ** 2 / (Phi * (1 - Phi))
        w = np.clip(w, 1e-10, 1e10)
        score = Z.T @ ((D - Phi) * phi / (Phi * (1 - Phi)))
        H = -(Z.T @ (w[:, None] * Z))

        try:
            delta = np.linalg.solve(H, score)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(H, score, rcond=None)[0]

        gamma -= delta
        if np.max(np.abs(delta)) < 1e-8:
            break

    # V(γ̂) = (Z' diag(w) Z)^{-1} = (−H)^{-1} at convergence.
    try:
        V_gamma = np.linalg.inv(Z.T @ (w[:, None] * Z))
    except np.linalg.LinAlgError:
        V_gamma = np.linalg.pinv(Z.T @ (w[:, None] * Z))
    return gamma, V_gamma


# Citation
CausalResult._CITATIONS['heckman'] = (
    "@article{heckman1979sample,\n"
    "  title={Sample Selection Bias as a Specification Error},\n"
    "  author={Heckman, James J.},\n"
    "  journal={Econometrica},\n"
    "  volume={47},\n"
    "  number={1},\n"
    "  pages={153--161},\n"
    "  year={1979},\n"
    "  publisher={Wiley}\n"
    "}"
)
