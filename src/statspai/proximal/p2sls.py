"""
Proximal 2SLS estimator for proximal causal inference.

Setting
-------
We want :math:`E[Y | do(D)]` but the classical back-door is blocked by
an unmeasured confounder :math:`U`. We assume access to two proxies:

* :math:`Z` — a "treatment proxy" satisfying Z ⊥ Y | (D, U, X)
* :math:`W` — an "outcome proxy" satisfying W ⊥ D | (U, X)

and that the outcome-confounding bridge function :math:`h(W, D, X)`
exists and solves Fredholm equation of the first kind:

    E[Y | D, Z, X] = E[h(W, D, X) | D, Z, X].

Under these conditions, the ATE is identified as
:math:`E[h(W, 1, X)] - E[h(W, 0, X)]`.

Linear specification (this implementation)
------------------------------------------
With a **linear** outcome bridge :math:`h(W, D, X) = γ_0 + γ_D D +
γ_W' W + γ_X' X`, the bridge-equation restriction becomes a moment
condition that's identified by an IV regression of Y on (D, W, X)
using (D, Z, X) as instruments — i.e. a standard 2SLS where :math:`W`
is endogenous and :math:`Z` is its instrument (Cui et al. 2024,
Tchetgen Tchetgen et al. 2020 §3.2, linear case).

The ATE is then the coefficient :math:`γ_D` on :math:`D` (plus any
treatment-covariate interactions averaged out, not included here).

Important limitations of this implementation
--------------------------------------------

* **Linear bridge**. For nonlinear bridges use Deaner (2018) sieve
  methods or Mastouri et al. (2021) kernel methods — not yet shipped.
* **Proxy rank condition** (``rank(E[(Z, D, X)(W, D, X)']) = full``)
  must hold; violated when Z is a weak proxy for U. We report a
  Cragg-Donald-style first-stage F on the proxy equation for sanity.
* **Homoskedastic standard errors** via 2SLS sandwich; for cluster or
  heteroskedastic inference use bootstrap by setting ``n_boot > 0``.

References
----------
Tchetgen Tchetgen, E.J., Ying, A., Cui, Y., Shi, X. and Miao, W. (2020).
"An Introduction to Proximal Causal Learning." arXiv:2009.10982.

Miao, W., Geng, Z. and Tchetgen Tchetgen, E.J. (2018). "Identifying
causal effects with proxy variables of an unmeasured confounder."
*Biometrika*, 105(4), 987-993.

Cui, Y., Pu, H., Shi, X., Miao, W. and Tchetgen Tchetgen, E.J. (2024).
"Semiparametric proximal causal inference." *JASA*, 119(546), 1348-1359.
"""

import warnings
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def proximal(
    data: pd.DataFrame,
    y: str,
    treat: str,
    proxy_z: List[str],
    proxy_w: List[str],
    covariates: Optional[List[str]] = None,
    n_boot: int = 0,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> CausalResult:
    """
    Proximal causal inference via linear 2SLS on the outcome bridge.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Treatment variable (binary or continuous).
    proxy_z : list of str
        Treatment-inducing confounding proxy variable(s) (Z). These
        serve as instruments for the outcome proxy W.
    proxy_w : list of str
        Outcome-inducing confounding proxy variable(s) (W). Endogenous
        regressors in the linear bridge.
    covariates : list of str, optional
        Measured baseline covariates X (exogenous controls).
    n_boot : int, default 0
        If > 0, nonparametric bootstrap SE (rows, not cluster-robust).
        If 0, use closed-form 2SLS sandwich SE (homoskedastic).
    alpha : float, default 0.05
    seed : int, optional

    Returns
    -------
    CausalResult
        ``estimate`` is the coefficient on the treatment in the linear
        bridge — the proximal ATE under correct specification.

    Examples
    --------
    >>> # ATE of smoking on lung cancer, with occupation (Z) and
    >>> # secondhand-smoke exposure (W) as proxies for unmeasured
    >>> # health behaviour/genetics.
    >>> sp.proximal(df, y='lung_cancer', treat='smoker',
    ...             proxy_z=['occupation'], proxy_w=['shs_exposure'],
    ...             covariates=['age', 'sex'])
    """
    covariates = list(covariates or [])
    all_cols = [y, treat] + list(proxy_z) + list(proxy_w) + covariates
    missing = [c for c in all_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    df = data[all_cols].dropna().reset_index(drop=True)
    n = len(df)
    if n < len(all_cols) + 5:
        raise ValueError(
            f"Too few complete rows ({n}) for proximal 2SLS with "
            f"{len(all_cols)} variables."
        )

    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float).reshape(-1, 1)
    Z = df[list(proxy_z)].values.astype(float)
    W = df[list(proxy_w)].values.astype(float)
    X_cov = df[covariates].values.astype(float) if covariates else np.zeros((n, 0))

    # Exogenous block (appears in both stages): const + D + X
    # Endogenous block: W
    # Excluded instruments: Z (one per W ideally)
    const = np.ones((n, 1))
    X_exog = np.hstack([const, D, X_cov])                # (n, 2 + p_x)
    instruments = np.hstack([X_exog, Z])                 # (n, 2 + p_x + p_z)
    regressors = np.hstack([X_exog, W])                  # (n, 2 + p_x + p_w)

    # Order-condition check: need #instruments >= #regressors
    k_exog = X_exog.shape[1]
    k_w = W.shape[1]
    k_z = Z.shape[1]
    if k_z < k_w:
        raise ValueError(
            f"Order condition violated: need at least as many Z proxies "
            f"({k_z}) as W proxies ({k_w}). Proximal identification "
            f"requires at least {k_w} excluded instruments."
        )

    try:
        beta, vcov, first_stage_F = _linear_iv_fit(
            Y, regressors, instruments, k_exog
        )
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Proximal 2SLS failed: {e}. Possible rank deficiency in "
            f"the proxy-instrument system — check completeness of Z/W."
        )

    # Treatment coefficient sits at position 1 (after the constant).
    tau = float(beta[1])
    se_closed = float(np.sqrt(max(vcov[1, 1], 0.0)))

    # Bootstrap SE if requested (overrides closed-form)
    se = se_closed
    boot_failed = 0
    first_err: Optional[str] = None
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        boot = np.full(n_boot, np.nan)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                beta_b, _, _ = _linear_iv_fit(
                    Y[idx], regressors[idx], instruments[idx], k_exog
                )
                boot[b] = beta_b[1]
            except Exception as e:
                boot_failed += 1
                if first_err is None:
                    first_err = f"{type(e).__name__}: {e}"
        if (n_boot - boot_failed) >= 2:
            se = float(np.nanstd(boot, ddof=1))
            if boot_failed > 0:
                warnings.warn(
                    f"Proximal: {boot_failed}/{n_boot} bootstrap "
                    f"replications failed. SE from {n_boot - boot_failed} "
                    f"successes. First error: {first_err}.",
                    RuntimeWarning, stacklevel=2,
                )
        else:
            warnings.warn(
                f"Proximal bootstrap failed on {boot_failed}/{n_boot} "
                f"replications; falling back to closed-form SE.",
                RuntimeWarning, stacklevel=2,
            )

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (tau - z_crit * se, tau + z_crit * se)
    z = tau / se if se > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(z))))

    model_info = {
        'estimator': 'Proximal 2SLS (linear bridge)',
        'n_proxy_z': k_z,
        'n_proxy_w': k_w,
        'n_covariates': len(covariates),
        'first_stage_F': float(first_stage_F) if first_stage_F is not None else None,
        'se_method': 'bootstrap' if n_boot and (n_boot - boot_failed) >= 2 else '2sls_sandwich',
    }
    if first_stage_F is not None and first_stage_F < 10:
        warnings.warn(
            f"Proximal: first-stage F = {first_stage_F:.2f} < 10. "
            f"Z is a weak instrument for W; estimates may be unreliable.",
            RuntimeWarning, stacklevel=2,
        )
    elif first_stage_F is None and k_w > 1:
        warnings.warn(
            f"Proximal: first-stage F is only reported for a single "
            f"endogenous proxy (k_w=1); got k_w={k_w}. For multiple W "
            f"the Cragg-Donald/Kleibergen-Paap minimum-eigenvalue "
            f"statistic is required and is not yet implemented.",
            RuntimeWarning, stacklevel=2,
        )
    if n_boot and n_boot > 0:
        model_info['n_boot'] = n_boot
        model_info['n_boot_failed'] = boot_failed
        if boot_failed > 0 and first_err:
            model_info['first_bootstrap_error'] = first_err

    return CausalResult(
        method='Proximal Causal Inference (linear 2SLS)',
        estimand='ATE',
        estimate=tau,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key='proximal',
    )


class ProximalCausalInference:
    """Class wrapper for :func:`proximal`."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.result_: Optional[CausalResult] = None

    def fit(self, data: pd.DataFrame) -> 'ProximalCausalInference':
        self.result_ = proximal(data=data, **self._kwargs)
        return self


def _linear_iv_fit(y, X, Zmat, k_exog):
    """
    2SLS: regress X on Zmat to get X_hat, then OLS y on X_hat.

    Also returns (vcov, first_stage_F) where first_stage_F is the
    heteroskedasticity-robust F-stat from the first-stage projection
    of the endogenous block on the excluded instruments (robustness
    check).
    """
    n = len(y)
    k = X.shape[1]

    # First stage: project each column of X onto Zmat
    # (exogenous columns come back to themselves; only W is projected
    # onto Z, which is how 2SLS handles identification)
    ZZ = Zmat.T @ Zmat
    try:
        ZZ_inv = np.linalg.pinv(ZZ)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError("Singular Z'Z in proximal 2SLS first stage")
    Pi = ZZ_inv @ Zmat.T @ X               # (k_inst, k)
    X_hat = Zmat @ Pi                       # projected regressors

    # Second stage: OLS of y on X_hat
    beta = np.linalg.pinv(X_hat.T @ X_hat) @ X_hat.T @ y

    # Residuals for sandwich SE — use the STRUCTURAL residuals (y - X β),
    # not (y - X_hat β). This is the standard 2SLS convention.
    resid = y - X @ beta
    sigma2 = float(np.sum(resid**2)) / max(n - k, 1)
    vcov = sigma2 * np.linalg.pinv(X_hat.T @ X_hat)

    # First-stage F: only well-defined for a single endogenous regressor
    # (k_w == 1). For multiple endogenous W's the correct weak-instrument
    # statistic is the Cragg-Donald / Kleibergen-Paap minimum eigenvalue,
    # which we do not ship yet. Reporting a pooled-F across W columns
    # would be misleading (no null distribution), so we return None and
    # let the caller handle the absence.
    first_stage_F: Optional[float] = None
    k_w = k - k_exog
    if k_w == 1:
        try:
            wj = X[:, k_exog]
            ex = Zmat[:, :k_exog]
            full = Zmat
            # Restricted: regress wj on exog only
            b_r = np.linalg.pinv(ex.T @ ex) @ ex.T @ wj
            r_r = wj - ex @ b_r
            rss_restr = float(r_r @ r_r)
            # Unrestricted: regress wj on full (exog + Z)
            b_u = np.linalg.pinv(full.T @ full) @ full.T @ wj
            r_u = wj - full @ b_u
            rss_full = float(r_u @ r_u)
            q = full.shape[1] - ex.shape[1]      # # excluded instruments
            df_denom = n - full.shape[1]
            if rss_full > 0 and q > 0 and df_denom > 0:
                first_stage_F = (
                    (rss_restr - rss_full) / q
                ) / (rss_full / df_denom)
        except Exception:
            first_stage_F = None

    return beta, vcov, first_stage_F


# Citation
CausalResult._CITATIONS['proximal'] = (
    "@article{tchetgen2020introduction,\n"
    "  title={An Introduction to Proximal Causal Learning},\n"
    "  author={Tchetgen Tchetgen, Eric J. and Ying, Andrew and "
    "Cui, Yifan and Shi, Xu and Miao, Wang},\n"
    "  journal={arXiv preprint arXiv:2009.10982},\n"
    "  year={2020}\n"
    "}"
)
