"""
Causal Mediation Analysis (Imai, Keele, Tingley 2010).

Decomposes total effect into:
- ACME (Average Causal Mediation Effect) — indirect via mediator
- ADE (Average Direct Effect) — not through mediator
- Total effect = ACME + ADE
- Proportion mediated = ACME / Total

Uses simulation-based inference (quasi-Bayesian) with bootstrap
confidence intervals.

Causal diagram:
    T → M → Y   (indirect / mediation path)
    T ------→ Y  (direct path)

Also provides :func:`mediate_interventional` for *interventional
(in)direct effects* (VanderWeele et al. 2014) — identified under weaker
assumptions than natural effects because they stochastically draw M
from its post-treatment marginal rather than fixing counterfactual
mediator values. Recommended when there are mediator-outcome
confounders affected by treatment.
"""

import warnings
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


def mediate(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[List[str]] = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    pvalue_method: str = 'bootstrap_sign',
    seed: int = 42,
) -> CausalResult:
    """
    Causal mediation analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Treatment variable (binary 0/1 or continuous).
    mediator : str
        Mediator variable.
    covariates : list of str, optional
        Pre-treatment covariates to control for.
    n_boot : int, default 1000
        Number of bootstrap replications for inference.
    alpha : float, default 0.05
        Significance level.
    pvalue_method : {'bootstrap_sign', 'wald'}, default 'bootstrap_sign'
        How each effect's p-value is computed:

        - ``'bootstrap_sign'`` (default, preserves v0.9.x behaviour):
          twice the fraction of bootstrap replications that fall on
          the side of zero opposite to the **bootstrap mean**. For
          well-behaved (symmetric) bootstrap distributions the mean
          and the point estimate coincide, but under skew this is the
          self-consistent "is the bootstrap distribution straddling
          zero?" test and is robust to skew.
        - ``'wald'``: conventional 2·(1−Φ(|θ̂/ŝe|)), using the
          bootstrap SE. Consistent with the Wald convention used
          across the rest of the causal-inference surface
          (:func:`sp.aipw`, :func:`sp.dml`, etc.).

        The matching kwarg is also accepted by
        :func:`sp.mediate_interventional`; pass the same value to
        keep reporting conventions aligned across a mediation study.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    CausalResult
        Contains ACME, ADE, total effect, and proportion mediated.

    Examples
    --------
    >>> # Job training → Skills → Wages
    >>> result = mediate(df, y='wage', treat='training',
    ...                  mediator='skills', covariates=['age', 'edu'])
    >>> print(result.summary())

    >>> # Wald p-values — align with sp.aipw / sp.dml convention
    >>> result = mediate(df, y='wage', treat='training',
    ...                  mediator='skills', pvalue_method='wald')
    """
    if pvalue_method not in ('bootstrap_sign', 'wald'):
        raise ValueError(
            f"pvalue_method must be 'bootstrap_sign' or 'wald'; "
            f"got {pvalue_method!r}"
        )
    analysis = MediationAnalysis(
        data=data, y=y, treat=treat, mediator=mediator,
        covariates=covariates, n_boot=n_boot, alpha=alpha,
        pvalue_method=pvalue_method, seed=seed,
    )
    _result = analysis.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.mediate",
            params={
                "y": y, "treat": treat, "mediator": mediator,
                "covariates": list(covariates) if covariates else None,
                "n_boot": n_boot, "alpha": alpha,
                "pvalue_method": pvalue_method, "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


class MediationAnalysis:
    """
    Causal Mediation Analysis estimator.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        mediator: str,
        covariates: Optional[List[str]] = None,
        n_boot: int = 1000,
        alpha: float = 0.05,
        pvalue_method: str = 'bootstrap_sign',
        seed: int = 42,
    ):
        if pvalue_method not in ('bootstrap_sign', 'wald'):
            raise ValueError(
                f"pvalue_method must be 'bootstrap_sign' or 'wald'; "
                f"got {pvalue_method!r}"
            )
        self.data = data
        self.y = y
        self.treat = treat
        self.mediator = mediator
        self.covariates = covariates or []
        self.pvalue_method = pvalue_method
        self.n_boot = n_boot
        self.alpha = alpha
        self.seed = seed

        self._validate()

    def _validate(self):
        for col in [self.y, self.treat, self.mediator] + self.covariates:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")

    def fit(self) -> CausalResult:
        """Estimate mediation effects with bootstrap inference."""
        cols = [self.y, self.treat, self.mediator] + self.covariates
        clean = self.data[cols].dropna()
        n = len(clean)

        Y = clean[self.y].values.astype(float)
        T = clean[self.treat].values.astype(float)
        M = clean[self.mediator].values.astype(float)

        if self.covariates:
            X = clean[self.covariates].values.astype(float)
        else:
            X = None

        # Point estimates
        acme, ade, total = self._estimate_effects(Y, T, M, X)
        prop_mediated = acme / total if abs(total) > 1e-10 else np.nan

        # Bootstrap inference
        rng = np.random.default_rng(self.seed)
        boot_acme = np.zeros(self.n_boot)
        boot_ade = np.zeros(self.n_boot)
        boot_total = np.zeros(self.n_boot)

        for b in range(self.n_boot):
            idx = rng.choice(n, size=n, replace=True)
            Y_b, T_b, M_b = Y[idx], T[idx], M[idx]
            X_b = X[idx] if X is not None else None

            try:
                a, d, t = self._estimate_effects(Y_b, T_b, M_b, X_b)
                boot_acme[b] = a
                boot_ade[b] = d
                boot_total[b] = t
            except Exception:
                boot_acme[b] = acme
                boot_ade[b] = ade
                boot_total[b] = total

        # Standard errors and CIs
        se_acme = float(np.std(boot_acme, ddof=1))
        se_ade = float(np.std(boot_ade, ddof=1))
        se_total = float(np.std(boot_total, ddof=1))

        lo = self.alpha / 2
        hi = 1 - self.alpha / 2
        ci_acme = (float(np.percentile(boot_acme, lo * 100)),
                   float(np.percentile(boot_acme, hi * 100)))
        ci_ade = (float(np.percentile(boot_ade, lo * 100)),
                  float(np.percentile(boot_ade, hi * 100)))
        ci_total = (float(np.percentile(boot_total, lo * 100)),
                    float(np.percentile(boot_total, hi * 100)))

        # P-values. Dispatch on the user's declared convention so a
        # single mediate()/mediate_interventional() study can report
        # consistent p-values regardless of which family member is used.
        pv_acme = self._pvalue(boot_acme, acme, se_acme)
        pv_ade = self._pvalue(boot_ade, ade, se_ade)
        pv_total = self._pvalue(boot_total, total, se_total)

        # Detail table
        detail = pd.DataFrame({
            'effect': ['ACME (indirect)', 'ADE (direct)', 'Total Effect', 'Prop. Mediated'],
            'estimate': [acme, ade, total, prop_mediated],
            'se': [se_acme, se_ade, se_total, np.nan],
            'ci_lower': [ci_acme[0], ci_ade[0], ci_total[0], np.nan],
            'ci_upper': [ci_acme[1], ci_ade[1], ci_total[1], np.nan],
            'pvalue': [pv_acme, pv_ade, pv_total, np.nan],
        })

        model_info = {
            'acme': acme,
            'ade': ade,
            'total_effect': total,
            'prop_mediated': prop_mediated,
            'n_boot': self.n_boot,
            'se_acme': se_acme,
            'se_ade': se_ade,
            'ci_acme': ci_acme,
            'ci_ade': ci_ade,
            'ci_total': ci_total,
            'pvalue_method': self.pvalue_method,
        }

        return CausalResult(
            method='Causal Mediation Analysis',
            estimand='ACME',
            estimate=acme,
            se=se_acme,
            pvalue=pv_acme,
            ci=ci_acme,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key='mediation',
        )

    def _estimate_effects(self, Y, T, M, X):
        """
        Estimate ACME, ADE, and total effect using the product method
        with OLS mediator and outcome models.

        Mediator model: M = a0 + a1*T + a2'*X + e_m
        Outcome model:  Y = b0 + b1*T + b2*M + b3'*X + e_y

        ACME = a1 * b2 (indirect effect)
        ADE  = b1       (direct effect)
        Total = a1*b2 + b1
        """
        n = len(Y)

        # Build design matrices
        if X is not None:
            Z_med = np.column_stack([np.ones(n), T, X])
            Z_out = np.column_stack([np.ones(n), T, M, X])
        else:
            Z_med = np.column_stack([np.ones(n), T])
            Z_out = np.column_stack([np.ones(n), T, M])

        # Mediator model: M ~ T + X
        try:
            a = np.linalg.lstsq(Z_med, M, rcond=None)[0]
        except np.linalg.LinAlgError:
            a = np.zeros(Z_med.shape[1])
        a1 = a[1]  # coefficient on T

        # Outcome model: Y ~ T + M + X
        try:
            b = np.linalg.lstsq(Z_out, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            b = np.zeros(Z_out.shape[1])
        b1 = b[1]  # direct effect (T coefficient)
        b2 = b[2]  # mediator coefficient

        acme = a1 * b2  # indirect effect
        ade = b1         # direct effect
        total = acme + ade

        return float(acme), float(ade), float(total)

    def _pvalue(self, boot_samples, point, se):
        """Per-effect p-value using the convention declared by self.pvalue_method."""
        if self.pvalue_method == 'wald':
            if se and se > 0 and np.isfinite(point):
                z = point / se
                return float(2 * (1 - stats.norm.cdf(abs(z))))
            return float('nan')
        return self._boot_pvalue(boot_samples)

    @staticmethod
    def _boot_pvalue(boot_samples):
        """Two-sided bootstrap-CI-inversion p-value."""
        n = len(boot_samples)
        # Proportion of bootstrap samples with sign opposite to point estimate
        mean = np.mean(boot_samples)
        if mean >= 0:
            p = 2 * np.mean(boot_samples <= 0)
        else:
            p = 2 * np.mean(boot_samples >= 0)
        return float(np.clip(p, 1 / n, 1.0))


# Citation
CausalResult._CITATIONS['mediation'] = (
    "@article{imai2010general,\n"
    "  title={A General Approach to Causal Mediation Analysis},\n"
    "  author={Imai, Kosuke and Keele, Luke and Tingley, Dustin},\n"
    "  journal={Psychological Methods},\n"
    "  volume={15},\n"
    "  number={4},\n"
    "  pages={309--334},\n"
    "  year={2010},\n"
    "  publisher={American Psychological Association}\n"
    "}"
)


def mediate_interventional(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[List[str]] = None,
    tv_confounders: Optional[List[str]] = None,
    n_mc: int = 500,
    n_boot: int = 500,
    alpha: float = 0.05,
    pvalue_method: str = 'bootstrap_sign',
    seed: int = 42,
) -> CausalResult:
    """
    Interventional (in)direct effects (VanderWeele, Vansteelandt,
    Robins 2014).

    Decomposes the total effect into:

    * **IIE** (Interventional Indirect Effect): E[Y(1, G_{M|1})] -
      E[Y(1, G_{M|0})], i.e. the effect of shifting M's
      *post-treatment distribution* from its D=0 draw to its D=1 draw,
      while holding D fixed at 1.
    * **IDE** (Interventional Direct Effect): E[Y(1, G_{M|0})] -
      E[Y(0, G_{M|0})].
    * **Total** = IIE + IDE = E[Y(1, G_{M|1})] - E[Y(0, G_{M|0})].

    Here :math:`G_{M|d}` is the random draw from the marginal
    distribution of :math:`M` under treatment :math:`D = d` (integrated
    over covariates).

    Interventional effects are identified under the standard mediation
    assumptions **minus** the cross-world independence requirement —
    which makes them valid even when there are **treatment-induced
    mediator-outcome confounders** (``tv_confounders``). Natural effects
    are not generally identified in that case.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Binary treatment (0/1).
    mediator : str
        Mediator variable.
    covariates : list of str, optional
        Baseline (pre-treatment) covariates.
    tv_confounders : list of str, optional
        Treatment-induced mediator-outcome confounders (variables
        affected by D that confound the M-Y relationship). These enter
        the outcome model but **not** the M-marginalization.
    n_mc : int, default 500
        Monte Carlo draws of M for the stochastic intervention.
    n_boot : int, default 500
        Nonparametric bootstrap replications.
    alpha : float, default 0.05
        Significance level.
    pvalue_method : {'bootstrap_sign', 'wald'}, default 'bootstrap_sign'
        How the per-effect p-value is computed:

        - ``'bootstrap_sign'`` (default, matches :func:`sp.mediate`):
          the bootstrap-CI-inversion p-value, i.e. twice the fraction
          of bootstrap replications on the opposite side of zero
          from the point estimate. Respects skewed bootstrap
          distributions.
        - ``'wald'``: the conventional 2*(1-Φ(|θ̂/ŝe|)) p-value,
          using the bootstrap SE. Consistent with the Wald
          pvalue convention used across the rest of StatsPAI's
          causal-inference surface (e.g. ``sp.aipw``, ``sp.dml``).
    seed : int, default 42
        Random seed.

    Returns
    -------
    CausalResult
        ``estimate`` is IIE; full decomposition lives in ``detail``.

    Notes
    -----
    **Linear outcome model assumption.** The current implementation
    hard-codes an OLS outcome regression Y ~ D + M + X_base + X_tv.
    This permits the Monte-Carlo integration to collapse analytically
    via the linearity of OLS predictions in the treatment-induced
    confounder block (X_tv), giving an O(n_mc + n) cost instead of
    O(n × n_mc). Non-linear outcome models (gradient boosting, neural
    nets, etc.) would break this vectorisation and are not currently
    supported — passing a custom learner is not exposed via the API.

    References
    ----------
    VanderWeele, T.J., Vansteelandt, S. and Robins, J.M. (2014).
    "Effect decomposition in the presence of an exposure-induced
    mediator-outcome confounder." *Epidemiology*, 25(2), 300-306. [@vanderweele2014effect]
    """
    if pvalue_method not in ('bootstrap_sign', 'wald'):
        raise ValueError(
            f"pvalue_method must be 'bootstrap_sign' or 'wald'; "
            f"got '{pvalue_method}'"
        )
    covariates = list(covariates or [])
    tv_confounders = list(tv_confounders or [])
    cols = [y, treat, mediator] + covariates + tv_confounders
    df = data[cols].dropna().reset_index(drop=True)
    n = len(df)

    Y = df[y].values.astype(float)
    D = df[treat].values.astype(float)
    M = df[mediator].values.astype(float)

    if not set(np.unique(D)).issubset({0, 1}):
        raise ValueError("treat must be binary (0/1).")

    X_base = df[covariates].values.astype(float) if covariates else np.zeros((n, 0))
    X_tv = df[tv_confounders].values.astype(float) if tv_confounders else np.zeros((n, 0))

    def _compute(Y_, D_, M_, Xb_, Xtv_, rng):
        n_ = len(Y_)
        p_base = Xb_.shape[1]
        p_tv = Xtv_.shape[1]

        # Outcome model: Y ~ [1, D, M, X_base, X_tv]
        design_y = np.column_stack([np.ones(n_), D_, M_, Xb_, Xtv_])
        beta_y, *_ = np.linalg.lstsq(design_y, Y_, rcond=None)
        b0, bD, bM = float(beta_y[0]), float(beta_y[1]), float(beta_y[2])
        b_base = beta_y[3 : 3 + p_base]
        b_tv = beta_y[3 + p_base : 3 + p_base + p_tv]

        # Mediator model: Gaussian M ~ [1, D, X_base]
        design_m = np.column_stack([np.ones(n_), D_, Xb_])
        beta_m, *_ = np.linalg.lstsq(design_m, M_, rcond=None)
        resid_m = M_ - design_m @ beta_m
        sigma_m = float(np.std(resid_m, ddof=max(1, design_m.shape[1])))
        sigma_m = max(sigma_m, 1e-6)

        # μ_M(d, x_base) = α + δ*d + β' x_base
        alpha_m, delta_m = float(beta_m[0]), float(beta_m[1])
        beta_m_base = beta_m[2 : 2 + p_base]

        # MC pool: each j is a (x_base_j, m_d1_j, m_d0_j) triple.
        # x_base drawn from the empirical marginal; m drawn from the
        # conditional Gaussian given D=d and that x_base row.
        sample_idx = rng.integers(0, n_, size=n_mc)
        Xb_pool = Xb_[sample_idx]                                   # (n_mc, p_base)
        eps = rng.standard_normal(n_mc)
        base_effect_m = Xb_pool @ beta_m_base                       # (n_mc,)
        draws_d1 = alpha_m + delta_m * 1.0 + base_effect_m + sigma_m * eps
        draws_d0 = alpha_m + delta_m * 0.0 + base_effect_m + sigma_m * eps

        # Y is linear in (D, M, X_base, X_tv) — we exploit this to fully
        # vectorize. The expectation over the original units' X_tv rows
        # reduces to b_tv · mean(X_tv) because the outcome is additive.
        tv_contrib = float(b_tv @ Xtv_.mean(axis=0)) if p_tv else 0.0
        base_mean_pool = Xb_pool @ b_base                           # (n_mc,)

        def _EY(d_val, m_draws):
            # b0 + bD*d + bM*m + b_base·x_base + b_tv·mean(X_tv)
            inner = b0 + bD * d_val + bM * m_draws + base_mean_pool
            return float(np.mean(inner)) + tv_contrib

        EY_11 = _EY(1.0, draws_d1)
        EY_10 = _EY(1.0, draws_d0)
        EY_00 = _EY(0.0, draws_d0)

        iie = EY_11 - EY_10
        ide = EY_10 - EY_00
        total = EY_11 - EY_00
        return float(iie), float(ide), float(total)

    rng = np.random.default_rng(seed)
    iie_hat, ide_hat, total_hat = _compute(Y, D, M, X_base, X_tv, rng)

    # Bootstrap — failures leave NaN and are reported rather than silently
    # substituted with the point estimate (which would shrink SE).
    boot_iie = np.full(n_boot, np.nan)
    boot_ide = np.full(n_boot, np.nan)
    boot_total = np.full(n_boot, np.nan)
    n_failed = 0
    first_err: Optional[str] = None
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            a, d, t = _compute(
                Y[idx], D[idx], M[idx], X_base[idx], X_tv[idx], rng
            )
            boot_iie[b], boot_ide[b], boot_total[b] = a, d, t
        except Exception as e:
            n_failed += 1
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"

    n_success = n_boot - n_failed
    if n_success < 2:
        raise RuntimeError(
            f"Interventional mediation bootstrap failed on {n_failed}/{n_boot} "
            f"replications (only {n_success} succeeded; need ≥2 for SE). "
            f"First error: {first_err}."
        )
    if n_failed > 0:
        frac = n_failed / n_boot
        warnings.warn(
            f"Interventional mediation: {n_failed}/{n_boot} bootstrap "
            f"replications failed ({frac:.1%}). SE/CI computed over "
            f"{n_success} successes. First error: {first_err}.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _ci_pv(boot, point):
        se = float(np.nanstd(boot, ddof=1))
        lo = float(np.nanpercentile(boot, 100 * alpha / 2))
        hi = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))
        boot_clean = boot[~np.isnan(boot)]
        if len(boot_clean) == 0:
            return se, (lo, hi), float('nan')

        if pvalue_method == 'wald':
            # Conventional Wald p-value: 2 * (1 - Φ(|θ̂/ŝe|)).
            # Consistent with sp.aipw / sp.dml family.
            if se > 0:
                z = point / se
                p = float(2 * (1 - stats.norm.cdf(abs(z))))
            else:
                p = float('nan')
            return se, (lo, hi), p

        # 'bootstrap_sign' (default): bootstrap CI-inversion p-value.
        if point >= 0:
            p = 2 * float(np.mean(boot_clean <= 0))
        else:
            p = 2 * float(np.mean(boot_clean >= 0))
        p = float(np.clip(p, 1 / max(len(boot_clean), 1), 1.0))
        return se, (lo, hi), p

    se_i, ci_i, pv_i = _ci_pv(boot_iie, iie_hat)
    se_d, ci_d, pv_d = _ci_pv(boot_ide, ide_hat)
    se_t, ci_t, pv_t = _ci_pv(boot_total, total_hat)

    detail = pd.DataFrame({
        'effect': ['IIE (interventional indirect)',
                   'IDE (interventional direct)',
                   'Total'],
        'estimate': [iie_hat, ide_hat, total_hat],
        'se': [se_i, se_d, se_t],
        'ci_lower': [ci_i[0], ci_d[0], ci_t[0]],
        'ci_upper': [ci_i[1], ci_d[1], ci_t[1]],
        'pvalue': [pv_i, pv_d, pv_t],
    })

    model_info = {
        'estimator': 'Interventional (in)direct effects',
        'iie': iie_hat,
        'ide': ide_hat,
        'total_effect': total_hat,
        'n_boot': n_boot,
        'n_boot_failed': n_failed,
        'n_boot_success': n_success,
        'n_mc': n_mc,
        'pvalue_method': pvalue_method,
        'tv_confounders': tv_confounders,
        'covariates': covariates,
    }
    if n_failed > 0:
        model_info['first_bootstrap_error'] = first_err

    _result = CausalResult(
        method='Interventional Mediation Analysis',
        estimand='IIE',
        estimate=iie_hat,
        se=se_i,
        pvalue=pv_i,
        ci=ci_i,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key='mediation_interventional',
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.mediate_interventional",
            params={
                "y": y, "treat": treat, "mediator": mediator,
                "covariates": list(covariates) if covariates else None,
                "tv_confounders": list(tv_confounders) if tv_confounders else None,
                "n_mc": n_mc, "n_boot": n_boot,
                "alpha": alpha,
                "pvalue_method": pvalue_method, "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


CausalResult._CITATIONS['mediation_interventional'] = (
    "@article{vanderweele2014effect,\n"
    "  title={Effect Decomposition in the Presence of an "
    "Exposure-Induced Mediator-Outcome Confounder},\n"
    "  author={VanderWeele, Tyler J. and Vansteelandt, Stijn and "
    "Robins, James M.},\n"
    "  journal={Epidemiology},\n"
    "  volume={25},\n"
    "  number={2},\n"
    "  pages={300--306},\n"
    "  year={2014}\n"
    "}"
)
