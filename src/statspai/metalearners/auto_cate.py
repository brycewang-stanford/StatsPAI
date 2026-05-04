"""
``sp.auto_cate`` — one-line multi-learner CATE race with honest scoring.

Fits several meta-learners on the same data, scores each on held-out
data via the R-loss (Nie & Wager 2021), runs the BLP calibration test
(Chernozhukov et al. 2018) on each, and picks a winner.

Motivation
----------
Section 8 of StatsPAI's 0.9.3+1 retrospective called out "ML CATE
scheduling isn't as good as econml" as a known gap. ``auto_cate``
closes it: every meta-learner in this package is a first-class
citizen, so racing them and reporting a leaderboard is a natural
agent-native API.

Why R-loss
----------
Unlike ATE bootstrap SE (which is about the variance of the mean),
the R-loss directly measures held-out CATE prediction quality. It
is uniform across S/T/X/R/DR because it only depends on held-out
``tau_hat(X)``, ``m_hat(X)`` and ``e_hat(X)``:

    R(tau_hat) = E[ ((Y - m_hat) - tau_hat(X) * (D - e_hat))^2 ]

Selection rule
--------------
The winner is the learner with the lowest held-out R-loss. BLP
**beta_1** (≈ the ATE) and **beta_2** (the heterogeneity signal,
along with its p-value) are computed per learner and reported in the
leaderboard as diagnostics — they help the analyst understand *what*
the chosen learner is saying, but they are not part of the selection
gate. A user who wants to pick by heterogeneity signal can inspect
``result.leaderboard[['learner', 'r_loss', 'blp_beta2_pvalue']]``
and override manually.

Why not gate on beta_1: in this parametrization beta_1 equals the
ATE in the units of Y, not a calibration factor around 1 — so a
"beta_1 ≈ 1" gate would be testing "ATE is close to 1", which is
not a sensible selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult

# sklearn imports moved to function bodies — keeps ``import statspai``
# from pulling sklearn.model_selection / sklearn.ensemble through this
# file when the user never touches auto_cate.
from .metalearners import (
    SLearner,
    TLearner,
    XLearner,
    RLearner,
    DRLearner,
    metalearner,
    _default_outcome_model,
    _default_propensity_model,
    _default_cate_model,
    _get_propensity,
    _prepare_data,
)


_LEARNER_NAMES = {
    's': 'S-Learner', 't': 'T-Learner', 'x': 'X-Learner',
    'r': 'R-Learner', 'dr': 'DR-Learner',
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class AutoCATEResult:
    """Leaderboard + winner from ``sp.auto_cate``.

    Attributes
    ----------
    leaderboard : pd.DataFrame
        One row per learner with ATE, SE, CI, R-loss, BLP calibration
        columns, and CATE dispersion. Sorted by R-loss ascending.
    best_learner : str
        Full name of the chosen winner (e.g. ``"DR-Learner"``).
    best_result : CausalResult
        Full fitted result for the winner — supports ``.summary()``,
        ``.tidy()``, ``.glance()``.
    results : dict[str, CausalResult]
        All fitted learners keyed by short code (``'s'``, ``'t'``, ...).
    agreement : pd.DataFrame
        Pearson-rho matrix of CATE vectors across learners
        (in-sample). High agreement suggests stable heterogeneity;
        low agreement suggests model dependence.
    selection_rule : str
        Human-readable description of the rule that picked the
        winner.
    n_obs : int
        Sample size (after dropping NA on modelled columns).
    """

    leaderboard: pd.DataFrame
    best_learner: str
    best_result: CausalResult
    results: Dict[str, CausalResult]
    agreement: pd.DataFrame
    selection_rule: str
    n_obs: int

    def summary(self) -> str:
        lines = [
            '=' * 72,
            'auto_cate: CATE Learner Race',
            '=' * 72,
            f'  N obs:           {self.n_obs:,}',
            f'  Learners raced:  {len(self.results)} '
            f'({", ".join(self.leaderboard["learner"].tolist())})',
            f'  Winner:          {self.best_learner}',
            f'  Selection rule:  {self.selection_rule}',
            '-' * 72,
            'Leaderboard (sorted by R-loss; lower is better)',
            '-' * 72,
        ]
        # Pretty-printed leaderboard with rounding
        show = self.leaderboard.copy()
        for col in ['ate', 'se', 'ci_lower', 'ci_upper', 'r_loss',
                    'blp_beta1', 'blp_beta1_pvalue',
                    'blp_beta2', 'blp_beta2_pvalue',
                    'cate_std', 'cate_iqr']:
            if col in show.columns:
                show[col] = show[col].astype(float).round(4)
        lines.append(show.to_string(index=False))
        lines.append('-' * 72)
        lines.append('Cross-learner CATE agreement (Pearson rho)')
        lines.append('-' * 72)
        lines.append(self.agreement.round(3).to_string())
        lines.append('=' * 72)
        return '\n'.join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<AutoCATEResult n={self.n_obs} learners={len(self.results)} "
                f"winner={self.best_learner!r}>")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_learner(
    code: str,
    outcome_model,
    propensity_model,
    cate_model,
    n_folds: int,
):
    """Instantiate a fresh learner for a given short code."""
    from sklearn.base import clone
    code = code.lower()
    if code == 's':
        return SLearner(model=outcome_model)
    if code == 't':
        return TLearner(
            model_0=outcome_model,
            model_1=clone(outcome_model) if outcome_model is not None else None,
        )
    if code == 'x':
        return XLearner(
            model_0=outcome_model,
            model_1=clone(outcome_model) if outcome_model is not None else None,
            cate_model_0=cate_model,
            cate_model_1=clone(cate_model) if cate_model is not None else None,
            propensity_model=propensity_model,
        )
    if code == 'r':
        return RLearner(
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            cate_model=cate_model,
            n_folds=n_folds,
        )
    if code == 'dr':
        return DRLearner(
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            cate_model=cate_model,
            n_folds=n_folds,
        )
    raise ValueError(
        f"Unknown learner '{code}'. Valid codes: 's','t','x','r','dr'."
    )


def _cross_fit_nuisance(
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    outcome_model,
    propensity_model,
    n_folds: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared K-fold E[Y|X] and P(D=1|X) estimates reused across learners."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    n = len(Y)
    m_hat = np.zeros(n)
    e_hat = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        m = clone(outcome_model)
        m.fit(X[train_idx], Y[train_idx])
        m_hat[test_idx] = m.predict(X[test_idx])

        p = clone(propensity_model)
        p.fit(X[train_idx], D[train_idx])
        e_hat[test_idx] = _get_propensity(p, X[test_idx])
    return m_hat, e_hat


def _honest_cate_predictions(
    code: str,
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    outcome_model,
    propensity_model,
    cate_model,
    n_folds: int,
    random_state: int,
) -> np.ndarray:
    """Out-of-fold CATE predictions via per-fold refit of the learner."""
    from sklearn.model_selection import KFold
    n = len(Y)
    tau_hat = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, test_idx in kf.split(X):
        est = _build_learner(
            code,
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            cate_model=cate_model,
            n_folds=n_folds,
        )
        est.fit(X[train_idx], Y[train_idx], D[train_idx])
        tau_hat[test_idx] = est.effect(X[test_idx])
    return tau_hat


def _r_loss(
    tau_hat: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    m_hat: np.ndarray,
    e_hat: np.ndarray,
) -> float:
    """Nie-Wager R-loss evaluated on held-out CATE predictions."""
    resid = (Y - m_hat) - tau_hat * (D - e_hat)
    return float(np.mean(resid ** 2))


def _blp_calibration(
    tau_hat: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    m_hat: np.ndarray,
    e_hat: np.ndarray,
) -> Dict[str, float]:
    """BLP-of-CATE calibration test (Chernozhukov-Demirer-Duflo-FV 2018).

    Regresses ``Y - m_hat`` on ``[1, (D - e_hat), (D - e_hat) * (tau_hat - mean)]``
    with HC1 covariance.

    - ``beta_1 ~ 1`` indicates the CATE is well-calibrated to the
      population ATE.
    - ``beta_2 > 0`` and significant indicates genuine heterogeneity
      (not noise).

    The constant column is required because ``m_hat`` is estimated,
    so ``Y - m_hat`` is not exactly mean-zero in finite samples.
    """
    import statsmodels.api as sm
    D_centered = D - e_hat
    tau_centered = tau_hat - np.mean(tau_hat)
    # Intercept first, then beta_1 and beta_2 — mirrors diagnostics.blp_test.
    Z = np.column_stack([np.ones(len(Y)), D_centered, D_centered * tau_centered])
    Y_res = Y - m_hat
    try:
        ols = sm.OLS(Y_res, Z).fit(cov_type='HC1')
        b1 = float(ols.params[1])
        b1_se = float(ols.bse[1])
        # Two-sided p-value for H0: beta_1 = 1
        if b1_se > 0:
            t1 = (b1 - 1.0) / b1_se
            p1 = float(2 * (1 - stats.norm.cdf(abs(t1))))
        else:
            p1 = np.nan
        b2 = float(ols.params[2])
        p2 = float(ols.pvalues[2])
    except Exception:
        b1, p1, b2, p2 = np.nan, np.nan, np.nan, np.nan
    return {
        'blp_beta1': b1,
        'blp_beta1_pvalue': p1,
        'blp_beta2': b2,
        'blp_beta2_pvalue': p2,
    }


def _validate_learners(learners: Sequence[str]) -> List[str]:
    valid = set(_LEARNER_NAMES.keys())
    out: List[str] = []
    for lr in learners:
        code = lr.lower()
        if code not in valid:
            raise ValueError(
                f"Unknown learner '{lr}'. Valid: {sorted(valid)}."
            )
        if code not in out:
            out.append(code)
    if not out:
        raise ValueError("At least one learner is required.")
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_cate(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    learners: Sequence[str] = ('s', 't', 'x', 'r', 'dr'),
    outcome_model: Optional[Any] = None,
    propensity_model: Optional[Any] = None,
    cate_model: Optional[Any] = None,
    n_folds: int = 5,
    score: str = 'r_loss',
    alpha: float = 0.05,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> AutoCATEResult:
    """Race several meta-learners and return a scored leaderboard + winner.

    Parameters
    ----------
    data : pd.DataFrame
        Input data. Must contain ``y``, ``treat`` and all ``covariates``.
    y : str
        Outcome column name.
    treat : str
        Binary treatment column name (values in {0, 1}).
    covariates : list of str
        Effect-modifier columns used as features for every nuisance and
        CATE model.
    learners : sequence of str, default ``('s','t','x','r','dr')``
        Short codes of the meta-learners to race. Duplicates are
        ignored.
    outcome_model, propensity_model, cate_model : sklearn estimator, optional
        Override the default gradient-boosting models used for
        nuisance and final CATE fitting.
    n_folds : int, default 5
        Number of folds used for both the shared nuisance cross-fit
        and each learner's honest CATE prediction.
    score : {'r_loss'}, default ``'r_loss'``
        Currently only ``'r_loss'`` is implemented. Reserved for
        future expansion.
    alpha : float, default 0.05
        Significance level for both learner confidence intervals and
        the BLP-beta1 acceptance region used by the selection rule.
    n_bootstrap : int, default 200
        Bootstrap iterations for ATE standard error on non-DR learners
        (passed through to ``metalearner``).
    random_state : int, default 42
        Seed for all K-fold splits.

    Returns
    -------
    AutoCATEResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.auto_cate(df, y='wage', treat='training',
    ...                       covariates=['age', 'edu', 'exp'])
    >>> print(result.summary())
    >>> result.best_result.tidy()
    """
    if score != 'r_loss':
        raise NotImplementedError(
            f"score={score!r} is not supported yet. Use 'r_loss'."
        )

    codes = _validate_learners(learners)

    # Extract and validate data once
    Y, D, X, n = _prepare_data(data, y, treat, covariates)
    unique_d = np.unique(D)
    if not (len(unique_d) == 2 and set(unique_d) == {0.0, 1.0}):
        raise ValueError(
            f"Treatment must be binary (0/1), got unique values: {unique_d}"
        )

    # Shared nuisance: one cross-fit for all learners' scoring
    om = outcome_model if outcome_model is not None else _default_outcome_model()
    pm = propensity_model if propensity_model is not None else _default_propensity_model()
    m_hat, e_hat = _cross_fit_nuisance(
        X, Y, D,
        outcome_model=om, propensity_model=pm,
        n_folds=n_folds, random_state=random_state,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)

    # Fit each learner + score
    fitted_results: Dict[str, CausalResult] = {}
    in_sample_cate: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for code in codes:
        # Full fit via the canonical metalearner() API (in-sample CATE + SE)
        res = metalearner(
            data, y=y, treat=treat, covariates=covariates,
            learner=code,
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            cate_model=cate_model,
            n_folds=n_folds,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )
        fitted_results[code] = res
        cate = np.asarray(res.model_info['cate'])
        in_sample_cate[code] = cate

        # Honest held-out CATE for R-loss and BLP
        tau_oof = _honest_cate_predictions(
            code, X, Y, D,
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            cate_model=cate_model,
            n_folds=n_folds, random_state=random_state,
        )
        r_loss = _r_loss(tau_oof, Y, D, m_hat, e_hat)
        blp = _blp_calibration(tau_oof, Y, D, m_hat, e_hat)

        rows.append({
            'learner': _LEARNER_NAMES[code],
            'code': code,
            'ate': res.estimate,
            'se': res.se,
            'ci_lower': res.ci[0],
            'ci_upper': res.ci[1],
            'pvalue': res.pvalue,
            'r_loss': r_loss,
            **blp,
            'cate_std': float(np.std(cate)),
            'cate_iqr': float(np.percentile(cate, 75) - np.percentile(cate, 25)),
        })

    leaderboard = pd.DataFrame(rows).sort_values('r_loss').reset_index(drop=True)

    # Selection rule: lowest held-out R-loss.
    # BLP beta_1 (approx. ATE) and beta_2 (heterogeneity signal) are
    # reported in the leaderboard as diagnostics but do not gate the
    # selection — beta_1 equals the ATE in units of Y, not a
    # calibration factor around 1, so there is no natural "beta_1 == 1"
    # gate in this parametrization. Users who want to prefer learners
    # that find genuine heterogeneity can sort by beta_2_pvalue.
    winner_row = leaderboard.iloc[0]
    rule = "lowest held-out Nie-Wager R-loss"

    winner_code = winner_row['code']
    best_result = fitted_results[winner_code]
    best_name = _LEARNER_NAMES[winner_code]

    # Cross-learner agreement on in-sample CATE vectors.
    # A learner whose CATE vector is constant (e.g. S-Learner on a
    # linear outcome) would produce NaN correlations; we replace those
    # pairs with NaN -> 0.0 on the off-diagonal so the printed matrix
    # stays readable, and leave the diagonal at 1.0 by definition.
    names = [_LEARNER_NAMES[c] for c in codes]
    cate_matrix = np.column_stack([in_sample_cate[c] for c in codes])
    if len(codes) == 1:
        agr = pd.DataFrame([[1.0]], index=names, columns=names)
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.corrcoef(cate_matrix, rowvar=False)
        # NaNs come from zero-variance CATE vectors; coerce to 0.0 for
        # readability (and leave identity diagonal in place).
        if np.isnan(corr).any():
            corr = np.where(np.isnan(corr), 0.0, corr)
            np.fill_diagonal(corr, 1.0)
        agr = pd.DataFrame(corr, index=names, columns=names)

    # Drop 'code' helper column from the public leaderboard
    public_leaderboard = leaderboard.drop(columns=['code']).reset_index(drop=True)

    return AutoCATEResult(
        leaderboard=public_leaderboard,
        best_learner=best_name,
        best_result=best_result,
        results=fitted_results,
        agreement=agr,
        selection_rule=rule,
        n_obs=n,
    )
