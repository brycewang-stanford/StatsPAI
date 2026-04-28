"""
Monte Carlo parametric g-formula (Robins 1986).

Classical formulation: fit a parametric model for the conditional
distribution of each time-varying confounder given the history, then
simulate counterfactual trajectories under a chosen treatment
intervention ``g`` and average the resulting outcomes.

This complements :func:`ice` (Bang-Robins 2005 ICE / iterative
conditional expectation) — MC g-formula is more flexible for dynamic
regimes and continuous/distributional contrasts, while ICE is
statistically more efficient for the expectation under a fixed
static regime.

Target estimand
---------------

.. math::

    \\psi(g) \\;=\\; \\mathbb{E}\\Big[ Y\\big(g(\\bar{L})\\big) \\Big]
    \\;=\\; \\int \\mathbb{E}[Y \\mid \\bar{L}, \\bar{A}=g(\\bar{L})]
         \\prod_{k=0}^{K-1} f(L_k \\mid \\bar{L}_{k-1}, \\bar{A}_{k-1}=g(\\bar{L}_{k-1}))
         \\, dL_0 \\cdots dL_{K-1}.

We approximate the integral by drawing ``n_simulations`` trajectories
from the fitted conditional densities and averaging the predicted
outcomes.

Reference
---------
Robins, J.M. (1986). "A new approach to causal inference in mortality
studies with a sustained exposure period—application to control of the
healthy worker survivor effect." *Mathematical Modelling*, 7, 1393-1512.

Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*,
Chapter 21.

Keil, A.P. et al. (2014). "The parametric g-formula for time-to-event
data: intuition and a worked example." *Epidemiology*, 25(6), 889-897.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats as _stats


@dataclass
class MCGFormulaResult:
    """Result of one or two Monte-Carlo g-formula arms."""

    value: float
    se: float
    ci: tuple
    n_simulations: int
    contrast_value: Optional[float] = None
    contrast_se: Optional[float] = None
    contrast_ci: Optional[tuple] = None
    strategies: Optional[dict] = None
    trajectories: Optional[pd.DataFrame] = None
    method: str = "MC-parametric-g-formula"

    def summary(self) -> str:
        s = (
            "Monte-Carlo parametric g-formula\n"
            f"  E[Y(g)]                      : {self.value:.4f}"
            f"  (SE {self.se:.4f}, 95% CI "
            f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}])\n"
            f"  n_simulations per arm        : {self.n_simulations}"
        )
        if self.contrast_value is not None:
            s += (
                "\n  Contrast (treat vs. control) : "
                f"{self.contrast_value:.4f}"
                f"  (SE {self.contrast_se:.4f}, 95% CI "
                f"[{self.contrast_ci[0]:.4f}, {self.contrast_ci[1]:.4f}])"
            )
        return s

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════
#  Internal fitters
# ═══════════════════════════════════════════════════════════════════════

def _fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients with an intercept column assumed in X."""
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _fit_logit(X: np.ndarray, y: np.ndarray, max_iter: int = 50,
               tol: float = 1e-7) -> np.ndarray:
    """Newton-Raphson MLE for binary logistic regression."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        W = mu * (1.0 - mu) + 1e-8
        grad = X.T @ (y - mu)
        H = -(X.T * W) @ X
        try:
            step = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, -grad, rcond=None)[0]
        new_beta = beta + step
        if np.max(np.abs(new_beta - beta)) < tol:
            beta = new_beta
            break
        beta = new_beta
    return beta


def _predict_ols(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ beta


def _predict_logit(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    eta = X @ beta
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))


def _design(*cols: np.ndarray) -> np.ndarray:
    """Stack a constant + the supplied 1-d or 2-d columns into a design matrix."""
    pieces = [np.ones((len(cols[0]), 1))]
    for c in cols:
        c = np.asarray(c, dtype=float)
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        pieces.append(c)
    return np.hstack(pieces)


def _is_binary(arr: np.ndarray) -> bool:
    u = np.unique(arr[~np.isnan(arr)])
    s = set(u.tolist())
    # True binary: observed values are a subset of {0, 1} AND at least
    # one of each level is present. A column that is identically 0 or
    # identically 1 is degenerate and should fall through to the
    # gaussian path (which will fit a constant with zero residual
    # variance) so we don't blow up in the logistic Newton-Raphson loop.
    return s.issubset({0.0, 1.0}) and len(s) == 2


# ═══════════════════════════════════════════════════════════════════════
#  Core fitter: one epoch over the wide-format history
# ═══════════════════════════════════════════════════════════════════════

class _FittedModels:
    """Container holding per-timepoint fitted confounder / outcome models."""

    def __init__(
        self,
        conf_models,          # list of list of (kind, beta, sd, feature_cols)
        outcome_model,        # (kind, beta, sd, feature_cols)
        conf_cols_by_t,       # list of list[str]
        treatment_cols,
        outcome_col,
        K,
    ):
        self.conf_models = conf_models
        self.outcome_model = outcome_model
        self.conf_cols_by_t = conf_cols_by_t
        self.treatment_cols = list(treatment_cols)
        self.outcome_col = outcome_col
        self.K = int(K)


def _fit_models(
    df: pd.DataFrame,
    treatment_cols: Sequence[str],
    conf_cols_by_t: Sequence[Sequence[str]],
    outcome_col: str,
) -> _FittedModels:
    """Fit conditional confounder and outcome models time by time."""
    K = len(treatment_cols)

    # conf_models[t][j] = (kind, beta, sd, feature_cols) for j-th confounder at time t
    conf_models: List[List[tuple]] = []
    history_cols: List[str] = []  # columns known up to the current timepoint

    for t in range(K):
        conf_list = list(conf_cols_by_t[t])
        per_t: List[tuple] = []
        for cname in conf_list:
            y = df[cname].values.astype(float)
            feat_cols = list(history_cols)
            X = _design(*(df[c].values.astype(float) for c in feat_cols)) \
                if feat_cols else np.ones((len(df), 1))
            if _is_binary(y):
                beta = _fit_logit(X, y)
                per_t.append(("binary", beta, None, feat_cols))
            else:
                beta = _fit_ols(X, y)
                resid = y - _predict_ols(beta, X)
                sd = float(np.std(resid, ddof=max(1, X.shape[1])))
                per_t.append(("gaussian", beta, sd, feat_cols))
            history_cols.append(cname)
        conf_models.append(per_t)
        # Treatment at t is assumed observed / part of strategy, but its
        # realised value enters subsequent confounder and outcome models.
        history_cols.append(treatment_cols[t])

    # Outcome model using full history
    Y = df[outcome_col].values.astype(float)
    X_out = _design(*(df[c].values.astype(float) for c in history_cols))
    if _is_binary(Y):
        beta_o = _fit_logit(X_out, Y)
        outcome_model = ("binary", beta_o, None, list(history_cols))
    else:
        beta_o = _fit_ols(X_out, Y)
        resid = Y - _predict_ols(beta_o, X_out)
        sd_o = float(np.std(resid, ddof=max(1, X_out.shape[1])))
        outcome_model = ("gaussian", beta_o, sd_o, list(history_cols))

    return _FittedModels(
        conf_models=conf_models,
        outcome_model=outcome_model,
        conf_cols_by_t=[list(c) for c in conf_cols_by_t],
        treatment_cols=list(treatment_cols),
        outcome_col=outcome_col,
        K=K,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Simulation under a chosen intervention
# ═══════════════════════════════════════════════════════════════════════

def _simulate(
    fitted: _FittedModels,
    strategy: Union[Sequence[float], Callable],
    seed_df: pd.DataFrame,       # baseline covariates are resampled from here
    n_sim: int,
    rng: np.random.Generator,
    return_trajectories: bool = False,
) -> tuple:
    """Draw ``n_sim`` trajectories and return (E[Y], trajectory DataFrame or None)."""
    # Resample row indices from observed data to approximate the baseline
    # covariate distribution f(L_0).
    idx = rng.integers(0, len(seed_df), size=n_sim)

    # history_values is a dict from column name -> vector of length n_sim
    history: dict = {}

    # Treatment strategy resolver: accept either list of ints/callable
    def _treat_at_t(t: int, hist: dict) -> np.ndarray:
        if callable(strategy):
            # strategy takes current timepoint index and history dict, returns vector.
            return np.asarray(strategy(t, hist), dtype=float)
        val = strategy[t]
        return np.full(n_sim, float(val))

    for t in range(fitted.K):
        conf_list = fitted.conf_cols_by_t[t]
        for j, cname in enumerate(conf_list):
            kind, beta, sd, feat_cols = fitted.conf_models[t][j]
            if t == 0 and not feat_cols:
                # Baseline confounder — draw from empirical distribution
                history[cname] = seed_df[cname].values.astype(float)[idx]
                continue
            feats = [history[c] for c in feat_cols]
            X = _design(*feats) if feats else np.ones((n_sim, 1))
            if kind == "binary":
                p = _predict_logit(beta, X)
                history[cname] = (rng.random(n_sim) < p).astype(float)
            else:
                mu = _predict_ols(beta, X)
                history[cname] = mu + rng.normal(0.0, sd, size=n_sim)
        # Now impose treatment at t
        a_t = _treat_at_t(t, history)
        history[fitted.treatment_cols[t]] = a_t

    # Predict terminal outcome
    kind_o, beta_o, sd_o, feat_cols_o = fitted.outcome_model
    X_o = _design(*(history[c] for c in feat_cols_o))
    if kind_o == "binary":
        y_pred = _predict_logit(beta_o, X_o)  # P(Y=1)
    else:
        y_pred = _predict_ols(beta_o, X_o)

    mean_y = float(np.mean(y_pred))

    traj_df = None
    if return_trajectories:
        traj = dict(history)
        traj["Y_pred"] = y_pred
        traj_df = pd.DataFrame(traj)
    return mean_y, traj_df


# ═══════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════

def gformula_mc(
    data: pd.DataFrame,
    treatment_cols: Sequence[str],
    confounder_cols: Union[Sequence[str], Sequence[Sequence[str]]],
    outcome_col: str,
    *,
    strategy: Union[Sequence[float], Callable] = (1, 1, 1),
    control_strategy: Union[Sequence[float], Callable, None] = None,
    id_col: Optional[str] = None,
    time_col: Optional[str] = None,
    n_simulations: int = 10_000,
    bootstrap: int = 200,
    alpha: float = 0.05,
    return_trajectories: bool = False,
    seed: Optional[int] = None,
) -> MCGFormulaResult:
    r"""
    Monte-Carlo parametric g-formula estimate of :math:`E[Y(g)]`.

    Fits a conditional density model for each time-varying confounder
    given the observed history, a regression for the outcome given the
    full history, and then simulates ``n_simulations`` counterfactual
    trajectories under the user-supplied treatment strategy.

    Parameters
    ----------
    data : DataFrame (wide format)
        One row per subject. Must contain all treatment, confounder and
        outcome columns.
    treatment_cols : list[str]
        Treatment column names, chronologically ordered.
    confounder_cols : list[str] | list[list[str]]
        Either a flat list (same confounders at every timepoint) or a
        list-of-lists with per-timepoint confounder sets. A confounder
        whose name ends in an index / time tag can be supplied as a
        flat list; for true time-varying confounders supply the nested
        form so the g-formula respects the correct temporal ordering.
    outcome_col : str
        Terminal outcome column.
    strategy : Sequence[float] or Callable, default (1,1,1)
        Either a static sequence of treatment values (e.g. ``[1]*K`` =
        always-treat), or a callable with signature
        ``strategy(t: int, history: dict) -> np.ndarray of length n_sim``
        for dynamic regimes that depend on simulated state.
    control_strategy : optional, same type as ``strategy``
        If provided, a second arm is simulated under the control
        strategy and a contrast (treat − control) is reported.
    id_col, time_col : str, optional
        Unused when ``data`` is already wide; kept for API symmetry
        with :func:`ice`.
    n_simulations : int, default 10 000
        Monte-Carlo trajectories per arm.
    bootstrap : int, default 200
        Non-parametric bootstrap replicates for SE / CI. Set to 0 to
        skip inference (point estimate only).
    alpha : float, default 0.05
    return_trajectories : bool, default False
        Attach the simulated trajectories DataFrame to the result.
    seed : int, optional

    Returns
    -------
    MCGFormulaResult

    Examples
    --------
    Static always-treat strategy:

    >>> res = sp.gformula_mc(
    ...     df,
    ...     treatment_cols=['A0', 'A1', 'A2'],
    ...     confounder_cols=[['L0'], ['L1'], ['L2']],
    ...     outcome_col='Y',
    ...     strategy=[1, 1, 1],
    ...     control_strategy=[0, 0, 0],
    ... )
    >>> print(res.summary())

    Dynamic regime: treat only when a time-varying biomarker is high.

    >>> def dynamic(t, hist):
    ...     return (hist[f'L{t}'] > 0).astype(float)
    >>> res = sp.gformula_mc(..., strategy=dynamic)
    """
    # ── Normalise column specs ────────────────────────────────────────
    if isinstance(confounder_cols[0], (list, tuple)):
        conf_cols_by_t = [list(c) for c in confounder_cols]
    else:
        conf_cols_by_t = [list(confounder_cols) for _ in range(len(treatment_cols))]

    K = len(treatment_cols)
    if len(conf_cols_by_t) != K:
        raise ValueError(
            f"confounder_cols has {len(conf_cols_by_t)} time slots but "
            f"treatment_cols has {K}."
        )
    if isinstance(strategy, (list, tuple, np.ndarray)) and len(strategy) != K:
        raise ValueError(
            f"strategy length {len(strategy)} does not match K={K}."
        )

    # Drop rows with any missing required column
    req = (
        list(treatment_cols)
        + [c for lst in conf_cols_by_t for c in lst]
        + [outcome_col]
    )
    df = data[req].dropna().copy()
    n = len(df)
    if n < 2:
        raise ValueError("Need at least 2 complete observations.")

    rng = np.random.default_rng(seed)

    # Point estimate
    fitted = _fit_models(df, treatment_cols, conf_cols_by_t, outcome_col)
    val_t, traj_t = _simulate(
        fitted, strategy, df, n_simulations, rng,
        return_trajectories=return_trajectories,
    )
    if control_strategy is not None:
        val_c, _ = _simulate(
            fitted, control_strategy, df, n_simulations, rng,
            return_trajectories=False,
        )
        contrast = val_t - val_c
    else:
        val_c = None
        contrast = None

    # Bootstrap inference
    if bootstrap and bootstrap > 0:
        boot_t = np.empty(bootstrap)
        boot_c = np.empty(bootstrap) if control_strategy is not None else None
        for b in range(bootstrap):
            b_idx = rng.integers(0, n, size=n)
            df_b = df.iloc[b_idx].reset_index(drop=True)
            fit_b = _fit_models(df_b, treatment_cols, conf_cols_by_t, outcome_col)
            mt, _ = _simulate(fit_b, strategy, df_b,
                              max(n_simulations // 4, 500), rng, False)
            boot_t[b] = mt
            if control_strategy is not None:
                mc, _ = _simulate(fit_b, control_strategy, df_b,
                                  max(n_simulations // 4, 500), rng, False)
                boot_c[b] = mc
        se_t = float(np.std(boot_t, ddof=1))
        z = float(_stats.norm.ppf(1 - alpha / 2))
        ci_t = (val_t - z * se_t, val_t + z * se_t)
        if control_strategy is not None:
            diffs = boot_t - boot_c
            se_c = float(np.std(diffs, ddof=1))
            ci_c = (contrast - z * se_c, contrast + z * se_c)
        else:
            se_c = None
            ci_c = None
    else:
        se_t = np.nan
        ci_t = (np.nan, np.nan)
        # Keep contrast SE consistent with point SE when bootstrap disabled:
        # NaN if a contrast was requested, None if it wasn't.
        se_c = np.nan if control_strategy is not None else None
        ci_c = (np.nan, np.nan) if control_strategy is not None else None

    _result = MCGFormulaResult(
        value=float(val_t),
        se=float(se_t),
        ci=(float(ci_t[0]), float(ci_t[1])),
        n_simulations=int(n_simulations),
        contrast_value=(float(contrast) if contrast is not None else None),
        contrast_se=(float(se_c) if se_c is not None else None),
        contrast_ci=(tuple(float(x) for x in ci_c) if ci_c is not None else None),
        strategies={
            "treat": list(strategy) if not callable(strategy) else "callable",
            "control": (list(control_strategy)
                        if (control_strategy is not None
                            and not callable(control_strategy))
                        else ("callable" if callable(control_strategy) else None)),
        },
        trajectories=traj_t if return_trajectories else None,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.gformula.gformula_mc",
            params={
                "treatment_cols": list(treatment_cols),
                "outcome_col": outcome_col,
                "id_col": id_col, "time_col": time_col,
                "n_simulations": n_simulations,
                "bootstrap": bootstrap,
                "alpha": alpha,
                "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["gformula_mc", "MCGFormulaResult"]
