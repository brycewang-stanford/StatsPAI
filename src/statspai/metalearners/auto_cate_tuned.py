"""
``sp.auto_cate_tuned`` — Optuna-tuned nuisance hyperparameters for
the CATE learner race.

Thin wrapper on top of :func:`statspai.metalearners.auto_cate`. First
tunes the GBM nuisance hyperparameters (outcome and propensity) via
Optuna's TPE sampler against held-out R-loss, then hands the tuned
models to :func:`auto_cate` and returns its :class:`AutoCATEResult`
unchanged except for two keys added to the winner's ``model_info``:

- ``tuned_params`` — the best trial's hyperparameter values;
- ``n_trials`` — number of Optuna trials actually evaluated.

Optuna is an **optional dependency**. If missing, this function
raises :class:`ImportError` with the install recipe; the rest of
``statspai`` works normally.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.model_selection import KFold

from ..core.results import CausalResult  # noqa: F401 - re-exported via result
from .auto_cate import (
    AutoCATEResult,
    auto_cate,
    _cross_fit_nuisance,
    _honest_cate_predictions,
    _r_loss,
)
from .metalearners import _prepare_data


_OPTUNA_INSTALL_HINT = (
    "sp.auto_cate_tuned requires optuna. Install with:\n"
    "    pip install 'statspai[tune]'\n"
    "or directly:\n"
    "    pip install optuna"
)


DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    'outcome_n_estimators': [100, 200, 400, 800],
    'outcome_max_depth': [2, 3, 4, 5, 6],
    'outcome_learning_rate': [0.01, 0.03, 0.05, 0.1],
    'outcome_subsample': [0.6, 0.8, 1.0],
    'propensity_n_estimators': [100, 200, 400],
    'propensity_max_depth': [2, 3, 4, 5],
    'propensity_learning_rate': [0.03, 0.05, 0.1],
}


# Per-learner search space — the CATE-stage GBM hyperparameters.
# Shared between outcome-style (S/T/X) and cate_model learners (R/DR)
# because all of them ultimately hit a GradientBoostingRegressor.
DEFAULT_PER_LEARNER_SEARCH_SPACE: Dict[str, List[Any]] = {
    'cate_n_estimators': [100, 200, 400],
    'cate_max_depth': [2, 3, 4, 5],
    'cate_learning_rate': [0.03, 0.05, 0.1],
    'cate_subsample': [0.6, 0.8, 1.0],
}


def _require_optuna():
    try:
        import optuna  # noqa: F401
    except ImportError as err:
        raise ImportError(_OPTUNA_INSTALL_HINT) from err
    return optuna


def _build_models_from_params(params: Dict[str, Any], random_state: int) -> Tuple[Any, Any]:
    outcome = GradientBoostingRegressor(
        n_estimators=int(params['outcome_n_estimators']),
        max_depth=int(params['outcome_max_depth']),
        learning_rate=float(params['outcome_learning_rate']),
        subsample=float(params['outcome_subsample']),
        random_state=random_state,
    )
    propensity = GradientBoostingClassifier(
        n_estimators=int(params['propensity_n_estimators']),
        max_depth=int(params['propensity_max_depth']),
        learning_rate=float(params['propensity_learning_rate']),
        random_state=random_state,
    )
    return outcome, propensity


def _sample_params(trial, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, choices in search_space.items():
        params[name] = trial.suggest_categorical(name, list(choices))
    return params


def _r_loss_on_nuisance(
    params: Dict[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    n_folds: int,
    random_state: int,
) -> float:
    """Held-out R-loss using these nuisance HPs and a naive ``tau_hat = 0``
    baseline. Lower is better.

    Why tau_hat=0: the nuisance hyperparameters affect ``m_hat`` and
    ``e_hat`` directly, and the R-loss lower-bound for a *correctly
    specified* nuisance is the variance of the residual ``Y - m(X)``
    projected through ``(D - e(X))``. We therefore tune nuisance
    *quality* independent of any specific CATE learner — better
    nuisance => lower residual variance => lower R-loss floor. This is
    the econml-style "nuisance cross-validation before CATE" pattern.
    """
    outcome, propensity = _build_models_from_params(params, random_state)
    m_hat, e_hat = _cross_fit_nuisance(
        X, Y, D,
        outcome_model=outcome, propensity_model=propensity,
        n_folds=n_folds, random_state=random_state,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    tau_zero = np.zeros_like(Y)
    return _r_loss(tau_zero, Y, D, m_hat, e_hat)


def _build_cate_model(params: Dict[str, Any], random_state: int):
    """GBM factory for the CATE-stage search space."""
    return GradientBoostingRegressor(
        n_estimators=int(params['cate_n_estimators']),
        max_depth=int(params['cate_max_depth']),
        learning_rate=float(params['cate_learning_rate']),
        subsample=float(params['cate_subsample']),
        random_state=random_state,
    )


def _r_loss_per_learner(
    code: str,
    params: Dict[str, Any],
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    m_hat: np.ndarray,
    e_hat: np.ndarray,
    outcome_model,
    propensity_model,
    n_folds: int,
    random_state: int,
) -> float:
    """Honest R-loss for a single learner with a specific CATE-stage HP."""
    cate_model = _build_cate_model(params, random_state)
    tau_oof = _honest_cate_predictions(
        code, X, Y, D,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        cate_model=cate_model,
        n_folds=n_folds, random_state=random_state,
    )
    return _r_loss(tau_oof, Y, D, m_hat, e_hat)


def auto_cate_tuned(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    learners: Sequence[str] = ('s', 't', 'x', 'r', 'dr'),
    *,
    tune: str = 'nuisance',
    n_trials: int = 25,
    n_trials_per_learner: Optional[int] = None,
    timeout: Optional[float] = None,
    search_space: Optional[Dict[str, List[Any]]] = None,
    per_learner_search_space: Optional[Dict[str, List[Any]]] = None,
    n_folds: int = 5,
    alpha: float = 0.05,
    n_bootstrap: int = 200,
    random_state: int = 42,
    sampler: Optional[Any] = None,
    verbose: bool = False,
) -> AutoCATEResult:
    """Optuna-tuned CATE learner race — nuisance, per-learner, or both.

    Parameters
    ----------
    data, y, treat, covariates, learners :
        Same semantics as :func:`auto_cate`.
    tune : {'nuisance', 'per_learner', 'both'}, default ``'nuisance'``
        Tuning regime:

        - ``'nuisance'`` — tune the shared outcome / propensity GBMs
          against held-out R-loss, then hand them to ``auto_cate``.
          (v0.9.5 behaviour.)
        - ``'per_learner'`` — keep default nuisance models; for each
          learner, tune its final-stage CATE model against held-out
          R-loss.
        - ``'both'`` — run ``'nuisance'`` first, then ``'per_learner'``
          using the tuned nuisance. Most expensive; most thorough.
    n_trials : int, default 25
        Budget for the nuisance-tuning study (ignored when
        ``tune == 'per_learner'``).
    n_trials_per_learner : int, optional
        Budget for each per-learner study. Defaults to
        ``max(5, n_trials // 3)``.
    timeout : float, optional
        Wall-clock limit per study (seconds).
    search_space, per_learner_search_space : dict, optional
        Override default spaces. See :data:`DEFAULT_SEARCH_SPACE` and
        :data:`DEFAULT_PER_LEARNER_SEARCH_SPACE`.
    n_folds, alpha, n_bootstrap, random_state, sampler, verbose :
        Passed through / see :func:`auto_cate`.

    Returns
    -------
    AutoCATEResult
        With winner's ``model_info`` populated based on ``tune``:

        - ``'nuisance'`` / ``'both'``: ``tuned_params`` (nuisance) +
          ``n_trials`` + ``best_r_loss_nuisance``.
        - ``'per_learner'`` / ``'both'``: ``per_learner_params``
          (dict keyed by learner short code, value = best CATE HP) +
          ``per_learner_r_loss`` (dict keyed by short code).
    """
    optuna = _require_optuna()

    if tune not in ('nuisance', 'per_learner', 'both'):
        raise ValueError(
            f"tune must be one of 'nuisance', 'per_learner', 'both'; "
            f"got {tune!r}"
        )

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = search_space if search_space is not None else DEFAULT_SEARCH_SPACE
    pl_space = (per_learner_search_space
                if per_learner_search_space is not None
                else DEFAULT_PER_LEARNER_SEARCH_SPACE)
    pl_trials = (n_trials_per_learner
                 if n_trials_per_learner is not None
                 else max(5, n_trials // 3))

    # Extract arrays once (same cleaning as auto_cate)
    Y, D, X, n = _prepare_data(data, y, treat, covariates)
    unique_d = np.unique(D)
    if not (len(unique_d) == 2 and set(unique_d) == {0.0, 1.0}):
        raise ValueError(
            f"Treatment must be binary (0/1), got unique values: {unique_d}"
        )

    # ------------------------------------------------------------------
    # Step 1: Nuisance tuning (modes 'nuisance' and 'both')
    # ------------------------------------------------------------------
    best_nuisance_params: Optional[Dict[str, Any]] = None
    best_r_loss_nuisance: Optional[float] = None
    n_completed_nuisance: int = 0
    best_outcome = None
    best_propensity = None

    if tune in ('nuisance', 'both'):
        def _objective_nuisance(trial):
            params = _sample_params(trial, space)
            return _r_loss_on_nuisance(
                params, X, Y, D,
                n_folds=n_folds, random_state=random_state,
            )

        nuisance_sampler = sampler or optuna.samplers.TPESampler(seed=random_state)
        nuisance_study = optuna.create_study(
            direction='minimize', sampler=nuisance_sampler,
        )
        nuisance_study.optimize(
            _objective_nuisance,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose,
        )
        best_nuisance_params = dict(nuisance_study.best_params)
        n_completed_nuisance = len([
            t for t in nuisance_study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        best_r_loss_nuisance = float(nuisance_study.best_value)
        best_outcome, best_propensity = _build_models_from_params(
            best_nuisance_params, random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Step 2: Per-learner CATE-stage tuning (modes 'per_learner' and 'both')
    # ------------------------------------------------------------------
    per_learner_params: Dict[str, Dict[str, Any]] = {}
    per_learner_r_loss: Dict[str, float] = {}

    if tune in ('per_learner', 'both'):
        # Pre-compute the nuisance once (shared across all learners' R-loss
        # evaluations) — uses the tuned nuisance if 'both', else defaults.
        om_shared = best_outcome
        pm_shared = best_propensity
        if om_shared is None:
            # Lazy import to avoid circular issues
            from .metalearners import (
                _default_outcome_model, _default_propensity_model,
            )
            om_shared = _default_outcome_model()
            pm_shared = _default_propensity_model()
        m_hat, e_hat = _cross_fit_nuisance(
            X, Y, D,
            outcome_model=om_shared, propensity_model=pm_shared,
            n_folds=n_folds, random_state=random_state,
        )
        e_hat = np.clip(e_hat, 0.01, 0.99)

        for code in learners:
            def _objective_pl(trial, _code=code):
                params = _sample_params(trial, pl_space)
                return _r_loss_per_learner(
                    _code, params, X, Y, D, m_hat, e_hat,
                    outcome_model=om_shared,
                    propensity_model=pm_shared,
                    n_folds=n_folds, random_state=random_state,
                )

            pl_sampler = optuna.samplers.TPESampler(seed=random_state)
            pl_study = optuna.create_study(direction='minimize', sampler=pl_sampler)
            pl_study.optimize(
                _objective_pl,
                n_trials=pl_trials,
                timeout=timeout,
                show_progress_bar=verbose,
            )
            per_learner_params[code] = dict(pl_study.best_params)
            per_learner_r_loss[code] = float(pl_study.best_value)

    # ------------------------------------------------------------------
    # Step 3: Run the learner race with the chosen configuration
    # ------------------------------------------------------------------
    # For per-learner mode we pick the CATE model corresponding to the
    # lowest-per-learner-R-loss learner as the *shared* cate_model hint;
    # auto_cate still races all learners and picks its winner by R-loss.
    if per_learner_params:
        best_pl_code = min(per_learner_r_loss, key=per_learner_r_loss.get)
        best_pl_cate_model = _build_cate_model(
            per_learner_params[best_pl_code], random_state=random_state,
        )
    else:
        best_pl_cate_model = None

    result = auto_cate(
        data, y=y, treat=treat, covariates=covariates,
        learners=learners,
        outcome_model=best_outcome,
        propensity_model=best_propensity,
        cate_model=best_pl_cate_model,
        n_folds=n_folds,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    # ------------------------------------------------------------------
    # Record metadata on the winner's CausalResult
    # ------------------------------------------------------------------
    info = result.best_result.model_info
    info['tune_mode'] = tune
    if best_nuisance_params is not None:
        info['tuned_params'] = best_nuisance_params
        info['n_trials'] = n_completed_nuisance
        info['best_r_loss_nuisance'] = best_r_loss_nuisance
    if per_learner_params:
        info['per_learner_params'] = per_learner_params
        info['per_learner_r_loss'] = per_learner_r_loss
        info['best_per_learner_code'] = best_pl_code
        info['n_trials_per_learner'] = pl_trials

    # Also expose on the AutoCATEResult's selection rule for transparency
    rule_parts: List[str] = []
    if best_nuisance_params is not None:
        rule_parts.append(
            f"nuisance tuned via {n_completed_nuisance} Optuna trials"
        )
    if per_learner_params:
        rule_parts.append(
            f"per-learner CATE tuned via {pl_trials} trials each "
            f"(best: {best_pl_code})"
        )
    if rule_parts:
        result.selection_rule = (
            result.selection_rule + " [" + "; ".join(rule_parts) + "]"
        )

    return result
