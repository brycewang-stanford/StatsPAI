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


def auto_cate_tuned(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    learners: Sequence[str] = ('s', 't', 'x', 'r', 'dr'),
    *,
    n_trials: int = 25,
    timeout: Optional[float] = None,
    search_space: Optional[Dict[str, List[Any]]] = None,
    n_folds: int = 5,
    alpha: float = 0.05,
    n_bootstrap: int = 200,
    random_state: int = 42,
    sampler: Optional[Any] = None,
    verbose: bool = False,
) -> AutoCATEResult:
    """Optuna-tune the nuisance GBM, then run :func:`auto_cate`.

    Parameters
    ----------
    data, y, treat, covariates, learners :
        Same semantics as :func:`auto_cate`.
    n_trials : int, default 25
        Upper bound on Optuna trials.
    timeout : float, optional
        Wall-clock limit for the tuning phase (seconds). If supplied
        tuning may stop before ``n_trials`` is reached.
    search_space : dict, optional
        Override the default search space. Keys must match
        :data:`DEFAULT_SEARCH_SPACE`; values are lists of categorical
        choices. Users with finer-grained needs should call
        :func:`auto_cate` directly with pre-built estimators.
    n_folds, alpha, n_bootstrap, random_state : see :func:`auto_cate`.
    sampler : ``optuna.samplers.BaseSampler``, optional
        Override Optuna's sampler. Defaults to :class:`~optuna.samplers.TPESampler`.
    verbose : bool, default False
        If ``False`` we silence Optuna logging (Optuna is chatty by
        default).

    Returns
    -------
    AutoCATEResult
        The standard result from :func:`auto_cate`, with the winner's
        ``model_info['tuned_params']`` and ``['n_trials']`` populated.
    """
    optuna = _require_optuna()

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    space = search_space if search_space is not None else DEFAULT_SEARCH_SPACE

    # Extract arrays once (same cleaning as auto_cate)
    Y, D, X, n = _prepare_data(data, y, treat, covariates)
    unique_d = np.unique(D)
    if not (len(unique_d) == 2 and set(unique_d) == {0.0, 1.0}):
        raise ValueError(
            f"Treatment must be binary (0/1), got unique values: {unique_d}"
        )

    def _objective(trial):
        params = _sample_params(trial, space)
        return _r_loss_on_nuisance(
            params, X, Y, D, n_folds=n_folds, random_state=random_state,
        )

    sampler = sampler or optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose,
    )

    best_params = dict(study.best_params)
    n_completed = len([t for t in study.trials
                       if t.state == optuna.trial.TrialState.COMPLETE])
    best_outcome, best_propensity = _build_models_from_params(
        best_params, random_state=random_state,
    )

    result = auto_cate(
        data, y=y, treat=treat, covariates=covariates,
        learners=learners,
        outcome_model=best_outcome,
        propensity_model=best_propensity,
        n_folds=n_folds,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    # Record tuning metadata on the winner's CausalResult
    result.best_result.model_info['tuned_params'] = best_params
    result.best_result.model_info['n_trials'] = n_completed
    result.best_result.model_info['best_r_loss_nuisance'] = float(study.best_value)

    # Also expose on the AutoCATEResult's selection rule for transparency
    result.selection_rule = (
        result.selection_rule
        + f" [nuisance tuned via {n_completed} Optuna trials]"
    )

    return result
