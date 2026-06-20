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

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# sklearn is imported lazily inside the functions that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches auto_cate_tuned.

from ..core.results import CausalResult  # noqa: F401 - re-exported via result
from ..exceptions import DataInsufficient, MethodIncompatibility
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
    "outcome_n_estimators": [100, 200, 400, 800],
    "outcome_max_depth": [2, 3, 4, 5, 6],
    "outcome_learning_rate": [0.01, 0.03, 0.05, 0.1],
    "outcome_subsample": [0.6, 0.8, 1.0],
    "propensity_n_estimators": [100, 200, 400],
    "propensity_max_depth": [2, 3, 4, 5],
    "propensity_learning_rate": [0.03, 0.05, 0.1],
}


# Per-learner search space — the CATE-stage GBM hyperparameters.
# Shared between outcome-style (S/T/X) and cate_model learners (R/DR)
# because all of them ultimately hit a GradientBoostingRegressor.
DEFAULT_PER_LEARNER_SEARCH_SPACE: Dict[str, List[Any]] = {
    "cate_n_estimators": [100, 200, 400],
    "cate_max_depth": [2, 3, 4, 5],
    "cate_learning_rate": [0.03, 0.05, 0.1],
    "cate_subsample": [0.6, 0.8, 1.0],
}


CovariatesArg = Union[Sequence[str], str]
LearnersArg = Union[Sequence[str], str]
_VALID_LEARNERS = {"s", "t", "x", "r", "dr"}

_AUTO_CATE_TUNED_ALTERNATIVES = [
    "sp.auto_cate_tuned",
    "sp.auto_cate",
    "sp.metalearner",
]


def _auto_cate_tuned_error(
    message: str,
    *,
    diagnostics: Optional[Dict[str, Any]] = None,
    recovery_hint: str = "Check auto_cate_tuned inputs and option values.",
) -> MethodIncompatibility:
    return MethodIncompatibility(
        message,
        recovery_hint=recovery_hint,
        diagnostics=diagnostics,
        alternative_functions=_AUTO_CATE_TUNED_ALTERNATIVES,
    )


def _normalize_covariates(raw: CovariatesArg) -> List[str]:
    if isinstance(raw, str):
        return [raw]
    try:
        covariates = list(raw)
    except TypeError as err:
        raise _auto_cate_tuned_error(
            "covariates must be a column name or a sequence of column names.",
            diagnostics={"covariates": repr(raw)},
            recovery_hint="Pass covariates=['x1', 'x2'] or a single column name.",
        ) from err
    if not covariates:
        raise DataInsufficient(
            "auto_cate_tuned requires at least one covariate.",
            recovery_hint="Pass one or more effect-modifier columns.",
            diagnostics={"n_covariates": 0},
            alternative_functions=_AUTO_CATE_TUNED_ALTERNATIVES,
        )
    for idx, covariate in enumerate(covariates):
        if not isinstance(covariate, str) or not covariate:
            raise _auto_cate_tuned_error(
                "covariates must contain non-empty column-name strings.",
                diagnostics={"index": idx, "value": repr(covariate)},
                recovery_hint="Pass covariates as column-name strings.",
            )
    return covariates


def _normalize_learners(raw: LearnersArg) -> List[str]:
    if isinstance(raw, str):
        raw_learners = [raw]
    else:
        try:
            raw_learners = list(raw)
        except TypeError as err:
            raise _auto_cate_tuned_error(
                "learners must be a learner code or a sequence of learner codes.",
                diagnostics={"learners": repr(raw)},
                recovery_hint="Pass learners=('t', 'dr') or a single learner code.",
            ) from err
    out: List[str] = []
    for idx, learner in enumerate(raw_learners):
        if not isinstance(learner, str) or not learner:
            raise _auto_cate_tuned_error(
                "learners must contain non-empty learner-code strings.",
                diagnostics={"index": idx, "value": repr(learner)},
                recovery_hint="Use learner codes 's', 't', 'x', 'r', or 'dr'.",
            )
        code = learner.lower()
        if code not in _VALID_LEARNERS:
            raise _auto_cate_tuned_error(
                f"Unknown learner {learner!r}.",
                diagnostics={
                    "learner": learner,
                    "valid": sorted(_VALID_LEARNERS),
                },
                recovery_hint="Use learner codes 's', 't', 'x', 'r', or 'dr'.",
            )
        if code not in out:
            out.append(code)
    if not out:
        raise DataInsufficient(
            "auto_cate_tuned requires at least one learner.",
            recovery_hint="Pass one or more learner codes to tune.",
            diagnostics={"n_learners": 0},
            alternative_functions=_AUTO_CATE_TUNED_ALTERNATIVES,
        )
    return out


def _validate_search_space(
    search_space: Dict[str, List[Any]],
    *,
    context: str,
) -> None:
    if not isinstance(search_space, dict):
        raise _auto_cate_tuned_error(
            f"{context} search space must be a dictionary.",
            diagnostics={"context": context, "type": type(search_space).__name__},
            recovery_hint="Pass a dict mapping hyperparameter names to choices.",
        )
    for name, choices in search_space.items():
        if not isinstance(name, str) or not name:
            raise _auto_cate_tuned_error(
                f"{context} search-space keys must be non-empty strings.",
                diagnostics={"context": context, "name": repr(name)},
                recovery_hint="Use string hyperparameter names in search_space.",
            )
        if isinstance(choices, (str, bytes)):
            raise _auto_cate_tuned_error(
                f"{context} search-space choices must be sequences.",
                diagnostics={
                    "context": context,
                    "name": name,
                    "choices": repr(choices),
                },
                recovery_hint="Give each tuned hyperparameter at least one choice.",
            )
        try:
            choices_list = list(choices)
        except TypeError as err:
            raise _auto_cate_tuned_error(
                f"{context} search-space choices must be sequences.",
                diagnostics={
                    "context": context,
                    "name": name,
                    "choices": repr(choices),
                },
                recovery_hint="Give each tuned hyperparameter at least one choice.",
            ) from err
        if not choices_list:
            raise _auto_cate_tuned_error(
                f"{context} search-space choices must be non-empty.",
                diagnostics={
                    "context": context,
                    "name": name,
                    "choices": repr(choices),
                },
                recovery_hint="Give each tuned hyperparameter at least one choice.",
            )


def _require_optuna() -> Any:
    try:
        import optuna  # noqa: F401
    except ImportError as err:
        raise ImportError(_OPTUNA_INSTALL_HINT) from err
    return optuna


def _build_models_from_params(
    params: Dict[str, Any],
    random_state: int,
) -> Tuple[Any, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    outcome = GradientBoostingRegressor(
        n_estimators=int(params["outcome_n_estimators"]),
        max_depth=int(params["outcome_max_depth"]),
        learning_rate=float(params["outcome_learning_rate"]),
        subsample=float(params["outcome_subsample"]),
        random_state=random_state,
    )
    propensity = GradientBoostingClassifier(
        n_estimators=int(params["propensity_n_estimators"]),
        max_depth=int(params["propensity_max_depth"]),
        learning_rate=float(params["propensity_learning_rate"]),
        random_state=random_state,
    )
    return outcome, propensity


def _sample_params(trial: Any, search_space: Dict[str, List[Any]]) -> Dict[str, Any]:
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
        X,
        Y,
        D,
        outcome_model=outcome,
        propensity_model=propensity,
        n_folds=n_folds,
        random_state=random_state,
    )
    e_hat = np.clip(e_hat, 0.01, 0.99)
    tau_zero = np.zeros_like(Y)
    return _r_loss(tau_zero, Y, D, m_hat, e_hat)


def _build_cate_model(params: Dict[str, Any], random_state: int) -> Any:
    """GBM factory for the CATE-stage search space."""
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=int(params["cate_n_estimators"]),
        max_depth=int(params["cate_max_depth"]),
        learning_rate=float(params["cate_learning_rate"]),
        subsample=float(params["cate_subsample"]),
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
    outcome_model: Any,
    propensity_model: Any,
    n_folds: int,
    random_state: int,
) -> float:
    """Honest R-loss for a single learner with a specific CATE-stage HP."""
    cate_model = _build_cate_model(params, random_state)
    tau_oof = _honest_cate_predictions(
        code,
        X,
        Y,
        D,
        outcome_model=outcome_model,
        propensity_model=propensity_model,
        cate_model=cate_model,
        n_folds=n_folds,
        random_state=random_state,
    )
    return _r_loss(tau_oof, Y, D, m_hat, e_hat)


def auto_cate_tuned(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: CovariatesArg,
    learners: LearnersArg = ("s", "t", "x", "r", "dr"),
    *,
    tune: str = "nuisance",
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

    Examples
    --------
    Requires the optional ``optuna`` dependency
    (``pip install 'statspai[tune]'``).

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.integers(0, 2, size=n)
    >>> tau = 1.0 + 0.5 * x1  # heterogeneous effect
    >>> y = 2.0 + x2 + tau * d + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    >>> res = sp.auto_cate_tuned(  # doctest: +SKIP
    ...     df, y="y", treat="d", covariates=["x1", "x2"],
    ...     learners=("s", "t"), tune="nuisance",
    ...     n_trials=3, n_folds=2, n_bootstrap=50, random_state=42,
    ... )
    >>> res.best_result.model_info["tune_mode"]  # doctest: +SKIP
    'nuisance'
    """
    if tune not in ("nuisance", "per_learner", "both"):
        raise _auto_cate_tuned_error(
            f"tune must be one of 'nuisance', 'per_learner', 'both'; " f"got {tune!r}",
            diagnostics={"tune": tune},
            recovery_hint="Use tune='nuisance', 'per_learner', or 'both'.",
        )
    if not isinstance(data, pd.DataFrame):
        raise _auto_cate_tuned_error(
            "data must be a pandas DataFrame.",
            diagnostics={"type": type(data).__name__},
            recovery_hint=(
                "Pass a pandas DataFrame with outcome, treatment, and " "covariates."
            ),
        )
    if data.empty:
        raise DataInsufficient(
            "auto_cate_tuned requires non-empty data.",
            recovery_hint="Provide rows before tuning CATE learners.",
            diagnostics={"n_rows": 0},
            alternative_functions=_AUTO_CATE_TUNED_ALTERNATIVES,
        )
    for argument, value in (("y", y), ("treat", treat)):
        if not isinstance(value, str) or not value:
            raise _auto_cate_tuned_error(
                f"{argument} must be a non-empty column-name string.",
                diagnostics={"argument": argument, "value": repr(value)},
                recovery_hint=(
                    f"Pass an existing DataFrame column name for `{argument}`."
                ),
            )

    covariate_list = _normalize_covariates(covariates)
    learner_codes = _normalize_learners(learners)
    required_columns = [y, treat] + covariate_list
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise _auto_cate_tuned_error(
            "auto_cate_tuned input columns are missing from data.",
            diagnostics={"missing_columns": sorted(missing)},
            recovery_hint="Pass column names present in the DataFrame.",
        )
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or n_folds < 2:
        raise _auto_cate_tuned_error(
            "n_folds must be an integer >= 2.",
            diagnostics={"n_folds": n_folds},
            recovery_hint="Use n_folds=2 or larger for cross-fitting.",
        )
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as err:
        raise _auto_cate_tuned_error(
            "alpha must be numeric.",
            diagnostics={"alpha": repr(alpha)},
            recovery_hint="Use a confidence level such as alpha=0.05.",
        ) from err
    if not (0 < alpha_value < 1):
        raise _auto_cate_tuned_error(
            "alpha must be between 0 and 1.",
            diagnostics={"alpha": alpha},
            recovery_hint="Use a confidence level such as alpha=0.05.",
        )
    if (
        not isinstance(n_bootstrap, int)
        or isinstance(n_bootstrap, bool)
        or n_bootstrap < 1
    ):
        raise _auto_cate_tuned_error(
            "n_bootstrap must be an integer >= 1.",
            diagnostics={"n_bootstrap": n_bootstrap},
            recovery_hint="Use n_bootstrap=50 or larger for stable intervals.",
        )
    if tune in ("nuisance", "both") and (
        not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1
    ):
        raise _auto_cate_tuned_error(
            "n_trials must be an integer >= 1 for nuisance tuning.",
            diagnostics={"n_trials": n_trials, "tune": tune},
            recovery_hint="Increase n_trials or use tune='per_learner'.",
        )
    if n_trials_per_learner is None and (
        not isinstance(n_trials, int) or isinstance(n_trials, bool)
    ):
        raise _auto_cate_tuned_error(
            "n_trials must be an integer when deriving per-learner trials.",
            diagnostics={"n_trials": n_trials, "tune": tune},
            recovery_hint="Pass an integer n_trials or explicit n_trials_per_learner.",
        )

    space = search_space if search_space is not None else DEFAULT_SEARCH_SPACE
    pl_space = (
        per_learner_search_space
        if per_learner_search_space is not None
        else DEFAULT_PER_LEARNER_SEARCH_SPACE
    )
    pl_trials = (
        n_trials_per_learner
        if n_trials_per_learner is not None
        else max(5, n_trials // 3)
    )
    _validate_search_space(space, context="nuisance")
    _validate_search_space(pl_space, context="per_learner")
    if tune in ("per_learner", "both") and (
        not isinstance(pl_trials, int) or isinstance(pl_trials, bool) or pl_trials < 1
    ):
        raise _auto_cate_tuned_error(
            "n_trials_per_learner must be an integer >= 1.",
            diagnostics={"n_trials_per_learner": pl_trials, "tune": tune},
            recovery_hint="Increase n_trials_per_learner before per-learner tuning.",
        )

    # Extract arrays once (same cleaning as auto_cate)
    Y, D, X, n = _prepare_data(data, y, treat, covariate_list)
    if n < n_folds:
        raise DataInsufficient(
            "auto_cate_tuned requires at least n_folds complete rows.",
            recovery_hint="Use fewer folds or provide more complete rows.",
            diagnostics={"n_complete": n, "n_folds": n_folds},
            alternative_functions=_AUTO_CATE_TUNED_ALTERNATIVES,
        )
    unique_d = np.unique(D)
    if not (len(unique_d) == 2 and set(unique_d) == {0.0, 1.0}):
        raise _auto_cate_tuned_error(
            f"Treatment must be binary (0/1), got unique values: {unique_d}",
            diagnostics={"treat_values": unique_d.tolist()},
            recovery_hint="Encode treatment as binary 0/1 before tuning.",
        )

    optuna = _require_optuna()
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ------------------------------------------------------------------
    # Step 1: Nuisance tuning (modes 'nuisance' and 'both')
    # ------------------------------------------------------------------
    best_nuisance_params: Optional[Dict[str, Any]] = None
    best_r_loss_nuisance: Optional[float] = None
    n_completed_nuisance: int = 0
    best_outcome = None
    best_propensity = None

    if tune in ("nuisance", "both"):

        def _objective_nuisance(trial: Any) -> float:
            params = _sample_params(trial, space)
            return _r_loss_on_nuisance(
                params,
                X,
                Y,
                D,
                n_folds=n_folds,
                random_state=random_state,
            )

        nuisance_sampler = sampler or optuna.samplers.TPESampler(seed=random_state)
        nuisance_study = optuna.create_study(
            direction="minimize",
            sampler=nuisance_sampler,
        )
        nuisance_study.optimize(
            _objective_nuisance,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=verbose,
        )
        best_nuisance_params = dict(nuisance_study.best_params)
        n_completed_nuisance = len(
            [
                t
                for t in nuisance_study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
        )
        best_r_loss_nuisance = float(nuisance_study.best_value)
        best_outcome, best_propensity = _build_models_from_params(
            best_nuisance_params,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Step 2: Per-learner CATE-stage tuning (modes 'per_learner' and 'both')
    # ------------------------------------------------------------------
    per_learner_params: Dict[str, Dict[str, Any]] = {}
    per_learner_r_loss: Dict[str, float] = {}

    if tune in ("per_learner", "both"):
        # Pre-compute the nuisance once (shared across all learners' R-loss
        # evaluations) — uses the tuned nuisance if 'both', else defaults.
        om_shared = best_outcome
        pm_shared = best_propensity
        if om_shared is None:
            # Lazy import to avoid circular issues
            from .metalearners import (
                _default_outcome_model,
                _default_propensity_model,
            )

            om_shared = _default_outcome_model()
            pm_shared = _default_propensity_model()
        m_hat, e_hat = _cross_fit_nuisance(
            X,
            Y,
            D,
            outcome_model=om_shared,
            propensity_model=pm_shared,
            n_folds=n_folds,
            random_state=random_state,
        )
        e_hat = np.clip(e_hat, 0.01, 0.99)

        for code in learner_codes:

            def _objective_pl(trial: Any, _code: str = code) -> float:
                params = _sample_params(trial, pl_space)
                return _r_loss_per_learner(
                    _code,
                    params,
                    X,
                    Y,
                    D,
                    m_hat,
                    e_hat,
                    outcome_model=om_shared,
                    propensity_model=pm_shared,
                    n_folds=n_folds,
                    random_state=random_state,
                )

            pl_sampler = optuna.samplers.TPESampler(seed=random_state)
            pl_study = optuna.create_study(direction="minimize", sampler=pl_sampler)
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
        best_pl_code = min(
            per_learner_r_loss,
            key=lambda code: per_learner_r_loss[code],
        )
        best_pl_cate_model = _build_cate_model(
            per_learner_params[best_pl_code],
            random_state=random_state,
        )
    else:
        best_pl_cate_model = None

    result = auto_cate(
        data,
        y=y,
        treat=treat,
        covariates=covariate_list,
        learners=learner_codes,
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
    info["tune_mode"] = tune
    if best_nuisance_params is not None:
        info["tuned_params"] = best_nuisance_params
        info["n_trials"] = n_completed_nuisance
        info["best_r_loss_nuisance"] = best_r_loss_nuisance
    if per_learner_params:
        info["per_learner_params"] = per_learner_params
        info["per_learner_r_loss"] = per_learner_r_loss
        info["best_per_learner_code"] = best_pl_code
        info["n_trials_per_learner"] = pl_trials

    # Also expose on the AutoCATEResult's selection rule for transparency
    rule_parts: List[str] = []
    if best_nuisance_params is not None:
        rule_parts.append(f"nuisance tuned via {n_completed_nuisance} Optuna trials")
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
