"""
Augmented Synthetic Control Method (ASCM).

Combines synthetic-control weights with a ridge correction to reduce
pre-treatment imbalance when a simplex SCM alone cannot perfectly match
the treated unit.

Model
-----
    w_aug = w_scm + (x1 - X0' w_scm)' (X0'X0 + λI)^(-1) X0'
    τ̂_ascm = mean(Y1_post - Y0_post' w_aug)

where pre-treatment outcomes are centered by the control mean before
the SCM and ridge steps. This matches the no-covariate
``augsynth::augsynth(..., progfunc="Ridge", scm=TRUE)`` convention used
by the Track-A parity harness.

References
----------
Ben-Michael, E., Feller, A. and Rothstein, J. (2021).
"The Augmented Synthetic Control Method."
*Journal of the American Statistical Association*, 116(536), 1789-1803. [@benmichael2021augmented]

Ben-Michael, E., Feller, A. and Rothstein, J. (2022).
"Synthetic Controls with Staggered Adoption."
*Journal of the Royal Statistical Society: Series B*, 84(2), 351-381. [@benmichael2022synthetic]
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._augsynth_r import _augsynth_r_backend, _find_rscript  # noqa: F401
from ._core import placebo_rank_pvalue

#: augsynth ``progfunc`` values implemented natively.
_NATIVE_PROGFUNCS = frozenset({"ridge", "none"})


@accepts_aliases(_strict=True, id="unit", y="outcome")
def augsynth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Optional[List[str]] = None,
    ridge_lambda: Optional[float] = None,
    placebo: bool = True,
    alpha: float = 0.05,
    backend: str = "native",
    *,
    progfunc: str = "ridge",
    scm: bool = True,
    fixedeff: bool = False,
    **kwargs: Any,
) -> CausalResult:
    """
    Augmented Synthetic Control Method (Ben-Michael, Feller & Rothstein 2021).

    Fits SCM weights on control-mean-centered pre-treatment outcomes and
    then applies the ridge-augmented weight correction used by
    ``augsynth::augsynth``:

        w_aug = w_scm + (x1 - X0' w_scm)' (X0'X0 + λI)^(-1) X0'.

    The augmented weights are applied directly to the full donor
    trajectory and may be negative. Large ``ridge_lambda`` approaches
    centered SCM; smaller values allow more extrapolation.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    outcome, unit, time : str
        Column names.
    treated_unit : scalar or list
        Treated-unit identifier. A list of units that all adopt at
        ``treatment_time`` is fitted as one treated unit whose trajectory is
        the average of theirs, as ``augsynth::augsynth`` does when several
        units share one treatment date. Units adopting at different dates
        need ``sp.staggered_synth`` (``augsynth::multisynth``).
    treatment_time : scalar
        First treatment period (shared by every treated unit).
    covariates : list of str, optional
        Auxiliary covariates, balanced alongside the pre-period outcomes as
        in ``augsynth::augsynth``'s default (``residualize = FALSE``): each
        covariate is averaged over the pre-treatment periods per unit,
        centred at the donor mean, rescaled to the standard deviation of
        the centred pre-period outcome matrix and appended to it.
    ridge_lambda : float, optional
        Ridge penalty. When ``None``, selected by time-holdout CV
        matching ``augsynth``'s one-standard-error rule. Ignored when
        ``progfunc='none'``.
    placebo : bool, default True
        Run in-space placebo permutation tests for SE / p-value.
    alpha : float, default 0.05
        Significance level.
    backend : {'native', 'augsynth', 'r'}, default 'native'
        ``'native'`` uses StatsPAI's Python ridge-augmented SCM
        implementation. ``'augsynth'``/``'r'`` delegates the point
        estimate and pre-period RMSPE to the R ``augsynth`` package
        through ``Rscript`` using ``progfunc='Ridge'`` and ``scm=TRUE``.
        The R backend is intended for exact reference-package parity;
        the native path remains the dependency-light default.
    progfunc : {'ridge', 'none'}, default 'ridge'
        Outcome model of the augmentation, as ``augsynth``'s ``progfunc``.
        ``'ridge'`` is ridge-augmented SCM; ``'none'`` returns the SCM
        weights without augmentation. augsynth's other outcome models
        (``'EN'``, ``'RF'``, ``'GSYN'``, ``'MCP'``, ``'CITS'``,
        ``'CausalImpact'``, ``'seq2seq'``) are not implemented natively and
        raise ``MethodIncompatibility``.
    scm : bool, default True
        Start the augmentation from SCM weights (``True``) or from uniform
        weights over the donors (``False``, pure ridge regression
        weights), as augsynth's ``scm``. With ``progfunc='none'`` there is
        no augmentation and the fit is SCM whatever ``scm`` says (as in R
        augsynth).
    fixedeff : bool, default False
        De-mean every unit by its pre-treatment average before fitting and
        add the treated average back (augsynth's ``fixedeff``).
    **kwargs
        Not accepted: any leftover keyword raises ``TypeError`` (they used
        to be ignored, so a misspelled option fell back to its default).

    Returns
    -------
    CausalResult
        ``detail`` has one row per post-treatment period with columns
        ``time, treated, counterfactual, effect``. ``model_info`` includes
        ``pre_rmspe, post_rmspe, weights, ridge_lambda, n_donors,
        n_pre_periods, n_post_periods, placebo_distribution``.

    References
    ----------
    Ben-Michael, E., Feller, A. and Rothstein, J. (2021). "The Augmented
    Synthetic Control Method." *JASA*, 116(536), 1789-1803. [@benmichael2021augmented]

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_tobacco()
    >>> result = sp.augsynth(df, outcome='cigsale', unit='state', time='year',
    ...                       treated_unit='California', treatment_time=1989)
    >>> bool(result.estimate < 0)  # Prop 99 lowered cigarette sales
    True
    """
    from ..core._vcov_spec import reject_unknown_kwargs

    reject_unknown_kwargs(kwargs, function="augsynth")
    progfunc_norm = str(progfunc).lower()
    if progfunc_norm not in _NATIVE_PROGFUNCS:
        raise MethodIncompatibility(
            f"augsynth: progfunc={progfunc!r} is not implemented natively; "
            f"supported values are {sorted(_NATIVE_PROGFUNCS)}",
            recovery_hint=(
                "Use progfunc='ridge' (ridge-augmented SCM) or 'none' (SCM "
                "weights). augsynth's 'EN', 'RF', 'GSYN', 'MCP', 'CITS', "
                "'CausalImpact' and 'seq2seq' outcome models are not "
                "available; for an interactive-fixed-effects outcome model "
                "see sp.gsynth."
            ),
            diagnostics={"progfunc": progfunc},
            alternative_functions=["sp.gsynth", "sp.synth"],
        )
    if _is_list_like(treatment_time):
        times = list(pd.unique(pd.Series(list(treatment_time))))
        if len(times) != 1:
            raise MethodIncompatibility(
                "augsynth fits treated units that share one treatment date; "
                f"got {len(times)} different dates",
                recovery_hint=(
                    "For staggered adoption use sp.staggered_synth (the "
                    "partially pooled SCM of augsynth::multisynth)."
                ),
                diagnostics={"treatment_times": times},
                alternative_functions=["sp.staggered_synth"],
            )
        treatment_time = times[0]
    backend_norm = backend.lower().replace("-", "_")
    if backend_norm in {"augsynth", "r", "augsynth_r"}:
        r_spec = progfunc_norm == "ridge" and scm and not fixedeff
        if _is_list_like(treated_unit) or not r_spec:
            raise NotImplementedError(
                "The augsynth R reference backend runs the single-unit "
                "progfunc='Ridge', scm=TRUE, fixedeff=FALSE specification; "
                "use backend='native' for several treated units, "
                "progfunc='none', scm=False or fixedeff=True."
            )
        if covariates:
            raise NotImplementedError(
                "The augsynth R reference backend currently supports the "
                "outcome/treatment specification used in the parity harness; "
                "use backend='native' for covariates."
            )
        if ridge_lambda is not None:
            raise NotImplementedError(
                "The augsynth R reference backend uses augsynth::augsynth's "
                "own Ridge regularisation convention; use backend='native' "
                "for an explicit ridge_lambda."
            )
        return _augsynth_r_backend(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            alpha=alpha,
        )
    if backend_norm != "native":
        raise ValueError("Unknown backend. Use 'native', 'augsynth', or 'r'.")

    # --- Input validation (unified with classic SCM contract) ---
    check_columns(
        data,
        {
            "outcome": outcome,
            "unit": unit,
            "time": time,
            "covariates": covariates,
        },
        function="sp.augsynth",
    )
    treated_list = _as_treated_list(treated_unit)
    missing = [u for u in treated_list if u not in set(data[unit].values)]
    if missing:
        raise ValueError(f"Treated unit(s) {missing!r} not found in '{unit}' column")

    # --- Reshape to wide format ---
    panel = data.pivot_table(index=unit, columns=time, values=outcome)
    all_times = sorted(panel.columns)
    pre_times = [t for t in all_times if t < treatment_time]
    post_times = [t for t in all_times if t >= treatment_time]

    if len(pre_times) < 2:
        raise DataInsufficient(
            "Need at least 2 pre-treatment periods",
            recovery_hint=(
                "Augmented SCM fits an outcome bridge on pre-periods — "
                "needs at least 2. Use sp.did if you only have 1 pre period."
            ),
            diagnostics={"n_pre_periods": int(len(pre_times))},
            alternative_functions=["sp.did"],
        )
    if len(post_times) < 1:
        raise DataInsufficient(
            "Need at least 1 post-treatment period",
            recovery_hint=("Verify treatment_time is within the panel window."),
            diagnostics={"n_post_periods": int(len(post_times))},
            alternative_functions=[],
        )

    # augsynth::augsynth's fixedeff: de-mean every unit by its own
    # pre-treatment average and add the treated units' average back to the
    # counterfactual (``mhat``).
    unit_means = panel[pre_times].mean(axis=1)
    if fixedeff:
        panel_fit = panel.sub(unit_means, axis=0)
    else:
        panel_fit = panel

    # Several treated units sharing one adoption date are averaged into one
    # treated trajectory, as augsynth's single_augsynth does (colMeans over
    # the treated rows); every treated unit leaves the donor pool.
    Y1_pre = panel_fit.loc[treated_list, pre_times].to_numpy(np.float64).mean(axis=0)
    Y1_post = panel_fit.loc[treated_list, post_times].to_numpy(np.float64).mean(axis=0)
    level_shift = float(unit_means.loc[treated_list].mean()) if fixedeff else 0.0

    donors = [u for u in panel.index if u not in set(treated_list)]
    Y0_pre = panel_fit.loc[donors, pre_times].values.astype(np.float64)  # (J, T0)
    Y0_post = panel_fit.loc[donors, post_times].values.astype(np.float64)  # (J, T1)

    J = len(donors)
    T0 = len(pre_times)
    T1 = len(post_times)

    Z1: Optional[np.ndarray] = None
    Z0: Optional[np.ndarray] = None
    if covariates:
        Z_all = _aggregate_covariates(data, unit, time, covariates, treatment_time)
        Z1 = Z_all.loc[treated_list].to_numpy(dtype=np.float64).mean(axis=0)
        Z0 = Z_all.loc[donors].to_numpy(dtype=np.float64)

    # --- Step 1: R-augsynth-compatible augmented weights ---
    #
    # ``augsynth::augsynth(..., progfunc="Ridge", scm=TRUE)`` centers
    # pre-treatment outcomes by the control mean, solves SCM on the
    # centered pre-period matrix, then adds a ridge correction to the
    # weights themselves.  The final counterfactual is the full donor
    # trajectory multiplied by these augmented weights, which may be
    # negative.  This is deliberately different from a post-on-pre
    # outcome-model imputation.
    weight_fit = _fit_ridge_augmented_weights(
        Y1_pre,
        Y0_pre,
        ridge_lambda=ridge_lambda,
        ridge=progfunc_norm == "ridge",
        scm=scm,
        cov_target=Z1,
        cov_control=Z0,
    )
    gamma = weight_fit["weights"]
    syn_weights = weight_fit["synthetic_weights"]
    ridge_lambda = weight_fit["ridge_lambda"]
    if ridge_lambda is not None:
        ridge_lambda = float(ridge_lambda)

    # --- Step 2: Counterfactual trajectory using augmented weights ---
    Y1_hat_scm_pre = Y0_pre.T @ syn_weights
    Y1_hat_aug_pre = Y0_pre.T @ gamma
    Y1_hat_aug_post = Y0_post.T @ gamma

    pre_residual_aug = Y1_pre - Y1_hat_aug_pre
    pre_residual_scm = Y1_pre - Y1_hat_scm_pre
    pre_rmspe = float(np.sqrt(np.mean(pre_residual_aug**2)))

    effects = Y1_post - Y1_hat_aug_post
    att = float(np.mean(effects))
    if fixedeff:
        # report trajectories in levels
        Y1_pre = Y1_pre + level_shift
        Y1_post = Y1_post + level_shift
        Y1_hat_aug_pre = Y1_hat_aug_pre + level_shift
        Y1_hat_aug_post = Y1_hat_aug_post + level_shift

    # --- Inference via placebo permutation ---
    placebo_effects: Any
    if placebo:
        placebo_effects = []
        for j in range(J):
            other_idx = [i for i in range(J) if i != j]
            Y_plac_pre = Y0_pre[j]
            Y_plac_post = Y0_post[j]
            Y_others_pre = Y0_pre[other_idx]
            Y_others_post = Y0_post[other_idx]

            plac_fit = _fit_ridge_augmented_weights(
                Y_plac_pre,
                Y_others_pre,
                ridge_lambda=ridge_lambda,
                ridge=progfunc_norm == "ridge",
                scm=scm,
                cov_target=None if Z0 is None else Z0[j],
                cov_control=None if Z0 is None else Z0[other_idx],
            )
            plac_weights = plac_fit["weights"]
            plac_hat = Y_others_post.T @ plac_weights
            plac_eff = float(np.mean(Y_plac_post - plac_hat))
            placebo_effects.append(plac_eff)

        placebo_effects = np.array(placebo_effects)
        se = float(np.std(placebo_effects, ddof=1))
        pvalue = placebo_rank_pvalue(abs(att), np.abs(placebo_effects))

        t_crit = sp_stats.norm.ppf(1 - alpha / 2)
        ci = (att - t_crit * se, att + t_crit * se)
    else:
        # ASCM inference is the placebo permutation; without it there is no
        # standard error to report. Say so rather than return bare NaN.
        warnings.warn(
            "augsynth(placebo=False): no placebo permutation was run, so the "
            "standard error, p-value and interval are NaN. Pass placebo=True "
            "for inference.",
            UserWarning,
            stacklevel=2,
        )
        placebo_effects = np.array([])
        se = float("nan")
        pvalue = float("nan")
        ci = (float("nan"), float("nan"))

    # Build period-level results
    effects_df = pd.DataFrame(
        {
            "time": post_times,
            "treated": Y1_post,
            "counterfactual": Y1_hat_aug_post,
            "effect": effects,
        }
    )

    # Unified gap table (full trajectory, matches classic SCM contract)
    all_times_arr = np.array(pre_times + post_times)
    Y1_all = np.concatenate([Y1_pre, Y1_post])
    Y1_hat_all = np.concatenate([Y1_hat_aug_pre, Y1_hat_aug_post])
    gap_all = Y1_all - Y1_hat_all
    gap_df = pd.DataFrame(
        {
            "time": all_times_arr,
            "treated": Y1_all,
            "synthetic": Y1_hat_all,
            "gap": gap_all,
            "post_treatment": np.concatenate(
                [
                    np.zeros(T0, dtype=bool),
                    np.ones(T1, dtype=bool),
                ]
            ),
        }
    )

    weight_df = (
        pd.DataFrame({"unit": donors, "weight": gamma})
        .assign(abs_weight=lambda df: np.abs(df["weight"]))
        .sort_values("abs_weight", ascending=False)
        .drop(columns=["abs_weight"])
        .reset_index(drop=True)
    )
    weight_df = weight_df[np.abs(weight_df["weight"]) > 1e-6]

    return CausalResult(
        method="Augmented Synthetic Control (ASCM)",
        _citation_key="augsynth",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=effects_df,
        model_info={
            "model_type": "Synthetic Control (Augmented)",
            "pre_rmspe": pre_rmspe,
            "pre_treatment_rmse": pre_rmspe,
            "pre_treatment_mspe": pre_rmspe**2,
            "scm_pre_treatment_rmse": float(np.sqrt(np.mean(pre_residual_scm**2))),
            "post_rmspe": float(np.sqrt(np.mean(effects**2))),
            "weights": weight_df,
            "weights_dict": dict(zip(donors, np.asarray(gamma))),
            "synthetic_weights": dict(zip(donors, np.asarray(syn_weights))),
            "augmented_weights_can_be_negative": True,
            "n_donors": J,
            "n_pre_periods": T0,
            "n_post_periods": T1,
            "treatment_time": treatment_time,
            "treated_unit": treated_unit,
            "treated_units": treated_list,
            "n_treated_units": len(treated_list),
            "progfunc": progfunc_norm,
            "scm": bool(scm),
            "fixedeff": bool(fixedeff),
            "covariates": list(covariates) if covariates else [],
            "ridge_lambda": ridge_lambda,
            "effects_by_period": effects_df,
            "gap_table": gap_df,
            "times": all_times_arr,
            "Y_synth": Y1_hat_all,
            "Y_treated": Y1_all,
            "placebo_distribution": placebo_effects,
            "n_placebos": len(placebo_effects),
        },
    )


# ====================================================================== #
#  Internal helpers
# ====================================================================== #


def _scm_weights(Y1_pre: np.ndarray, Y0_pre: np.ndarray) -> np.ndarray:
    """
    Standard SCM: find non-negative weights that minimise
    ||Y1_pre - Y0_pre' γ||² subject to sum(γ) = 1, γ >= 0.
    """
    from ._core import solve_simplex_weights

    return solve_simplex_weights(Y1_pre, Y0_pre.T)


def _solve_ridge_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a ridge normal equation with a pseudo-inverse fallback."""
    try:
        return np.asarray(np.linalg.solve(a, b))
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.pinv(a) @ b)


def _is_list_like(x: Any) -> bool:
    return isinstance(x, (list, tuple, set, frozenset, np.ndarray, pd.Index, pd.Series))


def _as_treated_list(treated_unit: Any) -> List[Any]:
    """Treated unit(s) as a de-duplicated list (input order kept)."""
    if not _is_list_like(treated_unit):
        return [treated_unit]
    out = list(dict.fromkeys(list(treated_unit)))
    if not out:
        raise DataInsufficient(
            "augsynth: treated_unit is an empty list",
            recovery_hint="Pass at least one treated unit identifier.",
        )
    return out


def _aggregate_covariates(
    data: pd.DataFrame,
    unit: str,
    time: str,
    covariates: List[str],
    treatment_time: Any,
) -> pd.DataFrame:
    """Per-unit pre-treatment means of the covariates (augsynth ``cov_agg``)."""
    pre = data.loc[data[time] < treatment_time, [unit] + list(covariates)]
    Z = pre.groupby(unit)[list(covariates)].mean()
    Z = Z.reindex(sorted(data[unit].unique()))
    if Z.isna().any().any():
        bad = Z.index[Z.isna().any(axis=1)].tolist()
        raise DataInsufficient(
            f"augsynth: units {bad[:5]!r} have no pre-treatment covariate data",
            recovery_hint="Fill or drop missing covariates before fitting.",
        )
    return Z


def _fit_ridge_augmented_weights(
    target_pre: np.ndarray,
    control_pre: np.ndarray,
    *,
    ridge_lambda: Optional[float] = None,
    ridge: bool = True,
    scm: bool = True,
    cov_target: Optional[np.ndarray] = None,
    cov_control: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """
    Fit R ``augsynth``-style ridge-augmented SCM weights.

    The installed R package centers pre-treatment outcomes by the
    control-unit mean, obtains simplex SCM weights on the centered
    matrix, then applies a ridge correction to the weights:

        w_aug = w_scm + (x1 - X0' w_scm)' (X0'X0 + lambda I)^(-1) X0'.

    The resulting weights need not be non-negative.  They are then used
    directly on the raw control trajectory for both pre- and post-period
    counterfactuals.

    ``ridge=False`` returns the starting weights (augsynth
    ``progfunc='None'``); ``scm=False`` starts from uniform weights. With
    covariates (one row per donor in ``cov_control``) each covariate is
    centred at its donor mean, rescaled to ``sd(X_c)`` (the standard
    deviation over every entry of the centred donor outcome matrix) divided
    by its own donor standard deviation, and appended to the outcome
    columns -- ``fit_ridgeaug_formatted`` with ``residualize = FALSE``.
    """
    target_pre = np.asarray(target_pre, dtype=np.float64).ravel()
    control_pre = np.asarray(control_pre, dtype=np.float64)
    control_mean = control_pre.mean(axis=0)
    x_c = control_pre - control_mean
    x_1 = target_pre - control_mean
    if cov_control is not None:
        Zc = np.asarray(cov_control, dtype=np.float64).reshape(x_c.shape[0], -1)
        z1 = np.asarray(cov_target, dtype=np.float64).ravel()
        z_mean = Zc.mean(axis=0)
        sdz = Zc.std(axis=0, ddof=1)
        if np.any(~(sdz > 0)):
            raise DataInsufficient(
                "augsynth: a covariate has no variation across donors",
                recovery_hint="Drop constant covariates.",
            )
        sdx = float(np.std(x_c, ddof=1))
        x_c = np.hstack([x_c, sdx * (Zc - z_mean) / sdz])
        x_1 = np.concatenate([x_1, sdx * (z1 - z_mean) / sdz])

    n0 = x_c.shape[0]
    # Without the ridge augmentation the fit is SCM itself, whatever scm=
    # says: R augsynth(progfunc = "None", scm = FALSE) returns the SCM
    # estimate (3.334 on the probe panel), and the uniform-weight average
    # the native path gave (6.878) is not an augsynth estimator.
    use_scm = scm or not ridge
    syn = _scm_weights(x_1, x_c) if use_scm else np.full(n0, 1.0 / n0)
    if not ridge:
        return {"weights": syn.copy(), "synthetic_weights": syn, "ridge_lambda": None}
    if ridge_lambda is None:
        ridge_lambda = _cv_ridge_lambda_augsynth(x_c, x_1, scm=scm)
    ridge_lambda = float(ridge_lambda)

    gram = x_c.T @ x_c + ridge_lambda * np.eye(x_c.shape[1])
    imbalance = x_1 - x_c.T @ syn
    ridge_weights = imbalance @ _solve_ridge_system(gram, x_c.T)
    weights = syn + ridge_weights
    return {
        "weights": np.asarray(weights, dtype=np.float64),
        "synthetic_weights": np.asarray(syn, dtype=np.float64),
        "ridge_lambda": ridge_lambda,
    }


def _cv_ridge_lambda_augsynth(
    x_c: np.ndarray,
    x_1: np.ndarray,
    *,
    lambda_min_ratio: float = 1e-8,
    n_lambda: int = 20,
    holdout_length: int = 1,
    min_1se: bool = True,
    scm: bool = True,
) -> float:
    """Time-holdout lambda CV matching ``augsynth``'s ridge path."""
    _, singular_values, _ = np.linalg.svd(x_c, full_matrices=False)
    if singular_values.size == 0:
        return 1.0
    lambda_max = float(singular_values[0] ** 2)
    if not np.isfinite(lambda_max) or lambda_max <= 0:
        return 1.0
    scaler = float(lambda_min_ratio ** (1.0 / n_lambda))
    lambdas = lambda_max * scaler ** np.arange(n_lambda + 1)

    n_periods = x_c.shape[1]
    n_holdouts = n_periods - holdout_length
    if n_holdouts <= 1:
        return float(lambdas[0])

    errors = np.zeros((n_holdouts, len(lambdas)), dtype=np.float64)
    for i in range(n_holdouts):
        holdout = np.arange(i, i + holdout_length)
        train_mask = np.ones(n_periods, dtype=bool)
        train_mask[holdout] = False
        x_0_train = x_c[:, train_mask]
        x_1_train = x_1[train_mask]
        x_0_val = x_c[:, holdout]
        x_1_val = x_1[holdout]

        if scm:
            syn = _scm_weights(x_1_train, x_0_train)
        else:
            syn = np.full(x_0_train.shape[0], 1.0 / x_0_train.shape[0])
        imbalance = x_1_train - x_0_train.T @ syn
        base_gram = x_0_train.T @ x_0_train
        for j, lam in enumerate(lambdas):
            gram = base_gram + float(lam) * np.eye(x_0_train.shape[1])
            ridge_weights = imbalance @ _solve_ridge_system(gram, x_0_train.T)
            aug_weights = syn + ridge_weights
            val_error = x_1_val - x_0_val.T @ aug_weights
            errors[i, j] = float(val_error @ val_error)

    lambda_errors = errors.mean(axis=0)
    lambda_errors_se = errors.std(axis=0, ddof=1) / np.sqrt(errors.shape[0])
    min_idx = int(np.argmin(lambda_errors))
    if not min_1se:
        return float(lambdas[min_idx])
    threshold = lambda_errors[min_idx] + lambda_errors_se[min_idx]
    candidates = lambdas[lambda_errors <= threshold]
    if candidates.size == 0:
        return float(lambdas[min_idx])
    return float(np.max(candidates))
