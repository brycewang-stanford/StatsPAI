"""Tuning the causal forest by out-of-bag R-loss.

The forest's defaults (leaves of at least five, half-samples, honest
splitting) are sensible but not always best: a signal that varies fast wants
smaller leaves, a noisy one larger. There is no ground truth for a treatment
effect to cross-validate against, but there is a loss that the true effect
function minimises, the R-loss of Nie and Wager,

    mean( (Y - m(X) - (W - e(X)) * tau(X))^2 ),

with ``m`` and ``e`` the conditional means of the outcome and the treatment.
Evaluated at each observation's *out-of-bag* prediction it is an honest
measure of fit, and candidate settings can be ranked on it. This is the
criterion behind ``grf::causal_forest(tune.parameters = ...)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..exceptions import MethodIncompatibility
from .causal_forest import causal_forest

__all__ = ["tune_causal_forest"]

_TUNABLE = (
    "min_samples_leaf",
    "max_samples",
    "mtry",
    "honesty_fraction",
    "honesty_prune_leaves",
    "imbalance_penalty",
)


def _draw(
    rng: np.random.Generator, n: int, p: int, names: Sequence[str]
) -> Dict[str, Any]:
    """One random setting, from ranges that scale with the sample."""
    out: Dict[str, Any] = {}
    if "min_samples_leaf" in names:
        # log-uniform between 1 and n / 16
        top = max(np.log2(max(n, 32)) - 4.0, 1.0)
        out["min_samples_leaf"] = int(max(1, np.floor(2.0 ** (rng.random() * top))))
    if "max_samples" in names:
        out["max_samples"] = float(rng.uniform(0.05, 0.5))
    if "mtry" in names:
        out["mtry"] = int(rng.integers(1, p + 1))
    if "honesty_fraction" in names:
        out["honesty_fraction"] = float(rng.uniform(0.5, 0.8))
    if "honesty_prune_leaves" in names:
        out["honesty_prune_leaves"] = bool(rng.random() < 0.5)
    if "imbalance_penalty" in names:
        out["imbalance_penalty"] = float(-np.log(rng.uniform(np.exp(-2.0), 1.0)))
    return out


@accepts_aliases(covariates="X")
def tune_causal_forest(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    Y: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    *,
    parameters: Sequence[str] = ("min_samples_leaf", "max_samples", "mtry"),
    n_draws: int = 40,
    tune_trees: int = 200,
    tune_reps: int = 2,
    n_estimators: int = 2000,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Choose causal-forest settings by out-of-bag R-loss, then fit.

    Parameters
    ----------
    formula, data, Y, T, X
        As in :func:`statspai.causal_forest`. ``covariates=`` is accepted
        for ``X``.
    parameters : sequence of str
        Which settings to tune, from ``'min_samples_leaf'``,
        ``'max_samples'``, ``'mtry'``, ``'honesty_fraction'``,
        ``'honesty_prune_leaves'`` and ``'imbalance_penalty'``. The default
        tunes the three that matter most.
    n_draws : int, default 40
        Random settings tried, besides the defaults.
    tune_trees : int, default 200
        Trees in each trial forest. Small forests are noisy, which is why
        a tuned setting is adopted only if it beats the defaults clearly
        (see Notes).
    tune_reps : int, default 2
        Trial forests per setting, with different seeds; their losses are
        averaged and their spread gives the noise of the comparison.
    n_estimators : int, default 2000
        Trees in the final forest.
    random_state : int, optional
    **kwargs
        Passed to every forest (for example ``clusters=``, ``model_y=``).

    Returns
    -------
    dict
        ``forest`` (the final :class:`~statspai.forest.causal_forest.CausalForest`),
        ``best_params``, ``tuned`` (whether a tuned setting replaced the
        defaults), ``default_error`` and ``tuned_error`` (out-of-bag
        R-loss), ``noise`` (standard deviation of the loss across
        repetitions of the same setting) and ``trials`` (a DataFrame, one
        row per setting, sorted by loss).

    Notes
    -----
    The nuisance functions ``m`` and ``e`` are estimated once, out of fold,
    by the forest with the default settings, and shared by every trial, so
    settings are compared on the same residuals. When no trial is adopted
    that default forest is what is returned. A tuned setting is adopted
    only when its loss is below the defaults' by more than twice the
    repetition noise; otherwise the defaults are kept and ``tuned`` is
    False. Picking the minimum of forty
    noisy losses without such a margin would select noise.

    What to expect, from 12 replications each of four designs with 1,500
    rows and 30 draws (root mean squared error of the out-of-bag effects
    against the truth, tuned against default): a constant effect, 0.082
    against 0.119, with a tuned setting adopted 10 times out of 12, always
    with much larger leaves; a step and a fast-varying effect, no
    difference, the defaults being kept almost every time; a linear
    effect, 0.184 against 0.172, slightly worse. The R-loss is a noisy
    criterion and a lower loss is not always a lower error of the effect
    itself. Tuning pays when the effect varies little; it is not a free
    improvement, and ``trials`` is there to be looked at.

    ``grf`` fits a smooth surface to the trial losses and minimises it.
    Here the best trial itself is taken, with the margin above; the two
    choose from the same kind of evidence but will not return the same
    setting.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 800
    >>> X = rng.normal(size=(n, 4))
    >>> W = rng.binomial(1, 0.5, n)
    >>> Y = X[:, 0] + W * (1 + 2 * (X[:, 1] > 0)) + rng.normal(size=n)
    >>> out = sp.tune_causal_forest(Y=Y, T=W, X=X, n_draws=10,
    ...                             n_estimators=200, random_state=1)
    >>> sorted(out)[:3]
    ['best_params', 'default_error', 'forest']
    >>> bool(out["tuned_error"] <= out["default_error"])
    True

    References
    ----------
    [@nie2021quasi] [@athey2019generalized]
    """
    names = [str(x) for x in parameters]
    unknown = [x for x in names if x not in _TUNABLE]
    if unknown or not names:
        raise MethodIncompatibility(
            f"tune_causal_forest: cannot tune {unknown or 'nothing'}.",
            recovery_hint=f"Choose from {list(_TUNABLE)}.",
        )
    clash = [x for x in names if x in kwargs]
    if clash:
        raise MethodIncompatibility(
            f"tune_causal_forest: {clash} are both tuned and fixed by keyword.",
            recovery_hint="Drop them from parameters= or from the keywords.",
        )
    if n_draws < 1 or tune_trees < 10 or tune_reps < 1:
        raise MethodIncompatibility(
            "tune_causal_forest: n_draws >= 1, tune_trees >= 10 and "
            "tune_reps >= 1 are needed."
        )
    rng = np.random.default_rng(random_state)
    seed0 = int(rng.integers(0, 2**31 - 1))

    # The forest with the default settings, at full size. Its nuisance
    # estimates are the ones every trial is scored on, and it is what is
    # returned when no trial beats it.
    base = causal_forest(
        formula=formula, data=data, Y=Y, T=T, X=X,
        n_estimators=int(n_estimators), random_state=seed0, **kwargs,
    )  # fmt: skip
    nuis = base.get_nuisances()
    y_res = np.asarray(base._Y_original, dtype=float) - np.asarray(nuis["Y_hat"])
    w_res = np.asarray(base._T_original, dtype=float) - np.asarray(nuis["W_hat"])
    n, p = np.asarray(base._X_original).shape
    shared = dict(kwargs)
    shared.update({"Y_hat": nuis["Y_hat"], "W_hat": nuis["W_hat"]})

    def losses(params: Dict[str, Any]) -> List[float]:
        out = []
        for rep in range(int(tune_reps)):
            forest = causal_forest(
                formula=formula, data=data, Y=Y, T=T, X=X,
                n_estimators=int(tune_trees),
                random_state=seed0 + 7919 * (rep + 1),
                **shared, **params,
            )  # fmt: skip
            tau = np.asarray(forest.oob_effect(), dtype=float)
            ok = np.isfinite(tau)
            out.append(float(np.mean((y_res[ok] - w_res[ok] * tau[ok]) ** 2)))
        return out

    rows: List[Dict[str, Any]] = []
    default_losses = losses({})
    rows.append({"setting": "default", "error": float(np.mean(default_losses)),
                 "spread": float(np.std(default_losses))})  # fmt: skip
    for _ in range(int(n_draws)):
        params = _draw(rng, n, p, names)
        ls = losses(params)
        rows.append({"setting": "draw", **params, "error": float(np.mean(ls)),
                     "spread": float(np.std(ls))})  # fmt: skip
    trials = pd.DataFrame(rows).sort_values("error").reset_index(drop=True)
    default_error = float(np.mean(default_losses))
    # noise of one setting's mean loss, pooled over the settings
    noise = float(np.sqrt(np.mean(np.square(trials["spread"]))) / np.sqrt(tune_reps))
    best = trials.iloc[0]
    tuned = bool(
        best["setting"] == "draw" and default_error - best["error"] > 2.0 * noise
    )
    best_params: Dict[str, Any] = {}
    if tuned:
        for name in names:
            value = best[name]
            if name in ("min_samples_leaf", "mtry"):
                best_params[name] = int(value)
            elif name == "honesty_prune_leaves":
                best_params[name] = bool(value)
            else:
                best_params[name] = float(value)
    final = (
        causal_forest(
            formula=formula, data=data, Y=Y, T=T, X=X,
            n_estimators=int(n_estimators),
            random_state=seed0, **shared, **best_params,
        )  # fmt: skip
        if tuned
        else base
    )
    return {
        "forest": final,
        "best_params": best_params,
        "tuned": tuned,
        "default_error": default_error,
        "tuned_error": float(best["error"]) if tuned else default_error,
        "noise": noise,
        "trials": trials,
    }
