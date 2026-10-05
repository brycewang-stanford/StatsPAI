"""Propensity-score weights and what they do to a sample.

Three small tools that sit between a fitted propensity score and an
outcome model:

- :func:`ps_weights` turns scores into weights for a chosen target
  population (ATE, ATT, ATC, overlap, matching weights), with optional
  stabilisation and truncation, and handles a continuous exposure through
  the ratio of normal densities.
- :func:`ess` is the effective sample size of a set of weights.
- :func:`energy_distance` is a single number for how far apart two
  weighted covariate distributions are, jointly rather than one margin at
  a time.

References
----------
[@li2018balancing] [@li2013weighting] [@robins2000marginal]
[@szekely2013energy] [@huling2024energy] [@chattopadhyay2023implied]
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility

_ESTIMANDS = ("ATE", "ATT", "ATC", "ATO", "ATM")


def _as_1d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise MethodIncompatibility(f"{name} is empty")
    return arr


def ps_weights(
    ps: Any,
    treat: Any,
    estimand: str = "ATE",
    *,
    stabilize: bool = False,
    truncate: Optional[Tuple[float, float]] = None,
    truncate_scale: str = "ps",
    exposure: str = "binary",
    sigma: Optional[float] = None,
) -> Union[np.ndarray, pd.Series]:
    """Propensity-score weights for a target population.

    Parameters
    ----------
    ps : array-like
        Binary exposure: the fitted probability of treatment,
        ``P(treat = 1 | X)``. Continuous exposure: the fitted conditional
        mean of the exposure, ``E[treat | X]``.
    treat : array-like
        The exposure: 0/1 for ``exposure='binary'``, any real number for
        ``exposure='continuous'``.
    estimand : {'ATE', 'ATT', 'ATC', 'ATO', 'ATM'}, default 'ATE'
        Target population (binary exposure). With tilting function
        ``h(e)`` the treated get ``h / e`` and the untreated
        ``h / (1 - e)``:

        ==========  =================  ==================================
        estimand    ``h(e)``           population
        ==========  =================  ==================================
        ``'ATE'``   ``1``              everyone
        ``'ATT'``   ``e``              the treated
        ``'ATC'``   ``1 - e``          the untreated (``'ATU'`` accepted)
        ``'ATO'``   ``e (1 - e)``      overlap (clinical equipoise)
        ``'ATM'``   ``min(e, 1 - e)``  the evenly matchable
        ==========  =================  ==================================

        A continuous exposure only has ``'ATE'``.
    stabilize : bool, default False
        Multiply by the marginal probability (binary) or marginal density
        (continuous) of the exposure received. Binary: only for
        ``'ATE'``, where the stabilised weights average one; the weighted
        contrast of group means is unchanged because the factor is
        constant within a group. Continuous: unstabilised density weights
        are rarely usable, so pass ``stabilize=True`` unless there is a
        reason not to.
    truncate : (lower, upper), optional
        Winsorise the propensity score before weighting. Scores below
        ``lower`` are set to ``lower``, above ``upper`` to ``upper``; no
        row is dropped. Binary exposure only.
    truncate_scale : {'ps', 'quantile'}, default 'ps'
        ``'ps'``: the bounds are propensity scores. ``'quantile'``: they
        are quantiles of the fitted scores, e.g. ``(0.01, 0.99)``.
    exposure : {'binary', 'continuous'}, default 'binary'
        Type of the exposure.
    sigma : float, optional
        Continuous exposure: residual standard deviation of the exposure
        model whose fitted values are passed as ``ps``. Required.

    Returns
    -------
    numpy.ndarray or pandas.Series
        One weight per row; a Series with the index of ``ps`` or
        ``treat`` when either is a Series.

    Notes
    -----
    For a continuous exposure the weight is ``1 / f(x | X)`` with ``f``
    the normal density at mean ``ps`` and standard deviation ``sigma``;
    stabilised, the numerator is the normal density at the sample mean
    and standard deviation of the exposure.

    Trimming, as opposed to truncation, removes rows and so changes the
    sample; use :func:`sp.trimming` for that and refit the propensity
    model on what is left.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> ps = np.array([0.2, 0.2, 0.5, 0.8])
    >>> treat = np.array([1, 0, 1, 0])
    >>> sp.ps_weights(ps, treat).round(2).tolist()
    [5.0, 1.25, 2.0, 5.0]
    >>> sp.ps_weights(ps, treat, 'ATT').round(2).tolist()
    [1.0, 0.25, 1.0, 4.0]
    >>> sp.ps_weights(ps, treat, 'ATO').round(2).tolist()
    [0.8, 0.2, 0.5, 0.8]
    >>> sp.ps_weights(ps, treat, 'ATM').round(2).tolist()
    [1.0, 0.25, 1.0, 1.0]

    References
    ----------
    [@li2018balancing] [@li2013weighting] [@robins2000marginal]
    """
    index = None
    for obj in (ps, treat):
        if isinstance(obj, pd.Series):
            index = obj.index
            break
    e = _as_1d(ps, "ps")
    t = _as_1d(treat, "treat")
    if e.shape != t.shape:
        raise MethodIncompatibility(
            f"ps and treat must have the same length; got {e.size} and {t.size}"
        )
    if not (np.isfinite(e).all() and np.isfinite(t).all()):
        raise MethodIncompatibility("ps and treat must not contain missing values")
    estimand = str(estimand).upper()
    if estimand == "ATU":
        estimand = "ATC"
    if estimand not in _ESTIMANDS:
        raise MethodIncompatibility(
            "estimand must be 'ATE', 'ATT', 'ATC' (or 'ATU'), 'ATO' or 'ATM'; "
            f"got {estimand!r}"
        )

    if exposure == "continuous":
        w = _continuous_weights(e, t, estimand, stabilize, truncate, sigma)
    elif exposure == "binary":
        w = _binary_weights(e, t, estimand, stabilize, truncate, truncate_scale)
    else:
        raise MethodIncompatibility(
            f"exposure must be 'binary' or 'continuous'; got {exposure!r}"
        )
    if index is not None:
        return pd.Series(w, index=index, name="ps_weight")
    return w


def _binary_weights(
    e: np.ndarray,
    t: np.ndarray,
    estimand: str,
    stabilize: bool,
    truncate: Optional[Tuple[float, float]],
    truncate_scale: str,
) -> np.ndarray:
    if not set(np.unique(t)).issubset({0.0, 1.0}):
        raise MethodIncompatibility(
            "treat must be 0/1 for exposure='binary'; for a continuous "
            "exposure pass exposure='continuous' and sigma="
        )
    if ((e <= 0) | (e >= 1)).any():
        raise MethodIncompatibility(
            "propensity scores must lie strictly between 0 and 1; a score of "
            "0 or 1 is a positivity violation and has no finite weight"
        )
    if truncate is not None:
        lower, upper = (float(b) for b in truncate)
        if truncate_scale == "quantile":
            if not 0 <= lower < upper <= 1:
                raise MethodIncompatibility(
                    "quantile bounds need 0 <= lower < upper <= 1"
                )
            lower, upper = np.quantile(e, [lower, upper])
        elif truncate_scale != "ps":
            raise MethodIncompatibility(
                f"truncate_scale must be 'ps' or 'quantile'; got {truncate_scale!r}"
            )
        if not lower < upper:
            raise MethodIncompatibility("truncate needs lower < upper")
        e = np.clip(e, lower, upper)
    if stabilize and estimand != "ATE":
        raise MethodIncompatibility(
            "stabilize=True is defined for ATE weights only: the other "
            "estimands already have bounded or unit weights in one group.",
            recovery_hint="Drop stabilize=True or use estimand='ATE'.",
        )
    if estimand == "ATE":
        tilt = np.ones_like(e)
    elif estimand == "ATT":
        tilt = e
    elif estimand == "ATC":
        tilt = 1 - e
    elif estimand == "ATO":
        tilt = e * (1 - e)
    else:
        tilt = np.minimum(e, 1 - e)
    w = tilt / np.where(t == 1, e, 1 - e)
    if stabilize:
        p = float(t.mean())
        w = w * np.where(t == 1, p, 1 - p)
    return np.asarray(w, dtype=float)


def _continuous_weights(
    mu: np.ndarray,
    x: np.ndarray,
    estimand: str,
    stabilize: bool,
    truncate: Optional[Tuple[float, float]],
    sigma: Optional[float],
) -> np.ndarray:
    if estimand != "ATE":
        raise MethodIncompatibility(
            "a continuous exposure has no treated or untreated group; only "
            "estimand='ATE' is defined.",
            recovery_hint="Use estimand='ATE'.",
        )
    if truncate is not None:
        raise MethodIncompatibility(
            "truncate= bounds a probability and does not apply to the "
            "fitted mean of a continuous exposure.",
            recovery_hint="Winsorise the returned weights instead.",
        )
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        raise MethodIncompatibility(
            "exposure='continuous' needs sigma=, the residual standard "
            "deviation of the exposure model"
        )
    w = 1.0 / stats.norm.pdf(x, loc=mu, scale=float(sigma))
    if stabilize:
        w = w * stats.norm.pdf(x, loc=x.mean(), scale=x.std(ddof=1))
    return np.asarray(w, dtype=float)


def ess(
    weights: Any,
    by: Any = None,
) -> Union[float, pd.Series]:
    """Effective sample size of a set of weights.

    ``ESS = (sum w)^2 / sum(w^2)``: the number of equally weighted
    observations that would give a weighted mean of the same variance. It
    equals ``n`` when all weights are equal and falls as a few rows come
    to dominate.

    Parameters
    ----------
    weights : array-like
        Weights. Negative entries are allowed, since the weights a
        regression implies can be negative (:func:`sp.implied_weights`);
        the formula is applied as it stands.
    by : array-like, optional
        Group labels (typically the exposure). The effective sample size
        is then returned for each group, which is what matters for a
        contrast: the precision of each group mean depends on its own ESS.

    Returns
    -------
    float or pandas.Series
        A float, or with ``by`` a Series indexed by group.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> sp.ess(np.ones(50))
    50.0
    >>> round(sp.ess([1, 1, 1, 9]), 2)
    1.71
    >>> sp.ess([1, 1, 1, 9], by=[0, 0, 1, 1]).round(2).tolist()
    [2.0, 1.22]

    """
    w = _as_1d(weights, "weights")
    if not np.isfinite(w).all():
        raise MethodIncompatibility("weights must not contain missing values")
    if by is None:
        return _ess(w)
    g = np.asarray(by).ravel()
    if g.shape != w.shape:
        raise MethodIncompatibility(
            f"by and weights must have the same length; got {g.size} and {w.size}"
        )
    out = {level: _ess(w[g == level]) for level in pd.unique(g)}
    return pd.Series(out, name="ess").sort_index()


def _ess(w: np.ndarray) -> float:
    denom = float(np.sum(w**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w) ** 2 / denom)


def energy_distance(
    data: pd.DataFrame,
    treat: str,
    covariates: Sequence[str],
    weights: Any = None,
) -> float:
    """Energy distance between the covariate distributions of two groups.

    ``2 E|X1 - X0| - E|X1 - X1'| - E|X0 - X0'|`` with Euclidean distance
    on standardised covariates and expectations taken over the (weighted)
    samples. It is zero only when the two weighted joint distributions
    coincide, so unlike one standardized difference per covariate it
    picks up imbalance in interactions and higher moments.

    Parameters
    ----------
    data : DataFrame
    treat : str
        Binary 0/1 column.
    covariates : sequence of str
        Numeric columns. Expand categorical variables into indicators
        first.
    weights : str or array-like, optional
        Weights; omitted, the unweighted distance.

    Returns
    -------
    float

    Notes
    -----
    Each covariate is centred and divided by its unweighted standard
    deviation (``sqrt(p (1 - p))`` for a 0/1 column), so the scaling is
    the same before and after weighting and two sets of weights can be
    compared on one scale. This is the between-group statistic that R
    ``halfmoon::bal_energy`` reports.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=400)
    >>> d = rng.binomial(1, 1 / (1 + np.exp(-x)))
    >>> df = pd.DataFrame({'d': d, 'x': x})
    >>> raw = sp.energy_distance(df, 'd', ['x'])
    >>> ps = sp.propensity_score(df, 'd', ['x'])
    >>> w = sp.ps_weights(ps, df['d'])
    >>> bool(sp.energy_distance(df, 'd', ['x'], weights=w) < raw)
    True

    References
    ----------
    [@szekely2013energy] [@huling2024energy]
    """
    from scipy.spatial.distance import cdist

    cols: List[str] = list(covariates)
    if not cols:
        raise MethodIncompatibility("covariates is empty")
    frame = data[[treat] + cols]
    if frame.isna().any().any():
        raise MethodIncompatibility(
            "energy_distance: missing values in treat or covariates; drop "
            "or impute them first"
        )
    t = frame[treat].to_numpy(dtype=float)
    if not set(np.unique(t)) == {0.0, 1.0}:
        raise MethodIncompatibility(
            f"'{treat}' must be binary 0/1 with both groups present"
        )
    if weights is None:
        w = np.ones(len(t))
    elif isinstance(weights, str):
        w = data[weights].to_numpy(dtype=float)
    else:
        w = _as_1d(weights, "weights")
    if w.shape != t.shape:
        raise MethodIncompatibility("weights must have one entry per row of data")
    if not np.isfinite(w).all() or (w < 0).any():
        raise MethodIncompatibility("weights must be finite and non-negative")

    X = frame[cols].to_numpy(dtype=float)
    scale = X.std(axis=0, ddof=1)
    for j in range(X.shape[1]):
        if set(np.unique(X[:, j])).issubset({0.0, 1.0}):
            p = X[:, j].mean()
            scale[j] = np.sqrt(p * (1 - p))
    scale[scale == 0] = 1.0
    Z = (X - X.mean(axis=0)) / scale
    s1, s0 = w[t == 1].sum(), w[t == 0].sum()
    if s1 <= 0 or s0 <= 0:
        raise MethodIncompatibility("each group needs positive total weight")
    a = np.where(t == 1, w / s1, -w / s0)
    return float(-(a @ cdist(Z, Z) @ a))


def implied_weights(
    data: pd.DataFrame,
    treat: str,
    covariates: Sequence[str],
    *,
    interactions: bool = False,
    estimand: str = "ATE",
) -> pd.Series:
    """Weights that a linear regression adjustment implicitly puts on each row.

    The coefficient on a binary treatment in ``y ~ treat + covariates`` is
    a difference of two weighted means of ``y``, with weights that depend
    only on the treatment and the covariates. Looking at those weights,
    without the outcome, shows what the regression is doing: which
    population it represents, how many observations effectively carry the
    estimate, and whether some rows enter with a negative weight, that
    is, are extrapolated.

    Parameters
    ----------
    data : DataFrame
    treat : str
        Binary 0/1 column.
    covariates : sequence of str
        Numeric regressors other than the treatment (expand categorical
        variables into indicators first).
    interactions : bool, default False
        ``False``: the regression ``y ~ treat + covariates`` with one set
        of covariate slopes ("uniform regression imputation"). ``True``:
        the treatment interacted with every centred covariate, i.e. a
        separate regression in each arm ("multi regression imputation").
    estimand : {'ATE', 'ATT', 'ATC'}, default 'ATE'
        With ``interactions=True``, the population the two arm
        regressions are averaged over. The regression without
        interactions has no choice here: its weights target a population
        of its own, between the two groups, and ``estimand`` must stay
        ``'ATE'``.

    Returns
    -------
    pandas.Series
        One weight per row, scaled to sum to the group size within each
        treatment group (so that 1 means "counts as one observation").
        The regression coefficient equals the weighted mean of ``y``
        among the treated minus the weighted mean among the controls.

    Notes
    -----
    Without interactions the weights are ``n_g * r_i / sum(r^2)`` in
    absolute value, with ``r`` the residual of the treatment regressed on
    the covariates. With interactions the weights of arm ``g`` are
    ``n_g * xbar' (X_g' X_g)^{-1} x_i`` with ``xbar`` the covariate mean
    of the target population. Both match R ``lmw::lmw``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=300)
    >>> d = rng.binomial(1, 1 / (1 + np.exp(-x)))
    >>> y = 1 + 2 * d + x + rng.normal(size=300)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x': x})
    >>> w = sp.implied_weights(df, 'd', ['x'])
    >>> ols = sp.regress('y ~ d + x', data=df).params['d']
    >>> t, c = df['d'] == 1, df['d'] == 0
    >>> diff = (np.average(df['y'][t], weights=w[t])
    ...         - np.average(df['y'][c], weights=w[c]))
    >>> bool(abs(diff - ols) < 1e-10)
    True

    References
    ----------
    [@chattopadhyay2023implied]
    """
    cols: List[str] = list(covariates)
    frame = data[[treat] + cols]
    if frame.isna().any().any():
        raise MethodIncompatibility(
            "implied_weights: missing values in treat or covariates; a "
            "regression would drop those rows, so drop them first"
        )
    t = frame[treat].to_numpy(dtype=float)
    if set(np.unique(t)) != {0.0, 1.0}:
        raise MethodIncompatibility(
            f"'{treat}' must be binary 0/1 with both groups present"
        )
    estimand = str(estimand).upper()
    if estimand == "ATU":
        estimand = "ATC"
    if estimand not in ("ATE", "ATT", "ATC"):
        raise MethodIncompatibility(
            f"estimand must be 'ATE', 'ATT' or 'ATC'; got {estimand!r}"
        )
    n = len(t)
    X = np.column_stack([np.ones(n), frame[cols].to_numpy(dtype=float)])
    treated = t == 1
    n1, n0 = int(treated.sum()), int((~treated).sum())
    w = np.empty(n)
    if not interactions:
        if estimand != "ATE":
            raise MethodIncompatibility(
                "Without interactions the regression has a single set of "
                "weights; it does not target the treated or the controls.",
                recovery_hint="Use interactions=True to choose estimand=.",
            )
        resid = t - X @ np.linalg.lstsq(X, t, rcond=None)[0]
        base = resid / float(resid @ resid)
        w[treated] = n1 * base[treated]
        w[~treated] = -n0 * base[~treated]
    else:
        if estimand == "ATT":
            target = X[treated].mean(axis=0)
        elif estimand == "ATC":
            target = X[~treated].mean(axis=0)
        else:
            target = X.mean(axis=0)
        for mask, size in ((treated, n1), (~treated, n0)):
            Xg = X[mask]
            if np.linalg.matrix_rank(Xg) < Xg.shape[1]:
                raise MethodIncompatibility(
                    "implied_weights: the covariates are collinear within a "
                    "treatment group, so the arm regression is not identified"
                )
            w[mask] = size * (Xg @ np.linalg.solve(Xg.T @ Xg, target))
    return pd.Series(w, index=data.index, name="implied_weight")
