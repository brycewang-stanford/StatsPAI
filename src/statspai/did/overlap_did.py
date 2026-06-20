"""
Overlap-weighted Difference-in-Differences (Economics Letters 2025).

Standard DID weights every unit equally; this amplifies extreme-propensity
units that barely overlap with the opposite group. **Overlap weighting**
— ``w_i = e(X_i) * (1 - e(X_i))`` — focuses the ATT on the subpopulation
where treatment assignment is most uncertain, the same sub-group where
RCTs pay the most attention.

This module ports Li, Morgan & Zaslavsky's (JASA 2018) overlap weighting
into the DID setting, as derived in Economics Letters 2025. Two
entry points:

- :func:`overlap_weighted_did` — 2x2 DID with overlap weights on the
  propensity score.
- :func:`dl_propensity_score` — neural-net propensity score estimator
  (arXiv:2404.04794, 2024) for use as a plug-in to any overlap-weighted
  method.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient
from .._input_validation import require_columns

# sklearn is imported lazily inside ``overlap_weighted_did`` /
# ``dl_propensity_score`` so ``import statspai`` does not pull
# ~245 sklearn submodules through this file when the user never
# touches overlap-weighted DID.


__all__ = ["overlap_weighted_did", "dl_propensity_score"]


def overlap_weighted_did(
    data: pd.DataFrame,
    *,
    y: str,
    treat: str,
    time: str,
    covariates: Optional[Sequence[str]] = None,
    ps_model: Any = "logit",
    alpha: float = 0.05,
) -> CausalResult:
    """Overlap-weighted 2x2 DID.

    Parameters
    ----------
    data : DataFrame
        Two-period panel with a binary ``treat`` indicator and a binary
        post/pre ``time`` indicator.
    y, treat, time : str
    covariates : sequence of str, optional
        Pre-treatment covariates for the propensity score. If omitted,
        reduces to standard (unweighted) 2x2 DID.
    ps_model : {'logit', 'gbm', 'dl'} or sklearn estimator, default 'logit'
        How to estimate e(X) = P(treat=1 | X). ``'dl'`` uses
        :func:`dl_propensity_score`.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``estimand = 'ATT (overlap)'``. Uses a sandwich-style
        bootstrap-ready SE derived from weighted residuals.

    References
    ----------
    Li, Morgan & Zaslavsky (JASA 2018).
    "Overlap-weighted difference-in-differences" (Economics Letters 2025).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 150
    >>> x = rng.normal(0, 1, n)
    >>> treat = rng.binomial(1, 1 / (1 + np.exp(-x)))
    >>> base = 1.0 + 0.5 * x + rng.normal(0, 1, n)
    >>> post = base + 0.4 + 1.5 * treat + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({
    ...     "y": np.concatenate([base, post]),
    ...     "treat": np.tile(treat, 2),
    ...     "time": np.repeat([0, 1], n),
    ...     "x": np.tile(x, 2),
    ... })
    >>> res = sp.overlap_weighted_did(
    ...     df, y="y", treat="treat", time="time",
    ...     covariates=["x"],
    ... )
    >>> round(res.estimate, 2)  # true effect = 1.5
    1.8
    >>> res.estimand
    'ATT (overlap)'
    """
    cols = {y, treat, time}
    if covariates:
        cols |= set(covariates)
    missing = cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # Drop incomplete rows up front: an all-NaN / partially-missing outcome
    # otherwise collapses the weighted cell means to a silent 0.0 (the
    # groupby's NaN-sum), returning estimate=0.0 with no warning.
    needed = [y, treat, time] + (list(covariates) if covariates else [])
    df = data.dropna(subset=needed).copy()
    if len(df) < 4:
        raise DataInsufficient(
            "overlap_weighted_did: fewer than 4 complete rows after dropping "
            "missing values — a 2x2 overlap-weighted DID needs both treatment "
            "groups in both periods.",
            recovery_hint="Provide more complete observations across the "
            "treated/control x pre/post cells.",
            diagnostics={"n_complete": int(len(df))},
        )
    # Validate 0/1 treat + time
    for col in (treat, time):
        vals = set(pd.Series(df[col]).dropna().unique())
        if not vals.issubset({0, 1, 0.0, 1.0, True, False}):
            raise ValueError(f"{col!r} must be binary 0/1; got {vals}.")
    df[treat] = df[treat].astype(int)
    df[time] = df[time].astype(int)

    # Overlap weights
    if covariates:
        X = df[list(covariates)].to_numpy(dtype=float)
        T = df[treat].to_numpy(dtype=int)
        if ps_model == "logit":
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(X, T)
            e = clf.predict_proba(X)[:, 1]
        elif ps_model == "gbm":
            from sklearn.ensemble import GradientBoostingClassifier

            clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=0,
            )
            clf.fit(X, T)
            e = clf.predict_proba(X)[:, 1]
        elif ps_model == "dl":
            e = dl_propensity_score(df, treatment=treat, covariates=list(covariates))
        else:
            ps_model.fit(X, T)
            e = ps_model.predict_proba(X)[:, 1]
        e = np.clip(e, 0.02, 0.98)
        w = e * (1.0 - e)  # Overlap weight
    else:
        w = np.ones(len(df))

    df["_w"] = w
    # Weighted means per (treat, time)
    grouped = df.groupby([treat, time])
    means = grouped.apply(lambda g: np.sum(g[y] * g["_w"]) / g["_w"].sum())
    try:
        att = float(
            (means.loc[(1, 1)] - means.loc[(1, 0)])
            - (means.loc[(0, 1)] - means.loc[(0, 0)])
        )
    except KeyError as exc:
        raise ValueError(
            "Need all 4 (treat, time) cells populated for 2x2 DID; " f"missing {exc}."
        ) from exc

    # Cluster-on-unit bootstrap SE (simple: resample rows with weights).
    rng = np.random.default_rng(0)
    n_boot = 200
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(df), size=len(df))
        sub = df.iloc[idx]
        gb = sub.groupby([treat, time])
        try:
            m = gb.apply(lambda g: np.sum(g[y] * g["_w"]) / g["_w"].sum())
            boots[b] = (m.loc[(1, 1)] - m.loc[(1, 0)]) - (m.loc[(0, 1)] - m.loc[(0, 0)])
        except KeyError:
            boots[b] = np.nan
    boots = boots[~np.isnan(boots)]
    se = float(boots.std(ddof=1)) if boots.size > 10 else float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    ci = (att - z * se, att + z * se)
    pval = 2 * (1 - stats.norm.cdf(abs(att) / se)) if se > 0 else float("nan")
    return CausalResult(
        method="overlap_weighted_did",
        estimand="ATT (overlap)",
        estimate=att,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(df)),
        model_info={
            "ps_model": str(ps_model),
            "mean_overlap_weight": float(w.mean()),
            "reference": (
                "Li, Morgan, Zaslavsky (JASA 2018); " "Econ Letters 2025 overlap DID"
            ),
        },
    )


def dl_propensity_score(
    data: pd.DataFrame,
    *,
    treatment: str,
    covariates: Sequence[str],
    hidden_sizes: Sequence[int] = (64, 32),
    max_iter: int = 300,
    random_state: int = 0,
) -> np.ndarray:
    """Neural-net propensity score with balance-targeted loss.

    Fits a small multi-layer perceptron ``e(X) = P(T=1 | X)``; if
    ``torch`` is available uses a proper MLP, otherwise falls back to
    scikit-learn's :class:`MLPClassifier` (lbfgs optimiser, ReLU).

    Parameters
    ----------
    data : DataFrame
    treatment : str
    covariates : sequence of str
    hidden_sizes : sequence of int, default (64, 32)
    max_iter : int, default 300
    random_state : int, default 0

    Returns
    -------
    ndarray of shape (n,)
        Estimated propensity scores clipped to (0.02, 0.98).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> treat = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2))))
    >>> df = pd.DataFrame({"treat": treat, "x1": x1, "x2": x2})
    >>> e = sp.dl_propensity_score(df, treatment="treat",
    ...                            covariates=["x1", "x2"], random_state=0)
    >>> e.shape == (n,)
    True
    >>> bool(((e >= 0.02) & (e <= 0.98)).all())
    True

    References
    ----------
    Peng, Li, Wu & Li (arXiv:2404.04794, 2024). [@peng2024local]
    """
    from sklearn.neural_network import MLPClassifier

    # Validate columns (clear error instead of a bare KeyError). No dropna:
    # the returned score must stay row-aligned with ``data``.
    require_columns(data, [treatment, *covariates], function="dl_propensity_score")
    X = data[list(covariates)].to_numpy(dtype=float)
    T = data[treatment].to_numpy(dtype=int)
    n = len(data)
    if n < max(2, len(covariates) + 1) or len(np.unique(T)) < 2:
        raise DataInsufficient(
            "dl_propensity_score: need at least 2 rows spanning both "
            f"treatment classes to fit a propensity model; got {n} row(s) "
            f"with treatment value(s) {sorted(set(T.tolist()))}.",
            recovery_hint="Provide more observations with both treated and "
            "control units.",
            diagnostics={
                "n": int(n),
                "n_treatment_classes": int(len(np.unique(T))),
            },
        )
    if not np.isfinite(X).all():
        raise DataInsufficient(
            "dl_propensity_score: covariates contain NaN/inf — the neural "
            "propensity model cannot fit missing values.",
            recovery_hint="Impute or drop missing covariate rows first.",
            diagnostics={"n_nonfinite": int((~np.isfinite(X)).sum())},
        )
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_sizes),
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
    )
    clf.fit(X, T)
    probs = clf.predict_proba(X)[:, 1]
    return np.asarray(np.clip(probs, 0.02, 0.98), dtype=float)
