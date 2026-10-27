"""Covariance, delta-method margins and prediction for ``sp.fmm``."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.utils import create_design_matrices
from ..exceptions import (
    ColumnNotFound,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
)
from . import _fmm_core as core
from ._optim_helpers import inverse_information, require_two_clusters, se_from_vcov


def _cons(name: str) -> str:
    return "_cons" if name in ("Intercept", "const", "_cons") else str(name)


def _check_outcome(y: np.ndarray, fam: str) -> None:
    if fam == "logit" and not np.all(np.isin(y, (0.0, 1.0))):
        raise MethodIncompatibility(
            "fmm: family='logit' needs a 0 / 1 outcome.",
            recovery_hint="Recode the outcome to 0 / 1, or use family='gaussian'.",
        )
    if fam == "poisson" and np.any(y < 0):
        raise MethodIncompatibility(
            "fmm: family='poisson' needs a non-negative outcome.",
            recovery_hint="Use family='gaussian' for an outcome that can be negative.",
        )
    if np.unique(y).size < 2:
        raise DataInsufficient(
            "fmm: the outcome is constant in the estimation sample.",
            recovery_hint="Check the sample restriction.",
        )


def covariance(
    H: np.ndarray,
    scores: np.ndarray,
    w: np.ndarray,
    vce: str,
    sampling: bool,
    clusters: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[int]]:
    A = -H
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise IdentificationFailure(
            "fmm: the information matrix is not positive definite at the "
            "maximum; some parameters are not identified.",
            recovery_hint="Use fewer classes or drop class covariates.",
            diagnostics={"min_eigenvalue": float(np.linalg.eigvalsh(A).min())},
        ) from None
    Ainv = inverse_information(H)
    if vce == "oim":
        return Ainv, None
    if vce == "robust":
        if sampling:
            u = w[:, None] * scores
            meat = u.T @ u
            N = float(len(w))
        else:
            meat = (scores * w[:, None]).T @ scores
            N = float(w.sum())
        return N / (N - 1.0) * Ainv @ meat @ Ainv, None
    G = require_two_clusters(clusters, "fmm")
    codes = pd.factorize(pd.Series(clusters))[0]
    u = np.zeros((G, scores.shape[1]))
    np.add.at(u, codes, w[:, None] * scores)
    return G / (G - 1.0) * Ainv @ (u.T @ u) @ Ainv, G


def _cs_jacobian(f: Any, theta: np.ndarray, h: float = 1e-30) -> np.ndarray:
    cols = []
    for j in range(theta.size):
        tc = theta.astype(complex)
        tc[j] += 1j * h
        cols.append(np.imag(f(tc)) / h)
    return np.column_stack(cols)


def margins(
    model: core.MixtureModel, theta: np.ndarray, V: np.ndarray, alpha: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``estat lcprob`` and ``estat lcmean`` with delta-method errors."""
    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    wn = model.w / model.w.sum()
    labels = [f"{j}.Class" for j in range(1, model.K + 1)]

    def prob(t: np.ndarray) -> np.ndarray:
        return wn @ np.exp(model.class_log_prior(t))

    def mean(t: np.ndarray) -> np.ndarray:
        return wn @ model.class_mean(t)

    out = []
    for f, logit_ci in ((prob, True), (mean, model.family == "logit")):
        est = np.real(f(theta.astype(complex)))
        Jm = _cs_jacobian(f, theta)
        se = se_from_vcov(Jm @ V @ Jm.T)
        if logit_ci:
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = np.log(est / (1.0 - est))
                half = crit * se / (est * (1.0 - est))
            lo, hi = 1.0 / (1.0 + np.exp(-(lg - half))), 1.0 / (
                1.0 + np.exp(-(lg + half))
            )
        else:
            lo, hi = est - crit * se, est + crit * se
        out.append(
            pd.DataFrame(
                {"estimate": est, "std_error": se, "conf_low": lo, "conf_high": hi},
                index=pd.Index(labels, name="class"),
            )
        )
    return out[0], out[1]


def make_predictor(
    model: core.MixtureModel,
    theta: np.ndarray,
    formula: str,
    xnames: List[str],
    zcols: List[str],
    labels: List[str],
    X_df: Any,
) -> Any:
    depvar_raw = formula.split("~", 1)[0].strip()
    design_cols = [str(c) for c in getattr(X_df, "columns", [])]

    def predict(newdata: pd.DataFrame, key: str) -> pd.DataFrame:
        nd = newdata.reset_index(drop=True)
        has_y = depvar_raw in nd.columns
        if key == "classposteriorpr" and not has_y:
            raise ColumnNotFound(
                f"fmm.predict: type='classposteriorpr' needs the outcome "
                f"{depvar_raw!r} in newdata.",
                recovery_hint="Add the outcome column, or use type='classpr'.",
            )
        frame = nd if has_y else nd.assign(**{depvar_raw: 0.0})
        y_new, X_new = create_design_matrices(formula, frame)
        rows = np.asarray(y_new.index)
        Xc = [_cons(str(c)) for c in X_new.columns]
        lost = [c for c in xnames if c not in Xc]
        if lost:
            raise MethodIncompatibility(
                "fmm.predict: newdata does not produce the fitted design "
                f"columns {lost}.",
                recovery_hint="Use factor levels seen in the estimation sample.",
                diagnostics={"design_columns": design_cols},
            )
        X = np.asarray(X_new, dtype=float)[:, [Xc.index(c) for c in xnames]]
        sub = nd.loc[rows]
        Z = np.column_stack(
            [sub[c].to_numpy(dtype=float) for c in zcols] + [np.ones(len(rows))]
        )
        idx = newdata.index[rows]
        if key == "mu":
            return pd.DataFrame(
                np.real(model.class_mean(theta, X)), index=idx, columns=labels
            )
        prior = np.exp(np.real(model.class_log_prior(theta, Z)))
        if key == "classpr":
            return pd.DataFrame(prior, index=idx, columns=labels)
        if key == "marginal":
            m = (prior * np.real(model.class_mean(theta, X))).sum(axis=1)
            return pd.DataFrame({"marginal": m}, index=idx)
        y = np.asarray(y_new, dtype=float).ravel()
        with np.errstate(all="ignore"):
            post = np.real(model.joint(theta, X=X, Z=Z, y=y)[2])
        return pd.DataFrame(post, index=idx, columns=labels)

    return predict
