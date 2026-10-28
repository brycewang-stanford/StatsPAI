"""Hendren's lower bound on private information from elicited beliefs.

People may not report their true loss probability ``P`` on a survey, but
they cannot report more than they know. If the elicitation ``Z`` carries no
information about the loss ``L`` beyond ``P`` (Assumption 1) and beliefs
are unbiased, ``Pr{L | X, P} = P`` (Assumption 2), then the predicted
value ``P_Z = Pr{L | X, Z}`` equals ``E[P | X, Z]``: the true beliefs are a
mean-preserving spread of ``P_Z``. Hendren's Proposition 2 turns this into a
bound on the average magnitude of private information::

    E[m_Z(P_Z) | X] <= E[m(P) | X],   m(p) = E[P | P >= p, X] - p,

the average amount by which the people with higher risk than a randomly
drawn person are riskier.

Estimation follows Section 6 of the paper and the author's
``makem.do``: within each sample (e.g. insurance rejectees), a probit of
``L`` on ``X`` and a flexible basis in ``Z`` gives ``P_Z``, a probit on
``X`` alone gives ``Pr{L | X}``, and ``m_Z`` is computed on the residual
``P_Z - Pr{L | X}``, whose distribution is assumed not to vary with ``X``
inside an aggregation cell (an age group in the paper). A Wald test of the
``Z`` terms is the test for the presence of private information (Test 1);
the bootstrap gives the standard error of the bound, which is not
asymptotically normal when there is no private information.

References
----------
[@hendren2013private]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core._covariates import expands_categorical_covariates
from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
)
from ._results import PrivateInfoBoundResult

__all__ = ["private_info_bound"]

_ALL = "all"


def _scale_elicitation(z: np.ndarray, name: str) -> np.ndarray:
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    if lo >= 0.0 and hi <= 1.0:
        return z
    if lo >= 0.0 and hi <= 100.0:
        return z / 100.0
    raise MethodIncompatibility(
        f"elicitation {name!r} has values in [{lo:g}, {hi:g}], neither a "
        "probability nor a percentage",
        recovery_hint="Pass a probability in [0, 1] or a 0-100 percentage, or "
        "build the basis yourself and pass its columns as a list.",
    )


def _hendren_basis(
    df: pd.DataFrame, z: str, focal: Sequence[float], interact: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Chebyshev terms of ``Z`` plus focal-point indicators, times ``interact``.

    ``T1(p) = p`` and ``T2(p) = 2 p^2 - 1`` evaluated on ``p in [0, 1]``, as in
    the author's ``createhrs_all.do``, plus ``1{p = f}`` for each focal value
    ``f``; every term is also multiplied by each ``interact`` column (the
    paper interacts with linear age).
    """
    p = _scale_elicitation(df[z].to_numpy(dtype=float), z)
    terms: Dict[str, np.ndarray] = {"T1": p, "T2": 2.0 * p**2 - 1.0}
    for f in focal:
        terms[f"focal_{f:g}"] = (p == float(f)).astype(float)
    cols: Dict[str, np.ndarray] = {}
    for t, v in terms.items():
        cols[f"{z}:{t}"] = v
        for a in interact:
            cols[f"{z}:{t}:{a}"] = v * df[a].to_numpy(dtype=float)
    out = pd.DataFrame(cols, index=df.index)
    return out, list(cols)


def _independent(M: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Order-preserving selection of linearly independent columns (Stata's
    ``_rmcoll``: a column is dropped when earlier ones span it)."""
    keep = np.zeros(M.shape[1], dtype=bool)
    basis: List[np.ndarray] = []
    for j in range(M.shape[1]):
        v = M[:, j].astype(float)
        norm0 = float(np.linalg.norm(v))
        if norm0 == 0.0:
            continue
        r = v.copy()
        for q in basis:
            r -= q * (q @ r)
        nr = float(np.linalg.norm(r))
        if nr > tol * norm0:
            keep[j] = True
            basis.append(r / nr)
    return keep


def _tail_m(r: np.ndarray) -> np.ndarray:
    """``E[r' | r' >= r_i] - r_i`` within one cell, ties entering together.

    People with the same covariates and answer get the same residual only up
    to rounding (the fitted values depend on row order in the last bits), so
    sorted values closer than ``1e-10`` times the largest residual form one
    tie.
    """
    order = np.argsort(r, kind="stable")
    rs = r[order]
    tol = 1e-10 * max(float(np.max(np.abs(rs), initial=0.0)), 1e-300)
    new = np.concatenate([[True], np.diff(rs) > tol])
    start = np.flatnonzero(new)
    inverse = np.cumsum(new) - 1
    tail_sum = np.cumsum(rs[::-1])[::-1][start]
    tail_n = (len(rs) - start).astype(float)
    m_sorted = (tail_sum / tail_n)[inverse] - rs
    out = np.empty_like(r)
    out[order] = m_sorted
    return out


def _fit_sample(
    d: pd.DataFrame,
    cov: List[str],
    zt: List[str],
    model: str,
    cluster: Optional[str],
    wald: bool,
) -> Dict[str, Any]:
    """Both fits on one sample, the residuals and (optionally) the Wald test."""
    from ..regression.logit_probit import probit as _probit
    from ..regression.ols import regress as _regress

    def fit(x: List[str]) -> Any:
        if model == "probit":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = _probit(data=d, y="_L", x=x, cluster=cluster)
            if res.model_info.get("n_perfect_prediction_dropped"):
                raise DataInsufficient(
                    "the probit predicts some losses perfectly in this sample",
                    recovery_hint="Merge sparse focal or covariate cells, or "
                    "use model='linear' (the paper's linear specification).",
                )
            return res
        rhs = " + ".join(x) if x else "1"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _regress(f"_L ~ {rhs}", data=d, cluster=cluster)

    def predict(res: Any, x: List[str]) -> np.ndarray:
        b = res.params.reindex(["Intercept"] + x).fillna(0.0).to_numpy()
        xb = b[0] + d[x].to_numpy(dtype=float) @ b[1:]
        return np.asarray(stats.norm.cdf(xb) if model == "probit" else xb)

    full = fit(cov + zt)
    used_z = [t for t in zt if t in full.params.index and full.params[t] != 0.0]
    restricted = fit(cov)
    pz = predict(full, cov + zt)
    px = predict(restricted, cov)
    out: Dict[str, Any] = {"pz": pz, "px": px, "r": pz - px}
    if wald:
        if not used_z:
            raise IdentificationFailure(
                "no elicitation term survives in this sample",
                recovery_hint="The elicitation is constant or collinear with "
                "the covariates here; check the basis.",
            )
        test = full.test(" = ".join(used_z) + " = 0")
        out.update(
            wald_stat=(
                float(test.get("chi2", test["statistic"]))
                if model == "probit"
                else float(test["statistic"])
            ),
            wald_df=int(np.atleast_1d(test["df"])[0]),
            wald_p=float(test["pvalue"]),
        )
    return out


def _bound(
    df: pd.DataFrame,
    cls: np.ndarray,
    labels: List[Any],
    cells: Optional[np.ndarray],
    cov: List[str],
    zt: List[str],
    model: str,
    cluster: Optional[str],
    wald: bool,
) -> Tuple[Dict[Any, Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    res: Dict[Any, Dict[str, Any]] = {}
    m = np.full(len(df), np.nan)
    pz = np.full(len(df), np.nan)
    px = np.full(len(df), np.nan)
    for lab in labels:
        idx = np.flatnonzero(cls == lab)
        d = df.iloc[idx]
        if d["_L"].nunique() < 2:
            raise DataInsufficient(
                f"the loss does not vary in class {lab!r}",
                recovery_hint="Each class needs both losses and non-losses.",
            )
        keep = _independent(
            np.column_stack([np.ones(len(d)), d[cov + zt].to_numpy(dtype=float)])
        )[1:]
        names = cov + zt
        cov_c = [c for c, k in zip(names[: len(cov)], keep[: len(cov)]) if k]
        zt_c = [c for c, k in zip(names[len(cov) :], keep[len(cov) :]) if k]
        fit = _fit_sample(d, cov_c, zt_c, model, cluster, wald)
        r = fit["r"]
        mi = np.empty(len(idx))
        groups = cells[idx] if cells is not None else np.zeros(len(idx))
        for g in pd.unique(groups):
            sel = groups == g
            mi[sel] = _tail_m(r[sel])
        m[idx], pz[idx], px[idx] = mi, fit["pz"], fit["px"]
        res[lab] = {
            "n": len(idx),
            "mean_loss": float(d["_L"].mean()),
            "estimate": float(mi.mean()),
            "sd_residual": float(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
            "z_terms": len(zt_c),
            **{k: v for k, v in fit.items() if k.startswith("wald")},
        }
    return res, m, pz, px


@expands_categorical_covariates("covariates")
def private_info_bound(
    data: pd.DataFrame,
    loss: str,
    elicitation: Union[str, Sequence[str]],
    covariates: Sequence[str],
    classes: Optional[str] = None,
    compare: Optional[Tuple[Any, Any]] = None,
    cells: Optional[str] = None,
    interact: Optional[Union[str, Sequence[str]]] = None,
    focal: Sequence[float] = (0.0, 0.5, 1.0),
    model: Literal["probit", "linear"] = "probit",
    cluster: Optional[str] = None,
    n_boot: int = 0,
    seed: Optional[int] = 0,
    alpha: float = 0.05,
) -> PrivateInfoBoundResult:
    """Hendren's (2013) nonparametric lower bound on private information.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per person.
    loss : str
        0/1 indicator that the insured event happened (e.g. entering a
        nursing home within five years).
    elicitation : str or list of str
        The elicited probability ``Z`` of the loss (a column in ``[0, 1]``
        or a 0-100 percentage); its basis is built as in the paper:
        Chebyshev terms ``T1(Z) = Z`` and ``T2(Z) = 2 Z^2 - 1`` plus an
        indicator for each ``focal`` value, each also multiplied by the
        ``interact`` columns. A list instead names basis columns already in
        ``data`` (e.g. the author's ``pnc*`` variables), used as given.
    covariates : list of str
        The public information ``X`` insurers price on. Categorical columns
        are expanded into indicators.
    classes : str, optional
        Sample classification (e.g. rejected / not rejected by insurers).
        Both probits and the bound are computed separately in each class,
        as in the paper. Default: one sample.
    compare : tuple, optional
        ``(a, b)``: report ``Delta_Z = bound[a] - bound[b]`` (the paper's
        rejectees minus non-rejectees) with the conservative one-sided
        p-value of Section 6.2. Needs ``classes`` and ``n_boot > 0``.
    cells : str, optional
        Aggregation cells inside a class (the paper's age groups): the
        distribution of ``P_Z - Pr{L | X}`` is assumed not to vary with
        ``X`` within a cell, and ``m_Z`` is computed within it.
    interact : str or list of str, optional
        Columns the elicitation basis is interacted with (linear age in the
        paper). Only with a column ``elicitation``.
    focal : sequence of float, default (0, 0.5, 1)
        Focal responses that get their own indicator, on the probability
        scale. The paper uses 0, 50 and 100 percent.
    model : {'probit', 'linear'}, default 'probit'
        ``Pr{L | X, Z}`` by probit (the paper's preferred specification) or
        a linear probability model.
    cluster : str, optional
        Cluster for the Wald test (the paper: household) and for block
        resampling in the bootstrap.
    n_boot : int, default 0
        Bootstrap replications that refit everything (0 = no standard
        error). The paper uses 1000.
    seed : int, optional
        Seed for the bootstrap.
    alpha : float, default 0.05
        Level of the percentile intervals.

    Returns
    -------
    PrivateInfoBoundResult
        ``table`` (one row per class: bound, bootstrap SE and interval,
        Wald test of no private information), ``delta`` (with
        ``compare``), and per-row ``P_Z``, ``Pr{L | X}`` and ``m_Z`` in
        ``rows``.

    Notes
    -----
    ``m_Z(r_i)`` is the mean of the residuals ``r_j >= r_i`` in the cell
    minus ``r_i``; people tied at ``r_i`` all enter, so the result does not
    depend on row order. The author's ``makem.do`` breaks ties by sort
    position, which gives the same number whenever no two residuals in a
    cell are equal.

    The bound is invariant to one-to-one transformations of ``Z`` only when
    the basis is saturated in ``Z``; a polynomial basis is not.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> from scipy.stats import norm
    >>> rng = np.random.default_rng(0)
    >>> n = 3000
    >>> x, v = rng.normal(size=n), rng.normal(size=n)
    >>> p = norm.cdf(-1 + 0.5 * x + 0.6 * v)
    >>> df = pd.DataFrame({"loss": (rng.uniform(size=n) < p).astype(int),
    ...                    "z": norm.cdf(v), "x": x})
    >>> res = sp.private_info_bound(df, "loss", "z", ["x"])
    >>> bool(res.table["wald_p"].iloc[0] < 0.01)
    True

    References
    ----------
    [@hendren2013private]
    """
    if model not in ("probit", "linear"):
        raise MethodIncompatibility(
            f"model must be 'probit' or 'linear', got {model!r}",
            recovery_hint="Use model='probit'.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"alpha must be in (0, 1), got {alpha!r}",
            recovery_hint="Use e.g. alpha=0.05.",
        )
    if n_boot < 0:
        raise MethodIncompatibility(
            f"n_boot must be >= 0, got {n_boot!r}",
            recovery_hint="Use n_boot=0 for no bootstrap or e.g. 1000.",
        )
    cov = list(covariates)
    inter = [interact] if isinstance(interact, str) else list(interact or [])
    user_basis = not isinstance(elicitation, str)
    if user_basis and inter:
        raise MethodIncompatibility(
            "interact= applies to the built-in basis only",
            recovery_hint="Include the interactions in your basis columns.",
        )
    z_cols = [elicitation] if isinstance(elicitation, str) else list(elicitation)
    if not z_cols:
        raise MethodIncompatibility(
            "elicitation names no column",
            recovery_hint="Pass the elicitation column or basis columns.",
        )
    extra = [c for c in (classes, cells, cluster) if c is not None]
    need = [loss] + z_cols + cov + inter + extra
    check_columns(data, need, function="private_info_bound")
    df = data[list(dict.fromkeys(need))].dropna().copy()
    if df.empty:
        raise DataInsufficient(
            "no complete rows", recovery_hint="Check the columns for missing values."
        )
    yl = df[loss].to_numpy(dtype=float)
    if not np.isin(yl, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"loss {loss!r} must be 0/1",
            recovery_hint="Code the loss as an indicator.",
        )
    df["_L"] = yl
    if user_basis:
        zt = z_cols
    else:
        basis, zt = _hendren_basis(df, z_cols[0], focal, inter)
        rename = {c: f"_z{j}" for j, c in enumerate(zt)}
        df = pd.concat([df, basis.rename(columns=rename)], axis=1)
        zt_labels = dict(zip(rename.values(), rename.keys()))
        zt = list(rename.values())
    cls = df[classes].to_numpy() if classes else np.full(len(df), _ALL, dtype=object)
    labels = list(pd.unique(cls))
    if compare is not None:
        if classes is None or len(compare) != 2:
            raise MethodIncompatibility(
                "compare=(a, b) needs classes= and two class labels",
                recovery_hint="e.g. classes='reject', compare=(1, 0).",
            )
        missing = [c for c in compare if c not in labels]
        if missing:
            raise MethodIncompatibility(
                f"compare labels {missing} are not values of {classes!r}",
                recovery_hint=f"Use labels from {labels}.",
            )
    cell_arr = df[cells].to_numpy() if cells else None

    point, m, pz, px = _bound(df, cls, labels, cell_arr, cov, zt, model, cluster, True)
    delta = None
    ca, cb = compare if compare is not None else (None, None)
    if compare is not None:
        delta = {
            "a": ca,
            "b": cb,
            "estimate": point[ca]["estimate"] - point[cb]["estimate"],
        }

    boot: Dict[Any, List[float]] = {lab: [] for lab in labels}
    boot_delta: List[float] = []
    n_failed = 0
    if n_boot:
        rng = np.random.default_rng(seed)
        units = df[cluster].to_numpy() if cluster else np.arange(len(df))
        uniq, inv = np.unique(units, return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
        for _ in range(n_boot):
            draw = rng.integers(0, len(uniq), size=len(uniq))
            rows = np.concatenate([members[g] for g in draw])
            db = df.iloc[rows].reset_index(drop=True)
            try:
                pb, _, _, _ = _bound(
                    db,
                    cls[rows],
                    labels,
                    cell_arr[rows] if cell_arr is not None else None,
                    cov,
                    zt,
                    model,
                    None,
                    False,
                )
            except (DataInsufficient, IdentificationFailure, np.linalg.LinAlgError):
                n_failed += 1
                continue
            for lab in labels:
                boot[lab].append(pb[lab]["estimate"])
            if delta is not None:
                boot_delta.append(pb[ca]["estimate"] - pb[cb]["estimate"])
        if n_failed:
            warnings.warn(
                f"{n_failed} of {n_boot} bootstrap replications failed and were "
                "skipped",
                AssumptionWarning,
                stacklevel=2,
            )

    q = (alpha / 2, 1 - alpha / 2)
    rows_out = []
    for lab in labels:
        b = np.asarray(boot[lab])
        se = float(np.std(b, ddof=1)) if len(b) > 1 else np.nan
        lo, hi = np.quantile(b, q) if len(b) > 1 else (np.nan, np.nan)
        rows_out.append(
            {"class": lab, **point[lab], "se": se, "ci_lower": lo, "ci_upper": hi}
        )
    table = pd.DataFrame(rows_out).set_index("class")
    if delta is not None:
        bd = np.asarray(boot_delta)
        p_boot = float(np.mean(bd <= 0.0)) if len(bd) else np.nan
        p_wald = point[ca]["wald_p"]
        delta.update(
            se=float(np.std(bd, ddof=1)) if len(bd) > 1 else np.nan,
            ci=tuple(np.quantile(bd, q)) if len(bd) > 1 else (np.nan, np.nan),
            p_boot=p_boot,
            p_wald=p_wald,
            pvalue=max(p_boot, p_wald) if len(bd) else np.nan,
        )
    rows = pd.DataFrame(
        {"class": cls, "p_z": pz, "p_x": px, "residual": pz - px, "m_z": m},
        index=df.index,
    )
    estimate = (
        delta["estimate"]
        if delta is not None
        else (point[labels[0]]["estimate"] if len(labels) == 1 else None)
    )
    basis_names = zt if user_basis else [zt_labels[t] for t in zt]
    return PrivateInfoBoundResult(
        table=table,
        estimate=estimate,
        delta=delta,
        rows=rows,
        alpha=alpha,
        n_obs=len(df),
        model_info={
            "model": model,
            "basis": basis_names,
            "cells": cells,
            "classes": classes,
            "cluster": cluster,
            "n_boot": n_boot,
            "n_boot_failed": n_failed,
            "seed": seed,
            "ties": "inclusive",
        },
    )
