"""Imputation-score inference for causal forests with fixed effects.

A forest fitted with ``fe="unit"`` / ``fe="twoway"`` predicts ``tau(x)`` but
has no doubly-robust score: the AIPW correction needs a propensity
``E[D | X]``, which a within-unit design does not define.  This module
supplies the missing unbiased signal from the same model the forest
assumes,

    Y_it = alpha_i + gamma_t + tau(X_it) D_it + e_it,

by *imputation* [borusyak2024revisiting]: the unit and period effects are
fitted on the untreated cells only, and every treated cell gets

    Gamma_it = Y_it - (alpha_hat_i + gamma_hat_t),

an estimate of its own effect that is unbiased under parallel trends and
no anticipation whatever the heterogeneity -- in ``x``, in calendar time or
in exposure.  The forest then plays the role of a proxy, as in the generic
machine-learning inference of [chernozhukov2025generic]:

* the average effect on the treated is the mean of ``Gamma`` (it is the
  imputation estimator of the ATT, so it inherits the robustness of that
  estimator to staggered adoption with dynamic effects);
* a group average (a country pair, a country, a period, a quantile of the
  forest's prediction) is the mean of ``Gamma`` over the group's treated
  cells;
* the best linear projection regresses ``Gamma`` on covariates, and the
  calibration regression regresses it on the out-of-bag forest prediction,
  which gives a heterogeneity test *and* a slope that does correct the
  forest's shrinkage.

Every such quantity is linear in the outcome, ``theta = v' y``; the weights
``v`` are the target weights on treated cells minus their least-squares
projection through the untreated design (the construction of
``did._bjs_variance``).  Variances are sums of cluster (or dyadic) score
sums of ``v * eps``.  Untreated cells use their ``Y(0)`` residual; treated
cells use ``Gamma`` minus a heterogeneity model, either the
``v^2``-weighted cohort-by-relative-time mean of the Borusyak--Jaravel--
Spiess convention (``variance="bjs"``, conservative) or the out-of-bag
forest prediction followed by the same block centring
(``variance="forest"``).  The forest prediction for a unit never uses that
unit's data (trees draw whole units), so it can model the heterogeneity
without absorbing the unit's own noise.

Dyadic data (country pairs) can use the dyadic-robust variance of
[aronow2015cluster]: any two rows that share a member are allowed to be
correlated.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from ..utils._attrs import attrs_array

_VARIANCES = ("forest", "bjs")


# --------------------------------------------------------------------------- #
#  The imputation design
# --------------------------------------------------------------------------- #


@dataclass
class ImputationDesign:
    """Everything the linear functionals need, computed once per forest."""

    n: int
    treated: np.ndarray  # D == 1
    target: np.ndarray  # treated and imputable
    Z: sparse.csr_matrix  # FE design, all rows (reference levels dropped)
    gram_lu: Any  # splu of Z0'Z0
    y: np.ndarray
    y0_hat: np.ndarray  # alpha_hat + gamma_hat, all rows
    gamma: np.ndarray  # Y - y0_hat on target rows, NaN elsewhere
    unit: np.ndarray
    time: np.ndarray
    cohort: np.ndarray  # first treated period code (-1: never treated)
    rel: np.ndarray  # time - cohort (0 for never treated)
    clusters: np.ndarray
    obs_weight: np.ndarray
    n_not_imputable: int
    absorbing: bool
    control_names: List[str]
    control_coef: np.ndarray


def _check_binary(T: np.ndarray, context: str) -> None:
    values = np.unique(T)
    if not np.all(np.isin(values, (0.0, 1.0))):
        raise MethodIncompatibility(
            f"{context}: imputation scores need a binary 0/1 treatment.",
            recovery_hint=(
                "Recode the treatment as 0/1; a continuous treatment has no "
                "untreated state to impute."
            ),
            diagnostics={"treatment_values": values[:10].tolist()},
        )


def _candidate_covariates(forest: Any, covariates: Any) -> Tuple[np.ndarray, List[str]]:
    """Resolve ``covariates`` into an (n, k) matrix and names."""
    n = int(len(forest._Y_original))
    X = np.asarray(forest._X_original, dtype=float)
    W = getattr(forest, "_fe_W", None)
    x_names = list(
        getattr(forest, "_feature_names", None) or [f"X{j}" for j in range(X.shape[1])]
    )
    w_names = list(getattr(forest, "_control_names", None) or [])
    if W is not None and len(w_names) != W.shape[1]:
        w_names = [f"W{j}" for j in range(W.shape[1])]
    pool = X if W is None else np.hstack([X, W])
    names = x_names + (w_names if W is not None else [])
    if covariates is None or (isinstance(covariates, str) and covariates == "none"):
        return np.zeros((n, 0)), []
    if isinstance(covariates, str) and covariates == "auto":
        return pool, names
    if isinstance(covariates, str):
        raise MethodIncompatibility(
            f"covariates must be 'auto', 'none', a list of names or an "
            f"(n, k) array, got {covariates!r}.",
            recovery_hint=(
                "Use covariates='auto' (the time-varying effect modifiers and "
                "controls of the forest)."
            ),
        )
    if isinstance(covariates, (list, tuple)) and all(
        isinstance(c, str) for c in covariates
    ):
        missing = [c for c in covariates if c not in names]
        if missing:
            raise MethodIncompatibility(
                f"covariates names {missing} are not effect modifiers or controls "
                "of the forest.",
                recovery_hint=f"Choose among {names}, or pass an (n, k) array.",
            )
        idx = [names.index(c) for c in covariates]
        return pool[:, idx], list(covariates)
    arr = np.asarray(covariates, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != n or not np.isfinite(arr).all():
        raise MethodIncompatibility(
            "covariates array must have one finite row per training row.",
            recovery_hint="Pass an (n, k) numeric array aligned with the fit rows.",
        )
    return arr, [f"control{j}" for j in range(arr.shape[1])]


def _cache_key(covariates: Any) -> Any:
    if covariates is None or isinstance(covariates, str):
        return ("spec", covariates)
    if isinstance(covariates, (list, tuple)) and all(
        isinstance(c, str) for c in covariates
    ):
        return ("names", tuple(covariates))
    # Content hash, never id(): a temporary array can be freed and a new one
    # reuse its address, which would silently return another design.
    import hashlib

    arr = np.ascontiguousarray(np.asarray(covariates, dtype=float))
    # Cache key only, never a security digest.
    digest = hashlib.sha1(arr.tobytes(), usedforsecurity=False).hexdigest()
    return ("array", arr.shape, digest)


def imputation_design(
    forest: Any, context: str = "imputation", covariates: Any = "none"
) -> ImputationDesign:
    """Fit the untreated two-way model and form the imputation scores.

    ``Y(0) = alpha_i + gamma_t + C_it' beta`` is fitted on untreated cells.
    ``covariates="none"`` (default) is the pure two-way model -- exactly
    :func:`statspai.did_imputation` without covariates; ``"auto"`` uses the
    forest's effect modifiers and covariates that vary within units
    (time-invariant columns are absorbed by the unit effect and dropped); a
    list of names or an array selects them explicitly.  Controls enter
    linearly, as in :func:`statspai.did_imputation`, and must not be
    affected by the treatment.

    Requires a forest fitted with ``fe="twoway"``: with ``fe="unit"`` there
    are no period ids, so neither period effects nor adoption cohorts are
    defined.

    Cached on the forest per ``covariates``.  Treated cells whose unit
    has no untreated period, or whose period has no untreated cell in the
    same connected component of the untreated unit-period graph, cannot be
    imputed; they are dropped from every target with a warning (and counted
    in ``n_not_imputable``) rather than imputed from an unidentified
    normalisation.
    """
    cache = getattr(forest, "_fe_imputation", None)
    if not isinstance(cache, dict):
        cache = {}
    key = _cache_key(covariates)
    if key in cache:
        return cache[key]  # type: ignore[no-any-return]
    if getattr(forest, "fe", None) != "twoway":
        raise MethodIncompatibility(
            f"{context}: imputation scores need period ids, i.e. a forest "
            "fitted with fe='twoway'.",
            recovery_hint=(
                "Refit with sp.causal_forest(..., id=, time=, fe='twoway'). "
                "With fe='unit' common period shocks are not removed, so a "
                "staggered design is not identified by unit effects alone."
            ),
            diagnostics={"fe": getattr(forest, "fe", None)},
        )
    Y = np.asarray(forest._Y_original, dtype=float)
    T = np.asarray(forest._T_original, dtype=float)
    _check_binary(T, context)
    unit = np.asarray(forest._fe_unit, dtype=np.int64)
    time = np.asarray(forest._fe_time, dtype=np.int64)
    n = Y.size
    n_u = int(unit.max()) + 1
    n_t = int(time.max()) + 1
    treated = T == 1.0
    untreated = ~treated
    if not untreated.any():
        raise DataInsufficient(
            f"{context}: there are no untreated cells to fit the unit and "
            "period effects on.",
            recovery_hint="Imputation needs untreated periods for treated units.",
        )

    # Components of the bipartite untreated graph: unit nodes 0..n_u-1,
    # period nodes n_u..n_u+n_t-1.
    rows0 = np.flatnonzero(untreated)
    graph = sparse.coo_matrix(
        (np.ones(rows0.size), (unit[rows0], n_u + time[rows0])),
        shape=(n_u + n_t, n_u + n_t),
    )
    _, comp = connected_components(graph, directed=False)
    unit_seen = np.bincount(unit[rows0], minlength=n_u) > 0
    time_seen = np.bincount(time[rows0], minlength=n_t) > 0
    imputable = unit_seen[unit] & time_seen[time] & (comp[unit] == comp[n_u + time])
    target = treated & imputable
    n_not = int(np.sum(treated & ~imputable))
    if not target.any():
        raise DataInsufficient(
            f"{context}: no treated cell can be imputed (every treated unit "
            "needs an untreated period, and every treated period an untreated "
            "cell connected to it).",
            recovery_hint=(
                "Imputation needs units observed before they are treated and "
                "not-yet- or never-treated comparison units."
            ),
        )
    if n_not:
        warnings.warn(
            f"{context}: {n_not} treated cell(s) cannot be imputed (always-"
            "treated units, or periods with no connected untreated cell) and "
            "are excluded from every average.",
            AssumptionWarning,
            stacklevel=3,
        )

    # Columns: every unit seen untreated, every period seen untreated minus
    # one reference period per component.
    unit_col = np.full(n_u, -1, dtype=np.int64)
    unit_col[unit_seen] = np.arange(int(unit_seen.sum()))
    next_col = int(unit_seen.sum())
    time_col = np.full(n_t, -1, dtype=np.int64)
    ref_taken = set()
    for t in range(n_t):
        if not time_seen[t]:
            continue
        c = int(comp[n_u + t])
        if c not in ref_taken:
            ref_taken.add(c)  # lowest period of each component is the reference
            continue
        time_col[t] = next_col
        next_col += 1
    ucols = unit_col[unit]
    tcols = time_col[time]
    ridx = np.arange(n)
    mu = ucols >= 0
    mt = tcols >= 0
    Z = sparse.csr_matrix(
        (
            np.ones(int(mu.sum() + mt.sum())),
            (
                np.concatenate([ridx[mu], ridx[mt]]),
                np.concatenate([ucols[mu], tcols[mt]]),
            ),
        ),
        shape=(n, next_col),
    )
    # Linear covariates: keep columns that vary within units and periods on
    # the untreated cells (two-way demeaned residual variation), dropping
    # collinear ones.
    C, cnames = _candidate_covariates(forest, covariates)
    kept: List[int] = []
    if C.shape[1]:
        from ._grf_engine import _fe_residualize

        idx0 = rows0.astype(np.int64)
        resid_cols: List[np.ndarray] = []
        for j in range(C.shape[1]):
            out = np.zeros(n)
            sweeps = _fe_residualize(
                idx0,
                np.ascontiguousarray(C[:, j]),
                np.ones(n),
                unit,
                time,
                np.zeros(n_u),
                np.zeros(n_u),
                np.zeros(n_t),
                np.zeros(n_t),
                out,
                100000,
                1e-12,
            )
            if sweeps < 0:
                raise DataInsufficient(
                    f"{context}: the within transformation of control "
                    f"{cnames[j]!r} did not converge.",
                    recovery_hint=(
                        "Check the untreated panel for disconnected "
                        "unit-period sets."
                    ),
                )
            r = out[rows0]
            scale = float(np.sum((C[rows0, j] - C[rows0, j].mean()) ** 2))
            if scale > 0 and float(r @ r) > 1e-8 * scale:
                if (
                    not resid_cols
                    or np.linalg.matrix_rank(np.column_stack(resid_cols + [r]))
                    == len(resid_cols) + 1
                ):
                    resid_cols.append(r)
                    kept.append(j)
        if kept:
            Cs = sparse.csr_matrix(C[:, kept])
            Z = sparse.hstack([Z, Cs], format="csr")
    control_names = [cnames[j] for j in kept]
    Z0 = Z[rows0]
    gram = (Z0.T @ Z0).tocsc()
    try:
        lu = splu(gram)
    except RuntimeError as exc:  # pragma: no cover - guarded by the column rules
        raise DataInsufficient(
            f"{context}: the untreated unit/period design is singular.",
            recovery_hint=(
                "Check the panel for units or periods with no untreated cells."
            ),
        ) from exc
    coef = lu.solve(np.asarray(Z0.T @ Y[rows0]).ravel())
    y0_hat = np.asarray(Z @ coef).ravel()
    gamma = np.full(n, np.nan)
    gamma[target] = Y[target] - y0_hat[target]

    # Cohort and relative time, for the BJS blocks.  The forest only sees
    # D, so the cohort is the first *observed* treated period; in a panel
    # whose onset row is missing it can be later than the true adoption
    # date, which changes the variance blocks but never the estimate.
    first = np.full(n_u, n_t, dtype=np.int64)
    np.minimum.at(first, unit[treated], time[treated])
    ever = first < n_t
    cohort = np.where(ever[unit], first[unit], -1)
    rel = np.where(ever[unit], time - first[unit], 0)
    # Absorbing: once treated, treated in every later observed period.
    absorbing = bool(np.all(treated[ever[unit] & (time >= first[unit])]))
    if not absorbing:
        warnings.warn(
            f"{context}: the treatment switches off for some units. Untreated "
            "cells after a treated spell are used to fit Y(0), which assumes "
            "no carry-over effects.",
            AssumptionWarning,
            stacklevel=3,
        )

    clusters = getattr(forest, "_clusters", None)
    clusters = unit.copy() if clusters is None else np.asarray(clusters, dtype=np.int64)
    ow = getattr(forest, "_observation_weight", None)
    ow = np.ones(n) if ow is None or len(ow) != n else np.asarray(ow, dtype=float)
    design = ImputationDesign(
        n=n,
        treated=treated,
        target=target,
        Z=Z,
        gram_lu=lu,
        y=Y,
        y0_hat=y0_hat,
        gamma=gamma,
        unit=unit,
        time=time,
        cohort=cohort,
        rel=rel,
        clusters=clusters,
        obs_weight=ow,
        n_not_imputable=n_not,
        absorbing=absorbing,
        control_names=control_names,
        control_coef=np.asarray(coef[Z.shape[1] - len(kept) :], dtype=float),
    )
    cache[key] = design
    try:
        forest._fe_imputation = cache
    except AttributeError:  # pragma: no cover
        pass
    return design


def functional_weights(design: ImputationDesign, W1: np.ndarray) -> np.ndarray:
    """Weights ``V`` (n x k) with ``theta_k = V[:, k]' y``.

    ``W1`` holds the target weights on the rows (non-zero only on target
    cells).  Untreated rows receive minus the projection of the target
    through the untreated design, ``-Z0 (Z0'Z0)^-1 Z1' w``.
    """
    W1 = np.asarray(W1, dtype=float)
    if W1.ndim == 1:
        W1 = W1[:, None]
    rhs = np.asarray(design.Z.T @ W1)
    solved = design.gram_lu.solve(rhs)
    if solved.ndim == 1:
        solved = solved[:, None]
    projection = np.asarray(design.Z @ solved)
    V = W1.copy()
    untreated = ~design.treated
    V[untreated] -= projection[untreated]
    return V


def _block_centre(
    values: np.ndarray, v: np.ndarray, rows: np.ndarray, design: ImputationDesign
) -> np.ndarray:
    """``values`` minus their ``v^2``-weighted cohort x relative-time mean."""
    out = values.copy()
    frame = pd.DataFrame(
        {
            "g": design.cohort[rows],
            "e": design.rel[rows],
            "v2": v[rows] ** 2,
            "val": values[rows],
        }
    )
    frame["num"] = frame["v2"] * frame["val"]
    grouped = frame.groupby(["g", "e"], sort=False)[["num", "v2"]].transform("sum")
    den = grouped["v2"].to_numpy()
    mean = np.divide(
        grouped["num"].to_numpy(), den, out=np.zeros_like(den), where=den > 0
    )
    out[rows] = values[rows] - mean
    return out


def score_matrix(
    design: ImputationDesign,
    V: np.ndarray,
    tau_forest: Optional[np.ndarray],
    variance: str,
) -> np.ndarray:
    """Row scores ``v * eps`` for every functional (column of ``V``)."""
    if variance not in _VARIANCES:
        raise MethodIncompatibility(
            f"variance must be one of {_VARIANCES}, got {variance!r}.",
            recovery_hint="Use variance='forest' (default) or 'bjs'.",
        )
    rows = np.flatnonzero(design.target)
    base = np.zeros(design.n)
    untreated = ~design.treated
    base[untreated] = design.y[untreated] - design.y0_hat[untreated]
    treated_signal = design.gamma.copy()
    if variance == "forest":
        if tau_forest is None:  # pragma: no cover - callers always pass it
            raise MethodIncompatibility("variance='forest' needs forest predictions.")
        treated_signal[rows] = design.gamma[rows] - tau_forest[rows]
    S = np.zeros_like(V)
    for k in range(V.shape[1]):
        eps = base.copy()
        eps[rows] = _block_centre(treated_signal, V[:, k], rows, design)[rows]
        S[:, k] = V[:, k] * eps
    return S


def vcov_from_scores(
    S: np.ndarray,
    clusters: Optional[np.ndarray] = None,
    dyads: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """``sum`` of outer products of cluster (or dyadic) score sums.

    With ``dyads=(i, j)`` every pair of rows sharing a member is allowed to
    be correlated: node sums minus unordered-pair sums, as in
    :func:`statspai.dyadic_regression` [aronow2015cluster].  No small-sample
    factor is applied (the BJS convention).
    """
    if dyads is not None:
        ci, cj = dyads
        n_nodes = int(max(ci.max(), cj.max())) + 1
        node = np.zeros((n_nodes, S.shape[1]))
        np.add.at(node, ci, S)
        np.add.at(node, cj, S)
        lo, hi = np.minimum(ci, cj), np.maximum(ci, cj)
        pair = pd.factorize(pd.Series(lo.astype(np.int64) * n_nodes + hi))[0]
        pairs = np.zeros((int(pair.max()) + 1, S.shape[1]))
        np.add.at(pairs, pair, S)
        return np.asarray(node.T @ node - pairs.T @ pairs)
    if clusters is None:
        return np.asarray(S.T @ S)
    G = int(clusters.max()) + 1
    summed = np.zeros((G, S.shape[1]))
    np.add.at(summed, clusters, S)
    return np.asarray(summed.T @ summed)


# --------------------------------------------------------------------------- #
#  Shared input handling
# --------------------------------------------------------------------------- #


def _codes(values: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size != n:
        raise MethodIncompatibility(
            f"{name} must have one value per training row (length {n}).",
            recovery_hint=f"Pass {name} aligned with the rows used to fit the forest.",
            diagnostics={f"n_{name}": int(arr.size), "n": int(n)},
        )
    if pd.isna(pd.Series(arr)).any():
        raise MethodIncompatibility(
            f"{name} contains missing values.",
            recovery_hint=f"Fill or drop rows with a missing {name}.",
        )
    return np.asarray(pd.factorize(pd.Series(arr), sort=True)[0], dtype=np.int64)


def dyad_codes(members: Any, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate an (n, 2) member array; return node codes and labels."""
    arr = np.asarray(members, dtype=object)
    if arr.ndim != 2 or arr.shape != (n, 2):
        raise MethodIncompatibility(
            "members must be an (n, 2) array with the two members of each row "
            "(e.g. exporter and importer).",
            recovery_hint=(
                "Pass np.column_stack([df['i'], df['j']]) aligned with the " "fit rows."
            ),
            diagnostics={"shape": list(arr.shape), "n": int(n)},
        )
    if pd.isna(pd.DataFrame(arr)).to_numpy().any():
        raise MethodIncompatibility(
            "members contains missing values.",
            recovery_hint="Drop rows with a missing member.",
        )
    both = pd.Series(np.concatenate([arr[:, 0], arr[:, 1]]))
    codes, labels = pd.factorize(both, sort=True)
    ci, cj = codes[:n].astype(np.int64), codes[n:].astype(np.int64)
    if np.any(ci == cj):
        raise MethodIncompatibility(
            "members: a row pairs a member with itself.",
            recovery_hint=(
                "Drop self-pairs; dyadic variances need two distinct members."
            ),
        )
    return ci, cj, np.asarray(labels, dtype=object)


def _variance_setup(
    design_clusters: Optional[np.ndarray],
    cluster: Any,
    members: Optional[Tuple[np.ndarray, np.ndarray]],
    n: int,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]], str]:
    if isinstance(cluster, str) and cluster == "dyadic":
        if members is None:
            raise MethodIncompatibility(
                "cluster='dyadic' needs members= (the two members of each row).",
                recovery_hint="Pass members=np.column_stack([i, j]).",
            )
        n_nodes = int(max(members[0].max(), members[1].max())) + 1
        if n_nodes < 50:
            warnings.warn(
                f"cluster='dyadic' with {n_nodes} members: the dyadic-robust "
                "variance is justified as the number of members grows and is "
                "unreliable (possibly negative) with few of them. Report the "
                "pair-clustered standard errors alongside it.",
                AssumptionWarning,
                stacklevel=4,
            )
        return None, members, "dyadic-robust (rows sharing a member)"
    if cluster is None:
        return (
            design_clusters,
            None,
            "clustered by the forest's clusters (default: unit)",
        )
    return _codes(cluster, n, "cluster"), None, "clustered by the supplied cluster ids"


def _safe_se(V: np.ndarray, context: str = "") -> np.ndarray:
    """Square roots of the diagonal; NaN (with a warning) where it is <= 0.

    Cluster sums of squares are never negative, but the dyadic estimator
    subtracts pair sums and can be; a zero-width interval would be the
    worst possible way to report that.
    """
    diag = np.diag(V).astype(float)
    bad = ~(diag > 0)
    if np.any(bad):
        warnings.warn(
            f"{context or 'variance'}: {int(bad.sum())} variance estimate(s) "
            "are not positive and are reported as NaN. This happens with the "
            "dyadic-robust variance when a target loads on few members.",
            AssumptionWarning,
            stacklevel=4,
        )
    return np.where(bad, np.nan, np.sqrt(np.where(bad, 1.0, diag)))


def _table(
    est: np.ndarray, V: np.ndarray, names: Sequence[str], alpha: float
) -> pd.DataFrame:
    se = _safe_se(V)
    z = float(stats.norm.ppf(1 - alpha / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, est / se, np.nan)
    return pd.DataFrame(
        {
            "estimate": est,
            "se": se,
            "z": t,
            "p": 2 * stats.norm.sf(np.abs(t)),
            "ci_low": est - z * se,
            "ci_high": est + z * se,
        },
        index=list(names),
    )


def _oob_tau(forest: Any) -> np.ndarray:
    return np.asarray(forest._oob_tau, dtype=float)


def _require_target_oob(
    forest: Any, design: ImputationDesign, context: str, needed: bool = True
) -> None:
    if not needed:
        return
    tau = _oob_tau(forest)
    missing = int(np.sum(~np.isfinite(tau[design.target])))
    if missing:
        raise DataInsufficient(
            f"{context}: {missing} treated row(s) have no out-of-bag CATE "
            "prediction.",
            recovery_hint="Refit with more trees (n_estimators).",
            diagnostics={"n_rows_without_oob_prediction": missing},
        )


# --------------------------------------------------------------------------- #
#  Functionals
# --------------------------------------------------------------------------- #


def _estimate(
    forest: Any,
    design: ImputationDesign,
    W1: np.ndarray,
    variance: str,
    cluster: Any = None,
    members: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    V = functional_weights(design, W1)
    est = V.T @ design.y
    S = score_matrix(design, V, _oob_tau(forest), variance)
    clusters, dyads, label = _variance_setup(
        design.clusters, cluster, members, design.n
    )
    return np.asarray(est).ravel(), vcov_from_scores(S, clusters, dyads), label


def average_effect_fe(
    forest: Any,
    target_sample: str = "treated",
    alpha: float = 0.05,
    variance: str = "forest",
    cluster: Any = None,
    members: Any = None,
    covariates: Any = "none",
) -> Dict[str, Any]:
    """ATT of an FE forest by imputation, with its exact linear-weight SE."""
    context = "average_treatment_effect()"
    if target_sample != "treated":
        raise MethodIncompatibility(
            f"{context}: target_sample={target_sample!r} is not identified for a "
            "causal forest with fixed effects. Parallel trends identifies the "
            "effect on treated cells; effects on untreated cells are "
            "extrapolations of tau(x).",
            recovery_hint=(
                "Use target_sample='treated'. For counterfactual effects of "
                "units never treated, predict with cf.effect(X_new) and check "
                "sp.forest_support(cf, X_new) first."
            ),
            alternative_functions=["sp.forest_support"],
        )
    design = imputation_design(forest, context, covariates)
    _require_target_oob(forest, design, context, needed=variance == "forest")
    n = design.n
    mem = None if members is None else dyad_codes(members, n)[:2]
    w = np.where(design.target, design.obs_weight, 0.0)
    w = w / w.sum()
    est, V, label = _estimate(forest, design, w, variance, cluster, mem)
    tab = _table(est, V, ["ATT"], alpha)
    tau = _oob_tau(forest)
    z = float(stats.norm.ppf(1 - alpha / 2))
    se = float(tab["se"].iloc[0])
    return {
        "estimate": float(est[0]),
        "se": se,
        "ci_low": float(est[0]) - z * se,
        "ci_high": float(est[0]) + z * se,
        "pvalue": float(tab["p"].iloc[0]),
        "target_sample": "treated",
        "estimand": "ATT",
        "method": "imputation",
        "method_detail": (
            "imputation ATT (unit and period effects fitted on untreated "
            f"cells); SE from exact linear weights, {label}; treated "
            "residuals centred by "
            f"{'the OOB forest and ' if variance == 'forest' else ''}"
            "cohort x event-time blocks"
        ),
        "variance": variance,
        "effective_sample_size": float(w.sum() ** 2 / np.sum(w**2)),
        "n": int(n),
        "n_treated_cells": int(design.target.sum()),
        "n_not_imputable": design.n_not_imputable,
        "n_clusters": int(len(np.unique(design.clusters))),
        "alpha": float(alpha),
        "forest_plug_in": (
            float(np.sum(w[design.target] * tau[design.target]))
            if np.all(np.isfinite(tau[design.target]))
            else float("nan")
        ),
        "imputation_covariates": list(design.control_names),
        "weighting": (
            "equal per treated cell"
            if np.allclose(design.obs_weight, design.obs_weight[0])
            else "forest observation weights (equalize_cluster_weights)"
        ),
        "cate_source": "imputation_scores",
    }


def best_linear_projection_fe(
    forest: Any,
    A: Optional[np.ndarray],
    names: Sequence[str],
    alpha: float = 0.05,
    variance: str = "forest",
    cluster: Any = None,
    members: Any = None,
    covariates: Any = "none",
) -> pd.DataFrame:
    """Regression of imputation scores on ``(1, A)`` over treated cells."""
    context = "best_linear_projection()"
    design = imputation_design(forest, context, covariates)
    _require_target_oob(forest, design, context, needed=variance == "forest")
    rows = np.flatnonzero(design.target)
    A_all = np.asarray(forest._X_original if A is None else A, dtype=float)
    if A_all.ndim == 1:
        A_all = A_all[:, None]
    if A_all.shape[0] != design.n:
        raise MethodIncompatibility(
            f"{context}: A must have one row per training row.",
            recovery_hint="Pass covariates aligned with the fit rows.",
        )
    D = np.column_stack([np.ones(rows.size), A_all[rows]])
    ow = design.obs_weight[rows]
    DtWD = D.T @ (D * ow[:, None])
    if np.linalg.matrix_rank(DtWD) < D.shape[1]:
        raise MethodIncompatibility(
            f"{context}: the covariates are collinear on the treated cells.",
            recovery_hint=(
                "Drop covariates that are constant or collinear among treated " "rows."
            ),
        )
    W1 = np.zeros((design.n, D.shape[1]))
    W1[rows] = (D * ow[:, None]) @ np.linalg.inv(DtWD)
    mem = None if members is None else dyad_codes(members, design.n)[:2]
    est, V, label = _estimate(forest, design, W1, variance, cluster, mem)
    tab = _table(est, V, ["Intercept", *names], alpha)
    out = pd.DataFrame(
        {
            "coef": tab["estimate"],
            "se": tab["se"],
            "t": tab["z"],
            "p": tab["p"],
            "ci_lower": tab["ci_low"],
            "ci_upper": tab["ci_high"],
        }
    )
    out.attrs["method"] = (
        f"OLS of imputation scores on covariates over {rows.size} treated "
        f"cells; {label}"
    )
    out.attrs["vcov"] = attrs_array(V)
    return out


def calibration_fe(
    forest: Any,
    alpha: float = 0.05,
    variance: str = "forest",
    cluster: Any = None,
    members: Any = None,
    covariates: Any = "none",
) -> pd.DataFrame:
    """BLP calibration of the OOB forest prediction against imputation scores.

    Regresses ``Gamma`` on ``mean(tau_hat)`` and ``tau_hat - mean(tau_hat)``
    (no intercept) over treated cells.  ``Gamma`` is unbiased for each
    cell's effect, so ``beta_differential`` is the best-linear-predictor
    slope of the true effect on the forest prediction: a one-sided test
    of ``beta_differential <= 0`` is a heterogeneity test, and the slope
    itself rescales the predictions around their mean.
    """
    context = "calibration_test(method='imputation')"
    design = imputation_design(forest, context, covariates)
    _require_target_oob(forest, design, context)
    rows = np.flatnonzero(design.target)
    tau = _oob_tau(forest)[rows]
    ow = design.obs_weight[rows]
    tau_bar = float(np.sum(ow * tau) / np.sum(ow))
    D = np.column_stack([np.full(rows.size, tau_bar), tau - tau_bar])
    DtWD = D.T @ (D * ow[:, None])
    if abs(tau_bar) < 1e-12 or np.linalg.matrix_rank(DtWD) < 2:
        raise NumericalInstability(
            f"{context}: the forest predictions have a zero mean or no "
            "variation on the treated cells, so the calibration regression "
            "is not identified.",
            recovery_hint=(
                "Use average_treatment_effect() for the level; the forest "
                "found no ranking to test."
            ),
        )
    W1 = np.zeros((design.n, 2))
    W1[rows] = (D * ow[:, None]) @ np.linalg.inv(DtWD)
    mem = None if members is None else dyad_codes(members, design.n)[:2]
    est, V, label = _estimate(forest, design, W1, variance, cluster, mem)
    names = ["mean_forest_prediction", "differential_forest_prediction"]
    se = _safe_se(V, context)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, est / se, np.nan)
    p_one = stats.norm.sf(t)
    z = float(stats.norm.ppf(1 - alpha / 2))
    # Same columns and reporting rule as grf's test_calibration (t against
    # 0, one-sided p), normal reference because the SE is a large-sample
    # linear-weight variance without a degrees-of-freedom correction.
    out = pd.DataFrame(
        {
            "coef": est,
            "se": se,
            "t": t,
            "p": p_one,
            "ci_low": est - z * se,
            "ci_high": est + z * se,
            "t_vs_zero": t,
            "p_one_sided": p_one,
        },
        index=names,
    )
    out.attrs["method"] = (
        "BLP calibration of out-of-bag predictions against imputation scores "
        f"on {rows.size} treated cells; {label}"
    )
    out.attrs["tau_bar"] = tau_bar
    return out


# --------------------------------------------------------------------------- #
#  Group effects (shared by FE and pooled forests)
# --------------------------------------------------------------------------- #


def _group_indicator(
    n: int,
    rows_mask: np.ndarray,
    by: Any,
    members: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    tau: np.ndarray,
    n_groups: int,
) -> Tuple[np.ndarray, List[Any], str]:
    """(n x G) 0/1 membership over eligible rows, labels and grouping kind."""
    if isinstance(by, str) and by == "cate_quantile":
        rows = np.flatnonzero(rows_mask)
        if n_groups < 2:
            raise MethodIncompatibility(
                "n_groups must be at least 2 for by='cate_quantile'.",
                recovery_hint="Use n_groups=4 (quartiles) or 5.",
            )
        ranks = pd.Series(tau[rows]).rank(method="first").to_numpy()
        q = np.minimum((ranks - 1) * n_groups // rows.size, n_groups - 1).astype(int)
        M = np.zeros((n, n_groups))
        M[rows, q] = 1.0
        return (
            M,
            [f"Q{g + 1}" for g in range(n_groups)],
            "quantile of the OOB forest prediction",
        )
    if by is None and members is not None:
        ci, cj, labels = members
        M = np.zeros((n, labels.size))
        M[np.arange(n), ci] = 1.0
        M[np.arange(n), cj] = 1.0
        M[~rows_mask] = 0.0
        return M, list(labels), "membership (a row counts for both of its members)"
    if by is None:
        M = np.zeros((n, 1))
        M[rows_mask, 0] = 1.0
        return M, ["all"], "all eligible rows"
    arr = np.asarray(by, dtype=object)
    if arr.ndim != 1 or arr.size != n:
        raise MethodIncompatibility(
            "by must be 'cate_quantile' or one label per training row.",
            recovery_hint="Pass df['group'].to_numpy() aligned with the fit rows.",
            diagnostics={"n_by": int(arr.size), "n": int(n)},
        )
    if pd.isna(pd.Series(arr)).any():
        raise MethodIncompatibility(
            "by contains missing labels.",
            recovery_hint="Fill or drop rows with a missing group label.",
        )
    codes, labels = pd.factorize(pd.Series(arr), sort=True)
    M = np.zeros((n, len(labels)))
    M[np.arange(n), codes] = 1.0
    M[~rows_mask] = 0.0
    return M, list(labels), "supplied labels"


def _group_tests(
    est: np.ndarray, V: np.ndarray, labels: Sequence[Any], alpha: float
) -> Dict[str, Any]:
    G = est.size
    out: Dict[str, Any] = {}
    if G >= 2:
        R = np.zeros((G - 1, G))
        R[:, 0] = -1.0
        R[np.arange(G - 1), np.arange(1, G)] = 1.0
        d = R @ est
        RVR = R @ V @ R.T
        try:
            stat = float(d @ np.linalg.solve(RVR, d))
            out["equality_wald"] = stat
            out["equality_df"] = G - 1
            out["equality_p"] = float(stats.chi2.sf(stat, G - 1))
        except np.linalg.LinAlgError:
            out["equality_wald"] = np.nan
            out["equality_df"] = G - 1
            out["equality_p"] = np.nan
        diff = float(est[-1] - est[0])
        var_d = float(V[-1, -1] + V[0, 0] - 2 * V[0, -1])
        se = float(np.sqrt(var_d)) if var_d > 0 else float("nan")
        z = float(stats.norm.ppf(1 - alpha / 2))
        out["last_minus_first"] = {
            "groups": (labels[-1], labels[0]),
            "estimate": diff,
            "se": se,
            "ci_low": diff - z * se,
            "ci_high": diff + z * se,
            "p": float(2 * stats.norm.sf(abs(diff) / se)) if se > 0 else np.nan,
        }
    return out


def group_effects(
    forest: Any,
    by: Any = None,
    *,
    members: Any = None,
    n_groups: int = 4,
    cluster: Any = None,
    variance: str = "forest",
    alpha: float = 0.05,
    scale: str = "level",
    min_rows: int = 1,
    covariates: Any = "none",
) -> pd.DataFrame:
    """Average effects by group with valid standard errors.

    FE forests: means of imputation scores over each group's treated cells
    (group ATTs).  Pooled forests: means of AIPW scores over each group's
    rows (group ATEs).  See :func:`statspai.forest_group_effects`.
    """
    from . import _grf_inference as gi

    context = "forest_group_effects()"
    if not gi.is_grf_forest(forest):
        raise MethodIncompatibility(
            f"{context} needs a forest fitted with the GRF engine "
            "(split_rule='grf', the default).",
            recovery_hint=(
                "Refit with sp.causal_forest(...) using the default split_rule."
            ),
        )
    if scale not in ("level", "percent"):
        raise MethodIncompatibility(
            f"{context}: scale must be 'level' or 'percent'.",
            recovery_hint="Use scale='percent' for log outcomes (100 * (exp(x) - 1)).",
        )
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"{context}: alpha must be in (0, 1).", recovery_hint="Use alpha=0.05."
        )
    n = int(len(forest._Y_original))
    mem = None if members is None else dyad_codes(members, n)
    tau = _oob_tau(forest)
    fe = gi.is_fe_forest(forest)
    if fe:
        design = imputation_design(forest, context, covariates)
        _require_target_oob(
            forest,
            design,
            context,
            needed=variance == "forest"
            or (isinstance(by, str) and by == "cate_quantile"),
        )
        eligible = design.target
        weights = design.obs_weight
    else:
        gi.require_finite_oob(forest, context)
        eligible = np.ones(n, dtype=bool)
        weights = gi._weights(forest, n) * n
    M, labels, kind = _group_indicator(n, eligible, by, mem, tau, int(n_groups))
    counts = M.sum(axis=0)
    keep = counts >= max(int(min_rows), 1)
    if not keep.any():
        raise DataInsufficient(
            f"{context}: no group has {max(int(min_rows), 1)} or more eligible rows.",
            recovery_hint="Lower min_rows or use a coarser grouping.",
        )
    M = M[:, keep]
    labels = [lab for lab, k in zip(labels, keep) if k]
    Wg = M * weights[:, None]
    Wg = Wg / Wg.sum(axis=0, keepdims=True)
    dyads = None if mem is None else (mem[0], mem[1])
    if fe:
        est, V, label = _estimate(forest, design, Wg, variance, cluster, dyads)
        units = design.unit
        method = (
            f"imputation-score group ATTs ({kind}); {label}; treated residuals "
            f"centred by {'the OOB forest and ' if variance == 'forest' else ''}"
            "cohort x event-time blocks"
        )
    else:
        scores, _, _, _ = gi.dr_scores(forest, tau, clip=0.0)
        est = Wg.T @ scores
        S = Wg * (scores[:, None] - est[None, :])
        clusters, dyads_, label = _variance_setup(
            (
                gi._cluster_codes(forest, n)
                if getattr(forest, "_clusters", None) is not None
                else None
            ),
            cluster,
            dyads,
            n,
        )
        V = vcov_from_scores(S, clusters, dyads_)
        if dyads_ is None:
            # grf's G / (G - 1) factor, so that the single all-rows group
            # reproduces average_treatment_effect()'s standard error.
            g_count = n if clusters is None else int(np.unique(clusters).size)
            V = V * g_count / max(g_count - 1, 1)
        units = gi._cluster_codes(forest, n)
        method = (
            f"AIPW-score group ATEs ({kind}); {label or 'heteroskedasticity-robust'}"
        )
    tab = _table(np.asarray(est).ravel(), V, labels, alpha)
    tab.insert(0, "n_rows", M.sum(axis=0).astype(int))
    tab.insert(
        1,
        "n_units",
        [int(np.unique(units[M[:, g] > 0]).size) for g in range(M.shape[1])],
    )
    tab["forest_mean"] = [
        (
            float(np.sum(Wg[M[:, g] > 0, g] * tau[M[:, g] > 0]))
            if np.all(np.isfinite(tau[M[:, g] > 0]))
            else float("nan")
        )
        for g in range(M.shape[1])
    ]
    if scale == "percent":
        for col in ("estimate", "ci_low", "ci_high", "forest_mean"):
            tab[f"{col}_pct"] = 100.0 * np.expm1(tab[col].to_numpy())
    tab.index.name = "group"
    tab.attrs["method"] = method
    tab.attrs["vcov"] = attrs_array(V)
    tab.attrs["tests"] = _group_tests(tab["estimate"].to_numpy(), V, labels, alpha)
    tab.attrs["estimand"] = "ATT (treated cells)" if fe else "ATE (all rows)"
    if fe:
        tab.attrs["imputation_covariates"] = list(design.control_names)
        if design.n_not_imputable:
            tab.attrs["n_not_imputable"] = design.n_not_imputable
    return tab


def rate_fe(
    forest: Any,
    target: str = "AUTOC",
    *,
    priorities: Any = None,
    q_grid: int = 100,
    alpha: float = 0.05,
    variance: str = "bjs",
    cluster: Any = None,
    members: Any = None,
    covariates: Any = "none",
    se_method: str = "imputation",
) -> Dict[str, Any]:
    r"""RATE of a fixed-effects forest, on the treated cells it identifies.

    The doubly-robust score behind :func:`statspai.rate` needs a propensity,
    which a within-unit design does not have; the imputation score
    :math:`\Gamma_{it} = Y_{it} - \hat\alpha_i - \hat\gamma_t` does the same
    job on treated cells, so the whole curve is read on that population:

    .. math::
        \mathrm{TOC}(q) = \mathrm{ATT}\bigl(\text{top } q
        \text{ by } S\bigr) - \mathrm{ATT},

    "how much larger is the effect among the cells this rule would have
    prioritised". Effects on untreated cells are not identified, so this is
    a retrospective targeting curve, not the population RATE that a
    randomised design gives.

    ``se_method="imputation"`` (default) composes the two linear maps --
    the rank weights of :func:`statspai.forest.forest_inference.
    rate_rank_weights` and the imputation weights of
    :func:`functional_weights` -- so the RATE is one more linear functional
    of ``y``, with the same exact, cluster- (or dyad-) robust variance as
    the ATT.  It conditions on the prioritisation, which is what you want
    when ``priorities`` were fitted elsewhere.  ``se_method="influence"``
    instead treats the imputation scores as data and applies the
    rank-corrected influence function of :func:`statspai.rate`: it carries
    the cost of estimating the ranking but not of estimating the fixed
    effects.  See :func:`statspai.rate` for which to report.
    """
    from .forest_inference import (
        _rate_influence_se,
        rate_from_scores,
        rate_rank_weights,
    )

    context = "rate()"
    key = str(target).upper().strip()
    if key not in ("AUTOC", "QINI"):
        raise MethodIncompatibility(
            f"{context}: target must be 'AUTOC' or 'QINI'.",
            recovery_hint="Use a supported RATE summary target.",
            diagnostics={"target": target},
        )
    if se_method not in ("imputation", "influence"):
        raise MethodIncompatibility(
            f"{context}: se_method must be 'imputation' or 'influence' for a "
            "forest with fixed effects.",
            recovery_hint=(
                "grf's half-sample bootstrap resamples units, which would have "
                "to refit the untreated two-way model in every draw; the exact "
                "linear-weight variance ('imputation') supersedes it here."
            ),
            diagnostics={"se_method": se_method},
        )
    design = imputation_design(forest, context, covariates)
    _require_target_oob(forest, design, context, needed=variance == "forest")
    rows = np.flatnonzero(design.target)
    if rows.size < 2:
        raise DataInsufficient(
            f"{context}: need at least two imputable treated cells.",
            recovery_hint="RATE ranks treated cells; this panel has too few.",
            diagnostics={"n_treated_cells": int(rows.size)},
        )
    tau = _oob_tau(forest)
    if priorities is None:
        prio = tau[rows]
        priority_source = "out_of_bag"
        warnings.warn(
            "rate(): ranking the treated cells by this same forest's own "
            "out-of-bag predictions does not give a valid test. Every "
            "imputation score carries -gamma_hat_t, estimated from the very "
            "periods the forest was trained on, so the ranking and the scores "
            "are correlated even though no unit predicts itself. Measured on "
            "a design with no heterogeneity at all (200 replications, N = 150 "
            "units, T = 8), AUTOC averaged -0.025 instead of 0 and a nominal "
            "5% test rejected 17.5% of the time (QINI 13.5%); sp.rate_split(), "
            "which fits the ranking and the scores on disjoint units, averaged "
            "+0.0008 and rejected 7.5% (QINI 4.0%). Report this as a "
            "diagnostic and sp.rate_split() as the test.",
            AssumptionWarning,
            stacklevel=3,
        )
    else:
        supplied = np.asarray(priorities, dtype=float).ravel()
        if supplied.size == design.n:
            prio = supplied[rows]
        elif supplied.size == rows.size:
            prio = supplied
        else:
            raise MethodIncompatibility(
                f"{context}: priorities must have one value per row "
                f"({design.n}) or per treated cell ({rows.size}), got "
                f"{supplied.size}.",
                recovery_hint="Pass cf.effect(X) or a vector over treated cells.",
            )
        priority_source = "supplied"
    if not np.isfinite(prio).all():
        raise DataInsufficient(
            f"{context}: priorities contain non-finite values.",
            recovery_hint="Drop or impute them before ranking.",
        )

    gamma = design.gamma[rows]
    q_targets = np.linspace(1.0 / int(q_grid), 1.0, int(q_grid))
    q_targets[-1] = 1.0
    core = rate_from_scores(gamma, prio, key, q_targets)

    a = rate_rank_weights(prio, key)
    W1 = np.zeros(design.n)
    W1[rows] = a
    mem = None if members is None else dyad_codes(members, design.n)[:2]
    est, V, label = _estimate(forest, design, W1, variance, cluster, mem)
    estimate = float(est[0])
    if se_method == "imputation":
        se = float(_safe_se(V, context)[0])
        detail = (
            f"exact linear weights (rank weights o imputation weights), {label}; "
            "treated residuals centred by "
            f"{'the OOB forest and ' if variance == 'forest' else ''}"
            "cohort x event-time blocks; conditional on the prioritisation"
        )
    else:
        codes, dyads, label = _variance_setup(design.clusters, cluster, mem, design.n)
        if dyads is not None:
            raise MethodIncompatibility(
                f"{context}: se_method='influence' has no dyadic form.",
                recovery_hint="Use the default se_method='imputation'.",
            )
        on_target = None
        if codes is not None:
            on_target = pd.factorize(codes[rows])[0]
        se = float(_rate_influence_se(gamma, prio, key, clusters=on_target))
        detail = (
            "rank-corrected influence function of the imputation scores, "
            f"{label}; carries the cost of estimating the ranking but treats "
            "the fixed effects as known"
        )

    z = float(stats.norm.ppf(1 - float(alpha) / 2))
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target": key,
        "toc_curve": np.column_stack([core["toc_q"], core["toc"]]),
        "n": int(rows.size),
        "method": f"imputation RATE ({se_method} SE)",
        "method_detail": detail,
        "priority_source": priority_source,
        "n_clusters": int(len(np.unique(design.clusters))),
        "estimand": (
            "TOC over treated cells: ATT(top q by priority) - ATT "
            "(retrospective targeting; effects on untreated cells are not "
            "identified)"
        ),
        "variance": variance,
        "se_method": se_method,
        "n_treated_cells": int(design.target.sum()),
        "n_not_imputable": design.n_not_imputable,
        "imputation_covariates": list(design.control_names),
        "alpha": float(alpha),
        "cate_source": "imputation_scores",
    }
