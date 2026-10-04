"""Two-stage DiD with a user-specified first and second stage.

:func:`statspai.gardner_did` in its original form takes the first-treatment
period and fits unit and period effects. R ``did2s`` and Stata ``did2s`` are
more general: the untreated rows are whatever a treatment dummy marks, the
first stage is any set of fixed effects and covariates, and the second stage
any set of regressors. This module is that form. The algebra, including the
Butts-Gardner covariance, is the one in :mod:`statspai.did.gardner_2s`; only
the two designs are built differently.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats as sp_stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["did2s_general"]

_FACTOR = re.compile(r"^i(?:b(\d+))?\.([A-Za-z_]\w*)$")


def _cell_codes(df: pd.DataFrame, spec: str) -> Tuple[np.ndarray, int]:
    """Integer codes of one fixed effect: a column, or ``a#b`` for the cell."""
    parts = [p.strip() for p in str(spec).split("#") if p.strip()]
    if not parts:
        raise MethodIncompatibility(f"gardner_did: empty fixed effect {spec!r}.")
    missing = [p for p in parts if p not in df.columns]
    if missing:
        raise ValueError(f"Column '{missing[0]}' not found in data")
    codes = df.groupby(parts, sort=True, dropna=False).ngroup().to_numpy()
    return codes, int(codes.max()) + 1 if len(codes) else 0


def _stage1_design(
    df: pd.DataFrame, fe: Sequence[str], controls: Sequence[str]
) -> sparse.csr_matrix:
    """Intercept, one block of dummies per fixed effect (first level dropped)
    and the covariates. Levels are coded over every row, so a level seen only
    among treated rows has a column: it is zero in the untreated fit and gets
    a zero coefficient, which is what Stata's omitted dummy contributes."""
    n = len(df)
    rows = np.arange(n)
    parts: List[sparse.spmatrix] = [sparse.csr_matrix(np.ones((n, 1)))]
    for spec in fe:
        codes, n_levels = _cell_codes(df, spec)
        keep = codes > 0
        parts.append(
            sparse.csr_matrix(
                (np.ones(int(keep.sum())), (rows[keep], codes[keep] - 1)),
                shape=(n, max(n_levels - 1, 0)),
            )
        )
    if controls:
        parts.append(sparse.csr_matrix(df[list(controls)].to_numpy(dtype=float)))
    return sparse.hstack(parts, format="csr")


def _level_label(level: Any, column: str) -> str:
    value = float(level)
    text = str(int(value)) if value.is_integer() else repr(value)
    return f"{text}.{column}"


def _stage2_design(
    df: pd.DataFrame, terms: Sequence[str]
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Columns of the second stage: a numeric column as it is, ``i.x`` /
    ``ib<k>.x`` as one indicator per level of ``x`` other than the base
    (the smallest level, or ``k``). Returns the design, its column names and
    the names of the base levels left out."""
    cols: List[np.ndarray] = []
    names: List[str] = []
    bases: List[str] = []
    for term in terms:
        m = _FACTOR.match(str(term).strip())
        if m is None:
            name = str(term).strip()
            if name not in df.columns:
                raise ValueError(f"Column '{name}' not found in data")
            cols.append(pd.to_numeric(df[name], errors="coerce").to_numpy(float))
            names.append(name)
            continue
        column = m.group(2)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in data")
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(float)
        levels = np.unique(values[np.isfinite(values)])
        base = float(m.group(1)) if m.group(1) is not None else float(levels[0])
        if base not in levels:
            raise MethodIncompatibility(
                f"gardner_did: base level {m.group(1)} of {term!r} does not "
                f"occur in column '{column}'.",
                recovery_hint="Name a level that occurs, or write "
                f"'i.{column}' for the smallest one.",
            )
        bases.append(_level_label(base, column))
        for level in levels:
            if level == base:
                continue
            cols.append((values == level).astype(float))
            names.append(_level_label(level, column))
    if not cols:
        raise MethodIncompatibility("gardner_did: second_stage has no regressor.")
    return np.column_stack(cols), names, bases


def did2s_general(
    data: pd.DataFrame,
    y: str,
    *,
    treat: str,
    fe: Sequence[str],
    controls: Sequence[str],
    second_stage: Optional[Sequence[str]],
    cluster: str,
    weights: Optional[str],
    alpha: float,
    vce: str,
) -> CausalResult:
    """Two-stage DiD from a treatment dummy and explicit stages."""
    from .gardner_2s import _did2s_vcov, _sparse_normal_solve

    fe = [str(f) for f in fe]
    controls = [str(c) for c in controls]
    terms = None if second_stage is None else [str(t) for t in second_stage]
    fe_columns = [p.strip() for spec in fe for p in spec.split("#") if p.strip()]
    stage2_columns = (
        []
        if terms is None
        else [(_FACTOR.match(t.strip()) or [None, None, t.strip()])[2] for t in terms]
    )
    needed = [y, treat, cluster, *fe_columns, *controls, *stage2_columns]
    if weights is not None:
        needed.append(weights)
    for col in needed:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")
    df = data.copy()
    df[y] = pd.to_numeric(df[y], errors="coerce")
    df = df.dropna(subset=list(dict.fromkeys(needed))).reset_index(drop=True)
    n = len(df)

    d = pd.to_numeric(df[treat], errors="coerce").to_numpy(float)
    if not np.all(np.isin(d, (0.0, 1.0))):
        raise MethodIncompatibility(
            f"gardner_did: treat='{treat}' must be a 0/1 indicator of the "
            "treated observations.",
            recovery_hint="Build it as (period >= first treatment period) "
            "for the units that are ever treated.",
            diagnostics={"values": np.unique(d)[:10].tolist()},
        )
    treated = d == 1.0
    untreated = ~treated
    if untreated.sum() < 10:
        raise ValueError("Not enough untreated observations for Stage 1 (<10).")
    if not treated.any():
        raise DataInsufficient(
            "gardner_did: no treated observations, the effect is undefined."
        )

    if weights is not None:
        w = pd.to_numeric(df[weights], errors="coerce").to_numpy(float)
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise MethodIncompatibility(
                f"weights column '{weights}' must be finite and strictly "
                "positive; drop zero-weight rows before calling gardner_did",
                diagnostics={"weights": weights},
            )
    else:
        w = np.ones(n, dtype=float)
    sw = np.sqrt(w)

    # ── Stage 1 on the untreated rows ─────────────────────────────── #
    A_full = _stage1_design(df, fe, controls)
    A_full_w = sparse.csr_matrix(sparse.diags(sw) @ A_full)
    A_un_w = A_full_w[untreated]
    y_all = df[y].to_numpy(dtype=float)
    coefs = _sparse_normal_solve(A_un_w, A_un_w.T @ (y_all * sw)[untreated])
    y_tilde = y_all - A_full @ coefs
    e1_un = y_tilde[untreated]

    # ── Stage 2: no intercept, as did2s ───────────────────────────── #
    names: List[str]
    bases: List[str]
    if terms is None:
        X2, names, bases = d.reshape(-1, 1), [treat], []
    else:
        X2, names, bases = _stage2_design(df, terms)
    present = np.abs(X2).sum(axis=0) > 0
    omitted = [nm for nm, ok in zip(names, present) if not ok]
    X2, kept = X2[:, present], [nm for nm, ok in zip(names, present) if ok]
    if X2.shape[1] == 0:
        raise DataInsufficient("gardner_did: every second-stage column is zero.")
    X2_w = X2 * sw[:, None]
    if np.linalg.matrix_rank(X2_w) < X2_w.shape[1]:
        raise MethodIncompatibility(
            "gardner_did: the second-stage regressors are collinear, so "
            "their coefficients are not identified.",
            recovery_hint="Drop one of the collinear columns; the second "
            "stage has no intercept, so a full set of indicators is fine "
            "but an indicator plus its complement and a constant is not.",
            diagnostics={"second_stage": kept},
        )
    coef2 = np.linalg.lstsq(X2_w, y_tilde * sw, rcond=None)[0]
    e2 = y_tilde - X2 @ coef2

    cl = df[cluster].to_numpy()
    V: Optional[np.ndarray] = None
    se = np.full(len(kept), np.nan)
    if vce == "analytic":
        V = _did2s_vcov(
            A_un_w, e1_un * sw[untreated], A_full_w, X2_w, e2 * sw, untreated, cl
        )
        se = np.sqrt(np.clip(np.diag(V), 0.0, None))

    z = float(sp_stats.norm.ppf(1 - alpha / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        pvalues = 2 * sp_stats.norm.sf(np.abs(coef2 / se))
    detail = pd.DataFrame(
        {
            "term": kept,
            "estimate": coef2,
            "se": se,
            "ci_lower": coef2 - z * se,
            "ci_upper": coef2 + z * se,
            "pvalue": pvalues,
            "n_obs": (X2 != 0).sum(axis=0).astype(int),
        }
    )
    single = len(kept) == 1
    estimate = float(coef2[0]) if single else float("nan")
    est_se = float(se[0]) if single else float("nan")
    model_info: Dict[str, Any] = {
        "method": "Gardner 2022 two-stage DID",
        "vce": vce,
        "se_convention": (
            "did2s corrected two-stage clustered variance (Gardner 2022): "
            "Stage-1 estimation error propagated, no small-sample factor; "
            "matches R did2s::did2s and Stata did2s"
            if vce == "analytic"
            else "no inference"
        ),
        "weights": weights,
        "covariates": "set" if controls else "none",
        "first_stage_fe": fe,
        "second_stage": kept,
        "second_stage_base": bases,
        "second_stage_omitted": omitted,
        "treat": treat,
        "n_obs": n,
        "n_clusters": int(pd.unique(cl).size),
        "alpha": alpha,
        "stage1_n": int(untreated.sum()),
        "vcov": V,
        "cell_labels": kept if V is not None else None,
        "event_study": None,
        "citation": (
            "Gardner, J. (2022). Two-stage differences in differences. "
            "arXiv:2207.05943. Butts & Gardner (2022), The R Journal."
        ),
    }
    return CausalResult(
        method="Gardner 2022 two-stage DID (did2s)",
        estimand=(
            "ATT" if terms is None else "second-stage coefficients (see .detail)"
        ),
        estimate=estimate,
        se=est_se,
        pvalue=float(pvalues[0]) if single else float("nan"),
        ci=(
            (estimate - z * est_se, estimate + z * est_se)
            if single
            else (float("nan"), float("nan"))
        ),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
    )
