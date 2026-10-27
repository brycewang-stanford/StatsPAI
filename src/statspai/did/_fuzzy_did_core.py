"""Engine of :func:`statspai.fuzzy_did` (Wald-DID / Wald-TC / Wald-CIC).

Kept apart from the public function: the design (rows coded by period,
group-change codes and treatment category), the comparisons it defines,
the estimators as functions of the rows, and the influence functions of
the Wald-DID / Wald-TC. See :mod:`statspai.did.fuzzy_did` for the
definitions and the conventions they follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
from scipy import stats

from ..exceptions import (
    ConvergenceFailure,
    IdentificationFailure,
    MethodIncompatibility,
)

_MODELS = ("ols", "logit", "probit")

# ----------------------------------------------------------------------
# Design
# ----------------------------------------------------------------------


@dataclass
class _Design:
    y: np.ndarray
    d: np.ndarray
    cat: np.ndarray  # treatment category used by TC / CIC (newcateg)
    tcode: np.ndarray  # period index 0..P-1
    gt: np.ndarray  # change into this period: -1/0/1, 9 = missing
    gf: np.ndarray  # change out of this period: -1/0/1, 9 = missing
    x: Optional[np.ndarray]  # covariates (no intercept) or None
    cluster: Optional[np.ndarray]
    periods: List[Any]
    n_cat: int


@dataclass
class _Comparison:
    pair: int  # index a of the pair (a, a + 1)
    sign: int


def _comparisons(des: _Design, rows: np.ndarray) -> List[_Comparison]:
    out = []
    t = des.tcode[rows]
    for a in range(len(des.periods) - 1):
        for s in (1, -1):
            pre = np.any((t == a) & (des.gf[rows] == s))
            post = np.any((t == a + 1) & (des.gt[rows] == s))
            if pre and post:
                out.append(_Comparison(a, s))
            elif pre or post:
                raise IdentificationFailure(
                    "fuzzy_did: a switching group is coded in only one period "
                    f"of the pair ({des.periods[a]}, {des.periods[a + 1]}): "
                    f"group_change_next == {s} before but no group_change == "
                    f"{s} after, or the reverse.",
                    recovery_hint="Code each group's change on both periods: "
                    "group_change on the later period's rows and "
                    "group_change_next on the earlier period's rows.",
                    diagnostics={"pair": a, "direction": s},
                )
    return out


# ----------------------------------------------------------------------
# Wald-DID / Wald-TC without covariates: functions of atom sums
# ----------------------------------------------------------------------


class _Atoms:
    """Rows grouped by (period, group_change, group_change_next, category).

    Without covariates the Wald-DID and Wald-TC depend on the data only
    through the count, sum of Y and sum of D in each atom, so the
    estimators are smooth functions of ``3 x n_atoms`` numbers and their
    influence functions follow from the gradient (complex step: exact to
    machine precision).
    """

    def __init__(self, des: _Design, rows: np.ndarray):
        # codes -1 / 0 / 1 / 9 (missing) -> 0 / 1 / 2 / 3; one integer per
        # row (a 1-D unique is ~100x faster than np.unique(axis=0))
        gt = np.where(des.gt[rows] == 9, 3, des.gt[rows] + 1).astype(np.int64)
        gf = np.where(des.gf[rows] == 9, 3, des.gf[rows] + 1).astype(np.int64)
        nc = int(des.n_cat)
        code = ((des.tcode[rows].astype(np.int64) * 4 + gt) * 4 + gf) * nc
        code = code + des.cat[rows]
        uniq, inv = np.unique(code, return_inverse=True)
        self.key = np.column_stack(
            [uniq // (16 * nc), (uniq // (4 * nc)) % 4, (uniq // nc) % 4, uniq % nc]
        )
        self.atom_of_row = inv.reshape(-1)
        self.rows = rows

    def stats(
        self, des: _Design, w: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        k = len(self.key)
        rows = self.rows
        ww = np.ones(len(rows)) if w is None else w
        n = np.bincount(self.atom_of_row, weights=ww, minlength=k)
        sy = np.bincount(self.atom_of_row, weights=ww * des.y[rows], minlength=k)
        sd = np.bincount(self.atom_of_row, weights=ww * des.d[rows], minlength=k)
        return n, sy, sd

    def mask(
        self, period: int, col: int, value: int, cat: Optional[int] = None
    ) -> np.ndarray:
        m = (self.key[:, 0] == period) & (self.key[:, col] == value + 1)
        if cat is not None:
            m &= self.key[:, 3] == cat
        return cast(np.ndarray, m)


def _did_tc_from_stats(
    atoms: _Atoms,
    comps: Sequence[_Comparison],
    n: np.ndarray,
    sy: np.ndarray,
    sd: np.ndarray,
    want: Sequence[str],
    numerator: bool,
    n_cat: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Wald-DID / Wald-TC (or numerators) from atom sums; complex-safe."""
    num = {k: 0.0 for k in want}
    den = {k: 0.0 for k in want}
    parts = []
    for c in comps:
        a, s = c.pair, c.sign
        pre = atoms.mask(a, 2, s)
        post = atoms.mask(a + 1, 1, s)
        cpre = atoms.mask(a, 2, 0)
        cpost = atoms.mask(a + 1, 1, 0)

        def mean(m: np.ndarray, v: np.ndarray) -> Any:
            return v[m].sum() / n[m].sum()

        dy = mean(post, sy) - mean(pre, sy)
        dd = mean(post, sd) - mean(pre, sd)
        did_y = dy - (mean(cpost, sy) - mean(cpre, sy))
        did_d = dd - (mean(cpost, sd) - mean(cpre, sd))
        weight = s * n[post].sum()
        rec = {"did_y": did_y, "did_d": did_d, "dd": dd, "weight": weight}
        if "did" in want:
            num["did"] += weight * did_y
            den["did"] += weight * did_d
            rec["did"] = did_y / did_d
            rec["did_num"] = did_y
        if "tc" in want:
            n_pre = n[pre].sum()
            corr = 0.0
            for q in range(n_cat):
                pq = atoms.mask(a, 2, s, q)
                npq = n[pq].sum()
                if npq.real == 0:
                    continue
                m1 = atoms.mask(a + 1, 1, 0, q)
                m0 = atoms.mask(a, 2, 0, q)
                if n[m1].sum().real == 0 or n[m0].sum().real == 0:
                    raise IdentificationFailure(
                        "Wald-TC: a treatment value present in the switching "
                        "group before the change is absent from the stable "
                        "group in one of the two periods, so its counterfactual "
                        "trend delta_d is not identified.",
                        recovery_hint="Group treatment values with "
                        "newcateg=[...] (upper bounds of each set), as in "
                        "fuzzydid's newcateg() option.",
                        diagnostics={"pair": a, "direction": s, "category": q},
                    )
                delta = mean(m1, sy) - mean(m0, sy)
                corr = corr + (npq / n_pre) * delta
            tc_num = dy - corr
            num["tc"] += weight * did_d * tc_num / dd
            den["tc"] += weight * did_d
            rec["tc"] = tc_num / dd
            rec["tc_num"] = tc_num
        parts.append(rec)
    if numerator:
        return np.array([parts[0][k + "_num"] for k in want]), parts
    return np.array([num[k] / den[k] for k in want]), parts


# ----------------------------------------------------------------------
# Wald-CIC
# ----------------------------------------------------------------------


def _qq(
    c0: np.ndarray, c1: np.ndarray, y: np.ndarray, rule: str = "exact"
) -> np.ndarray:
    """``F^{-1}_{c1}(F_{c0}(y))`` with empirical cdf / left-continuous inverse.

    ``rule='exact'`` finds ``inf{z : F_{c1}(z) >= u}`` in integer arithmetic
    (``k = ceil(count * n1 / n0)``). ``rule='stata'`` computes
    ``ceil((count / n0) * n1)`` in floating point as fuzzydid does: when
    ``count * n1 / n0`` is an integer the product can round up past it and
    pick the next order statistic (always, when the two control cells have
    the same size).
    """
    s0 = np.sort(c0)
    s1 = np.sort(c1)
    count = np.searchsorted(s0, y, side="right")  # n0 * F_{c0}(y)
    if rule == "stata":
        k = np.ceil(count / len(s0) * len(s1)).astype(np.int64)
    else:
        k = -((-count * len(s1)) // len(s0))
    k = np.clip(k, 1, len(s1))
    return s1[k - 1]


def _cic(
    des: _Design,
    rows: np.ndarray,
    comps: Sequence[_Comparison],
    numerator: bool,
    rule: str,
) -> Tuple[Any, List[Dict[str, Any]]]:
    t = des.tcode[rows]
    gt = des.gt[rows]
    gf = des.gf[rows]
    y = des.y[rows]
    d = des.d[rows]
    cat = des.cat[rows]
    num = den = 0.0
    parts = []
    for c in comps:
        a, s = c.pair, c.sign
        pre = (t == a) & (gf == s)
        post = (t == a + 1) & (gt == s)
        cpre = (t == a) & (gf == 0)
        cpost = (t == a + 1) & (gt == 0)
        cf = 0.0
        for q in np.unique(cat[pre]):
            m0 = cpre & (cat == q)
            m1 = cpost & (cat == q)
            if not m0.any() or not m1.any():
                raise IdentificationFailure(
                    "Wald-CIC: a treatment value present in the switching "
                    "group before the change is absent from the stable group "
                    "in one of the two periods.",
                    recovery_hint="Group treatment values with newcateg=[...].",
                    diagnostics={"pair": a, "direction": s, "category": int(q)},
                )
            cf += _qq(y[m0], y[m1], y[pre & (cat == q)], rule).sum()
        cf /= pre.sum()
        dd = d[post].mean() - d[pre].mean()
        did_d = dd - (d[cpost].mean() - d[cpre].mean())
        cic_num = y[post].mean() - cf
        weight = s * post.sum()
        num += weight * did_d * cic_num / dd
        den += weight * did_d
        parts.append({"cic": cic_num / dd, "cic_num": cic_num})
    if numerator:
        return parts[0]["cic_num"], parts
    return num / den, parts


# ----------------------------------------------------------------------
# Covariates: parametric conditional means
# ----------------------------------------------------------------------


def _keep_columns(x: np.ndarray) -> np.ndarray:
    """Columns kept by a forward rank check (intercept first), as Stata drops
    later collinear regressors."""
    keep: List[int] = []
    cur = np.empty((x.shape[0], 0))
    rank = 0
    for j in range(x.shape[1]):
        trial = np.column_stack([cur, x[:, j]])
        r = np.linalg.matrix_rank(trial)
        if r > rank:
            keep.append(j)
            cur, rank = trial, r
    return np.array(keep, dtype=int)


def _glm_fit(x: np.ndarray, v: np.ndarray, link: str) -> np.ndarray:
    """Newton-Raphson for a logit / probit with intercept-included ``x``."""
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        eta = x @ beta
        if link == "logit":
            p = 1.0 / (1.0 + np.exp(-eta))
            grad = x.T @ (v - p)
            hess = (x * (p * (1 - p))[:, None]).T @ x
        else:
            p = stats.norm.cdf(eta)
            phi = stats.norm.pdf(eta)
            p = np.clip(p, 1e-300, 1 - 1e-16)
            lam = phi * (v / p - (1 - v) / (1 - p))
            grad = x.T @ lam
            w = phi**2 / (p * (1 - p))
            hess = (x * w[:, None]).T @ x
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-13 * (1 + np.max(np.abs(beta))):
            return beta
    raise ConvergenceFailure(
        f"fuzzy_did: the {link} model of a conditional mean did not converge "
        "(perfect prediction or separation in a cell).",
        recovery_hint="Use modelx with 'ols' for that equation, or drop the "
        "covariate that separates the cell.",
        diagnostics={"link": link},
    )


def _fit_predict(
    xtr: np.ndarray, v: np.ndarray, xte: np.ndarray, model: str
) -> np.ndarray:
    xtr1 = np.column_stack([np.ones(len(xtr)), xtr])
    xte1 = np.column_stack([np.ones(len(xte)), xte])
    keep = _keep_columns(xtr1)
    xtr1, xte1 = xtr1[:, keep], xte1[:, keep]
    if model == "ols":
        beta = np.linalg.lstsq(xtr1, v, rcond=None)[0]
        return cast(np.ndarray, xte1 @ beta)
    vals = np.unique(v)
    if not np.all(np.isin(vals, (0.0, 1.0))):
        raise MethodIncompatibility(
            f"fuzzy_did: modelx '{model}' needs a 0/1 variable; this one takes "
            f"values {vals[:6].tolist()}.",
            recovery_hint="Use 'ols' for non-binary outcomes / treatments.",
            diagnostics={"model": model},
        )
    if len(vals) == 1:
        return np.full(len(xte), vals[0])
    beta = _glm_fit(xtr1, v, model)
    eta = xte1 @ beta
    if model == "logit":
        return cast(np.ndarray, 1.0 / (1.0 + np.exp(-eta)))
    return cast(np.ndarray, stats.norm.cdf(eta))


def _did_tc_cov(
    des: _Design,
    rows: np.ndarray,
    comps: Sequence[_Comparison],
    want: Sequence[str],
    numerator: bool,
    modelx: Sequence[str],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    t = des.tcode[rows]
    gt = des.gt[rows]
    gf = des.gf[rows]
    y = des.y[rows]
    d = des.d[rows]
    cat = des.cat[rows]
    assert des.x is not None  # only called when covariates are given
    x = des.x[rows]
    m_y, m_d, m_p = modelx
    binary_d = des.n_cat == 2 and set(np.unique(des.d)) <= {0.0, 1.0}
    num = {k: 0.0 for k in want}
    den = {k: 0.0 for k in want}
    parts = []
    for c in comps:
        a, s = c.pair, c.sign
        pre = (t == a) & (gf == s)
        post = (t == a + 1) & (gt == s)
        cpre = (t == a) & (gf == 0)
        cpost = (t == a + 1) & (gt == 0)
        xt = x[post]

        def m(cell: np.ndarray, v: np.ndarray, model: str) -> Any:
            return _fit_predict(x[cell], v[cell], xt, model).mean()

        my_pre = m(pre, y, m_y)
        md_pre = m(pre, d, m_d)
        did_y = y[post].mean() - (my_pre + m(cpost, y, m_y) - m(cpre, y, m_y))
        did_d = d[post].mean() - (md_pre + m(cpost, d, m_d) - m(cpre, d, m_d))
        weight = s * post.sum()
        rec = {"did_y": did_y, "did_d": did_d, "weight": weight}
        if "did" in want:
            num["did"] += weight * did_y
            den["did"] += weight * did_d
            rec["did"] = did_y / did_d
            rec["did_num"] = did_y
        if "tc" in want:
            dd = d[post].mean() - md_pre
            cf = _fit_predict(x[pre], y[pre], xt, m_y)
            for q in np.unique(cat[pre]):
                m0 = cpre & (cat == q)
                m1 = cpost & (cat == q)
                if not m0.any() or not m1.any():
                    raise IdentificationFailure(
                        "Wald-TC: a treatment value present in the switching "
                        "group before the change is absent from the stable "
                        "group in one of the two periods.",
                        recovery_hint="Group treatment values with newcateg=[...].",
                        diagnostics={"pair": a, "direction": s, "category": int(q)},
                    )
                ind = (cat[pre] == q).astype(float)
                if binary_d:
                    p1 = _fit_predict(x[pre], d[pre], xt, m_d)
                    pq = p1 if q == 1 else 1.0 - p1
                else:
                    pq = _fit_predict(x[pre], ind, xt, m_p)
                delta = _fit_predict(x[m1], y[m1], xt, m_y) - _fit_predict(
                    x[m0], y[m0], xt, m_y
                )
                cf = cf + pq * delta
            tc_num = y[post].mean() - cf.mean()
            num["tc"] += weight * did_d * tc_num / dd
            den["tc"] += weight * did_d
            rec["tc"] = tc_num / dd
            rec["tc_num"] = tc_num
        parts.append(rec)
    if numerator:
        return np.array([parts[0][k + "_num"] for k in want]), parts
    return np.array([num[k] / den[k] for k in want]), parts


# ----------------------------------------------------------------------
# One full estimation on a set of rows
# ----------------------------------------------------------------------


def _estimate(
    des: _Design,
    rows: np.ndarray,
    want: Sequence[str],
    numerator: bool,
    modelx: Sequence[str],
    rule: str = "exact",
) -> Tuple[Dict[str, float], List[_Comparison], Dict[str, Any]]:
    comps = _comparisons(des, rows)
    if not comps:
        raise IdentificationFailure(
            "fuzzy_did: no switching group is observed in two consecutive periods.",
            recovery_hint="Check the group_change / group_change_next coding.",
        )
    for c in comps:
        t = des.tcode[rows]
        if not (np.any((t == c.pair) & (des.gf[rows] == 0))) or not np.any(
            (t == c.pair + 1) & (des.gt[rows] == 0)
        ):
            raise IdentificationFailure(
                "fuzzy_did: a pair of periods has switching groups but no "
                "stable (group_change == 0) control group.",
                recovery_hint="The point estimators need a stable group; "
                "fuzzydid's partial option (bounds without a stable group) is "
                "not implemented.",
                diagnostics={"pair": [des.periods[c.pair], des.periods[c.pair + 1]]},
            )
    out: Dict[str, float] = {}
    parts: Dict[str, Any] = {}
    dt = [k for k in want if k in ("did", "tc")]
    with np.errstate(divide="ignore", invalid="ignore"):
        if dt:
            if des.x is None:
                atoms = _Atoms(des, rows)
                n, sy, sd = atoms.stats(des)
                vals, p = _did_tc_from_stats(
                    atoms, comps, n, sy, sd, dt, numerator, des.n_cat
                )
            else:
                vals, p = _did_tc_cov(des, rows, comps, dt, numerator, modelx)
            for k, v in zip(dt, vals):
                out[k] = float(np.real(v))
            parts["did_tc"] = p
        if "cic" in want:
            v, p = _cic(des, rows, comps, numerator, rule)
            out["cic"] = float(v)
            parts["cic"] = p
    bad = [k for k, v in out.items() if not np.isfinite(v)]
    if bad:
        raise IdentificationFailure(
            "fuzzy_did: the estimator's denominator is zero (the treatment "
            "rate does not change more in the switching groups than in the "
            f"stable ones, or not at all): {bad} not finite.",
            recovery_hint="Check that group_change codes the groups whose "
            "treatment changes; a Wald ratio needs a first stage.",
            diagnostics={"estimators": bad},
        )
    return out, comps, parts


def _analytic_if(
    des: _Design, rows: np.ndarray, want: Sequence[str], numerator: bool
) -> np.ndarray:
    """Per-row derivative of each estimator w.r.t. its row weight."""
    comps = _comparisons(des, rows)
    atoms = _Atoms(des, rows)
    n, sy, sd = atoms.stats(des)
    base = np.concatenate([n, sy, sd]).astype(complex)
    k = len(n)
    h = 1e-30
    grad = np.empty((3 * k, len(want)))
    for j in range(3 * k):
        z = base.copy()
        z[j] += 1j * h
        v, _ = _did_tc_from_stats(
            atoms, comps, z[:k], z[k : 2 * k], z[2 * k :], want, numerator, des.n_cat
        )
        grad[j] = np.imag(v) / h
    a = atoms.atom_of_row
    # Every estimator is a function of cell means, hence homogeneous of
    # degree 0 in the row weights: these derivatives sum to zero (Euler).
    return cast(
        np.ndarray,
        grad[a]
        + des.y[rows][:, None] * grad[k + a]
        + des.d[rows][:, None] * grad[2 * k + a],
    )
