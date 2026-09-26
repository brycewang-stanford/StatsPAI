"""Stata / fixest small-sample conventions for ``sp.panel`` standard errors.

``sp.panel`` fits its static models through linearmodels, whose covariance
scaling follows linearmodels' own ``debiased`` / ``count_effects`` rules and
matches neither Stata nor R ``fixest`` once standard errors are robust or
clustered. ``ssc="stata"`` / ``ssc="fixest"`` recompute the covariance from
the transformed design on the same estimation sample and apply the reference
package's factor and reference distribution. Every rule below is pinned
against Stata 18 and fixest 0.14 on an unbalanced panel by
``tests/reference_parity/test_panel_ssc_stata_parity.py``.

Notation: ``N`` observations used, ``k`` slope regressors (plus the constant
where the model has one), ``G`` clusters, ``K_full`` every estimated
parameter including absorbed fixed-effect levels.

Stata (the command each method corresponds to):

* ``fe`` -> ``xtreg, fe``; ``twoway`` -> ``xtreg ... i.t, fe``;
  ``pooled`` -> ``regress``; ``fd`` -> ``regress D.y D.x, nocons``;
  ``re`` -> ``xtreg, re``; ``be`` -> ``xtreg, be``.
* Unadjusted: ``s^2 = e'e / (N - K_full)``, t(N - K_full); ``re`` uses z.
* ``robust``: for ``fe`` / ``twoway`` / ``re`` Stata's ``vce(robust)`` *is*
  clustering on the panel variable; ``pooled`` / ``fd`` get HC1,
  ``N / (N - K_full)``.
* Cluster: ``G/(G-1) * (N-1)/(N-K)``, t(G - 1) (z for ``re``). When the
  cluster nests the panel variable (``xtreg``), the unit effects are not
  counted beyond the constant: ``K = K_full - (G_unit - 1)``. When it does
  not, ``xtreg`` refuses and ``areg, absorb(id)`` counts everything:
  ``K = K_full``.

fixest (``ssc(adj=TRUE, fixef.K="nested", cluster.adj=TRUE, t.df="min")``):

* Unadjusted and ``hetero``: ``N - K_full``, t(N - K_full).
* Cluster: ``G/(G-1) * (N-1)/(N-K)`` with every fixed-effect dimension
  nested in the cluster counted as one level:
  ``K = K_full - sum(levels_d - 1)`` over nested dimensions; t(G - 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

SSC_CHOICES = ("stata", "fixest")

_STATA_COMMAND = {
    "fe": "xtreg, fe",
    "twoway": "xtreg ... i.t, fe",
    "pooled": "regress",
    "fd": "regress D.y D.x, nocons",
    "re": "xtreg, re",
    "be": "xtreg, be",
    "mundlak": "xtreg ... (unit means), re",
    "chamberlain": "xtreg ... (unit means), re",
}


def resolve_ssc(ssc: Optional[str], method: str) -> Optional[str]:
    """Validate ``ssc=`` for ``method``; ``None`` keeps linearmodels' scaling."""
    if ssc is None:
        return None
    key = str(ssc).lower()
    if key == "linearmodels":
        return None
    if key not in SSC_CHOICES:
        raise MethodIncompatibility(
            f"ssc={ssc!r} is not supported; use 'stata' or 'fixest'.",
            diagnostics={"ssc": ssc},
            recovery_hint="Pass ssc='stata' (xtreg / regress) or "
            "ssc='fixest' (R fixest defaults); omit it for linearmodels.",
        )
    supported = (
        set(_STATA_COMMAND) if key == "stata" else {"fe", "twoway", "pooled", "fd"}
    )
    if method not in supported:
        raise MethodIncompatibility(
            f"ssc={key!r} is not defined for method={method!r}.",
            diagnostics={"ssc": key, "method": method},
            recovery_hint=(
                "fixest has no random-effects or between estimator; use "
                "ssc='stata' for those."
                if key == "fixest"
                else "Drop ssc= for this method."
            ),
        )
    return key


@dataclass
class SSCInference:
    """Covariance and reference distribution under a named convention."""

    vcov: pd.DataFrame
    df_inference: Optional[float]  # None -> z
    vce: str  # 'unadjusted' | 'robust' | 'cluster'
    n_clusters: Optional[int]
    reference: str
    convention: str = "stata"  # 'stata' | 'fixest'


def _cluster_groups(
    cluster: Any, entity: str, time: str, frame: pd.DataFrame
) -> Optional[np.ndarray]:
    if cluster is None or cluster is False:
        return None
    if cluster == "twoway" or isinstance(cluster, (list, tuple)):
        raise MethodIncompatibility(
            "ssc= supports one-way clustering only.",
            diagnostics={"cluster": cluster},
            recovery_hint="Cluster on one variable, or use "
            "sp.feols(..., vcov={'CRV1': 'a+b'}) for two-way clustering.",
        )
    if cluster is True or cluster in ("entity", entity):
        col = entity
    elif cluster in ("time", time):
        col = time
    else:
        col = cluster
    codes: np.ndarray = pd.factorize(frame[col])[0]
    return codes


def _nests(inner: np.ndarray, outer: np.ndarray) -> bool:
    """True when every level of ``inner`` sits inside one level of ``outer``."""
    pairs = pd.DataFrame({"i": inner, "o": outer}).drop_duplicates()
    return bool(pairs["i"].is_unique)


def _ols(Xw: np.ndarray, yw: np.ndarray) -> np.ndarray:
    coef: np.ndarray = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    return coef


def _check_params(
    b: np.ndarray, reference: pd.Series, names: List[str], method: str
) -> None:
    ref = reference.reindex(names).to_numpy(dtype=float)
    scale = max(1.0, float(np.max(np.abs(ref))))
    gap = float(np.max(np.abs(b - ref)))
    # Two-way demeaning is iterative on both sides; 1e-7 is far above either
    # tolerance and far below any real disagreement.
    if not np.isfinite(gap) or gap > 1e-7 * scale:
        raise RuntimeError(
            f"sp.panel(ssc=...): the transformed {method} design reproduces "
            f"the coefficients only to {gap:.2e}; the covariance would not "
            "describe the reported estimates."
        )


def panel_ssc_inference(
    *,
    ssc: str,
    method: str,
    data: pd.DataFrame,
    dep_var: str,
    exog: List[str],
    entity: str,
    time: str,
    robust: str,
    cluster: Any,
    weights: Optional[str],
    lm_result: Any,
) -> SSCInference:
    """Covariance of ``lm_result.params`` under ``ssc`` (see module doc)."""
    if robust in ("kernel", "driscoll-kraay"):
        raise MethodIncompatibility(
            "ssc= applies to unadjusted, robust and one-way cluster SEs; "
            f"robust={robust!r} has no Stata / fixest counterpart here.",
            diagnostics={"robust": robust, "ssc": ssc},
            recovery_hint="Drop ssc= for Driscoll-Kraay SEs.",
        )
    cols = [dep_var, *exog, entity, time]
    extra = [
        c
        for c in (weights, cluster)
        if isinstance(c, str) and c in data.columns and c not in cols
    ]
    frame = data[cols + extra].dropna().sort_values([entity, time], kind="stable")
    frame = frame.reset_index(drop=True)
    params = lm_result.params
    names = list(params.index)
    has_const = "const" in names
    w = (
        frame[weights].to_numpy(dtype=float)
        if weights is not None
        else np.ones(len(frame))
    )

    y = frame[dep_var].to_numpy(dtype=float)
    X = frame[exog].to_numpy(dtype=float)
    unit = pd.factorize(frame[entity])[0]
    period = pd.factorize(frame[time])[0]
    n_unit = int(unit.max()) + 1
    n_period = int(period.max()) + 1
    fe_dims: Dict[str, np.ndarray] = {}

    if method in ("fe", "twoway"):
        from .hdfe import Absorber

        fe = pd.DataFrame({"u": unit})
        fe_dims["u"] = unit
        if method == "twoway":
            fe["t"] = period
            fe_dims["t"] = period
        ab = Absorber(
            fe,
            weights=w,
            drop_singletons=False,
            tol=1e-13,
            maxiter=100_000,
            n_obs=len(frame),
        )
        Xt = ab.demean(X)
        yt = ab.demean(y)
        k_full = X.shape[1] + n_unit + (n_period - 1 if method == "twoway" else 0)
    elif method == "fd":
        levels = np.sort(frame[time].unique())
        pos = np.searchsorted(levels, frame[time].to_numpy())
        same = np.r_[False, unit[1:] == unit[:-1]]
        consecutive = np.r_[False, pos[1:] == pos[:-1] + 1]
        keep = same & consecutive
        Xt = (X[1:] - X[:-1])[keep[1:]]
        yt = (y[1:] - y[:-1])[keep[1:]]
        frame = frame.loc[keep].reset_index(drop=True)
        w = w[keep]
        unit = unit[keep]
        k_full = X.shape[1]
    elif method in ("re", "mundlak", "chamberlain"):
        theta = lm_result.theta.iloc[:, 0]
        th = theta.reindex(frame[entity]).to_numpy(dtype=float)
        Xc = np.column_stack([np.ones(len(frame)), X]) if has_const else X
        means_X = pd.DataFrame(Xc).groupby(unit).transform("mean").to_numpy()
        means_y = pd.Series(y).groupby(unit).transform("mean").to_numpy()
        Xt = Xc - th[:, None] * means_X
        yt = y - th * means_y
        k_full = Xc.shape[1]
    elif method == "be":
        Xc = np.column_stack([np.ones(len(frame)), X])
        Xt = pd.DataFrame(Xc).groupby(unit).mean().to_numpy()
        yt = pd.Series(y).groupby(unit).mean().to_numpy()
        w = np.ones(len(yt))
        k_full = Xc.shape[1]
    else:  # pooled
        Xt = np.column_stack([np.ones(len(frame)), X]) if has_const else X
        yt = y
        k_full = Xt.shape[1]

    if method in ("re", "mundlak", "chamberlain", "be", "pooled") and has_const:
        order = ["const", *exog]
    else:
        order = list(exog)

    n = len(yt)
    if n != int(lm_result.nobs):
        raise RuntimeError(
            f"sp.panel(ssc=...): rebuilt {n} observations but the fit used "
            f"{int(lm_result.nobs)}."
        )
    w_norm = w * (n / w.sum())  # Stata aweights sum to N
    sw = np.sqrt(w_norm)
    Xw = Xt * sw[:, None]
    yw = yt * sw
    b = _ols(Xw, yw)
    _check_params(b, params, order, method)
    e = yw - Xw @ b
    bread = np.linalg.inv(Xw.T @ Xw)

    groups = _cluster_groups(cluster, entity, time, frame)
    if method == "be" and (groups is not None or robust != "nonrobust"):
        raise MethodIncompatibility(
            "Stata's xtreg, be has conventional standard errors only.",
            diagnostics={"method": method, "robust": robust, "cluster": cluster},
            recovery_hint="Drop robust= / cluster= for method='be'.",
        )
    z_dist = ssc == "stata" and method in ("re", "mundlak", "chamberlain")
    if groups is None and robust == "robust":
        if ssc == "stata" and method in (
            "fe",
            "twoway",
            "re",
            "mundlak",
            "chamberlain",
        ):
            groups = unit  # xtreg: vce(robust) is vce(cluster panelvar)
        else:
            S = Xw * e[:, None]
            V = (n / (n - k_full)) * (bread @ (S.T @ S) @ bread)
            return SSCInference(
                vcov=pd.DataFrame(V, index=order, columns=order).loc[names, names],
                df_inference=None if z_dist else float(n - k_full),
                vce="robust",
                n_clusters=None,
                reference=_reference(ssc, method, "robust"),
                convention=ssc,
            )

    if groups is None:
        V = (float(e @ e) / (n - k_full)) * bread
        return SSCInference(
            vcov=pd.DataFrame(V, index=order, columns=order).loc[names, names],
            df_inference=None if z_dist else float(n - k_full),
            vce="unadjusted",
            n_clusters=None,
            reference=_reference(ssc, method, "unadjusted"),
            convention=ssc,
        )

    G = int(groups.max()) + 1
    if G < 2:
        raise MethodIncompatibility(
            "Cluster-robust SEs need at least two clusters.",
            diagnostics={"n_clusters": G},
            recovery_hint="Cluster on a variable with more than one value.",
        )
    k_c = k_full
    if ssc == "stata" and method in ("fe", "twoway") and _nests(unit, groups):
        k_c = k_full - (n_unit - 1)
    elif ssc == "fixest":
        for dim in fe_dims.values():
            if _nests(dim, groups):
                k_c -= int(dim.max())  # levels - 1
    S = Xw * e[:, None]
    sg = np.zeros((G, S.shape[1]))
    np.add.at(sg, groups, S)
    factor = (G / (G - 1)) * ((n - 1) / (n - k_c))
    V = factor * (bread @ (sg.T @ sg) @ bread)
    return SSCInference(
        vcov=pd.DataFrame(V, index=order, columns=order).loc[names, names],
        df_inference=None if z_dist else float(G - 1),
        vce="cluster",
        n_clusters=G,
        reference=_reference(ssc, method, "cluster"),
        convention=ssc,
    )


def _reference(ssc: str, method: str, vce: str) -> str:
    if ssc == "fixest":
        v = {"unadjusted": "iid", "robust": "hetero", "cluster": "cluster"}[vce]
        return f"R fixest::feols(vcov='{v}') default ssc()"
    return f"Stata {_STATA_COMMAND[method]} ({vce})"
