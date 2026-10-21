"""Heterogeneous treatment effects in sharp RD designs.

Clean-room implementation (2026-10-21) of Calonico, Cattaneo, Farrell,
Palomba and Titiunik (2025): conditional average treatment effects
``kappa(w) = E[Y(1) - Y(0) | X = c, W = w]`` from a kernel-weighted local
polynomial regression fully interacted with the moderators ``W``, with
robust bias-corrected inference (order ``p + 1`` fit on the same window) and
MSE/CER-optimal bandwidths. R ``rdhte`` 0.2.0 (GPL-3) was used only as a
black box; the conventions it fixed are listed in
``docs/dev/cleanroom/rdhte.md`` and pinned by
``tests/reference_parity/test_rdhte_R_parity.py``.

Two kinds of moderator:

* **continuous** (one or more numeric columns): ``kappa(w) = theta + xi'w``
  (functional-coefficient model), one bandwidth;
* **subgroups** (a single 0/1, boolean or categorical column): one effect per
  level, each level with its own bandwidth (subgroup analysis).
"""

import types
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._cct_bandwidth import BW_SELECTORS, cct_bandwidth
from ._hte_core import (
    canonical_kernel,
    evaluation_grid,
    interacted_fit,
    normalize_vce,
    subgroup_levels,
)

__all__ = ["rdhte", "rdbwhte", "rdhte_lincom"]

CausalResult._CITATIONS["rdhte"] = (
    "@misc{calonico2025treatment,\n"
    "  title={Treatment Effect Heterogeneity in Regression Discontinuity Designs},\n"
    "  author={Calonico, Sebastian and Cattaneo, Matias D. and Farrell, Max H. "
    "and Palomba, Filippo and Titiunik, Roc{\\'\\i}o},\n"
    "  year={2025},\n"
    "  eprint={2503.13696},\n"
    "  archivePrefix={arXiv}\n"
    "}"
)


def _bad(msg: str, hint: str, **diag: Any) -> MethodIncompatibility:
    return MethodIncompatibility(msg, recovery_hint=hint, diagnostics=dict(diag))


# --------------------------------------------------------------------------
# data preparation
# --------------------------------------------------------------------------


def _is_subgroup_column(col: pd.Series) -> bool:
    if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object:
        return True
    if col.dtype == bool:
        return True
    vals = set(pd.unique(col.dropna()))
    return vals == {0, 1}


def _prepare(
    data: pd.DataFrame,
    y: str,
    x: str,
    z: Union[str, List[str], None],
    c: float,
    cluster: Optional[str],
    function: str,
) -> Dict[str, Any]:
    zs = [] if z is None else ([z] if isinstance(z, str) else list(z))
    if not zs:
        raise _bad(
            f"{function}: z= (the moderator column or columns) is required.",
            "Pass z='moderator'; for the average effect use sp.rdrobust.",
        )
    cols = [y, x] + zs
    check_columns(data, cols, function=function)
    if cluster is not None:
        if cluster not in data.columns:
            raise _bad(
                f"{function}: cluster column {cluster!r} not found in data.",
                "Pass the name of an existing column as cluster=.",
                cluster=cluster,
            )
        cols = cols + [cluster]
    df = data[cols].dropna()
    if len(df) == 0:
        raise DataInsufficient(
            f"{function}: no complete rows in {cols}.",
            recovery_hint="Check the columns for missing values.",
        )
    xc = df[x].to_numpy(dtype=float) - float(c)
    subgroups = len(zs) == 1 and _is_subgroup_column(df[zs[0]])
    out: Dict[str, Any] = {
        "y": df[y].to_numpy(dtype=float),
        "xc": xc,
        "zs": zs,
        "cluster": None if cluster is None else df[cluster].to_numpy(),
        "subgroups": subgroups,
        "n": len(df),
    }
    if subgroups:
        col = df[zs[0]]
        levels = subgroup_levels(col)
        if all(isinstance(lv, (int, float, np.integer, np.floating)) for lv in levels):
            levels = [float(lv) for lv in levels]
        if len(levels) < 2:
            raise _bad(
                f"{function}: the moderator {zs[0]!r} takes a single value.",
                "Use a moderator that varies, or sp.rdrobust for the average.",
            )
        codes = np.empty(len(col), dtype=int)
        vals = col.to_numpy()
        if isinstance(levels[0], float):
            vals = vals.astype(float)
        for g, lv in enumerate(levels):
            codes[vals == lv] = g
        out.update(levels=levels, groups=codes)
        out["W"] = np.column_stack(
            [(codes == g).astype(float) for g in range(1, len(levels))]
        )
    else:
        try:
            out["W"] = df[zs].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise _bad(
                f"{function}: moderators {zs} must be numeric, or a single "
                "categorical column.",
                "Convert the moderators to numbers, or pass one categorical "
                "column for subgroup effects.",
            ) from exc
    return out


def _level_key(level: Any) -> str:
    return "None" if level is None else str(level)


def _level_label(z: str, level: Any) -> str:
    if isinstance(level, (int, float, np.integer, np.floating)) and not isinstance(
        level, bool
    ):
        return f"{z}={float(level):g}"
    return f"{z}={level}"


# --------------------------------------------------------------------------
# bandwidths
# --------------------------------------------------------------------------


def _check_orders(p: int, q: Optional[int]) -> Tuple[int, int]:
    if int(p) != p or p < 0:
        raise _bad(f"p must be a non-negative integer; got {p!r}", "Use p=1.", p=p)
    p = int(p)
    q = p + 1 if q is None else int(q)
    if q <= p:
        raise _bad(f"q must exceed p; got q={q}, p={p}", "Use q=p+1.", p=p, q=q)
    return p, q


def _selector_vce(vce: str) -> str:
    return "hc1" if vce == "cr1" else vce


def _bandwidth_pairs(
    prep: Dict[str, Any],
    p: int,
    q: int,
    kernel: str,
    bwselect: str,
    vce: str,
) -> Dict[Any, Tuple[float, float]]:
    """MSE/CER-optimal ``(h_left, h_right)`` per level (or one pair)."""
    if bwselect not in BW_SELECTORS:
        raise ValueError(f"bwselect must be one of {BW_SELECTORS}, got {bwselect!r}")
    cl = prep["cluster"]
    selector_vce = _selector_vce(vce)

    def select(mask: np.ndarray) -> Tuple[float, float]:
        xc = prep["xc"][mask]
        if min(np.sum(xc < 0), np.sum(xc >= 0)) < p + 3:
            raise DataInsufficient(
                "rdbwhte: too few observations on a side of the cutoff for "
                "bandwidth selection.",
                recovery_hint="Pass h= or use a coarser moderator.",
                diagnostics={"n_left": int(np.sum(xc < 0))},
            )
        bw = cct_bandwidth(
            prep["y"][mask],
            xc,
            0.0,
            p=p,
            q=q,
            kernel=kernel,
            bwselect=bwselect,
            vce=selector_vce,
            cluster=None if cl is None else cl[mask],
        )
        return float(bw["h_left"]), float(bw["h_right"])

    if not prep["subgroups"]:
        return {None: select(np.ones(prep["n"], dtype=bool))}
    return {lv: select(prep["groups"] == g) for g, lv in enumerate(prep["levels"])}


def _user_pair(h: Any, name: str) -> Tuple[float, float]:
    if isinstance(h, (tuple, list, np.ndarray)):
        if len(h) != 2:
            raise _bad(
                f"{name} must be a number or a (left, right) pair; got {h!r}",
                f"Pass {name}=0.4 or {name}=(0.3, 0.5).",
            )
        pair = (float(h[0]), float(h[1]))
    else:
        pair = (float(h), float(h))
    if not all(np.isfinite(v) and v > 0 for v in pair):
        raise _bad(f"{name} must be positive; got {h!r}", f"Pass a positive {name}.")
    return pair


def _per_obs(
    prep: Dict[str, Any], pairs: Dict[Any, Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    n = prep["n"]
    if not prep["subgroups"]:
        h_left, h_right = pairs[None]
        return np.full(n, h_left), np.full(n, h_right)
    hl = np.empty(n)
    hr = np.empty(n)
    for g, lv in enumerate(prep["levels"]):
        m = prep["groups"] == g
        hl[m], hr[m] = pairs[lv]
    return hl, hr


# --------------------------------------------------------------------------
# rdhte
# --------------------------------------------------------------------------


def _wald(est: np.ndarray, V: np.ndarray) -> Dict[str, float]:
    est = np.atleast_1d(est)
    stat = float(est @ np.linalg.solve(V, est))
    df = int(est.size)
    return {"statistic": stat, "pvalue": float(stats.chi2.sf(stat, df)), "df": df}


@records_call("rdhte")
def rdhte(
    data: pd.DataFrame,
    y: str,
    x: str,
    z: Union[str, List[str]],
    c: float = 0,
    p: int = 1,
    h: Optional[Union[float, Tuple[float, float]]] = None,
    b: Optional[float] = None,
    kernel: str = "triangular",
    bwselect: str = "mserd",
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    eval_points: Optional[np.ndarray] = None,
    n_eval: int = 20,
    q: Optional[int] = None,
    vce: Optional[str] = None,
) -> CausalResult:
    """Conditional average treatment effects (CATE) in sharp RD designs.

    Fits ``y`` on ``[1, T] x [1, x-c, ..., (x-c)^p] x [1, z]`` by kernel
    weighted least squares on ``|x - c| <= h`` and reports
    ``CATE(z) = beta_T + z' beta_{T:z}``, with robust bias-corrected inference
    from the order-``q`` fit on the same window.

    Parameters
    ----------
    data : DataFrame
    y, x : str
        Outcome and running variable.
    z : str or list of str
        Moderator(s). A single 0/1, boolean or categorical column gives one
        effect per level (subgroup analysis, a bandwidth per level); numeric
        columns give a linear CATE function.
    c : float, default 0
        Cutoff.
    p : int, default 1
        Polynomial order of the point estimate (``0`` = local constant).
    h : float or (float, float), optional
        Bandwidth, or ``(h_left, h_right)``; default MSE-optimal
        (``bwselect``), per level for subgroups.
    b : float, optional
        Window of the bias-correction fit (default ``h``).
    kernel : {'triangular', 'uniform', 'epanechnikov'}
    bwselect : str, default 'mserd'
        One of the CCT selectors (``mserd``, ``msetwo``, ``cerrd``, ...).
    cluster : str, optional
        Cluster column (CR1 variance).
    alpha : float, default 0.05
    eval_points : array, optional
        Moderator values at which to report the CATE (continuous mode).
    n_eval : int, default 20
        Number of default evaluation points (quantiles of ``z``).
    q : int, optional
        Order of the bias-correction fit (default ``p + 1``).
    vce : {'hc0', 'hc1', 'hc2', 'hc3', 'cr1'}, optional
        Default ``'hc3'``, or ``'cr1'`` with ``cluster``.

    Returns
    -------
    CausalResult
        ``detail``: one row per evaluation point (``z_value``, ``cate``,
        ``cate_bc``, ``se``, ``ci_lower``, ``ci_upper``, ``pvalue``); the
        headline is their equally weighted average. ``model_info`` carries the
        coefficients (``coef``, ``coef_bc``, ``se_rb``, ``vcov``), bandwidths
        and a Wald test of homogeneity.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 2000); z = rng.integers(0, 2, 2000)
    >>> y = x + (2 + 3 * z) * (x >= 0) + rng.normal(0, 0.5, 2000)
    >>> df = pd.DataFrame({"y": y, "x": x, "z": z})
    >>> r = sp.rdhte(df, y="y", x="x", z="z")
    >>> r.detail["cate"].round(0).tolist()
    [2.0, 5.0]

    References
    ----------
    calonico2025treatment, calonico2025rdhte
    """
    if not (0.0 < float(alpha) < 1.0):
        raise _bad(f"alpha must lie in (0, 1); got {alpha!r}", "Use alpha=0.05.")
    if int(n_eval) != n_eval or n_eval < 1:
        raise _bad(
            f"n_eval must be a positive integer; got {n_eval!r}", "Use n_eval=20."
        )
    kernel = canonical_kernel(kernel)
    p, q = _check_orders(p, q)
    prep = _prepare(data, y, x, z, c, cluster, "sp.rdhte")
    vce_used = normalize_vce(vce, cluster is not None)

    if h is None:
        pairs = _bandwidth_pairs(prep, p, q, kernel, bwselect, vce_used)
    else:
        pair = _user_pair(h, "h")
        keys = prep["levels"] if prep["subgroups"] else [None]
        pairs = {lv: pair for lv in keys}
    hl, hr = _per_obs(prep, pairs)
    if b is None:
        bl, br = hl, hr
    else:
        b_pair = _user_pair(b, "b")
        bl, br = np.full(prep["n"], b_pair[0]), np.full(prep["n"], b_pair[1])

    W = prep["W"]
    fit_p = interacted_fit(
        prep["y"], prep["xc"], W, hl, hr, p, kernel, vce_used, prep["cluster"]
    )
    fit_q = interacted_fit(
        prep["y"], prep["xc"], W, bl, br, q, kernel, vce_used, prep["cluster"]
    )

    k = len(fit_p.treat_index)
    L_coef: np.ndarray
    if prep["subgroups"]:
        G = len(prep["levels"])
        L_coef = np.zeros((G, k))
        L_coef[:, 0] = 1.0
        for g in range(1, G):
            L_coef[g, g] = 1.0
        names = [_level_label(prep["zs"][0], lv) for lv in prep["levels"]]
        mode = "subgroups"
    else:
        L_coef = np.eye(k)
        names = ["T"] + [f"T:{zz}" for zz in prep["zs"]]
        mode = "continuous"
    coef = L_coef @ fit_p.beta[fit_p.treat_index]
    coef_bc = L_coef @ fit_q.beta[fit_q.treat_index]
    V_t = fit_q.vcov[np.ix_(fit_q.treat_index, fit_q.treat_index)]
    V = L_coef @ V_t @ L_coef.T
    V = 0.5 * (V + V.T)
    se_rb = np.sqrt(np.diag(V))

    # Evaluation design: rows over the reported coefficients.
    if prep["subgroups"]:
        L_eval = np.eye(len(names))
        z_values: List[Any] = list(prep["levels"])
    else:
        d = W.shape[1]
        if eval_points is None:
            grid = evaluation_grid(W, int(n_eval))
        else:
            ep = np.asarray(eval_points, dtype=float)
            if ep.size % d:
                raise _bad(
                    f"eval_points has {ep.size} values, not a multiple of the "
                    f"{d} moderators.",
                    "Pass an array of shape (n_points, n_moderators).",
                )
            grid = ep.reshape(-1, d)
        L_eval = np.column_stack([np.ones(grid.shape[0]), grid])
        z_values = (
            [float(r[0]) for r in grid]
            if d == 1
            else [tuple(map(float, r)) for r in grid]
        )
    cate = L_eval @ coef
    cate_bc = L_eval @ coef_bc
    cov_eval = L_eval @ V @ L_eval.T
    se = np.sqrt(np.clip(np.diag(cov_eval), 0.0, None))
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
    zstat = cate_bc / se
    detail = pd.DataFrame(
        {
            "z_value": z_values,
            "cate": cate,
            "cate_bc": cate_bc,
            "se": se,
            "ci_lower": cate_bc - zcrit * se,
            "ci_upper": cate_bc + zcrit * se,
            "pvalue": 2.0 * stats.norm.sf(np.abs(zstat)),
        }
    )

    w_avg = np.full(len(cate), 1.0 / len(cate))
    ate = float(w_avg @ cate)
    ate_bc = float(w_avg @ cate_bc)
    ate_se = float(np.sqrt(w_avg @ cov_eval @ w_avg))
    ate_p = float(2.0 * stats.norm.sf(abs(ate_bc / ate_se)))
    ate_ci = (ate_bc - zcrit * ate_se, ate_bc + zcrit * ate_se)

    if prep["subgroups"]:
        D = np.column_stack([-np.ones(len(names) - 1), np.eye(len(names) - 1)])
        het = _wald(D @ coef_bc, D @ V @ D.T)
    else:
        het = _wald(coef_bc[1:], V[1:, 1:])

    by_level = {_level_key(lv): pairs[lv] for lv in pairs}
    if prep["subgroups"]:
        bandwidth_h: Any = {lv: pairs[lv] for lv in pairs}
    else:
        hl0, hr0 = pairs[None]
        bandwidth_h = hl0 if hl0 == hr0 else {None: (hl0, hr0)}
    xc = prep["xc"]
    model_info: Dict[str, Any] = {
        "rd_type": "Sharp",
        "mode": mode,
        "polynomial_p": p,
        "polynomial_q": q,
        "kernel": kernel,
        "vce": vce_used,
        "bandwidth_h": bandwidth_h,
        "bandwidth_by_level": by_level,
        "bandwidth_b": "h" if b is None else b,
        "bwselect": bwselect if h is None else "manual",
        "cutoff": c,
        "z_covariates": prep["zs"],
        "n_z": len(prep["zs"]),
        "n_left": int(np.sum(xc < 0)),
        "n_right": int(np.sum(xc >= 0)),
        "n_effective": int(fit_p.n_window),
        "n_eval_points": len(cate),
        "coef_names": names,
        "coef": coef.tolist(),
        "coef_bc": coef_bc.tolist(),
        "se_rb": se_rb.tolist(),
        "vcov": V.tolist(),
        "ate": ate,
        "ate_bc": ate_bc,
        "ate_se": ate_se,
        "ate_pvalue": ate_p,
        "ate_ci": ate_ci,
        "heterogeneity_test": het,
        "_L": L_eval.tolist(),
    }
    result = CausalResult(
        method="RD Heterogeneous Treatment Effects (rdhte)",
        estimand="CATE",
        estimate=ate,
        se=ate_se,
        pvalue=ate_p,
        ci=ate_ci,
        alpha=alpha,
        n_obs=prep["n"],
        detail=detail,
        model_info=model_info,
        _citation_key="rdhte",
    )
    result.plot = types.MethodType(_rdhte_plot, result)  # type: ignore[method-assign]
    return result


@records_call("rdbwhte")
def rdbwhte(
    data: pd.DataFrame,
    y: str,
    x: str,
    z: Union[str, List[str]],
    c: float = 0,
    p: int = 1,
    kernel: str = "triangular",
    q: Optional[int] = None,
    bwselect: str = "mserd",
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
) -> Any:
    """Bandwidth selection for :func:`rdhte`.

    Parameters
    ----------
    data : DataFrame
    y, x : str
        Outcome and running variable.
    z : str or list of str
        Moderator(s), as in :func:`rdhte`.
    c : float, default 0
    p : int, default 1
    kernel : {'triangular', 'uniform', 'epanechnikov'}
    q : int, optional
        Bias-correction order (default ``p + 1``).
    bwselect : str, default 'mserd'
    vce : str, optional
        Variance used inside the selector (default ``'hc3'``; ``'cr1'`` with
        ``cluster``).
    cluster : str, optional

    Returns
    -------
    float or DataFrame
        One common bandwidth as a float; otherwise a frame with columns
        ``level``, ``h_left``, ``h_right`` (one row per subgroup, or one row
        with ``level=None`` for side-specific bandwidths).

    Notes
    -----
    Continuous moderators: the CCT selector on ``(y, x)``. Subgroups: the
    selector on each level's observations (one bandwidth pair per level).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 2000); z = rng.normal(size=2000)
    >>> y = x + (2 + z) * (x >= 0) + rng.normal(0, 0.5, 2000)
    >>> h = sp.rdbwhte(pd.DataFrame({"y": y, "x": x, "z": z}), y="y", x="x", z="z")
    >>> h > 0
    True

    References
    ----------
    calonico2025treatment, calonico2020optimal
    """
    kernel = canonical_kernel(kernel)
    p, q = _check_orders(p, q)
    prep = _prepare(data, y, x, z, c, cluster, "sp.rdbwhte")
    vce_used = normalize_vce(vce, cluster is not None)
    pairs = _bandwidth_pairs(prep, p, q, kernel, bwselect, vce_used)
    if not prep["subgroups"]:
        hl, hr = pairs[None]
        if hl == hr:
            return float(hl)
    return pd.DataFrame(
        [{"level": lv, "h_left": v[0], "h_right": v[1]} for lv, v in pairs.items()]
    )


def rdhte_lincom(
    result: CausalResult,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    linfct: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Linear combinations of :func:`rdhte` effects.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`rdhte`.
    weights : array, optional
        Weights over the rows of ``result.detail`` (evaluation points or
        subgroups).
    alpha : float, default 0.05
    linfct : array, optional
        Rows of weights over the reported coefficients
        (``model_info['coef_names']``); takes precedence over ``weights``.

    Returns
    -------
    dict
        ``estimate`` (from the conventional fit), ``estimate_bc``, ``se``,
        ``z``, ``ci``, ``pvalue`` (robust bias-corrected), and ``joint``
        (Wald statistic, df, p-value). Scalars for one combination, arrays
        otherwise.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 2000); z = rng.integers(0, 2, 2000)
    >>> y = x + (2 + 3 * z) * (x >= 0) + rng.normal(0, 0.5, 2000)
    >>> r = sp.rdhte(pd.DataFrame({"y": y, "x": x, "z": z}), y="y", x="x", z="z")
    >>> out = sp.rdhte_lincom(r, linfct=[-1, 1])
    >>> round(out["estimate"])
    3

    References
    ----------
    calonico2025rdhte
    """
    if not (0.0 < float(alpha) < 1.0):
        raise _bad(f"alpha must lie in (0, 1); got {alpha!r}", "Use alpha=0.05.")
    mi = result.model_info
    coef = np.asarray(mi["coef"], dtype=float)
    coef_bc = np.asarray(mi["coef_bc"], dtype=float)
    V = np.asarray(mi["vcov"], dtype=float)
    names = mi["coef_names"]
    if linfct is not None:
        L = np.atleast_2d(np.asarray(linfct, dtype=float))
        if L.shape[1] != coef.size:
            raise _bad(
                f"linfct needs {coef.size} columns ({names})",
                "Give one weight per reported coefficient.",
            )
    elif weights is not None:
        wv = np.asarray(weights, dtype=float).ravel()
        L_eval = np.asarray(mi["_L"], dtype=float)
        if wv.size != L_eval.shape[0]:
            raise _bad(
                f"weights has length {wv.size}, expected {L_eval.shape[0]} "
                "(number of rows in result.detail)",
                "Give one weight per row of result.detail.",
            )
        L = (wv @ L_eval)[None, :]
    else:
        raise _bad(
            "pass linfct= (over coefficients) or weights= (over rows)",
            "e.g. linfct=[-1, 1] for the difference of two subgroups.",
        )
    est = L @ coef
    est_bc = L @ coef_bc
    cov = L @ V @ L.T
    se = np.sqrt(np.diag(cov))
    zstat = est_bc / se
    zcrit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci = np.column_stack([est_bc - zcrit * se, est_bc + zcrit * se])
    pval = 2.0 * stats.norm.sf(np.abs(zstat))
    joint = _wald(est_bc, cov)
    if L.shape[0] == 1:
        return {
            "estimate": float(est[0]),
            "estimate_bc": float(est_bc[0]),
            "se": float(se[0]),
            "z": float(zstat[0]),
            "ci": (float(ci[0, 0]), float(ci[0, 1])),
            "pvalue": float(pval[0]),
            "joint": joint,
        }
    return {
        "estimate": est,
        "estimate_bc": est_bc,
        "se": se,
        "z": zstat,
        "ci": ci,
        "pvalue": pval,
        "joint": joint,
    }


def _rdhte_plot(
    result: CausalResult,
    ax: Any = None,
    ci_alpha: float = 0.2,
    cate_color: str = "#2171B5",
    ate_color: str = "#CB181D",
    zero_color: str = "gray",
    xlabel: Optional[str] = None,
    ylabel: str = "CATE",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Any:
    """Plot CATE(z) vs z with confidence bands.

    One moderator only; subgroups are drawn as points with interval bars.
    """
    mi = result.model_info
    if mi.get("n_z", 1) > 1:
        raise NotImplementedError(
            "rdhte plot draws one moderator; with several, plot result.detail "
            "against the moderator of interest."
        )
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for rdhte plots.") from exc
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    det = result.detail
    if det is None:
        raise _bad(
            "rdhte plot needs result.detail, which this result does not carry.",
            "Plot a result returned by sp.rdhte.",
        )
    zcol = mi["z_covariates"][0]
    if mi.get("mode") == "subgroups":
        pos = np.arange(len(det))
        ax.errorbar(
            pos,
            det["cate_bc"],
            yerr=[det["cate_bc"] - det["ci_lower"], det["ci_upper"] - det["cate_bc"]],
            fmt="o",
            color=cate_color,
            capsize=4,
        )
        ax.set_xticks(pos)
        ax.set_xticklabels([str(v) for v in det["z_value"]])
    else:
        det = det.sort_values("z_value")
        zv = det["z_value"].to_numpy(dtype=float)
        ax.plot(zv, det["cate"], color=cate_color, label="CATE")
        ax.fill_between(
            zv, det["ci_lower"], det["ci_upper"], color=cate_color, alpha=ci_alpha
        )
    ax.axhline(result.estimate, color=ate_color, linestyle="--", label="average")
    ax.axhline(0.0, color=zero_color, linewidth=0.8)
    ax.set_xlabel(xlabel if xlabel is not None else zcol)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return ax
