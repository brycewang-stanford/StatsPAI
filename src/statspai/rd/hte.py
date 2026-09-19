"""
Heterogeneous treatment effects in regression discontinuity designs.

Implements the methodology of Calonico, Cattaneo, Farrell, Palomba, and
Titiunik (2025) for estimating conditional average treatment effects (CATE)
in RD designs using fully interacted local polynomial models.

``rdhte``, ``rdbwhte`` and ``rdhte_lincom`` reproduce the authors' R package
``rdhte`` (1.x): the point estimate is the order-``p`` interacted fit
``lm(Y ~ T * Xp * W)``, inference is robust bias-corrected (order ``q = p+1``
fit on the same window, ``sandwich::vcovCL`` of that fit, HC3 by default),
the default bandwidth is ``rdrobust::rdbwselect`` on the running variable
(per subgroup for a binary ``z``), and binary covariates are subgroups.

The core idea: standard RD estimates tau = E[Y(1)-Y(0)|X=c]. When treatment
effects vary with covariates Z, we estimate CATE(z) = E[Y(1)-Y(0)|X=c, Z=z]
by fitting a fully interacted local linear model on each side of the cutoff.

References
----------
Calonico, S., Cattaneo, M.D., Farrell, M.H., Palomba, F. and Titiunik, R.
(2025). "Treatment Effect Heterogeneity in Regression Discontinuity Designs."
arXiv preprint arXiv:2503.13696. [@calonico2025treatment]
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult

# ======================================================================
# Citation
# ======================================================================

# Same identifiers as the verified master entry ``calonico2025treatment`` in
# paper.bib (CLAUDE.md §10); ``sp.bibtex("calonico2025treatment")`` returns the
# canonical text.
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


# ======================================================================
# Public API
# ======================================================================


_VCE = ("hc0", "hc1", "hc2", "hc3", "cr1")


def _normalize_vce(vce: Optional[str], cluster: bool) -> str:
    """R ``rdhte:::.rdhte_normalize_vce`` (CR2/CR3 not ported)."""
    if vce is None:
        return "cr1" if cluster else "hc3"
    vce = vce.lower()
    if vce not in ("hc0", "hc1", "hc2", "hc3", "cr0", "cr1", "cr2", "cr3"):
        raise ValueError(f"vce must be one of hc0-hc3 / cr1; got {vce!r}")
    if cluster:
        mapped = {"hc0": "cr1", "hc1": "cr1", "cr0": "cr1", "hc2": "cr2", "hc3": "cr3"}
        vce = mapped.get(vce, vce)
        if vce in ("cr2", "cr3"):
            raise NotImplementedError(
                "Clustered HC2/HC3 (vcovCL types CR2/CR3) are not ported; use "
                "vce='cr1' with cluster=."
            )
        return "cr1"
    return {"cr0": "hc0", "cr1": "hc1", "cr2": "hc2", "cr3": "hc3"}.get(vce, vce)


def _hte_prepare(
    data: pd.DataFrame,
    y: str,
    x: str,
    z: Union[str, List[str], None],
    c: float,
    cluster: Optional[str],
) -> Dict[str, Any]:
    z_cols = [] if z is None else ([z] if isinstance(z, str) else list(z))
    cols = [y, x] + z_cols + ([cluster] if cluster else [])
    for col in cols:
        if col not in data.columns:
            role = "Cluster column" if col == cluster else "Column"
            raise ValueError(f"{role} '{col}' not found in data")
    frame = data[cols].dropna()
    Y = frame[y].to_numpy(dtype=float)
    Xc = frame[x].to_numpy(dtype=float) - c
    Z = frame[z_cols].to_numpy(dtype=float) if z_cols else np.zeros((len(Y), 0))
    cl = frame[cluster].to_numpy() if cluster else None
    # R: a single 0/1 covariate is a factor (subgroups), anything else linear.
    groups = None
    if Z.shape[1] == 1 and np.all(np.isin(Z[:, 0], (0.0, 1.0))):
        groups = Z[:, 0]
    return {"Y": Y, "Xc": Xc, "Z": Z, "cl": cl, "z_cols": z_cols, "groups": groups}


def _bw_select(
    Y: np.ndarray,
    Xc: np.ndarray,
    p: int,
    q: int,
    kernel: str,
    bwselect: str,
    vce: str,
    cl: Optional[np.ndarray],
) -> Tuple[float, float]:
    """``rdrobust::rdbwselect`` on the running variable (StatsPAI port)."""
    from ._cct_bandwidth import cct_bandwidth

    # R passes vce = "cr1" to rdbwselect with a cluster, where it means
    # hc1-type residuals summed within cluster -- the port's "hc1" + cluster.
    bw = cct_bandwidth(
        Y,
        Xc,
        c=0.0,
        p=p,
        q=q,
        kernel=kernel,
        bwselect=bwselect,
        vce="hc1" if vce == "cr1" else vce,
        cluster=cl,
    )
    return float(bw["h_left"]), float(bw["h_right"])


def _hte_bandwidths(
    prep: Dict[str, Any],
    h: Any,
    p: int,
    q: int,
    kernel: str,
    bwselect: str,
    vce: str,
) -> Tuple[np.ndarray, Dict[Any, Tuple[float, float]], str]:
    """Per-observation bandwidth vector and per-level (h_left, h_right)."""
    Xc, groups = prep["Xc"], prep["groups"]
    left = Xc < 0
    levels = [None] if groups is None else sorted(np.unique(groups).tolist())
    h_lev: Dict[Any, Tuple[float, float]] = {}
    if h is not None:
        hl, hr = (float(h), float(h)) if np.ndim(h) == 0 else map(float, h)
        for lv in levels:
            h_lev[lv] = (hl, hr)
        how = "manual"
    else:
        for lv in levels:
            m = np.ones(len(Xc), bool) if lv is None else groups == lv
            cl = prep["cl"][m] if prep["cl"] is not None else None
            h_lev[lv] = _bw_select(prep["Y"][m], Xc[m], p, q, kernel, bwselect, vce, cl)
        how = bwselect
    h_vec = np.empty(len(Xc))
    for lv in levels:
        m = np.ones(len(Xc), bool) if lv is None else groups == lv
        h_vec[m & left] = h_lev[lv][0]
        h_vec[m & ~left] = h_lev[lv][1]
    return h_vec, h_lev, how


def _hte_design(
    Xc: np.ndarray, W: np.ndarray, order: int
) -> Tuple[np.ndarray, List[str]]:
    """Columns of ``model.matrix(Y ~ T * Xp * W)`` (R's order not needed;
    coefficients are looked up by name)."""
    T = (Xc >= 0).astype(float)
    cols, names = [np.ones_like(Xc), T], ["(Intercept)", "T"]
    Xp = [Xc**j for j in range(1, order + 1)]
    for j, v in enumerate(Xp, 1):
        cols.append(v)
        names.append(f"X{j}")
    for k in range(W.shape[1]):
        cols.append(W[:, k])
        names.append(f"W{k}")
    for j, v in enumerate(Xp, 1):
        cols.append(T * v)
        names.append(f"T:X{j}")
    for k in range(W.shape[1]):
        cols.append(T * W[:, k])
        names.append(f"T:W{k}")
    for j, v in enumerate(Xp, 1):
        for k in range(W.shape[1]):
            cols.append(v * W[:, k])
            names.append(f"X{j}:W{k}")
            cols.append(T * v * W[:, k])
            names.append(f"T:X{j}:W{k}")
    return np.column_stack(cols), names


def _hte_fit(
    Y: np.ndarray,
    Xc: np.ndarray,
    W: np.ndarray,
    h_vec: np.ndarray,
    order: int,
    kernel: str,
    vce: str,
    cl: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Weighted ``lm`` on ``|Xc| <= h`` and ``sandwich::vcovCL(type=vce)``."""
    r = np.abs(Xc) <= h_vec
    Xd, names = _hte_design(Xc[r], W[r], order)
    u = Xc[r] / h_vec[r]
    if kernel == "triangular":
        w = 1 - np.abs(u)
    elif kernel == "epanechnikov":
        w = 1 - u**2
    else:
        w = np.ones_like(u)
    y = Y[r]
    XtW = Xd.T * w
    bread = np.linalg.inv(XtW @ Xd)
    beta = bread @ (XtW @ y)
    e = y - Xd @ beta
    psi = Xd * (w * e)[:, None]
    n, k = Xd.shape
    if vce == "cr1":
        _, codes = np.unique(cl[r], return_inverse=True)
        G = int(codes.max()) + 1
        S = np.zeros((G, k))
        np.add.at(S, codes, psi)
        V = bread @ (S.T @ S) @ bread * (G / (G - 1) * (n - 1) / (n - k))
    elif vce in ("hc2", "hc3"):
        lev = w * np.einsum("ij,jk,ik->i", Xd, bread, Xd)
        a = np.sqrt(1 - lev) if vce == "hc2" else (1 - lev)
        P = psi / a[:, None]
        V = bread @ (P.T @ P) @ bread
    else:
        scale = n / (n - 1) if vce == "hc0" else n / (n - k)
        V = bread @ (psi.T @ psi) @ bread * scale
    return beta, V, names, int(r.sum())


def _hte_coefficients(
    beta: np.ndarray, V: np.ndarray, names: List[str], dz: int, subgroup: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """R's reported vector and covariance.

    Continuous ``z``: ``(T, T:W_1, ..., T:W_dz)``. Subgroups: the effect in
    each group, ``T`` for the base group and ``T + T:W`` for the other.
    """
    idx = [names.index("T")] + [names.index(f"T:W{k}") for k in range(dz)]
    b = beta[idx]
    Vs = V[np.ix_(idx, idx)]
    if not subgroup:
        return b, Vs
    A = np.eye(len(idx))
    A[1:, 0] = 1.0
    return A @ b, A @ Vs @ A.T


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
    """
    Conditional average treatment effects (CATE) in sharp RD designs.

    The Python port of R ``rdhte`` (Calonico, Cattaneo, Farrell, Palomba &
    Titiunik). Fits ``Y ~ T * Xp * W`` by kernel-weighted least squares on
    ``|x - c| <= h`` and reports:

    * the conventional coefficients of the order-``p`` fit -- ``T`` and
      ``T:z`` for continuous ``z`` (so ``CATE(z) = T + z'(T:z)``), or the
      effect in each group when ``z`` is a single 0/1 variable;
    * robust bias-corrected inference: the order-``q`` fit on the same
      window, with its ``sandwich::vcovCL`` covariance (HC3 by default).

    ``result.model_info`` carries R's ``coef``, ``coef_bc``, ``se_rb`` and
    ``vcov`` under those names.

    Parameters
    ----------
    data : pd.DataFrame
    y, x : str
        Outcome and running variable.
    z : str or list of str
        Covariate(s) the effect varies with. A single 0/1 column is treated
        as two subgroups (R's factor rule).
    c : float, default 0
        Cutoff.
    p : int, default 1
        Polynomial order of the point estimate.
    h : float or (float, float), optional
        Bandwidth, or ``(h_left, h_right)``. Default: ``rdrobust``'s
        ``rdbwselect`` (``bwselect``) on the running variable -- separately
        in each subgroup for a binary ``z``, as R does.
    b : float, optional
        Bandwidth of the bias-correction fit. Defaults to ``h`` (R has no
        separate ``b``). Through 1.28.0 this argument was accepted and
        ignored.
    kernel : {'triangular', 'epanechnikov', 'uniform'}
    bwselect : str, default 'mserd'
        Any ``rdbwselect`` rule.
    cluster : str, optional
        Cluster variable (CR1).
    alpha : float, default 0.05
    eval_points : array-like, optional
        ``z`` values at which to report ``CATE(z)`` (continuous ``z``).
        Default: ``n_eval`` points from the 10th to the 90th percentile.
    n_eval : int, default 20
    q : int, optional
        Order of the bias-correction fit, default ``p + 1``.
    vce : {'hc0', 'hc1', 'hc2', 'hc3', 'cr1'}, optional
        Default ``'hc3'``, or ``'cr1'`` with ``cluster`` (R's defaults).

    Returns
    -------
    CausalResult
        ``estimate``: the average of ``CATE`` over the evaluation points
        (the groups, for a binary ``z``), with robust bias-corrected
        ``se`` / ``ci`` / ``pvalue``. ``detail``: one row per evaluation
        point with ``cate`` (conventional), ``cate_bc``, ``se`` (robust),
        ``ci_lower``, ``ci_upper``, ``pvalue``.

    Notes
    -----
    Through 1.28.0 inference was conventional (the order-``p`` fit's own
    HC1 SE, no bias correction), the default bandwidth was a rule of thumb
    no reference computes, and binary ``z`` was not treated as subgroups.
    Point estimates at a given ``h`` were already R's.

    References
    ----------
    calonico2025treatment, calonico2025rdhtepackage

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> X = rng.uniform(-1, 1, n)
    >>> Z = rng.normal(0, 1, n)
    >>> tau_z = 2.0 + 1.5 * Z  # CATE varies with Z
    >>> Y = 0.5 * X + tau_z * (X >= 0) + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X, 'z': Z})
    >>> import statspai as sp
    >>> result = sp.rdhte(df, y='y', x='x', z='z', c=0)
    >>> bool(abs(result.estimate - 2.0) < 1.0)  # average CATE near 2
    True
    """
    if kernel not in ("triangular", "uniform", "epanechnikov"):
        raise ValueError(
            f"kernel must be 'triangular', 'uniform', or 'epanechnikov', "
            f"got '{kernel}'"
        )
    if p < 0:
        raise ValueError(f"p must be >= 0, got {p}")
    q = p + 1 if q is None else int(q)
    if q <= p:
        raise ValueError(f"q must exceed p; got q={q}, p={p}")
    vce_n = _normalize_vce(vce, cluster is not None)

    prep = _hte_prepare(data, y, x, z, c, cluster)
    Y, Xc, Z, cl, z_cols = prep["Y"], prep["Xc"], prep["Z"], prep["cl"], prep["z_cols"]
    groups = prep["groups"]
    subgroup = groups is not None
    dz = Z.shape[1]
    if subgroup:
        levels = sorted(np.unique(groups).tolist())
        W = (groups == levels[1]).astype(float).reshape(-1, 1) if len(levels) > 1 else Z
        if len(levels) < 2:
            raise ValueError(f"z '{z_cols[0]}' has a single value; nothing to compare")
    else:
        W = Z

    h_vec, h_lev, bw_how = _hte_bandwidths(prep, h, p, q, kernel, bwselect, vce_n)
    b_vec = h_vec if b is None else np.full(len(Xc), float(b))

    n_left, n_right = int((Xc < 0).sum()), int((Xc >= 0).sum())
    for vec, order, lab in ((h_vec, p, "h"), (b_vec, q, "b")):
        n_in = int((np.abs(Xc) <= vec).sum())
        if n_in <= len(_hte_design(Xc[:1], W[:1], order)[1]):
            raise ValueError(
                f"Not enough observations inside the {lab} window ({n_in}) for "
                f"the order-{order} interacted fit"
            )
    beta_p, _, names_p, n_eff = _hte_fit(Y, Xc, W, h_vec, p, kernel, vce_n, cl)
    beta_q, V_q, names_q, _ = _hte_fit(Y, Xc, W, b_vec, q, kernel, vce_n, cl)
    coef, _ = _hte_coefficients(
        beta_p, np.zeros((len(beta_p),) * 2), names_p, W.shape[1], subgroup
    )
    coef_bc, vcov = _hte_coefficients(beta_q, V_q, names_q, W.shape[1], subgroup)
    se_rb = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # ---- evaluation points / subgroup rows ---------------------------------
    if subgroup:
        L = np.eye(len(coef))
        z_display: Any = levels
    else:
        if eval_points is not None:
            ev = np.asarray(eval_points, dtype=float)
            ev = ev.reshape(-1, dz) if dz == 1 else np.atleast_2d(ev)
            if ev.shape[1] != dz:
                raise ValueError(f"eval_points must have {dz} columns, got {ev.shape}")
        elif dz == 1:
            ev = np.percentile(Z[:, 0], np.linspace(10, 90, n_eval)).reshape(-1, 1)
        else:
            pct = np.linspace(10, 90, max(int(n_eval ** (1 / dz)), 3))
            grids = [np.percentile(Z[:, j], pct) for j in range(dz)]
            ev = np.column_stack(
                [m.ravel() for m in np.meshgrid(*grids, indexing="ij")]
            )
            if len(ev) > n_eval:
                ev = ev[np.round(np.linspace(0, len(ev) - 1, n_eval)).astype(int)]
        L = np.column_stack([np.ones(len(ev)), ev])
        z_display = ev[:, 0] if dz == 1 else [tuple(r) for r in ev]
    cate = L @ coef
    cate_bc = L @ coef_bc
    se_pts = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", L, vcov, L), 0.0))
    t_pts = np.where(se_pts > 0, cate_bc / np.where(se_pts > 0, se_pts, 1.0), np.nan)
    detail = pd.DataFrame(
        {
            "z_value": z_display,
            "cate": cate,
            "cate_bc": cate_bc,
            "se": se_pts,
            "ci_lower": cate_bc - z_crit * se_pts,
            "ci_upper": cate_bc + z_crit * se_pts,
            "pvalue": 2 * stats.norm.sf(np.abs(t_pts)),
        }
    )

    # ---- headline: average over the rows -------------------------------------
    lbar = L.mean(axis=0)
    ate = float(lbar @ coef)
    ate_bc = float(lbar @ coef_bc)
    ate_se = float(np.sqrt(max(lbar @ vcov @ lbar, 0.0)))
    ate_pv = (
        float(2 * stats.norm.sf(abs(ate_bc / ate_se))) if ate_se > 0 else float("nan")
    )
    ate_ci = (ate_bc - z_crit * ate_se, ate_bc + z_crit * ate_se)

    # ---- heterogeneity: joint Wald on the bias-corrected contrasts ------------
    if subgroup:
        R_ = np.column_stack([-np.ones(len(coef) - 1), np.eye(len(coef) - 1)])
    else:
        R_ = np.column_stack([np.zeros(dz), np.eye(dz)])
    rb = R_ @ coef_bc
    wald = float(rb @ np.linalg.pinv(R_ @ vcov @ R_.T) @ rb)
    het_test = {
        "statistic": wald,
        "pvalue": float(stats.chi2.sf(wald, df=R_.shape[0])),
        "df": int(R_.shape[0]),
    }

    names_out = (
        [f"{z_cols[0]}={lv:g}" for lv in levels]
        if subgroup
        else ["T"] + [f"T:{zc}" for zc in z_cols]
    )
    hl_list = list(h_lev.values())
    model_info: Dict[str, Any] = {
        "rd_type": "Sharp",
        "mode": "subgroups" if subgroup else "continuous",
        "polynomial_p": p,
        "polynomial_q": q,
        "kernel": kernel,
        "vce": vce_n,
        "bandwidth_h": (
            hl_list[0][0]
            if len(set(hl_list)) == 1 and hl_list[0][0] == hl_list[0][1]
            else h_lev
        ),
        "bandwidth_by_level": {str(k): v for k, v in h_lev.items()},
        "bandwidth_b": float(b) if b is not None else "h",
        "bwselect": bw_how,
        "cutoff": c,
        "z_covariates": z_cols,
        "n_z": dz,
        "n_left": n_left,
        "n_right": n_right,
        "n_effective": n_eff,
        "n_eval_points": len(detail),
        "coef_names": names_out,
        "coef": coef.tolist(),
        "coef_bc": coef_bc.tolist(),
        "se_rb": se_rb.tolist(),
        "vcov": vcov.tolist(),
        "ate": ate,
        "ate_bc": ate_bc,
        "ate_se": ate_se,
        "ate_pvalue": ate_pv,
        "ate_ci": ate_ci,
        "heterogeneity_test": het_test,
        "_L": L.tolist(),
    }

    result = CausalResult(
        method="RD Heterogeneous Treatment Effects (rdhte)",
        estimand="CATE",
        estimate=ate,
        se=ate_se,
        pvalue=ate_pv,
        ci=ate_ci,
        alpha=alpha,
        n_obs=len(Y),
        detail=detail,
        model_info=model_info,
        _citation_key="rdhte",
    )
    setattr(result, "plot", lambda **kw: _rdhte_plot(result, **kw))
    return result


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
    """
    Bandwidth selection for :func:`rdhte`, as R ``rdhte::rdbwhte``.

    For continuous ``z`` this is ``rdrobust::rdbwselect`` on the running
    variable (the covariates do not enter); for a single 0/1 ``z`` it is
    ``rdbwselect`` within each subgroup.

    Returns
    -------
    float or pd.DataFrame
        The common bandwidth when there is one level and ``h_left ==
        h_right``; otherwise a frame with columns ``level``, ``h_left``,
        ``h_right``.

    Notes
    -----
    Through 1.28.0 this inflated the local-linear bandwidth by a
    parameter-count factor and refined it with a pilot rule of thumb --
    a quantity no reference computes.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.uniform(-1, 1, n)
    >>> z = rng.normal(size=n)
    >>> y = (0.8 * (x >= 0) + 0.3 * z * (x >= 0) + 0.5 * x
    ...      + rng.normal(0, 0.3, n))
    >>> df = pd.DataFrame({"x": x, "y": y, "z": z})
    >>> h = sp.rdbwhte(df, y="y", x="x", z="z", c=0.0)
    >>> bool(h > 0)
    True
    """
    q = p + 1 if q is None else int(q)
    vce_n = _normalize_vce(vce, cluster is not None)
    prep = _hte_prepare(data, y, x, z, c, cluster)
    _, h_lev, _ = _hte_bandwidths(prep, None, p, q, kernel, bwselect, vce_n)
    if len(h_lev) == 1:
        hl, hr = next(iter(h_lev.values()))
        if hl == hr:
            return hl
    return pd.DataFrame(
        [{"level": k, "h_left": v[0], "h_right": v[1]} for k, v in h_lev.items()]
    )


def rdhte_lincom(
    result: CausalResult,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    linfct: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Linear combinations of :func:`rdhte` effects, as R ``rdhte_lincom``.

    Either ``linfct`` -- a matrix (or vector) over R's coefficient vector
    ``model_info['coef']`` (``T, T:z...`` or the group effects) -- or
    ``weights`` over the rows of ``result.detail``. Each row's estimate is
    ``L coef`` (conventional); its z-statistic, p-value and interval use
    ``L coef_bc`` and ``sqrt(L V L')`` (robust bias-corrected), and the rows
    are tested jointly by a Wald chi-square, as ``rdhte_lincom`` does.

    Returns
    -------
    dict
        ``estimate``, ``se``, ``ci``, ``pvalue`` (scalars for one row,
        arrays otherwise) and ``joint`` (``statistic``, ``df``, ``pvalue``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> X = rng.uniform(-1, 1, n)
    >>> Z = rng.normal(0, 1, n)
    >>> Y = 0.5 * X + (2.0 + 1.5 * Z) * (X >= 0) + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({'y': Y, 'x': X, 'z': Z})
    >>> result = sp.rdhte(df, y='y', x='x', z='z', c=0)
    >>> out = sp.rdhte_lincom(result, linfct=[1.0, 1.0])  # CATE at z = 1
    >>> bool(abs(out["estimate"] - 3.5) < 0.5)
    True
    """
    mi = result.model_info
    coef = np.asarray(mi["coef"], dtype=float)
    coef_bc = np.asarray(mi["coef_bc"], dtype=float)
    V = np.asarray(mi["vcov"], dtype=float)
    if linfct is None:
        if weights is None:
            raise ValueError("pass linfct= (over coefficients) or weights= (over rows)")
        wts = np.asarray(weights, dtype=float)
        L_rows = np.asarray(mi["_L"], dtype=float)
        if len(wts) != len(L_rows):
            raise ValueError(
                f"weights has length {len(wts)}, expected {len(L_rows)} "
                "(number of rows in result.detail)"
            )
        L = (wts @ L_rows).reshape(1, -1)
    else:
        L = np.atleast_2d(np.asarray(linfct, dtype=float))
        if L.shape[1] != len(coef):
            raise ValueError(f"linfct needs {len(coef)} columns ({mi['coef_names']})")
    est = L @ coef
    est_bc = L @ coef_bc
    LVL = L @ V @ L.T
    se = np.sqrt(np.maximum(np.diag(LVL), 0.0))
    zc = stats.norm.ppf(1 - alpha / 2)
    tstat = est_bc / se
    pv = 2 * stats.norm.sf(np.abs(tstat))
    wald = float(est_bc @ np.linalg.pinv(LVL) @ est_bc)
    one = L.shape[0] == 1

    def _s(a):
        return float(a[0]) if one else a

    return {
        "estimate": _s(est),
        "estimate_bc": _s(est_bc),
        "se": _s(se),
        "z": _s(tstat),
        "ci": (
            (float(est_bc[0] - zc * se[0]), float(est_bc[0] + zc * se[0]))
            if one
            else np.column_stack([est_bc - zc * se, est_bc + zc * se])
        ),
        "pvalue": _s(pv),
        "joint": {
            "statistic": wald,
            "df": int(L.shape[0]),
            "pvalue": float(stats.chi2.sf(wald, df=L.shape[0])),
        },
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
    """
    Plot CATE(z) vs z with confidence bands.

    Parameters
    ----------
    result : CausalResult
        Result from rdhte().
    ax : matplotlib Axes, optional
        Axes to plot on. If None, a new figure is created.
    ci_alpha : float, default 0.2
        Transparency for confidence band shading.
    cate_color : str, default '#2171B5'
        Color for the CATE line.
    ate_color : str, default '#CB181D'
        Color for the average treatment effect line.
    zero_color : str, default 'gray'
        Color for the zero-effect line.
    xlabel : str, optional
        X-axis label. Defaults to covariate name(s).
    ylabel : str, default 'CATE'
        Y-axis label.
    title : str, optional
        Plot title.
    figsize : tuple, default (8, 5)
        Figure size.

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. "  # pragma: no cover
            "Install it with: pip install matplotlib"
        )

    detail = result.detail
    if detail is None:
        raise ValueError("rdhte plot requires result.detail.")
    mi = result.model_info
    dz = mi["n_z"]

    if dz > 1:
        raise NotImplementedError(  # pragma: no cover
            "Plotting is only supported for scalar Z (dim=1). "
            "For multivariate Z, construct custom plots from result.detail."
        )

    z_vals = detail["z_value"].values.astype(float)
    cate = detail["cate"].values
    ci_lo = detail["ci_lower"].values
    ci_hi = detail["ci_upper"].values
    ate = mi["ate"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Sort by z for smooth line
    order = np.argsort(z_vals)
    z_sorted = z_vals[order]
    cate_sorted = cate[order]
    ci_lo_sorted = ci_lo[order]
    ci_hi_sorted = ci_hi[order]

    # CATE line + CI band
    ax.plot(z_sorted, cate_sorted, color=cate_color, linewidth=2, label="CATE(z)")
    ax.fill_between(
        z_sorted,
        ci_lo_sorted,
        ci_hi_sorted,
        color=cate_color,
        alpha=ci_alpha,
        label=f"{int((1 - result.alpha) * 100)}% CI",
    )

    # Horizontal lines
    ax.axhline(y=0, color=zero_color, linestyle="--", linewidth=1, label="Zero effect")
    ax.axhline(
        y=ate, color=ate_color, linestyle="-.", linewidth=1.5, label=f"ATE = {ate:.3f}"
    )

    # Labels
    z_names = mi.get("z_covariates", ["Z"])
    ax.set_xlabel(xlabel or z_names[0])
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Heterogeneous Treatment Effects in RD")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    return ax
