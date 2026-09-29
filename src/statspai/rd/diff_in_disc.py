"""Difference-in-discontinuities (Grembi, Nannicini and Troiano 2016).

Two cross-sections of a regression-discontinuity design -- before and after
a policy that switches on at the cutoff only in the post period -- and the
effect is the *change* in the discontinuity. The estimator is one local
polynomial regression on the pooled sample within the bandwidth, fully
interacted in side of the cutoff and period, with kernel weights; its
``post x above`` coefficient is the difference in discontinuities (it equals
the post-period local-linear RD minus the pre-period one fitted with the same
bandwidth and kernel). Pooling is what gives the standard error: clustered
by the unit or site, it carries the covariance between the two periods'
estimates that a difference of two separate ``rdrobust`` runs ignores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility


def _kernel(u: np.ndarray, kernel: str) -> np.ndarray:
    a = np.abs(u)
    if kernel == "triangular":
        return np.where(a <= 1, 1 - a, 0.0)
    if kernel == "uniform":
        return np.where(a <= 1, 1.0, 0.0)
    if kernel == "epanechnikov":
        return np.where(a <= 1, 0.75 * (1 - a**2), 0.0)
    raise MethodIncompatibility(
        f"kernel={kernel!r}; use 'triangular', 'uniform' or 'epanechnikov'."
    )


@accepts_aliases(covs="covariates")
def rd_diff_in_disc(
    data: pd.DataFrame,
    y: str,
    x: str,
    post: str,
    c: float = 0.0,
    h: Optional[float] = None,
    p: int = 1,
    kernel: str = "triangular",
    cluster: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """Difference-in-discontinuities estimate with a pooled local polynomial.

    Parameters
    ----------
    data : pd.DataFrame
        Pooled pre- and post-period observations.
    y, x : str
        Outcome and running variable.
    post : str
        0/1 period indicator (1 = the period in which the cutoff policy is
        active).
    c : float, default 0
        Cutoff; observations with ``x >= c`` are above it.
    h : float, optional
        Bandwidth, common to both sides and periods. Default: the smaller of
        the two periods' MSE-optimal ``rdbwselect`` bandwidths (``mserd``) --
        a conservative choice; pass ``h`` to set it.
    p : int, default 1
        Polynomial order on each side and period.
    kernel : {'triangular', 'uniform', 'epanechnikov'}
    cluster : str, optional
        Cluster the standard error (e.g. by unit or site, which lets the two
        periods' errors correlate): CR1 as Stata ``regress [aw=], vce(cluster
        c)``. Otherwise HC1 (``vce(robust)``).
    covariates : list of str, optional
        Additive covariates in the pooled regression (``covs=`` is accepted,
        as in ``rdrobust``).
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``estimate`` is the change in the discontinuity; ``detail`` has the
        pre- and post-period discontinuities; ``model_info`` the bandwidth
        and sample sizes.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> d = pd.DataFrame({"x": rng.uniform(-1, 1, n), "post": rng.integers(0, 2, n)})
    >>> d["y"] = d.x + 0.3 * (d.x >= 0) + 0.5 * (d.x >= 0) * d.post + rng.normal(0, 0.3, n)
    >>> r = sp.rd_diff_in_disc(d, y="y", x="x", post="post", h=0.5)
    >>> round(r.estimate, 1)
    0.5

    References
    ----------
    Grembi, V., Nannicini, T. and Troiano, U. (2016). "Do Fiscal Rules Matter?"
    *American Economic Journal: Applied Economics*, 8(3), 1-30.
    """
    covs = list(covariates or [])
    cols = [y, x, post] + covs + ([cluster] if cluster else [])
    missing = [col for col in cols if col not in data.columns]
    if missing:
        raise MethodIncompatibility(f"Columns not found in data: {missing}")
    df = data[cols].dropna().reset_index(drop=True)
    P = pd.to_numeric(df[post], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isin(P, (0.0, 1.0))) or len(np.unique(P)) < 2:
        raise MethodIncompatibility(f"'{post}' must be 0/1 with both periods present.")
    if int(p) < 0:
        raise MethodIncompatibility("p must be >= 0.")
    p = int(p)
    X = df[x].to_numpy(dtype=float) - float(c)
    Y = df[y].to_numpy(dtype=float)

    if h is None:
        from .bandwidth import rdbwselect

        hs = []
        for per in (0.0, 1.0):
            sub = df.loc[P == per]
            bw = rdbwselect(sub, y=y, x=x, c=c, p=p, kernel=kernel)
            hs.append(float(bw["h_left"].iloc[0]))
        h = min(hs)
        h_rule = "min(mserd pre, mserd post)"
    else:
        h = float(h)
        if not h > 0:
            raise MethodIncompatibility("h must be positive.")
        h_rule = "user"

    w = _kernel(X / h, kernel)
    keep = w > 0
    X, Y, P, w = X[keep], Y[keep], P[keep], w[keep]
    T = (X >= 0).astype(float)
    sub = df.loc[keep].reset_index(drop=True)

    # fully interacted local polynomial: [1, T, x^j, T x^j] x [1, post]
    base = [np.ones_like(X), T]
    names = ["const", "above"]
    for j in range(1, p + 1):
        base += [X**j, T * X**j]
        names += [f"x^{j}", f"above*x^{j}"]
    B = np.column_stack(base)
    D = np.column_stack([B, B * P[:, None]])
    names = names + [f"post*{nm}" for nm in names]
    if covs:
        D = np.column_stack([D, sub[covs].to_numpy(dtype=float)])
        names += covs
    k = D.shape[1]
    n = len(Y)
    idx = names.index("post*above")

    sw = np.sqrt(w)
    Dw = D * sw[:, None]
    XtX_inv = np.linalg.inv(Dw.T @ Dw)
    beta = XtX_inv @ (Dw.T @ (Y * sw))
    e = Y - D @ beta
    sc = D * (w * e)[:, None]
    if cluster:
        codes = pd.factorize(sub[cluster])[0]
        G = int(codes.max()) + 1
        S = np.zeros((G, k))
        np.add.at(S, codes, sc)
        meat = S.T @ S
        V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtX_inv @ meat @ XtX_inv
        dof = G - 1
        vce = f"cluster({cluster})"
    else:
        V = (n / (n - k)) * XtX_inv @ (sc.T @ sc) @ XtX_inv
        dof = n - k
        vce = "HC1"
    est = float(beta[idx])
    se = float(np.sqrt(V[idx, idx]))
    tval = est / se if se > 0 else np.nan
    pval = float(2 * stats.t.sf(abs(tval), dof))
    q = stats.t.ppf(1 - alpha / 2, dof)

    pre_rd = float(beta[names.index("above")])
    post_rd = pre_rd + est
    rows: List[Dict[str, Any]] = [
        {
            "period": "pre",
            "discontinuity": pre_rd,
            "n_left": int(((P == 0) & (T == 0)).sum()),
            "n_right": int(((P == 0) & (T == 1)).sum()),
        },
        {
            "period": "post",
            "discontinuity": post_rd,
            "n_left": int(((P == 1) & (T == 0)).sum()),
            "n_right": int(((P == 1) & (T == 1)).sum()),
        },
    ]
    return CausalResult(
        method="Difference-in-discontinuities (Grembi et al. 2016)",
        estimand="Change in the discontinuity",
        estimate=est,
        se=se,
        pvalue=pval,
        ci=(est - q * se, est + q * se),
        alpha=alpha,
        n_obs=n,
        detail=pd.DataFrame(rows),
        model_info={
            "bandwidth": h,
            "bandwidth_rule": h_rule,
            "polynomial_p": p,
            "kernel": kernel,
            "cutoff": c,
            "vce": vce,
            "coefficients": pd.Series(beta, index=names),
            "df_inference": dof,
        },
    )
