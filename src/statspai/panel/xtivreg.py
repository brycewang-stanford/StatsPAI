"""Instrumental variables for panel data (Stata ``xtivreg``).

``y_it = Z_it d + mu_i + nu_it`` with ``Z = [Y, X1]`` (endogenous and
exogenous regressors) and instruments ``X = [X1, X2]``:

* ``method='fe'`` -- 2SLS on the within transform ``w - w_i. + w..``; the
  conventional variance carries ``(N - K) / (N - n - K + 1)`` for the
  ``n - 1`` absorbed means;
* ``method='be'`` -- 2SLS on the panel means, each mean entering ``T_i``
  times;
* ``method='re'`` -- feasible GLS. The variance components come from the
  within and between 2SLS residuals (Stata's extension of Swamy-Arora to
  unbalanced panels; ``nosa=True`` for the Baltagi-Chang estimators), every
  variable is quasi-demeaned ``w - theta_i w_i.`` with
  ``theta_i = 1 - sqrt(s2_nu / (T_i s2_mu + s2_nu))``, and the GLS
  instruments are the quasi-demeaned ``X`` (G2SLS, Balestra and
  Varadharajan-Krishnakumar, the default) or ``X`` within-transformed plus
  its panel means (``ec2sls=True``, Baltagi's EC2SLS).

The formulas are those of Stata's documentation for ``xtivreg``; the
evidence is ``tests/test_xtivreg_stata.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["xtivreg"]


def _as_list(v: Union[str, Sequence[str], None]) -> List[str]:
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def _gmean(values: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    total = np.zeros((n,) + values.shape[1:])
    np.add.at(total, codes, values)
    cnt = np.bincount(codes, minlength=n).astype(float)
    return total / cnt.reshape((-1,) + (1,) * (values.ndim - 1))


def _span(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Columns of ``A`` with the linearly dependent ones removed."""
    scale = np.sqrt((A * A).sum(axis=0))
    keep: List[int] = []
    basis = np.zeros((A.shape[0], 0))
    for j in range(A.shape[1]):
        if scale[j] == 0:
            continue
        v = A[:, j] / scale[j]
        if basis.shape[1]:
            v = v - basis @ (basis.T @ v)
        nv = np.linalg.norm(v)
        if nv > tol:
            keep.append(j)
            basis = np.column_stack([basis, v / nv])
    return A[:, keep]


def _tsls(y: np.ndarray, Z: np.ndarray, X: np.ndarray) -> Dict[str, np.ndarray]:
    """2SLS of ``y`` on ``Z`` with instruments ``X``."""
    X = _span(X)
    if X.shape[1] < Z.shape[1]:
        raise MethodIncompatibility(
            "xtivreg: fewer independent instruments than regressors "
            f"({X.shape[1]} < {Z.shape[1]}); the equation is not identified.",
            recovery_hint="Add excluded instruments, at least one per "
            "endogenous regressor.",
        )
    Zhat = X @ np.linalg.lstsq(X, Z, rcond=None)[0]
    A = Zhat.T @ Zhat
    try:
        bread = np.linalg.inv(A)
    except np.linalg.LinAlgError as exc:
        raise MethodIncompatibility(
            "xtivreg: the regressors are collinear once projected on the "
            "instruments.",
            recovery_hint="Drop a regressor that is a combination of others "
            "or constant within panels (with method='fe').",
        ) from exc
    d = bread @ (Zhat.T @ y)
    return {"d": d, "bread": bread, "e": y - Z @ d, "Zhat": Zhat}


def _cluster_vcov(
    fit: Dict[str, np.ndarray], codes: np.ndarray, n_groups: int, N: int, K: int
) -> np.ndarray:
    if n_groups < 2:
        raise DataInsufficient(
            f"xtivreg: the cluster variable has {n_groups} group; a "
            "cluster-robust variance needs at least two.",
            recovery_hint="Cluster on a variable with two or more groups, or "
            "use vce='robust'.",
        )
    S = fit["Zhat"] * fit["e"][:, None]
    G = np.zeros((n_groups, S.shape[1]))
    np.add.at(G, codes, S)
    meat = G.T @ G
    adj = n_groups / (n_groups - 1) * (N - 1) / (N - K)
    return np.asarray(adj * fit["bread"] @ meat @ fit["bread"])


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


@records_call("xtivreg")
@accepts_aliases(entity="id", robust="vce", covariates="x")
def xtivreg(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str], None] = None,
    endog: Union[str, Sequence[str], None] = None,
    instruments: Union[str, Sequence[str], None] = None,
    id: Optional[str] = None,
    method: str = "re",
    ec2sls: bool = False,
    nosa: bool = False,
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Instrumental-variables regression for panel data.

    Equivalent to Stata's ``xtivreg y x (endog = instruments), re | fe |
    be`` with ``ec2sls`` / ``nosa``. The random-effects estimator is
    feasible-GLS 2SLS: G2SLS (Balestra and Varadharajan-Krishnakumar) by
    default, Baltagi's EC2SLS with ``ec2sls=True``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Dependent variable.
    x : str or list of str, optional
        Exogenous regressors (they instrument themselves). A constant is
        added.
    endog : str or list of str
        Endogenous regressors.
    instruments : str or list of str
        Excluded instruments, at least as many as ``endog``.
    id : str
        Panel identifier.
    method : {'re', 'fe', 'be'}, default 're'
        Random effects (feasible-GLS 2SLS, Stata's default), fixed effects
        (within 2SLS) or between effects (2SLS on the panel means).
    ec2sls : bool, default False
        ``method='re'`` only: Baltagi's EC2SLS instruments (within and
        between parts of the instruments) instead of G2SLS.
    nosa : bool, default False
        ``method='re'`` only: the Baltagi-Chang variance components instead
        of the Swamy-Arora ones with a degrees-of-freedom correction.
    vce : {None, 'robust', 'cluster'}, optional
        Conventional (default) or cluster-robust variance. ``'robust'``
        clusters on the panel identifier, as Stata does after ``xtivreg``.
    cluster : str, optional
        Cluster column (default: ``id``).
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        Coefficients with z inference (Stata's default). ``model_info``
        holds ``sigma_u``, ``sigma_e``, ``rho``, ``theta`` (per-panel
        values summarised as min / median / max when the panel is
        unbalanced), the within / between / overall R-squared and the
        number of panels.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> ids = np.repeat(np.arange(100), 5)
    >>> mu = rng.normal(size=100)[ids]
    >>> z = rng.normal(size=500)
    >>> v = rng.normal(size=500)
    >>> d = z + 0.5 * v + rng.normal(size=500)
    >>> df = pd.DataFrame({"id": ids, "z": z, "d": d,
    ...                    "x": rng.normal(size=500)})
    >>> df["y"] = 1 + 0.5 * df["d"] + df["x"] + mu + v
    >>> fit = sp.xtivreg(df, y="y", x="x", endog="d", instruments="z", id="id")
    >>> round(float(fit.params["d"]), 1)  # doctest: +SKIP
    0.5

    References
    ----------
    balestra1987full
    baltagi1981simultaneous
    """
    from ..core._vcov_spec import parse_se_request

    m = str(method).lower()
    if m not in ("re", "fe", "be"):
        raise MethodIncompatibility(
            f"xtivreg: method must be 're', 'fe' or 'be'; got {method!r}.",
        )
    if m != "re" and (ec2sls or nosa):
        raise MethodIncompatibility(
            "xtivreg: ec2sls= and nosa= apply to method='re' only.",
        )
    if id is None:
        raise MethodIncompatibility(
            "xtivreg: id= names the panel identifier.",
            recovery_hint="Pass id='<panel column>'.",
        )
    xs, en, iv = _as_list(x), _as_list(endog), _as_list(instruments)
    req = parse_se_request(
        vce, cluster, function="xtivreg", supported=("nonrobust", "robust", "cluster")
    )
    se_kind, cluster = req.kind, req.cluster
    if se_kind == "robust":
        se_kind, cluster = "cluster", id
    if se_kind == "cluster" and cluster is None:
        cluster = id
    if len(iv) < len(en):
        raise MethodIncompatibility(
            f"xtivreg: {len(en)} endogenous regressor(s) need at least as many "
            f"excluded instruments; got {len(iv)}.",
        )
    check_columns(
        data,
        {"y": y, "x": xs, "endog": en, "instruments": iv, "id": id, "cluster": cluster},
        function="xtivreg",
    )
    cols = list(dict.fromkeys([y, id] + xs + en + iv + ([cluster] if cluster else [])))
    df = data.loc[data[cols].notna().all(axis=1)]
    codes, uniq = pd.factorize(df[id], sort=True)
    n = len(uniq)
    N = len(df)
    Ti = np.bincount(codes, minlength=n).astype(float)
    yv = df[y].to_numpy(dtype=float)
    Zn = en + xs  # Stata orders the endogenous regressors first
    Z = np.column_stack([df[v].to_numpy(dtype=float) for v in Zn] + [np.ones(N)])
    X = np.column_stack([df[v].to_numpy(dtype=float) for v in xs + iv] + [np.ones(N)])
    names = Zn + ["_cons"]
    if n < 2:
        raise DataInsufficient("xtivreg: needs at least two panels.")
    ybar, Zbar, Xbar = (_gmean(a, codes, n) for a in (yv, Z, X))

    def within(a: np.ndarray, abar: np.ndarray) -> np.ndarray:
        return np.asarray(a - abar[codes] + a.mean(axis=0))

    info: Dict[str, Any] = {"method": m, "n_groups": n, "id": id}
    if m in ("fe", "re"):
        yw, Zw, Xw = within(yv, ybar), within(Z, Zbar), within(X, Xbar)
        keep = [
            j
            for j in range(Z.shape[1])
            if j == Z.shape[1] - 1 or np.ptp(Z[:, j] - Zbar[codes, j]) > 1e-12
        ]
        if m == "fe":
            dropped = [names[j] for j in range(len(names)) if j not in keep]
            Z, Zw, Zbar = Z[:, keep], Zw[:, keep], Zbar[:, keep]
            names = [names[j] for j in keep]
        fw = _tsls(yw, Zw[:, keep] if m == "re" else Zw, Xw)
        # K counts every regressor of the model being fitted: after
        # method='re' that includes the time-invariant ones the within
        # regression cannot use (Stata's xtivreg, re does the same).
        Kw = len(keep) if m == "fe" else Z.shape[1]
        s2_nu = float(fw["e"] @ fw["e"]) / (N - n - Kw + 1)
    K = Z.shape[1]
    if m == "fe":
        fit = fw
        V = s2_nu * fit["bread"]
        if se_kind == "cluster":
            ccodes, _ = pd.factorize(df[cluster], sort=True)
            G = int(ccodes.max()) + 1
            V = _cluster_vcov(fit, ccodes, G, N, K)
            info["n_clusters"] = G
        e_u = ybar - Zbar @ fit["d"]  # mu_i up to the constant
        sigma_u = float(np.std(e_u, ddof=1))  # across panels, as Stata
        info.update(
            omitted=dropped,
            sigma_e=float(np.sqrt(s2_nu)),
            sigma_u=sigma_u,
            rho=float(sigma_u**2 / (sigma_u**2 + s2_nu)),
        )
    elif m == "be":
        # one row per panel: Stata's xtivreg, be does not weight the means
        # by T_i (its manual says each mean enters T_i times; the numbers
        # are those of the unweighted regression)
        fit = _tsls(ybar, Zbar, Xbar)
        s2 = float(fit["e"] @ fit["e"]) / (n - K)
        V = s2 * fit["bread"]
        if se_kind == "cluster":
            raise MethodIncompatibility(
                "xtivreg: method='be' has the conventional variance only.",
            )
    else:
        # between 2SLS for sigma_mu
        fb = _tsls(ybar, Zbar, Xbar)  # unweighted, as method='be'
        ub = ybar - Zbar @ fb["d"]
        ssb = float(np.sum(Ti * ub * ub))
        if nosa:
            s2_nu = float(fw["e"] @ fw["e"]) / (N - n)
            s2_mu = (ssb - n * s2_nu) / N
        else:
            ZZ = (Zbar * Ti[:, None]).T @ Zbar
            ZmZ = (Zbar * (Ti * Ti)[:, None]).T @ Zbar
            r = float(np.trace(np.linalg.solve(ZZ, ZmZ)))
            s2_mu = (ssb - (n - K) * s2_nu) / (N - r)
        s2_mu = max(s2_mu, 0.0)
        theta = 1.0 - np.sqrt(s2_nu / (Ti * s2_mu + s2_nu))
        th = theta[codes]
        ys = yv - th * ybar[codes]
        Zs = Z - th[:, None] * Zbar[codes]
        Xs = X - th[:, None] * Xbar[codes]
        if ec2sls:
            # Baltagi's EC2SLS instruments: the within part and the panel
            # means of X. Stata adds the GLS-transformed constant
            # (1 - theta_i), a separate column from the constant mean (1)
            # on an unbalanced panel; with it the estimates match
            # xtivreg, ec2sls to 1e-14 (found by testing instrument sets
            # against its output; without it they differ by ~2e-3).
            Xg = np.column_stack([(X - Xbar[codes])[:, :-1], Xbar[codes], 1.0 - th])
        else:
            Xg = Xs
        fit = _tsls(ys, Zs, Xg)
        s2 = float(fit["e"] @ fit["e"]) / (N - K)
        V = s2 * fit["bread"]
        if se_kind == "cluster":
            ccodes, _ = pd.factorize(df[cluster], sort=True)
            G = int(ccodes.max()) + 1
            V = _cluster_vcov(fit, ccodes, G, N, K)
            info["n_clusters"] = G
        info.update(
            sigma_u=float(np.sqrt(s2_mu)),
            sigma_e=float(np.sqrt(s2_nu)),
            rho=float(s2_mu / (s2_mu + s2_nu)) if s2_mu + s2_nu > 0 else np.nan,
            theta=(
                float(theta[0])
                if np.ptp(theta) < 1e-12
                else {
                    "min": float(theta.min()),
                    "median": float(np.median(theta)),
                    "max": float(theta.max()),
                }
            ),
            estimator="EC2SLS" if ec2sls else "G2SLS",
            variance_components="Baltagi-Chang" if nosa else "Swamy-Arora",
        )
    d = fit["d"]
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    xb = Z @ d
    yd = yv - ybar[codes]
    r2_w = _r2((Z - Zbar[codes]) @ d, yd)
    r2_b = _r2(Zbar @ d, ybar)
    if m == "fe":  # the R-squared of the mean-deviated regression
        e = yd - (Z - Zbar[codes]) @ d
        r2_w = float(1.0 - (e @ e) / (yd @ yd))
    if m == "be":  # the R-squared of the between regression
        e = ybar - Zbar @ d
        r2_b = float(1.0 - (e @ e) / np.sum((ybar - ybar.mean()) ** 2))
    info.update(
        r2_within=r2_w,
        r2_between=r2_b,
        r2_overall=_r2(xb, yv),
        vce=se_kind,
        cluster=cluster if se_kind == "cluster" else None,
        endog=en,
        instruments=iv,
        alpha=alpha,
        model_type="Panel IV",
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = d / se
    params = pd.Series(d, index=names)
    res = EconometricResults(
        params=params,
        std_errors=pd.Series(se, index=names),
        model_info=info,
        data_info={
            "nobs": N,
            "n_obs": N,
            "dependent_var": y,
            "var_cov": V,
            "var_names": names,
            "df_resid": N - K,
            "inference": "z",
        },
        diagnostics={
            "z": dict(zip(names, zstat)),
            "p": dict(zip(names, 2 * stats.norm.sf(np.abs(zstat)))),
        },
    )
    return res
