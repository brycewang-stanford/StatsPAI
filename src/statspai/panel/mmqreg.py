"""Method-of-moments quantile regression (MMQR) with absorbed fixed effects.

Machado and Santos Silva's (2019) location-scale model

    y = x'b + a_g + (x'g + d_g) * eps,   eps independent of x,

whose conditional quantiles are ``x'(b + q(tau) g) + a_g + q(tau) d_g``
with ``q(tau)`` the ``tau``-quantile of ``eps``. Three steps, each a moment
condition: the location coefficients ``b`` by least squares, the scale
coefficients ``g`` by least squares of the absolute location residual on
the same regressors, and ``q(tau)`` as the sample ``tau``-quantile of the
standardized residuals ``nu / sigma``. Fixed effects (one or several sets)
are partialled out and the variables recentred on their means, as in
Rios-Avila, Siles and Canavire-Bacarreza (2024), so that the regressions
keep a constant.

Standard errors come from the influence functions of the stacked moment
conditions (Rios-Avila, Siles and Canavire-Bacarreza 2024): ``'robust'``
(the sum of squared influence functions), ``'cluster'`` (summed within
clusters first) and ``'gls'``, the estimator that factors each block into a
moment of the standardized error times a moment of the regressors, valid
when the scale model is right (Machado and Santos Silva's Theorem 3 form).
The density of ``eps`` at ``q(tau)`` is Stata ``qreg``'s default: the
"fitted" difference quotient of sample quantiles at ``tau +/- h`` with the
Hall-Sheather bandwidth ``h``.

Conventions are those of Stata ``mmqreg`` 2.6 (Rios-Avila's implementation
of the paper), used as a black box: coefficients and the full covariance of
the location, scale and quantile equations agree to about 1e-13 on the
fixtures of ``tests/test_mmqreg.py`` for each ``vce``, with one and two sets
of fixed effects and without any. The influence function of ``q(tau)`` is
the one ``mmqreg`` uses, ``[sigma_i (tau - 1{eps_i <= q}) / f(q) - nu_i -
q (nu~_i - sigma_i)] / mean(sigma)``; the paper's displayed equation (13)
omits the factor ``sigma_i / mean(sigma)`` on the first term.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core._collinear import independent_columns
from ..core.results import EconometricResults
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility

__all__ = ["mmqreg"]

_VCE = ("gls", "robust", "cluster")


def _as_list(v: Union[None, str, Sequence[str]]) -> List[str]:
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def _partial_out(W: np.ndarray, fe: pd.DataFrame) -> np.ndarray:
    """Columns of ``W`` net of the fixed effects, recentred on their means."""
    if fe.shape[1] == 0:
        return W
    means = W.mean(axis=0)
    if fe.shape[1] == 1:
        codes = pd.factorize(fe.iloc[:, 0], sort=True)[0]
        cnt = np.bincount(codes)
        dem = np.empty_like(W)
        for j in range(W.shape[1]):
            gm = np.bincount(codes, weights=W[:, j]) / cnt
            dem[:, j] = W[:, j] - gm[codes]
    else:
        from .hdfe import demean

        dem, _ = demean(W, fe, drop_singletons=False, tol=1e-14, maxiter=100000)
    return dem + means


def _drop_singletons(frame: pd.DataFrame, fes: List[str]) -> Any:
    """Drop observations alone in a group of any absorbed set, repeatedly
    (dropping one can create another in a different set)."""
    start = len(frame)
    while fes:
        alone = np.zeros(len(frame), dtype=bool)
        for c in fes:
            alone |= frame.groupby(c)[c].transform("size").to_numpy() == 1
        if not alone.any():
            break
        frame = frame.loc[~alone]
    return frame, start - len(frame)


def _lp_quantile(sorted_e: np.ndarray, tau: float) -> float:
    """The vertex Stata ``qreg`` returns for a constant-only model: the
    ``(floor(n tau) + 1)``-th order statistic (the upper end of the set of
    minimizers when ``n tau`` is an integer)."""
    n = len(sorted_e)
    k = int(np.floor(n * tau + 1e-12 * n))
    return float(sorted_e[min(k, n - 1)])


def _hs_density(e: np.ndarray, tau: float) -> Dict[str, float]:
    """Stata ``qreg``'s default density at the ``tau``-quantile: the fitted
    difference quotient with the Hall-Sheather bandwidth (alpha = 0.05)."""
    n = len(e)
    x = stats.norm.ppf(tau)
    h = (
        n ** (-1 / 3)
        * stats.norm.ppf(0.975) ** (2 / 3)
        * ((1.5 * stats.norm.pdf(x) ** 2) / (2 * x**2 + 1)) ** (1 / 3)
    )
    if not (0 < tau - h and tau + h < 1):
        raise DataInsufficient(
            f"mmqreg: the Hall-Sheather bandwidth {h:.4f} puts tau +/- h "
            f"outside (0, 1) at tau = {tau} with {n} observations.",
            recovery_hint="Use a quantile further from 0 and 1, or more data.",
            diagnostics={"tau": tau, "bandwidth": float(h), "n": n},
        )
    s = np.sort(e)
    width = _lp_quantile(s, tau + h) - _lp_quantile(s, tau - h)
    if width <= 1e-12 * max(1.0, float(np.max(np.abs(s)))):
        raise DataInsufficient(
            "mmqreg: the standardized residuals are tied around q(tau); the "
            "density at the quantile is not estimable.",
            recovery_hint="Check for a discrete outcome; MMQR assumes a "
            "continuous error.",
        )
    return {"density": float(2 * h / width), "bandwidth": float(h)}


@records_call("mmqreg")
def mmqreg(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str]],
    absorb: Union[None, str, Sequence[str]] = None,
    quantile: float = 0.5,
    vce: str = "gls",
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """Quantile regression via moments (MMQR), with absorbed fixed effects.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome.
    x : str or list of str
        Regressors (numeric columns).
    absorb : str or list of str, optional
        Fixed effects to partial out (one or more sets). One set is Stata
        ``xtqreg`` / ``mmqreg, absorb(id)``.
    quantile : float, default 0.5
        The quantile, strictly between 0 and 1.
    vce : {'gls', 'robust', 'cluster'} or 'cluster <var>', default 'gls'
        ``'gls'`` is valid when the location-scale model is right (Stata
        ``mmqreg``'s default, kept here so results match it); ``'robust'``
        and ``'cluster'`` use the influence functions without that
        factorization. The method paper (Rios-Avila, Siles and
        Canavire-Bacarreza 2024) favours them: it finds the GLS standard
        errors unstable when fitted scales are near zero or negative,
        robust ones better behaved even under a correct model (if somewhat
        small in small samples), and recommends ``'cluster'`` when errors
        are correlated within groups and the sample is large. See Notes.
        Stata's ``vce(cluster id)`` is ``vce='cluster id'``, the same as
        ``vce='cluster', cluster='id'`` (so ``sp.vcov(fit, 'cluster id')``
        works).
    cluster : str, optional
        Cluster variable for ``vce='cluster'``.
    alpha : float, default 0.05
        Level of the confidence intervals.

    Returns
    -------
    EconometricResults
        ``params`` / ``std_errors`` are the quantile coefficients (the
        constant is the recentred one and has no meaning when effects are
        absorbed). ``model_info`` has ``q`` (the quantile of the
        standardized error), ``density`` and ``bandwidth``, ``location`` and
        ``scale`` (coefficient tables), ``vcov_all`` (the joint covariance
        of the location, scale and quantile equations), ``n_nonpositive_scale``
        and ``n_singletons_dropped``.

    Notes
    -----
    Observations alone in a group of an absorbed set are dropped (with a
    warning): both their residual and their fitted scale are zero. Stata
    keeps them and ranks the resulting 0/0 rounding noise among the
    standardized residuals, which can move ``q(tau)`` by one order
    statistic.

    The scale model is linear, so its fitted values can be negative; the
    standardized residuals and the variance then lose their meaning, and a
    warning counts those observations (as ``mmqreg`` does).

    The default ``vce='gls'`` follows Stata ``mmqreg``. Its validity rests
    on the location-scale model holding exactly; in Rios-Avila, Siles and
    Canavire-Bacarreza's (2024) simulations it is biased when fitted
    scales are near zero or negative, where the robust and clustered
    standard errors stay stable. Prefer ``vce='robust'`` or
    ``vce='cluster'`` (for panels, clustered on the unit) unless the aim
    is to reproduce ``mmqreg``'s default output.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> df = pd.DataFrame({"id": rng.integers(0, 40, n), "x": rng.uniform(0, 2, n)})
    >>> df["y"] = 1 + df["x"] + (1 + 0.5 * df["x"]) * rng.normal(size=n)
    >>> r = sp.mmqreg(df, "y", "x", absorb="id", quantile=0.9)
    >>> bool(r.params["x"] > 1)
    True

    References
    ----------
    [@machado2019quantiles],
    [@riosavila2024estimating]
    """
    xs = _as_list(x)
    fes = _as_list(absorb)
    if not xs:
        raise MethodIncompatibility(
            "mmqreg: give at least one regressor.",
            recovery_hint="Pass x='col' or a list of columns.",
        )
    if not 0 < float(quantile) < 1:
        raise MethodIncompatibility(
            f"mmqreg: quantile={quantile} is not strictly between 0 and 1.",
            recovery_hint="Pass e.g. quantile=0.25 (not 25).",
        )
    if isinstance(vce, str) and vce.split()[:1] == ["cluster"] and len(vce.split()) > 1:
        words = vce.split()
        if len(words) != 2:
            raise MethodIncompatibility(
                f"mmqreg: vce={vce!r} names {len(words) - 1} cluster variables; "
                "mmqreg clusters on one.",
                recovery_hint="Pass vce='cluster <one column>'.",
            )
        if cluster is not None and cluster != words[1]:
            raise MethodIncompatibility(
                f"mmqreg: vce={vce!r} and cluster={cluster!r} disagree.",
                recovery_hint="Name the cluster variable once.",
            )
        vce, cluster = "cluster", words[1]
    if vce not in _VCE:
        raise MethodIncompatibility(
            f"mmqreg: vce={vce!r}.",
            recovery_hint="Use 'gls', 'robust', 'cluster' or 'cluster <var>'.",
        )
    if (vce == "cluster") != (cluster is not None):
        raise MethodIncompatibility(
            "mmqreg: vce='cluster' and cluster= go together.",
            recovery_hint="Pass vce='cluster', cluster='<column>'.",
        )
    cols = [y] + xs + fes + ([cluster] if cluster else [])
    check_columns(data, list(dict.fromkeys(cols)), function="mmqreg")
    frame = data[list(dict.fromkeys(cols))].dropna()
    frame, n_single = _drop_singletons(frame, fes)
    if n_single:
        warnings.warn(
            f"mmqreg: dropped {n_single} observation(s) that are alone in their "
            "absorbed group; their location residual and fitted scale are both "
            "zero, so their standardized residual is undefined (Stata mmqreg / "
            "xtqreg keep them and use rounding noise).",
            AssumptionWarning,
            stacklevel=2,
        )
    n = len(frame)
    tau = float(quantile)
    try:
        Y = frame[y].to_numpy(dtype=float)
        X = frame[xs].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"mmqreg: outcome and regressors must be numeric ({exc}).",
            recovery_hint="Encode categorical regressors as indicator columns.",
        ) from exc

    W = _partial_out(np.column_stack([Y, X]), frame[fes])
    Yr, Xr = W[:, 0], W[:, 1:]
    names = xs + ["_cons"]
    Z = np.column_stack([Xr, np.ones(n)])
    keep, omitted = independent_columns(Z, names)
    if omitted:
        raise MethodIncompatibility(
            "mmqreg: collinear regressors (after absorbing the fixed effects): "
            + ", ".join(o["variable"] for o in omitted),
            recovery_hint="Drop regressors that are constant within the absorbed "
            "groups or linear combinations of others.",
            diagnostics={"omitted": omitted},
        )
    k = Z.shape[1]
    if n <= k + 1:
        raise DataInsufficient(
            f"mmqreg: {n} observations for {k} coefficients.",
            recovery_hint="Use more data or fewer regressors.",
        )
    ZZi = np.linalg.inv(Z.T @ Z)
    b = ZZi @ (Z.T @ Yr)
    nu = Yr - Z @ b
    absnu = np.abs(nu)
    ar = _partial_out(absnu[:, None], frame[fes])[:, 0]
    g = ZZi @ (Z.T @ ar)
    sig = absnu - (ar - Z @ g)  # fitted scale, absorbed effects included
    bad = int(np.sum(sig <= 0))
    if bad:
        warnings.warn(
            f"mmqreg: the fitted scale is not positive for {bad} of {n} "
            "observations; the location-scale model cannot hold there and the "
            "standard errors are unreliable.",
            AssumptionWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = nu / sig
    q = _lp_quantile(np.sort(eps), tau)
    dens = _hs_density(eps, tau)
    fq = dens["density"]
    sbar = float(sig.mean())

    # Influence functions (sum form: V = IF' IF), Rios-Avila et al. (2024)
    pos = (nu >= 0).astype(float)
    nut = 2 * nu * (pos - pos.mean())
    ind = tau - (eps <= q)
    if_q = (sig * ind / fq - nu - q * (nut - sig)) / sbar
    if vce == "gls":
        et = nut / sig
        kap = [eps, et - 1, if_q / sig]
        P = n * (Z @ ZZi) * sig[:, None]
        psi = [P, P, sig[:, None]]
        dims = [k, k, 1]
        off = np.cumsum([0] + dims)
        V_all = np.zeros((2 * k + 1, 2 * k + 1))
        for i in range(3):
            for j in range(3):
                omega = psi[i].T @ psi[j] / n
                V_all[off[i] : off[i + 1], off[j] : off[j + 1]] = (
                    np.mean(kap[i] * kap[j]) * omega / n
                )
    else:
        IF = np.column_stack(
            [(Z * nu[:, None]) @ ZZi, (Z * (nut - sig)[:, None]) @ ZZi, if_q / n]
        )
        if vce == "cluster":
            codes = pd.factorize(frame[cluster], sort=True)[0]
            S = np.zeros((codes.max() + 1, IF.shape[1]))
            np.add.at(S, codes, IF)
            IF = S
        V_all = IF.T @ IF
    T = np.vstack(
        [
            np.hstack([np.eye(2 * k), np.zeros((2 * k, 1))]),
            np.hstack([np.eye(k), q * np.eye(k), g[:, None]]),
        ]
    )
    V_eq = T @ V_all @ T.T
    beta = b + q * g
    zc = stats.norm.ppf(1 - alpha / 2)

    def table(coef: np.ndarray, var: np.ndarray) -> pd.DataFrame:
        se = np.sqrt(np.diag(var))
        return pd.DataFrame(
            {
                "coef": coef,
                "se": se,
                "z": coef / se,
                "pvalue": 2 * stats.norm.sf(np.abs(coef / se)),
                "ci_lower": coef - zc * se,
                "ci_upper": coef + zc * se,
            },
            index=names,
        )

    eq_names = (
        [f"location:{v}" for v in names]
        + [f"scale:{v}" for v in names]
        + [f"qtile:{v}" for v in names]
    )
    Vq = V_eq[2 * k :, 2 * k :]
    info: Dict[str, Any] = {
        "model_type": "MMQR (quantiles via moments)",
        "quantile": tau,
        "q": q,
        "density": fq,
        "bandwidth": dens["bandwidth"],
        "vce": vce,
        "cluster": cluster,
        "absorb": fes,
        "location": table(b, V_eq[:k, :k]),
        "scale": table(g, V_eq[k : 2 * k, k : 2 * k]),
        "vcov_all": pd.DataFrame(V_eq, index=eq_names, columns=eq_names),
        "n_nonpositive_scale": bad,
        "n_singletons_dropped": n_single,
        "alpha": alpha,
    }
    if cluster is not None:
        info["n_clusters"] = int(frame[cluster].nunique())
    data_info: Dict[str, Any] = {
        "nobs": n,
        "n_obs": n,
        "dependent_var": y,
        "var_cov": Vq,
        "var_names": names,
        "inference": "z",
    }
    return EconometricResults(
        params=pd.Series(beta, index=names),
        std_errors=pd.Series(np.sqrt(np.diag(Vq)), index=names),
        model_info=info,
        data_info=data_info,
    )
