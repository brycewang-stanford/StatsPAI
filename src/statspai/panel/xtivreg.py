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

The formulas are those of Stata's documentation for ``xtivreg``, with the
conventions its output shows where the manual is silent (balanced panels
with time-invariant variables; no time-varying excluded instrument); the
evidence is ``tests/test_xtivreg_stata.py``.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._xtivreg_extras import fd_pairs as _fd_pairs
from ._xtivreg_extras import first_stage as _first_stage
from ._xtivreg_extras import model_test as _model_test
from ._xtivreg_extras import ols_fit as _ols_fit

__all__ = ["xtivreg"]


def _as_list(v: Union[str, Sequence[str], None]) -> List[str]:
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def _gmean(values: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    total = np.zeros((n,) + values.shape[1:])
    np.add.at(total, codes, values)
    cnt = np.bincount(codes, minlength=n).astype(float)
    _ret: np.ndarray = total / cnt.reshape((-1,) + (1,) * (values.ndim - 1))
    return _ret


def _span(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Columns of ``A`` with the linearly dependent ones removed."""
    return A[:, _span_index(A, tol)]


def _span_index(A: np.ndarray, tol: float = 1e-10) -> List[int]:
    """Indices of the columns :func:`_span` keeps."""
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
    return keep


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


def _re_within_unidentified(
    df: pd.DataFrame,
    codes: np.ndarray,
    n: int,
    en: List[str],
    iv: List[str],
    keep: List[int],
    Xw: np.ndarray,
    n_endog: int,
) -> List[str]:
    """Endogenous regressors the within 2SLS of ``method='re'`` cannot use.

    When no excluded instrument varies within panels, Stata's ``xtivreg,
    re`` estimates ``sigma_e`` from the within regression without the
    endogenous regressors (checked against Stata 18). When some do but too
    few, which regressor Stata leaves out is not documented; that case is
    refused.
    """
    if not en or _span(Xw).shape[1] >= len(keep):
        return []
    varies = [
        v
        for v in iv
        if np.ptp(
            (
                df[v].to_numpy(dtype=float)
                - _gmean(df[v].to_numpy(dtype=float), codes, n)[codes]
            )
        )
        > 1e-12
    ]
    if varies:
        raise MethodIncompatibility(
            f"xtivreg: method='re' estimates sigma_e from the within 2SLS, and "
            f"the instruments that vary within panels ({varies}) are fewer than "
            f"the endogenous regressors that do ({en}).",
            recovery_hint="Add a time-varying excluded instrument per "
            "endogenous regressor, or use method='be'.",
        )
    return [en[j] for j in keep if j < n_endog]


def _re_sigma_e_k(
    n_cols: int,
    n_keep: int,
    n_within_omitted: int,
    balanced: bool,
    se_kind: str,
    n_endog: int,
) -> int:
    """The ``K`` of ``s2_nu = SSR_within / (N - n - K + 1)`` in ``re``.

    Every column of the model (the manual's ``K = g2 + k1`` plus the
    constant); on a balanced panel only those the within regression uses.
    ``n_endog`` is unused: Stata 18 counts two or more endogenous
    regressors as one here (and only here; its ``sigma_u`` uses the full
    ``K``), which is not followed. ``tests/test_xtivreg_stata.py`` pins
    Stata's numbers by running with that count.
    """
    if balanced:
        return n_keep - (n_within_omitted if se_kind != "cluster" else 0)
    return n_cols


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
    time: Optional[str] = None,
    small: bool = False,
    first: bool = False,
    regress: bool = False,
) -> EconometricResults:
    """
    Instrumental-variables regression for panel data.

    Equivalent to Stata's ``xtivreg y x (endog = instruments), re | fe |
    be | fd`` with ``ec2sls`` / ``nosa`` / ``small`` / ``first`` /
    ``regress``. The random-effects estimator is feasible-GLS 2SLS: G2SLS
    (Balestra and Varadharajan-Krishnakumar) by default, Baltagi's EC2SLS
    with ``ec2sls=True``.

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
    method : {'re', 'fe', 'be', 'fd'}, default 're'
        Random effects (feasible-GLS 2SLS, Stata's default), fixed effects
        (within 2SLS), between effects (2SLS on the panel means) or first
        differences (2SLS of ``D.y`` on ``D.endog``, ``D.x`` and a
        constant with instruments ``D.x``, ``D.instruments`` and the
        constant; Anderson-Hsiao). Regressors constant within panels drop
        out of ``'fe'`` and ``'fd'``.
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
    time : str, optional
        Time variable; required with ``method='fd'``. A difference is taken
        between consecutive periods of a panel, the periods being the
        distinct values of ``time`` in the sample (Stata's ``D.`` on an
        ``xtset`` panel); observations on either side of a gap form no
        difference.
    small : bool, default False
        Stata's ``small``: t statistics with the residual degrees of
        freedom instead of z, and the model test as F instead of chi2. The
        degrees of freedom are Stata's ``e(df_r)``: ``N - n - K + 1``
        (``'fe'``, also when clustering), ``n - K`` (``'be'``),
        ``N - K + 1`` (``'re'``; ``G`` with ``G`` clusters) and ``N_D - K``
        (``'fd'``, on the ``N_D`` differences; ``G - 1`` with ``G``
        clusters), with ``K`` regressors including the constant.
        Coefficients and standard errors do not change.
    first : bool, default False
        Stata's ``first``: also fit the first-stage regression of each
        endogenous regressor on all instruments in the estimator's own
        transformation (within for ``'fe'``, panel means for ``'be'``,
        quasi-demeaned for ``'re'``, first differences for ``'fd'``) and
        store it in ``model_info['first_stage']``.
    regress : bool, default False
        Stata's ``regress``: treat the endogenous regressors as exogenous
        and ignore ``instruments``, i.e. fit the same panel estimator of
        ``y`` on ``endog`` and ``x`` without instrumenting. ``'fe'`` then
        equals Stata's ``xtreg, fe``; ``'re'`` keeps the variance
        components of ``xtivreg``, which are not those of ``xtreg, re``.

    Returns
    -------
    EconometricResults
        Coefficients with z inference (Stata's default; t with
        ``small=True``). ``model_info`` holds ``sigma_u``, ``sigma_e``,
        ``rho``, ``theta`` (per-panel values summarised as min / median /
        max when the panel is unbalanced), the within / between / overall
        R-squared, the number of panels, and the Wald test that all
        coefficients but the constant are zero (``chi2`` / ``chi2_p``, or
        ``F`` / ``F_p`` with ``small``, and ``df_m``). With ``first=True``,
        ``model_info['first_stage']`` maps each endogenous regressor to the
        coefficients and standard errors of its first stage.

    Notes
    -----
    Agrees with Stata 18 ``xtivreg`` to rounding on ``fe``, ``be``, ``fd``,
    ``re`` with ``nosa`` and ``re`` with one endogenous regressor. With two
    or more endogenous regressors on an unbalanced panel, Stata's ``re``
    divides the within sum of squares behind ``sigma_e`` by
    ``N - n - K + 1`` with the endogenous regressors counted as one (its
    ``sigma_u`` uses the full ``K``, as its manual does for both); the full
    ``K`` is used here, so ``sigma_e`` is Stata's times
    ``sqrt((N - n - K + g2) / (N - n - K + 1))`` for ``g2`` endogenous
    regressors, and the estimates move by less than that
    (``tests/test_xtivreg_stata.py`` reproduces Stata's numbers under its
    count). On a balanced panel with ``vce='robust'`` Stata's count also
    includes a continuous time-invariant regressor, through rounding noise
    in its within transform; that is not followed either.

    Reporting conventions of Stata's options that are not followed (each
    pinned in ``tests/test_xtivreg_stata.py``): with ``small`` and a
    cluster-robust variance Stata's ``fd`` divides the residual sum of
    squares behind ``sigma_e`` by ``G - 1`` instead of ``N_D - K``; its
    ``fe`` with ``small`` reports an F test with ``n - 1 + K`` numerator
    degrees of freedom that takes in the panel effects, where this is the
    Wald test of the coefficients; its ``re`` first stage under
    ``vce(robust)`` uses heteroskedasticity-robust (HC1) standard errors
    while the main fit clusters on the panel, where this clusters both;
    and its ``fd`` between R-squared is not the manual's
    ``corr(Zbar_i b, ybar_i)^2``, which is what is reported here.

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
    if m not in ("re", "fe", "be", "fd"):
        raise MethodIncompatibility(
            f"xtivreg: method must be 're', 'fe', 'be' or 'fd'; got {method!r}.",
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
    if m == "fd" and time is None:
        raise MethodIncompatibility(
            "xtivreg: method='fd' needs time=, the time variable that orders "
            "the periods of each panel.",
            recovery_hint="Pass time='<time column>'.",
        )
    xs, en, iv = _as_list(x), _as_list(endog), _as_list(instruments)
    if regress:
        if first:
            raise MethodIncompatibility(
                "xtivreg: first=True reports the first stages of the "
                "endogenous regressors, and regress=True treats them as "
                "exogenous, so there is none.",
                recovery_hint="Drop one of first= and regress=.",
            )
        iv = list(en)  # every regressor instruments itself
    if first and not en:
        raise MethodIncompatibility(
            "xtivreg: first=True needs at least one endogenous regressor.",
        )
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
        {
            "y": y,
            "x": xs,
            "endog": en,
            "instruments": iv,
            "id": id,
            "cluster": cluster,
            "time": time if m == "fd" else None,
        },
        function="xtivreg",
    )
    extra = [cluster] if cluster else []
    if m == "fd":
        extra.append(time)
    cols = list(dict.fromkeys([y, id] + xs + en + iv + extra))
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
    x_names = xs + iv + ["_cons"]
    if n < 2:
        raise DataInsufficient("xtivreg: needs at least two panels.")
    ybar, Zbar, Xbar = (_gmean(a, codes, n) for a in (yv, Z, X))

    def within(a: np.ndarray, abar: np.ndarray) -> np.ndarray:
        return np.asarray(a - abar[codes] + a.mean(axis=0))

    def cluster_codes(rows: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        cc, _ = pd.factorize(df[cluster], sort=True)
        if rows is not None:
            cc, _ = pd.factorize(cc[rows], sort=True)
        return cc, int(cc.max()) + 1

    info: Dict[str, Any] = {"method": m, "n_groups": n, "id": id}
    df_r: Optional[float] = None  # residual df under small=True
    stage1: Dict[str, Tuple[Dict[str, np.ndarray], List[str]]] = {}
    s1_df: Callable[[int], float] = lambda k: float(N - k)  # noqa: E731
    s1_rows: Optional[np.ndarray] = None
    s1_n = N
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
        within_omitted: List[str] = []
        balanced = bool(np.ptp(Ti) == 0)
        if m == "re":
            within_omitted = _re_within_unidentified(
                df, codes, n, en, iv, keep, Xw, len(en)
            )
            wkeep = [j for j in keep if names[j] not in within_omitted]
        fw = _tsls(yw, Zw[:, wkeep] if m == "re" else Zw, Xw)
        # K counts every regressor of the model being fitted. On an
        # unbalanced panel Stata's xtivreg, re counts the time-invariant
        # ones the within regression cannot use (its manual's formula); on
        # a balanced panel it counts only the regressors the within
        # regression uses, and an endogenous regressor left out of it for
        # lack of a time-varying instrument counts only with the
        # cluster-robust variance (Stata 18, tests/test_xtivreg_stata.py).
        Kw = (
            len(keep)
            if m == "fe"
            else _re_sigma_e_k(
                Z.shape[1], len(keep), len(within_omitted), balanced, se_kind, len(en)
            )
        )
        s2_nu = float(fw["e"] @ fw["e"]) / (N - n - Kw + 1)
        if within_omitted:
            warnings.warn(
                f"xtivreg: no excluded instrument varies within panels, so the "
                f"within regression that estimates sigma_e leaves out "
                f"{within_omitted} (as Stata's xtivreg, re does). That sigma_e "
                "absorbs their effect, so the variance components and theta "
                "are not consistent; the coefficients remain a consistent IV "
                "estimator, but the conventional SEs assume the components "
                "are right: prefer vce='robust' (clustered on the panel).",
                UserWarning,
                stacklevel=2,
            )
            info["within_omitted"] = within_omitted
    K = Z.shape[1]
    if m == "fe":
        fit = fw
        V = s2_nu * fit["bread"]
        df_r = float(N - n - K + 1)
        if se_kind == "cluster":
            ccodes, G = cluster_codes()
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
        if first:
            stage1 = {
                v: _ols_fit(_tsls, _span_index, Zw[:, names.index(v)], Xw, x_names)
                for v in en
                if v in names
            }
            s1_df = lambda k: float(N - n - k + 1)  # noqa: E731
    elif m == "be":
        # one row per panel: Stata's xtivreg, be does not weight the means
        # by T_i (its manual says each mean enters T_i times; the numbers
        # are those of the unweighted regression)
        fit = _tsls(ybar, Zbar, Xbar)
        s2 = float(fit["e"] @ fit["e"]) / (n - K)
        V = s2 * fit["bread"]
        df_r = float(n - K)
        if se_kind == "cluster":
            raise MethodIncompatibility(
                "xtivreg: method='be' has the conventional variance only.",
            )
        if first:
            stage1 = {
                v: _ols_fit(_tsls, _span_index, Zbar[:, j], Xbar, x_names)
                for j, v in enumerate(en)
            }
            s1_df = lambda k: float(n - k)  # noqa: E731
            s1_n = n
    elif m == "fd":
        cur, prv = _fd_pairs(codes, df[time], str(time))
        if cur.size == 0:
            raise DataInsufficient(
                "xtivreg: method='fd' found no two consecutive periods of a "
                "panel in the sample.",
                recovery_hint="Check time=; each panel needs observations at "
                "two adjacent periods.",
            )
        dy = yv[cur] - yv[prv]
        dZ = Z[cur] - Z[prv]
        dX = X[cur] - X[prv]
        dZ[:, -1] = 1.0
        dX[:, -1] = 1.0
        keep = [j for j in range(K) if j == K - 1 or np.ptp(dZ[:, j]) > 1e-12]
        dropped = [names[j] for j in range(K) if j not in keep]
        Z, dZ = Z[:, keep], dZ[:, keep]
        names = [names[j] for j in keep]
        K = len(keep)
        Nd = int(cur.size)
        fit = _tsls(dy, dZ, dX)
        s2_d = float(fit["e"] @ fit["e"]) / (Nd - K)
        V = s2_d * fit["bread"]
        df_r = float(Nd - K)
        if se_kind == "cluster":
            ccodes, G = cluster_codes(cur)
            V = _cluster_vcov(fit, ccodes, G, Nd, K)
            info["n_clusters"] = G
            df_r = float(G - 1)
        # the levels of the differenced rows (Stata's e(sample))
        pc, _ = pd.factorize(codes[cur], sort=True)
        n_fd = int(pc.max()) + 1
        xb_s = Z[cur][:, :-1] @ fit["d"][:-1]
        y_s = yv[cur]
        u_i = _gmean(y_s - xb_s, pc, n_fd)
        sigma_u = float(np.std(u_i, ddof=1))
        xb_bar, y_bar = _gmean(xb_s, pc, n_fd), _gmean(y_s, pc, n_fd)
        info.update(
            omitted=dropped,
            n_groups=n_fd,
            n_differences=Nd,
            sigma_e=float(np.sqrt(s2_d)),
            sigma_u=sigma_u,
            rho=float(sigma_u**2 / (sigma_u**2 + s2_d)),
            corr=float(np.corrcoef(u_i[pc], xb_s)[0, 1]),
        )
        fd_r2 = (
            _r2(xb_s - xb_bar[pc], y_s - y_bar[pc]),
            _r2(xb_bar, y_bar),
            _r2(xb_s, y_s),
        )
        if first:
            stage1 = {
                v: _ols_fit(_tsls, _span_index, dZ[:, names.index(v)], dX, x_names)
                for v in en
                if v in names
            }
            s1_df = lambda k: float(Nd - k)  # noqa: E731
            s1_rows, s1_n = cur, Nd
    else:
        # between 2SLS for sigma_mu, unweighted as method='be'. On a
        # balanced panel Stata leaves out the excluded instruments that do
        # not vary within panels (unless none is left), as checked against
        # Stata 18 on a balanced panel.
        Xb = Xbar
        if balanced and iv:
            n_x = len(xs)
            drop = [
                n_x + k
                for k in range(len(iv))
                if np.ptp(X[:, n_x + k] - Xbar[codes, n_x + k]) <= 1e-12
            ]
            if drop and len(drop) < len(iv):
                Xb = np.delete(Xbar, drop, axis=1)
        fb = _tsls(ybar, Zbar, Xb)
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
            g_names = (
                [v + "_d" for v in xs + iv]
                + [v + "_m" for v in xs + iv]
                + ["_cons_m", "_cons"]
            )
        else:
            Xg = Xs
            g_names = x_names
        fit = _tsls(ys, Zs, Xg)
        s2 = float(fit["e"] @ fit["e"]) / (N - K)
        V = s2 * fit["bread"]
        df_r = float(N - K + 1)  # Stata's e(df_r) after xtivreg, re small
        if se_kind == "cluster":
            ccodes, G = cluster_codes()
            V = _cluster_vcov(fit, ccodes, G, N, K)
            info["n_clusters"] = G
            df_r = float(G)
        if first:
            stage1 = {
                v: _ols_fit(_tsls, _span_index, Zs[:, j], Xg, g_names)
                for j, v in enumerate(en)
            }
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
    if m == "fd":
        r2_w, r2_b, r2_o = fd_r2
    else:
        xb = Z @ d
        yd = yv - ybar[codes]
        r2_w = _r2((Z - Zbar[codes]) @ d, yd)
        r2_b = _r2(Zbar @ d, ybar)
        r2_o = _r2(xb, yv)
        if m == "fe":  # the R-squared of the mean-deviated regression
            e = yd - (Z - Zbar[codes]) @ d
            r2_w = float(1.0 - (e @ e) / (yd @ yd))
        if m == "be":  # the R-squared of the between regression
            e = ybar - Zbar @ d
            r2_b = float(1.0 - (e @ e) / np.sum((ybar - ybar.mean()) ** 2))
    if first:
        cv = None
        if se_kind == "cluster":
            c1, G1 = cluster_codes(s1_rows)
            cv = lambda f, k: _cluster_vcov(f, c1, G1, s1_n, k)  # noqa: E731
        info["first_stage"] = _first_stage(
            stage1, s1_df, cv, s1_n, "t" if m in ("fe", "be", "fd") else "z"
        )
    info.update(
        _model_test(d, V, small, df_r),
        r2_within=r2_w,
        r2_between=r2_b,
        r2_overall=r2_o,
        vce=se_kind,
        cluster=cluster if se_kind == "cluster" else None,
        endog=[] if regress else en,
        instruments=[] if regress else iv,
        alpha=alpha,
        small=bool(small),
        model_type="Panel IV",
    )
    if regress:
        info["regress"] = True
    if m == "fd":
        info["time"] = time
    n_used = info.get("n_differences", N)
    if small:
        inference: Dict[str, Any] = {"inference": "t", "df_resid": df_r}
        info["df_r"] = df_r
    else:
        inference = {"df_resid": n_used - K, "inference": "z"}
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = d / se
    if small:
        pvals = 2 * stats.t.sf(np.abs(zstat), df_r)
    else:
        pvals = 2 * stats.norm.sf(np.abs(zstat))
    params = pd.Series(d, index=names)
    res = EconometricResults(
        params=params,
        std_errors=pd.Series(se, index=names),
        model_info=info,
        data_info={
            "nobs": n_used,
            "n_obs": n_used,
            "dependent_var": y,
            "var_cov": V,
            "var_names": names,
            **inference,
        },
        diagnostics={
            ("t" if small else "z"): dict(zip(names, zstat)),
            "p": dict(zip(names, pvals)),
        },
    )
    return res
