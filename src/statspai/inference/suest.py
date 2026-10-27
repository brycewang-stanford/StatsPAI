"""Seemingly unrelated estimation: joint covariance of separate OLS fits.

Stata ``suest`` after ``regress``: each equation is estimated on its own
sample, the per-observation influence functions ``(X_j'X_j)^{-1} x_ij e_ij``
are stacked across equations, and their cross-equation sandwich gives the
joint covariance -- robust (``n/(n-1)``) or clustered (``G/(G-1)``). That is
what a "joint test" row across outcomes needs (e.g. equality of a treatment
effect across outcomes, all effects zero), which ``sp.sureg`` (Zellner FGLS,
no clusters, common sample) cannot provide.

Equations may carry factor terms (``i.f`` / ``C(f)``) and absorbed fixed
effects (``y ~ x | fe``, swept per equation by FWL), and the covariance may
be clustered on several variables (Cameron-Gelbach-Miller inclusion-
exclusion). Stata's ``suest`` has neither, so those cases are pinned to the
equivalent stacked regression in ``reghdfe`` / ``regress``
(``tests/reference_parity/test_suest_multiway_fe_parity.py``).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..core._call_record import records_call
from ..core._vcov_spec import accepts_vce
from ..exceptions import ColumnNotFound, MethodIncompatibility


@dataclass
class SuestResult(ResultProtocolMixin):
    """Joint estimates of several OLS equations.

    Attributes
    ----------
    params : pd.Series
        Coefficients, indexed ``"<equation>:<term>"`` (the equation name is
        its dependent variable).
    vcov : pd.DataFrame
        Joint covariance of all coefficients.
    n_obs : dict
        Estimation sample size of each equation.
    n_clusters : int or None
        Number of clusters (the smallest count under multi-way clustering).
    cluster_info : dict
        Cluster variables, their counts, the multi-way small-sample factor
        and the number of negative eigenvalues set to zero.
    omitted : dict
        Regressors omitted for collinearity, by equation.
    absorbed : dict
        Absorbed fixed effects, by equation.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> d = pd.DataFrame({"x": rng.normal(size=100)})
    >>> d["y1"], d["y2"] = d.x + rng.normal(size=100), rng.normal(size=100)
    >>> r = sp.suest(d, ["y1 ~ x", "y2 ~ x"])
    >>> type(r).__name__, r.test_zero("x")["df"]
    ('SuestResult', 2)
    """

    params: pd.Series
    vcov: pd.DataFrame
    n_obs: Dict[str, int]
    n_clusters: Optional[int] = None
    vce: str = "robust"
    equations: List[str] = field(default_factory=list)
    cluster_info: Dict[str, Any] = field(default_factory=dict)
    omitted: Dict[str, List[str]] = field(default_factory=dict)
    absorbed: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def std_errors(self) -> pd.Series:
        return pd.Series(
            np.sqrt(np.diag(self.vcov.to_numpy())), index=self.params.index
        )

    def wald(self, R: Any, q: Optional[Sequence[float]] = None) -> Dict[str, float]:
        """Wald test of ``R b = q`` (chi-squared, as Stata ``test``).

        ``R`` is a matrix over ``params`` (columns in ``params`` order) or a
        list of dicts mapping coefficient names to weights, one per
        restriction.
        """
        names = list(self.params.index)
        if isinstance(R, (list, tuple)) and R and isinstance(R[0], dict):
            M = np.zeros((len(R), len(names)))
            for i, row in enumerate(R):
                for k, v in row.items():
                    if k not in names:
                        raise MethodIncompatibility(
                            f"Unknown coefficient {k!r}.",
                            recovery_hint=f"Use names like {names[0]!r}.",
                        )
                    M[i, names.index(k)] = float(v)
        else:
            M = np.atleast_2d(np.asarray(R, dtype=float))
        qv = np.zeros(M.shape[0]) if q is None else np.asarray(q, dtype=float)
        diff = M @ self.params.to_numpy() - qv
        V = M @ self.vcov.to_numpy() @ M.T
        chi2 = float(diff @ np.linalg.solve(V, diff))
        df = int(np.linalg.matrix_rank(V))
        return {"chi2": chi2, "df": df, "p_value": float(stats.chi2.sf(chi2, df))}

    def test_equal(
        self, term: str, equations: Optional[Sequence[str]] = None
    ) -> Dict[str, float]:
        """Test that ``term`` has the same coefficient in every equation."""
        eqs = list(equations or self.equations)
        rows = [{f"{eqs[0]}:{term}": 1.0, f"{e}:{term}": -1.0} for e in eqs[1:]]
        return self.wald(rows)

    def test_zero(
        self, term: str, equations: Optional[Sequence[str]] = None
    ) -> Dict[str, float]:
        """Test that ``term`` is zero in every equation (joint significance)."""
        eqs = list(equations or self.equations)
        return self.wald([{f"{e}:{term}": 1.0} for e in eqs])

    def summary(self) -> str:
        se = self.std_errors
        lines = [f"suest: {len(self.equations)} equations, vce={self.vce}"]
        if self.cluster_info:
            lines.append(
                "  clusters: "
                + ", ".join(
                    f"{c} = {g}"
                    for c, g in zip(
                        self.cluster_info["cluster"], self.cluster_info["n_clusters"]
                    )
                )
            )
        for eq, fe in self.absorbed.items():
            lines.append(f"  {eq}: absorbed {' + '.join(fe)}")
        for eq, om in self.omitted.items():
            lines.append(f"  {eq}: omitted (collinear) {', '.join(om)}")
        for k in self.params.index:
            lines.append(f"  {k:<30s} {self.params[k]: .6f}  ({se[k]:.6f})")
        return "\n".join(lines)


_C_TERM = re.compile(r"\bC\(\s*([^\W\d]\w*)\s*\)")


def _fail(message: str, hint: str, **diag: Any) -> MethodIncompatibility:
    return MethodIncompatibility(message, recovery_hint=hint, diagnostics=diag)


def _equation_spec(data: pd.DataFrame, eq: Any) -> Dict[str, Any]:
    """``{y, plain, xs | terms, fe_terms, columns}`` for one equation.

    Plain column lists keep the original fast path (any column name, an
    intercept appended); anything else goes through the ``sp.hdfe_ols``
    term grammar, with patsy's ``C(x)`` read as Stata's ``i.x``.
    """
    if not isinstance(eq, str):
        y, xs = eq
        xs = list(xs)
        if all(c in data.columns for c in xs):
            return {"y": y, "plain": True, "xs": xs, "columns": [y] + xs}
        eq = f"{y} ~ {' + '.join(xs) if xs else '1'}"
    if eq.count("~") != 1:
        raise _fail(
            f"sp.suest: cannot read the equation {eq!r}.",
            "Write each equation as 'y ~ x1 + x2' (optionally '| fe1 + fe2').",
            equation=eq,
        )
    lhs, rhs = eq.split("~")
    tokens = [t.strip() for t in rhs.split("+")]
    xs = [t for t in tokens if t and t != "1"]
    if "|" not in rhs and all(t in data.columns for t in xs):
        return {
            "y": lhs.strip(),
            "plain": True,
            "xs": xs,
            "columns": [lhs.strip()] + xs,
        }
    from ..panel.feols import _parse_formula

    text = _C_TERM.sub(r"i.\1", eq)
    if "C(" in text:
        raise _fail(
            f"sp.suest: only C(x) with a bare column is read ({eq!r}).",
            "Write C(x) or i.x; for another base level use ib<k>.x.",
            equation=eq,
        )
    try:
        y, x_terms, fe_terms = _parse_formula(text)
    except ValueError as exc:
        raise _fail(
            f"sp.suest: {exc}",
            "Use the sp.hdfe_ols term grammar: x, i.f, C(f), a:b, a*b, '| fe'.",
            equation=eq,
        ) from exc
    cols = [y] + [c for t in x_terms + fe_terms for c in t.columns]
    return {
        "y": y,
        "plain": False,
        "terms": x_terms,
        "fe_terms": fe_terms,
        "columns": list(dict.fromkeys(cols)),
    }


def _fit_equation(
    data: pd.DataFrame, spec: Dict[str, Any], tol: float
) -> Dict[str, Any]:
    """OLS of one equation on its own complete rows (FEs swept by FWL)."""
    from ..core._collinear import drop_collinear

    y = spec["y"]
    rows = np.flatnonzero(data[spec["columns"]].notna().all(axis=1).to_numpy())
    sub = data.iloc[rows]
    if spec["plain"]:
        X = np.column_stack([sub[spec["xs"]].to_numpy(dtype=float), np.ones(len(rows))])
        names = spec["xs"] + ["_cons"]
        fe_names: List[str] = []
    else:
        from ..panel.feols import _materialize_fe, _materialize_rhs

        Xt, names = (
            _materialize_rhs(sub, spec["terms"])
            if spec["terms"]
            else (np.empty((len(rows), 0)), [])
        )
        fe_names = [t.label for t in spec["fe_terms"]]
        if spec["fe_terms"]:
            X = np.asarray(Xt, dtype=float)
        else:
            X = np.column_stack([Xt, np.ones(len(rows))])
            names = list(names) + ["_cons"]
    yv = sub[y].to_numpy(dtype=float)
    n_singletons = 0
    if fe_names:
        from ..panel.hdfe import Absorber, _reghdfe_kept_columns

        fe_mat, slopes, _ = _materialize_fe(sub, spec["fe_terms"])
        ab = Absorber(fe_mat, tol=tol, maxiter=100_000, slopes=slopes, n_obs=len(rows))
        keep = ab.keep_mask
        n_singletons = int(ab.n_dropped)
        M = ab.demean(np.column_stack([yv, X]))
        yv, Xw = M[:, 0], M[:, 1:]
        kept = _reghdfe_kept_columns(X[keep], Xw, None, tol)
        omitted = [names[j] for j in range(len(names)) if j not in kept]
        if omitted:
            warnings.warn(
                f"sp.suest: in the equation for {y!r}, "
                + ", ".join(omitted)
                + " omitted because of collinearity (with the absorbed "
                "effects or the other regressors).",
                UserWarning,
                stacklevel=4,
            )
        X, names, rows = Xw[:, kept], [names[j] for j in kept], rows[keep]
    else:
        X, names, recs, _ = drop_collinear(
            X, names, f"sp.suest (equation for {y!r})", stacklevel=4
        )
        omitted = [r["variable"] for r in recs]
    if X.shape[1] == 0:
        raise _fail(
            f"sp.suest: the equation for {y!r} has no identified regressor.",
            "Check that the regressors vary within the absorbed fixed effects.",
            equation=y,
        )
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ yv)
    e = yv - X @ b
    return {
        "rows": rows,
        "b": b,
        "IF": (X * e[:, None]) @ XtX_inv,
        "names": list(names),
        "omitted": omitted,
        "absorbed": fe_names,
        "n_singletons": n_singletons,
    }


def _multiway_vcov(IF: np.ndarray, keys: List[np.ndarray], cluster_df: str) -> tuple:
    """Cameron-Gelbach-Miller inclusion-exclusion on the stacked scores.

    Each intersection term is the one-way sum of outer products of the
    cluster-summed influence functions -- ``panel.hdfe._cluster_sandwich``
    with the identity as bread, so the cluster meat and the intersection
    coding are the ones ``sp.hdfe_ols`` uses against ``reghdfe`` -- scaled by
    ``G_min/(G_min-1)`` (``cluster_df='min'``, reghdfe / fixest) or by its
    own ``G/(G-1)`` (``'per_term'``). Negative eigenvalues of the sum are set
    to zero (``inference.jackknife.psd_adjust``) and counted.
    """
    from itertools import combinations

    from ..panel.hdfe import _cluster_sandwich, _factorize, _factorize_multi
    from .jackknife import psd_adjust

    m, k = IF.shape
    ones, eye = np.ones(m), np.eye(k)
    counts = [int(_factorize(c)[1]) for c in keys]
    g_min = min(counts)
    V = np.zeros((k, k))
    for r in range(1, len(keys) + 1):
        for combo in combinations(range(len(keys)), r):
            codes, G = _factorize_multi([keys[i] for i in combo])
            # one-way call: scale G/(G-1) x (m-1)/(m-1)
            term = _cluster_sandwich(IF, ones, None, eye, codes, 0, n_absorbed=1)
            if cluster_df == "min":
                term = term * (g_min / (g_min - 1)) / (G / (G - 1))
            V += ((-1) ** (r + 1)) * term
    V, n_negative = psd_adjust(V)
    return V, counts, int(n_negative)


@records_call("suest")
@accepts_vce(default="robust", multiway=True)
def suest(
    data: pd.DataFrame,
    equations: Sequence[Any],
    cluster: Optional[Union[str, Sequence[str]]] = None,
    cluster_df: str = "min",
) -> SuestResult:
    """Joint (seemingly unrelated) inference across separately fitted OLS equations.

    Parameters
    ----------
    data : pd.DataFrame
    equations : sequence
        Each equation is ``(y, [x1, x2, ...])`` or a formula string. Plain
        column names get an intercept. Formulas also take factor terms
        ``i.f`` / ``C(f)`` (level dummies, smallest level omitted, labelled
        ``f::<level>``), ``ib<k>.f``, interactions ``a:b`` / ``a*b``, and
        absorbed fixed effects after a bar, ``"y ~ x + i.f | fe1 + fe2"``
        (the ``sp.hdfe_ols`` grammar). Absorbed effects are swept out of
        ``y`` and the regressors of that equation alone (Frisch-Waugh-Lovell,
        singletons dropped as ``reghdfe`` does) and the equation has no
        ``_cons``. Each equation is fitted on its own complete rows; a
        collinear regressor is omitted with a warning.
    cluster : str or list of str, optional
        Cluster variable(s). One: the joint covariance sums the stacked
        influence functions within clusters and scales by ``G/(G-1)``
        (Stata ``suest, vce(cluster c)``). Several: the Cameron-Gelbach-
        Miller inclusion-exclusion sum over every intersection of them, the
        negative eigenvalues of the result set to zero. Without ``cluster``
        it is the robust ``n/(n-1)`` version, observations (rows of
        ``data``) being the clusters.
    cluster_df : {'min', 'per_term'}, default 'min'
        Small-sample factor of a multi-way covariance: ``G_min/(G_min-1)``
        on every term, with ``G_min`` the smallest cluster count (reghdfe,
        fixest), or each term's own ``G/(G-1)`` (Stata ``regress, vce(cluster
        a b)``). One-way clustering is the same either way.

    Returns
    -------
    SuestResult
        ``params``, joint ``vcov``, and ``wald`` / ``test_equal`` /
        ``test_zero`` for cross-equation hypotheses.

    Notes
    -----
    Degrees of freedom follow ``suest``: the covariance carries ``G/(G-1)``
    (or ``n/(n-1)``) and nothing else -- no ``(N-1)/(N-K)`` and no charge
    for absorbed effects. The covariance is the one of the stacked
    regression of all equations (each regressor and fixed effect interacted
    with an equation indicator, clustered on the same variables, observed
    rows of the same unit sharing a cluster). ``reghdfe`` on that stacked
    regression reports it times ``(N_s - 1) / (N_s - K_s - df_a)``, with
    ``N_s`` the stacked rows, ``K_s`` its regressors and ``df_a`` the
    absorbed levels not nested in a cluster variable; the identity is
    pinned in ``tests/reference_parity/test_suest_multiway_fe_parity.py``.
    Already fitted result objects are not accepted: the influence functions
    need the rows of each equation, which results do not keep.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> d = pd.DataFrame({"x": rng.normal(size=200), "c": np.repeat(range(20), 10)})
    >>> d["y1"] = 0.5 * d.x + rng.normal(size=200)
    >>> d["y2"] = 0.5 * d.x + rng.normal(size=200)
    >>> res = sp.suest(d, ["y1 ~ x", "y2 ~ x"], cluster="c")
    >>> round(res.test_equal("x")["p_value"], 2) > 0.05
    True
    >>> d["t"] = np.tile(range(10), 20)
    >>> res2 = sp.suest(d, ["y1 ~ x | t", "y2 ~ x | t"], cluster=["c", "t"])
    >>> res2.vce
    'cluster(c t)'
    """
    if cluster_df not in ("min", "per_term"):
        raise _fail(
            f"sp.suest: cluster_df must be 'min' or 'per_term'; got {cluster_df!r}.",
            "Pass cluster_df='min' (reghdfe) or 'per_term' (Stata regress).",
            cluster_df=cluster_df,
        )
    if cluster is None:
        cl_names: List[str] = []
    elif isinstance(cluster, str):
        cl_names = [cluster]
    else:
        cl_names = list(dict.fromkeys(cluster))
        if not cl_names:
            raise _fail(
                "sp.suest: cluster= is an empty list.",
                "Pass a column name, a list of them, or None.",
            )
    specs = [_equation_spec(data, eq) for eq in equations]
    names = [s["y"] for s in specs]
    if len(set(names)) != len(names):
        raise MethodIncompatibility(
            "Each equation needs a distinct dependent variable.",
            recovery_hint="Copy the column under a new name for a second equation.",
        )
    n = len(data)
    named = {c: c for s in specs for c in s["columns"]}
    for c in cl_names:
        named["cluster" if "cluster" not in named else f"cluster={c}"] = c
    missing = {k: c for k, c in named.items() if c not in data.columns}
    if missing:
        raise ColumnNotFound.from_missing(missing, data.columns, context="sp.suest")

    blocks, coefs, labels, n_obs = [], [], [], {}
    omitted: Dict[str, List[str]] = {}
    absorbed: Dict[str, List[str]] = {}
    # rows in at least one equation's estimation sample (Stata's union of
    # the e(sample)s): the clusters and the robust n are counted on them
    used = np.zeros(n, dtype=bool)
    for spec in specs:
        y = spec["y"]
        fit = _fit_equation(data, spec, tol=1e-12)
        used[fit["rows"]] = True
        F = np.zeros((n, len(fit["names"])))
        F[fit["rows"]] = fit["IF"]
        blocks.append(F)
        coefs.append(fit["b"])
        labels += [f"{y}:{t}" for t in fit["names"]]
        n_obs[y] = int(len(fit["rows"]))
        if fit["omitted"]:
            omitted[y] = fit["omitted"]
        if fit["absorbed"]:
            absorbed[y] = fit["absorbed"]
    IF = np.hstack(blocks)
    cluster_info: Dict[str, Any] = {}
    if cl_names:
        # A row that no equation uses carries no score: neither its cluster
        # nor a missing cluster value on it enters the count. Counting every
        # cluster of data inflated G (G/(G-1) too small) and a missing value
        # there was refused.
        keys = []
        for c in cl_names:
            k = data[c].to_numpy()[used]
            if pd.isna(k).any():
                raise MethodIncompatibility(
                    f"cluster has missing values on the estimation sample ({c!r})."
                )
            keys.append(k)
        counts = [int(pd.Series(k).nunique()) for k in keys]
        if min(counts) < 2:
            raise _fail(
                "sp.suest: a cluster variable has a single value on the "
                "estimation sample.",
                "Cluster on a variable with at least two values, or drop it.",
                n_clusters=dict(zip(cl_names, counts)),
            )
        if len(keys) == 1:
            codes = pd.factorize(keys[0])[0]
            G = int(codes.max()) + 1
            S = np.zeros((G, IF.shape[1]))
            np.add.at(S, codes, IF[used])
            V = (G / (G - 1)) * (S.T @ S)
            n_negative = 0
        else:
            V, counts, n_negative = _multiway_vcov(IF[used], keys, cluster_df)
            G = min(counts)
            if n_negative:
                warnings.warn(
                    "sp.suest: the multi-way clustered covariance was not "
                    f"positive semi-definite ({n_negative} negative "
                    "eigenvalue(s)); they were set to zero, as reghdfe and "
                    "fixest do. Treat joint tests with care.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        vce = f"cluster({' '.join(cl_names)})"
        cluster_info = {
            "cluster": cl_names,
            "n_clusters": counts,
            "cluster_df": cluster_df if len(cl_names) > 1 else "G/(G-1)",
            "n_negative_eigenvalues": n_negative,
        }
    else:
        m = int(used.sum())
        V = (m / (m - 1)) * (IF.T @ IF)
        G = None
        vce = "robust"
    params = pd.Series(np.concatenate(coefs), index=labels)
    return SuestResult(
        params=params,
        vcov=pd.DataFrame(V, index=labels, columns=labels),
        n_obs=n_obs,
        n_clusters=G,
        vce=vce,
        equations=names,
        cluster_info=cluster_info,
        omitted=omitted,
        absorbed=absorbed,
    )
