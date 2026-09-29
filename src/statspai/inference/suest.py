"""Seemingly unrelated estimation: joint covariance of separate OLS fits.

Stata ``suest`` after ``regress``: each equation is estimated on its own
sample, the per-observation influence functions ``(X_j'X_j)^{-1} x_ij e_ij``
are stacked across equations, and their cross-equation sandwich gives the
joint covariance -- robust (``n/(n-1)``) or clustered (``G/(G-1)``). That is
what a "joint test" row across outcomes needs (e.g. equality of a treatment
effect across outcomes, all effects zero), which ``sp.sureg`` (Zellner FGLS,
no clusters, common sample) cannot provide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


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
        for k in self.params.index:
            lines.append(f"  {k:<30s} {self.params[k]: .6f}  ({se[k]:.6f})")
        return "\n".join(lines)


def suest(
    data: pd.DataFrame,
    equations: Sequence[Any],
    cluster: Optional[str] = None,
) -> SuestResult:
    """Joint (seemingly unrelated) inference across separately fitted OLS equations.

    Parameters
    ----------
    data : pd.DataFrame
    equations : sequence
        Each equation is ``(y, [x1, x2, ...])`` or a formula string
        ``"y ~ x1 + x2"`` (plain column names; an intercept is added). Each
        is fitted on its own complete rows.
    cluster : str, optional
        Cluster variable: the joint covariance sums the stacked influence
        functions within clusters and scales by ``G/(G-1)`` (Stata ``suest,
        vce(cluster c)``); otherwise it is the robust ``n/(n-1)`` version.

    Returns
    -------
    SuestResult
        ``params``, joint ``vcov``, and ``wald`` / ``test_equal`` /
        ``test_zero`` for cross-equation hypotheses.

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
    """
    specs = []
    for eq in equations:
        if isinstance(eq, str):
            lhs, rhs = eq.split("~")
            xs = [t.strip() for t in rhs.split("+") if t.strip() and t.strip() != "1"]
            specs.append((lhs.strip(), xs))
        else:
            y, xs = eq
            specs.append((y, list(xs)))
    names = [s[0] for s in specs]
    if len(set(names)) != len(names):
        raise MethodIncompatibility(
            "Each equation needs a distinct dependent variable.",
            recovery_hint="Copy the column under a new name for a second equation.",
        )
    n = len(data)
    cols = {c for y, xs in specs for c in [y] + xs} | ({cluster} if cluster else set())
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"Columns not found: {missing}")
    if cluster is not None and data[cluster].isna().any():
        raise MethodIncompatibility("cluster has missing values.")

    blocks, coefs, labels, n_obs = [], [], [], {}
    for y, xs in specs:
        sub = data[[y] + xs].notna().all(axis=1).to_numpy()
        X = np.column_stack([data[xs].to_numpy(dtype=float), np.ones(n)])
        Xs, ys = X[sub], data[y].to_numpy(dtype=float)[sub]
        XtX_inv = np.linalg.inv(Xs.T @ Xs)
        b = XtX_inv @ (Xs.T @ ys)
        e = ys - Xs @ b
        F = np.zeros((n, X.shape[1]))
        F[sub] = (Xs * e[:, None]) @ XtX_inv
        blocks.append(F)
        coefs.append(b)
        labels += [f"{y}:{t}" for t in xs + ["_cons"]]
        n_obs[y] = int(sub.sum())
    IF = np.hstack(blocks)
    if cluster is not None:
        codes = pd.factorize(data[cluster])[0]
        G = int(codes.max()) + 1
        S = np.zeros((G, IF.shape[1]))
        np.add.at(S, codes, IF)
        V = (G / (G - 1)) * (S.T @ S)
        vce = f"cluster({cluster})"
    else:
        rows = np.zeros(n, dtype=bool)
        for F in blocks:
            rows |= np.any(F != 0, axis=1)
        m = int(rows.sum())
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
    )
