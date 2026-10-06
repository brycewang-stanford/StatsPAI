"""Exploratory factor analysis of a correlation matrix.

The model is ``x = L f + u`` with ``k`` common factors of unit variance and
uncorrelated unique parts, so that the correlation matrix is
``L L' + Psi`` with ``Psi`` diagonal. Four ways to estimate the loadings
``L`` are offered: principal factors (the default, as in Stata), iterated
principal factors, principal-component factors, and maximum likelihood.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from .pca import _columns, _ordered_eigen

__all__ = ["factor", "FactorResult"]

_METHODS = ("pf", "ipf", "pcf", "ml")
_NAMES = {
    "pf": "principal factors",
    "ipf": "iterated principal factors",
    "pcf": "principal-component factors",
    "ml": "maximum likelihood",
}


class FactorResult(ResultProtocolMixin):
    """Outcome of :func:`factor`.

    Attributes
    ----------
    loadings : pandas.DataFrame
        Factor loadings (pattern matrix), variables in rows, retained
        factors in columns; unrotated.
    uniqueness : pandas.Series
        One minus the communality of each variable: the share of its
        variance that the retained factors do not reproduce.
    eigenvalues : pandas.DataFrame
        ``eigenvalue``, ``difference``, ``proportion`` and ``cumulative``.
        For maximum likelihood the eigenvalues are the sums of squared
        loadings of the retained factors.
    n_factors, n_obs : int
    loglik, aic, bic : float
        Maximum likelihood only. The log likelihood is ``-n / 2``
        times the minimised discrepancy between the fitted and the sample
        correlation matrix, which is zero for a perfect fit.
    lr_independence : dict
        ``chi2``, ``df``, ``pvalue`` of the likelihood-ratio test that the
        variables are uncorrelated.
    lr_factors : dict or None
        Maximum likelihood only: the test of ``k`` factors against an
        unrestricted correlation matrix (Bartlett-corrected).
    heywood : bool
        True when a uniqueness is at its lower bound (an improper
        solution): 0.005 for maximum likelihood, below zero otherwise.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> f = rng.normal(size=400)
    >>> df = pd.DataFrame({f"x{j}": f + rng.normal(size=400) for j in range(4)})
    >>> res = sp.factor(df, method="pcf", n_factors=1)
    >>> list(res.loadings.columns)
    ['Factor1']
    >>> res.scores(df).shape
    (400, 1)
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, **fields: Any) -> None:
        self.method = f"Factor analysis / {_NAMES[fields.pop('kind')]}"
        self._correlation = fields.pop("correlation")
        self._means = fields.pop("means")
        self._scales = fields.pop("scales")
        self.loadings: pd.DataFrame = fields.pop("loadings")
        self.uniqueness: pd.Series = fields.pop("uniqueness")
        self.eigenvalues: pd.DataFrame = fields.pop("eigenvalues")
        self.n_factors = self.loadings.shape[1]
        self.n_obs: int = fields.pop("n_obs")
        self.loglik: Optional[float] = fields.pop("loglik", None)
        self.aic: Optional[float] = fields.pop("aic", None)
        self.bic: Optional[float] = fields.pop("bic", None)
        self.lr_independence: Dict[str, float] = fields.pop("lr_independence")
        self.lr_factors: Optional[Dict[str, float]] = fields.pop("lr_factors", None)
        self.heywood: bool = fields.pop("heywood", False)
        #: set by :meth:`rotate`
        self.rotation: Optional[str] = None
        self.rotation_matrix: Optional[pd.DataFrame] = None
        self.factor_correlation: Optional[pd.DataFrame] = None
        self.variance: Optional[pd.DataFrame] = None

    def rotate(
        self,
        method: str = "varimax",
        *,
        normalize: bool = False,
        power: float = 3.0,
        tol: float = 1e-12,
        maxiter: int = 5000,
    ) -> "FactorResult":
        """Rotate the retained factors.

        Parameters
        ----------
        method : {'varimax', 'promax'}, default 'varimax'
            ``'varimax'`` is the orthogonal rotation that maximises the
            variance of the squared loadings within each factor.
            ``'promax'`` starts from varimax and lets the factors
            correlate: the target is the varimax loadings raised to
            ``power`` with their signs kept.
        normalize : bool, default False
            Kaiser normalisation: each variable's loadings are scaled to
            unit length before the rotation and scaled back after it.
        power : float, default 3.0
            Promax power.

        Returns
        -------
        FactorResult
            A copy with rotated ``loadings``, ``rotation_matrix``
            (``loadings = unrotated @ rotation_matrix`` for varimax; for
            promax the matrix that rotates the factors, whose inverse
            transpose rotates the loadings), ``factor_correlation`` (the
            identity for varimax) and ``variance`` (the sum of squared
            correlations of the variables with each rotated factor).
            Uniquenesses and the tests do not change with a rotation.
            Factors are ordered by the variance they account for and
            signed so that their loadings sum to a positive number, which
            is what Stata's ``rotate`` prints.

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> import statspai as sp
        >>> rng = np.random.default_rng(0)
        >>> f, g = rng.normal(size=(2, 500))
        >>> df = pd.DataFrame({f"a{j}": f + rng.normal(size=500) for j in range(3)})
        >>> for j in range(3):
        ...     df[f"b{j}"] = g + rng.normal(size=500)
        >>> rot = sp.factor(df, method="pcf", n_factors=2).rotate("varimax")
        >>> bool((rot.loadings.abs().max(axis=1) > 0.6).all())
        True
        """
        import copy

        kind = str(method).lower()
        if kind not in ("varimax", "promax"):
            raise MethodIncompatibility(
                f"FactorResult.rotate: method={method!r} is not available.",
                recovery_hint="Use 'varimax' or 'promax'.",
            )
        L = self.loadings.to_numpy(dtype=float)
        p, k = L.shape
        if k < 2:
            raise MethodIncompatibility(
                "FactorResult.rotate: one factor cannot be rotated.",
                recovery_hint="Retain at least two factors (n_factors=2).",
            )
        scale = np.sqrt((L**2).sum(axis=1)) if normalize else np.ones(p)
        scale = np.where(scale > 0, scale, 1.0)
        A = L / scale[:, None]
        T = np.eye(k)
        last = 0.0
        for _ in range(maxiter):
            B = A @ T
            u, sv, vt = np.linalg.svd(A.T @ (B**3 - B * (B**2).sum(axis=0) / p))
            T = u @ vt
            if sv.sum() - last <= tol * max(sv.sum(), 1.0):
                break
            last = sv.sum()
        phi = np.eye(k)
        rotated = (A @ T) * scale[:, None]
        reported = T
        if kind == "promax":
            # Stata's order of operations: the target is built from the
            # (normalised) varimax loadings, and the oblique transformation
            # is fitted to the loadings in their own scale
            B = A @ T
            target = np.sign(B) * np.abs(B) ** power
            U = np.linalg.lstsq(rotated, target, rcond=None)[0]
            U = U * np.sqrt(np.diag(np.linalg.inv(U.T @ U)))
            T = T @ U
            phi = np.linalg.inv(U.T @ U)
            rotated = rotated @ U
            # for correlated factors the matrix printed is the one that
            # rotates the factors, the inverse transpose of the one that
            # rotates the loadings
            reported = np.linalg.inv(T.T)
        # the variance of a factor: squared correlations of the variables
        # with it (the structure matrix), which for uncorrelated factors
        # are the squared loadings
        structure = rotated @ phi
        variance = (structure**2).sum(axis=0)
        order = np.argsort(-variance, kind="stable")
        sign = np.where(rotated[:, order].sum(axis=0) < 0, -1.0, 1.0)
        rotated = rotated[:, order] * sign
        T = reported[:, order] * sign
        phi = phi[np.ix_(order, order)] * np.outer(sign, sign)
        variance = variance[order]
        names = list(self.loadings.columns)
        out = copy.copy(self)
        out.loadings = pd.DataFrame(rotated, index=self.loadings.index, columns=names)
        out.rotation = kind + (" (Kaiser normalisation)" if normalize else "")
        out.rotation_matrix = pd.DataFrame(T, index=names, columns=names)
        out.factor_correlation = pd.DataFrame(phi, index=names, columns=names)
        out.variance = pd.DataFrame(
            {"variance": variance, "proportion": variance / p}, index=names
        )
        return out

    def scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Regression-method factor scores, ``z R^{-1} S``, with ``z`` the
        variables standardised by the estimation sample's mean and standard
        deviation and ``S`` the correlations of the variables with the
        factors (the loadings, times the factor correlation after an
        oblique rotation)."""
        names = list(self.loadings.index)
        z = (data[names].astype(float) - self._means) / self._scales
        structure = self.loadings.to_numpy()
        if self.factor_correlation is not None:
            structure = structure @ self.factor_correlation.to_numpy()
        weights = np.linalg.solve(self._correlation, structure)
        return pd.DataFrame(
            z.to_numpy() @ weights, index=data.index, columns=self.loadings.columns
        )

    def summary(self) -> str:
        lines = [
            f"{self.method:<48}Number of obs    = {self.n_obs:>8,}",
            f"{'':<48}Retained factors = {self.n_factors:>8}",
        ]
        if self.loglik is not None:
            lines.append(
                f"Log likelihood = {self.loglik:.5f}   AIC = {self.aic:.3f}   "
                f"BIC = {self.bic:.3f}"
            )
        lines += ["", self.eigenvalues.to_string(float_format=lambda v: f"{v:.4f}")]
        t = self.lr_independence
        lines.append(
            f"LR test: independent vs. saturated: chi2({t['df']:.0f}) = "
            f"{t['chi2']:.2f}, p = {t['pvalue']:.4f}"
        )
        if self.lr_factors is not None:
            t = self.lr_factors
            lines.append(
                f"LR test: {self.n_factors} factors vs. saturated: "
                f"chi2({t['df']:.0f}) = {t['chi2']:.2f}, p = {t['pvalue']:.4f}"
            )
        lines += [
            "",
            "Factor loadings (pattern matrix) and unique variances",
            self.loadings.assign(Uniqueness=self.uniqueness).to_string(
                float_format=lambda v: f"{v:.4f}"
            ),
        ]
        if self.heywood:
            lines.append("Note: a uniqueness is at zero (Heywood case).")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _ml_loadings(R: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Loadings, uniquenesses and the minimised discrepancy
    ``log|S| - log|R| + tr(R S^{-1}) - p`` of the ``k``-factor model.

    For given uniquenesses ``psi`` the best loadings are known in closed
    form from the eigen-decomposition of ``Psi^{-1/2} R Psi^{-1/2}``, so the
    search is over ``psi`` only (Joreskog's concentrated likelihood).
    """
    p = R.shape[0]

    def parts(psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = 1.0 / np.sqrt(psi)
        values, vectors = np.linalg.eigh(R * np.outer(scale, scale))
        return values[::-1], vectors[:, ::-1]

    def objective(psi: np.ndarray) -> float:
        tail = parts(psi)[0][k:]
        return float(np.sum(tail - np.log(tail)) - (p - k))

    def gradient(psi: np.ndarray) -> np.ndarray:
        values, vectors = parts(psi)
        lead = vectors[:, :k] * np.sqrt(np.clip(values[:k] - 1.0, 0.0, None))
        load = lead * np.sqrt(psi)[:, None]
        fitted = load @ load.T + np.diag(psi)
        inv = np.linalg.inv(fitted)
        out: np.ndarray = np.diag(inv @ (fitted - R) @ inv)
        return out

    start = np.clip((1.0 - 0.5 * k / p) / np.diag(np.linalg.inv(R)), 0.005, 1.0)
    fit = optimize.minimize(
        objective,
        start,
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(0.005, 1.0)] * p,
        options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10},
    )
    psi = np.asarray(fit.x, dtype=float)
    values, vectors = parts(psi)
    lead = vectors[:, :k] * np.sqrt(np.clip(values[:k] - 1.0, 0.0, None))
    return lead * np.sqrt(psi)[:, None], psi, float(fit.fun)


def factor(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    method: str = "pf",
    n_factors: Optional[int] = None,
    min_eigenvalue: Optional[float] = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> FactorResult:
    """Exploratory factor analysis of the correlation matrix.

    Parameters
    ----------
    data : pandas.DataFrame
        The data. Rows with a missing value in any analysed variable are
        dropped.
    variables : list of str, optional
        Columns to analyse. Default: every numeric column.
    method : {'pf', 'ipf', 'pcf', 'ml'}, default 'pf'
        ``'pf'``: principal factors. The diagonal of the correlation matrix
        is replaced by the squared multiple correlation of each variable
        with the others before the eigen-decomposition.
        ``'ipf'``: the same, iterated until the communalities settle.
        ``'pcf'``: principal-component factors (communalities of one): the
        loadings are the principal components scaled by the square root of
        their eigenvalue.
        ``'ml'``: maximum likelihood under joint normality.
    n_factors : int, optional
        Number of factors to keep. Default: those with an eigenvalue above
        ``min_eigenvalue``; for ``'ml'`` it is required.
    min_eigenvalue : float, optional
        Threshold when ``n_factors`` is not given. Default 0 for ``'pf'``
        and ``'ipf'`` and 1 for ``'pcf'``, as in Stata.
    tol : float, default 1e-8
        Convergence of the communalities for ``'ipf'``.
    maxiter : int, default 1000
        Iteration limit for ``'ipf'``.

    Returns
    -------
    FactorResult
        ``loadings``, ``uniqueness``, ``eigenvalues``, the likelihood-ratio
        tests, and ``scores(data)``.

    Notes
    -----
    Loadings are identified only up to an orthogonal rotation. What is
    returned is the unrotated solution in which the factors are ordered by
    the variance they account for and each column is signed to have a
    positive sum; it reproduces Stata's ``factor``. No rotation is applied.

    The likelihood-ratio statistics carry Bartlett's correction in the
    form Stata uses: the multiplier is ``n - (2p + 5) / 6`` for independence
    and ``n - (2p + 5) / 6 - 2k / 3`` for the ``k``-factor model. (Texts
    that write the likelihood with ``n - 1`` have ``n - 1`` there too; the
    two differ by a factor ``1 - 1/n``.)

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> f = rng.normal(size=(800, 2))
    >>> load = np.array([[.8, 0], [.7, .1], [.6, .2], [0, .8], [.1, .7], [.2, .6]])
    >>> x = f @ load.T + rng.normal(scale=0.6, size=(800, 6))
    >>> df = pd.DataFrame(x, columns=[f"x{j}" for j in range(6)])
    >>> res = sp.factor(df, method="ml", n_factors=2)
    >>> res.loadings.shape
    (6, 2)
    >>> bool(res.lr_factors["pvalue"] > 0.01)
    True
    """
    method = str(method).lower()
    if method not in _METHODS:
        raise MethodIncompatibility(
            f"sp.factor: method={method!r} is not one of {', '.join(_METHODS)}.",
            recovery_hint="Use 'pf', 'ipf', 'pcf' or 'ml'.",
        )
    names, frame = _columns(data, variables, "factor")
    p, n = len(names), len(frame)
    R = np.corrcoef(frame.to_numpy(dtype=float), rowvar=False)
    if n_factors is not None and not 1 <= int(n_factors) <= p:
        raise MethodIncompatibility(
            f"sp.factor: n_factors={n_factors} is not between 1 and {p}.",
            recovery_hint="Ask for fewer factors than variables.",
        )
    if method == "ml" and n_factors is None:
        raise MethodIncompatibility(
            "sp.factor: method='ml' needs n_factors.",
            recovery_hint="Pass n_factors=; compare fits with lr_factors, "
            "aic and bic.",
        )

    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        raise MethodIncompatibility(
            "sp.factor: the correlation matrix is singular.",
            recovery_hint="Drop a variable that is a linear combination of "
            "the others.",
        )
    df_ind = p * (p - 1) / 2
    chi2_ind = -(n - (2 * p + 5) / 6) * logdet
    lr_independence = {
        "chi2": float(chi2_ind),
        "df": float(df_ind),
        "pvalue": float(stats.chi2.sf(chi2_ind, df_ind)),
    }
    extra: Dict[str, Any] = {}

    if method == "ml":
        k = int(n_factors)  # type: ignore[arg-type]
        df_k = ((p - k) ** 2 - (p + k)) / 2
        if df_k < 0:
            raise MethodIncompatibility(
                f"sp.factor: {k} factors are not identified from {p} variables.",
                recovery_hint="Ask for fewer factors.",
            )
        load, psi, discrepancy = _ml_loadings(R, k)
        values = (load**2).sum(axis=0)
        order = np.argsort(values)[::-1]
        load, values = load[:, order], values[order]
        load = load * np.where(load.sum(axis=0) < 0, -1.0, 1.0)
        shown = values
        loglik = -n / 2.0 * discrepancy
        n_par = p * k - k * (k - 1) / 2
        chi2_k = (n - (2 * p + 5) / 6 - 2 * k / 3) * discrepancy
        extra = {
            "loglik": float(loglik),
            "aic": float(-2 * loglik + 2 * n_par),
            "bic": float(-2 * loglik + n_par * np.log(n)),
            "lr_factors": (
                {
                    "chi2": float(chi2_k),
                    "df": float(df_k),
                    "pvalue": float(stats.chi2.sf(chi2_k, df_k)),
                }
                if df_k > 0
                else None
            ),
            "heywood": bool(np.any(psi <= 0.005 + 1e-9)),
        }
        uniqueness = psi
    else:
        if method == "pcf":
            communality = np.ones(p)
        else:
            communality = 1.0 - 1.0 / np.diag(np.linalg.inv(R))
        threshold = (
            float(min_eigenvalue)
            if min_eigenvalue is not None
            else (1.0 if method == "pcf" else 0.0)
        )

        def decompose(h2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
            reduced = R.copy()
            np.fill_diagonal(reduced, h2)
            vals, vecs = _ordered_eigen(reduced)
            keep = (
                int(n_factors)
                if n_factors is not None
                else max(int((vals > threshold).sum()), 1)
            )
            return vals, vecs, keep

        values, vectors, k = decompose(communality)
        if method == "ipf":
            for _ in range(maxiter):
                load = vectors[:, :k] * np.sqrt(np.clip(values[:k], 0.0, None))
                new = (load**2).sum(axis=1)
                done = np.max(np.abs(new - communality)) < tol
                communality = new
                values, vectors, _ = decompose(communality)
                if done:
                    break
        if np.any(values[:k] <= 0):
            raise MethodIncompatibility(
                f"sp.factor: only {int((values > 0).sum())} factor(s) have a "
                f"positive eigenvalue; {k} were requested.",
                recovery_hint="Ask for fewer factors, or use method='pcf'.",
            )
        load = vectors[:, :k] * np.sqrt(values[:k])
        uniqueness = 1.0 - (load**2).sum(axis=1)
        shown = values
        extra = {"heywood": bool(np.any(uniqueness < 0))}

    labels = [f"Factor{j + 1}" for j in range(len(shown))]
    total = shown.sum() if method != "pcf" else float(p)
    if method in ("pf", "ipf"):
        total = shown[shown > 0].sum() if method == "ipf" else shown.sum()
    table = pd.DataFrame(
        {
            "eigenvalue": shown,
            "difference": np.append(shown[:-1] - shown[1:], np.nan),
            "proportion": shown / total,
            "cumulative": np.cumsum(shown) / total,
        },
        index=labels,
    )
    return FactorResult(
        kind=method,
        correlation=R,
        means=frame.mean(),
        scales=frame.std(ddof=1),
        loadings=pd.DataFrame(load, index=names, columns=labels[: load.shape[1]]),
        uniqueness=pd.Series(uniqueness, index=names),
        eigenvalues=table,
        n_obs=n,
        lr_independence=lr_independence,
        **extra,
    )
