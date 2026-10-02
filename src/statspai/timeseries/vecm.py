"""Vector error-correction model by Johansen's maximum likelihood.

With ``K`` series integrated of order one and ``r`` cointegrating relations,

    dY_t = alpha (beta' Y_{t-1} + mu) + sum_i Gamma_i dY_{t-i} + gamma + e_t

``beta`` (``K x r``) holds the long-run relations, ``alpha`` the speed at
which each series corrects a deviation from them, and the ``Gamma_i`` the
short-run dynamics. ``beta`` is identified by Johansen's normalisation: its
first ``r`` rows are the identity matrix.

The estimates and standard errors follow Stata's ``vec``: coefficient
standard errors use the residual covariance with divisor ``T - d`` (``d``
coefficients per equation), fit statistics use the maximum-likelihood one.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["vec", "VECResult"]

_TREND_ALIASES = {
    "n": "n", "none": "n",
    "rc": "rc", "rconstant": "rc",
    "c": "c", "constant": "c",
    "rt": "rt", "rtrend": "rt",
    "ct": "ct", "trend": "ct",
}  # fmt: skip


class VECResult(ResultProtocolMixin):
    """Fitted vector error-correction model.

    Attributes
    ----------
    alpha : pandas.DataFrame
        Adjustment coefficients, one row per equation and one column per
        cointegrating relation.
    beta : pandas.DataFrame
        Cointegrating vectors (columns ``_ce1`` ...), normalised so that the
        first ``rank`` rows are the identity. Rows ``_cons`` / ``_trend``
        hold the deterministic terms inside the relation.
    beta_se : pandas.DataFrame
        Standard errors of ``beta``; ``NaN`` for the normalised entries and
        for a constant that is identified only through the equations.
    coefs : dict of DataFrame
        For each equation ``D_<name>``: ``coef`` and ``se`` of the lagged
        cointegrating relation(s) (``L._ce1``), the lagged differences
        (``LD.<name>``, ``L2D.<name>`` ...) and the deterministic terms.
    sigma_ml : pandas.DataFrame
        Maximum-likelihood residual covariance.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> common = np.cumsum(rng.normal(size=300))
    >>> df = pd.DataFrame({
    ...     "y": common + rng.normal(scale=0.5, size=300),
    ...     "x": 0.5 * common + rng.normal(scale=0.5, size=300),
    ... })
    >>> fit = sp.vec(df, ["y", "x"], lags=1, rank=1)
    >>> type(fit).__name__
    'VECResult'
    >>> list(fit.beta.index)
    ['y', 'x', '_cons']
    >>> list(fit.coefs)
    ['D_y', 'D_x']
    >>> fit.lm_test(2).shape
    (2, 4)
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("johansen1991estimation",)

    def __init__(self, **fields: Any) -> None:
        self.var_names: List[str] = fields["var_names"]
        self.rank: int = fields["rank"]
        self.lags: int = fields["lags"]
        self.trend: str = fields["trend"]
        self.n_obs: int = fields["n_obs"]
        self.alpha: pd.DataFrame = fields["alpha"]
        self.beta: pd.DataFrame = fields["beta"]
        self.beta_se: pd.DataFrame = fields["beta_se"]
        self.coefs: Dict[str, pd.DataFrame] = fields["coefs"]
        self.residuals: pd.DataFrame = fields["residuals"]
        self.sigma_ml: pd.DataFrame = fields["sigma_ml"]
        self.log_likelihood: float = fields["log_likelihood"]
        self.det_sigma: float = fields["det_sigma"]
        self.aic: float = fields["aic"]
        self.hqic: float = fields["hqic"]
        self.bic: float = fields["bic"]
        self.n_params: int = fields["n_params"]
        self.eigenvalues: np.ndarray = fields["eigenvalues"]
        self._X: np.ndarray = fields["X"]
        self._dY: np.ndarray = fields["dY"]
        self._levels: np.ndarray = fields["levels"]
        self._B: np.ndarray = fields["B"]
        self._omega: np.ndarray = fields["omega"]
        self._bread: np.ndarray = fields["bread"]

    # ------------------------------------------------------------- tables
    def equation_table(self) -> pd.DataFrame:
        """Per equation: number of coefficients, root MSE (divisor ``T -
        d``), R-squared about zero, and the Wald chi-squared that every
        coefficient of the equation is zero."""
        T, d = self._X.shape
        resid = self.residuals.to_numpy()
        inverse = np.linalg.inv(self._bread)
        rows = []
        for i, name in enumerate(self.var_names):
            rss = float(resid[:, i] @ resid[:, i])
            y = self._dY[:, i]
            b = self._B[:, i]
            chi2 = float(b @ inverse @ b) / self._omega[i, i]
            rows.append(
                {
                    "equation": f"D_{name}",
                    "parms": d,
                    "rmse": float(np.sqrt(rss / (T - d))),
                    "r2": 1.0 - rss / float(y @ y),
                    "chi2": chi2,
                    "p": float(stats.chi2.sf(chi2, d)),
                }
            )
        return pd.DataFrame(rows).set_index("equation")

    def companion(self) -> np.ndarray:
        """Companion matrix of the VAR in levels the model implies."""
        k, r, m = len(self.var_names), self.rank, self.lags
        alpha = self.alpha.to_numpy()
        beta = self.beta.to_numpy()[:k, :]
        gammas = [self._B[r + j * k : r + (j + 1) * k, :].T for j in range(m)]
        A = []
        for i in range(1, m + 2):
            a = np.zeros((k, k))
            if i == 1:
                a += np.eye(k) + alpha @ beta.T
            if i <= m:
                a += gammas[i - 1]
            if i >= 2:
                a -= gammas[i - 2]
            A.append(a)
        p = m + 1
        comp = np.zeros((k * p, k * p))
        comp[:k, :] = np.hstack(A)
        if p > 1:
            comp[k:, :-k] = np.eye(k * (p - 1))
        return comp

    def stability(self) -> pd.DataFrame:
        """Eigenvalues of the companion matrix (Stata ``vecstable``).

        A VECM with ``K`` series and rank ``r`` has ``K - r`` eigenvalues
        equal to one by construction (``attrs['unit_moduli']``); the model
        is stable when the others lie inside the unit circle
        (``attrs['stable']``).
        """
        eig = np.linalg.eigvals(self.companion())
        eig = eig[np.argsort(-np.abs(eig), kind="stable")]
        table = pd.DataFrame(
            {"real": eig.real, "imaginary": eig.imag, "modulus": np.abs(eig)}
        )
        imposed = len(self.var_names) - self.rank
        table.attrs["unit_moduli"] = imposed
        table.attrs["stable"] = bool(np.all(np.abs(eig[imposed:]) < 1.0 - 1e-8))
        return table

    def lm_test(self, lags: int = 2) -> pd.DataFrame:
        """LM test for residual autocorrelation at each lag order (Stata
        ``veclmar``): ``(T - d - 0.5) log(|Sigma| / |Sigma_s|)`` from the
        equations re-estimated, at the fitted cointegrating relation, with
        the residuals lagged ``s`` periods added (zeros where the lag does
        not exist); ``d`` counts the coefficients of one augmented
        equation. Chi-squared with ``K^2`` degrees of freedom."""
        resid = self.residuals.to_numpy()
        T, k = resid.shape
        det0 = float(np.linalg.det(resid.T @ resid / T))
        rows = []
        for s in range(1, lags + 1):
            if s >= T:
                raise DataInsufficient(
                    f"lag {s} is not available with {T} observations.",
                    recovery_hint="Lower the number of lags.",
                )
            lagged = np.zeros_like(resid)
            lagged[s:] = resid[:-s]
            aug = np.column_stack([self._X, lagged])
            u = self._dY - aug @ np.linalg.lstsq(aug, self._dY, rcond=None)[0]
            det_s = float(np.linalg.det(u.T @ u / T))
            chi2 = (T - aug.shape[1] - 0.5) * np.log(det0 / det_s)
            rows.append((s, chi2, k * k, float(stats.chi2.sf(chi2, k * k))))
        return pd.DataFrame(rows, columns=["lag", "chi2", "df", "p"])

    def summary(self) -> str:
        lines = [
            "Vector error-correction model",
            "=" * 66,
            f"Rank: {self.rank:<6d} Lagged differences: {self.lags:<6d} "
            f"N obs: {self.n_obs}",
            f"Log-lik: {self.log_likelihood:.4f}   AIC: {self.aic:.4f}   "
            f"HQIC: {self.hqic:.4f}   SBIC: {self.bic:.4f}",
            "=" * 66,
        ]
        for eq, table in self.coefs.items():
            lines += ["", f"Equation: {eq}", "-" * 66]
            lines.append(f"{'':<16s}{'Coef':>12s}{'SE':>12s}{'z':>9s}{'P>|z|':>9s}")
            for name, row in table.iterrows():
                z = row["coef"] / row["se"] if row["se"] > 0 else np.nan
                lines.append(
                    f"{name:<16s}{row['coef']:>12.6f}{row['se']:>12.6f}"
                    f"{z:>9.2f}{2 * stats.norm.sf(abs(z)):>9.3f}"
                )
        lines += ["", "Cointegrating equations (beta)", "-" * 66]
        lines.append(self.beta.to_string(float_format=lambda v: f"{v:.6f}"))
        return "\n".join(lines)


def _residualise(A: np.ndarray, Z: np.ndarray) -> np.ndarray:
    if Z.shape[1] == 0:
        return A
    return A - Z @ np.linalg.lstsq(Z, A, rcond=None)[0]


def vec(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    lags: int = 1,
    rank: int = 1,
    trend: str = "c",
    alpha: float = 0.05,
) -> VECResult:
    """Vector error-correction model (Johansen maximum likelihood).

    Parameters
    ----------
    data : pandas.DataFrame
        The series in levels, in time order.
    variables : sequence of str, optional
        The series of the system. Default: every numeric column. Their order
        matters for the normalisation: the first ``rank`` are the ones each
        cointegrating relation is solved for.
    lags : int, default 1
        Number of lagged **differences**, as in :func:`statspai.johansen`.
        Stata's ``vec, lags(p)`` counts lags of the levels: ``lags = p - 1``.
    rank : int, default 1
        Number of cointegrating relations, ``1 <= rank < K``. Choose it
        with :func:`statspai.johansen`.
    trend : {'c', 'rc', 'n', 'rt', 'ct'}, default 'c'
        ``'c'``: unrestricted constant (the levels may drift); ``'rc'``:
        constant inside the cointegrating relation only; ``'n'``: no
        deterministic term; ``'rt'``: a linear trend inside the relation and
        an unrestricted constant; ``'ct'``: unrestricted constant and trend
        (a quadratic trend in the levels). Stata's names ``constant`` /
        ``rconstant`` / ``none`` / ``rtrend`` / ``trend`` are accepted.
    alpha : float, default 0.05
        One minus the confidence level (kept on the result for reporting).

    Returns
    -------
    VECResult
        ``alpha``, ``beta`` (with ``beta_se``), per-equation coefficient
        tables in ``coefs``, the log likelihood and information criteria,
        and the methods ``equation_table()``, ``stability()``,
        ``lm_test()`` and ``summary()``.

    Notes
    -----
    With an unrestricted constant ``v`` the model writes ``v = alpha * mu +
    gamma`` with ``gamma`` orthogonal to ``alpha``: ``mu`` is reported as
    the constant of the cointegrating relation (no standard error, it is
    not separately estimated) and ``gamma`` as the constant of each
    equation.

    The cointegrating vector is super-consistent, but its normalisation is a
    choice: a relation that is insignificant in the first variable cannot be
    solved for it. If ``alpha`` of an equation is not different from zero,
    that series is weakly exogenous for the long-run parameters.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> common = np.cumsum(rng.normal(size=400))
    >>> df = pd.DataFrame({
    ...     "y": common + rng.normal(scale=0.5, size=400),
    ...     "x": 0.5 * common + rng.normal(scale=0.5, size=400),
    ... })
    >>> fit = sp.vec(df, ["y", "x"], lags=1, rank=1)
    >>> bool(abs(fit.beta.loc["x", "_ce1"] + 2.0) < 0.3)
    True
    >>> int(fit.stability().attrs["unit_moduli"])
    1

    References
    ----------
    johansen1991estimation; lutkepohl2005new
    """
    case = _TREND_ALIASES.get(str(trend).lower())
    if case is None:
        raise MethodIncompatibility(
            f"sp.vec: trend={trend!r} is not one of 'c', 'rc', 'n', 'rt', 'ct' "
            "(or Stata's constant / rconstant / none / rtrend / trend).",
            recovery_hint="Use trend='c' for an unrestricted constant.",
        )
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    names = list(variables)
    unknown = [v for v in names if v not in data.columns]
    if unknown or len(names) < 2:
        raise MethodIncompatibility(
            (
                f"sp.vec: variable(s) {unknown} are not in the data."
                if unknown
                else "sp.vec needs at least two series."
            ),
            recovery_hint="Name the series of the system.",
        )
    k = len(names)
    lags, rank = int(lags), int(rank)
    if lags < 0 or not 1 <= rank < k:
        raise MethodIncompatibility(
            f"sp.vec: lags={lags}, rank={rank} with {k} series; need lags >= 0 "
            f"and 1 <= rank <= {k - 1}.",
            recovery_hint="rank = K means the levels are stationary (fit "
            "sp.var); rank = 0 means no cointegration (fit a VAR in "
            "differences).",
        )
    levels = data[names].dropna().to_numpy(dtype=float)
    n = levels.shape[0]
    T = n - lags - 1
    d = rank + k * lags + {"n": 0, "rc": 0, "c": 1, "rt": 1, "ct": 2}[case]
    if T <= d + k:
        raise DataInsufficient(
            f"sp.vec: {n} observations are too few for {lags} lagged "
            f"difference(s) of {k} series.",
            recovery_hint="Use fewer lags or a longer sample.",
        )

    dY = np.diff(levels, axis=0)
    target = dY[lags:]
    lagged_levels = levels[lags:-1]
    blocks = [dY[lags - j : len(dY) - j] for j in range(1, lags + 1)]
    Z = np.hstack(blocks) if blocks else np.empty((T, 0))
    # the trend counts the estimation sample: 1 at its first observation
    # (Stata's convention; it fixes the constants, not the slopes)
    clock = np.arange(1, T + 1, dtype=float)
    if case in ("c", "rt", "ct"):
        Z = np.column_stack([Z, np.ones(T)])
    if case == "ct":
        Z = np.column_stack([Z, clock])
    if case == "rc":
        lagged_levels = np.column_stack([lagged_levels, np.ones(T)])
    elif case == "rt":
        # the relation is dated t - 1, and so is its trend
        lagged_levels = np.column_stack([lagged_levels, clock - 1.0])
    k1 = lagged_levels.shape[1]

    R0, R1 = _residualise(target, Z), _residualise(lagged_levels, Z)
    S00, S11, S01 = R0.T @ R0 / T, R1.T @ R1 / T, R0.T @ R1 / T
    chol = np.linalg.cholesky(S11)
    chol_inv = np.linalg.inv(chol)
    sym = chol_inv @ S01.T @ np.linalg.solve(S00, S01) @ chol_inv.T
    eigvals, eigvecs = np.linalg.eigh((sym + sym.T) / 2.0)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    V = chol_inv.T @ eigvecs[:, order]
    head = V[:rank, :rank]
    if abs(np.linalg.det(head)) < 1e-12:
        raise MethodIncompatibility(
            "sp.vec: the cointegrating relation cannot be normalised on the "
            f"first {rank} variable(s); they do not enter it.",
            recovery_hint="Reorder `variables` so that a series that belongs "
            "to the long-run relation comes first.",
        )
    beta = V[:, :rank] @ np.linalg.inv(head)

    ce = lagged_levels @ beta
    X = np.column_stack([ce, Z])
    B = np.linalg.lstsq(X, target, rcond=None)[0]
    resid = target - X @ B
    sigma_ml = resid.T @ resid / T
    # degrees of freedom of the coefficient covariance: the coefficients of
    # one equation, plus the constant when it sits inside the relation
    d_omega = d + (1 if case in ("rc", "rt") else 0)
    omega = resid.T @ resid / (T - d_omega)
    a = B[:rank].T  # K x r

    inside = {"rc": ["_cons"], "rt": ["_trend"]}.get(case, [])
    beta_full = pd.DataFrame(
        beta,
        index=names + inside,
        columns=[f"_ce{j + 1}" for j in range(rank)],
    )
    # Var(free rows of beta) = (H' S11 H)^{-1} (x) (alpha' Omega^{-1} alpha)^{-1} / T
    free = list(range(rank, k1))
    cov_beta = (
        np.kron(
            np.linalg.inv(S11[np.ix_(free, free)]),
            np.linalg.inv(a.T @ np.linalg.inv(omega) @ a),
        )
        / T
    )
    se_free = np.sqrt(np.diag(cov_beta)).reshape(len(free), rank)
    beta_se = pd.DataFrame(np.nan, index=beta_full.index, columns=beta_full.columns)
    beta_se.iloc[rank:, :] = se_free

    if case in ("c", "rt", "ct"):
        # An unrestricted term v splits into the part alpha can reach, which
        # belongs to the relation, and the rest, which stays in the
        # equations: v = alpha * mu + gamma with gamma orthogonal to alpha.
        # The equation coefficients and their standard errors are those of
        # the regression on the relation that carries mu (and rho).
        project = np.linalg.solve(a.T @ a, a.T)
        shift = np.zeros((T, rank))
        if case == "ct":
            rho = project @ B[-1]
            beta_full.loc["_trend"] = rho
            beta_se.loc["_trend"] = np.nan
            shift = shift + np.outer(clock - 1.0, rho)  # dated t - 1
            mu = project @ B[-2]
        else:
            mu = project @ B[-1]
        beta_full.loc["_cons"] = mu
        beta_se.loc["_cons"] = np.nan
        X = np.column_stack([ce + mu + shift, Z])
        B = np.linalg.lstsq(X, target, rcond=None)[0]
    bread = np.linalg.inv(X.T @ X)

    labels = [f"L._ce{j + 1}" for j in range(rank)]
    for j in range(1, lags + 1):
        prefix = "LD." if j == 1 else f"L{j}D."
        labels += [prefix + name for name in names]
    if case in ("c", "rt", "ct"):
        labels.append("_cons")
    if case == "ct":
        labels.append("_trend")
    coefs = {}
    for i, name in enumerate(names):
        coefs[f"D_{name}"] = pd.DataFrame(
            {"coef": B[:, i], "se": np.sqrt(omega[i, i] * np.diag(bread))},
            index=labels,
        )

    det = float(np.linalg.det(sigma_ml))
    ll = -0.5 * T * k * (1.0 + np.log(2.0 * np.pi)) - 0.5 * T * np.log(det)
    n_params = (
        k * rank
        + (k1 - rank) * rank
        + k * k * lags
        + k * {"n": 0, "rc": 0, "c": 1, "rt": 1, "ct": 2}[case]
    )
    return VECResult(
        var_names=names,
        rank=rank,
        lags=lags,
        trend=case,
        n_obs=T,
        alpha=pd.DataFrame(
            a, index=[f"D_{v}" for v in names], columns=beta_full.columns
        ),
        beta=beta_full,
        beta_se=beta_se,
        coefs=coefs,
        residuals=pd.DataFrame(resid, columns=names),
        sigma_ml=pd.DataFrame(sigma_ml, index=names, columns=names),
        log_likelihood=float(ll),
        det_sigma=det,
        aic=(-2.0 * ll + 2.0 * n_params) / T,
        hqic=(-2.0 * ll + 2.0 * np.log(np.log(T)) * n_params) / T,
        bic=(-2.0 * ll + np.log(T) * n_params) / T,
        n_params=int(n_params),
        eigenvalues=np.clip(eigvals[:k], 0.0, 1.0),
        X=X,
        dY=target,
        levels=levels,
        B=B,
        omega=omega,
        bread=bread,
    )
