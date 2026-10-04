"""Structural vector autoregressions: short-run, long-run and sign restrictions.

A reduced-form VAR has innovations ``e_t`` with covariance ``Sigma``. A
structural VAR writes them as ``e_t = P u_t`` with uncorrelated unit-variance
shocks ``u_t``; ``P P' = Sigma`` leaves ``K (K - 1) / 2`` degrees of freedom
that the data cannot pin down. Three ways of pinning them down are offered:

* **short-run** restrictions on the contemporaneous matrices of the AB model
  ``A e_t = B u_t`` (so ``P = A^{-1} B``), estimated by maximum likelihood
  (Stata ``svar, aeq() beq()``);
* **long-run** restrictions on ``C = (I - A_1 - ... - A_p)^{-1} P``, the
  cumulative effect of the shocks (Blanchard and Quah; Stata ``svar,
  lreq()``);
* **sign** restrictions on the impulse responses, which give a set of
  admissible ``P`` instead of one (Uhlig; Rubio-Ramirez, Waggoner and Zha).

The reduced form is taken from :func:`statspai.var` as estimated; this
module only adds the identification step.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, IdentificationFailure, MethodIncompatibility
from .var import VARResult

__all__ = ["svar", "SVARResult"]

_Matrix = Union[np.ndarray, Sequence[Sequence[float]], pd.DataFrame]


# ------------------------------------------------------------ reduced form
def _lag_matrices(var_result: VARResult) -> List[np.ndarray]:
    B = var_result._B
    if B is None:
        raise MethodIncompatibility(
            "svar: the VAR result carries no coefficient matrices.",
            recovery_hint="Pass the result of sp.var(...).",
        )
    k, p = var_result._k, var_result._lags
    return [
        np.asarray(B[lag * k : (lag + 1) * k, :], dtype=float).T for lag in range(p)
    ]


def ma_coefficients(lag_matrices: List[np.ndarray], periods: int) -> np.ndarray:
    """``Phi_0 .. Phi_periods`` of the moving-average representation."""
    k = lag_matrices[0].shape[0]
    p = len(lag_matrices)
    phi = np.zeros((periods + 1, k, k))
    phi[0] = np.eye(k)
    for s in range(1, periods + 1):
        for j in range(min(s, p)):
            phi[s] += phi[s - j - 1] @ lag_matrices[j]
    return phi


def fevd_shares(theta: np.ndarray) -> np.ndarray:
    """Forecast-error variance shares from structural responses
    ``theta[s, response, shock]``.

    ``out[h, response, shock]`` is the share of the ``h``-step forecast-error
    variance of ``response`` due to ``shock``, using ``theta[0 .. h - 1]``
    (so ``h = 0`` is all zeros, as Stata's ``irf table fevd`` prints it).
    """
    cum = np.cumsum(theta**2, axis=0)
    out = np.zeros_like(theta)
    total = cum.sum(axis=2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = np.where(total[:-1] > 0, cum[:-1] / total[:-1], 0.0)
    return out


# ------------------------------------------------------------------ result
class SVARResult(ResultProtocolMixin):
    """Outcome of :func:`svar`.

    Attributes
    ----------
    identification : str
        ``"short-run"``, ``"long-run"`` or ``"sign"``.
    impact : pandas.DataFrame
        The structural impact matrix ``P`` (rows: variables, columns:
        shocks): the response in the same period to a one-standard-deviation
        shock. With sign restrictions, the pointwise median over the
        admissible set (which is not itself an admissible matrix).
    A, B : pandas.DataFrame or None
        Short-run identification: the estimated matrices of ``A e = B u``.
    C : pandas.DataFrame or None
        Long-run identification: the long-run impact matrix.
    table : pandas.DataFrame or None
        The free parameters with standard error, z and p-value.
    log_likelihood : float or None
    overid : dict or None
        Over-identified models: the likelihood-ratio test of the
        over-identifying restrictions (``statistic``, ``df``, ``pvalue``).
    n_accepted, n_tried : int or None
        Sign restrictions: draws kept, and rotations tried to get them.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> T = 300
    >>> u = rng.normal(size=(T, 2))
    >>> e = u @ np.array([[1.0, 0.0], [0.5, 0.8]]).T
    >>> y = np.zeros((T, 2))
    >>> for t in range(1, T):
    ...     y[t] = 0.5 * y[t - 1] + e[t]
    >>> fit = sp.var(pd.DataFrame(y, columns=["x", "y"]), lags=1)
    >>> res = sp.svar(fit, B=[[np.nan, 0], [np.nan, np.nan]])
    >>> res.identification
    'short-run'
    >>> res.impact.shape
    (2, 2)
    >>> res.irf(4).shape   # (periods + 1) x variables x shocks, long format
    (20, 4)
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("lutkepohl2005new",)

    def __init__(
        self,
        *,
        identification: str,
        var_names: List[str],
        shock_names: List[str],
        impact: np.ndarray,
        lag_matrices: List[np.ndarray],
        n_obs: int,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        C: Optional[np.ndarray] = None,
        table: Optional[pd.DataFrame] = None,
        log_likelihood: Optional[float] = None,
        overid: Optional[Dict[str, float]] = None,
        draws: Optional[np.ndarray] = None,
        n_tried: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:
        self.identification = identification
        self.method = f"Structural VAR ({identification} restrictions)"
        self.var_names = var_names
        self.shock_names = shock_names
        self.n_obs = n_obs
        self.alpha = alpha
        self.table = table
        self.log_likelihood = log_likelihood
        self.overid = overid
        self._lag_matrices = lag_matrices
        self._impact = impact
        self._draws = draws
        self.n_accepted = None if draws is None else int(draws.shape[0])
        self.n_tried = n_tried

        def frame(m: Optional[np.ndarray], cols: List[str]) -> Optional[pd.DataFrame]:
            return None if m is None else pd.DataFrame(m, index=var_names, columns=cols)

        self.impact = frame(impact, shock_names)
        self.A = frame(A, var_names)
        self.B = frame(B, shock_names)
        self.C = frame(C, shock_names)

    # ---------------------------------------------------------- responses
    def _theta(self, periods: int, impact: np.ndarray) -> np.ndarray:
        phi = ma_coefficients(self._lag_matrices, periods)
        return np.asarray(np.einsum("sij,jk->sik", phi, impact))

    def _long(self, values: Dict[str, np.ndarray], periods: int) -> pd.DataFrame:
        rows = []
        for j, shock in enumerate(self.shock_names):
            for i, name in enumerate(self.var_names):
                for s in range(periods + 1):
                    row = {"shock": shock, "response": name, "period": s}
                    row.update({key: float(v[s, i, j]) for key, v in values.items()})
                    rows.append(row)
        return pd.DataFrame(rows)

    def irf(self, periods: int = 20, cumulative: bool = False) -> pd.DataFrame:
        """Structural impulse responses, one row per shock, response and
        period.

        Column ``irf`` is the response to a one-standard-deviation shock.
        With sign restrictions it is the pointwise median over the
        admissible set and ``lower`` / ``upper`` are its ``alpha / 2`` and
        ``1 - alpha / 2`` quantiles: the spread of the identified set at the
        estimated reduced form, not a confidence interval.
        """
        if periods < 0:
            raise MethodIncompatibility("svar: periods must be non-negative.")

        def path(impact: np.ndarray) -> np.ndarray:
            theta = self._theta(periods, impact)
            return np.cumsum(theta, axis=0) if cumulative else theta

        if self._draws is None:
            return self._long({"irf": path(self._impact)}, periods)
        stack = np.stack([path(p) for p in self._draws])
        lo, hi = 100 * self.alpha / 2, 100 * (1 - self.alpha / 2)
        return self._long(
            {
                "irf": np.median(stack, axis=0),
                "lower": np.percentile(stack, lo, axis=0),
                "upper": np.percentile(stack, hi, axis=0),
            },
            periods,
        )

    def fevd(self, periods: int = 20) -> pd.DataFrame:
        """Forecast-error variance decomposition: the share of the
        ``period``-step forecast-error variance of each response due to each
        shock (zero at ``period = 0``, as in Stata's ``irf table fevd``).
        With sign restrictions, the pointwise median over the admissible
        set, with its quantile band."""
        if self._draws is None:
            return self._long(
                {"fevd": fevd_shares(self._theta(periods, self._impact))}, periods
            )
        stack = np.stack([fevd_shares(self._theta(periods, p)) for p in self._draws])
        lo, hi = 100 * self.alpha / 2, 100 * (1 - self.alpha / 2)
        return self._long(
            {
                "fevd": np.median(stack, axis=0),
                "lower": np.percentile(stack, lo, axis=0),
                "upper": np.percentile(stack, hi, axis=0),
            },
            periods,
        )

    def summary(self) -> str:
        lines = [self.method, "=" * len(self.method)]
        lines.append(f"Observations: {self.n_obs}")
        if self.log_likelihood is not None:
            lines.append(f"Log likelihood: {self.log_likelihood:.4f}")
        if self.table is not None:
            lines += ["", self.table.to_string(float_format=lambda v: f"{v:.6g}")]
        if self.overid is not None:
            lines.append(
                "LR test of over-identifying restrictions: "
                f"chi2({self.overid['df']:g}) = {self.overid['statistic']:.4f}, "
                f"p = {self.overid['pvalue']:.4f}"
            )
        if self._draws is not None:
            lines.append(
                f"Admissible rotations kept: {self.n_accepted} of {self.n_tried} "
                "tried. Bands describe the identified set at the estimated "
                "reduced form, not sampling uncertainty."
            )
        lines += [
            "",
            "Impact matrix (rows: variables, columns: shocks)",
            pd.DataFrame(
                self._impact, index=self.var_names, columns=self.shock_names
            ).to_string(float_format=lambda v: f"{v:.6g}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SVARResult({self.identification} restrictions, K={len(self.var_names)})"
        )


# ------------------------------------------------------- AB-model likelihood
def _pattern(value: Optional[_Matrix], k: int, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(
        value.to_numpy() if isinstance(value, pd.DataFrame) else value, dtype=float
    )
    if arr.shape != (k, k):
        raise MethodIncompatibility(
            f"svar: {name} must be a {k} x {k} matrix, got shape {arr.shape}.",
            recovery_hint="One row and one column per variable of the VAR; "
            "np.nan marks a free element.",
        )
    return arr


def _loglik(M: np.ndarray, sigma: np.ndarray, T: int) -> float:
    k = sigma.shape[0]
    sign, logdet = np.linalg.slogdet(M)
    if sign == 0:
        return -np.inf
    return float(
        -0.5 * T * k * np.log(2.0 * np.pi)
        + T * logdet
        - 0.5 * T * np.trace(M @ sigma @ M.T)
    )


def _fit_ab(
    sigma: np.ndarray, T: int, A0: np.ndarray, B0: np.ndarray, what: str
) -> Dict[str, Any]:
    """Maximum likelihood for ``A e = B u`` with ``np.nan`` marking the free
    elements of ``A0`` / ``B0``."""
    k = sigma.shape[0]
    free_a, free_b = np.isnan(A0), np.isnan(B0)
    na, nb = int(free_a.sum()), int(free_b.sum())
    n_free = na + nb
    n_moments = k * (k + 1) // 2
    if n_free == 0:
        raise MethodIncompatibility(
            f"svar: the {what} restrictions leave nothing to estimate.",
            recovery_hint="Mark the free elements with np.nan.",
        )
    if n_free > n_moments:
        raise IdentificationFailure(
            f"svar: {n_free} free parameters cannot be identified from the "
            f"{n_moments} distinct elements of the residual covariance.",
            recovery_hint=f"Restrict at least {n_free - n_moments} more "
            "element(s) (the order condition).",
            diagnostics={"n_free": n_free, "n_moments": n_moments},
        )
    scale = float(np.sqrt(np.mean(np.diag(sigma))))

    def unpack(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        A, B = A0.copy(), B0.copy()
        A[free_a] = theta[:na]
        B[free_b] = theta[na:] * scale
        return A, B

    def objective(
        theta: np.ndarray, sigma: np.ndarray = sigma
    ) -> Tuple[float, np.ndarray]:
        A, B = unpack(theta)
        try:
            B_inv = np.linalg.inv(B)
            M = B_inv @ A
            M_inv_t = np.linalg.inv(M).T
        except np.linalg.LinAlgError:
            return 1e300, np.zeros_like(theta)
        ll = _loglik(M, sigma, T)
        if not np.isfinite(ll):
            return 1e300, np.zeros_like(theta)
        G = T * (M_inv_t - M @ sigma)
        grad_a = B_inv.T @ G
        grad_b = -B_inv.T @ G @ M.T
        grad = np.concatenate([grad_a[free_a], grad_b[free_b] * scale])
        return -ll, -grad

    # start: A free elements at zero, B from the Cholesky factor of the
    # covariance of A e; then a few perturbed restarts for patterns the
    # Cholesky start does not suit
    A_start = np.where(free_a, 0.0, A0)
    try:
        chol = np.linalg.cholesky(A_start @ sigma @ A_start.T)
    except np.linalg.LinAlgError:
        chol = np.diag(np.sqrt(np.diag(sigma)))
    base_b = np.where(free_b, chol, 0.0)[free_b] / scale
    empty_diag = free_b & np.eye(k, dtype=bool) & (np.where(free_b, chol, 0.0) == 0)
    if empty_diag.any():
        start_b = np.where(free_b, chol, 0.0)
        start_b[empty_diag] = np.sqrt(np.diag(sigma))[np.diag(empty_diag)]
        base_b = start_b[free_b] / scale
    rng = np.random.default_rng(0)
    best: Optional[optimize.OptimizeResult] = None
    for attempt in range(6):
        start = np.concatenate([np.zeros(na), base_b])
        if attempt:
            start = start + rng.normal(scale=0.3, size=start.size)
        res = optimize.minimize(
            objective, start, jac=True, method="BFGS",
            options={"gtol": 1e-10, "maxiter": 2000},
        )  # fmt: skip
        if np.isfinite(res.fun) and res.fun < 1e299:
            if best is None or res.fun < best.fun - 1e-9:
                best = res
            if attempt == 0 and float(np.max(np.abs(res.jac))) < 1e-6:
                break
    if best is None:
        raise IdentificationFailure(
            f"svar: the likelihood could not be evaluated under the {what} "
            "restrictions (A or B is singular for every start).",
            recovery_hint="Check that no row or column of A or B is "
            "restricted to zero.",
        )
    theta = best.x

    # Newton polish and the information matrix, from the analytic score
    def hessian(t: np.ndarray, at: np.ndarray = sigma) -> np.ndarray:
        H = np.zeros((t.size, t.size))
        for i in range(t.size):
            h = 1e-5 * max(1.0, abs(t[i]))
            step = np.zeros(t.size)
            step[i] = h
            up, down = objective(t + step, at)[1], objective(t - step, at)[1]
            H[:, i] = (up - down) / (2 * h)
        return 0.5 * (H + H.T)

    def gradient(t: np.ndarray) -> np.ndarray:
        return objective(t)[1]

    for _ in range(5):
        H = hessian(theta)
        try:
            delta = np.linalg.solve(H, gradient(theta))
        except np.linalg.LinAlgError:
            break
        if objective(theta - delta)[0] > objective(theta)[0] + 1e-10:
            break
        theta = theta - delta
        if float(np.max(np.abs(delta))) < 1e-12:
            break
    H = hessian(theta)
    eig = np.linalg.eigvalsh(H)
    if eig.min() <= 1e-8 * max(1.0, eig.max()):
        raise IdentificationFailure(
            f"svar: the {what} restrictions do not identify the model: the "
            "likelihood is flat in some direction of the free parameters.",
            recovery_hint="Each shock needs restrictions that tell it apart "
            "from the others; a recursive (triangular) pattern always does.",
            diagnostics={"smallest_eigenvalue": float(eig.min())},
        )
    A, B = unpack(theta)
    # sign normalisation: a free diagonal element of B is positive
    flip = np.ones(k)
    for j in range(k):
        if free_b[j, j] and B[j, j] < 0 and not free_a.any():
            flip[j] = -1.0
    B = B * flip
    # Standard errors from the expected information: the Hessian with the
    # sample covariance replaced by the one the model implies. The two
    # coincide in an exactly identified model; with over-identifying
    # restrictions this is the estimator Stata's svar reports.
    P = np.linalg.solve(A, B)
    cov = np.linalg.inv(hessian(theta, P @ P.T))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    se[na:] *= scale
    M = np.linalg.inv(B) @ A
    return {
        "A": A, "B": B, "se": se, "free_a": free_a, "free_b": free_b,
        "loglik": _loglik(M, sigma, T), "n_free": n_free,
        "df_overid": n_moments - n_free,
    }  # fmt: skip


def _table(fit: Dict[str, Any], labels: Tuple[str, str]) -> pd.DataFrame:
    rows = []
    position = 0
    for key, free, label in (
        ("A", fit["free_a"], labels[0]),
        ("B", fit["free_b"], labels[1]),
    ):
        matrix = fit[key]
        for i, j in zip(*np.nonzero(free)):
            est, se = float(matrix[i, j]), float(fit["se"][position])
            z = est / se if se > 0 else np.nan
            rows.append(
                {
                    "parameter": f"{label}[{i + 1},{j + 1}]",
                    "estimate": est,
                    "se": se,
                    "z": z,
                    "pvalue": float(2 * stats.norm.sf(abs(z))),
                }
            )
            position += 1
    return pd.DataFrame(rows).set_index("parameter")


# -------------------------------------------------------------------- signs
def _sign_spec(
    sign: Mapping[str, Mapping[str, Any]], var_names: List[str]
) -> List[Tuple[str, np.ndarray]]:
    spec = []
    for shock, wanted in sign.items():
        row = np.zeros(len(var_names))
        for name, direction in wanted.items():
            if name not in var_names:
                raise MethodIncompatibility(
                    f"svar: sign restriction on {name!r}, which is not a "
                    "variable of the VAR.",
                    recovery_hint=f"Variables: {var_names}.",
                )
            value = {"+": 1.0, "-": -1.0}.get(direction, direction)  # type: ignore[call-overload]
            if value not in (1, -1, 1.0, -1.0):
                raise MethodIncompatibility(
                    f"svar: the sign of {name!r} under shock {shock!r} must "
                    f"be '+' or '-', got {direction!r}.",
                    recovery_hint="Leave a variable out to leave its "
                    "response unrestricted.",
                )
            row[var_names.index(name)] = float(value)
        if not row.any():
            raise MethodIncompatibility(
                f"svar: shock {shock!r} has no sign restriction.",
                recovery_hint="Give at least one response a sign.",
            )
        spec.append((str(shock), row))
    if not spec:
        raise MethodIncompatibility(
            "svar: sign= is empty.",
            recovery_hint="sign={'demand': {'output': '+', 'prices': '+'}}",
        )
    if len(spec) > len(var_names):
        raise MethodIncompatibility(
            f"svar: {len(spec)} shocks for {len(var_names)} variables.",
            recovery_hint="A VAR in K variables has K shocks.",
        )
    return spec


def _sign_draws(
    sigma: np.ndarray,
    lag_matrices: List[np.ndarray],
    spec: List[Tuple[str, np.ndarray]],
    horizon: int,
    n_draws: int,
    max_tries: int,
    seed: Optional[int],
) -> Tuple[np.ndarray, int]:
    k = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)
    phi = ma_coefficients(lag_matrices, horizon)
    rng = np.random.default_rng(seed)
    kept: List[np.ndarray] = []
    tried = 0
    while len(kept) < n_draws and tried < max_tries:
        tried += 1
        # a rotation uniform on the orthogonal group: QR of a Gaussian
        # matrix with the diagonal of R made positive
        Q, R = np.linalg.qr(rng.standard_normal((k, k)))
        Q = Q * np.sign(np.diag(R))
        impact = chol @ Q
        theta = np.einsum("sij,jk->sik", phi, impact)  # s, response, column
        columns: List[int] = []
        flips: List[float] = []
        ok = True
        for _, row in spec:
            restricted = row != 0
            found = False
            for j in range(k):
                if j in columns:
                    continue
                path = theta[:, restricted, j] * row[restricted]
                if np.all(path > 0):
                    columns.append(j)
                    flips.append(1.0)
                    found = True
                elif np.all(path < 0):
                    columns.append(j)
                    flips.append(-1.0)
                    found = True
                if found:
                    break
            if not found:
                ok = False
                break
        if not ok:
            continue
        rest = [j for j in range(k) if j not in columns]
        order = columns + rest
        signs = np.array(flips + [1.0] * len(rest))
        kept.append(impact[:, order] * signs)
    if not kept:
        raise IdentificationFailure(
            f"svar: none of {tried} rotations satisfied the sign restrictions.",
            recovery_hint="The restrictions may contradict the estimated "
            "reduced form; relax them, shorten sign_horizon or raise "
            "max_tries.",
            diagnostics={"n_tried": tried},
        )
    return np.stack(kept), tried


# --------------------------------------------------------------------- svar
def svar(
    var_result: VARResult,
    *,
    A: Optional[_Matrix] = None,
    B: Optional[_Matrix] = None,
    long_run: Optional[_Matrix] = None,
    sign: Optional[Mapping[str, Mapping[str, Any]]] = None,
    sign_horizon: int = 0,
    n_draws: int = 1000,
    max_tries: int = 200_000,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> SVARResult:
    """Structural VAR: identify the shocks behind a fitted :func:`var`.

    Exactly one kind of restriction is given.

    ============================================  ==========================
    Call                                          Stata
    ============================================  ==========================
    ``sp.svar(fit, A=A, B=B)``                    ``svar ..., aeq(A) beq(B)``
    ``sp.svar(fit, long_run=C)``                  ``svar ..., lreq(C)``
    ``sp.svar(fit, sign={...})``                  no official command
    ============================================  ==========================

    Parameters
    ----------
    var_result : VARResult
        A VAR fitted by :func:`statspai.var`. The identification uses its
        residual covariance with divisor ``T`` (the maximum-likelihood
        estimate, as Stata's ``svar``).
    A, B : K x K array-like, optional
        Short-run restrictions in the AB model ``A e_t = B u_t``. A number
        fixes an element, ``np.nan`` leaves it free. With only ``B`` given,
        ``A`` is the identity; with only ``A`` given, ``B`` is diagonal with
        a free diagonal. A lower-triangular ``B`` with ``A = I`` is the
        recursive (Cholesky) identification in the order of the variables.
    long_run : K x K array-like, optional
        Restrictions on ``C = (I - A_1 - ... - A_p)^{-1} P``, the cumulative
        response of each variable to each shock, in the same notation. A
        lower-triangular pattern is Blanchard and Quah's: the second shock
        has no long-run effect on the first variable, and so on.
    sign : mapping, optional
        ``{shock name: {variable: '+' or '-'}}``: the sign of the response
        of the named variables to each named shock. Variables left out are
        unrestricted, and so are shocks left out.
    sign_horizon : int, default 0
        The sign restrictions hold for the responses at periods
        ``0 .. sign_horizon``.
    n_draws : int, default 1000
        Admissible rotations to collect.
    max_tries : int, default 200000
        Rotations to try before giving up collecting ``n_draws``.
    seed : int, optional
        Seed of the rotation draws.
    alpha : float, default 0.05
        Sign restrictions: the bands of ``irf`` / ``fevd`` are the
        ``alpha / 2`` and ``1 - alpha / 2`` quantiles over the draws.

    Returns
    -------
    SVARResult
        ``.impact`` is the structural impact matrix; ``.irf(periods)`` and
        ``.fevd(periods)`` give the structural impulse responses and the
        variance decomposition; ``.table`` the free parameters with
        standard errors (short- and long-run).

    Raises
    ------
    MethodIncompatibility
        More or fewer than one kind of restriction, a matrix of the wrong
        shape, an unknown variable in ``sign``.
    IdentificationFailure
        More free parameters than the covariance has distinct elements, a
        likelihood that is flat in some direction, or sign restrictions no
        rotation satisfies.

    Notes
    -----
    Short- and long-run restrictions are estimated by maximum likelihood on
    ``-T/2 [K ln 2 pi - ln |M|^2 + tr(M' M Sigma)]``, ``M = B^{-1} A``
    (``M = C^{-1} (I - A_1 - ... - A_p)^{-1}`` in the long-run case), with
    standard errors from the expected information. An exactly identified
    model reproduces the reduced-form likelihood; with fewer free parameters
    the likelihood-ratio statistic tests the over-identifying restrictions.

    Sign restrictions do not select one model. Each accepted draw is a
    rotation of the Cholesky factor that is uniform on the orthogonal group
    and satisfies the signs; ``irf`` reports the pointwise median and
    quantiles over them. Two cautions follow. The bands are the spread of
    the identified set at the *estimated* reduced form and carry no
    sampling uncertainty. And the uniform distribution over rotations is
    itself an assumption that shapes where inside the identified set the
    median falls; only the set, not the median, is what the restrictions
    deliver.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> T = 400
    >>> u = rng.normal(size=(T, 2))                    # supply, demand
    >>> impact = np.array([[1.0, 0.5], [-0.6, 0.8]])   # output, prices
    >>> y = np.zeros((T, 2))
    >>> for t in range(1, T):
    ...     y[t] = 0.4 * y[t - 1] + impact @ u[t]
    >>> fit = sp.var(pd.DataFrame(y, columns=["output", "prices"]), lags=1)
    >>> chol = sp.svar(fit, B=[[np.nan, 0], [np.nan, np.nan]])
    >>> bool(chol.impact.iloc[0, 1] == 0)
    True
    >>> res = sp.svar(
    ...     fit,
    ...     sign={"supply": {"output": "+", "prices": "-"},
    ...           "demand": {"output": "+", "prices": "+"}},
    ...     n_draws=200, seed=0,
    ... )
    >>> res.n_accepted
    200
    >>> list(res.irf(2).columns)
    ['shock', 'response', 'period', 'irf', 'lower', 'upper']

    References
    ----------
    [@lutkepohl2005new],
    [@sims1980macroeconomics],
    [@blanchard1988dynamic],
    [@uhlig2005effects],
    [@rubioramirez2010structural]
    """
    if not isinstance(var_result, VARResult):
        raise MethodIncompatibility(
            "svar: the first argument is a VAR fitted by sp.var.",
            recovery_hint="fit = sp.var(df, variables=[...], lags=p); "
            "sp.svar(fit, ...)",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"svar: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="alpha is 1 minus the coverage of the bands.",
        )
    kinds = [A is not None or B is not None, long_run is not None, sign is not None]
    if sum(kinds) != 1:
        raise MethodIncompatibility(
            "svar: give exactly one kind of restriction: A= / B= (short-run), "
            "long_run= or sign=.",
            recovery_hint="A recursive identification is "
            "B=np.tril(np.full((K, K), np.nan)).",
        )
    names = list(var_result.var_names)
    k = len(names)
    T = int(var_result.n_obs)
    sigma = np.asarray(
        (
            var_result.sigma_u.to_numpy()
            if isinstance(var_result.sigma_u, pd.DataFrame)
            else var_result.sigma_u
        ),
        dtype=float,
    )
    lags = _lag_matrices(var_result)
    shocks = [f"shock{i + 1}" for i in range(k)]

    if sign is not None:
        if sign_horizon < 0 or n_draws < 1:
            raise MethodIncompatibility(
                "svar: sign_horizon must be >= 0 and n_draws >= 1."
            )
        spec = _sign_spec(sign, names)
        draws, tried = _sign_draws(
            sigma, lags, spec, int(sign_horizon), int(n_draws), int(max_tries), seed
        )
        if draws.shape[0] < n_draws:
            import warnings

            from ..exceptions import AssumptionWarning

            warnings.warn(
                f"svar: only {draws.shape[0]} of the {n_draws} requested "
                f"rotations satisfied the sign restrictions in {tried} tries; "
                "the quantiles rest on that many draws.",
                AssumptionWarning,
                stacklevel=2,
            )
        labels = [s for s, _ in spec]
        labels += [f"shock{i + 1}" for i in range(len(labels), k)]
        return SVARResult(
            identification="sign", var_names=names, shock_names=labels,
            impact=np.median(draws, axis=0), lag_matrices=lags, n_obs=T,
            draws=draws, n_tried=tried, alpha=float(alpha),
        )  # fmt: skip

    if T <= k:
        raise DataInsufficient(
            "svar: too few observations for the residual covariance.",
            recovery_hint="Use a longer sample or fewer variables.",
        )
    ll_reduced = _loglik(np.linalg.cholesky(np.linalg.inv(sigma)).T, sigma, T)

    def overid(fit: Dict[str, Any]) -> Optional[Dict[str, float]]:
        if fit["df_overid"] <= 0:
            return None
        stat = max(2.0 * (ll_reduced - fit["loglik"]), 0.0)
        return {
            "statistic": float(stat),
            "df": float(fit["df_overid"]),
            "pvalue": float(stats.chi2.sf(stat, fit["df_overid"])),
        }

    if long_run is not None:
        C0 = _pattern(long_run, k, "long_run")
        assert C0 is not None
        total = np.eye(k) - sum(lags)
        try:
            a_bar = np.linalg.inv(total)
        except np.linalg.LinAlgError as exc:
            raise IdentificationFailure(
                "svar: I - A_1 - ... - A_p is singular (a unit root in the "
                "VAR), so the long-run effects are not finite.",
                recovery_hint="Difference the integrated variables first.",
            ) from exc
        # the likelihood in C is the AB likelihood with "A" fixed at a_bar
        fit = _fit_ab(sigma, T, a_bar, C0, "long-run")
        C = fit["B"]
        return SVARResult(
            identification="long-run", var_names=names, shock_names=shocks,
            impact=total @ C, lag_matrices=lags, n_obs=T, C=C,
            table=_table(fit, ("A", "C")), log_likelihood=fit["loglik"],
            overid=overid(fit), alpha=float(alpha),
        )  # fmt: skip

    A0 = _pattern(A, k, "A")
    B0 = _pattern(B, k, "B")
    if A0 is None:
        A0 = np.eye(k)
    if B0 is None:
        B0 = np.where(np.eye(k, dtype=bool), np.nan, 0.0)
    fit = _fit_ab(sigma, T, A0, B0, "short-run")
    return SVARResult(
        identification="short-run", var_names=names, shock_names=shocks,
        impact=np.linalg.solve(fit["A"], fit["B"]), lag_matrices=lags, n_obs=T,
        A=fit["A"], B=fit["B"], table=_table(fit, ("A", "B")),
        log_likelihood=fit["loglik"], overid=overid(fit), alpha=float(alpha),
    )  # fmt: skip
