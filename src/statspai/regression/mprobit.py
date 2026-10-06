"""Multinomial probit for choice data in long form.

Each case chooses one of ``J`` alternatives; the utility of alternative
``j`` is ``x_ij'b + z_i'g_j + e_ij`` with jointly normal errors. Only
differences of utilities matter, so one alternative (``base``) fixes the
location and another (``scale``) fixes the scale: the model is written for
the ``J - 1`` errors differenced with the base, whose covariance has the
variance of the scale alternative set to 2. With independent errors of
variance one that covariance is ``I + 11'``, which is why 2 is the
normalisation (it is also Stata's).

A choice probability is a ``J - 1`` dimensional normal orthant
probability. It is written in the sequentially conditioned form of
Geweke, Hajivassiliou and Keane and then *integrated by Gauss-Legendre
quadrature* when there are at most four alternatives, so the likelihood is
a smooth deterministic function with no simulation noise and no seed. With
more alternatives the same recursion is averaged over Halton points.

References
----------
[@hansen2022econometrics]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, special

from ..core.results import EconometricResults
from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)

__all__ = ["mprobit"]

_TINY = 1e-300


def _as_list(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _halton(n: int, dim: int) -> np.ndarray:
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    out = np.empty((n, dim))
    for d in range(dim):
        base, seq = primes[d], np.zeros(n)
        index: np.ndarray = np.arange(11, n + 11, dtype=float)
        f = 1.0 / base
        while np.any(index > 0):
            seq += f * (index % base)
            index = np.floor(index / base)
            f /= base
        out[:, d] = seq
    return out


def _nodes(dim: int, n_points: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Points in the unit cube of ``dim`` dimensions and their weights."""
    if dim == 0:
        return np.zeros((1, 0)), np.ones(1)
    if dim <= 2:
        m = int(n_points) if n_points else (32 if dim == 1 else 20)
        x: np.ndarray
        w: np.ndarray
        x, w = np.polynomial.legendre.leggauss(m)
        x, w = (x + 1.0) / 2.0, w / 2.0
        # The integrand has unbounded derivatives at the ends of the unit
        # interval (it goes through the normal quantile function). The
        # substitution u = s^3 (10 - 15 s + 6 s^2) has two vanishing
        # derivatives at both ends, which restores the fast convergence of
        # the Gauss rule: 24 points per dimension give 1e-7 where the plain
        # rule gives 1e-4.
        w = w * 30.0 * x**2 * (1.0 - x) ** 2
        x = x**3 * (10.0 - 15.0 * x + 6.0 * x**2)
        if dim == 1:
            return x[:, None], w
        grid = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
        return grid, np.outer(w, w).ravel()
    m = int(n_points) if n_points else 500
    return _halton(m, dim), np.full(m, 1.0 / m)


def _orthant(
    limits: np.ndarray, chol: np.ndarray, points: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """``P(w < limits)`` for ``w ~ N(0, chol chol')``, one row per case."""
    n, d = limits.shape
    a = limits[:, [0]] / chol[0, 0]
    prob = special.ndtr(a) * np.ones((1, len(weights)))
    if d == 1:
        return np.asarray(prob[:, 0])
    z = np.empty((d - 1, n, len(weights)))
    upper = prob
    for k in range(1, d):
        u = np.clip(points[None, :, k - 1] * upper, 1e-300, 1.0 - 1e-16)
        z[k - 1] = special.ndtri(u)
        mean = np.tensordot(chol[k, :k], z[:k], axes=(0, 0))
        upper = special.ndtr((limits[:, [k]] - mean) / chol[k, k])
        prob = prob * upper
    return np.asarray(prob @ weights)


def mprobit(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str], None] = None,
    case_vars: Union[str, Sequence[str], None] = None,
    chid: Optional[str] = None,
    alt: Optional[str] = None,
    base: Optional[Any] = None,
    scale: Optional[Any] = None,
    correlation: str = "unstructured",
    stddev: str = "heteroskedastic",
    constants: bool = True,
    n_points: Optional[int] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Multinomial probit choice model, by maximum likelihood.

    Parameters
    ----------
    data : pandas.DataFrame
        Long form: one row per case and alternative.
    y : str
        1 for the chosen alternative, 0 otherwise.
    x : str or sequence of str, optional
        Regressors that vary over alternatives (one coefficient each).
    case_vars : str or sequence of str, optional
        Regressors that are constant within a case (one coefficient per
        alternative but the base).
    chid : str
        Case identifier.
    alt : str
        The variable naming the alternatives.
    base : optional
        Alternative that fixes the location. Default: the first in sorted
        order.
    scale : optional
        Alternative that fixes the scale. Default: the first one that is
        not the base.
    correlation : {'unstructured', 'independent'}, default 'unstructured'
    stddev : {'heteroskedastic', 'homoskedastic'}, default 'heteroskedastic'
        Structure of the error covariance. ``('independent',
        'homoskedastic')`` has no covariance parameter;
        ``('independent', 'heteroskedastic')`` estimates a standard
        deviation for each alternative but the base and the scale;
        ``('unstructured', 'heteroskedastic')`` estimates the whole
        covariance of the differenced errors.
    constants : bool, default True
        A constant for each alternative but the base.
    n_points : int, optional
        Quadrature points per dimension (default 32 with three
        alternatives, 20 with four), or the number of Halton points with
        five alternatives or more (default 500).
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params``: the coefficients of ``x``, then ``<alt>:<var>`` and
        ``<alt>:_cons`` for each alternative but the base, then the
        covariance parameters (``lnl2_2``, ``l2_1``, ... the Cholesky
        factor of the differenced covariance as Stata's ``cmmprobit``
        names it, or ``lnsigma:<alt>``). ``model_info`` has
        ``covariance`` and ``correlation`` of the errors differenced with
        the base, the log likelihood and ``probabilities`` (cases by
        alternatives).

    Notes
    -----
    Standard errors are from the observed information, computed by finite
    differences of the log likelihood.

    The covariance parameters of an unstructured model are weakly
    identified in most data: large changes in them move the likelihood
    little. Check ``model_info['correlation']`` for values at the boundary
    before reading the coefficients.

    Stata's ``cmmprobit`` simulates the same probabilities (Hammersley
    points by default), so its estimates carry simulation error that
    shrinks with ``intpoints()``; with a few thousand points the two agree
    to two or three digits.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> long = pd.DataFrame({
    ...     "case": np.repeat(np.arange(n), 3),
    ...     "mode": np.tile(["bus", "car", "train"], n),
    ...     "cost": rng.normal(size=3 * n),
    ... })
    >>> u = -1.0 * long.cost + rng.normal(size=3 * n)
    >>> long["choice"] = (u == u.groupby(long.case).transform("max")).astype(int)
    >>> fit = sp.mprobit(long, y="choice", x="cost", chid="case", alt="mode",
    ...                  correlation="independent", stddev="homoskedastic")
    >>> bool(abs(fit.params["cost"] + 1.0) < 0.3)
    True

    References
    ----------
    [@hansen2022econometrics]
    """
    xs, zs = _as_list(x), _as_list(case_vars)
    correlation, stddev = str(correlation).lower(), str(stddev).lower()
    structure = (correlation, stddev)
    allowed = (
        ("independent", "homoskedastic"),
        ("independent", "heteroskedastic"),
        ("unstructured", "heteroskedastic"),
    )
    if structure not in allowed:
        raise MethodIncompatibility(
            f"sp.mprobit: correlation={correlation!r} with stddev={stddev!r} "
            "is not available.",
            recovery_hint=f"Available combinations: {allowed}.",
        )
    if chid is None or alt is None:
        raise MethodIncompatibility(
            "sp.mprobit: chid= (case identifier) and alt= (alternatives "
            "variable) are required.",
            recovery_hint="The data are in long form, one row per case and "
            "alternative.",
        )
    cols = [y, chid, alt] + xs + zs
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"sp.mprobit: column(s) {missing} are not in the data.",
            recovery_hint="Check y=, x=, case_vars=, chid=, alt=.",
        )
    df = data[list(dict.fromkeys(cols))].dropna().sort_values([chid, alt])
    alts = sorted(pd.unique(df[alt]).tolist())
    J = len(alts)
    if J < 3:
        raise MethodIncompatibility(
            f"sp.mprobit: {J} alternatives; with two this is sp.probit.",
            recovery_hint="Use sp.probit for a binary choice.",
        )
    sizes = df.groupby(chid, sort=False)[alt].agg(["size", "nunique"])
    if not (sizes["size"].eq(J).all() and sizes["nunique"].eq(J).all()):
        raise MethodIncompatibility(
            "sp.mprobit: every case must list each of the "
            f"{J} alternatives exactly once.",
            recovery_hint="Drop the incomplete cases first.",
        )
    n = len(sizes)
    Y = df[y].to_numpy(dtype=float).reshape(n, J)
    if not (np.all(np.isin(Y, (0.0, 1.0))) and np.all(Y.sum(axis=1) == 1.0)):
        raise MethodIncompatibility(
            f"sp.mprobit: y={y!r} must mark exactly one chosen alternative "
            "per case.",
            recovery_hint="y is 1 on the chosen row and 0 on the others.",
        )
    if base is None:
        base = alts[0]
    if base not in alts:
        raise MethodIncompatibility(
            f"sp.mprobit: base={base!r} is not an alternative.",
            recovery_hint=f"Alternatives: {alts}.",
        )
    others = [a for a in alts if a != base]
    if scale is None:
        scale = others[0]
    if scale not in others:
        raise MethodIncompatibility(
            f"sp.mprobit: scale={scale!r} has to be an alternative other "
            "than the base.",
            recovery_hint=f"Choose one of {others}.",
        )
    # the differenced errors, the scale alternative first
    order = [scale] + [a for a in others if a != scale]
    d = J - 1
    col_of = {a: alts.index(a) for a in alts}
    b_col = col_of[base]

    X = (
        df[xs].to_numpy(dtype=float).reshape(n, J, len(xs))
        if xs
        else np.zeros((n, J, 0))
    )
    Zc = df[zs].to_numpy(dtype=float).reshape(n, J, len(zs))[:, 0, :]
    if zs and not np.allclose(
        df[zs].to_numpy(dtype=float).reshape(n, J, len(zs)), Zc[:, None, :]
    ):
        raise MethodIncompatibility(
            "sp.mprobit: a case_vars column varies within a case.",
            recovery_hint="Put regressors that vary over alternatives in x=.",
        )
    if constants:
        Zc = np.column_stack([Zc, np.ones(n)])
    kx, kz = len(xs), Zc.shape[1]
    k_beta = kx + kz * d
    if structure == ("independent", "homoskedastic"):
        cov_names: List[str] = []
    elif structure == ("independent", "heteroskedastic"):
        cov_names = [f"lnsigma:{a}" for a in order[1:]]
    else:
        cov_names = [f"lnl{i + 1}_{i + 1}" for i in range(1, d)] + [
            f"l{i + 1}_{j + 1}" for i in range(1, d) for j in range(i)
        ]
    k_cov = len(cov_names)
    k_total = k_beta + k_cov
    if n <= k_total:
        raise DataInsufficient(
            f"sp.mprobit: {n} cases for {k_total} parameters.",
            recovery_hint="Use fewer regressors or a simpler covariance.",
        )
    names = (
        xs
        + [f"{a}:{v}" for a in others for v in zs + (["_cons"] if constants else [])]
        + cov_names
    )
    chosen = np.argmax(Y, axis=1)
    fine = _nodes(d - 1, n_points)
    # a coarse rule carries the search; the fine one finishes it
    coarse = _nodes(d - 1, {0: None, 1: 16, 2: 12}.get(d - 1, 100))
    rule: Dict[str, Tuple[np.ndarray, np.ndarray]] = {"now": coarse}

    # d V / d beta, cases by alternatives by coefficients
    design = np.zeros((n, J, k_beta))
    design[:, :, :kx] = X
    for row, a in enumerate(others):
        design[:, col_of[a], kx + row * kz : kx + (row + 1) * kz] = Zc

    def covariance(tail: np.ndarray, kind: Tuple[str, str]) -> np.ndarray:
        """Covariance of the errors differenced with the base, in `order`."""
        if kind[0] == "independent":
            sd = np.ones(d)
            if tail.size:
                sd[1:] = np.exp(tail)
            return np.asarray(np.diag(sd**2) + 1.0)
        L = np.zeros((d, d))
        L[0, 0] = np.sqrt(2.0)
        L[np.arange(1, d), np.arange(1, d)] = np.exp(tail[: d - 1])
        position = d - 1
        for i in range(1, d):
            for j in range(i):
                L[i, j] = tail[position]
                position += 1
        return np.asarray(L @ L.T)

    # for each alternative j: the differences e_k - e_j, k != j, as a linear
    # map of the base-differenced errors (e_base = 0)
    maps: Dict[int, Tuple[np.ndarray, List[int]]] = {}
    for a in alts:
        j = col_of[a]
        rest = [col_of[k] for k in alts if k != a]
        A = np.zeros((d, d))
        for row, k_col in enumerate(rest):
            if k_col != b_col:
                A[row, order.index(alts[k_col])] += 1.0
            if j != b_col:
                A[row, order.index(a)] -= 1.0
        maps[j] = (A, rest)

    def probability(V: np.ndarray, sigma: np.ndarray, j: int, rows: Any) -> np.ndarray:
        A, rest = maps[j]
        chol = np.linalg.cholesky(A @ sigma @ A.T)
        limits = V[rows][:, [j]] - V[rows][:, rest]
        return _orthant(limits, chol, *rule["now"])

    groups = {j: np.flatnonzero(chosen == j) for j in maps}

    def logp(V: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Log probability of the chosen alternative, case by case."""
        out = np.zeros(n)
        try:
            for j, rows in groups.items():
                if rows.size:
                    out[rows] = probability(V, sigma, j, rows)
        except np.linalg.LinAlgError:
            return np.full(n, -np.inf)
        return np.asarray(np.log(np.maximum(out, _TINY)))

    h_v = 1e-5

    def d_utility(V: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """d log P / d V. A common shift of the utilities changes nothing,
        so the last column is minus the sum of the others."""
        G = np.zeros((n, J))
        for k in range(J - 1):
            up, down = V.copy(), V.copy()
            up[:, k] += h_v
            down[:, k] -= h_v
            G[:, k] = (logp(up, sigma) - logp(down, sigma)) / (2.0 * h_v)
        G[:, J - 1] = -G[:, : J - 1].sum(axis=1)
        return G

    def fit_structure(
        kind: Tuple[str, str], start: np.ndarray, kc: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Maximise for one covariance structure. Returns the estimate, the
        Hessian of minus the log likelihood, the gradient and the value."""

        def split(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return design @ theta[:k_beta], covariance(theta[k_beta:], kind)

        def value(theta: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                v = float(np.sum(logp(*split(theta))))
            return -v if np.isfinite(v) else 1e300

        def steps(theta: np.ndarray) -> np.ndarray:
            return np.asarray(1e-5 * np.maximum(1.0, np.abs(theta[k_beta:])))

        def grad(theta: np.ndarray) -> np.ndarray:
            V, sigma = split(theta)
            g = np.zeros(theta.size)
            g[:k_beta] = -np.einsum("ij,ijk->k", d_utility(V, sigma), design)
            for m, h in enumerate(steps(theta)):
                up, down = theta.copy(), theta.copy()
                up[k_beta + m] += h
                down[k_beta + m] -= h
                g[k_beta + m] = (value(up) - value(down)) / (2.0 * h)
            return g

        def hess(theta: np.ndarray) -> np.ndarray:
            V, sigma = split(theta)
            k = theta.size
            H = np.zeros((k, k))
            # second derivatives in the utilities, then the chain rule
            HV: np.ndarray = np.zeros((n, J, J))
            for a in range(J - 1):
                up, down = V.copy(), V.copy()
                up[:, a] += 1e-4
                down[:, a] -= 1e-4
                HV[:, a, :] = (d_utility(up, sigma) - d_utility(down, sigma)) / 2e-4
            HV[:, J - 1, :] = -HV[:, : J - 1, :].sum(axis=1)
            HV = (HV + HV.transpose(0, 2, 1)) / 2.0
            H[:k_beta, :k_beta] = -np.einsum("ija,ijl,ilb->ab", design, HV, design)
            hc = 1e-4 * np.maximum(1.0, np.abs(theta[k_beta:]))
            f0 = value(theta)
            single = np.zeros((kc, 2))
            for m in range(kc):
                tails = []
                for s_, sign in enumerate((1.0, -1.0)):
                    t = theta.copy()
                    t[k_beta + m] += sign * hc[m]
                    single[m, s_] = value(t)
                    Vt, st = split(t)
                    tails.append(d_utility(Vt, st))
                cross = -np.einsum(
                    "ij,ijk->k", (tails[0] - tails[1]) / (2.0 * hc[m]), design
                )
                H[:k_beta, k_beta + m] = H[k_beta + m, :k_beta] = cross
                H[k_beta + m, k_beta + m] = (
                    single[m, 0] - 2.0 * f0 + single[m, 1]
                ) / hc[m] ** 2
            for a in range(kc):
                for b in range(a):
                    t = theta.copy()
                    t[k_beta + a] += hc[a]
                    t[k_beta + b] += hc[b]
                    pp = value(t)
                    t = theta.copy()
                    t[k_beta + a] -= hc[a]
                    t[k_beta + b] -= hc[b]
                    mm = value(t)
                    H[k_beta + a, k_beta + b] = H[k_beta + b, k_beta + a] = (
                        pp - single[a, 0] - single[b, 0] + 2.0 * f0
                        - single[a, 1] - single[b, 1] + mm
                    ) / (2.0 * hc[a] * hc[b])  # fmt: skip
            return H

        rule["now"] = coarse
        found = optimize.minimize(
            value, start, jac=grad, method="BFGS",
            options={"gtol": 1e-5 * n, "maxiter": 1000},
        )  # fmt: skip
        theta = np.asarray(found.x, dtype=float)
        rule["now"] = fine
        H = hess(theta)
        g = grad(theta)
        for _ in range(6):
            try:
                trial = theta - np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            if not value(trial) < value(theta):
                break
            moved = float(
                np.max(np.abs(trial - theta) / np.maximum(1.0, np.abs(theta)))
            )
            theta = trial
            H, g = hess(theta), grad(theta)
            if moved < 1e-7:
                break
        return theta, H, g, -value(theta)

    def start_covariance() -> np.ndarray:
        if structure[0] == "independent":
            return np.zeros(k_cov)
        L = np.linalg.cholesky(np.eye(d) + 1.0)
        return np.concatenate(
            [
                np.log(np.diag(L)[1:]),
                [L[i, j] for i in range(1, d) for j in range(i)],
            ]
        )

    # the model with independent homoskedastic errors first: it is the
    # start of the richer ones, and its likelihood is their reference
    simple = ("independent", "homoskedastic")
    theta, H, grad_final, ll = fit_structure(simple, np.zeros(k_beta), 0)
    ll_independent = ll
    if k_cov:
        theta, H, grad_final, ll = fit_structure(
            structure, np.concatenate([theta, start_covariance()]), k_cov
        )
    grad_norm = float(np.max(np.abs(grad_final)))
    converged = bool(grad_norm < 1e-3 * max(1.0, np.sqrt(n)))
    if not converged:
        warnings.warn(
            f"sp.mprobit: the gradient is {grad_norm:.3g} at the reported "
            "estimate; the likelihood is nearly flat in some direction, "
            "usually a covariance parameter. Try a simpler covariance.",
            ConvergenceWarning,
            stacklevel=2,
        )
    eigen = np.linalg.eigvalsh((H + H.T) / 2.0)
    if eigen.min() <= 1e-10 * max(eigen.max(), 1.0):
        raise NumericalInstability(
            "sp.mprobit: the information matrix is singular at the estimate; "
            "a parameter is not identified.",
            recovery_hint="Use correlation='independent', or drop a regressor "
            "that does not vary enough across alternatives.",
        )
    cov = np.linalg.inv((H + H.T) / 2.0)
    se = np.sqrt(np.diag(cov))

    def probabilities(theta: np.ndarray, V: Optional[np.ndarray] = None) -> np.ndarray:
        if V is None:
            V = design @ theta[:k_beta]
        sigma_ = covariance(theta[k_beta:], structure)
        return np.column_stack(
            [probability(V, sigma_, j, np.arange(n)) for j in range(J)]
        )

    sigma = covariance(theta[k_beta:], structure)
    scale_sd = np.sqrt(np.diag(sigma))
    labels = [str(a) for a in order]
    P = probabilities(theta)
    info: Dict[str, Any] = {
        "alpha": alpha,
        "model_type": "Multinomial probit",
        "method": "Maximum likelihood ("
        + ("Gauss-Legendre quadrature" if d - 1 <= 2 else "Halton points")
        + ")",
        "correlation_structure": correlation,
        "stddev_structure": stddev,
        "alternatives": alts,
        "base": base,
        "scale": scale,
        "covariance": pd.DataFrame(sigma, index=labels, columns=labels),
        "correlation": pd.DataFrame(
            sigma / np.outer(scale_sd, scale_sd), index=labels, columns=labels
        ),
        "ll": ll,
        "log_likelihood": ll,
        "ll_independent": ll_independent,
        "aic": -2.0 * ll + 2.0 * k_total,
        "bic": -2.0 * ll + np.log(n) * k_total,
        "gradient_norm": grad_norm,
        "converged": converged,
        "n_points": int(len(fine[1])),
        "probabilities": pd.DataFrame(
            P, index=sizes.index, columns=[str(a) for a in alts]
        ),
        "robust": "nonrobust",
    }
    result = EconometricResults(
        params=pd.Series(theta, index=names),
        std_errors=pd.Series(se, index=names),
        model_info=info,
        data_info={
            "nobs": int(n * J),
            "n_cases": int(n),
            "df_model": k_total,
            "df_resid": int(n - k_total),
            "dependent_var": y,
            "var_names": names,
            "var_cov": cov,
        },
        diagnostics={
            "Log likelihood": ll,
            "AIC": info["aic"],
            "BIC": info["bic"],
        },
    )

    def effect(variable: str, outcome: Any, alternative: Any) -> Tuple[float, float]:
        """Average effect on ``P(outcome)`` of a unit change in ``variable``
        for ``alternative``, with a delta-method standard error."""
        if variable not in xs:
            raise MethodIncompatibility(
                f"sp.mprobit: {variable!r} is not an alternative-specific "
                "regressor.",
                recovery_hint=f"Choose one of {xs}.",
            )
        position = xs.index(variable)
        out_col, at_col = col_of[outcome], col_of[alternative]
        step = 1e-4 * max(1.0, float(np.std(X[:, :, position])))

        def average(t: np.ndarray) -> float:
            V = design @ t[:k_beta]
            sigma_ = covariance(t[k_beta:], structure)
            up, down = V.copy(), V.copy()
            up[:, at_col] += t[position] * step
            down[:, at_col] -= t[position] * step
            everyone = np.arange(n)
            hi = probability(up, sigma_, out_col, everyone)
            lo = probability(down, sigma_, out_col, everyone)
            return float(np.mean((hi - lo) / (2.0 * step)))

        g = np.zeros(theta.size)
        for m in range(theta.size):
            h = 1e-5 * max(1.0, abs(theta[m]))
            hi_t, lo_t = theta.copy(), theta.copy()
            hi_t[m] += h
            lo_t[m] -= h
            g[m] = (average(hi_t) - average(lo_t)) / (2.0 * h)
        return average(theta), float(np.sqrt(g @ cov @ g))

    result._choice_effect = effect  # type: ignore[attr-defined]
    return result
