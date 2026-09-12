"""
Olley-Pakes / Levinsohn-Petrin / Ackerberg-Caves-Frazer production function
estimators.

OP and LP are the two-step estimators computed by Stata's and R's
``prodest``:

1. ``y = free'beta_f + poly_d(state, proxy) + eta`` by OLS. This gives the
   free-input coefficients and ``phi = fitted - free'beta_f``.
2. The state coefficients ``theta`` minimise the sum of squared
   ``xi = (y - free'beta_f) - state'theta - g(omega_lag)`` with
   ``omega = phi - state'theta``, ``omega_lag = phi_{t-1} - state_{t-1}'theta``
   and ``g`` a polynomial whose coefficients are concentrated out by OLS of
   ``omega`` on powers of ``omega_lag``.

OP and LP differ only in the proxy: investment (OP) or an intermediate input
(LP).

ACF puts every input into stage 1 (``y = poly_d(free, state, proxy) + eta``,
``phi = fitted``) and identifies all coefficients in stage 2 from
``E[xi * (free_{t-1}, state)] = 0``, with ``omega = phi - inputs'beta`` and
``xi = omega - g(omega_lag)``. Its criterion is ``m' (Z'Z)^{-1} m / n`` with
``m = Z'xi``. The system is just identified, so the estimate is a root of
``m``. A cubic ``g`` can give ``m`` more than one root; every root found from
a grid of starts is reported in ``diagnostics["acf_roots"]``.

Lags are calendar lags: the value at ``time - 1`` of the same firm, ``NaN``
when that year is missing.

Cobb-Douglas is supported by all three; translog by ACF only (OP and LP read
the free-input coefficients off the linear stage-1 regression).

References
----------
Olley & Pakes (1996, Econometrica) [@olley1996dynamics]
Levinsohn & Petrin (2003, Rev. Econ. Stud.) [@levinsohn2003estimating]
Ackerberg, Caves & Frazer (2015, Econometrica) [@ackerberg2015identification]
"""

from __future__ import annotations

import warnings
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from ...exceptions import ConvergenceFailure
from ._core import elasticities_at, expand_inputs, panel_lag, polynomial_basis
from ._result import ProductionResult

_PID = "__panel_id__"
_TIME = "__time__"

# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def _normalise_form(functional_form: str) -> str:
    fform = functional_form.lower().replace("_", "-")
    if fform in ("cobb-douglas", "cd"):
        return "cobb-douglas"
    if fform == "translog":
        return "translog"
    raise ValueError(
        f"Unknown functional_form {functional_form!r}; "
        "choose 'cobb-douglas' or 'translog'."
    )


def _prepare_panel(
    data: pd.DataFrame,
    output: str,
    free: Sequence[str],
    state: Sequence[str],
    proxy: str,
    panel_id: str,
    time: str,
) -> pd.DataFrame:
    """Select, drop missing, sort by (id, time) and attach internal id/time.

    ``diagnostics["n_calendar_gaps"]`` later reports how many rows follow a
    skipped period; their lags are ``NaN`` (see :func:`_core.panel_lag`).
    """
    cols = list(dict.fromkeys([output, *free, *state, proxy, panel_id, time]))
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    df = data[cols].dropna().sort_values([panel_id, time]).reset_index(drop=True)
    if not pd.api.types.is_numeric_dtype(df[time]):
        raise TypeError(
            f"time column {time!r} must be numeric (e.g. an integer year); "
            f"got dtype {df[time].dtype}."
        )
    if df.duplicated([panel_id, time]).any():
        raise ValueError(
            f"({panel_id!r}, {time!r}) must uniquely identify rows; "
            "found duplicate firm-period pairs."
        )
    df[_PID] = df[panel_id].to_numpy()
    df[_TIME] = df[time].to_numpy()
    return df


def _lags(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return np.column_stack(
        [panel_lag(df, c, _PID, _TIME).to_numpy(dtype=float) for c in columns]
    )


def _n_gaps(df: pd.DataFrame) -> int:
    step = df.groupby(_PID, sort=False)[_TIME].diff()
    return int((step > 1).sum())


def _markov_residual(
    omega: np.ndarray,
    omega_lag: np.ndarray,
    d_omega: np.ndarray,
    d_omega_lag: np.ndarray,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``r = (I - H) omega``, the Markov coefficients, and ``dr/dtheta``.

    ``H`` projects on ``G = (1, u, ..., u**degree)`` with ``u = omega_lag``;
    ``d_omega`` and ``d_omega_lag`` (n x p) are the derivatives of ``omega``
    and ``u`` with respect to the p parameters. With ``gamma = (G'G)^{-1}
    G'omega`` and ``dG_j`` the derivative of ``G``,

        dr_j = (I - H)(d_omega_j - dG_j gamma) - G (G'G)^{-1} dG_j' r.
    """
    G, _ = polynomial_basis(omega_lag.reshape(-1, 1), degree=degree)
    Q, R = np.linalg.qr(G)
    gamma = np.linalg.solve(R, Q.T @ omega)
    r = omega - Q @ (Q.T @ omega)
    dG_du = np.column_stack(
        [np.zeros_like(omega_lag)]
        + [q * omega_lag ** (q - 1) for q in range(1, degree + 1)]
    )
    J = np.empty_like(d_omega)
    for j in range(d_omega.shape[1]):
        dG = dG_du * d_omega_lag[:, [j]]
        a = d_omega[:, j] - dG @ gamma
        J[:, j] = a - Q @ (Q.T @ a) - Q @ np.linalg.solve(R.T, dG.T @ r)
    return r, gamma, J


def _min_obs(n: int) -> None:
    if n < 10:
        raise ValueError(
            f"Only {n} valid observations for stage 2; need at least 10. "
            "Check panel structure (each firm needs consecutive periods)."
        )


def _firm_bootstrap(
    df: pd.DataFrame,
    fit: Callable[[pd.DataFrame], np.ndarray],
    reps: int,
    seed: Optional[int],
    label: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
    """Resample whole firms with replacement, refit, return (se, cov, n_ok).

    Each draw of a firm becomes its own panel, so a firm drawn twice does not
    collide with itself in the lag operator.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby(_PID, sort=False).indices
    firms = list(groups)
    draws: List[np.ndarray] = []
    failures: Dict[str, int] = {}
    for _ in range(int(reps)):
        pick = rng.integers(0, len(firms), size=len(firms))
        rows = [groups[firms[j]] for j in pick]
        boot = df.iloc[np.concatenate(rows)].reset_index(drop=True)
        boot[_PID] = np.repeat(np.arange(len(rows)), [len(r) for r in rows])
        try:
            draws.append(np.asarray(fit(boot), dtype=float))
        except (ValueError, np.linalg.LinAlgError, ConvergenceFailure) as exc:
            name = type(exc).__name__
            failures[name] = failures.get(name, 0) + 1
    n_ok = len(draws)
    if failures:
        warnings.warn(
            f"{label} bootstrap: {sum(failures.values())}/{int(reps)} "
            f"replications failed ({failures}); SEs use the {n_ok} that "
            "succeeded.",
            RuntimeWarning,
            stacklevel=3,
        )
    if n_ok < 2:
        warnings.warn(
            f"{label} bootstrap: only {n_ok} replication(s) succeeded (need "
            "at least 2); standard errors are NaN.",
            RuntimeWarning,
            stacklevel=3,
        )
        return np.full(0, np.nan), None, n_ok
    B = np.vstack(draws)
    return B.std(axis=0, ddof=1), np.atleast_2d(np.cov(B.T, ddof=1)), n_ok


def _pack(
    *,
    method: str,
    names: List[str],
    beta: np.ndarray,
    se: np.ndarray,
    cov: Optional[np.ndarray],
    omega: np.ndarray,
    eta: np.ndarray,
    gamma: np.ndarray,
    xi: np.ndarray,
    sample: pd.DataFrame,
    raw_input_names: List[str],
    free: List[str],
    state: List[str],
    proxy: str,
    functional_form: str,
    diagnostics: Dict[str, Any],
) -> ProductionResult:
    if se.size != beta.size:
        se = np.full(beta.size, np.nan)
    coef = {name: float(beta[i]) for i, name in enumerate(names)}
    rho = float(gamma[1]) if gamma.size > 1 else float("nan")
    sigma_xi = float(np.std(xi, ddof=1))
    sample = sample.copy()
    sample["omega"] = omega
    sample["eta"] = eta
    elasticities = elasticities_at(
        sample[raw_input_names].to_numpy(dtype=float),
        raw_input_names,
        coef,
        functional_form=functional_form,
    )
    diagnostics = {
        **diagnostics,
        "ar_rho": rho,
        "ar_sigma_xi": sigma_xi,
        "markov_coef": [float(g) for g in gamma],
    }
    return ProductionResult(
        method=method,
        params=pd.Series(beta, index=names, name="elasticity"),
        std_errors=pd.Series(se, index=names, name="std_error"),
        coef=coef,
        tfp=omega,
        residuals=eta,
        productivity_process={"rho": rho, "sigma": sigma_xi},
        sample=sample,
        diagnostics=diagnostics,
        model_info={
            "free_inputs": free,
            "state_inputs": state,
            "raw_input_names": raw_input_names,
            "proxy": proxy,
            "functional_form": functional_form,
            "elasticities": elasticities,
        },
        cov=cov,
    )


# ---------------------------------------------------------------------------
# Olley-Pakes / Levinsohn-Petrin
# ---------------------------------------------------------------------------


class _OPLPStage2:
    """Stage-2 objects for OP/LP on one (possibly resampled) panel."""

    def __init__(
        self,
        df: pd.DataFrame,
        output: str,
        free: List[str],
        state: List[str],
        proxy: str,
        polynomial_degree: int,
        productivity_degree: int,
    ) -> None:
        y = df[output].to_numpy(dtype=float)
        F = df[free].to_numpy(dtype=float)
        S = df[state].to_numpy(dtype=float)
        P, _ = polynomial_basis(
            df[[*state, proxy]].to_numpy(dtype=float),
            degree=polynomial_degree,
            include_intercept=False,
        )
        X1 = np.column_stack([np.ones(len(y)), F, P])
        b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        nf = len(free)
        self.fitted = X1 @ b1
        self.beta_free = b1[1 : 1 + nf]
        # Linear state terms lead the degree-1 block of the polynomial.
        self.theta0 = b1[1 + nf : 1 + nf + len(state)].copy()
        self.stage1_r2 = float(1.0 - np.var(y - self.fitted) / np.var(y))
        phi = self.fitted - F @ self.beta_free
        work = df[[_PID, _TIME]].copy()
        work["__phi__"] = phi
        lag_phi = panel_lag(work, "__phi__", _PID, _TIME).to_numpy(dtype=float)
        lag_S = _lags(df, state)
        self.valid = np.isfinite(lag_phi) & np.isfinite(lag_S).all(axis=1)
        _min_obs(int(self.valid.sum()))
        v = self.valid
        self.y = y[v]
        self.phi = phi[v]
        self.lag_phi = lag_phi[v]
        self.S = S[v]
        self.lag_S = lag_S[v]
        self.res = (y - F @ self.beta_free)[v]
        self.degree = productivity_degree

    def _residual(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        omega = self.phi - self.S @ theta
        r, gamma, J = _markov_residual(
            omega,
            self.lag_phi - self.lag_S @ theta,
            -self.S,
            -self.lag_S,
            self.degree,
        )
        # xi = (y - free'b) - state'theta - g = (y - fitted) + (I - H) omega
        return (self.res - self.phi) + r, omega, gamma, J

    def xi(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xi, omega, gamma, _ = self._residual(theta)
        return xi, omega, gamma

    def solve(self) -> optimize.OptimizeResult:
        fit = optimize.least_squares(
            lambda th: self._residual(th)[0],
            self.theta0,
            jac=lambda th: self._residual(th)[3],
            method="lm",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            max_nfev=20000,
        )
        if not np.all(np.isfinite(fit.x)):
            raise ConvergenceFailure("OP/LP stage 2 returned non-finite estimates.")
        # LM stops once the relative fall in the sum of squares is below
        # ftol, which on this flat criterion leaves theta ~1e-7 short.
        # Finish on the first-order condition J'xi = 0 (analytic J).

        def gradient(th: np.ndarray) -> np.ndarray:
            xi, _, _, J = self._residual(th)
            return J.T @ xi

        polish = optimize.root(gradient, fit.x, method="hybr", options={"xtol": 1e-15})
        sse = lambda th: float(np.sum(self._residual(th)[0] ** 2))  # noqa: E731
        if np.all(np.isfinite(polish.x)) and sse(polish.x) <= sse(fit.x):
            fit.x = polish.x
        # Converged when xi is orthogonal to every Jacobian column (cosine).
        xi, _, _, J = self._residual(fit.x)
        denom = np.linalg.norm(J, axis=0) * np.linalg.norm(xi)
        cosine = np.abs(J.T @ xi) / np.where(denom > 0, denom, 1.0)
        fit.status = int(np.all(cosine <= 1e-8))
        return fit


def _estimate_oplp(
    df: pd.DataFrame,
    *,
    method: str,
    output: str,
    free: List[str],
    state: List[str],
    proxy: str,
    polynomial_degree: int,
    productivity_degree: int,
    functional_form: str,
    boot_reps: int,
    seed: Optional[int],
) -> ProductionResult:
    if _normalise_form(functional_form) != "cobb-douglas":
        raise NotImplementedError(
            f"method={method!r} supports functional_form='cobb-douglas' only: "
            "OP and LP read the free-input coefficients off the linear "
            "stage-1 regression. Use sp.ackerberg_caves_frazer for translog."
        )
    clash = sorted(set(free) & ({proxy} | set(state)))
    if clash:
        raise ValueError(
            f"{clash} appear among the free inputs and the state/proxy "
            "variables; OP and LP need them to be distinct."
        )

    def build(frame: pd.DataFrame) -> _OPLPStage2:
        return _OPLPStage2(
            frame, output, free, state, proxy, polynomial_degree, productivity_degree
        )

    st = build(df)
    fit = st.solve()
    theta = fit.x
    xi, omega, gamma = st.xi(theta)
    beta = np.concatenate([st.beta_free, theta])

    se = np.full(0, np.nan)
    cov = None
    n_boot = 0
    if boot_reps and boot_reps > 0:

        def refit(frame: pd.DataFrame) -> np.ndarray:
            b = build(frame)
            return np.concatenate([b.beta_free, b.solve().x])

        se, cov, n_boot = _firm_bootstrap(df, refit, boot_reps, seed, method.upper())

    sample = df.loc[st.valid].reset_index(drop=True)
    y_all = df[output].to_numpy(dtype=float)
    eta = (y_all - st.fitted)[st.valid]
    return _pack(
        method=method,
        names=[*free, *state],
        beta=beta,
        se=se,
        cov=cov,
        omega=omega,
        eta=eta,
        gamma=gamma,
        xi=xi,
        sample=sample,
        raw_input_names=[*free, *state],
        free=free,
        state=state,
        proxy=proxy,
        functional_form="cobb-douglas",
        diagnostics={
            "stage1_r2": st.stage1_r2,
            "stage2_objective": float(xi @ xi),
            "stage2_converged": bool(fit.status > 0),
            "boot_reps_effective": n_boot,
            "polynomial_degree": int(polynomial_degree),
            "productivity_degree": int(productivity_degree),
            "n_calendar_gaps": _n_gaps(df),
        },
    )


# ---------------------------------------------------------------------------
# Ackerberg-Caves-Frazer
# ---------------------------------------------------------------------------


class _ACFStage2:
    """Stage-2 moment system for ACF on one (possibly resampled) panel."""

    def __init__(
        self,
        df: pd.DataFrame,
        output: str,
        free: List[str],
        state: List[str],
        proxy: str,
        polynomial_degree: int,
        productivity_degree: int,
        functional_form: str,
    ) -> None:
        y = df[output].to_numpy(dtype=float)
        poly_vars = list(dict.fromkeys([*free, *state, proxy]))
        P, _ = polynomial_basis(
            df[poly_vars].to_numpy(dtype=float),
            degree=polynomial_degree,
            include_intercept=False,
        )
        X1 = np.column_stack([np.ones(len(y)), P])
        b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        self.fitted = X1 @ b1
        self.stage1_r2 = float(1.0 - np.var(y - self.fitted) / np.var(y))
        inputs = [*free, *state]
        X, names = expand_inputs(
            df[inputs].to_numpy(dtype=float), inputs, functional_form
        )
        LX, _ = expand_inputs(_lags(df, inputs), inputs, functional_form)
        Zraw = np.column_stack([_lags(df, free), df[state].to_numpy(dtype=float)])
        Z, _ = expand_inputs(Zraw, inputs, functional_form)
        work = df[[_PID, _TIME]].copy()
        work["__phi__"] = self.fitted
        lag_phi = panel_lag(work, "__phi__", _PID, _TIME).to_numpy(dtype=float)
        self.valid = (
            np.isfinite(lag_phi)
            & np.isfinite(LX).all(axis=1)
            & np.isfinite(Z).all(axis=1)
        )
        _min_obs(int(self.valid.sum()))
        v = self.valid
        self.names = names
        self.phi, self.lag_phi = self.fitted[v], lag_phi[v]
        self.X, self.LX, self.Z = X[v], LX[v], Z[v]
        ZZ = self.Z.T @ self.Z
        # W = (Z'Z)^{-1} / n, factored so that |L'm|^2 = m'Wm.
        self.L = np.linalg.cholesky(np.linalg.inv(ZZ) / len(self.Z))
        self.degree = productivity_degree
        # Start at the stage-1 linear coefficients of the inputs (zero on
        # translog terms), as prodest does, without its random perturbation.
        start = np.zeros(len(names))
        for j, nm in enumerate(inputs):
            start[j] = b1[1 + poly_vars.index(nm)]
        self.theta0 = start
        self.z_norm = float(np.linalg.norm(self.Z))

    def _residual(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        omega = self.phi - self.X @ theta
        xi, gamma, J = _markov_residual(
            omega, self.lag_phi - self.LX @ theta, -self.X, -self.LX, self.degree
        )
        return xi, omega, gamma, J

    def parts(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xi, omega, gamma, _ = self._residual(theta)
        return xi, omega, gamma

    def scaled_moments(self, theta: np.ndarray) -> np.ndarray:
        return self.L.T @ (self.Z.T @ self._residual(theta)[0])

    def scaled_jacobian(self, theta: np.ndarray) -> np.ndarray:
        return self.L.T @ (self.Z.T @ self._residual(theta)[3])

    def crit(self, theta: np.ndarray) -> float:
        r = self.scaled_moments(theta)
        return float(r @ r)

    def is_root(self, theta: np.ndarray) -> bool:
        """Moments zero relative to their scale: |Z'xi| <= 1e-12 |Z| |xi|.

        Scale-free on purpose. Far out along a nearly deterministic input
        (capital that barely moves) the criterion can fall below any absolute
        cutoff while the normalised moments stay near 1e-9; exact roots sit
        at 1e-16.
        """
        xi = self._residual(theta)[0]
        denom = self.z_norm * float(np.linalg.norm(xi))
        if not np.isfinite(denom) or denom == 0.0:
            return False
        return float(np.linalg.norm(self.Z.T @ xi)) <= 1e-12 * denom

    def _solve_from(self, start: np.ndarray) -> Optional[np.ndarray]:
        try:
            fit = optimize.least_squares(
                self.scaled_moments,
                start,
                jac=self.scaled_jacobian,
                method="lm",
                xtol=1e-15,
                ftol=1e-15,
                gtol=1e-15,
                max_nfev=5000,
            )
        except (ValueError, np.linalg.LinAlgError):
            return None
        return fit.x if np.all(np.isfinite(fit.x)) else None

    def solve(self, search: bool = True) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
        """Return (estimate, is_root, distinct roots found)."""
        primary = self._solve_from(self.theta0)
        candidates: List[np.ndarray] = [] if primary is None else [primary]
        if search:
            n_lin = min(len(self.theta0), 3)
            grid = (0.1, 0.4, 0.7, 1.0)
            for combo in product(grid, repeat=n_lin):
                start = np.zeros(len(self.theta0))
                start[:n_lin] = combo
                if len(self.theta0) > n_lin:
                    start[n_lin : len(self.names)] = 0.0
                sol = self._solve_from(start)
                if sol is not None:
                    candidates.append(sol)
        roots: List[np.ndarray] = []
        for cand in candidates:
            if not self.is_root(cand):
                continue
            if not any(np.allclose(cand, r, rtol=1e-6, atol=1e-6) for r in roots):
                roots.append(cand)
        if roots:
            # The root nearest the stage-1 coefficients, whichever start or
            # solver path reached it.
            dist = [np.linalg.norm(r - self.theta0) for r in roots]
            return roots[int(np.argmin(dist))], True, roots
        if not candidates:
            raise ConvergenceFailure("ACF stage 2 failed from every start.")
        best = min(candidates, key=self.crit)
        return best, False, roots


def _estimate_acf(
    df: pd.DataFrame,
    *,
    output: str,
    free: List[str],
    state: List[str],
    proxy: str,
    polynomial_degree: int,
    productivity_degree: int,
    functional_form: str,
    boot_reps: int,
    seed: Optional[int],
) -> ProductionResult:
    fform = _normalise_form(functional_form)

    def build(frame: pd.DataFrame) -> _ACFStage2:
        return _ACFStage2(
            frame,
            output,
            free,
            state,
            proxy,
            polynomial_degree,
            productivity_degree,
            fform,
        )

    st = build(df)
    beta, rooted, roots = st.solve(search=True)
    if not rooted:
        warnings.warn(
            "ACF: no exact root of the moment conditions was found; the "
            f"estimate minimises the GMM criterion at {st.crit(beta):.3g}. "
            "Try a different polynomial_degree / productivity_degree.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif len(roots) > 1:
        warnings.warn(
            f"ACF: the moment conditions have {len(roots)} distinct roots "
            f"({', '.join(np.array2string(r, precision=4) for r in roots)}); "
            "the estimate is the root nearest the stage-1 coefficients. "
            "All roots are in diagnostics['acf_roots'].",
            RuntimeWarning,
            stacklevel=3,
        )
    xi, omega, gamma = st.parts(beta)

    se = np.full(0, np.nan)
    cov = None
    n_boot = 0
    if boot_reps and boot_reps > 0:

        def refit(frame: pd.DataFrame) -> np.ndarray:
            b = build(frame)
            b.theta0 = beta.copy()
            est = b._solve_from(beta)
            if est is None or not b.is_root(est):
                raise ConvergenceFailure("no ACF root near the full-sample estimate")
            return est

        se, cov, n_boot = _firm_bootstrap(df, refit, boot_reps, seed, "ACF")

    inputs = [*free, *state]
    sample = df.loc[st.valid].reset_index(drop=True)
    y_all = df[output].to_numpy(dtype=float)
    eta = (y_all - st.fitted)[st.valid]
    return _pack(
        method="acf",
        names=st.names,
        beta=beta,
        se=se,
        cov=cov,
        omega=omega,
        eta=eta,
        gamma=gamma,
        xi=xi,
        sample=sample,
        raw_input_names=inputs,
        free=free,
        state=state,
        proxy=proxy,
        functional_form=fform,
        diagnostics={
            "stage1_r2": st.stage1_r2,
            "stage2_objective": st.crit(beta),
            "stage2_converged": bool(rooted),
            "acf_roots": [[float(v) for v in r] for r in roots],
            "boot_reps_effective": n_boot,
            "polynomial_degree": int(polynomial_degree),
            "productivity_degree": int(productivity_degree),
            "n_calendar_gaps": _n_gaps(df),
        },
    )


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------


def _resolve_inputs(
    free: Optional[Sequence[str] | str],
    state: Optional[Sequence[str] | str],
    free_default: Sequence[str],
    state_default: Sequence[str],
) -> Tuple[List[str], List[str]]:
    def _to_list(
        x: Optional[Sequence[str] | str],
        default: Sequence[str],
    ) -> List[str]:
        if x is None:
            return list(default)
        if isinstance(x, str):
            return [x]
        return list(x)

    return _to_list(free, free_default), _to_list(state, state_default)


def olley_pakes(
    data: pd.DataFrame,
    output: str = "y",
    free: Sequence[str] | str | None = None,
    state: Sequence[str] | str | None = None,
    proxy: str = "i",
    panel_id: str = "id",
    time: str = "year",
    polynomial_degree: int = 3,
    productivity_degree: int = 3,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
    drop_zero_proxy: bool = True,
) -> ProductionResult:
    """Olley-Pakes (1996) production function estimator.

    Uses **investment** as the proxy for unobserved productivity. The
    free-input coefficients come from the stage-1 regression of output on
    the free inputs and a polynomial in (state, investment); the state
    coefficients minimise the sum of squared productivity innovations in
    stage 2. This is the estimator of Stata's and R's ``prodest``
    (``method(op)`` / ``prodestOP``).

    Parameters
    ----------
    data : DataFrame
        Long-form panel: one row per (firm, year).
    output : str, default ``"y"``
        Log output column.
    free : str or list, default ``["l"]``
        Freely chosen inputs (e.g. labor). Multiple are allowed.
    state : str or list, default ``["k"]``
        State inputs (capital, predetermined).
    proxy : str, default ``"i"``
        Investment column.
    panel_id, time : str
        Firm and year identifiers. ``time`` must be numeric; the lag of a
        year is the firm's value one year earlier (``NaN`` across a gap).
    polynomial_degree : int, default 3
        Degree of the stage-1 polynomial in (state, proxy); ``prodest``'s
        Stata default is 3, R ``prodest`` uses 2.
    productivity_degree : int, default 3
        Degree of the polynomial ``g`` in the productivity process (cubic in
        both ``prodest`` implementations).
    functional_form : {'cobb-douglas'}, default 'cobb-douglas'
        Translog raises ``NotImplementedError``; use
        :func:`ackerberg_caves_frazer`.
    boot_reps : int, default 0
        Firm-cluster bootstrap replications. ``0`` ⇒ NaN standard errors.
    seed : int, optional
        Bootstrap RNG seed.
    drop_zero_proxy : bool, default True
        Drop rows with non-positive proxy (the OP inversion needs strictly
        positive investment in levels). Pass ``False`` when the proxy is in
        logs. Dropping period ``t`` for a firm also removes period ``t+1``
        from stage 2, whose lag is then missing.

    Returns
    -------
    ProductionResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for fid in range(30):
    ...     k = rng.normal(2.0, 0.5)
    ...     omega = rng.normal(0.0, 0.3)
    ...     for yr in range(8):
    ...         omega = 0.7 * omega + rng.normal(0.0, 0.2)
    ...         k = 0.9 * k + 0.3 * rng.normal(1.0, 0.3)
    ...         l = 0.6 * k + omega + rng.normal(1.0, 0.2)
    ...         inv = 0.4 * k + omega + rng.normal(1.0, 0.2)  # investment > 0
    ...         y = 0.6 * l + 0.3 * k + omega + rng.normal(0.0, 0.1)
    ...         rows.append(dict(id=fid, year=yr, y=y, l=l, k=k, i=inv))
    >>> df = pd.DataFrame(rows)
    >>> res = sp.olley_pakes(df, output="y", free="l", state="k", proxy="i")
    >>> sorted(res.coef.keys())
    ['k', 'l']
    >>> res.method
    'op'

    References
    ----------
    Olley, G.S. & Pakes, A. (1996). The dynamics of productivity in the
    telecommunications equipment industry. Econometrica, 64(6), 1263-1297.
    """
    free, state = _resolve_inputs(free, state, ["l"], ["k"])
    cols = [output, *free, *state, proxy, panel_id, time]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")
    frame = data.loc[data[proxy] > 0] if drop_zero_proxy else data
    df = _prepare_panel(frame, output, free, state, proxy, panel_id, time)
    return _estimate_oplp(
        df,
        method="op",
        output=output,
        free=free,
        state=state,
        proxy=proxy,
        polynomial_degree=polynomial_degree,
        productivity_degree=productivity_degree,
        functional_form=functional_form,
        boot_reps=boot_reps,
        seed=seed,
    )


def levinsohn_petrin(
    data: pd.DataFrame,
    output: str = "y",
    free: Sequence[str] | str | None = None,
    state: Sequence[str] | str | None = None,
    proxy: str = "m",
    panel_id: str = "id",
    time: str = "year",
    polynomial_degree: int = 3,
    productivity_degree: int = 3,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
) -> ProductionResult:
    """Levinsohn-Petrin (2003) production function estimator.

    Uses an **intermediate input** (materials / energy) as the productivity
    proxy, which avoids the OP zero-investment selection problem. Same two
    steps as :func:`olley_pakes` (``prodest``'s ``method(lp)`` /
    ``prodestLP``).

    Parameters
    ----------
    data : DataFrame
        Long-form panel.
    output : str
        Log output.
    free : str or list, default ``["l"]``
        Free inputs.
    state : str or list, default ``["k"]``
        State inputs.
    proxy : str, default ``"m"``
        Intermediate input (materials).
    panel_id, time : str
        Firm and (numeric) year identifiers.
    polynomial_degree : int, default 3
        Degree of the stage-1 polynomial in (state, proxy).
    productivity_degree : int, default 3
        Degree of the productivity polynomial ``g``.
    functional_form : {'cobb-douglas'}, default 'cobb-douglas'
        Translog raises ``NotImplementedError``.
    boot_reps : int
        Firm-cluster bootstrap replications.
    seed : int
        Bootstrap RNG seed.

    Returns
    -------
    ProductionResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for fid in range(30):
    ...     k = rng.normal(2.0, 0.5)
    ...     omega = rng.normal(0.0, 0.3)
    ...     for yr in range(8):
    ...         omega = 0.7 * omega + rng.normal(0.0, 0.2)
    ...         k = 0.9 * k + 0.3 * rng.normal(1.0, 0.3)
    ...         l = 0.6 * k + omega + rng.normal(1.0, 0.2)
    ...         m = 0.5 * k + 0.5 * l + omega + rng.normal(0.5, 0.2)
    ...         y = 0.6 * l + 0.3 * k + omega + rng.normal(0.0, 0.1)
    ...         rows.append(dict(id=fid, year=yr, y=y, l=l, k=k, m=m))
    >>> df = pd.DataFrame(rows)
    >>> res = sp.levinsohn_petrin(df, output="y", free="l", state="k", proxy="m")
    >>> sorted(res.coef.keys())
    ['k', 'l']
    >>> res.method
    'lp'

    References
    ----------
    Levinsohn, J. & Petrin, A. (2003). Estimating production functions
    using inputs to control for unobservables. Review of Economic
    Studies, 70(2), 317-341.
    """
    free, state = _resolve_inputs(free, state, ["l"], ["k"])
    df = _prepare_panel(data, output, free, state, proxy, panel_id, time)
    return _estimate_oplp(
        df,
        method="lp",
        output=output,
        free=free,
        state=state,
        proxy=proxy,
        polynomial_degree=polynomial_degree,
        productivity_degree=productivity_degree,
        functional_form=functional_form,
        boot_reps=boot_reps,
        seed=seed,
    )


def ackerberg_caves_frazer(
    data: pd.DataFrame,
    output: str = "y",
    free: Sequence[str] | str | None = None,
    state: Sequence[str] | str | None = None,
    proxy: str = "m",
    panel_id: str = "id",
    time: str = "year",
    polynomial_degree: int = 3,
    productivity_degree: int = 3,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
) -> ProductionResult:
    """Ackerberg-Caves-Frazer (2015) production function estimator.

    Corrects the OP / LP "functional dependence" identification problem:
    when free inputs (labor) are chosen at the same time as the proxy,
    the labor coefficient is *not* identified in the stage-1 polynomial.
    ACF moves all coefficient identification to stage 2, instrumenting
    free inputs with their *lagged* values and state inputs at the
    contemporaneous level (``prodest``'s ``acf`` option /
    ``prodestACF``).

    The just-identified moment conditions can have several roots. They are
    searched from the stage-1 coefficients and a grid of other starts; the
    estimate is the root nearest the stage-1 coefficients, every distinct
    root is listed in
    ``diagnostics["acf_roots"]``, and a ``RuntimeWarning`` names them when
    there is more than one.

    Parameters
    ----------
    data : DataFrame
        Long-form panel.
    output : str
        Log output.
    free : str or list, default ``["l"]``
        Free inputs — instrumented with their lag in stage 2.
    state : str or list, default ``["k"]``
        State inputs.
    proxy : str, default ``"m"``
        Intermediate input (materials).
    panel_id, time : str
        Firm and (numeric) year identifiers.
    polynomial_degree : int, default 3
        Degree of the stage-1 polynomial in (free, state, proxy).
    productivity_degree : int, default 3
        Degree of the productivity polynomial ``g``.
    functional_form : {'cobb-douglas', 'translog'}, default 'cobb-douglas'
        Translog adds ``0.5 * x_j**2`` and ``x_j * x_k`` terms, instrumented
        by the same expansion of (lagged free, state).
    boot_reps : int
        Firm-cluster bootstrap replications; each replication solves from
        the full-sample estimate and must reach a root.
    seed : int
        Bootstrap RNG seed.

    Returns
    -------
    ProductionResult

    Notes
    -----
    Requires at least two consecutive time periods per firm so that
    lagged labor exists.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for fid in range(30):
    ...     k = rng.normal(2.0, 0.5)
    ...     omega = rng.normal(0.0, 0.3)
    ...     for yr in range(8):
    ...         omega = 0.7 * omega + rng.normal(0.0, 0.2)
    ...         k = 0.9 * k + 0.3 * rng.normal(1.0, 0.3)
    ...         l = 0.6 * k + omega + rng.normal(1.0, 0.2)
    ...         m = 0.5 * k + 0.5 * l + omega + rng.normal(0.5, 0.2)
    ...         y = 0.6 * l + 0.3 * k + omega + rng.normal(0.0, 0.1)
    ...         rows.append(dict(id=fid, year=yr, y=y, l=l, k=k, m=m))
    >>> df = pd.DataFrame(rows)
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     res = sp.ackerberg_caves_frazer(
    ...         df, output="y", free="l", state="k", proxy="m"
    ...     )
    >>> sorted(res.coef.keys())
    ['k', 'l']
    >>> res.method
    'acf'

    References
    ----------
    Ackerberg, D.A., Caves, K. & Frazer, G. (2015). Identification
    properties of recent production function estimators. Econometrica,
    83(6), 2411-2451.
    """
    free, state = _resolve_inputs(free, state, ["l"], ["k"])
    df = _prepare_panel(data, output, free, state, proxy, panel_id, time)
    return _estimate_acf(
        df,
        output=output,
        free=free,
        state=state,
        proxy=proxy,
        polynomial_degree=polynomial_degree,
        productivity_degree=productivity_degree,
        functional_form=functional_form,
        boot_reps=boot_reps,
        seed=seed,
    )
