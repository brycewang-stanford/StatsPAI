"""Mediation sensitivity analysis for linear models (Imai, Keele & Yamamoto, 2010).

Linear structural equations with a treatment ``T``, mediator ``M``,
outcome ``Y`` and pre-treatment covariates ``X`` [@imai2010identification]:

    M = a0 + a_T T + X'a_X + e2                (mediator model)
    Y = b0 + b_T T + b_M M + X'b_X + e3        (outcome model)

Under sequential ignorability the ACME is ``a_T * b_M``. The sensitivity
parameter is ``rho = Corr(e2, e3)``: an unobserved pre-treatment
confounder of the mediator and the outcome makes it non-zero, and for
each ``rho`` in ``(-1, 1)`` the ACME is again identified (Theorem 4 of
Imai, Keele & Yamamoto 2010). As that paper recommends, the two equations
are estimated jointly by iterated feasible GLS for a seemingly unrelated
regression whose error correlation is held at ``rho``, and the ACME's
standard error is the delta method applied to the GLS covariance.
Theorem 4 also gives the crossing point in closed form: ACME(rho) = 0
exactly at ``rho = Corr(e1, e2)``, with ``e1`` the residual of
``Y ~ T + X`` (the total-effect regression).

The coefficient-of-determination forms of Section 5.2 of the same paper
are reported at the crossing point: ``R2*_M R2*_Y = rho^2`` and
``R~2_M R~2_Y = rho^2 (1 - R2_M)(1 - R2_Y)``.

Clean-room implementation written from Imai, Keele & Yamamoto (2010) and
Imai, Keele & Tingley (2010); the reference (R ``mediation`` 4.5.1,
``medsens``, GPL >= 2) was used only as a black box (see
``docs/dev/cleanroom/mediate_sensitivity.md``).

References
----------
[@imai2010identification],
[@imai2010general]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ColumnNotFound, ConvergenceWarning, MethodIncompatibility


@dataclass
class MediateSensitivityResult(ResultProtocolMixin):
    rho_grid: np.ndarray  # (n_grid,)
    acme_at_rho: np.ndarray  # (n_grid,)
    rho_at_zero: Optional[float]  # ρ where ACME(ρ) = 0 (closed form)
    acme_at_zero: float  # ACME at ρ=0 (the baseline)
    se_at_rho: np.ndarray = field(default_factory=lambda: np.empty(0))
    ci_lower: np.ndarray = field(default_factory=lambda: np.empty(0))
    ci_upper: np.ndarray = field(default_factory=lambda: np.empty(0))
    rho_grid_zero: Optional[float] = None  # grid ρ closest to ACME = 0
    r2_mediator: float = float("nan")
    r2_outcome: float = float("nan")
    r2star_threshold: float = float("nan")  # rho_at_zero ** 2
    r2tilde_threshold: float = float("nan")
    conf_level: float = 0.95

    def plot(
        self,
        ax: Any = None,
        *,
        fill: bool = True,
        annotate: bool = True,
        figsize: Any = (7.0, 4.5),
        **kwargs: Any,
    ) -> Any:
        """Publication-style sensitivity plot.

        Shows ACME(ρ) vs the mediator-outcome confounder strength ρ,
        with a coloured fill for the *region of nullability* (any ρ in
        ``[ρ_at_zero, 1]`` flips the ACME sign), the ρ at which the
        ACME crosses zero (annotated), and reference lines at ρ=0
        (i.e. sequential-ignorability) and ACME=0.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        fill : bool, default True
            Fill the {ACME(ρ) > 0} region in light blue and the
            {ACME(ρ) < 0} region in light red, à la sensemakr.
        annotate : bool, default True
            Annotate ρ_at_zero, baseline ACME, and an interpretive note.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib required for plot()") from e
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        rho = self.rho_grid
        acme = self.acme_at_rho

        if fill:
            ax.fill_between(
                rho,
                0,
                acme,
                where=(acme >= 0),
                color="#1f77b4",
                alpha=0.18,
                label="ACME ≥ 0",
            )
            ax.fill_between(
                rho,
                0,
                acme,
                where=(acme < 0),
                color="#d62728",
                alpha=0.18,
                label="ACME < 0",
            )
        ax.plot(rho, acme, color="#1f77b4", lw=1.8)
        ax.axhline(0, color="#333", lw=0.7)
        ax.axvline(
            0, color="#333", lw=0.7, ls=":", label=r"$\rho=0$ (sequential ignorability)"
        )

        # Mark the baseline ACME and ρ_at_zero
        ax.scatter([0.0], [self.acme_at_zero], color="#1f77b4", s=60, zorder=5)
        if annotate:
            ax.annotate(
                f"baseline ACME = {self.acme_at_zero:+.3f}",
                xy=(0.0, self.acme_at_zero),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
            )

        if self.rho_at_zero is not None:
            ax.axvline(
                self.rho_at_zero,
                color="#d62728",
                lw=1.2,
                ls="--",
                label=rf"ACME=0 at $\rho={self.rho_at_zero:.3f}$",
            )
            if annotate:
                ax.annotate(
                    "robustness threshold",
                    xy=(self.rho_at_zero, 0),
                    xytext=(6, -20),
                    textcoords="offset points",
                    fontsize=8,
                    color="#d62728",
                )

        ax.set_xlabel(r"$\rho$ — confounder strength on (mediator, outcome) errors")
        ax.set_ylabel(r"$\widehat{\mathrm{ACME}}(\rho)$")
        ax.set_title("Mediation sensitivity analysis (Imai-Keele-Yamamoto 2010)")
        ax.legend(loc="best", fontsize=8)
        return fig, ax

    def summary(self) -> str:
        lines = [
            "Mediation Sensitivity (Imai et al. 2010)",
            "-" * 45,
            f"Baseline ACME (ρ=0) : {self.acme_at_zero: .4f}",
        ]
        if self.rho_at_zero is not None:
            lines.append(f"ρ at which ACME = 0  : {self.rho_at_zero: .4f}")
            lines.append(
                f"Interpretation: unobserved confounding with |ρ| > "
                f"{abs(self.rho_at_zero):.2f} would explain away the "
                f"estimated mediation effect."
            )
            lines.append(
                f"R²* (product) at ACME = 0 : {self.r2star_threshold: .4f}   "
                f"R̃² (product) : {self.r2tilde_threshold: .4f}"
            )
        else:
            lines.append("ACME(ρ) never reaches zero for |ρ| < 1.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# Default stopping rule (``eps=None``): the squared step of the stacked
# coefficient vector is below (1e-13)^2 * (1 + |coef|^2), i.e. the iteration
# has reached its fixed point to the last few digits.
_FIXED_POINT_TOL = 1e-13


def _gls_fixed_correlation(Xm, m, Xy, y, rho, sd_m, sd_y):
    """One GLS step for the two-equation system with error covariance
    ``[[sd_m^2, rho sd_m sd_y], [rho sd_m sd_y, sd_y^2]] (x) I_n``.

    Returns the stacked coefficients ``[a; b]`` and ``X' Omega^{-1} X``.
    """
    cov = np.array([[sd_m**2, rho * sd_m * sd_y], [rho * sd_m * sd_y, sd_y**2]])
    p = np.linalg.inv(cov)
    info = np.block(
        [
            [p[0, 0] * Xm.T @ Xm, p[0, 1] * Xm.T @ Xy],
            [p[1, 0] * Xy.T @ Xm, p[1, 1] * Xy.T @ Xy],
        ]
    )
    rhs = np.concatenate(
        [Xm.T @ (p[0, 0] * m + p[0, 1] * y), Xy.T @ (p[1, 0] * m + p[1, 1] * y)]
    )
    return np.linalg.solve(info, rhs), info


def _fixed_correlation_sur(Xm, m, Xy, y, rho, eps, max_iter):
    """Iterated feasible GLS with the error correlation held at ``rho``.

    Starts from equation-by-equation least squares; each step re-estimates
    the two error standard deviations from the current residuals (divisor
    ``n - 1``) and re-solves the GLS normal equations. It stops when the
    sum of squared changes of the stacked coefficients falls below ``eps``
    (``eps=None``: the fixed point, see ``_FIXED_POINT_TOL``). The returned
    covariance is ``(X' Omega^{-1} X)^{-1}`` of the final step.

    Returns ``(coef, cov, converged, n_steps)``.
    """
    km = Xm.shape[1]
    coef = np.concatenate(
        [
            np.linalg.lstsq(Xm, m, rcond=None)[0],
            np.linalg.lstsq(Xy, y, rcond=None)[0],
        ]
    )
    info = None
    converged = False
    steps = 0
    for steps in range(1, max_iter + 1):
        sd_m = np.std(m - Xm @ coef[:km], ddof=1)
        sd_y = np.std(y - Xy @ coef[km:], ddof=1)
        new, info = _gls_fixed_correlation(Xm, m, Xy, y, rho, sd_m, sd_y)
        change = float(np.sum((new - coef) ** 2))
        coef = new
        tol = eps if eps is not None else _FIXED_POINT_TOL**2 * (1 + coef @ coef)
        if change < tol:
            converged = True
            break
    return coef, np.linalg.inv(info), converged, steps


def _ols_resid(X, v):
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def _r_squared(X, v):
    e = _ols_resid(X, v)
    c = v - v.mean()
    return float(1 - (e @ e) / (c @ c))


def _require_full_rank(X, label):
    sv = np.linalg.svd(X, compute_uv=False)
    if sv[-1] <= 1e-10 * sv[0]:
        raise MethodIncompatibility(
            f"mediate_sensitivity: the {label} design is not full rank "
            "(a covariate is constant or a linear combination of the others)",
            recovery_hint="Drop the redundant or constant covariate.",
        )


def mediate_sensitivity(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    covariates: Optional[list] = None,
    rho_range: tuple = (-0.9, 0.9),
    n_grid: int = 41,
    *,
    conf_level: float = 0.95,
    eps: Optional[float] = None,
    max_iter: int = 10_000,
) -> MediateSensitivityResult:
    """Sensitivity analysis for causal mediation (linear models).

    For each ``rho`` on a grid, the mediator model ``M ~ T + X`` and the
    outcome model ``Y ~ T + M + X`` are re-estimated jointly with their
    error correlation fixed at ``rho``, and the ACME ``a_T * b_M`` is
    reported with a delta-method confidence interval (Imai, Keele &
    Yamamoto 2010, Section 5). ``rho = 0`` is sequential ignorability.

    Parameters
    ----------
    data : pandas.DataFrame
        Analysis data; rows with a missing value in a used column are
        dropped.
    y : str
        Outcome column (linear model).
    treat : str
        Treatment column. With a 0/1 treatment the ACME is the effect of
        moving from control to treatment; with a numeric treatment it is
        the effect per unit.
    mediator : str
        Mediator column (linear model).
    covariates : list of str, optional
        Pre-treatment covariates entering both models.
    rho_range : tuple of float, default (-0.9, 0.9)
        ``(low, high)`` with ``-1 < low <= high < 1``.
    n_grid : int, default 41
        Number of equally spaced ``rho`` values (``numpy.linspace``).
    conf_level : float, default 0.95
        Level of the normal-approximation interval at each ``rho``.
    eps : float, optional
        Stopping tolerance of the iterated FGLS: iteration stops once the
        sum of squared changes of the coefficients is below ``eps``
        (``mediation::medsens`` uses ``sqrt(.Machine$double.eps)``). The
        default ``None`` iterates to the fixed point.
    max_iter : int, default 10000
        Maximum FGLS steps per ``rho``; a ``ConvergenceWarning`` names the
        ``rho`` values that reach it.

    Returns
    -------
    MediateSensitivityResult
        ``rho_grid``, ``acme_at_rho``, ``se_at_rho``, ``ci_lower`` /
        ``ci_upper``; ``acme_at_zero`` (least squares, ``rho = 0``);
        ``rho_at_zero``, the exact ``rho`` at which the ACME is zero
        (``Corr(resid(M ~ T + X), resid(Y ~ T + X))``); ``rho_grid_zero``,
        the grid value with the smallest ``|ACME|``; ``r2_mediator`` /
        ``r2_outcome`` (R^2 of the two models); ``r2star_threshold``
        ``= rho_at_zero^2`` and ``r2tilde_threshold = rho_at_zero^2
        (1 - R^2_M)(1 - R^2_Y)``, the products of the partial and total
        R^2 an unobserved confounder needs to remove the ACME.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> t = rng.binomial(1, 0.5, 400)
    >>> m = 0.5 * t + rng.normal(size=400)
    >>> df = pd.DataFrame({"t": t, "m": m,
    ...                    "y": t + 0.8 * m + rng.normal(size=400)})
    >>> res = sp.mediate_sensitivity(df, y="y", treat="t", mediator="m",
    ...                              n_grid=19)
    >>> float(round(res.rho_grid[0], 2)), res.rho_at_zero is not None
    (-0.9, True)

    References
    ----------
    [@imai2010identification], [@imai2010general]

    Clean-room implementation; the reference (R ``mediation`` 4.5.1,
    ``medsens``, GPL >= 2) was used only as a black box.
    """
    if not 0 < conf_level < 1:
        raise MethodIncompatibility(
            f"conf_level must be in (0, 1); got {conf_level}",
            recovery_hint="Pass a level such as conf_level=0.95.",
        )
    lo, hi = (float(v) for v in rho_range)
    if not -1 < lo <= hi < 1:
        raise MethodIncompatibility(
            "rho_range must lie strictly inside (-1, 1) with low <= high; "
            f"got {tuple(rho_range)}",
            recovery_hint="Pass a range such as rho_range=(-0.9, 0.9).",
        )
    if int(n_grid) != n_grid or n_grid < 1:
        raise MethodIncompatibility(
            f"n_grid must be a positive integer; got {n_grid}",
            recovery_hint="Pass n_grid=41 (or any integer >= 1).",
        )
    if eps is not None and not eps > 0:
        raise MethodIncompatibility(
            f"eps must be positive or None; got {eps}",
            recovery_hint="Leave eps=None to iterate to the fixed point.",
        )
    covariates = list(covariates or [])
    cols = [y, treat, mediator] + covariates
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ColumnNotFound(
            f"mediate_sensitivity: column(s) {missing} not found in data",
            recovery_hint="Check the outcome, treatment, mediator and covariate names.",
        )
    frame = data[cols].dropna()
    n = len(frame)
    Yv = frame[y].to_numpy(dtype=float)
    Mv = frame[mediator].to_numpy(dtype=float)
    Tv = frame[treat].to_numpy(dtype=float)
    Xc = frame[covariates].to_numpy(dtype=float).reshape(n, -1)
    Xm = np.column_stack([np.ones(n), Tv, Xc])
    Xy = np.column_stack([np.ones(n), Tv, Mv, Xc])
    if n <= Xy.shape[1] + 1:
        raise MethodIncompatibility(
            f"mediate_sensitivity: {n} complete rows for {Xy.shape[1]} "
            "outcome-model coefficients",
            recovery_hint="Use fewer covariates or more data.",
        )
    _require_full_rank(Xm, "mediator-model")
    _require_full_rank(Xy, "outcome-model")
    km = Xm.shape[1]
    i_t, i_m = 1, km + 2  # a_T in the mediator block, b_M in the outcome block

    a_hat = np.linalg.lstsq(Xm, Mv, rcond=None)[0]
    b_hat = np.linalg.lstsq(Xy, Yv, rcond=None)[0]
    acme_zero = float(a_hat[1] * b_hat[2])
    r2_m = _r_squared(Xm, Mv)
    r2_y = _r_squared(Xy, Yv)
    e_med = _ols_resid(Xm, Mv)
    e_tot = _ols_resid(Xm, Yv)  # Y ~ T + X: same regressors as the mediator model
    denom = np.sqrt((e_med @ e_med) * (e_tot @ e_tot))
    rho_tilde = float(e_med @ e_tot / denom) if denom > 0 else np.nan
    rho_at_zero = rho_tilde if np.isfinite(rho_tilde) else None

    grid = np.linspace(lo, hi, int(n_grid))
    acme = np.empty(grid.size)
    se = np.empty(grid.size)
    stalled = []
    for k, rho in enumerate(grid):
        coef, cov, ok, _ = _fixed_correlation_sur(
            Xm, Mv, Xy, Yv, float(rho), eps, int(max_iter)
        )
        a_t, b_m = coef[i_t], coef[i_m]
        acme[k] = a_t * b_m
        var = (
            b_m**2 * cov[i_t, i_t]
            + a_t**2 * cov[i_m, i_m]
            + 2 * a_t * b_m * cov[i_t, i_m]
        )
        se[k] = np.sqrt(max(var, 0.0))
        if not ok:
            stalled.append(float(rho))
    if stalled:
        warnings.warn(
            f"mediate_sensitivity: the iterated FGLS reached max_iter={max_iter} "
            f"before converging at rho = {', '.join(f'{r:.4f}' for r in stalled)}; "
            "those estimates are the last iterate.",
            ConvergenceWarning,
            stacklevel=2,
        )
    z = stats.norm.ppf(0.5 + conf_level / 2)
    r2star = rho_at_zero**2 if rho_at_zero is not None else float("nan")
    return MediateSensitivityResult(
        rho_grid=grid,
        acme_at_rho=acme,
        rho_at_zero=rho_at_zero,
        acme_at_zero=acme_zero,
        se_at_rho=se,
        ci_lower=acme - z * se,
        ci_upper=acme + z * se,
        rho_grid_zero=float(grid[int(np.argmin(np.abs(acme)))]),
        r2_mediator=r2_m,
        r2_outcome=r2_y,
        r2star_threshold=r2star,
        r2tilde_threshold=r2star * (1 - r2_m) * (1 - r2_y),
        conf_level=conf_level,
    )
