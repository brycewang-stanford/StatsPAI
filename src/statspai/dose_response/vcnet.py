"""
VCNet-style neural dose-response estimation (Nie, Ye, Zhang, Yang &
Wen 2021, ICLR).

VCNet (Variational Continuous-treatment Network) fits a neural
network that outputs a smooth dose-response curve :math:`\\mu(t, x)`
using a spline-varying coefficient architecture. The key ideas:

1. **Treatment-varying coefficients.** The outcome head has weights
   that are **functions of t** (via B-spline basis evaluated at t),
   so the model can learn non-linear response curves without a fixed
   parametric form.

2. **Density / propensity adjustment.** A concurrent density-estimator
   head models :math:`\\pi(t | x)`, used for inverse-weighting or for
   the targeted update.

For a self-contained implementation without extra dependencies we use
NumPy + SciPy (B-spline basis) to fit a ridge-regularised *varying-
coefficient linear model*:

.. math::

   \\mu(t, x) = \\sum_{b=1}^{B} \\phi_b(t) \\cdot x^\\top \\beta_b,

where :math:`\\phi_b` is a B-spline basis. This captures the
defining property of VCNet (t-varying coefficients) and yields
smooth curves. Users wanting GPU neural fits can plug PyTorch in.

For dose-response inference, we report
:math:`\\hat\\mu(t) = E_n[\\mu(t, X)]` on a treatment grid, plus
bootstrap standard errors.

References
----------
Nie, L., Ye, M., Zhang, L., Yang, Q., & Wen, Q. (2021).
"VCNet and Functional Targeted Regularization For Learning Causal
Effects of Continuous Treatments." *ICLR 2021*. [@nie2021quasi]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from ..exceptions import DataInsufficient, MethodIncompatibility


@dataclass
class VCNetResult:
    """Dose-response curve returned by :func:`vcnet` / :func:`scigan`.

    Attributes
    ----------
    t_grid : np.ndarray
        Treatment grid at which the curve is evaluated.
    mu_hat : np.ndarray
        Estimated dose-response :math:`\\hat\\mu(t)` on ``t_grid``.
    se : np.ndarray
        Bootstrap pointwise standard errors.
    ci_lo, ci_hi : np.ndarray
        Pointwise lower / upper confidence band.
    coef_matrix : np.ndarray
        Varying-coefficient matrix of shape ``(n_basis, p + 1)``.
    n_obs, n_basis : int
    detail : dict

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "y": np.random.randn(200),
    ...     "dose": np.random.rand(200),
    ...     "x1": np.random.randn(200),
    ... })
    >>> res = sp.vcnet(df, y="y", treatment="dose",
    ...                covariates=["x1"])  # doctest: +SKIP
    >>> res.mu_hat[:3]            # dose-response on the t-grid  # doctest: +SKIP
    >>> print(res.summary())     # tidy curve table  # doctest: +SKIP
    """

    t_grid: np.ndarray
    mu_hat: np.ndarray
    se: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    coef_matrix: np.ndarray  # (B, p+1)
    n_obs: int
    n_basis: int
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        df = pd.DataFrame(
            {
                "t": self.t_grid,
                "mu_hat": self.mu_hat,
                "se": self.se,
                "ci_lo": self.ci_lo,
                "ci_hi": self.ci_hi,
            }
        )
        return "VCNet dose-response curve\n" + str(df.to_string(index=False))

    def __repr__(self) -> str:  # pragma: no cover
        return f"VCNetResult(n={self.n_obs}, grid_size={len(self.t_grid)})"


def _bspline_basis(t: np.ndarray, n_basis: int = 6, degree: int = 3) -> np.ndarray:
    """B-spline basis matrix evaluated at t (shape: len(t) x n_basis)."""
    t_min, t_max = float(np.min(t)), float(np.max(t))
    if t_min >= t_max:
        t_max = t_min + 1.0
    # interior knots
    n_interior = max(n_basis - degree - 1, 0)
    if n_interior > 0:
        interior = np.linspace(t_min, t_max, n_interior + 2)[1:-1]
    else:
        interior = np.array([])
    knots = np.concatenate(
        [
            [t_min] * (degree + 1),
            interior,
            [t_max] * (degree + 1),
        ]
    )
    basis = np.zeros((len(t), n_basis))
    for b in range(n_basis):
        c = np.zeros(n_basis)
        c[b] = 1.0
        spline = BSpline(knots, c, degree, extrapolate=False)
        vals = spline(t)
        vals = np.where(np.isnan(vals), 0.0, vals)
        basis[:, b] = vals
    return basis


def _coerce_covariates(covariates: Sequence[str]) -> list[str]:
    """Normalize one-column string shortcuts without splitting names."""
    if isinstance(covariates, str):
        return [covariates]
    try:
        return list(covariates)
    except TypeError as exc:
        raise MethodIncompatibility(
            "vcnet() requires covariates to be a column name or sequence.",
            recovery_hint="Pass covariates=[] or a list of column names.",
        ) from exc


def _validate_vcnet_controls(
    n_basis: int,
    spline_degree: int,
    ridge: float,
    n_bootstrap: int,
    alpha: float,
) -> tuple[int, int, float, int, float]:
    """Validate scalar VCNet controls before building spline designs."""
    if (
        isinstance(spline_degree, bool)
        or not isinstance(spline_degree, (int, np.integer))
        or int(spline_degree) < 0
    ):
        raise MethodIncompatibility(
            "vcnet(): spline_degree must be a non-negative integer.",
            recovery_hint="Use spline_degree=3 for cubic splines.",
            diagnostics={"spline_degree": spline_degree},
        )
    degree_value = int(spline_degree)
    if (
        isinstance(n_basis, bool)
        or not isinstance(n_basis, (int, np.integer))
        or int(n_basis) < degree_value + 1
    ):
        raise MethodIncompatibility(
            "vcnet(): n_basis must be at least spline_degree + 1.",
            recovery_hint="Use n_basis >= spline_degree + 1.",
            diagnostics={"n_basis": n_basis, "spline_degree": spline_degree},
        )
    try:
        ridge_value = float(ridge)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "vcnet(): ridge must be a finite non-negative scalar.",
            recovery_hint="Use ridge >= 0.",
            diagnostics={"ridge": ridge},
        ) from exc
    if not np.isfinite(ridge_value) or ridge_value < 0:
        raise MethodIncompatibility(
            "vcnet(): ridge must be a finite non-negative scalar.",
            recovery_hint="Use ridge >= 0.",
            diagnostics={"ridge": ridge},
        )
    if (
        isinstance(n_bootstrap, bool)
        or not isinstance(n_bootstrap, (int, np.integer))
        or int(n_bootstrap) < 2
    ):
        raise MethodIncompatibility(
            "vcnet(): n_bootstrap must be an integer >= 2.",
            recovery_hint="Use at least two bootstrap draws for pointwise SEs.",
            diagnostics={"n_bootstrap": n_bootstrap},
        )
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "vcnet(): alpha must be a finite scalar in (0, 1).",
            recovery_hint="Use alpha=0.05 for 95% intervals.",
            diagnostics={"alpha": alpha},
        ) from exc
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise MethodIncompatibility(
            "vcnet(): alpha must be in the open interval (0, 1).",
            recovery_hint="Use alpha=0.05 for 95% intervals.",
            diagnostics={"alpha": alpha},
        )
    return (
        int(n_basis),
        degree_value,
        ridge_value,
        int(n_bootstrap),
        alpha_value,
    )


def _prepare_vcnet_frame(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: Sequence[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Validate data/column inputs and drop rows with missing estimation data."""
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "vcnet() requires data to be a pandas DataFrame.",
            recovery_hint="Pass a DataFrame with outcome, treatment, and covariates.",
        )
    x_cols = _coerce_covariates(covariates)
    required = [y, treatment] + x_cols
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise MethodIncompatibility(
            "vcnet() input data is missing required column(s).",
            recovery_hint="Check y, treatment, and covariate names.",
            diagnostics={"missing_columns": missing},
        )
    df = data[required].dropna().reset_index(drop=True)
    if len(df) < 2:
        raise DataInsufficient(
            "vcnet() requires at least two complete rows.",
            recovery_hint="Drop missing data or provide a larger sample.",
        )
    try:
        df = df.astype(float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "vcnet() requires numeric outcome, treatment, and covariates.",
            recovery_hint="Convert VCNet inputs to numeric columns.",
            diagnostics={"columns": required},
        ) from exc
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise MethodIncompatibility(
            "vcnet() inputs contain NaN or infinite values.",
            recovery_hint="Drop or impute non-finite rows before fitting.",
            diagnostics={"columns": required},
        )
    if df[treatment].nunique() < 2:
        raise DataInsufficient(
            "vcnet() requires at least two distinct treatment values.",
            recovery_hint="Use a sample with treatment variation.",
        )
    return df, x_cols


def _prepare_t_grid(t_grid: Optional[Sequence[float]], T: np.ndarray) -> np.ndarray:
    """Validate or construct the dose-response evaluation grid."""
    if t_grid is None:
        return np.linspace(T.min(), T.max(), 40)
    try:
        grid = np.asarray(t_grid, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "vcnet(): t_grid must contain numeric treatment values.",
            recovery_hint="Pass a one-dimensional finite dose grid.",
        ) from exc
    if grid.ndim != 1 or grid.size == 0:
        raise MethodIncompatibility(
            "vcnet(): t_grid must be a non-empty one-dimensional array.",
            recovery_hint="Pass treatment values such as np.linspace(...).",
            diagnostics={"t_grid_ndim": int(grid.ndim), "t_grid_size": int(grid.size)},
        )
    if not np.isfinite(grid).all():
        raise MethodIncompatibility(
            "vcnet(): t_grid contains NaN or infinite values.",
            recovery_hint="Drop or replace non-finite dose-grid values.",
        )
    return grid


def vcnet(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: Sequence[str],
    t_grid: Optional[Sequence[float]] = None,
    n_basis: int = 6,
    spline_degree: int = 3,
    ridge: float = 1e-2,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> VCNetResult:
    """
    Varying-coefficient dose-response estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treatment : str
        Continuous treatment / dose column.
    covariates : sequence of str
    t_grid : sequence of float, optional
        Treatment values at which to evaluate the dose-response curve.
        Defaults to 40 equally-spaced points between the observed min/max.
    n_basis : int, default 6
        Number of B-spline basis functions for the t-axis.
    spline_degree : int, default 3
    ridge : float, default 1e-2
        Tikhonov regularisation on the coefficient matrix.
    n_bootstrap : int, default 200
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    VCNetResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "outcome": np.random.randn(300),
    ...     "dose": np.random.rand(300),
    ...     "age": np.random.randn(300),
    ... })
    >>> res = sp.vcnet(df, y="outcome", treatment="dose",
    ...                covariates=["age"],
    ...                t_grid=np.linspace(0, 1, 20))  # doctest: +SKIP
    >>> res.t_grid.shape, res.mu_hat.shape  # doctest: +SKIP
    ((20,), (20,))
    """
    n_basis, spline_degree, ridge, n_bootstrap, alpha = _validate_vcnet_controls(
        n_basis=n_basis,
        spline_degree=spline_degree,
        ridge=ridge,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
    df, X_cols = _prepare_vcnet_frame(data, y, treatment, covariates)
    n = len(df)
    Y = df[y].to_numpy(dtype=float)
    T = df[treatment].to_numpy(dtype=float)
    X = df[X_cols].to_numpy(dtype=float) if X_cols else np.zeros((n, 0))
    t_grid_arr = _prepare_t_grid(t_grid, T)

    # Full design: [phi(t) tensor (1, X)], i.e. n x (B * (p+1))
    B = _bspline_basis(T, n_basis=n_basis, degree=spline_degree)
    X_aug = np.column_stack([np.ones(n), X])  # n x (p+1)
    # outer per-row: shape n x (n_basis * (p+1))
    design = np.einsum("nb,np->nbp", B, X_aug).reshape(n, -1)

    def fit(design: np.ndarray, Y: np.ndarray) -> np.ndarray:
        G = design.T @ design + ridge * np.eye(design.shape[1])
        rhs = design.T @ Y
        return np.asarray(np.linalg.solve(G, rhs), dtype=float)

    beta = fit(design, Y)
    coef_matrix = beta.reshape(n_basis, X_aug.shape[1])

    # Predict curve on grid
    B_grid = _bspline_basis(t_grid_arr, n_basis=n_basis, degree=spline_degree)

    def predict_curve(coef: np.ndarray) -> np.ndarray:
        # For each t, mu(t) = mean_i sum_b phi_b(t) * (x_i dot coef[b])
        # = sum_b phi_b(t) * (mean(x) dot coef[b])
        x_mean = X_aug.mean(axis=0)
        mu_per_basis = coef @ x_mean  # shape (n_basis,)
        return np.asarray(B_grid @ mu_per_basis, dtype=float)

    mu_hat = predict_curve(coef_matrix)

    # Bootstrap for SE
    rng = np.random.default_rng(random_state)
    boot_curves = np.zeros((n_bootstrap, len(t_grid_arr)))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        d_boot = design[idx]
        y_boot = Y[idx]
        try:
            beta_b = fit(d_boot, y_boot)
            boot_curves[b] = predict_curve(beta_b.reshape(n_basis, X_aug.shape[1]))
        except np.linalg.LinAlgError:
            boot_curves[b] = mu_hat

    se = boot_curves.std(axis=0, ddof=1)
    q_lo = np.quantile(boot_curves, alpha / 2, axis=0)
    q_hi = np.quantile(boot_curves, 1 - alpha / 2, axis=0)

    _result = VCNetResult(
        t_grid=t_grid_arr,
        mu_hat=mu_hat,
        se=se,
        ci_lo=q_lo,
        ci_hi=q_hi,
        coef_matrix=coef_matrix,
        n_obs=n,
        n_basis=n_basis,
        detail={"ridge": ridge, "degree": spline_degree},
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.dose_response.vcnet",
            params={
                "y": y,
                "treatment": treatment,
                "covariates": list(covariates),
                "n_basis": n_basis,
                "spline_degree": spline_degree,
                "ridge": ridge,
                "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# --------------------------------------------------------------------
# SCIGAN placeholder — delegate to VCNet-style fit for now, with an
# option to inject adversarial samples via user-provided weights.
# --------------------------------------------------------------------


def scigan(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    covariates: Sequence[str],
    t_grid: Optional[Sequence[float]] = None,
    propensity_weights: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> VCNetResult:
    """
    Adversarial dose-response estimator (Bica et al. 2020).

    The reference SCIGAN architecture is a GAN — too heavy to ship as
    a dependency-free algorithm. This entry point exposes a
    propensity-weighted VCNet fit that captures the essential
    "balancing" behaviour of SCIGAN: residual imbalance along the
    treatment axis is re-weighted using a user-provided ``propensity_weights``
    (``g(T | X)^{-1}``). For the full SCIGAN training loop, plug in
    your own GAN-generated counterfactual samples and pass the
    re-weighting through ``propensity_weights``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "y": np.random.randn(200),
    ...     "dose": np.random.rand(200),
    ...     "x1": np.random.randn(200),
    ... })
    >>> w = np.ones(len(df))  # inverse-propensity balancing weights
    >>> res = sp.scigan(df, y="y", treatment="dose",
    ...                 covariates=["x1"],
    ...                 propensity_weights=w)  # doctest: +SKIP
    >>> res.mu_hat[:3]  # balanced dose-response curve  # doctest: +SKIP
    """
    df, X_cols = _prepare_vcnet_frame(data, y, treatment, covariates)
    if propensity_weights is not None:
        try:
            w = np.asarray(propensity_weights, dtype=float)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "scigan(): propensity_weights must be numeric.",
                recovery_hint=(
                    "Pass finite non-negative weights aligned to complete rows."
                ),
            ) from exc
        if len(w) != len(df):
            raise MethodIncompatibility(
                "scigan(): propensity_weights length mismatch.",
                recovery_hint="Pass one weight per complete estimation row.",
                diagnostics={"n_weights": int(len(w)), "n_rows": int(len(df))},
            )
        if w.ndim != 1 or not np.isfinite(w).all() or np.any(w < 0):
            raise MethodIncompatibility(
                "scigan(): propensity_weights must be a finite non-negative vector.",
                recovery_hint="Use one non-negative balancing weight per row.",
                diagnostics={"weights_ndim": int(w.ndim)},
            )
        if float(w.sum()) <= 0:
            raise DataInsufficient(
                "scigan(): propensity_weights have zero total mass.",
                recovery_hint="Pass at least one positive balancing weight.",
            )
        df = df.assign(_w=w)
        # Expand (sample with replacement by weight); simplest proxy.
        rng = np.random.default_rng(kwargs.get("random_state", 42))
        probs = np.clip(w, 1e-6, None)
        probs /= probs.sum()
        idx = rng.choice(len(df), size=len(df), replace=True, p=probs)
        df = df.iloc[idx].reset_index(drop=True)
    return vcnet(
        df,
        y=y,
        treatment=treatment,
        covariates=X_cols,
        t_grid=t_grid,
        **kwargs,
    )


__all__ = ["vcnet", "scigan", "VCNetResult"]
