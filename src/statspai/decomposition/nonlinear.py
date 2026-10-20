"""
Nonlinear (logit / probit) decomposition methods.

- **Fairlie (1999, 2005)**: simulation-based decomposition for binary
  outcomes using rank-based matching of predicted probabilities.
- **Bauer-Sinning (2008, 2010)**: Extension of Yun (2005) for arbitrary
  nonlinear models, providing analytical detailed decomposition based
  on weighted predictions.
- **Yun (2004, 2005)**: Detailed decomposition with weights that sum to
  1 (linearisation of nonlinear models).

All return both overall (explained / unexplained) and variable-level
detailed contributions.

References
----------
Fairlie, R.W. (2005). "An Extension of the Blinder-Oaxaca Decomposition
Technique to Logit and Probit Models." *Journal of Economic and Social
Measurement*, 30, 305-316. [@fairlie2005extension]

Bauer, T.K. & Sinning, M. (2008). "An Extension of the Blinder-Oaxaca
Decomposition to Nonlinear Models." *AStA Advances in Statistical
Analysis*, 92, 197-206. [@bauer2008extension]

Yun, M.-S. (2004). "Decomposing Differences in the First Moment."
*Economics Letters*, 82, 275-280. [@yun2004decomposing]

Yun, M.-S. (2005). "A Simple Solution to the Identification Problem in
Detailed Wage Decompositions." *Economic Inquiry*, 43(4), 766-772. [@yun2005simple]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility
from ._common import add_constant, logit_fit, logit_predict, prepare_frame
from ._results import DecompResultMixin

# ════════════════════════════════════════════════════════════════════════
# Probit helpers (optional)
# ════════════════════════════════════════════════════════════════════════


def _probit_fit(
    y: np.ndarray, X: np.ndarray, max_iter: int = 100, tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
    """Probit MLE via Newton-Raphson."""
    from scipy.stats import norm

    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -8, 8)
        phi = norm.pdf(eta)
        Phi = norm.cdf(eta)
        Phi = np.clip(Phi, 1e-10, 1 - 1e-10)
        # lambda
        lam = phi / Phi * y - phi / (1 - Phi) * (1 - y)
        grad = X.T @ lam
        # Hessian (approximate, expected information)
        w = phi**2 / (Phi * (1 - Phi))
        H = -(X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    # Fisher scoring converges only linearly for probit, so stopping at a
    # 1e-8 step left the estimates ~1e-7 short of the MLE (visible against
    # Stata's probit in the fifth-to-seventh digit of downstream
    # decompositions). Finish with full Newton steps on the observed
    # information, which converge quadratically from here.
    for _ in range(25):
        eta_u = X @ beta
        q = 2.0 * y - 1.0
        zq = q * eta_u
        lam_o = q * np.exp(norm.logpdf(zq) - norm.logcdf(zq))
        h = lam_o * (lam_o + eta_u)
        info_o = (X * h[:, None]).T @ X
        try:
            step = np.linalg.solve(info_o, X.T @ lam_o)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    eta = np.clip(X @ beta, -8, 8)
    phi = norm.pdf(eta)
    Phi = np.clip(norm.cdf(eta), 1e-10, 1 - 1e-10)
    w = phi**2 / (Phi * (1 - Phi))
    info = (X * w[:, None]).T @ X
    # info = X'WX is symmetric positive-semidefinite by construction, so a valid
    # covariance has a non-negative diagonal. On a rank-deficient (collinear)
    # design info is numerically singular; whether np.linalg.inv *raises*
    # LinAlgError there is LAPACK/BLAS-backend dependent — some builds silently
    # return an indefinite garbage matrix (huge negative variances) instead of
    # raising. Guard on the condition number rather than trusting the exception,
    # and fall back to the SVD pseudo-inverse, which stays PSD. Well-conditioned
    # fits keep the exact np.linalg.inv path, so their numbers are unchanged.
    if np.linalg.cond(info) < 1.0 / np.finfo(info.dtype).eps:
        vcov = np.linalg.inv(info)
    else:
        vcov = np.linalg.pinv(info)
    # A covariance is symmetric by definition; inv/pinv of the (symmetric)
    # information matrix can pick up asymmetric float noise on singular or
    # ill-conditioned designs, and that noise is BLAS-backend dependent. Force
    # symmetry — on a well-conditioned fit this is a ~1e-15 no-op.
    vcov = 0.5 * (vcov + vcov.T)
    return beta, vcov


def _probit_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return np.asarray(norm.cdf(np.clip(X @ beta, -8, 8)))


def _probit_oim_vcov(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Inverse observed information of the probit log likelihood (Stata e(V))."""
    from scipy.stats import norm

    eta = X @ beta
    q = 2.0 * y - 1.0
    z = q * eta
    lam = q * np.exp(norm.logpdf(z) - norm.logcdf(z))
    h = lam * (lam + eta)
    info = (X * h[:, None]).T @ X
    vcov = np.linalg.inv(info)
    return 0.5 * (vcov + vcov.T)


# ════════════════════════════════════════════════════════════════════════
# Result
# ════════════════════════════════════════════════════════════════════════


@dataclass
class NonlinearDecompResult(DecompResultMixin):
    method_name: ClassVar[str] = "Nonlinear Decomposition"
    bib_keys: ClassVar[Tuple[str, ...]] = (
        "fairlie2005extension",
        "bauer2008extension",
        "yun2004decomposing",
    )

    method: str
    model: str
    gap: float
    explained: float
    unexplained: float
    detailed: pd.DataFrame
    rate_a: float
    rate_b: float
    reference: int
    n_a: int
    n_b: int
    se: Optional[Dict[str, float]] = None
    detailed_unexplained: Optional[pd.DataFrame] = None
    vcov: Optional[pd.DataFrame] = None

    def summary(self) -> str:
        lines = [
            "━" * 62,
            f"  {self.method} Decomposition ({self.model.upper()})",
            "━" * 62,
            f"  Group A: mean(Y) = {self.rate_a:.4f}   N = {self.n_a}",
            f"  Group B: mean(Y) = {self.rate_b:.4f}   N = {self.n_b}",
            f"  Raw gap:      {self.gap:.4f}",
            f"  Explained:    {self.explained:.4f}"
            + (f"   SE={self.se['explained']:.4f}" if self.se else ""),
            f"  Unexplained:  {self.unexplained:.4f}"
            + (f"   SE={self.se['unexplained']:.4f}" if self.se else ""),
        ]
        if not self.detailed.empty:
            lines.append("")
            lines.append("  Detailed explained:")
            lines.append(self.detailed.round(4).to_string(index=False))
        if self.detailed_unexplained is not None:
            lines.append("")
            lines.append("  Detailed unexplained:")
            lines.append(self.detailed_unexplained.round(4).to_string(index=False))
        lines.append("━" * 62)
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs: Any) -> Any:
        from .plots import detailed_waterfall

        return detailed_waterfall(
            self.detailed, value_col="contribution", label_col="variable", **kwargs
        )

    def to_latex(self) -> str:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{self.method} Decomposition}}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"Component & Estimate \\",
            r"\midrule",
            f"Gap & {self.gap:.4f} \\\\",
            f"Explained & {self.explained:.4f} \\\\",
            f"Unexplained & {self.unexplained:.4f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (
            f"<div style='font-family:monospace;'>"
            f"<h3>{self.method} Decomposition</h3>"
            f"<p>Gap={self.gap:.4f}, Explained={self.explained:.4f}, "
            f"Unexplained={self.unexplained:.4f}</p></div>"
        )

    def __repr__(self) -> str:
        return (
            f"NonlinearDecompResult(method={self.method}, "
            f"model={self.model}, gap={self.gap:.4f})"
        )


# ════════════════════════════════════════════════════════════════════════
# Fairlie (2005)
# ════════════════════════════════════════════════════════════════════════


def fairlie(
    data: pd.DataFrame,
    y: str,
    group: str,
    x: Sequence[str],
    model: str = "logit",
    reference: int = 0,
    n_sim: int = 500,
    seed: Optional[int] = 12345,
) -> NonlinearDecompResult:
    """
    Fairlie (2005) nonlinear decomposition for binary outcomes.

    Procedure (as in Jann's Stata ``fairlie``): fit the binary model on the
    reference group; sort each group by its predicted probability; if the
    groups differ in size, draw a random subsample of the larger one the
    size of the smaller (kept in rank order); pair the two groups rank to
    rank; then switch the covariates from group A's (``group == 0``) values
    to group B's one variable at a time, in the order given. Variable
    ``j``'s contribution is the change in the mean predicted probability at
    its switch, averaged over ``n_sim`` subsamples. With equal group sizes
    there is no subsample and the decomposition is deterministic.

    Parameters
    ----------
    data : pd.DataFrame
    y : str — binary {0, 1}
    group : str — binary; ``0`` is group A, ``1`` group B (gap = A - B)
    x : Sequence[str]
    model : {'logit', 'probit'}
    reference : {0, 1} — whose coefficients to use (``fairlie``'s
        ``reference()``). The switching always runs from A's covariates to
        B's, so the contributions share the sign of the explained gap.
    n_sim : int — number of random subsamples to average over (``reps()``)
    seed : int or None

    Returns
    -------
    NonlinearDecompResult
        ``detailed`` carries each variable's contribution and its
        delta-method standard error (``fairlie``'s ``e(V)``: the variance
        of the mean-probability change given the model's coefficient
        covariance, averaged over subsamples). The contributions are NOT
        rescaled: they sum to the explained gap of the matched sample,
        which equals ``explained`` exactly when the groups are the same
        size.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> res = sp.fairlie(df, y="union", group="female",
    ...                  x=["education", "experience"], model="logit",
    ...                  n_sim=50, seed=0)
    >>> res.method
    'Fairlie'
    >>> list(res.detailed["variable"])
    ['education', 'experience']
    """
    if model not in ("logit", "probit"):
        raise MethodIncompatibility(f"model must be 'logit' or 'probit', got {model!r}")
    if reference not in (0, 1):
        raise MethodIncompatibility(f"reference must be 0 or 1, got {reference!r}")
    if int(n_sim) < 1:
        raise MethodIncompatibility("n_sim must be a positive integer.")
    cols = [y, group] + list(x)
    df, _ = prepare_frame(data, cols)
    g = df[group].astype(int).to_numpy()
    y_vec = df[y].astype(int).to_numpy()
    X_raw = df[list(x)].to_numpy(dtype=float)

    X = add_constant(X_raw)
    X_a = X[g == 0]
    X_b = X[g == 1]
    y_a = y_vec[g == 0]
    y_b = y_vec[g == 1]

    if len(y_a) < 10 or len(y_b) < 10:
        raise ValueError("Need ≥10 obs per group for Fairlie.")

    predict = logit_predict if model == "logit" else _probit_predict
    fit = logit_fit if model == "logit" else _probit_fit

    y_ref, X_ref = (y_a, X_a) if reference == 0 else (y_b, X_b)
    beta_ref, V_ref = fit(y_ref, X_ref)
    if model == "probit":
        V_ref = _probit_oim_vcov(y_ref, X_ref, beta_ref)

    def _dens(eta: np.ndarray) -> np.ndarray:
        if model == "logit":
            pr = 1.0 / (1.0 + np.exp(-eta))
            return np.asarray(pr * (1.0 - pr))
        from scipy.stats import norm

        return np.asarray(norm.pdf(eta))

    rate_a = float(np.mean(y_a))
    rate_b = float(np.mean(y_b))
    gap = rate_a - rate_b

    # Rank each group by its predicted probability under the reference
    # coefficients (stable sort, like Mata's order()).
    ord_a = np.argsort(predict(beta_ref, X_a), kind="mergesort")
    ord_b = np.argsort(predict(beta_ref, X_b), kind="mergesort")
    n_match = min(len(ord_a), len(ord_b))
    rng = np.random.default_rng(seed)

    k = X_raw.shape[1]
    contributions = np.zeros(k)
    variances = np.zeros(k)
    for _ in range(int(n_sim)):
        # Random subset of the larger group's ranks, kept in rank order.
        sa = (
            ord_a
            if len(ord_a) == n_match
            else ord_a[np.sort(rng.permutation(len(ord_a))[:n_match])]
        )
        sb = (
            ord_b
            if len(ord_b) == n_match
            else ord_b[np.sort(rng.permutation(len(ord_b))[:n_match])]
        )
        x_left = X_a[sa].copy()
        for j in range(k):
            x_right = x_left.copy()
            x_right[:, j + 1] = X_b[sb, j + 1]
            eta_l = x_left @ beta_ref
            eta_r = x_right @ beta_ref
            contributions[j] += float(
                np.mean(predict(beta_ref, x_left) - predict(beta_ref, x_right))
            )
            dx = (_dens(eta_l)[:, None] * x_left).mean(axis=0) - (
                _dens(eta_r)[:, None] * x_right
            ).mean(axis=0)
            variances[j] += float(dx @ V_ref @ dx)
            x_left = x_right
    contributions = contributions / n_sim
    variances = variances / n_sim

    # Total explained: reference coefficients on each group's full sample.
    explained = float(predict(beta_ref, X_a).mean() - predict(beta_ref, X_b).mean())
    unexplained = gap - explained

    detailed = pd.DataFrame(
        {
            "variable": list(x),
            "contribution": contributions,
            "se": np.sqrt(np.clip(variances, 0.0, None)),
            "pct_of_explained": (
                contributions / explained * 100
                if abs(explained) > 1e-12
                else np.zeros_like(contributions)
            ),
        }
    )

    return NonlinearDecompResult(
        method="Fairlie",
        model=model,
        gap=float(gap),
        explained=float(explained),
        unexplained=float(unexplained),
        detailed=detailed,
        rate_a=rate_a,
        rate_b=rate_b,
        reference=reference,
        n_a=int(len(y_a)),
        n_b=int(len(y_b)),
    )


# ════════════════════════════════════════════════════════════════════════
# Bauer-Sinning / Yun nonlinear
# ════════════════════════════════════════════════════════════════════════


def bauer_sinning(
    data: pd.DataFrame,
    y: str,
    group: str,
    x: Sequence[str],
    model: str = "logit",
    reference: int = 0,
    variant: str = "yun",
) -> NonlinearDecompResult:
    """
    Bauer-Sinning (2008) nonlinear Oaxaca-Blinder decomposition with
    Yun (2004, 2005) weights for detailed contributions.

    Implements the three-fold equivalent:
        gap = [E(p_a(X_a)) - E(p_r(X_a))]   # not used here
    but uses Yun's weight decomposition:
        explained_j = w_j · (E(p_r(X_a)) - E(p_r(X_b)))
    where w_j = (Δx̄_j · β_r_j) / Σ_k (Δx̄_k · β_r_k)

    Parameters
    ----------
    model : {'logit', 'probit'}
    reference : {0, 1}
    variant : {'yun'}  — reserved for future extensions

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> res = sp.bauer_sinning(df, y="union", group="female",
    ...                        x=["education", "experience"], model="logit")
    >>> res.method
    'Bauer-Sinning (Yun weights)'
    >>> bool(abs((res.explained + res.unexplained) - res.gap) < 1e-8)
    True
    """
    cols = [y, group] + list(x)
    df, _ = prepare_frame(data, cols)
    g = df[group].astype(int).to_numpy()
    y_vec = df[y].astype(int).to_numpy()
    X_raw = df[list(x)].to_numpy(dtype=float)

    X = add_constant(X_raw)
    X_a = X[g == 0]
    X_b = X[g == 1]
    y_a = y_vec[g == 0]
    y_b = y_vec[g == 1]

    predict = logit_predict if model == "logit" else _probit_predict
    fit = logit_fit if model == "logit" else _probit_fit

    beta_a, V_a = fit(y_a, X_a)
    beta_b, V_b = fit(y_b, X_b)
    if model == "probit":
        # Stata's probit e(V) is the inverse observed information; the fit
        # helper returns the expected information, which differs off the
        # canonical link. Use the observed Hessian for inference.
        V_a = _probit_oim_vcov(y_a, X_a, beta_a)
        V_b = _probit_oim_vcov(y_b, X_b, beta_b)

    p_a_obs = predict(beta_a, X_a).mean()
    p_b_obs = predict(beta_b, X_b).mean()
    gap = p_a_obs - p_b_obs

    # Yun (2004, 2005) detailed decomposition; the outputs are checked
    # against Stata mvdcmp (Powers, Yoshioka & Yun 2011) run as a black box.
    # Group "0" supplies the reference coefficients: A for reference=0, B for
    # reference=1, in which case every mvdcmp term is the negative of this
    # function's (the gap stays A - B).
    if reference == 0:
        x0, b0, V0, x1, b1, V1, sign = X_a, beta_a, V_a, X_b, beta_b, V_b, 1.0
    else:
        x0, b0, V0, x1, b1, V1, sign = X_b, beta_b, V_b, X_a, beta_a, V_a, -1.0

    def _pdf(Xm: np.ndarray, bm: np.ndarray) -> np.ndarray:
        if model == "logit":
            p = logit_predict(bm, Xm)
            return np.asarray(p * (1.0 - p))
        from scipy.stats import norm

        return np.asarray(norm.pdf(Xm @ bm))

    E = float(predict(b0, x0).mean() - predict(b0, x1).mean())
    C = float(predict(b0, x1).mean() - predict(b1, x1).mean())
    explained = sign * E
    unexplained = gap - explained

    m0 = x0.mean(axis=0)
    m1 = x1.mean(axis=0)
    k = len(b0)
    A_x = float((m0 - m1) @ b0)
    A_b = float(m1 @ (b0 - b1))
    Wdx = (m0 - m1) * b0 / A_x if abs(A_x) > 1e-300 else np.zeros(k)
    Wdb = m1 * (b0 - b1) / A_b if abs(A_b) > 1e-300 else np.zeros(k)
    eye = np.eye(k)
    dW = eye * ((m0 - m1) / A_x)[:, None] - np.outer(b0 * (m0 - m1), m0 - m1) / A_x**2
    dwA = eye * (m1 / A_b)[:, None] - np.outer(m1 * (b0 - b1), m1) / A_b**2
    dwB = -dwA
    g00 = (x0 * _pdf(x0, b0)[:, None]).mean(axis=0)
    g10 = (x1 * _pdf(x1, b0)[:, None]).mean(axis=0)
    g11 = (x1 * _pdf(x1, b1)[:, None]).mean(axis=0)
    dEdb = np.outer(Wdx, g00 - g10) + dW * E
    dCdb1 = np.outer(Wdb, g10) + dwA * C
    dCdb2 = dwB * C - np.outer(Wdb, g11)
    Z = np.zeros((k, k))
    J = np.block([[dEdb, Z], [dCdb1, dCdb2]])
    Vb = np.block([[V0, Z], [Z, V1]])
    eV = J @ Vb @ J.T  # covariance of (E_k, C_k); invariant to the sign flip
    se_E = float(np.sqrt(max((g00 - g10) @ V0 @ (g00 - g10), 0.0)))
    se_C = float(np.sqrt(max(g10 @ V0 @ g10 + g11 @ V1 @ g11, 0.0)))
    se_R = float(np.sqrt(max(eV.sum(), 0.0)))
    se_Ek = np.sqrt(np.clip(np.diag(eV)[:k], 0.0, None))
    se_Ck = np.sqrt(np.clip(np.diag(eV)[k:], 0.0, None))

    contributions = sign * Wdx * E
    detailed = pd.DataFrame(
        {
            "variable": list(x),
            "contribution": contributions[1:],
            "se": se_Ek[1:],
            "pct_of_explained": (
                contributions[1:] / explained * 100
                if abs(explained) > 1e-12
                else np.zeros(len(x))
            ),
        }
    )
    names = ["_cons"] + list(x)
    order = list(range(1, k)) + [0]
    detailed_unexplained = pd.DataFrame(
        {
            "variable": [names[i] for i in order],
            "contribution": [sign * Wdb[i] * C for i in order],
            "se": [se_Ck[i] for i in order],
        }
    )
    labels = [f"explained:{n}" for n in names] + [f"unexplained:{n}" for n in names]
    vcov = pd.DataFrame(eV, index=labels, columns=labels)
    se = {"explained": se_E, "unexplained": se_C, "gap": se_R}

    return NonlinearDecompResult(
        method="Bauer-Sinning (Yun weights)",
        model=model,
        gap=float(gap),
        explained=float(explained),
        unexplained=float(unexplained),
        detailed=detailed,
        rate_a=float(y_a.mean()),
        rate_b=float(y_b.mean()),
        reference=reference,
        n_a=int(len(y_a)),
        n_b=int(len(y_b)),
        se=se,
        detailed_unexplained=detailed_unexplained,
        vcov=vcov,
    )


# Alias for Yun nonlinear
yun_nonlinear = bauer_sinning
