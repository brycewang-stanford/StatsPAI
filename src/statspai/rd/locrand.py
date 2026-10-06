"""
Local randomization inference for regression discontinuity designs.

Implements the methodology of Cattaneo, Titiunik, and Vazquez-Bare (2016) for
inference in RD designs under a local randomization assumption. Within a small
window around the cutoff, units are treated as if randomly assigned to
treatment or control.

Functions
---------
rdrandinf : Main randomization inference for RD designs.
rdwinselect : Data-driven window selection for local randomization.
rdsensitivity : Sensitivity of results across different windows.
rdrbounds : Rosenbaum sensitivity bounds for hidden bias.

References
----------
Cattaneo, M.D., Titiunik, R. and Vazquez-Bare, G. (2016).
"Inference in Regression Discontinuity Designs under Local Randomization."
*The Stata Journal*, 16(2), 331-367. [@cattaneo2016inference]
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import special as sp_special
from scipy import stats as sp_stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, IdentificationFailure, MethodIncompatibility
from . import _locrand_core as _lr
from ._core import _complete_cases


def _drop_incomplete(frame: Any, columns: Any, *, where: str) -> Any:
    """Drop non-finite rows and say so, rather than propagating a NaN.

    The local-randomization entry points had no missing-data handling,
    while ``sp.rdrobust`` in the same subpackage has always dropped
    incomplete rows. The asymmetry was not benign: a missing outcome
    inside the window turned the difference in means into NaN, and since
    ``abs(nan) >= abs(nan)`` is False for every draw, the permutation
    count stayed at zero and the reported p-value came out exactly 0.000
    -- the most significant answer the test can give, produced by a
    statistic that never computed. Dropping silently would trade one
    quiet failure for another, so the count is warned (CLAUDE.md §3.7).
    """
    cleaned, n_dropped = _complete_cases(frame, columns)
    if n_dropped:
        warnings.warn(
            f"{where}: dropped {n_dropped} observation(s) inside the window "
            f"with a missing or non-finite value in {[c for c in columns if c]}. "
            f"{len(cleaned)} remain. Randomization inference is run on the "
            "complete cases only.",
            UserWarning,
            stacklevel=3,
        )
    return cleaned, n_dropped


# ======================================================================
# Citation registration
# ======================================================================

CausalResult._CITATIONS["rdlocrand"] = (
    "@article{cattaneo2016inference,\n"
    "  title={Inference in Regression Discontinuity Designs under\n"
    "  Local Randomization},\n"
    "  author={Cattaneo, Matias D and Titiunik, Roc{\\'\\i}o and\n"
    "  Vazquez-Bare, Gonzalo},\n"
    "  journal={The Stata Journal},\n"
    "  volume={16},\n"
    "  number={2},\n"
    "  pages={331--367},\n"
    "  year={2016}\n"
    "}"
)


# ======================================================================
# Internal helpers
# ======================================================================


def _window_bounds(
    c: float, wl: Optional[float], wr: Optional[float]
) -> Tuple[float, float]:
    """Validate the window endpoints, which are on the running-variable scale."""
    if wl is None or wr is None:
        raise MethodIncompatibility(
            "Window bounds wl and wr must be specified. "
            "Use rdwinselect() to choose a data-driven window."
        )
    wl_value, wr_value = float(wl), float(wr)
    if not (wl_value <= c <= wr_value):
        raise MethodIncompatibility(
            f"The window [wl, wr] = [{wl_value}, {wr_value}] does not contain "
            f"the cutoff c = {c}. wl and wr are the window's endpoints on the "
            "scale of the running variable, as in rdlocrand -- not offsets "
            f"from the cutoff. For a window of half-width w use wl={c} - w, "
            f"wr={c} + w."
        )
    return wl_value, wr_value


def _select_window(data: pd.DataFrame, x: str, wl: float, wr: float) -> np.ndarray:
    """Return mask for observations with the running variable in [wl, wr]."""
    xv = data[x].values
    return np.asarray((xv >= wl) & (xv <= wr), dtype=bool)


def _polynomial_residuals(
    y: np.ndarray, x: np.ndarray, p: int, covs: Optional[np.ndarray] = None
) -> np.ndarray:
    """Partial out polynomial in X (and optional covariates) from Y."""
    n = len(y)
    parts = []
    # Polynomial terms x^1, ..., x^p (if p > 0)
    if p > 0:
        parts.extend([x**k for k in range(1, p + 1)])
    # Covariates
    if covs is not None:
        if covs.ndim == 1:
            parts.append(covs.reshape(-1, 1))
        elif covs.shape[1] > 0:
            parts.append(covs)

    if len(parts) == 0:
        return np.asarray(y - np.mean(y), dtype=float)

    X_design = np.column_stack(parts)
    X_design = np.column_stack([np.ones(n), X_design])
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    return np.asarray(y - X_design @ beta, dtype=float)


def _diffmeans(y: np.ndarray, d: np.ndarray) -> float:
    """Difference in means: E[Y|D=1] - E[Y|D=0]."""
    return float(y[d == 1].mean() - y[d == 0].mean())


def _ks_stat(y: np.ndarray, d: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic."""
    stat, _ = sp_stats.ks_2samp(y[d == 1], y[d == 0])
    return float(stat)


def _ranksum_stat(y: np.ndarray, d: np.ndarray) -> float:
    """Standardised Wilcoxon rank-sum statistic, as in ``rdrandinf``.

    ``T`` is the rank sum of the CONTROL group, standardised by

        E[T] = n0 (n + 1) / 2,   Var[T] = n0 n1 s^2 / n

    where ``s^2`` is the sample variance of the midranks. Using ``s^2``
    rather than the closed form ``(n + 1) / 12`` is what makes this
    tie-robust, and the senate running variable is heavily tied.

    The previous implementation returned ``|U - mu| / sigma`` from scipy's
    Mann-Whitney U with the no-ties variance, which is a different statistic
    -- roughly 2-3x off rdlocrand's on ties-heavy data -- and discarded the
    sign.
    """
    n1 = int((d == 1).sum())
    n0 = int((d == 0).sum())
    n = n1 + n0
    if n1 == 0 or n0 == 0 or n < 2:
        return 0.0
    ri = sp_stats.rankdata(y)  # midranks for ties, matching R's rank()
    t_stat = ri[d == 0].sum()
    s2 = float(np.var(ri, ddof=1))
    var_t = n0 * n1 * s2 / n
    if var_t <= 0:
        return 0.0
    return float((t_stat - n0 * (n + 1) / 2.0) / np.sqrt(var_t))


_STAT_FUNCS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "diffmeans": _diffmeans,
    "ksmirnov": _ks_stat,
    "ranksum": _ranksum_stat,
}


def _compute_stat(y: np.ndarray, d: np.ndarray, stat_name: str) -> float:
    """Dispatch to the requested test statistic."""
    return _STAT_FUNCS[stat_name](y, d)


def _permutation_pvalue(
    y: np.ndarray,
    d: np.ndarray,
    stat_name: str,
    n_perms: int,
    rng: np.random.Generator,
    two_sided: bool = True,
) -> Tuple[float, float]:
    """
    Fisher randomization p-value via permutation.

    Returns (observed_stat, perm_pvalue).
    """
    obs_stat = _compute_stat(y, d, stat_name)
    draws = (rng.permutation(d) for _ in range(n_perms))
    perm_pval = _pvalue_from_assignments(y, obs_stat, draws, stat_name, two_sided)
    return obs_stat, perm_pval


def _pvalue_from_assignments(
    y: np.ndarray,
    obs_stat: float,
    assignments: Any,
    stat_name: str,
    two_sided: bool = True,
) -> float:
    """Randomization p-value over a given sequence of assignment vectors.

    The share of ``assignments`` whose statistic is at least as extreme as
    ``obs_stat``. Separated from the random draws so that the p-value can
    be recomputed on any fixed set of assignments -- the reference-parity
    test feeds it R's own ``sample()`` draws and reproduces
    ``rdlocrand::rdrandinf``'s p-value exactly.
    """
    abs_obs = abs(obs_stat) if two_sided else obs_stat
    count = 0
    total = 0
    for d_perm in assignments:
        perm_stat = _compute_stat(y, np.asarray(d_perm), stat_name)
        perm_abs = abs(perm_stat) if two_sided else perm_stat
        if perm_abs >= abs_obs - 1e-14:
            count += 1
        total += 1
    return count / total


def _asymptotic_pvalue(
    y: np.ndarray, d: np.ndarray, stat_name: str
) -> Tuple[float, float]:
    """
    Asymptotic p-value for the chosen test statistic.

    Returns (stat, pvalue).
    """
    if stat_name == "diffmeans":
        y1, y0 = y[d == 1], y[d == 0]
        n1, n0 = len(y1), len(y0)
        if n1 < 2 or n0 < 2:
            return _diffmeans(y, d), np.nan
        diff = y1.mean() - y0.mean()
        se = np.sqrt(y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0)
        if se < 1e-14:
            return diff, 0.0 if abs(diff) > 1e-14 else 1.0
        t = diff / se
        # Normal, not t. Two reasons, and they agree:
        #
        # * ``se`` above is the Welch (unequal-variance) standard error, so
        #   referring it to a t with ``n1 + n0 - 2`` degrees of freedom mixes
        #   two different tests -- Welch's needs Satterthwaite's df, not the
        #   pooled one.
        # * rdlocrand calls this quantity the *asymptotic* p-value and uses
        #   ``2 * pnorm(-|t|)``, which is what the name means.
        #
        # The old form was conservative but wrong by a factor that grows with
        # the statistic: on rdsenate at w=+/-5 it reported 1.68e-10 where
        # rdlocrand reports 2.49e-11, a factor of 6.7.
        pval = 2 * sp_stats.norm.cdf(-abs(t))
        return float(diff), float(pval)
    elif stat_name == "ksmirnov":
        y1, y0 = y[d == 1], y[d == 0]
        n1, n0 = len(y1), len(y0)
        stat = float(_lr.ks_from_labels(y, np.asarray(d)[None, :])[0])
        # R's ks.test, which rdlocrand calls, uses the exact conditional
        # distribution (ties included) while n1 * n0 < 10000 and
        # Kolmogorov's limit beyond that.
        if n1 * n0 < 10000:
            pval = _lr.ks_exact_pvalue(y1, y0, stat)
        else:
            pval = float(sp_special.kolmogorov(np.sqrt(n1 * n0 / (n1 + n0)) * stat))
        return stat, float(pval)
    elif stat_name == "ranksum":
        # The statistic is already standardised, so the asymptotic p-value
        # is the normal tail -- same as rdlocrand. scipy's mannwhitneyu
        # p-value does not correspond to this statistic.
        stat = _ranksum_stat(y, d)
        return float(stat), float(2 * sp_stats.norm.cdf(-abs(stat)))
    else:
        raise MethodIncompatibility(
            f"Unknown statistic: {stat_name}"
        )  # pragma: no cover


def _tsls_wald(
    y: np.ndarray, d_actual: np.ndarray, z: np.ndarray
) -> Tuple[float, float, float]:
    """Wald / 2SLS estimate with one binary instrument.

    Returns ``(estimate, se, first_stage)``. The standard error is the
    heteroskedasticity-robust (HC1) 2SLS one, which is what ``rdlocrand``
    refers to the normal for its ``tsls`` statistic.
    """
    n = y.shape[0]
    first_stage = float(d_actual[z == 1].mean() - d_actual[z == 0].mean())
    if abs(first_stage) < 1e-12:
        raise IdentificationFailure(
            "Fuzzy RD: treatment take-up does not change at the cutoff inside "
            "this window (first stage = 0), so the Wald ratio is undefined."
        )
    Z = np.column_stack([np.ones(n), z.astype(float)])
    X = np.column_stack([np.ones(n), d_actual])
    A = np.linalg.inv(Z.T @ X)
    beta = A @ (Z.T @ y)
    resid = y - X @ beta
    meat = (Z * (resid**2)[:, None]).T @ Z
    V = A @ meat @ A.T * n / (n - 2)
    return float(beta[1]), float(np.sqrt(V[1, 1])), first_stage


def _resolve_vce(vce: str, p: int) -> str:
    """The variance of the large-sample test: ``vce`` when ``p > 0``.

    Without polynomial adjustment the statistic is a difference in
    (weighted) means and its standard error is the HC2 one, Welch's with
    equal weights, whatever ``vce`` says: ``rdlocrand`` ignores the option
    there too.
    """
    key = str(vce).lower()
    if key not in ("hc1", "hc2", "hc3"):
        raise MethodIncompatibility(
            f"vce must be 'hc1', 'hc2' or 'hc3', got {vce!r}.",
            diagnostics={"vce": repr(vce)},
        )
    return key if p > 0 else "hc2"


def _randomization_test(
    y: np.ndarray,
    xc: np.ndarray,
    z: np.ndarray,
    *,
    shift: np.ndarray,
    stat_names: List[str],
    p: int,
    kernel: str,
    bw: Tuple[float, float],
    evals: Tuple[float, float],
    n_perms: int,
    rng: np.random.Generator,
    prob: Optional[np.ndarray],
    nulltau: float,
    ci_grid: Optional[np.ndarray],
    alpha: float,
    keep_draws: bool = False,
    vce: str = "hc2",
) -> Dict[str, Any]:
    """Observed statistics, randomization p-values and the inverted interval.

    ``shift`` is what a unit of treatment effect adds to the outcome: the
    assignment indicator in a sharp design, the treatment actually received
    in a fuzzy one. Under ``H0: tau = tau0`` the outcome ``y - tau0 * shift``
    is fixed, which is what gets re-randomized.

    Two randomization schemes, chosen by what the statistic needs:

    * ``p = 0``: assignment labels are redrawn (fixed margins, or
      independent Bernoulli draws when ``prob`` is given), each unit
      keeping its kernel weight;
    * ``p > 0``: outcomes are permuted against the (score, assignment)
      pairs. Assignment is a function of the score, so re-randomizing
      labels while holding scores fixed would compare a boundary
      extrapolation with interior fits -- see the Notes of
      :func:`rdrandinf`.
    """
    n = y.shape[0]
    w = _lr.kernel_weights(xc, z, bw[0], bw[1], kernel)
    g, _ = _lr.linear_functional(xc, z, w, p, evals[0], evals[1])
    se = _lr.hc_se(y, xc, z, w, p, evals[0], evals[1], vce)
    labels_scheme = p == 0
    y0 = y - nulltau * shift

    obs = {"diffmeans": float(g @ y0)}
    ranks = sp_stats.rankdata(y0)
    if "ranksum" in stat_names:
        obs["ranksum"] = float(_lr.ranksum_from_labels(ranks, z[None, :])[0])
    if "ksmirnov" in stat_names:
        obs["ksmirnov"] = float(_lr.ks_from_labels(y0, z[None, :])[0])

    exceed = {s: 0 for s in stat_names}
    total = 0
    kept: List[np.ndarray] = []
    grid = ci_grid
    grid_exceed = np.zeros(0 if grid is None else grid.shape[0])
    tau_hat = float(g @ y)
    g_shift = float(g @ shift)
    for size in _lr.chunks(n_perms, n):
        if labels_scheme:
            if prob is None:
                lab = z[_lr.permutation_indices(rng, size, n)].astype(float)
            else:
                lab = _lr.bernoulli_labels(rng, size, prob).astype(float)
            if lab.shape[0] == 0:
                continue
            # Kernel weights belong to the unit (they depend on its
            # distance from the cutoff), so they travel with its outcome.
            w1 = lab @ w
            w0 = w.sum() - w1
            ok = (w1 > 0) & (w0 > 0)
            if not ok.all():
                lab, w1, w0 = lab[ok], w1[ok], w0[ok]
            wy, ws = w * y, w * shift
            a = (lab @ wy) / w1 - ((1.0 - lab) @ wy) / w0
            b = (lab @ ws) / w1 - ((1.0 - lab) @ ws) / w0
            if "ranksum" in stat_names:
                rs = _lr.ranksum_from_labels(ranks, lab)
                exceed["ranksum"] += int(
                    np.sum(np.abs(rs) >= abs(obs["ranksum"]) - 1e-14)
                )
            if "ksmirnov" in stat_names:
                ks = _lr.ks_from_labels(y0, lab)
                exceed["ksmirnov"] += int(np.sum(ks >= obs["ksmirnov"] - 1e-14))
            total += lab.shape[0]
        else:
            idx = _lr.permutation_indices(rng, size, n)
            a = y[idx] @ g
            b = shift[idx] @ g
            total += size
        if keep_draws:
            kept.append(np.asarray(a, dtype=float))
        if "diffmeans" in stat_names:
            exceed["diffmeans"] += int(
                np.sum(np.abs(a - nulltau * b) >= abs(obs["diffmeans"]) - 1e-14)
            )
        if grid is not None:
            # H0: tau = t0. Observed g @ (y - t0 * shift); each draw is
            # a - t0 * b, so the whole grid costs one outer product.
            obs_grid = np.abs(tau_hat - grid * g_shift)
            draws = np.abs(a[:, None] - b[:, None] * grid[None, :])
            grid_exceed += np.sum(draws >= obs_grid[None, :] - 1e-14, axis=0)

    if total == 0:
        raise DataInsufficient(
            "No valid randomization draw: every Bernoulli draw left one arm "
            "empty. Check the bernoulli= probabilities."
        )
    out: Dict[str, Any] = {
        "observed": obs,
        "se": se,
        "pvalue": {s: exceed[s] / total for s in stat_names},
        "n_draws": total,
        "ci": None,
        "ci_truncated": False,
        "draws": np.concatenate(kept) if kept else None,
    }
    if grid is not None:
        keep = grid[(grid_exceed / total) > alpha]
        if keep.size:
            out["ci"] = (float(keep.min()), float(keep.max()))
            out["ci_truncated"] = bool(
                keep.min() <= grid.min() or keep.max() >= grid.max()
            )
    return out


def _asymptotic_for(
    stat: str, y0: np.ndarray, z: np.ndarray, observed: float, se: float
) -> float:
    """Large-sample p-value for one statistic on the null-adjusted outcome."""
    if stat == "diffmeans":
        if not np.isfinite(se):
            return float("nan")
        if se < 1e-14:
            return 0.0 if abs(observed) > 1e-14 else 1.0
        return float(2 * sp_stats.norm.cdf(-abs(observed / se)))
    return _asymptotic_pvalue(y0, z, stat)[1]


# ======================================================================
# Public API
# ======================================================================


@accepts_aliases(covariates="covs")
def rdrandinf(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    wl: Optional[float] = None,
    wr: Optional[float] = None,
    statistic: str = "diffmeans",
    p: int = 0,
    covs: Optional[List[str]] = None,
    kernel: str = "uniform",
    n_perms: int = 1000,
    fuzzy: Optional[str] = None,
    alpha: float = 0.05,
    seed: int = 42,
    *,
    fuzzy_stat: str = "itt",
    nulltau: float = 0.0,
    bernoulli: Optional[Union[str, float]] = None,
    ci: Union[bool, Sequence[float], None] = None,
    d: Optional[float] = None,
    dscale: float = 0.5,
    evall: Optional[float] = None,
    evalr: Optional[float] = None,
    interfci: Optional[float] = None,
    vce: str = "hc3",
) -> CausalResult:
    """
    Randomization inference for regression discontinuity designs.

    Under the local randomization assumption, units within a small window
    around the cutoff are treated as if randomly assigned. Inference is
    based on Fisher's randomization test, with a large-sample approximation
    reported alongside.

    .. versionchanged:: 1.39.0
       ⚠️ **Results changed in three cases. Re-run anything that used
       them.** See MIGRATION.md.

       * ``wl`` / ``wr`` are now the window's endpoints on the scale of the
         running variable, as in ``rdlocrand``. They used to be offsets
         from the cutoff, which is the same thing only when ``c = 0``.
       * ``p > 0`` used to residualize the outcome on a polynomial pooled
         across both sides, which absorbs the effect itself (the score is
         collinear with treatment inside the window). It now fits a
         polynomial on each side and reports the difference in intercepts.
       * ``fuzzy=`` used to report a permutation p-value for the Wald
         ratio, whose randomization distribution has no finite scale when
         the permuted first stage passes through zero. The default is now
         the Anderson-Rubin / intention-to-treat randomization test.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff value.
    wl, wr : float
        Left and right endpoints of the window, on the scale of the running
        variable (so ``wl <= c <= wr``). Use :func:`rdwinselect` to choose
        a window from covariate balance.
    statistic : str, default 'diffmeans'
        Test statistic: 'diffmeans', 'ksmirnov', 'ranksum', or 'all'.
        ``'ttest'`` is accepted as an alias for ``'diffmeans'``, matching
        rdlocrand, where both names select the same statistic. The rank
        and Kolmogorov-Smirnov statistics need ``p=0`` and the uniform
        kernel.
    p : int, default 0
        Order of the polynomial in the score fitted on each side. The
        statistic is then the difference in fitted values at the cutoff
        (or at ``evall`` / ``evalr``).
    covs : list of str, optional
        Covariate names to partial out of the outcome before testing.
    kernel : str, default 'uniform'
        'uniform', 'triangular' or 'epanechnikov' weights. The bandwidth on
        each side is the distance from the cutoff to that side's window
        edge.
    n_perms : int, default 1000
        Number of randomization draws.
    fuzzy : str, optional
        Column holding the treatment actually received (fuzzy RD). The
        estimate is then the Wald ratio inside the window, with the
        heteroskedasticity-robust 2SLS standard error.
    alpha : float, default 0.05
        Significance level for the confidence interval and the power
        calculation.
    seed : int, default 42
        Random seed for reproducibility.
    fuzzy_stat : {'itt', 'tsls'}, default 'itt'
        Which test backs ``pvalue`` in a fuzzy design. ``'itt'`` (alias
        ``'ar'``) is the randomization test on the reduced form, valid in
        finite samples: the effect on compliers is zero exactly when the
        intention-to-treat effect is. ``'tsls'`` is the large-sample z-test
        on the Wald ratio; no randomization p-value is reported for it.
    nulltau : float, default 0
        Treatment effect under the null hypothesis.
    bernoulli : str or float, optional
        Assignment probabilities when the mechanism is independent
        Bernoulli trials rather than a fixed number of treated units: a
        column name, or one probability for every unit. Needs ``p=0`` and
        the uniform kernel.
    ci : bool or sequence of float, optional
        Confidence interval by inverting the randomization test of the
        difference in means. ``None`` / ``True`` uses a grid of 201 points
        over the estimate plus or minus five standard errors; a sequence is
        used as the grid of effects to test; ``False`` skips it. The
        interval is the range of grid values not rejected at ``alpha``, so
        its resolution is the grid's.
    d : float, optional
        Effect size for the large-sample power calculation. Defaults to
        ``dscale`` times the control group's outcome standard deviation.
    dscale : float, default 0.5
    evall, evalr : float, optional
        Points on the running-variable scale at which the fitted
        polynomials are evaluated. Both default to the cutoff.
    interfci : float, optional
        Level (such as 0.05) of Rosenbaum's confidence interval under
        arbitrary interference between units, reported in
        ``model_info['interf_ci']``. It is the observed difference in
        means minus the upper and lower quantiles of its randomization
        distribution, and covers the difference between the statistic and
        what it would have been had no unit been treated. Needs ``p=0``
        and a sharp design.
    vce : {'hc3', 'hc2', 'hc1'}, default 'hc3'
        Variance of the large-sample test when ``p > 0``: the
        heteroskedasticity-robust standard error of the regression of the
        outcome on treatment, the polynomial and their interaction. The
        default is the one of ``rdlocrand`` 3.0. Ignored when ``p = 0``,
        where the standard error is the HC2 one (Welch's with a uniform
        kernel).

        .. versionchanged:: 1.39.0
           ⚠️ The large-sample p-value, standard error and power for
           ``p > 0`` used HC2, as ``rdlocrand`` up to 2.0 did. The
           reference moved to HC3 in 3.0 and so does the default here;
           ``vce='hc2'`` gives the earlier numbers.

    Returns
    -------
    CausalResult
        ``estimate`` is the difference in means (or in fitted values at the
        cutoff when ``p > 0``); ``pvalue`` is the randomization p-value of
        the first requested statistic; ``ci`` is the test-inversion
        interval. ``model_info`` carries the large-sample p-value, the
        power against ``d``, per-side means and standard deviations, and,
        for fuzzy designs, the intention-to-treat and first-stage
        estimates.

    Notes
    -----
    **What is randomized when** ``p > 0``. Treatment is a function of the
    score, so the randomization that local randomization posits is of
    *scores* to units. For a difference in means, weighted or not, that
    is the same as permuting treatment labels. With a polynomial it is
    not: the statistic extrapolates each side's fit to the cutoff, and its
    randomization distribution has to be built from the same
    extrapolation. For ``p > 0`` this function therefore permutes outcomes
    against (score, assignment) pairs. ``rdlocrand`` up to 2.0 produced
    much smaller p-values for ``p > 0``; on a design with no effect, 40
    observations and ``p = 1``, re-randomizing labels with the scores held
    fixed rejects a true null 37% of the time at the 5% level, against 4.7%
    here (``tests/reference_parity/test_rdlocrand_extensions_parity.py``).
    The observed statistic and the large-sample p-value agree with
    ``rdlocrand`` to 1e-13.

    The large-sample p-value uses the normal distribution and the
    variance ``vce`` selects (HC2, Welch's standard error, when
    ``p = 0``). With ``p > 0`` ``rdlocrand`` 3.0 reports this p-value only
    and no longer computes a randomization p-value; the one reported here
    is the outcome-permutation p-value described above.

    References
    ----------
    Cattaneo, M.D., Titiunik, R. and Vazquez-Bare, G. (2016).
    "Inference in Regression Discontinuity Designs under Local
    Randomization." *The Stata Journal*, 16(2), 331-367. [@cattaneo2016inference],

    [@cattaneo2015randomization], [@cattaneo2024extensions],
    [@rosenbaum2007interference]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.8 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> res = sp.rdrandinf(df, y="y", x="x", c=0.0, wl=-0.3,
    ...                    wr=0.3, n_perms=500, seed=42)
    >>> round(float(res.estimate), 3)
    0.926
    >>> res.pvalue is not None
    True
    """
    rng = np.random.default_rng(seed)
    vce_used = _resolve_vce(vce, int(p))
    wl_value, wr_value = _window_bounds(c, wl, wr)
    kernel = _lr.canonical_kernel(kernel)
    p = int(p)
    if p < 0:
        raise MethodIncompatibility("p must be a non-negative integer.")
    fuzzy_stat = {"ar": "itt"}.get(str(fuzzy_stat).lower(), str(fuzzy_stat).lower())
    if fuzzy_stat not in ("itt", "tsls"):
        raise MethodIncompatibility("fuzzy_stat must be 'itt' (alias 'ar') or 'tsls'.")

    # rdlocrand accepts 'ttest' and 'diffmeans' as names for the same
    # statistic, so an R script written with either must run unchanged.
    if statistic == "ttest":
        statistic = "diffmeans"
    if statistic != "all" and statistic not in _STAT_FUNCS:
        raise MethodIncompatibility(
            f"Unknown statistic '{statistic}'. "
            f"Choose from: 'diffmeans' (alias 'ttest'), 'ksmirnov', "
            f"'ranksum', 'all'."
        )
    stat_names = list(_STAT_FUNCS.keys()) if statistic == "all" else [statistic]
    adjusted = p > 0 or kernel != "uniform"
    if adjusted and stat_names != ["diffmeans"]:
        raise MethodIncompatibility(
            "statistic='ksmirnov' / 'ranksum' / 'all' compare the two "
            "outcome distributions as they are; they are not defined for "
            "p > 0 or a non-uniform kernel. Use statistic='diffmeans'."
        )
    if p > 0 and bernoulli is not None:
        raise MethodIncompatibility("bernoulli= needs p=0.")
    if adjusted and fuzzy is not None:
        raise MethodIncompatibility("fuzzy= needs p=0 and kernel='uniform'.")
    if interfci is not None:
        if not 0 < float(interfci) < 1:
            raise MethodIncompatibility("interfci= is a level in (0, 1), such as 0.05.")
        if p > 0 or fuzzy is not None:
            raise MethodIncompatibility("interfci= needs p=0 and a sharp design.")

    # --- subset to window ---
    bern_col = bernoulli if isinstance(bernoulli, str) else None
    mask = _select_window(data, x, wl_value, wr_value)
    df_w = data.loc[mask].copy()
    df_w, _n_missing = _drop_incomplete(
        df_w, [y, x, covs, fuzzy, bern_col], where="rdrandinf"
    )
    n_obs = len(df_w)
    if n_obs < 4:
        raise DataInsufficient(
            f"Only {n_obs} observations in window [{wl_value}, {wr_value}]. "
            "Widen the window or check your data."
        )

    yv = df_w[y].values.astype(float)
    xv = df_w[x].values.astype(float)
    z = (xv >= c).astype(int)
    n_right = int(z.sum())
    n_left = n_obs - n_right
    if n_left < 2 or n_right < 2:
        raise DataInsufficient(
            f"Need >= 2 observations on each side of the cutoff; "
            f"got {n_left} left and {n_right} right."
        )
    y_raw = yv.copy()
    if covs is not None and len(covs) > 0:
        yv = _polynomial_residuals(yv, xv - c, 0, df_w[covs].values.astype(float))

    prob = None
    if bernoulli is not None:
        prob = (
            df_w[bern_col].values.astype(float)
            if bern_col is not None
            else np.full(n_obs, float(bernoulli))  # type: ignore[arg-type]
        )
        if np.any((prob <= 0) | (prob >= 1)):
            raise MethodIncompatibility(
                "bernoulli= probabilities must lie strictly in (0, 1)."
            )

    d_actual = df_w[fuzzy].values.astype(float) if fuzzy is not None else None
    shift = d_actual if d_actual is not None else z.astype(float)
    bw = (c - wl_value, wr_value - c)
    evals = (
        0.0 if evall is None else float(evall) - c,
        0.0 if evalr is None else float(evalr) - c,
    )

    # --- point estimate and its large-sample standard error ---
    w_kern = _lr.kernel_weights(xv - c, z, bw[0], bw[1], kernel)
    g, _ = _lr.linear_functional(xv - c, z, w_kern, p, evals[0], evals[1])
    itt = float(g @ yv)
    se_itt = _lr.hc_se(yv, xv - c, z, w_kern, p, evals[0], evals[1], vce_used)

    # --- interval grid ---
    if ci is False:
        grid = None
    elif ci is None or ci is True:
        if d_actual is not None or not np.isfinite(se_itt) or se_itt < 1e-14:
            grid = None
        else:
            grid = np.linspace(itt - 5 * se_itt, itt + 5 * se_itt, 201)
    else:
        grid = np.asarray(list(ci), dtype=float)  # type: ignore[arg-type]
        if grid.ndim != 1 or grid.size < 2:
            raise MethodIncompatibility(
                "ci= must be a sequence of at least two effects."
            )
        if d_actual is not None and fuzzy_stat == "tsls":
            raise MethodIncompatibility(
                "ci= inverts the randomization test; with fuzzy_stat='tsls' "
                "the interval is the large-sample one."
            )

    test = _randomization_test(
        yv,
        xv - c,
        z,
        shift=shift,
        stat_names=stat_names,
        p=p,
        kernel=kernel,
        bw=bw,
        evals=evals,
        n_perms=n_perms,
        rng=rng,
        prob=prob,
        nulltau=float(nulltau),
        ci_grid=grid,
        alpha=alpha,
        keep_draws=interfci is not None,
        vce=vce_used,
    )
    y0 = yv - float(nulltau) * shift
    results = {}
    for sname in stat_names:
        results[sname] = {
            "observed_stat": test["observed"][sname],
            "pvalue_permutation": test["pvalue"][sname],
            "pvalue_asymptotic": _asymptotic_for(
                sname, y0, z, test["observed"][sname], test["se"]
            ),
        }
    primary = stat_names[0]
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)

    sd_left = float(y_raw[z == 0].std(ddof=1))
    power_d = float(d) if d is not None else float(dscale) * sd_left
    info: Dict[str, Any] = {
        "cutoff": c,
        "window": (wl_value, wr_value),
        "n_left": n_left,
        "n_right": n_right,
        "mean_left": float(y_raw[z == 0].mean()),
        "mean_right": float(y_raw[z == 1].mean()),
        "sd_left": sd_left,
        "sd_right": float(y_raw[z == 1].std(ddof=1)),
        "statistic": statistic,
        "polynomial_order": p,
        "kernel": kernel,
        "n_perms": n_perms,
        "n_draws": test["n_draws"],
        "randomization": "bernoulli" if prob is not None else "fixed margins",
        "nulltau": float(nulltau),
        "vce": vce_used if p > 0 else None,
        "covariates": covs,
        "results_by_stat": results,
        "pvalue_permutation": results[primary]["pvalue_permutation"],
        "pvalue_asymptotic": results[primary]["pvalue_asymptotic"],
        "power_d": power_d,
        "power": _lr.asymptotic_power(power_d, se_itt, alpha),
    }

    if test["ci"] is not None:
        ci_out, ci_method = test["ci"], "randomization test inversion"
        if test["ci_truncated"]:
            warnings.warn(
                "rdrandinf: the confidence interval reaches the edge of the "
                "grid of effects tested, so it is truncated. Pass a wider "
                "grid through ci=.",
                UserWarning,
                stacklevel=2,
            )
    elif grid is not None:
        # Every grid value was rejected: the grid missed the interval.
        warnings.warn(
            "rdrandinf: every effect on the ci= grid was rejected, so the "
            "grid does not cover the confidence interval. Reporting the "
            "large-sample interval instead.",
            UserWarning,
            stacklevel=2,
        )
        ci_out = (itt - z_crit * se_itt, itt + z_crit * se_itt)
        ci_method = "large-sample (grid missed the interval)"
    elif ci is False:
        ci_out, ci_method = (float("nan"), float("nan")), "not requested"
    else:
        ci_out = (itt - z_crit * se_itt, itt + z_crit * se_itt)
        ci_method = "large-sample"
    info["ci_method"] = ci_method
    info["ci_grid_truncated"] = bool(test["ci_truncated"])
    if interfci is not None and test["draws"] is not None:
        # Rosenbaum (2007): under interference the statistic minus its
        # value in the uniformity trial is what can be bounded, and the
        # uniformity-trial statistic has the randomization distribution.
        hi_q, lo_q = np.quantile(
            test["draws"], [1 - float(interfci) / 2, float(interfci) / 2]
        )
        info["interf_ci"] = (float(itt - hi_q), float(itt - lo_q))
        info["interf_ci_level"] = float(interfci)

    if d_actual is not None:
        tau_iv, se_iv, first_stage = _tsls_wald(yv, d_actual, z)
        tsls_p = float(2 * sp_stats.norm.sf(abs((tau_iv - float(nulltau)) / se_iv)))
        info.update(
            {
                "statistic": "itt" if fuzzy_stat == "itt" else "tsls",
                "fuzzy_stat": fuzzy_stat,
                "fuzzy_treatment": fuzzy,
                "first_stage": first_stage,
                "itt": itt,
                "itt_se": se_itt,
                "itt_ci": ci_out if test["ci"] is not None else None,
                "pvalue_tsls": tsls_p,
            }
        )
        if fuzzy_stat == "tsls":
            # A large-sample statistic: there is no randomization p-value
            # for it, and inventing one by permuting the ratio is what the
            # previous version did.
            info["pvalue_permutation"] = float("nan")
            info["pvalue_asymptotic"] = tsls_p
            pval_main = tsls_p
        else:
            pval_main = float(results[primary]["pvalue_permutation"])
        if fuzzy_stat == "tsls" or test["ci"] is None:
            ci_final = (tau_iv - z_crit * se_iv, tau_iv + z_crit * se_iv)
            info["ci_method"] = "large-sample (2SLS)"
        else:
            # Anderson-Rubin: the grid was tested as y - tau0 * D.
            ci_final = ci_out
        return CausalResult(
            method="RD Local Randomization (Fuzzy)",
            estimand="LATE",
            estimate=float(tau_iv),
            se=float(se_iv),
            pvalue=float(pval_main),
            ci=ci_final,
            alpha=alpha,
            n_obs=n_obs,
            model_info=info,
            _citation_key="rdlocrand",
        )

    detail = None
    if statistic == "all":
        detail = pd.DataFrame(
            [
                {
                    "statistic": sname,
                    "observed": res["observed_stat"],
                    "pvalue_perm": res["pvalue_permutation"],
                    "pvalue_asym": res["pvalue_asymptotic"],
                }
                for sname, res in results.items()
            ]
        )

    return CausalResult(
        method="RD Local Randomization",
        estimand="ATE (local)",
        estimate=float(itt),
        se=float(se_itt),
        pvalue=float(results[primary]["pvalue_permutation"]),
        ci=ci_out,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=info,
        _citation_key="rdlocrand",
    )


_WINSELECT_STATS = ("diffmeans", "ksmirnov", "ranksum", "hotelling")


@accepts_aliases(covariates="covs")
def rdwinselect(
    data: pd.DataFrame,
    x: str,
    c: float = 0,
    covs: Optional[List[str]] = None,
    wmin: Optional[float] = None,
    wstep: Optional[float] = None,
    nwindows: int = 10,
    statistic: str = "diffmeans",
    p: int = 0,
    seed: int = 42,
    alpha: float = 0.15,
    *,
    obsmin: int = 10,
    wobs: Optional[int] = None,
    wasymmetric: bool = False,
    approx: bool = False,
    n_perms: int = 1000,
    kernel: str = "uniform",
    dropmissing: bool = False,
    wmasspoints: bool = False,
    vce: str = "hc3",
) -> pd.DataFrame:
    """
    Data-driven window selection for local randomization RD.

    Builds a sequence of nested windows around the cutoff and, in each,
    tests the balance of every covariate and runs a binomial test on the
    number of observations either side. The recommended window is the
    largest one such that it and every window inside it have a minimum
    balance p-value of at least ``alpha``.

    .. versionchanged:: 1.39.0
       ⚠️ **The window sequence and the p-values changed. Re-run anything
       that used this function.** The default sequence used to start at a
       fraction of the running variable's range and step to its full
       extent, so on the U.S. Senate data it tested windows of 5, 15.6, ...,
       100 points where the procedure is about windows holding ten, twelve,
       fourteen observations a side. It now starts at the smallest window
       with ``obsmin`` observations on each side and adds ``wobs`` per
       side per step, as described by Cattaneo, Idrobo and Titiunik (2024).
       Without covariates the function used to test made-up quantile
       dummies of the score; it now reports the binomial test only. See
       MIGRATION.md.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff value.
    covs : list of str, optional
        Predetermined covariates to test balance for. Without them only
        the binomial test is reported and no window is recommended.
    wmin : float, optional
        Half-width of the smallest window. Defaults to the smallest
        symmetric window with ``obsmin`` observations on each side.
    wstep : float, optional
        Increment in half-width. If omitted, each window adds at least
        ``wobs`` observations on each side.
    nwindows : int, default 10
        Number of windows to evaluate. Fewer come back if the data run out.
    statistic : str, default 'diffmeans'
        Balance statistic: 'diffmeans' (alias 'ttest'), 'ksmirnov',
        'ranksum', or 'hotelling'. The first three test each covariate and
        report the smallest p-value; 'hotelling' is one joint test of all
        covariates (Hotelling's T-squared), so ``variable`` is empty.
    p : int, default 0
        Polynomial order for the covariate adjustment model.
    seed : int, default 42
        Random seed.
    alpha : float, default 0.15
        Smallest acceptable minimum p-value (``level`` in rdlocrand). It is
        deliberately larger than 0.05: the concern is failing to detect
        imbalance, not falsely detecting it.
    obsmin : int, default 10
        Minimum observations on each side in the smallest window.
    wobs : int, optional
        Observations added on each side per step (5 if neither ``wobs``
        nor ``wstep`` is given).
    wasymmetric : bool, default False
        Let the two sides of the window grow separately.
    approx : bool, default False
        Use the large-sample p-values instead of randomization inference.
        Deterministic and much faster.
    n_perms : int, default 1000
        Randomization draws per covariate and window.
    kernel : str, default 'uniform'
    dropmissing : bool, default False
        Drop rows with a missing covariate before the windows are built.
    vce : {'hc3', 'hc2', 'hc1'}, default 'hc3'
        Variance of the large-sample balance tests when ``p > 0``; see
        :func:`rdrandinf`. Ignored when ``p = 0``.
        By default the windows are built on every row with an observed
        score and incomplete rows are dropped inside each window.
    wmasspoints : bool, default False
        For a discrete running variable: window ``k`` runs from the
        ``k``-th mass point below the cutoff to the ``k``-th at or above
        it, so each step adds one support point on each side. Cannot be
        combined with ``obsmin``, ``wmin``, ``wobs`` or ``wstep``.

    Returns
    -------
    pd.DataFrame
        One row per window: ``window_left``, ``window_right`` (endpoints on
        the running-variable scale), ``n_left``, ``n_right``, ``p_value``
        (smallest balance p-value across covariates), ``variable`` (the
        covariate attaining it), ``binom_pvalue`` and ``balanced``.
        ``attrs['recommended_window']`` holds ``(left, right)`` or
        ``None``; ``attrs['recommended_n']`` the counts inside it.

    Notes
    -----
    The window sequences, counts, binomial p-values and large-sample
    balance p-values agree with ``rdlocrand`` 1.0 and 3.0 to 1e-9,
    mass-point windows included
    (``tests/reference_parity/test_rdlocrand_v1_parity.py``). Releases 1.1
    and 2.0 start the default sequence one observation short below the
    cutoff (9 with ``obsmin = 10`` on the Senate data) and shift the left
    edge of mass-point windows by one support point; the help page and
    Cattaneo, Idrobo and Titiunik (2024, Snippet 2.5) describe what 1.0
    and 3.0 do. With ``wmin=`` and ``wstep=`` all releases agree.

    References
    ----------
    [@cattaneo2016inference], [@cattaneo2015randomization],
    [@cattaneo2024extensions]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.uniform(-1, 1, n)
    >>> z1 = 0.2 * x + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"x": x, "z1": z1})
    >>> tab = sp.rdwinselect(df, x="x", c=0.0, covs=["z1"],
    ...                      nwindows=5, approx=True)
    >>> tab.shape
    (5, 8)
    >>> list(tab.columns)[:5]
    ['window_left', 'window_right', 'n_left', 'n_right', 'p_value']
    >>> bool(tab["n_left"].iloc[0] >= 10 and tab["n_right"].iloc[0] >= 10)
    True
    """
    rng = np.random.default_rng(seed)
    vce_used = _resolve_vce(vce, int(p))
    if statistic == "ttest":
        statistic = "diffmeans"
    if statistic not in _WINSELECT_STATS:
        raise MethodIncompatibility(
            f"Unknown statistic '{statistic}'. Choose from: 'diffmeans' "
            "(alias 'ttest'), 'ksmirnov', 'ranksum', 'hotelling'."
        )
    kernel = _lr.canonical_kernel(kernel)
    covs = list(covs) if covs else []

    # The windows are a property of the score, so by default only rows with
    # a missing score are dropped up front; rows with a missing covariate
    # are dropped inside each window, and the counts reported are of the
    # rows the balance tests actually used.
    up_front = [x, covs] if dropmissing else [x]
    data, _n_missing = _drop_incomplete(data, up_front, where="rdwinselect")
    xv = data[x].values.astype(float)
    if (xv < c).sum() == 0 or (xv >= c).sum() == 0:
        raise DataInsufficient("Need observations on both sides of the cutoff.")
    complete = np.ones(len(data), dtype=bool)
    if covs:
        complete = np.isfinite(data[covs].to_numpy(dtype=float)).all(axis=1)
    cov_values = {cv: data[cv].to_numpy(dtype=float) for cv in covs}

    if wmasspoints:
        if wmin is not None or wobs is not None or wstep is not None or obsmin != 10:
            raise MethodIncompatibility(
                "wmasspoints=True takes its windows from the support points "
                "of the score; obsmin=, wmin=, wobs= and wstep= do not apply."
            )
        windows = _lr.mass_point_windows(xv, c, nwindows=int(nwindows))
    else:
        windows = _lr.window_sequence(
            xv,
            c,
            nwindows=int(nwindows),
            obsmin=int(obsmin),
            wmin=wmin,
            wobs=wobs,
            wstep=wstep,
            wasymmetric=wasymmetric,
        )

    rows = []
    for half_l, half_r in windows:
        lo, hi = c - half_l, c + half_r
        mask = (xv >= lo) & (xv <= hi) & complete
        xw = xv[mask]
        z = (xw >= c).astype(int)
        n_right = int(z.sum())
        n_left = int(z.shape[0] - n_right)
        row: Dict[str, Any] = {
            "window_left": lo,
            "window_right": hi,
            "n_left": n_left,
            "n_right": n_right,
            "p_value": np.nan,
            "variable": None,
            "binom_pvalue": (
                float(sp_stats.binomtest(n_left, n_left + n_right, 0.5).pvalue)
                if n_left + n_right > 0
                else np.nan
            ),
            "balanced": False,
        }
        if covs and statistic == "hotelling" and n_left >= 2 and n_right >= 2:
            if int(p) > 0 or kernel != "uniform":
                raise MethodIncompatibility(
                    "statistic='hotelling' needs p=0 and kernel='uniform'."
                )
            Z = np.column_stack([cov_values[cv][mask] for cv in covs])
            keep = np.ptp(Z, axis=0) > 1e-14
            Z = Z[:, keep]
            k_cov = Z.shape[1]
            if k_cov and Z.shape[0] - k_cov - 1 > 0:
                t2 = float(_lr.hotelling_t2(Z, z[None, :])[0])
                if approx:
                    pval = _lr.hotelling_pvalue_f(t2, Z.shape[0], k_cov)
                elif np.isfinite(t2):
                    hits = total = 0
                    for size in _lr.chunks(n_perms, Z.shape[0]):
                        lab = z[_lr.permutation_indices(rng, size, Z.shape[0])]
                        draws = _lr.hotelling_t2(Z, lab)
                        hits += int(np.sum(draws >= t2 - 1e-12))
                        total += size
                    pval = hits / total
                else:
                    pval = float("nan")
                if np.isfinite(pval):
                    row["p_value"] = float(pval)
                    row["balanced"] = bool(pval >= alpha)
        elif covs and n_left >= 2 and n_right >= 2:
            best, best_name = np.inf, None
            for cv in covs:
                vals = cov_values[cv][mask]
                pval = _balance_pvalue(
                    vals,
                    xw - c,
                    z,
                    statistic=statistic,
                    p=int(p),
                    kernel=kernel,
                    bw=(half_l, half_r),
                    approx=approx,
                    n_perms=n_perms,
                    rng=rng,
                    vce=vce_used,
                )
                if np.isfinite(pval) and pval < best:
                    best, best_name = pval, cv
            if best_name is not None:
                row["p_value"] = float(best)
                row["variable"] = best_name
                row["balanced"] = bool(best >= alpha)
        rows.append(row)

    result = pd.DataFrame(rows)
    recommended = None
    recommended_n = None
    if covs:
        passed = result["balanced"].to_numpy(dtype=bool)
        n_ok = int(np.argmin(passed)) if not passed.all() else len(passed)
        if n_ok == 0:
            warnings.warn(
                "rdwinselect: the smallest window already fails the balance "
                f"test (minimum p-value {result['p_value'].iloc[0]:.3f} < "
                f"{alpha}), so no window is recommended. Try a smaller "
                "obsmin= / wmin=, or reconsider local randomization here.",
                UserWarning,
                stacklevel=2,
            )
        else:
            best_row = result.iloc[n_ok - 1]
            recommended = (
                float(best_row["window_left"]),
                float(best_row["window_right"]),
            )
            recommended_n = (int(best_row["n_left"]), int(best_row["n_right"]))
            if n_ok == len(passed):
                warnings.warn(
                    "rdwinselect: every window tested passes the balance "
                    "test, so the recommended window is just the largest "
                    "one tried. Increase nwindows= to find where balance "
                    "breaks down.",
                    UserWarning,
                    stacklevel=2,
                )
    result.attrs["recommended_window"] = recommended
    result.attrs["recommended_n"] = recommended_n
    result.attrs["cutoff"] = c
    result.attrs["level"] = alpha
    result.attrs["approx"] = bool(approx)
    return result


def _balance_pvalue(
    vals: np.ndarray,
    xc: np.ndarray,
    z: np.ndarray,
    *,
    statistic: str,
    p: int,
    kernel: str,
    bw: Tuple[float, float],
    approx: bool,
    n_perms: int,
    rng: np.random.Generator,
    vce: str = "hc2",
) -> float:
    """Balance p-value for one covariate in one window; NaN if constant."""
    if np.ptp(vals) < 1e-14:
        return float("nan")
    adjusted = p > 0 or kernel != "uniform"
    if adjusted and statistic != "diffmeans":
        raise MethodIncompatibility(
            "statistic='ksmirnov' / 'ranksum' need p=0 and kernel='uniform'."
        )
    try:
        if approx:
            w = _lr.kernel_weights(xc, z, bw[0], bw[1], kernel)
            if statistic != "diffmeans":
                return _asymptotic_pvalue(vals, z, statistic)[1]
            g, _ = _lr.linear_functional(xc, z, w, p)
            se = _lr.hc_se(vals, xc, z, w, p, 0.0, 0.0, vce)
            return _asymptotic_for("diffmeans", vals, z, float(g @ vals), se)
        test = _randomization_test(
            vals,
            xc,
            z,
            shift=z.astype(float),
            stat_names=[statistic],
            p=p,
            kernel=kernel,
            bw=bw,
            evals=(0.0, 0.0),
            n_perms=n_perms,
            rng=rng,
            prob=None,
            nulltau=0.0,
            ci_grid=None,
            alpha=0.05,
            vce=vce,
        )
    except ValueError:
        # The polynomial is not identified in this window for this
        # covariate; it cannot speak to balance here.
        return float("nan")
    return float(test["pvalue"][statistic])


def rdsensitivity(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    wlist: Optional[List[float]] = None,
    nwindows: int = 20,
    statistic: str = "diffmeans",
    p: int = 0,
    n_perms: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
    plot: bool = False,
    vce: str = "hc3",
) -> pd.DataFrame:
    """
    Sensitivity of RD estimates across different window widths.

    For each window, runs ``rdrandinf`` and records the estimate, standard
    error, and p-value. Optionally produces a plot if matplotlib is
    available.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff value.
    wlist : list of float, optional
        Symmetric half-window widths to evaluate. If None, an
        evenly-spaced grid is generated automatically.
    nwindows : int, default 20
        Number of windows when ``wlist`` is None.
    statistic : str, default 'diffmeans'
        Test statistic for inference.
    p : int, default 0
        Polynomial order for adjustment.
    n_perms : int, default 500
        Number of permutations per window.
    seed : int, default 42
        Random seed.
    alpha : float, default 0.05
        Significance level.
    vce : {'hc3', 'hc2', 'hc1'}, default 'hc3'
        Variance of the large-sample test when ``p > 0``; see
        :func:`rdrandinf`.

    Returns
    -------
    pd.DataFrame
        Columns: window, estimate, se, pvalue, ci_lower, ci_upper,
        significant.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> x = rng.uniform(-1, 1, 500)
    >>> y = 0.5 * (x >= 0) + 0.8 * x + rng.normal(0, 0.3, 500)
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> sens = sp.rdsensitivity(
    ...     df, y="y", x="x", wlist=[0.25, 0.5, 0.75],
    ...     n_perms=200, seed=42,
    ... )
    >>> sens["estimate"].round(3).tolist()
    [0.66, 0.834, 1.078]
    >>> bool(sens["significant"].all())
    True
    """
    # Cleaned once here rather than inside each per-window rdrandinf call:
    # the default window grid is derived from the running variable's own
    # spacing, so it has to be built on the sample that will actually be
    # estimated, and one warning is more useful than `nwindows` of them.
    data, _n_missing = _drop_incomplete(data, [y, x], where="rdsensitivity")
    xv = data[x].values.astype(float)

    if wlist is None:
        x_left = xv[xv < c]
        x_right = xv[xv >= c]
        max_left = c - x_left.min() if len(x_left) > 0 else 1.0
        max_right = x_right.max() - c if len(x_right) > 0 else 1.0
        max_w = min(max_left, max_right)
        # Start from a small window; ensure enough obs
        sorted_gaps = np.sort(np.abs(xv - c))
        # Need at least 4 obs, so start from the 4th closest
        min_w = sorted_gaps[min(3, len(sorted_gaps) - 1)] * 1.1
        min_w = max(min_w, max_w / (nwindows * 2))
        wlist = np.linspace(min_w, max_w * 0.95, nwindows).tolist()

    rows = []
    for w in wlist:
        wl = c - w
        wr = c + w
        mask = (xv >= wl) & (xv <= wr)
        z_in = (xv[mask] >= c).astype(int)
        n_left = int((z_in == 0).sum())
        n_right = int((z_in == 1).sum())

        if n_left < 2 or n_right < 2:
            rows.append(
                {
                    "window": w,
                    "estimate": np.nan,
                    "se": np.nan,
                    "pvalue": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "significant": False,
                }
            )
            continue  # pragma: no cover

        try:
            res = rdrandinf(
                data,
                y,
                x,
                c=c,
                wl=wl,
                wr=wr,
                statistic=statistic,
                p=p,
                n_perms=n_perms,
                alpha=alpha,
                seed=seed,
                vce=vce,
            )
            rows.append(
                {
                    "window": w,
                    "estimate": res.estimate,
                    "se": res.se,
                    "pvalue": res.pvalue,
                    "ci_lower": res.ci[0],
                    "ci_upper": res.ci[1],
                    "significant": res.pvalue <= alpha,
                }
            )
        except (ValueError, RuntimeError):  # pragma: no cover
            rows.append(
                {
                    "window": w,
                    "estimate": np.nan,
                    "se": np.nan,
                    "pvalue": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "significant": False,
                }
            )

    result = pd.DataFrame(rows)

    # Figure built only on request, and never shown.
    #
    # This block used to run unconditionally and end in ``plt.show()``.
    # Under an interactive backend -- ``macosx`` is the default on the
    # platform this is developed on -- ``show()`` blocks until a human
    # closes the window, so `sp.rdsensitivity(...)` never returned in a
    # script, a test run, a CI job or an agent session. That is a hard
    # hang in an estimation function, in a package whose stated purpose
    # is to be callable by agents.
    #
    # Displaying is the caller's decision in every case: the returned
    # figure is attached to ``result.attrs["figure"]`` so a notebook user
    # can render it and everyone else can ignore it.
    try:
        import matplotlib.pyplot as plt

        valid = result.dropna(subset=["estimate"]) if plot else result.iloc[:0]
        if len(valid) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Panel 1: Estimates with CI
            ax = axes[0]
            ax.plot(valid["window"], valid["estimate"], "o-", color="#2c3e50")
            ax.fill_between(
                valid["window"],
                valid["ci_lower"],
                valid["ci_upper"],
                alpha=0.2,
                color="#3498db",
            )
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Window half-width")
            ax.set_ylabel("Treatment effect estimate")
            ax.set_title("Sensitivity: Estimates across windows")

            # Panel 2: p-values
            ax = axes[1]
            ax.plot(valid["window"], valid["pvalue"], "o-", color="#e74c3c")
            ax.axhline(
                alpha,
                color="grey",
                linestyle="--",
                linewidth=0.8,
                label=f"alpha = {alpha}",
            )
            ax.set_xlabel("Window half-width")
            ax.set_ylabel("Permutation p-value")
            ax.set_title("Sensitivity: P-values across windows")
            ax.legend()

            plt.tight_layout()
            result.attrs["figure"] = fig
    except ImportError:  # pragma: no cover
        pass  # pragma: no cover

    return result


def _rdrbounds_rows(
    yv: np.ndarray,
    z: np.ndarray,
    gamma_list: Any,
    U: np.ndarray,
    statistic: str,
) -> List[Dict[str, float]]:
    """Rosenbaum-bound p-values given a (reps, n) matrix of uniforms ``U``.

    Every p-value reuses the same ``U`` (R ``rdrbounds`` re-seeds before
    each one), and unit ``order[j]`` receives column ``j``. Separated from
    the draws so the reference-parity test can feed R's own ``runif``
    stream and reproduce ``rdlocrand::rdrbounds`` exactly.
    """
    n = len(yv)
    ranks = sp_stats.rankdata(yv)
    rank_var = float(np.var(ranks, ddof=1))

    def _stats(D: np.ndarray) -> np.ndarray:
        """Statistic for each row of a (reps, n) assignment matrix."""
        n1 = D.sum(axis=1)
        n0 = n - n1
        with np.errstate(divide="ignore", invalid="ignore"):
            if statistic == "ranksum":
                T = (1 - D) @ ranks
                out = (T - n0 * (n + 1) / 2) / np.sqrt(n0 * n1 * rank_var / n)
            else:
                out = (D @ yv) / n1 - ((1 - D) @ yv) / n0
        out[(n1 == 0) | (n0 == 0)] = np.nan
        return out

    obs = float(_stats(z[None, :])[0])

    def _pvalue(prob: np.ndarray, order: np.ndarray) -> float:
        # Draws are attached to units in `order`, as R attaches runif(n) to
        # the rows of its sorted data.
        D = np.empty_like(U)
        D[:, order] = (U <= prob[None, :]).astype(float)
        st = _stats(D)
        ok = np.isfinite(st)
        return float(np.mean(np.abs(st[ok]) >= abs(obs) - 1e-14))

    dec = np.argsort(-yv, kind="mergesort")
    inc = np.argsort(yv, kind="mergesort")
    rows = []
    for gamma in gamma_list:
        if abs(gamma - 1.0) < 1e-14:
            pv = _pvalue(np.full(n, z.mean()), inc)
            rows.append({"gamma": gamma, "pvalue_upper": pv, "pvalue_lower": pv})
            continue
        phigh, plow = gamma / (1 + gamma), 1 / (1 + gamma)
        ub, lb = [], []
        pos = np.arange(n)
        for u in range(1, n + 1):
            # R: uplus on the decreasing sort, uminus on the increasing one;
            # both give the high probability to the u largest outcomes.
            ub.append(_pvalue(np.where(pos < u, phigh, plow), dec))
            lb.append(_pvalue(np.where(pos >= n - u, phigh, plow), inc))
        rows.append({"gamma": gamma, "pvalue_upper": max(ub), "pvalue_lower": min(lb)})
    return rows


def rdrbounds(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    wl: Optional[float] = None,
    wr: Optional[float] = None,
    gamma_list: Optional[List[float]] = None,
    statistic: str = "ranksum",
    n_perms: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Rosenbaum sensitivity bounds for RD under local randomization.

    Assesses how much hidden bias (departure from random assignment)
    would be needed to explain away the estimated treatment effect, as
    R ``rdlocrand::rdrbounds`` does. Under Rosenbaum's model with odds
    ratio ``Gamma`` each unit in the window is treated independently with
    probability ``Gamma/(1+Gamma)`` or ``1/(1+Gamma)``. As in R, the bounds
    are taken over the monotone patterns in which the ``u`` units with the
    largest outcomes get the high probability, ``u = 1..n``: the upper
    bound is the largest of those p-values and the lower bound the
    smallest. ``Gamma = 1`` is randomization with the treated share as the
    common probability.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable name.
    x : str
        Running variable name.
    c : float, default 0
        RD cutoff value.
    wl, wr : float
        Window endpoints on the running-variable scale (``wl <= c <= wr``).
    gamma_list : list of float, optional
        Odds ratios. Defaults to ``[1, 1.5, 2, 2.5, 3, 4, 5]``.
    statistic : {'ranksum', 'diffmeans'}, default 'ranksum'
        ``'ranksum'`` is R's standardised rank sum (control-rank sum minus
        its mean over ``sqrt(n0 n1 var(ranks) / n)``); ``'diffmeans'`` the
        difference in means.
    n_perms : int, default 1000
        Bernoulli assignment draws per p-value. The same draws are reused
        across thresholds and ``Gamma`` values (common random numbers, as R
        reseeds before each p-value).
    seed : int, default 42
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: gamma, pvalue_upper, pvalue_lower.

    Notes
    -----
    Randomisation p-values: agreement with R is within Monte-Carlo error
    only. Through 1.28.0 this function split units at the *median*
    outcome only -- one threshold instead of the extremum over all of them
    -- drew a fixed number of treated units rather than Bernoulli
    assignments, and gave the high probability to the *smallest* outcomes
    for the lower bound; R's ``uminus`` on its increasing sort selects the
    largest outcomes, the same patterns as the upper bound, which is what
    this port follows (with R's draw-to-unit attachment for each bound).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.8 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> tab = sp.rdrbounds(df, y="y", x="x", c=0.0, wl=-0.3,
    ...                    wr=0.3, gamma_list=[1.0, 1.5, 2.0],
    ...                    n_perms=500, seed=42)
    >>> tab.shape
    (3, 3)
    >>> list(tab.columns)
    ['gamma', 'pvalue_upper', 'pvalue_lower']
    >>> bool((tab["pvalue_upper"] >= tab["pvalue_lower"]).all())
    True
    """
    if wl is None or wr is None:
        raise MethodIncompatibility(
            "Window bounds wl and wr must be specified. "
            "Use rdwinselect() to choose a data-driven window."
        )
    if statistic not in ("ranksum", "diffmeans"):
        raise MethodIncompatibility("statistic must be 'ranksum' or 'diffmeans'")
    if gamma_list is None:
        gamma_list = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    wl_value, wr_value = _window_bounds(c, wl, wr)
    mask = _select_window(data, x, wl_value, wr_value)
    df_w = data.loc[mask].copy()
    df_w, _n_missing = _drop_incomplete(df_w, [y, x], where="rdrbounds")
    yv = df_w[y].to_numpy(dtype=float)
    z = (df_w[x].to_numpy(dtype=float) >= c).astype(float)
    n = len(yv)
    if n < 4 or z.sum() < 2 or (n - z.sum()) < 2:
        raise DataInsufficient("Need >= 2 observations on each side of the cutoff.")

    for gamma in gamma_list:
        if gamma < 1.0:
            raise MethodIncompatibility("gamma must be >= 1.")
    rng = np.random.default_rng(seed)
    U = rng.random((int(n_perms), n))
    rows = _rdrbounds_rows(yv, z, gamma_list, U, statistic)
    return pd.DataFrame(rows)
