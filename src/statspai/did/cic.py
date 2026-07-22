"""
Changes-in-Changes (CIC) estimator — Athey & Imbens (2006).

CIC relaxes the standard DID parallel-trends assumption to the weaker
"rank invariance" condition.  Instead of assuming additive group effects,
it uses the full outcome distributions to construct the counterfactual.

Algorithm (continuous case)
---------------------------
1. Estimate empirical CDFs for each (group, time) cell:
   F_{00}, F_{01}, F_{10}, F_{11}.
2. Counterfactual CDF for treated-post absent treatment:
   F_{Y^N,11}(y) = F_{10}( F_{00}^{-1}( F_{01}(y) ) )
3. ATT  = mean(Y_{11}) - integral of F_{Y^N,11}^{-1}
4. QTE(τ) = F_{11}^{-1}(τ) - F_{Y^N,11}^{-1}(τ)

.. versionchanged:: 1.20.x
   **⚠️ Correctness fix to the step-2 counterfactual.** Earlier releases
   composed the empirical CDFs with the control-post (``y01``) and
   treated-pre (``y10``) cells transposed relative to A&I eq. 9, and
   evaluated linearly-interpolated CDF / quantile functions on a finite τ
   grid rather than the step-function ECDF and its generalized inverse. The
   ATT converged to a value ~0.5% away from the reference (2.8% in the
   covariate case). It now computes ``k(y) = F_01^{-1}(F_00(y))`` on the
   step ECDF and reproduces Kranker's Stata ``cic`` (a direct port of the
   Athey-Imbens Matlab) to the printed digits. See CHANGELOG / MIGRATION.

Covariates
----------
``cic(..., covariates=[...])`` implements the parametric covariate approach
of Athey & Imbens (2006, p. 466) — apply CIC to the residuals of an OLS
regression of the outcome on the covariates *and* the design dummies, with
the dummy effects added back in — and bootstraps both steps jointly.

Reference
---------
Athey, S. & Imbens, G. W. (2006).
Identification and Inference in Nonlinear Difference-in-Differences Models.
*Econometrica*, 74(2), 431-497.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult

_SUPPORTED_FIRST_STAGE = ("feols",)

# ── Helpers ───────────────────────────────────────────────────────────


def _ecdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empirical CDF of *x* evaluated at *grid* (right-continuous step).

    ``F(q) = #{x_i <= q} / n`` — the genuine empirical CDF, *not* the linear
    interpolation used previously. Athey & Imbens (2006) define the CIC
    estimator on the step-function ECDF; interpolating it smooths the
    counterfactual and pulls the estimate off the reference implementation
    (Kranker's Stata ``cic``, a direct port of A&I's Matlab) by ~0.5%.
    """
    xs = np.sort(x)
    return np.asarray(np.searchsorted(xs, grid, side="right") / len(xs), dtype=float)


def _quantile_func(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Generalized inverse CDF: ``inf{ y : F(y) >= p }``.

    The left-continuous generalized inverse of the step ECDF, as in A&I
    (2006). For ``p`` in ``(0, 1]`` this is the ``ceil(p * n)``-th order
    statistic; ``p <= 0`` maps to the minimum. Replaces the previous linear
    interpolation between order statistics, which is a different (smoothed)
    quantile definition and does not match Stata ``cic``.
    """
    xs = np.sort(x)
    n = len(xs)
    p = np.asarray(probs, dtype=float)
    k = np.clip(np.ceil(p * n).astype(int), 1, n)
    return np.asarray(xs[k - 1], dtype=float)


class CICResult(CausalResult):
    """CIC result with estimator-specific plot and summary methods."""

    _cic_plot_data: Dict[str, Any]

    def plot(self, type: str = "auto", **kwargs: Any) -> Tuple[Any, Any]:
        """CIC-specific plot: QTE plot if quantiles given, else CDF comparison."""
        import matplotlib.pyplot as plt

        ax = kwargs.pop("ax", None)
        plot_data = self._cic_plot_data
        has_qte = plot_data["qte_taus"] is not None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        if has_qte:
            taus = plot_data["qte_taus"]
            qte = plot_data["qte_point"]
            se = plot_data["qte_se"]
            z = stats.norm.ppf(1 - plot_data["alpha"] / 2)
            ax.plot(taus, qte, "o-", color="#2c7bb6", linewidth=2, label="QTE")
            ax.fill_between(
                taus,
                qte - z * se,
                qte + z * se,
                alpha=0.2,
                color="#2c7bb6",
            )
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            ax.axhline(
                self.estimate,
                color="#d7191c",
                linestyle=":",
                linewidth=1.2,
                label=f"ATT = {self.estimate:.4f}",
            )
            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel("Treatment Effect")
            ax.set_title("Changes-in-Changes: Quantile Treatment Effects")
            ax.legend()
        else:
            tau = plot_data["tau_grid"]
            obs_q = _quantile_func(plot_data["y11"], tau)
            cf_q_vals = plot_data["cf_quantiles"]
            ax.plot(
                tau,
                obs_q,
                color="#2c7bb6",
                linewidth=2,
                label="Observed (treated-post)",
            )
            ax.plot(
                tau,
                cf_q_vals,
                color="#d7191c",
                linewidth=2,
                linestyle="--",
                label="Counterfactual",
            )
            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel("Outcome")
            ax.set_title("Changes-in-Changes: Observed vs Counterfactual")
            ax.legend()

        fig.tight_layout()
        return fig, ax

    def summary(self, alpha: Optional[float] = None) -> str:
        a = self.alpha if alpha is None else alpha
        lines = []
        lines.append("━" * 60)
        lines.append("  Changes-in-Changes (Athey & Imbens, 2006)")
        lines.append("━" * 60)
        stars = CausalResult._stars(self.pvalue)
        lines.append(f"  ATT:          {self.estimate:.4f}{stars}")
        lines.append(f"  Bootstrap SE: {self.se:.4f}")
        pct = int(100 * (1 - a))
        lines.append(f"  {pct}% CI:      [{self.ci[0]:.4f}, {self.ci[1]:.4f}]")
        lines.append("")

        if self.detail is not None:
            lines.append("  Quantile Treatment Effects:")
            for _, row in self.detail.iterrows():
                s = CausalResult._stars(row["pvalue"])
                lines.append(
                    f"    τ = {row['quantile']:.2f}:   "
                    f"{row['qte']:.4f}  ({row['se']:.4f}) {s}"
                )
        lines.append("━" * 60)
        lines.append(f"  Observations: {self.n_obs:,}")
        cov = self.model_info.get("covariates")
        if cov:
            lines.append(f"  Covariates:   {', '.join(cov)}")
            lines.append(
                f"  First stage:  {self.model_info['first_stage']} "
                "(re-fit in every bootstrap replicate)"
            )
        lines.append(f"  Bootstrap replications: {self.model_info['n_boot']}")
        lines.append("━" * 60)
        out = "\n".join(lines)
        print(out)
        return out


def _counterfactual_map(y00: np.ndarray, y01: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Athey-Imbens counterfactual transform ``k(y) = F_01^{-1}(F_00(y))``.

    Maps a treated-pre outcome ``y`` through the control group's temporal
    change: read its rank in the control-pre distribution (``F_00``), then
    read off the control-post outcome at that rank (``F_01^{-1}``). Applying
    this to the treated-pre sample ``y10`` gives the counterfactual
    treated-post distribution ``F_{Y^N,11}`` (A&I 2006, eq. 9).
    """
    return _quantile_func(y01, _ecdf(y00, y))


def _counterfactual_quantiles(
    y00: np.ndarray,
    y01: np.ndarray,
    y10: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Counterfactual quantile function ``F_{Y^N,11}^{-1}(τ)``.

    The counterfactual treated-post distribution is the empirical distribution
    of the transformed treated-pre points ``k(y10)`` (see
    :func:`_counterfactual_map`); its quantile function is the generalized
    inverse of that empirical distribution.

    Previously this composed the CDFs as ``F_10^{-1}(F_00(F_01^{-1}(τ)))`` --
    the treated-pre (``y10``) and control-post (``y01``) cells transposed
    relative to A&I eq. 9 -- and on interpolated (not step) CDFs. Both are
    corrected here; the estimator now reproduces Stata ``cic`` to the printed
    digits. See CHANGELOG (⚠️ correctness fix).
    """
    cf_sample = _counterfactual_map(y00, y01, y10)
    return _quantile_func(cf_sample, grid)


# ── First stage (Melly & Santangelo two-step) ─────────────────────────


def _parse_covariates(
    covariates: Union[str, Sequence[str]],
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
) -> Tuple[List[str], List[str]]:
    """Split *covariates* into (absorbed FE terms, linear regressor terms).

    ``"C(state)"`` / ``"c(state)"`` / ``"i.state"`` mark a term for
    absorption as a fixed effect; anything else is a numeric regressor.

    Raises loudly on unknown columns, empty specs, and terms that collide
    with the CIC design columns.
    """
    if isinstance(covariates, str):
        covariates = [covariates]
    terms = [str(c).strip() for c in covariates]

    if not terms:
        raise ValueError(
            "cic(): covariates=[] is empty. Pass covariates=None for the "
            "unconditional estimator, or name at least one column, e.g. "
            "sp.cic(df, y='y', group='g', time='t', covariates=['x1'])."
        )

    fe_terms: List[str] = []
    num_terms: List[str] = []
    for term in terms:
        low = term.lower()
        if low.startswith("c(") and term.endswith(")"):
            fe_terms.append(term[2:-1].strip())
        elif low.startswith("i."):
            fe_terms.append(term[2:].strip())
        else:
            num_terms.append(term)

    wanted = fe_terms + num_terms
    missing = [c for c in wanted if c not in data.columns]
    if missing:
        available = ", ".join(map(repr, list(data.columns)[:12]))
        raise ValueError(
            f"cic(): covariate column(s) {missing!r} not found in `data`. "
            f"Available columns include: {available}. "
            "Fix the names, e.g. "
            f"sp.cic(data, y={y!r}, group={group!r}, time={time!r}, "
            f"covariates={[c for c in wanted if c in data.columns] or ['x1']!r})."
        )

    clash = [c for c in wanted if c in (y, group, time)]
    if clash:
        raise ValueError(
            f"cic(): covariate(s) {clash!r} duplicate the outcome/group/time "
            f"columns (y={y!r}, group={group!r}, time={time!r}). The group and "
            "time main effects are already absorbed by the CIC design. Drop "
            "them, e.g. covariates="
            f"{[c for c in wanted if c not in (y, group, time)] or ['x1']!r}."
        )

    dup = sorted({c for c in wanted if wanted.count(c) > 1})
    if dup:
        raise ValueError(
            f"cic(): covariate(s) {dup!r} listed more than once, which makes "
            "the first-stage design matrix rank-deficient. Pass each term "
            f"once, e.g. covariates={list(dict.fromkeys(wanted))!r}."
        )

    return fe_terms, num_terms


def _first_stage_residuals(
    frame: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    fe_terms: List[str],
    num_terms: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariate-adjusted outcome, Athey & Imbens (2006, p. 466).

    Runs OLS of ``y`` on the covariates *and* the (group × time) design
    dummies, then returns the residuals **with the design-dummy effects
    added back in** — A&I's parametric covariate approach, the same one
    Kranker's Stata ``cic`` implements.

    Including the design dummies in the first stage is what keeps the
    ATT intact: a first stage without them lets the covariate slope soak
    up the treatment effect whenever the covariate is imbalanced across
    cells.

    Returns ``(y_adj, keep_mask, coef)`` where ``keep_mask`` is a boolean
    mask over the *rows of* ``frame`` and ``y_adj`` has length
    ``keep_mask.sum()`` — callers MUST subset every other row-aligned
    array by ``keep_mask`` before pairing it with ``y_adj``.
    """
    from ..panel.hdfe import absorb_ols

    n = len(frame)
    yv = frame[y].to_numpy(dtype=float)

    # The 2x2 design cell is always absorbed (it spans the group, time
    # and group x time dummies) and is added back afterwards.
    cell = frame[group].to_numpy(dtype=np.int64) * 2 + frame[time].to_numpy(
        dtype=np.int64
    )
    fe = pd.DataFrame({"_cic_cell": cell}, index=frame.index)
    for term in fe_terms:
        fe[term] = frame[term].to_numpy()

    if num_terms:
        X = frame[num_terms].to_numpy(dtype=float)
    else:
        X = np.empty((n, 0), dtype=float)

    out = absorb_ols(yv, X, fe, return_absorber=True)
    keep_mask = np.asarray(out["absorber"].keep_mask, dtype=bool)
    resid = np.asarray(out["resid"], dtype=float)
    coef = np.asarray(out["coef"], dtype=float)

    # keep_mask alignment is the silent-corruption failure mode: if HDFE
    # returns a residual vector whose length does not match the surviving
    # rows we cannot pair residuals with (group, time) — raise, never guess.
    if keep_mask.shape[0] != n or resid.shape[0] != int(keep_mask.sum()):
        raise RuntimeError(
            "cic(): first-stage residuals could not be aligned back to the "
            f"input rows (frame has {n} rows, keep_mask has "
            f"{keep_mask.shape[0]}, residuals have {resid.shape[0]}, "
            f"expected {int(keep_mask.sum())}). Refusing to guess an "
            "alignment. Please report this with a reproducible example."
        )

    # Add the design-dummy effects back in. Because the cell is absorbed,
    # the HDFE residuals are exactly orthogonal to the cell dummies, so
    # the cell means of ``y - X b`` are precisely the fitted cell effects.
    yk = yv[keep_mask]
    cell_k = cell[keep_mask]
    partial = yk - (X[keep_mask] @ coef if coef.size else 0.0)
    cell_dense, _ = pd.factorize(cell_k, sort=True)
    counts = np.bincount(cell_dense)
    sums = np.bincount(cell_dense, weights=partial)
    y_adj = resid + (sums / counts)[cell_dense]

    return y_adj, keep_mask, coef


def _cell_arrays(
    yv: np.ndarray, g: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split *yv* into the four (group x time) cells."""
    return (
        yv[(g == 0) & (t == 0)],
        yv[(g == 0) & (t == 1)],
        yv[(g == 1) & (t == 0)],
        yv[(g == 1) & (t == 1)],
    )


_CELL_LABELS = ("control-pre", "control-post", "treated-pre", "treated-post")


def _check_cells(
    cells: Sequence[np.ndarray],
    *,
    context: str,
    hint: str,
) -> None:
    for label, arr in zip(_CELL_LABELS, cells):
        if len(arr) < 2:
            raise ValueError(
                f"Too few observations in the {label} cell ({len(arr)}) "
                f"{context}. CIC requires data in all four (group × time) "
                f"cells. {hint}"
            )


# ── Main estimator ────────────────────────────────────────────────────


def cic(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    quantiles: Optional[List[float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    n_grid: int = 200,
    covariates: Optional[Union[str, Sequence[str]]] = None,
    first_stage: str = "feols",
) -> CausalResult:
    """Changes-in-Changes estimator (Athey & Imbens 2006).

    Parameters
    ----------
    data : DataFrame
        Panel or repeated cross-section.
    y : str
        Outcome variable.
    group : str
        Binary group indicator (0 = control, 1 = treated).
    time : str
        Binary time indicator (0 = pre, 1 = post).
    quantiles : list of float, optional
        Quantiles at which to compute QTE.  ``None`` → ATT only.
    n_boot : int
        Number of bootstrap replications for SEs / CIs.
    alpha : float
        Significance level.
    seed : int
        Random seed for reproducibility.
    n_grid : int
        Retained for backwards compatibility only.  The corrected
        Athey-Imbens estimator is exact on the sample points (step-function
        ECDF and its generalized inverse), so no internal τ grid enters the
        point estimate, the QTEs, or the bootstrap.  ``n_grid`` now only
        sets the resolution of the observed-vs-counterfactual quantile
        curves stored for ``result.plot()``.

        .. versionchanged:: 1.20.x
           Before the step-2 correctness fix the estimator integrated a
           linearly-interpolated counterfactual quantile function over an
           ``n_grid``-point τ grid, so this parameter perturbed the ATT.
           It no longer does.
    covariates : str or list of str, optional
        Covariates for the Melly & Santangelo (2015) two-step estimator.
        ``None`` (default) → the unconditional Athey-Imbens estimator,
        bit-identical to previous releases.  Terms written ``"C(col)"`` or
        ``"i.col"`` are absorbed as high-dimensional fixed effects; every
        other term is a linear regressor.
    first_stage : str, default ``"feols"``
        First-stage residualizer.  Only ``"feols"`` (OLS-HDFE) is
        supported; the argument exists so future first stages can be
        added without a breaking signature change.

    Returns
    -------
    CausalResult

    Notes
    -----
    **Two-step estimation.**  With ``covariates`` the estimator follows
    Athey & Imbens (2006, p. 466): step 1 regresses ``y`` on the covariates
    *and* the (group × time) design dummies by OLS-HDFE and forms the
    adjusted outcome as the residuals **with the design-dummy effects added
    back in**; step 2 runs the unconditional CIC on that adjusted outcome.
    Keeping the design dummies in the first stage is essential — without
    them the covariate slope absorbs part of the treatment effect whenever
    the covariate is imbalanced across cells.  This is the same parametric
    covariate approach implemented by Kranker's Stata ``cic``.

    **Inference.**  The bootstrap resamples *both* steps jointly: every
    replicate re-draws the sample (cell-wise, holding the four cell sizes
    fixed), **re-fits the first stage**, and re-runs CIC.  Residualizing
    once and bootstrapping only step 2 — the natural hand-rolled ``feols``
    → residuals → ``cic`` pipeline — holds the first-stage coefficients
    fixed at their full-sample values and misstates the standard error.
    See ``tests/test_cic_covariates.py`` for the Monte Carlo that calibrates
    both schemes against the true sampling distribution.

    High-dimensional fixed effects drop singleton and incomplete rows.  The
    residual vector is realigned to the surviving rows via the absorber's
    ``keep_mask`` before step 2; if that alignment cannot be verified the
    estimator raises rather than risk silently mis-pairing observations.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> g = rng.integers(0, 2, n)
    >>> t = rng.integers(0, 2, n)
    >>> y = (1.0 + 0.5 * g + 0.3 * t + 1.5 * g * t
    ...      + rng.normal(0, 1, n))
    >>> df = pd.DataFrame({"y": y, "g": g, "t": t})
    >>> res = sp.cic(df, y="y", group="g", time="t",
    ...              quantiles=[0.25, 0.5, 0.75], n_boot=50)
    >>> round(res.estimate, 2)  # ATT, true effect = 1.5
    1.44
    >>> res.model_info["qte"].shape  # one row per quantile
    (3, 6)
    """
    if first_stage not in _SUPPORTED_FIRST_STAGE:
        raise ValueError(
            f"cic(): first_stage={first_stage!r} is not supported; expected "
            f"one of {list(_SUPPORTED_FIRST_STAGE)!r}. Use "
            f"sp.cic(data, y={y!r}, group={group!r}, time={time!r}, "
            "covariates=[...], first_stage='feols')."
        )

    use_cov = covariates is not None
    fe_terms: List[str] = []
    num_terms: List[str] = []
    if use_cov:
        fe_terms, num_terms = _parse_covariates(covariates, data, y, group, time)

    cov_cols = fe_terms + num_terms
    keep_cols = list(dict.fromkeys([y, group, time] + cov_cols))
    n_raw = len(data)
    df = data[keep_cols].dropna().copy()
    if use_cov and len(df) == 0:
        raise ValueError(
            f"cic(): no complete rows left after dropping missing values in "
            f"{keep_cols!r} ({n_raw} input rows). Check the covariates for "
            "all-missing columns, e.g. "
            f"data[{cov_cols!r}].isna().mean()."
        )

    g_all = df[group].astype(int).values
    t_all = df[time].astype(int).values
    y_all = df[y].values.astype(float)

    first_stage_info: Dict[str, Any] = {}
    if use_cov:
        resid, keep_mask, fs_coef = _first_stage_residuals(
            df, y, group, time, fe_terms, num_terms
        )
        yv = resid
        g = g_all[keep_mask]
        t = t_all[keep_mask]

        # A first stage that explains everything leaves no distributional
        # variation for CIC to work with.
        raw_var = float(np.var(y_all[keep_mask]))
        res_var = float(np.var(yv))
        if res_var <= 1e-12 * max(raw_var, 1.0):
            raise ValueError(
                f"cic(): the first stage removed essentially all variation in "
                f"{y!r} (residual variance {res_var:.3e} vs outcome variance "
                f"{raw_var:.3e}). covariates={cov_cols!r} saturate the design "
                "— typically a term that is collinear with, or a finer "
                "partition than, group × time. Drop it, e.g. "
                f"sp.cic(data, y={y!r}, group={group!r}, time={time!r}, "
                f"covariates={cov_cols[:1] or ['x1']!r})."
            )

        first_stage_info = {
            "first_stage": first_stage,
            "covariates": list(cov_cols),
            "fe_terms": list(fe_terms),
            "linear_terms": list(num_terms),
            "first_stage_coef": dict(zip(num_terms, fs_coef.tolist())),
            "n_dropped_first_stage": int((~keep_mask).sum()),
            "n_dropped_missing": int(n_raw - len(df)),
            "bootstrap": "two-step (first stage re-fit in each replicate)",
        }
    else:
        yv = y_all
        g = g_all
        t = t_all
        keep_mask = np.ones(len(df), dtype=bool)

    # Split into four cells
    y00, y01, y10, y11 = _cell_arrays(yv, g, t)

    if use_cov:
        _check_cells(
            (y00, y01, y10, y11),
            context=(
                f"after dropping rows with missing covariates and "
                f"first-stage singletons "
                f"({n_raw} input rows → {len(df)} complete rows → "
                f"{int(keep_mask.sum())} estimation rows)"
            ),
            hint=(
                "Either drop the high-cardinality fixed effect that is "
                f"pruning the sample (covariates={cov_cols!r}) or fit the "
                "unconditional estimator with "
                f"sp.cic(data, y={y!r}, group={group!r}, time={time!r})."
            ),
        )
    else:
        for label, arr in [
            ("control-pre", y00),
            ("control-post", y01),
            ("treated-pre", y10),
            ("treated-post", y11),
        ]:
            if len(arr) < 2:
                raise ValueError(
                    f"Too few observations in the {label} cell ({len(arr)}). "
                    "CIC requires data in all four (group × time) cells."
                )

    # Quantile grid
    tau_grid = np.linspace(1 / n_grid, 1 - 1 / n_grid, n_grid)

    # ── Point estimates ───────────────────────────────────────────── #
    # The mean ATT uses the counterfactual distribution's own mean, i.e. the
    # mean of the transformed treated-pre sample k(y10) (A&I 2006). This is
    # exact and grid-free; grid-integrating the counterfactual quantile
    # function would only approximate it. ``cf_q`` (on the τ grid) is retained
    # for the observed-vs-counterfactual plot.
    cf_sample = _counterfactual_map(y00, y01, y10)
    cf_q = _quantile_func(cf_sample, tau_grid)
    att_point = np.mean(y11) - np.mean(cf_sample)

    qte_taus = np.asarray(quantiles) if quantiles is not None else None
    qte_point = None
    if qte_taus is not None:
        obs_q11 = _quantile_func(y11, qte_taus)
        cf_q_at_tau = _quantile_func(cf_sample, qte_taus)
        qte_point = obs_q11 - cf_q_at_tau

    # ── Bootstrap ─────────────────────────────────────────────────── #
    rng = np.random.RandomState(seed)
    boot_att = np.empty(n_boot)
    boot_qte = np.empty((n_boot, len(qte_taus))) if qte_taus is not None else None

    idx00 = np.where((g == 0) & (t == 0))[0]
    idx01 = np.where((g == 0) & (t == 1))[0]
    idx10 = np.where((g == 1) & (t == 0))[0]
    idx11 = np.where((g == 1) & (t == 1))[0]

    # Row positions in `df` corresponding to each estimation-sample row —
    # needed to rebuild a bootstrap frame for the first-stage re-fit.
    df_pos = np.where(keep_mask)[0]
    n_boot_failed = 0

    for b in range(n_boot):
        s00 = rng.choice(idx00, len(idx00), replace=True)
        s01 = rng.choice(idx01, len(idx01), replace=True)
        s10 = rng.choice(idx10, len(idx10), replace=True)
        s11 = rng.choice(idx11, len(idx11), replace=True)

        if not use_cov:
            b00, b01, b10, b11 = yv[s00], yv[s01], yv[s10], yv[s11]
        else:
            # Two-step bootstrap: re-draw the sample, RE-FIT the first
            # stage on the drawn sample, then re-run CIC on the new
            # residuals. Bootstrapping step 2 alone would hold the
            # first-stage coefficients fixed and understate the SE.
            sel = np.concatenate([s00, s01, s10, s11])
            boot_frame = df.iloc[df_pos[sel]]
            try:
                b_resid, b_keep, _ = _first_stage_residuals(
                    boot_frame, y, group, time, fe_terms, num_terms
                )
            except Exception:
                boot_att[b] = np.nan
                if boot_qte is not None:
                    boot_qte[b] = np.nan
                n_boot_failed += 1
                continue
            bg = boot_frame[group].astype(int).values[b_keep]
            bt = boot_frame[time].astype(int).values[b_keep]
            b00, b01, b10, b11 = _cell_arrays(b_resid, bg, bt)
            if min(len(b00), len(b01), len(b10), len(b11)) < 2:
                boot_att[b] = np.nan
                if boot_qte is not None:
                    boot_qte[b] = np.nan
                n_boot_failed += 1
                continue

        bcf_sample = _counterfactual_map(b00, b01, b10)
        boot_att[b] = np.mean(b11) - np.mean(bcf_sample)

        if qte_taus is not None:
            assert boot_qte is not None
            bq11 = _quantile_func(b11, qte_taus)
            bcf_tau = _quantile_func(bcf_sample, qte_taus)
            boot_qte[b] = bq11 - bcf_tau

    if n_boot_failed:
        frac = n_boot_failed / n_boot
        if frac > 0.10:
            raise ValueError(
                f"cic(): {n_boot_failed}/{n_boot} bootstrap replicates "
                f"({frac:.0%}) collapsed — the re-fitted first stage left "
                "fewer than 2 observations in some (group × time) cell. The "
                f"fixed effects in covariates={cov_cols!r} are too fine for "
                "this sample. Coarsen them or fit the unconditional "
                f"estimator: sp.cic(data, y={y!r}, group={group!r}, "
                f"time={time!r})."
            )
        warnings.warn(
            f"cic(): {n_boot_failed}/{n_boot} bootstrap replicates were "
            "discarded because the re-fitted first stage emptied a "
            f"(group × time) cell; SEs use the remaining "
            f"{n_boot - n_boot_failed} replicates.",
            UserWarning,
            stacklevel=2,
        )

    att_se = np.nanstd(boot_att, ddof=1) if use_cov else np.std(boot_att, ddof=1)
    _pct = np.nanpercentile if use_cov else np.percentile
    att_ci = (
        _pct(boot_att, 100 * alpha / 2),
        _pct(boot_att, 100 * (1 - alpha / 2)),
    )
    att_z = att_point / att_se if att_se > 0 else np.nan
    att_pvalue = float(2 * (1 - stats.norm.cdf(np.abs(att_z))))

    # ── Build detail DataFrame ────────────────────────────────────── #
    detail = None
    model_info: Dict[str, Any] = {
        "n_control_pre": len(y00),
        "n_control_post": len(y01),
        "n_treated_pre": len(y10),
        "n_treated_post": len(y11),
        "n_boot": n_boot,
    }
    if use_cov:
        model_info.update(first_stage_info)
        model_info["n_boot_failed"] = n_boot_failed

    if qte_taus is not None and boot_qte is not None:
        _std = np.nanstd if use_cov else np.std
        qte_se = _std(boot_qte, axis=0, ddof=1)
        qte_ci_lo = _pct(boot_qte, 100 * alpha / 2, axis=0)
        qte_ci_hi = _pct(boot_qte, 100 * (1 - alpha / 2), axis=0)
        qte_z = np.where(qte_se > 0, qte_point / qte_se, np.nan)
        qte_pv = 2 * (1 - stats.norm.cdf(np.abs(qte_z)))

        detail = pd.DataFrame(
            {
                "quantile": qte_taus,
                "qte": qte_point,
                "se": qte_se,
                "ci_lower": qte_ci_lo,
                "ci_upper": qte_ci_hi,
                "pvalue": qte_pv,
            }
        )
        model_info["qte"] = detail

    n_obs = int(keep_mask.sum()) if use_cov else len(df)

    method = "Changes-in-Changes (Athey & Imbens, 2006)"
    if use_cov:
        method += " + covariates (Melly & Santangelo, 2015)"

    result = CICResult(
        method=method,
        estimand="ATT",
        estimate=float(att_point),
        se=float(att_se),
        pvalue=float(att_pvalue),
        ci=att_ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=model_info,
        _citation_key="cic",
    )

    result._cic_plot_data = {
        "y11": y11,
        "cf_quantiles": cf_q,
        "tau_grid": tau_grid,
        "qte_taus": qte_taus,
        "qte_point": qte_point,
        "qte_se": (
            (np.nanstd if use_cov else np.std)(boot_qte, axis=0, ddof=1)
            if boot_qte is not None
            else None
        ),
        "alpha": alpha,
    }
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.did.cic",
            params={
                "y": y,
                "group": group,
                "time": time,
                "quantiles": list(quantiles) if quantiles else None,
                "n_boot": n_boot,
                "alpha": alpha,
                "seed": seed,
                "n_grid": n_grid,
                "covariates": list(cov_cols) if use_cov else None,
                "first_stage": first_stage if use_cov else None,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return result


__all__ = ["cic"]
