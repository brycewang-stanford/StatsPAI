"""Stata ``psmatch2``-faithful propensity-score matching with a full
post-matching toolkit (balance, common-support plot, PSM-DID).

``sp.psmatch2`` is the migration-friendly front door for analysts coming
from Stata's ``psmatch2`` (Leuven & Sianesi 2003).  It wraps the supported
nearest-neighbour, kernel, and radius propensity-score matching paths in a
:class:`PSMatch2Result` that exposes the matched-sample variables
(``_pscore``, ``_treated``, ``_support``, ``_weight``, ``_y``; plus
``_n1`` … ``_nk``, ``_nn``, ``_pdif`` for nearest-neighbour matching) and the three
operations that ``sp.match`` alone could not previously support:

1. ``.balance()``  — covariate balance on the *matched, weighted* sample
   (the post-matching analogue of Stata ``pstest``).
2. ``.psplot()``   — propensity-score density before/after matching, with
   the controls reweighted by ``_weight`` (the common-support diagnostic).
3. ``.psm_did()``  — frequency-weighted PSM-DID: merge ``_weight`` into a
   panel, keep the matched sample, and run the weighted
   ``y ~ treat * post`` regression (Stata's ``reg y i.treat##i.post
   [fweight=_weight]``).

For the pinned Stata 18 ``psmatch2`` paths (nearest-neighbour,
Epanechnikov kernel, and radius matching), the point estimate, analytic
SE, and emitted matched-frame columns match the reference fixtures — see
``tests/reference_parity``.

References
----------
Leuven, E. and Sianesi, B. (2003). PSMATCH2: Stata module to perform full
    Mahalanobis and propensity score matching, common support graphing, and
    covariate imbalance testing. Statistical Software Components S432001.
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
Heckman, J.J., Ichimura, H. and Todd, P.E. (1997). Review of Economic
    Studies, 64(4), 605-654.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..core.results import CausalResult, SummaryText
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _matched_frame as _mf

if TYPE_CHECKING:
    from .ps_diagnostics import BalanceDiagnosticsResult

_PSMATCH2_ALTERNATIVES = ["sp.psmatch2", "sp.match", "sp.psm"]


def _psmatch2_error(
    message: str,
    *,
    diagnostics: Optional[dict[str, Any]] = None,
    recovery_hint: str = "Check psmatch2 inputs and option names.",
) -> MethodIncompatibility:
    return MethodIncompatibility(
        message,
        recovery_hint=recovery_hint,
        diagnostics=diagnostics,
        alternative_functions=_PSMATCH2_ALTERNATIVES,
    )


def _require_dataframe(data: Any, context: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise _psmatch2_error(
            f"{context} must be a pandas DataFrame.",
            diagnostics={"context": context, "type": type(data).__name__},
            recovery_hint="Pass a pandas DataFrame.",
        )
    if data.empty:
        raise DataInsufficient(
            f"{context} is empty.",
            recovery_hint="Provide a non-empty matching sample.",
            diagnostics={"context": context, "n_rows": 0},
            alternative_functions=_PSMATCH2_ALTERNATIVES,
        )
    return data


def _require_column(data: pd.DataFrame, column: Any, role: str) -> str:
    if not isinstance(column, str) or not column:
        raise _psmatch2_error(
            f"{role} must be a non-empty column-name string.",
            diagnostics={"role": role, "value": repr(column)},
            recovery_hint="Pass column names as strings.",
        )
    if column not in data.columns:
        raise _psmatch2_error(
            f"{role} column {column!r} not found in data.",
            diagnostics={
                "role": role,
                "column": column,
                "available_columns": [str(c) for c in data.columns],
            },
            recovery_hint="Check the column spelling or rename the DataFrame.",
        )
    return column


def _normalize_columns(columns: Any, role: str) -> List[str]:
    if columns is None:
        return []
    if isinstance(columns, str):
        raw = [columns]
    else:
        try:
            raw = list(columns)
        except TypeError as exc:
            raise _psmatch2_error(
                f"{role} must be a sequence of column-name strings.",
                diagnostics={"role": role, "type": type(columns).__name__},
                recovery_hint=f"Pass {role} as ['x1', 'x2'] or a single string.",
            ) from exc
    out: List[str] = []
    for idx, column in enumerate(raw):
        if not isinstance(column, str) or not column:
            raise _psmatch2_error(
                f"{role}[{idx}] must be a non-empty column-name string.",
                diagnostics={"role": role, "index": idx, "value": repr(column)},
                recovery_hint="Pass column names as strings.",
            )
        out.append(column)
    return out


# ======================================================================
# Result object
# ======================================================================


class PSMatch2Result:
    """Container for a ``sp.psmatch2`` run.

    Attributes
    ----------
    matched_data : DataFrame
        The input data plus the psmatch2 columns (``_pscore``,
        ``_treated``, ``_support``, ``_weight``, ``_y``; plus ``_n1`` …,
        ``_nn``, ``_pdif`` for nearest-neighbour matching).  Also available
        as ``.data``.
    att, se, pvalue, ci : float / tuple
        Average treatment effect on the treated and its inference.
    estimand : str
        Always ``'ATT'`` for ``psmatch2``.
    result : CausalResult
        The underlying :func:`sp.match` result.

    Methods
    -------
    matched_sample(on_support=True)
        Rows that entered the matched sample (``_weight`` not missing).
    balance(covariates=None)
        Post-matching covariate balance on the weighted matched sample.
    psplot(...) / overlap_plot(...)
        Propensity-score density before/after matching.
    psm_did(panel, id, ...)
        Frequency-weighted PSM-DID regression.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> m = sp.psmatch2(df, outcome='log_wage', treat='union',
    ...                 covariates=['education', 'experience', 'tenure'])
    >>> '_weight' in m.matched_data.columns
    True
    >>> bal = m.balance()                      # post-matching balance
    >>> fig, ax = m.psplot()                   # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        result: CausalResult,
        matched_data: pd.DataFrame,
        treat: str,
        covariates: List[str],
        outcome: Optional[str],
        n_matches: int,
        common_support: str,
        method: str = "neighbor",
    ):
        self.result = result
        self.matched_data = matched_data
        self.data = matched_data  # alias
        self.treat = treat
        self.covariates = list(covariates)
        self.outcome = outcome
        self.n_matches = int(n_matches)
        self.common_support = common_support
        self.method = method

        self.estimand = result.estimand
        self.att = result.estimate
        self.se = result.se
        self.pvalue = result.pvalue
        self.ci = result.ci
        self.alpha = result.alpha
        mi = result.model_info or {}
        self.n_treated = int(mi.get("n_treated", 0))
        self.n_control = int(mi.get("n_control", 0))
        self.n_on_support = int(
            mi.get("n_treated_on_support", mi.get("n_on_support", self.n_treated))
        )
        self.n_total_on_support = int(mi.get("n_on_support", self.n_on_support))
        self.n_matched_treated = int(mi.get("n_matched_treated", 0))

    # ------------------------------------------------------------------
    # Matched sample extraction
    # ------------------------------------------------------------------

    def matched_sample(
        self,
        *,
        on_support: bool = True,
        drop_unmatched: bool = True,
    ) -> pd.DataFrame:
        """Return the rows that make up the matched sample.

        Parameters
        ----------
        on_support : bool, default True
            Keep only rows with ``_support == 1``.  Has no effect when
            matching was run with ``common_support='none'`` (every row is
            on support).
        drop_unmatched : bool, default True
            Drop rows with a missing ``_weight`` — i.e. controls never used
            as a match and treated units that found no match.  This is the
            sample Stata uses for ``[fweight=_weight]`` regressions.

        Returns
        -------
        DataFrame
        """
        df = self.matched_data
        mask = pd.Series(True, index=df.index)
        if drop_unmatched:
            mask &= df[_mf.COL_WEIGHT].notna()
        if on_support:
            mask &= df[_mf.COL_SUPPORT] == 1
        return df.loc[mask].copy()

    # ------------------------------------------------------------------
    # Post-matching balance (post-match pstest)
    # ------------------------------------------------------------------

    def balance(
        self,
        covariates: Optional[Sequence[str]] = None,
        *,
        threshold: float = 0.1,
    ) -> BalanceDiagnosticsResult:
        """Covariate balance before vs after matching (Stata ``pstest``).

        Standardized mean differences are reported two ways, exactly like
        ``pstest``:

        - ``smd_raw``      — *before* matching: unweighted SMD over the full
          treated vs control sample.
        - ``smd_weighted`` — *after* matching: SMD with the ``_weight``
          frequency weights, so a control used twice counts twice and
          unmatched / off-support units drop out (weight 0).

        Parameters
        ----------
        covariates : list of str, optional
            Variables to assess.  Defaults to the matching covariates.
        threshold : float, default 0.1
            |SMD| balance threshold.

        Returns
        -------
        BalanceDiagnosticsResult
            ``.table`` (per-covariate before vs after SMD, variance ratio,
            KS) and ``.summary_stats``.
        """
        from .ps_diagnostics import balance_diagnostics

        covs = list(covariates) if covariates is not None else self.covariates
        df = self.matched_data
        # Frequency weights: matched treated = 1, matched controls = their
        # accumulated share, everything else 0 (drops out of the "after").
        w = df[_mf.COL_WEIGHT].fillna(0.0).to_numpy(dtype=float)
        return balance_diagnostics(
            df,
            treatment=self.treat,
            covariates=covs,
            weights=w,
            ps=df[_mf.COL_PSCORE].to_numpy(dtype=float),
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Common-support / propensity-score plot
    # ------------------------------------------------------------------

    def psplot(
        self,
        *,
        before: bool = True,
        n_grid: int = 300,
        ax: Any = None,
        figsize: tuple[float, float] = (8.0, 4.5),
        title: Optional[str] = None,
    ) -> tuple[Any, Any]:
        """Propensity-score density by treatment group, after matching.

        Controls are reweighted by ``_weight`` so the plotted control
        density reflects the matched sample, not the raw pool.  With
        ``before=True`` the raw (unweighted) densities are overlaid as
        dashed lines so the user can see how matching tightened overlap.

        Returns
        -------
        (fig, ax)
        """
        try:
            import matplotlib.pyplot as plt
            from scipy import stats as _stats
        except ImportError as exc:  # pragma: no cover
            raise ImportError("matplotlib + scipy required for psplot().") from exc

        df = self.matched_data
        ps = df[_mf.COL_PSCORE].to_numpy(dtype=float)
        treat = df[self.treat].to_numpy(dtype=float)
        w = df[_mf.COL_WEIGHT].to_numpy(dtype=float)
        finite = np.isfinite(ps)

        cw = np.where(np.isfinite(w), w, 0.0)
        t_mask = finite & (treat == 1)
        tmatched = t_mask & (cw > 0)
        c_mask = finite & (treat == 0)
        # Matched control sample: positive weight.
        cmatched = c_mask & (cw > 0)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        grid = np.linspace(0.0, 1.0, n_grid)

        def _kde(x: Any, weights: Optional[Any] = None) -> Optional[Any]:
            if len(x) < 2 or np.allclose(x, x[0]):
                return None
            kde = _stats.gaussian_kde(x, weights=weights)
            return kde(grid)

        # After-matching: both groups are restricted to rows with positive
        # matching weight, so caliper failures / off-support treated units do
        # not appear in the post-match density.
        d_t = _kde(ps[tmatched], weights=cw[tmatched])
        d_c = _kde(ps[cmatched], weights=cw[cmatched])
        if d_t is not None:
            ax.fill_between(
                grid,
                d_t,
                alpha=0.35,
                color="#2171b5",
                label=f"Matched treated (n={int(tmatched.sum())})",
            )
            ax.plot(grid, d_t, color="#2171b5", lw=1.2)
        if d_c is not None:
            ax.fill_between(
                grid,
                -d_c,
                alpha=0.35,
                color="#cb181d",
                label=f"Matched control (n={int(cmatched.sum())})",
            )
            ax.plot(grid, -d_c, color="#cb181d", lw=1.2)

        if before:
            d_t0 = _kde(ps[t_mask])  # treated unchanged
            d_c0 = _kde(ps[c_mask])  # all controls, unweighted
            if d_c0 is not None:
                ax.plot(
                    grid,
                    -d_c0,
                    color="#cb181d",
                    lw=1.0,
                    ls="--",
                    alpha=0.7,
                    label="Control (raw)",
                )
            if d_t0 is not None:
                ax.plot(grid, d_t0, color="#2171b5", lw=1.0, ls="--", alpha=0.4)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Propensity score")
        ax.set_ylabel("Density")
        ax.set_title(title or "Propensity score: matched sample (psmatch2)")
        ax.legend(frameon=False, fontsize=9)
        ax.set_xlim(0, 1)
        fig.tight_layout()
        return fig, ax

    # overlap_plot is a common name for the same thing.
    overlap_plot = psplot

    # ------------------------------------------------------------------
    # PSM-DID
    # ------------------------------------------------------------------

    def psm_did(
        self,
        panel: pd.DataFrame,
        *,
        id: str,
        y: str,
        time: Optional[str] = None,
        post: Optional[str] = None,
        treat: Optional[str] = None,
        treat_time: Optional[Any] = None,
        covariates: Optional[Sequence[str]] = None,
        fixed_effects: Optional[Sequence[str]] = None,
        cluster: Optional[Union[str, List[str]]] = None,
        on_support: bool = True,
        weight: str = "fweight",
        alpha: float = 0.05,
    ) -> CausalResult:
        """Frequency-weighted PSM-DID on a panel.

        Implements the Stata workflow

        .. code-block:: stata

            psmatch2 d x1 x2, out(y) ...        // produces _weight
            // merge _weight back onto the panel by id, then
            reg y i.treat##i.post [fweight=_weight] if _support==1

        The matching ``_weight`` (and ``_support``) are merged onto ``panel``
        by ``id``, the matched sample is selected, and the weighted
        difference-in-differences regression

        ``y ~ treat + post + treat:post  (+ covariates | fixed_effects)``

        is fitted with :func:`sp.feols`.  The ``treat:post`` coefficient is
        the PSM-DID treatment effect.

        Parameters
        ----------
        panel : DataFrame
            Long panel (one row per unit-period).
        id : str
            Unit identifier.  Must also exist in the matching data so the
            per-unit ``_weight`` can be merged in.
        y : str
            Outcome in the panel.
        time : str, optional
            Time variable.  Used with ``treat_time`` to build ``post`` if
            ``post`` is not supplied directly.
        post : str, optional
            Binary post-period indicator.  Provide this *or* ``time`` +
            ``treat_time``.
        treat : str, optional
            Time-invariant treated-group indicator in the panel.  Defaults
            to the matching treatment variable.
        treat_time : scalar, optional
            First treated period; ``post = time >= treat_time``.
        covariates : list of str, optional
            Additional time-varying controls.
        fixed_effects : list of str, optional
            Columns absorbed as fixed effects (e.g. ``[id, time]`` for TWFE).
        cluster : str or list, optional
            Cluster variable(s) for the standard errors.
        on_support : bool, default True
            Keep only matched units on common support.
        weight : {'fweight', 'none'}, default 'fweight'
            ``'fweight'`` weights the regression by ``_weight``; ``'none'``
            runs the matched-sample DiD unweighted.
        alpha : float, default 0.05
            Significance level for the returned CI.

        Returns
        -------
        CausalResult
            ``.estimate`` is the DiD (``treat:post``) coefficient; the full
            weighted regression is stored in ``model_info['feols_result']``.
        """
        from ..panel.feols import feols

        if weight not in {"fweight", "none"}:
            raise _psmatch2_error(
                "weight must be 'fweight' or 'none'.",
                diagnostics={"weight": weight},
                recovery_hint="Use weight='fweight' or weight='none'.",
            )

        panel = _require_dataframe(panel, "psm_did panel")
        treat = treat or self.treat
        id = _require_column(panel, id, "id")
        y = _require_column(panel, y, "outcome")
        treat = _require_column(panel, treat, "treat")
        covariates = _normalize_columns(covariates, "covariates")
        fixed_effects = _normalize_columns(fixed_effects, "fixed_effects")
        for covariate in covariates:
            _require_column(panel, covariate, "covariate")
        for fixed_effect in fixed_effects:
            _require_column(panel, fixed_effect, "fixed effect")
        if cluster is not None:
            if isinstance(cluster, str):
                _require_column(panel, cluster, "cluster")
            else:
                for cluster_col in _normalize_columns(cluster, "cluster"):
                    _require_column(panel, cluster_col, "cluster")

        # --- merge per-unit _weight / _support onto the panel ---
        md = self.matched_data
        if id not in md.columns:
            raise _psmatch2_error(
                f"id column {id!r} not found in the matching data; psm_did "
                f"needs it to merge _weight onto the panel.",
                diagnostics={"id": id},
                recovery_hint="Run sp.psmatch2() on data that includes the "
                "same unit id column as the panel.",
            )
        used_cols = set(panel.columns)
        psm_weight_col = _fresh_column("__statspai_psm_weight__", used_cols)
        used_cols.add(psm_weight_col)
        psm_support_col = _fresh_column("__statspai_psm_support__", used_cols)
        wmap = (
            md[[id, _mf.COL_WEIGHT, _mf.COL_SUPPORT]]
            .drop_duplicates(subset=[id])
            .rename(
                columns={
                    _mf.COL_WEIGHT: psm_weight_col,
                    _mf.COL_SUPPORT: psm_support_col,
                }
            )
        )
        merged = panel.merge(wmap, on=id, how="left")

        # --- build the post indicator ---
        if post is None:
            if time is None or treat_time is None:
                raise _psmatch2_error(
                    "Provide either post=<column> or both time=<column> and "
                    "treat_time=<scalar> so psm_did can build the post period.",
                    diagnostics={"post": post, "time": time, "treat_time": treat_time},
                    recovery_hint="Pass post='post' or pass both time= and "
                    "treat_time=.",
                )
            time = _require_column(panel, time, "time")
            post = "_post"
            merged[post] = (merged[time] >= treat_time).astype(float)
        else:
            post = _require_column(panel, post, "post")

        # --- restrict to the matched sample ---
        mask = merged[psm_weight_col].notna()
        if on_support:
            mask &= merged[psm_support_col] == 1
        samp = merged.loc[mask].copy()
        if samp.empty:
            raise DataInsufficient(
                "No matched panel rows after merging _weight.",
                recovery_hint="Check that the panel id overlaps the psmatch2 "
                "matched sample and relax common-support/caliper trimming if "
                "needed.",
                diagnostics={"on_support": on_support, "id": id},
                alternative_functions=_PSMATCH2_ALTERNATIVES,
            )

        # --- DiD interaction (feols needs a bare column) ---
        did_col = "_did"
        samp[treat] = samp[treat].astype(float)
        samp[post] = samp[post].astype(float)
        samp[did_col] = samp[treat] * samp[post]

        # Drop a main effect when a fixed effect already absorbs it (e.g.
        # the time-invariant treated indicator under unit FE, or post under
        # time FE) — otherwise the design matrix is singular.  The DiD
        # interaction is always kept.
        def _absorbed(col: str) -> bool:
            for fe in fixed_effects:
                if samp.groupby(fe)[col].nunique().max() <= 1:
                    return True
            return False

        mains = [c for c in (treat, post) if not _absorbed(c)]
        rhs = mains + [did_col] + covariates
        formula = f"{y} ~ " + " + ".join(rhs)
        if fixed_effects:
            formula += " | " + " + ".join(fixed_effects)

        w_arg = psm_weight_col if weight == "fweight" else None
        fres = feols(formula, samp, weights=w_arg, cluster=cluster, alpha=alpha)

        # --- pull out the DiD coefficient as a CausalResult ---
        coef = fres.coef if hasattr(fres, "coef") else fres.params
        se_s = fres.se if hasattr(fres, "se") else fres.std_errors
        beta = float(coef[did_col])
        se = float(se_s[did_col])
        from scipy import stats as _stats

        z = _stats.norm.ppf(1 - alpha / 2)
        tstat = beta / se if se > 0 else 0.0
        pval = float(2 * (1 - _stats.norm.cdf(abs(tstat))))
        ci = (beta - z * se, beta + z * se)

        out = CausalResult(
            method="PSM-DID (psmatch2 + weighted DiD)",
            estimand="ATT",
            estimate=beta,
            se=se,
            pvalue=pval,
            ci=ci,
            alpha=alpha,
            n_obs=int(len(samp)),
            detail=None,
            model_info={
                "did_term": did_col,
                "weight": weight,
                "on_support": on_support,
                "formula": formula,
                "n_units_matched": int(samp[id].nunique()),
                "weight_column": psm_weight_col if weight == "fweight" else None,
                "support_column": psm_support_col,
                "feols_result": fres,
            },
            _citation_key="matching",
        )
        return out

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> SummaryText:
        """Stata-style text summary of the matched ATT."""
        mi = self.result.model_info or {}
        se_method = mi.get("se_method", "ai")
        if se_method == "abadie_imbens":
            se_method = f"AI-robust({mi.get('ai_matches', 1)})"
        if self.method in ("kernel", "radius"):
            design = (
                f"  Kernel            : {mi.get('kernel', '?')}"
                f"  (bwidth: {mi.get('bwidth', '?')})"
            )
            cols = "_pscore _treated _support _weight _y"
        else:
            design = f"  Neighbours (k)    : {self.n_matches}"
            cols = "_pscore _treated _support _weight _n1 through _nn _pdif _y"
        lines = [
            "Propensity Score Matching (psmatch2-style)",
            "=" * 62,
            f"  Outcome           : {self.outcome}",
            f"  Treatment         : {self.treat}",
            f"  Covariates        : {', '.join(self.covariates)}",
            f"  Method            : {self.method}",
            design,
            f"  Common support    : {self.common_support}",
            "-" * 62,
            f"  Treated           : {self.n_treated}"
            f"  (on support: {self.n_on_support},"
            f" matched: {self.n_matched_treated})",
            f"  Control           : {self.n_control}",
            "-" * 62,
            f"  ATT               : {self.att:.4f}",
            f"  Std. err. ({se_method:<8}): {self.se:.4f}",
            f"  p-value           : {self.pvalue:.4f}",
            f"  {int((1-self.alpha)*100)}% CI            : "
            f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}]",
            "=" * 62,
            f"Matched-sample variables on .matched_data: {cols}",
        ]
        return SummaryText("\n".join(lines))

    def cite(self, format: str = "bibtex") -> Any:
        """Citation for the matching estimator (delegates to the result)."""
        return self.result.cite(format=format)

    def __repr__(self) -> str:
        return self.summary()

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><td style='text-align:left'>{k}</td>"
            f"<td style='text-align:right'>{v}</td></tr>"
            for k, v in [
                ("ATT", f"{self.att:.4f}"),
                ("Std. err.", f"{self.se:.4f}"),
                ("p-value", f"{self.pvalue:.4f}"),
                (
                    f"{int((1-self.alpha)*100)}% CI",
                    f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}]",
                ),
                ("Treated (matched)", f"{self.n_matched_treated} / {self.n_treated}"),
                ("Neighbours (k)", self.n_matches),
            ]
        )
        return (
            "<h4>Propensity Score Matching (psmatch2-style)</h4>"
            f"<table>{rows}</table>"
            "<p style='color:#888'>matched-sample variables on "
            "<code>.matched_data</code></p>"
        )


def _fresh_column(base: str, used: set) -> str:
    """Return a column name not present in *used*."""
    if base not in used:
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


# ======================================================================
# Public function
# ======================================================================


def psmatch2(
    data: pd.DataFrame,
    *,
    treat: Optional[str] = None,
    covariates: Optional[Union[Sequence[str], str]] = None,
    outcome: Optional[str] = None,
    y: Optional[str] = None,
    neighbor: int = 1,
    n_matches: Optional[int] = None,
    caliper: Optional[float] = None,
    common_support: str = "none",
    method: str = "neighbor",
    kernel: str = "epan",
    bwidth: float = 0.06,
    se: str = "psmatch2",
    ai: int = 0,
    replace: bool = True,
    ps_poly: int = 1,
    distance: str = "propensity",
    alpha: float = 0.05,
) -> PSMatch2Result:
    """Stata ``psmatch2``-faithful supported propensity-score matching.

    Runs nearest-neighbour propensity-score matching and returns a
    :class:`PSMatch2Result` carrying the psmatch2 matched-sample variables
    (``_pscore`` ``_treated`` ``_support`` ``_weight`` ``_y``; plus
    ``_n1`` … ``_nn`` ``_pdif`` for nearest-neighbour matching) plus
    post-matching balance, common-support plotting, and PSM-DID helpers.

    This is the Stata-migration front door over :func:`sp.match`; the pinned
    Stata 18 nearest-neighbour, kernel, and radius paths are covered by
    reference fixtures for the point estimate, analytic SE, and emitted
    matched-frame columns (Leuven & Sianesi 2003).

    Parameters
    ----------
    data : DataFrame
        Cross-section with one row per unit (the matching sample).
    treat : str
        Binary treatment indicator (0/1).  Stata's ``treated``.
    covariates : list of str
        Pre-treatment covariates entering the propensity-score model.
    outcome, y : str, optional
        Outcome variable (``outcome`` mirrors Stata's ``outcome()``; ``y``
        is accepted as an alias).  **Optional**, exactly like Stata: when
        omitted, the matched frame (``_weight`` etc.) is still produced for
        downstream PSM-DID, but the cross-sectional ATT is left ``NaN``.
    neighbor : int, default 1
        Number of nearest neighbours ``k`` (Stata ``neighbor(k)``).
        ``n_matches`` is accepted as an alias.
    caliper : float, optional
        Maximum propensity-score distance for a valid match
        (Stata ``caliper()``).
    common_support : {'none', 'minmax'}, default 'none'
        ``'none'`` matches every treated unit (raw ``psmatch2``).
        ``'minmax'`` drops treated units outside the control PS range
        before matching (Stata ``common``) and the ATT is taken over the
        on-support treated.
    method : {'neighbor', 'kernel', 'radius'}, default 'neighbor'
        Matching algorithm. ``'neighbor'`` is k-nearest-neighbour matching
        (Stata default; uses ``neighbor`` / ``caliper``). ``'kernel'`` is
        kernel matching (uses ``kernel`` + ``bwidth``). ``'radius'`` is
        radius matching (all controls within ``caliper``; a uniform kernel).
    kernel : {'epan', 'normal', 'biweight', 'uniform', 'tricube'}, default 'epan'
        Kernel type for ``method='kernel'`` (Stata ``kerneltype()``).
    bwidth : float, default 0.06
        Kernel bandwidth on the propensity score for ``method='kernel'``
        (Stata ``bwidth()`` default).
    se : {'psmatch2', 'ai', 'abadie_imbens'}, default 'psmatch2'
        Standard-error estimator. ``'psmatch2'`` reproduces Stata's
        homoskedastic analytic ATT SE digit for digit; ``'ai'`` is the simple
        matched-pair SE; ``'abadie_imbens'`` is the Abadie-Imbens (2006)
        heteroskedasticity-robust SE (Stata ``psmatch2 , ai(J)``).
    ai : int, default 0
        Shorthand for the Abadie-Imbens (2006) robust SE with ``J = ai``
        within-arm matches (Stata's ``ai(J)``). Any ``ai > 0`` selects the
        robust SE and overrides ``se``. Reproduces Stata's ``r(seatt)``
        digit for digit.
    replace : bool, default True
        Match with replacement (psmatch2 default).
    ps_poly : int, default 1
        Polynomial degree of the logit propensity-score model.
    distance : str, default 'propensity'
        Matching metric; ``'propensity'`` reproduces psmatch2.
    alpha : float, default 0.05
        Significance level for the ATT confidence interval.

    Returns
    -------
    PSMatch2Result

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> m = sp.psmatch2(df, outcome='log_wage', treat='union',
    ...                 covariates=['education', 'experience', 'tenure'])
    >>> round(float(m.att), 4) == round(float(m.result.estimate), 4)
    True
    >>> sorted(c for c in m.matched_data.columns if c.startswith('_'))[:4]
    ['_id', '_n1', '_nn', '_pdif']

    Post-matching balance (the analogue of Stata ``pstest`` — ``smd_raw`` is
    before matching, ``smd_weighted`` is the matched, ``_weight``-weighted
    sample):

    >>> bal = m.balance()
    >>> {'smd_raw', 'smd_weighted'} <= set(bal.table.columns)
    True
    """
    from .match import match as _match

    if treat is None or covariates is None:
        raise _psmatch2_error(
            "psmatch2 requires treat= and covariates=.",
            diagnostics={
                "has_treat": treat is not None,
                "has_covariates": covariates is not None,
            },
            recovery_hint="Pass treat='d' and covariates=['x1', 'x2'].",
        )
    data = _require_dataframe(data, "psmatch2 data")
    treat = _require_column(data, treat, "treat")
    covariates = _normalize_columns(covariates, "covariates")
    for covariate in covariates:
        _require_column(data, covariate, "covariate")
    out_var = outcome if outcome is not None else y
    if out_var is not None:
        out_var = _require_column(data, out_var, "outcome")
    if out_var is not None and out_var in covariates:
        raise _psmatch2_error(
            f"outcome={out_var!r} is also listed in covariates; pass a "
            f"distinct outcome (or outcome=None to only build _weight).",
            diagnostics={"outcome": out_var, "covariates": covariates},
            recovery_hint="Remove the outcome from covariates or set " "outcome=None.",
        )
    k = n_matches if n_matches is not None else neighbor

    # Map the psmatch2 method names onto sp.match's method space.
    method = str(method).lower()
    _method_map = {
        "neighbor": "psm",
        "nearest": "psm",
        "nn": "psm",
        "kernel": "kernel",
        "radius": "radius",
    }
    if method not in _method_map:
        raise _psmatch2_error(
            "method must be 'neighbor', 'kernel', or 'radius', " f"got {method!r}.",
            diagnostics={"method": method, "valid_methods": list(_method_map)},
            recovery_hint="Use method='neighbor', 'kernel', or 'radius'.",
        )
    match_method = _method_map[method]
    if method == "radius" and not caliper:
        raise _psmatch2_error(
            "method='radius' requires caliper=<radius>.",
            diagnostics={"method": method, "caliper": caliper},
            recovery_hint="Pass a positive caliper for radius matching.",
        )
    se_key = str(se).lower()
    if se_key not in {"psmatch2", "stata", "ai", "abadie_imbens", "ai_robust"}:
        raise _psmatch2_error(
            "se must be 'psmatch2' (or 'stata'), 'ai', or 'abadie_imbens'.",
            diagnostics={"se": se},
            recovery_hint="Use se='psmatch2', se='ai', or " "se='abadie_imbens'.",
        )
    if se_key in ("psmatch2", "stata"):
        se_method = "psmatch2"
    elif se_key in ("abadie_imbens", "ai_robust"):
        se_method = "abadie_imbens"
    else:
        se_method = "ai"
    # ``ai=J`` (Stata's ai(J)) is shorthand for the Abadie-Imbens robust SE
    # with J within-arm matches; it overrides ``se``.
    ai_matches = 1
    if ai and int(ai) > 0:
        se_method = "abadie_imbens"
        ai_matches = int(ai)

    # Stata's outcome() is optional: when omitted we still produce the
    # matched frame (the PSM-DID use case needs only _weight), so match on a
    # synthetic constant outcome and leave the ATT undefined.
    if out_var is None:
        fit_data = data.copy()
        fit_y = "__psmatch2_no_outcome__"
        fit_data[fit_y] = 0.0
    else:
        fit_data = data
        fit_y = out_var

    # Deliberately do PSM (the user asked for psmatch2); silence the generic
    # King & Nielsen (2019) caveat that sp.match raises for PSM — we hand the
    # user the balance diagnostics to judge it themselves.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="PSM can increase imbalance", category=UserWarning
        )
        result = _match(
            data=fit_data,
            y=fit_y,
            treat=treat,
            covariates=covariates,
            method=match_method,
            distance=distance,
            estimand="ATT",
            n_matches=k,
            caliper=caliper,
            replace=replace,
            ps_poly=ps_poly,
            common_support=common_support,
            kernel=kernel,
            bwidth=bwidth,
            se_method=se_method,
            ai_matches=ai_matches,
            alpha=alpha,
        )

    matched = getattr(result, "matched_data", None)
    if matched is None:  # pragma: no cover — nearest path always sets it
        raise _psmatch2_error(
            "psmatch2: matched frame was not produced.",
            diagnostics={"method": method},
            recovery_hint="Report this internal invariant failure with the "
            "matching inputs.",
        )
    model_info = dict(result.model_info or {})
    model_info.update(
        {
            "psmatch2_method": method,
            "propensity_model": "logit",
            "estimand_scope": "ATT",
            "outcome_status": "observed" if out_var is not None else "omitted",
            "att_defined": out_var is not None,
            "matched_frame_semantics": (
                "Stata psmatch2 ATT matched-frame columns; do not reuse as "
                "a generic ATE bookkeeping surface."
            ),
        }
    )
    if out_var is None:
        # Drop the synthetic outcome and its matched-outcome column; the ATT
        # is meaningless without a real outcome.
        matched = matched.drop(columns=[fit_y, _mf.COL_Y], errors="ignore")
        result.estimate = float("nan")
        result.se = float("nan")
        result.pvalue = float("nan")
        result.ci = (float("nan"), float("nan"))
    result.model_info = model_info

    return PSMatch2Result(
        result=result,
        matched_data=matched,
        treat=treat,
        covariates=covariates,
        outcome=out_var,
        n_matches=k,
        common_support=common_support,
        method=method,
    )
