"""
Synthetic Control Analysis Report.

Generates an SCM report combining main estimation,
placebo inference, and sensitivity diagnostics into a single formatted
output (plain text, Markdown, or LaTeX).

Usage
-----
>>> import statspai as sp
>>> report = sp.synth_report(
...     df, outcome='cigsale', unit='state', time='year',
...     treated_unit='California', treatment_time=1989,
... )
>>> print(report)

>>> sp.synth_report_to_file(
...     df, outcome='cigsale', unit='state', time='year',
...     treated_unit='California', treatment_time=1989,
...     filename='scm_report.md', output='markdown',
... )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .scm import synth
from .sensitivity import synth_sensitivity


# ======================================================================
# Method labels
# ======================================================================

_METHOD_LABELS: Dict[str, str] = {
    "classic": "Classic SCM (Abadie, Diamond & Hainmueller, 2010)",
    "penalized": "Penalized / Ridge SCM",
    "ridge": "Penalized / Ridge SCM",
    "demeaned": "De-meaned SCM (Ferman & Pinto, 2021)",
    "detrended": "De-trended SCM (Ferman & Pinto, 2021)",
    "unconstrained": "Unconstrained SCM (Doudchenko & Imbens, 2016)",
    "elastic_net": "Elastic-net SCM (Doudchenko & Imbens, 2016)",
    "augmented": "Augmented SCM (Ben-Michael, Feller & Rothstein, 2021)",
    "ascm": "Augmented SCM (Ben-Michael, Feller & Rothstein, 2021)",
    "sdid": "Synthetic DID (Arkhangelsky et al., 2021)",
    "factor": "Factor / GSynth (Xu, 2017)",
    "gsynth": "Factor / GSynth (Xu, 2017)",
    "staggered": "Staggered SCM (Ben-Michael, Feller & Rothstein, 2022)",
    "mc": "Matrix Completion (Athey et al., 2021)",
    "matrix_completion": "Matrix Completion (Athey et al., 2021)",
    "discos": "Distributional SCM (Gunsilius, 2023)",
    "scpi": "SC Prediction Intervals (Cattaneo, Feng & Titiunik, 2021)",
    "conformal": "Conformal SC (Chernozhukov, Wüthrich & Zhu, 2021)",
    "bayesian": "Bayesian SCM with MCMC posterior",
    "bsts": "Bayesian Structural Time Series (Brodersen et al., 2015)",
    "causal_impact": "Bayesian Structural Time Series (Brodersen et al., 2015)",
    "penscm": "Penalized SCM (Abadie & L'Hour, 2021)",
    "abadie_lhour": "Penalized SCM (Abadie & L'Hour, 2021)",
    "fdid": "Forward DID (Li, 2024)",
    "forward_did": "Forward DID (Li, 2024)",
    "cluster": "Cluster SCM (Rho et al., 2025)",
    "sparse": "Sparse SCM with L1 penalty (Amjad, Shah & Shen, 2018)",
    "lasso": "Sparse SCM with L1 penalty (Amjad, Shah & Shen, 2018)",
    "kernel": "Kernel SCM",
    "kernel_ridge": "Kernel-ridge SCM",
    "multi_outcome": "Multi-outcome SCM (Sun, Ben-Michael & Feller, 2023)",
    "sequential_sdid": "Sequential SDID for staggered adoption",
    "synth_survival": "Synthetic Survival Control (Han & Shah, 2025)",
}

# Method-specific citations. Each entry is the human-readable string used
# in the report's "Citation" section; bibtex equivalents live in
# paper.bib under the listed keys, never duplicated here.
_METHOD_CITATIONS: Dict[str, str] = {
    "classic": (
        "Abadie, A., Diamond, A. and Hainmueller, J. (2010). "
        '"Synthetic Control Methods for Comparative Case Studies: '
        "Estimating the Effect of California's Tobacco Control Program.\" "
        "Journal of the American Statistical Association, 105(490), 493–505. "
        "[@abadie2010synthetic]"
    ),
    "augmented": (
        "Ben-Michael, E., Feller, A. and Rothstein, J. (2021). "
        '"The Augmented Synthetic Control Method." '
        "Journal of the American Statistical Association, 116(536), 1789–1803. "
        "[@benmichael2021augmented]"
    ),
    "ascm": (
        "Ben-Michael, E., Feller, A. and Rothstein, J. (2021). "
        '"The Augmented Synthetic Control Method." '
        "Journal of the American Statistical Association, 116(536), 1789–1803. "
        "[@benmichael2021augmented]"
    ),
    "sdid": (
        "Arkhangelsky, D., Athey, S., Hirshberg, D., Imbens, G. and Wager, S. "
        '(2021). "Synthetic Difference-in-Differences." '
        "American Economic Review, 111(12), 4088–4118. "
        "[@arkhangelsky2021synthetic]"
    ),
    "gsynth": (
        "Xu, Y. (2017). "
        '"Generalized Synthetic Control Method: Causal Inference with '
        "Interactive Fixed Effects Models.\" "
        "Political Analysis, 25(1), 57–76. [@xu2017generalized]"
    ),
    "factor": (
        "Xu, Y. (2017). "
        '"Generalized Synthetic Control Method." '
        "Political Analysis, 25(1), 57–76. [@xu2017generalized]"
    ),
    "staggered": (
        "Ben-Michael, E., Feller, A. and Rothstein, J. (2022). "
        '"Synthetic Controls with Staggered Adoption." '
        "Journal of the Royal Statistical Society Series B, 84(2), 351–381. "
        "[@benmichael2022synthetic]"
    ),
    "mc": (
        "Athey, S., Bayati, M., Doudchenko, N., Imbens, G. and Khosravi, K. "
        '(2021). "Matrix Completion Methods for Causal Panel Data Models." '
        "Journal of the American Statistical Association, 116(536), 1716–1730. "
        "[@athey2021matrix]"
    ),
    "matrix_completion": (
        "Athey, S., Bayati, M., Doudchenko, N., Imbens, G. and Khosravi, K. "
        '(2021). "Matrix Completion Methods for Causal Panel Data Models." '
        "Journal of the American Statistical Association, 116(536), 1716–1730. "
        "[@athey2021matrix]"
    ),
    "discos": (
        "Gunsilius, F.F. (2023). "
        '"Distributional Synthetic Controls." '
        "Econometrica, 91(3). [@gunsilius2023distributional]"
    ),
    "scpi": (
        "Cattaneo, M.D., Feng, Y. and Titiunik, R. (2021). "
        '"Prediction Intervals for Synthetic Control Methods." '
        "Journal of the American Statistical Association, 116(536), 1865–1880. "
        "[@cattaneo2021prediction]"
    ),
    "conformal": (
        "Chernozhukov, V., Wüthrich, K. and Zhu, Y. (2021). "
        '"An Exact and Robust Conformal Inference Method for '
        "Counterfactual and Synthetic Controls.\" "
        "Journal of the American Statistical Association, 116(536), 1849–1864. "
        "[@chernozhukov2021exact]"
    ),
    "demeaned": (
        "Ferman, B. and Pinto, C. (2021). "
        '"Synthetic Controls with Imperfect Pre-Treatment Fit." '
        "Quantitative Economics, 12(4), 1197–1221. [@ferman2021synthetic]"
    ),
    "detrended": (
        "Ferman, B. and Pinto, C. (2021). "
        '"Synthetic Controls with Imperfect Pre-Treatment Fit." '
        "Quantitative Economics, 12(4), 1197–1221. [@ferman2021synthetic]"
    ),
    "unconstrained": (
        "Doudchenko, N. and Imbens, G.W. (2016). "
        '"Balancing, Regression, Difference-in-Differences and Synthetic '
        "Control Methods: A Synthesis.\" NBER Working Paper No. 22791. "
        "[@doudchenko2016balancing]"
    ),
    "elastic_net": (
        "Doudchenko, N. and Imbens, G.W. (2016). "
        '"Balancing, Regression, Difference-in-Differences and Synthetic '
        "Control Methods: A Synthesis.\" NBER Working Paper No. 22791. "
        "[@doudchenko2016balancing]"
    ),
    "penscm": (
        "Abadie, A. and L'Hour, J. (2021). "
        '"A Penalized Synthetic Control Estimator for Disaggregated Data." '
        "Journal of the American Statistical Association, 116(536), 1817–1834. "
        "[@abadie2021penalized]"
    ),
    "abadie_lhour": (
        "Abadie, A. and L'Hour, J. (2021). "
        '"A Penalized Synthetic Control Estimator for Disaggregated Data." '
        "Journal of the American Statistical Association, 116(536), 1817–1834. "
        "[@abadie2021penalized]"
    ),
    "bsts": (
        "Brodersen, K.H., Gallusser, F., Koehler, J., Remy, N. and Scott, S.L. "
        '(2015). "Inferring Causal Impact Using Bayesian Structural Time-Series '
        "Models.\" Annals of Applied Statistics, 9(1), 247–274. "
        "[@brodersen2015inferring]"
    ),
    "causal_impact": (
        "Brodersen, K.H., Gallusser, F., Koehler, J., Remy, N. and Scott, S.L. "
        '(2015). "Inferring Causal Impact Using Bayesian Structural Time-Series '
        "Models.\" Annals of Applied Statistics, 9(1), 247–274. "
        "[@brodersen2015inferring]"
    ),
    "fdid": (
        "Li, K.T. (2024). "
        '"Frontiers: A Simple Forward Difference-in-Differences Method." '
        "[@li2024forward]"
    ),
    "forward_did": (
        "Li, K.T. (2024). "
        '"Frontiers: A Simple Forward Difference-in-Differences Method." '
        "[@li2024forward]"
    ),
    "multi_outcome": (
        "Sun, L., Ben-Michael, E. and Feller, A. (2023). "
        '"Using Multiple Outcomes to Improve the Synthetic Control Method." '
        "[@sun2023multiple]"
    ),
    "cluster": (
        "Rho, S., Tang, A. et al. (2025). "
        '"ClusterSC: Cluster-Aware Synthetic Control." '
        "arXiv:2503.21629. [@rho2025clustersc]"
    ),
    "sparse": (
        "Amjad, M., Shah, D. and Shen, D. (2018). "
        '"Robust Synthetic Control." '
        "Journal of Machine Learning Research, 19(22), 1–51."
    ),
    "lasso": (
        "Amjad, M., Shah, D. and Shen, D. (2018). "
        '"Robust Synthetic Control." '
        "Journal of Machine Learning Research, 19(22), 1–51."
    ),
}

# Backwards-compatible default — used when the report can't infer the method
_CITATION_TEXT = _METHOD_CITATIONS["classic"]

_CITATION_BIBTEX = (
    "@article{abadie2010synthetic,\n"
    "  title={Synthetic Control Methods for Comparative Case Studies: "
    "Estimating the Effect of California's Tobacco Control Program},\n"
    "  author={Abadie, Alberto and Diamond, Alexis and Hainmueller, Jens},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={105},\n"
    "  number={490},\n"
    "  pages={493--505},\n"
    "  year={2010}\n"
    "}"
)


def _citation_for(method: str) -> str:
    """Look up the human-readable citation for a method.

    Falls back to the classic Abadie–Diamond–Hainmueller (2010) reference
    when the variant has no specific entry, so the report always closes
    with a citation rather than a placeholder.
    """
    return _METHOD_CITATIONS.get(method, _CITATION_TEXT)


def _canonicalise_mi(
    result: Any, treated_unit: Any, treatment_time: Any,
) -> Dict[str, Any]:
    """Return a model_info dict normalised to the report's expected schema.

    Different SCM variants store identical concepts under different
    keys. SDID, for example, uses ``T_pre`` / ``T_post`` / ``n_control``
    instead of ``n_pre_periods`` / ``n_post_periods`` / ``n_donors``,
    and exposes ``Y_obs`` rather than building a ``gap_table``. This
    helper backfills the canonical names plus a recomputed
    ``gap_table`` and ``pre_treatment_rmse`` so the existing text /
    Markdown / LaTeX formatters behave uniformly across variants.

    The returned dict is a shallow copy with the canonical keys
    overwritten — callers that need the original schema should keep
    working with ``result.model_info`` directly.
    """
    from .exports import _gap_table as _exp_gap_table  # local import
    from .exports import _pre_rmspe as _exp_pre_rmspe

    raw = result.model_info or {}
    mi: Dict[str, Any] = dict(raw)

    # Treated unit / treatment time
    if "treated_unit" not in mi:
        if treated_unit is not None:
            mi["treated_unit"] = treated_unit
        elif "treated_units" in raw and isinstance(raw["treated_units"], list):
            tu = raw["treated_units"]
            mi["treated_unit"] = tu[0] if len(tu) == 1 else ", ".join(map(str, tu))
    if "treatment_time" not in mi:
        if treatment_time is not None:
            mi["treatment_time"] = treatment_time
        elif "treat_time" in raw:
            mi["treatment_time"] = raw["treat_time"]

    # Period / donor counts
    if "n_pre_periods" not in mi and "T_pre" in raw:
        mi["n_pre_periods"] = raw["T_pre"]
    if "n_post_periods" not in mi and "T_post" in raw:
        mi["n_post_periods"] = raw["T_post"]
    if "n_donors" not in mi and "n_control" in raw:
        mi["n_donors"] = raw["n_control"]

    # Pre-RMSPE (recompute from gap if absent)
    if (
        mi.get("pre_treatment_rmse") is None
        and mi.get("pre_treatment_mspe") is None
    ):
        pre = _exp_pre_rmspe(result)
        if not (isinstance(pre, float) and (pre != pre)):  # not nan
            mi["pre_treatment_rmse"] = pre

    # Gap table (recompute from Y_obs / Y_synth if absent)
    if not isinstance(mi.get("gap_table"), pd.DataFrame):
        gt = _exp_gap_table(result)
        if isinstance(gt, pd.DataFrame):
            mi["gap_table"] = gt

    return mi


# ======================================================================
# Text formatter
# ======================================================================

def _format_text(
    result: Any,
    mi: Dict[str, Any],
    method: str,
    sensitivity: Optional[Dict[str, Any]],
    alpha: float,
) -> str:
    """Build a plain-text report."""
    lines: List[str] = []
    w = 56  # box width

    # Header
    lines.append("\u2550" * w)
    lines.append("  SYNTHETIC CONTROL ANALYSIS REPORT")
    lines.append("\u2550" * w)
    lines.append("")

    # 1. SETUP
    lines.append("1. SETUP")
    lines.append(f"   - Treated unit: {mi.get('treated_unit', 'N/A')}")
    lines.append(f"   - Treatment time: {mi.get('treatment_time', 'N/A')}")
    lines.append(f"   - Method: {_METHOD_LABELS.get(method, method)}")
    lines.append(f"   - Pre-treatment periods: {mi.get('n_pre_periods', 'N/A')}")
    lines.append(f"   - Post-treatment periods: {mi.get('n_post_periods', 'N/A')}")
    lines.append(f"   - Donor pool: {mi.get('n_donors', 'N/A')} units")
    lines.append("")

    # 2. MAIN RESULTS
    lines.append("2. MAIN RESULTS")
    ci = result.ci if hasattr(result, "ci") else (np.nan, np.nan)
    ci_lo, ci_hi = ci if ci is not None else (np.nan, np.nan)
    pct = int((1 - alpha) * 100)
    box = [
        f"   \u250c{'─' * 38}\u2510",
        f"   \u2502 {'ATT Estimate:':<20s} {result.estimate:>12.4f}   \u2502",
        f"   \u2502 {'Standard Error:':<20s} {result.se:>12.4f}   \u2502",
        f"   \u2502 {'P-value:':<20s} {result.pvalue:>12.4f}   \u2502",
        f"   \u2502 {f'{pct}% CI:':<20s} [{ci_lo:.4f}, {ci_hi:.4f}]"
        + " " * max(0, 10 - len(f"[{ci_lo:.4f}, {ci_hi:.4f}]")) + "  \u2502",
        f"   \u2514{'─' * 38}\u2518",
    ]
    lines.extend(box)
    lines.append("")

    # 3. PRE-TREATMENT FIT
    pre_rmse = mi.get("pre_treatment_rmse", np.nan)
    lines.append("3. PRE-TREATMENT FIT")
    lines.append(f"   - Pre-RMSPE: {pre_rmse:.6f}")
    gap_table = mi.get("gap_table")
    if gap_table is not None and isinstance(gap_table, pd.DataFrame):
        pre_rows = gap_table[~gap_table["post_treatment"]]
        if len(pre_rows) > 0:
            outcome_sd = pre_rows["treated"].std()
            if outcome_sd > 1e-10:
                pct_sd = (pre_rmse / outcome_sd) * 100
                if pct_sd < 5:
                    quality = "Excellent"
                elif pct_sd < 10:
                    quality = "Good"
                elif pct_sd < 20:
                    quality = "Acceptable"
                else:
                    quality = "Poor"
                lines.append(
                    f"   - Fit quality: {quality} "
                    f"({pct_sd:.1f}% of outcome SD)"
                )
    lines.append("")

    # 4. DONOR WEIGHTS
    weights_df = mi.get("weights")
    if weights_df is not None and isinstance(weights_df, pd.DataFrame):
        lines.append("4. DONOR WEIGHTS")
        lines.append(f"   {'Unit':<25s} {'Weight':>8s}")
        lines.append(f"   {'─' * 35}")
        for _, row in weights_df.head(15).iterrows():
            unit_name = str(row.get("unit", row.iloc[0]))
            w_val = row["weight"]
            lines.append(f"   {unit_name:<25s} {w_val:>8.3f}")
        if len(weights_df) > 15:
            lines.append(f"   ... ({len(weights_df) - 15} more with w > 0)")
        lines.append("")

    # 5. TREATMENT EFFECTS BY PERIOD
    if gap_table is not None and isinstance(gap_table, pd.DataFrame):
        post_rows = gap_table[gap_table["post_treatment"]]
        if len(post_rows) > 0:
            lines.append("5. TREATMENT EFFECTS BY PERIOD")
            lines.append(
                f"   {'Time':>8s}    {'Treated':>10s}    "
                f"{'Synthetic':>10s}    {'Gap':>10s}"
            )
            lines.append(f"   {'─' * 48}")
            display_rows = post_rows.head(20)
            for _, row in display_rows.iterrows():
                lines.append(
                    f"   {str(row['time']):>8s}    {row['treated']:>10.2f}    "
                    f"{row['synthetic']:>10.2f}    {row['gap']:>10.2f}"
                )
            if len(post_rows) > 20:
                lines.append(f"   ... ({len(post_rows) - 20} more periods)")
            lines.append("")

    # 6. PLACEBO INFERENCE
    placebo_atts = mi.get("placebo_atts")
    if placebo_atts is not None:
        n_plac = mi.get("n_placebos", len(placebo_atts))
        total_units = n_plac + 1
        # Rank: how many placebos have |ATT| >= |treated ATT|
        treated_att_abs = abs(result.estimate)
        rank = int(np.sum(np.abs(placebo_atts) >= treated_att_abs)) + 1
        ratio_treated = mi.get("treated_ratio", np.nan)

        lines.append("6. PLACEBO INFERENCE")
        lines.append(f"   - Placebos run: {n_plac}")
        lines.append(f"   - Treated unit rank: {rank}/{total_units}")
        if not np.isnan(ratio_treated):
            lines.append(f"   - Post/Pre RMSPE ratio: {ratio_treated:.2f}")
        lines.append(f"   - P-value (ratio test): {result.pvalue:.4f}")
        lines.append("")

    # 7. SENSITIVITY ANALYSIS
    if sensitivity is not None:
        lines.append("7. SENSITIVITY ANALYSIS")

        # a) Leave-one-out
        loo_df = sensitivity.get("loo")
        if loo_df is not None and len(loo_df) > 0:
            lines.append(
                f"   a) Leave-one-out: ATT range "
                f"[{loo_df['att'].min():.4f}, {loo_df['att'].max():.4f}]"
            )
        else:
            lines.append("   a) Leave-one-out: N/A")

        # b) Time placebos
        tp_df = sensitivity.get("time_placebo")
        if tp_df is not None and len(tp_df) > 0:
            n_sig = int((tp_df["pvalue"] < alpha).sum())
            if n_sig == 0:
                lines.append(
                    "   b) Time placebos: No significant "
                    "pre-treatment effects"
                )
            else:
                lines.append(
                    f"   b) Time placebos: {n_sig}/{len(tp_df)} "
                    f"significant at {alpha:.0%}"
                )
        else:
            lines.append("   b) Time placebos: N/A")

        # c) Donor pool sensitivity
        ds_df = sensitivity.get("donor_sensitivity")
        if ds_df is not None and len(ds_df) > 0:
            q025 = ds_df["att"].quantile(0.025)
            q975 = ds_df["att"].quantile(0.975)
            lines.append(
                f"   c) Donor pool: ATT 95% range "
                f"[{q025:.4f}, {q975:.4f}]"
            )
        else:
            lines.append("   c) Donor pool: N/A")

        # d) RMSPE filter
        rp_df = sensitivity.get("rmspe_filter")
        if rp_df is not None and len(rp_df) > 0:
            lines.append("   d) RMSPE-filtered p-values:")
            for _, row in rp_df.iterrows():
                thr = row["threshold"]
                thr_label = f"{thr:.0f}x" if np.isfinite(thr) else "all"
                lines.append(
                    f"      {thr_label:>5s}: p = {row['pvalue']:.3f} "
                    f"(n = {int(row['n_placebos'])})"
                )
        lines.append("")

    # 8. CITATION
    section_num = 8 if sensitivity is not None else 7
    lines.append(f"{section_num}. CITATION")
    lines.append(f"   {_citation_for(method)}")
    lines.append("")
    lines.append("\u2550" * w)

    return "\n".join(lines)


# ======================================================================
# Markdown formatter
# ======================================================================

def _format_markdown(
    result: Any,
    mi: Dict[str, Any],
    method: str,
    sensitivity: Optional[Dict[str, Any]],
    alpha: float,
) -> str:
    """Build a Markdown report."""
    lines: List[str] = []

    lines.append("# Synthetic Control Analysis Report")
    lines.append("")

    # 1. Setup
    lines.append("## 1. Setup")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|:----------|:------|")
    lines.append(f"| **Treated unit** | {mi.get('treated_unit', 'N/A')} |")
    lines.append(f"| **Treatment time** | {mi.get('treatment_time', 'N/A')} |")
    lines.append(f"| **Method** | {_METHOD_LABELS.get(method, method)} |")
    lines.append(f"| **Pre-treatment periods** | {mi.get('n_pre_periods', 'N/A')} |")
    lines.append(f"| **Post-treatment periods** | {mi.get('n_post_periods', 'N/A')} |")
    lines.append(f"| **Donor pool** | {mi.get('n_donors', 'N/A')} units |")
    lines.append("")

    # 2. Main Results
    ci = result.ci if hasattr(result, "ci") else (np.nan, np.nan)
    ci_lo, ci_hi = ci if ci is not None else (np.nan, np.nan)
    pct = int((1 - alpha) * 100)

    lines.append("## 2. Main Results")
    lines.append("")
    lines.append(f"| Statistic | Value |")
    lines.append(f"|:----------|------:|")
    lines.append(f"| **ATT Estimate** | {result.estimate:.4f} |")
    lines.append(f"| **Standard Error** | {result.se:.4f} |")
    lines.append(f"| **P-value** | {result.pvalue:.4f} |")
    lines.append(f"| **{pct}% CI** | [{ci_lo:.4f}, {ci_hi:.4f}] |")
    lines.append("")

    # 3. Pre-Treatment Fit
    pre_rmse = mi.get("pre_treatment_rmse", np.nan)
    lines.append("## 3. Pre-Treatment Fit")
    lines.append("")
    lines.append(f"- **Pre-RMSPE:** {pre_rmse:.6f}")
    gap_table = mi.get("gap_table")
    if gap_table is not None and isinstance(gap_table, pd.DataFrame):
        pre_rows = gap_table[~gap_table["post_treatment"]]
        if len(pre_rows) > 0:
            outcome_sd = pre_rows["treated"].std()
            if outcome_sd > 1e-10:
                pct_sd = (pre_rmse / outcome_sd) * 100
                if pct_sd < 5:
                    quality = "Excellent"
                elif pct_sd < 10:
                    quality = "Good"
                elif pct_sd < 20:
                    quality = "Acceptable"
                else:
                    quality = "Poor"
                lines.append(
                    f"- **Fit quality:** {quality} "
                    f"({pct_sd:.1f}% of outcome SD)"
                )
    lines.append("")

    # 4. Donor Weights
    weights_df = mi.get("weights")
    if weights_df is not None and isinstance(weights_df, pd.DataFrame):
        lines.append("## 4. Donor Weights")
        lines.append("")
        lines.append("| Unit | Weight |")
        lines.append("|:-----|-------:|")
        for _, row in weights_df.head(15).iterrows():
            unit_name = str(row.get("unit", row.iloc[0]))
            w_val = row["weight"]
            lines.append(f"| {unit_name} | {w_val:.3f} |")
        if len(weights_df) > 15:
            lines.append(f"| *... {len(weights_df) - 15} more* | |")
        lines.append("")

    # 5. Treatment Effects by Period
    if gap_table is not None and isinstance(gap_table, pd.DataFrame):
        post_rows = gap_table[gap_table["post_treatment"]]
        if len(post_rows) > 0:
            lines.append("## 5. Treatment Effects by Period")
            lines.append("")
            lines.append("| Time | Treated | Synthetic | Gap |")
            lines.append("|-----:|--------:|----------:|----:|")
            display_rows = post_rows.head(20)
            for _, row in display_rows.iterrows():
                lines.append(
                    f"| {row['time']} | {row['treated']:.2f} "
                    f"| {row['synthetic']:.2f} | {row['gap']:.2f} |"
                )
            if len(post_rows) > 20:
                lines.append(f"| *... {len(post_rows) - 20} more* | | | |")
            lines.append("")

    # 6. Placebo Inference
    placebo_atts = mi.get("placebo_atts")
    if placebo_atts is not None:
        n_plac = mi.get("n_placebos", len(placebo_atts))
        total_units = n_plac + 1
        treated_att_abs = abs(result.estimate)
        rank = int(np.sum(np.abs(placebo_atts) >= treated_att_abs)) + 1
        ratio_treated = mi.get("treated_ratio", np.nan)

        lines.append("## 6. Placebo Inference")
        lines.append("")
        lines.append(f"- **Placebos run:** {n_plac}")
        lines.append(f"- **Treated unit rank:** {rank}/{total_units}")
        if not np.isnan(ratio_treated):
            lines.append(
                f"- **Post/Pre RMSPE ratio:** {ratio_treated:.2f}"
            )
        lines.append(f"- **P-value (ratio test):** {result.pvalue:.4f}")
        lines.append("")

    # 7. Sensitivity
    if sensitivity is not None:
        lines.append("## 7. Sensitivity Analysis")
        lines.append("")

        loo_df = sensitivity.get("loo")
        if loo_df is not None and len(loo_df) > 0:
            lines.append(
                f"**a) Leave-one-out:** ATT range "
                f"[{loo_df['att'].min():.4f}, {loo_df['att'].max():.4f}]"
            )
            lines.append("")

        tp_df = sensitivity.get("time_placebo")
        if tp_df is not None and len(tp_df) > 0:
            n_sig = int((tp_df["pvalue"] < alpha).sum())
            if n_sig == 0:
                lines.append(
                    "**b) Time placebos:** No significant "
                    "pre-treatment effects"
                )
            else:
                lines.append(
                    f"**b) Time placebos:** {n_sig}/{len(tp_df)} "
                    f"significant at {alpha:.0%}"
                )
            lines.append("")

        ds_df = sensitivity.get("donor_sensitivity")
        if ds_df is not None and len(ds_df) > 0:
            q025 = ds_df["att"].quantile(0.025)
            q975 = ds_df["att"].quantile(0.975)
            lines.append(
                f"**c) Donor pool:** ATT 95% range "
                f"[{q025:.4f}, {q975:.4f}]"
            )
            lines.append("")

        rp_df = sensitivity.get("rmspe_filter")
        if rp_df is not None and len(rp_df) > 0:
            lines.append("**d) RMSPE-filtered p-values:**")
            lines.append("")
            lines.append("| Threshold | P-value | N placebos |")
            lines.append("|----------:|--------:|-----------:|")
            for _, row in rp_df.iterrows():
                thr = row["threshold"]
                thr_label = f"{thr:.0f}x" if np.isfinite(thr) else "all"
                lines.append(
                    f"| {thr_label} | {row['pvalue']:.3f} "
                    f"| {int(row['n_placebos'])} |"
                )
            lines.append("")

    # Citation
    section_num = 8 if sensitivity is not None else 7
    lines.append(f"## {section_num}. Citation")
    lines.append("")
    lines.append(f"> {_citation_for(method)}")
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by StatsPAI synth_report()*")

    return "\n".join(lines)


# ======================================================================
# LaTeX formatter
# ======================================================================

def _format_latex(
    result: Any,
    mi: Dict[str, Any],
    method: str,
    sensitivity: Optional[Dict[str, Any]],
    alpha: float,
) -> str:
    """Build a LaTeX report."""
    lines: List[str] = []

    lines.append(r"\section*{Synthetic Control Analysis Report}")
    lines.append("")

    # 1. Setup
    lines.append(r"\subsection*{1. Setup}")
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\hline")
    lines.append(
        r"Treated unit & "
        + _latex_escape(str(mi.get("treated_unit", "N/A")))
        + r" \\"
    )
    lines.append(
        r"Treatment time & "
        + _latex_escape(str(mi.get("treatment_time", "N/A")))
        + r" \\"
    )
    lines.append(
        r"Method & "
        + _latex_escape(_METHOD_LABELS.get(method, method))
        + r" \\"
    )
    lines.append(
        f"Pre-treatment periods & {mi.get('n_pre_periods', 'N/A')}"
        + r" \\"
    )
    lines.append(
        f"Post-treatment periods & {mi.get('n_post_periods', 'N/A')}"
        + r" \\"
    )
    lines.append(
        f"Donor pool & {mi.get('n_donors', 'N/A')} units"
        + r" \\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("")

    # 2. Main Results
    ci = result.ci if hasattr(result, "ci") else (np.nan, np.nan)
    ci_lo, ci_hi = ci if ci is not None else (np.nan, np.nan)
    pct = int((1 - alpha) * 100)

    lines.append(r"\subsection*{2. Main Results}")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\hline")
    lines.append(f"ATT Estimate & {result.estimate:.4f}" + r" \\")
    lines.append(f"Standard Error & {result.se:.4f}" + r" \\")
    lines.append(f"P-value & {result.pvalue:.4f}" + r" \\")
    lines.append(
        f"{pct}\\% CI & [{ci_lo:.4f}, {ci_hi:.4f}]"
        + r" \\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append("")

    # 3. Pre-Treatment Fit
    pre_rmse = mi.get("pre_treatment_rmse", np.nan)
    lines.append(r"\subsection*{3. Pre-Treatment Fit}")
    lines.append(f"Pre-RMSPE: {pre_rmse:.6f}")

    gap_table = mi.get("gap_table")
    if gap_table is not None and isinstance(gap_table, pd.DataFrame):
        pre_rows = gap_table[~gap_table["post_treatment"]]
        if len(pre_rows) > 0:
            outcome_sd = pre_rows["treated"].std()
            if outcome_sd > 1e-10:
                pct_sd = (pre_rmse / outcome_sd) * 100
                if pct_sd < 5:
                    quality = "Excellent"
                elif pct_sd < 10:
                    quality = "Good"
                elif pct_sd < 20:
                    quality = "Acceptable"
                else:
                    quality = "Poor"
                lines.append(
                    f"\\\\Fit quality: {quality} "
                    f"({pct_sd:.1f}\\% of outcome SD)"
                )
    lines.append("")

    # 4. Donor Weights
    weights_df = mi.get("weights")
    if weights_df is not None and isinstance(weights_df, pd.DataFrame):
        lines.append(r"\subsection*{4. Donor Weights}")
        lines.append(r"\begin{tabular}{lr}")
        lines.append(r"\hline")
        lines.append(r"\textbf{Unit} & \textbf{Weight} \\")
        lines.append(r"\hline")
        for _, row in weights_df.head(15).iterrows():
            unit_name = _latex_escape(str(row.get("unit", row.iloc[0])))
            w_val = row["weight"]
            lines.append(f"{unit_name} & {w_val:.3f}" + r" \\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        lines.append("")

    # 5. Treatment Effects by Period
    if gap_table is not None and isinstance(gap_table, pd.DataFrame):
        post_rows = gap_table[gap_table["post_treatment"]]
        if len(post_rows) > 0:
            lines.append(r"\subsection*{5. Treatment Effects by Period}")
            lines.append(r"\begin{tabular}{rrrr}")
            lines.append(r"\hline")
            lines.append(
                r"\textbf{Time} & \textbf{Treated} & "
                r"\textbf{Synthetic} & \textbf{Gap} \\"
            )
            lines.append(r"\hline")
            display_rows = post_rows.head(20)
            for _, row in display_rows.iterrows():
                lines.append(
                    f"{row['time']} & {row['treated']:.2f} & "
                    f"{row['synthetic']:.2f} & {row['gap']:.2f}"
                    + r" \\"
                )
            lines.append(r"\hline")
            lines.append(r"\end{tabular}")
            lines.append("")

    # 6. Placebo Inference
    placebo_atts = mi.get("placebo_atts")
    if placebo_atts is not None:
        n_plac = mi.get("n_placebos", len(placebo_atts))
        total_units = n_plac + 1
        treated_att_abs = abs(result.estimate)
        rank = int(np.sum(np.abs(placebo_atts) >= treated_att_abs)) + 1
        ratio_treated = mi.get("treated_ratio", np.nan)

        lines.append(r"\subsection*{6. Placebo Inference}")
        lines.append(r"\begin{itemize}")
        lines.append(f"  \\item Placebos run: {n_plac}")
        lines.append(f"  \\item Treated unit rank: {rank}/{total_units}")
        if not np.isnan(ratio_treated):
            lines.append(
                f"  \\item Post/Pre RMSPE ratio: {ratio_treated:.2f}"
            )
        lines.append(
            f"  \\item P-value (ratio test): {result.pvalue:.4f}"
        )
        lines.append(r"\end{itemize}")
        lines.append("")

    # 7. Sensitivity
    if sensitivity is not None:
        lines.append(r"\subsection*{7. Sensitivity Analysis}")

        loo_df = sensitivity.get("loo")
        if loo_df is not None and len(loo_df) > 0:
            lines.append(
                f"\\textbf{{a) Leave-one-out:}} ATT range "
                f"[{loo_df['att'].min():.4f}, {loo_df['att'].max():.4f}]"
            )
            lines.append("")

        tp_df = sensitivity.get("time_placebo")
        if tp_df is not None and len(tp_df) > 0:
            n_sig = int((tp_df["pvalue"] < alpha).sum())
            if n_sig == 0:
                lines.append(
                    r"\textbf{b) Time placebos:} No significant "
                    "pre-treatment effects"
                )
            else:
                lines.append(
                    f"\\textbf{{b) Time placebos:}} {n_sig}/{len(tp_df)} "
                    f"significant at {alpha:.0%}"
                )
            lines.append("")

        ds_df = sensitivity.get("donor_sensitivity")
        if ds_df is not None and len(ds_df) > 0:
            q025 = ds_df["att"].quantile(0.025)
            q975 = ds_df["att"].quantile(0.975)
            lines.append(
                f"\\textbf{{c) Donor pool:}} ATT 95\\% range "
                f"[{q025:.4f}, {q975:.4f}]"
            )
            lines.append("")

        rp_df = sensitivity.get("rmspe_filter")
        if rp_df is not None and len(rp_df) > 0:
            lines.append(r"\textbf{d) RMSPE-filtered p-values:}")
            lines.append("")
            lines.append(r"\begin{tabular}{rrr}")
            lines.append(r"\hline")
            lines.append(
                r"\textbf{Threshold} & \textbf{P-value} & "
                r"\textbf{N placebos} \\"
            )
            lines.append(r"\hline")
            for _, row in rp_df.iterrows():
                thr = row["threshold"]
                thr_label = (
                    f"{thr:.0f}$\\times$" if np.isfinite(thr) else "all"
                )
                lines.append(
                    f"{thr_label} & {row['pvalue']:.3f} & "
                    f"{int(row['n_placebos'])}"
                    + r" \\"
                )
            lines.append(r"\hline")
            lines.append(r"\end{tabular}")
            lines.append("")

    # Citation
    section_num = 8 if sensitivity is not None else 7
    lines.append(f"\\subsection*{{{section_num}. Citation}}")
    lines.append("")
    lines.append(_CITATION_BIBTEX)
    lines.append("")

    return "\n".join(lines)


# ======================================================================
# Helpers
# ======================================================================

def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


# ======================================================================
# Public API
# ======================================================================

def synth_report(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any = None,
    treatment_time: Any = None,
    method: str = "classic",
    output: str = "text",
    sensitivity: bool = True,
    alpha: float = 0.05,
    **kwargs,
) -> str:
    """
    Generate a comprehensive Synthetic Control analysis report.

    Runs ``synth()`` for the main estimation and optionally
    ``synth_sensitivity()`` for robustness diagnostics, then formats
    everything into a structured report.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any, optional
        Identifier of the treated unit.
    treatment_time : any, optional
        First treatment period (inclusive).
    method : str, default 'classic'
        SCM variant passed to ``synth()``.
    output : str, default 'text'
        Output format: ``'text'``, ``'markdown'``, or ``'latex'``.
    sensitivity : bool, default True
        Whether to include the sensitivity analysis section.
    alpha : float, default 0.05
        Significance level for CIs and hypothesis tests.
    **kwargs
        Additional keyword arguments forwarded to ``synth()``.

    Returns
    -------
    str
        Formatted analysis report.

    Examples
    --------
    >>> import statspai as sp
    >>> report = sp.synth_report(
    ...     df, outcome='cigsale', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ... )
    >>> print(report)

    >>> md = sp.synth_report(
    ...     df, outcome='cigsale', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     output='markdown',
    ... )
    """
    if output not in ("text", "markdown", "latex"):
        raise ValueError(
            f"output must be 'text', 'markdown', or 'latex', got {output!r}"
        )

    # --- Main estimation ---
    result = synth(
        data=data,
        outcome=outcome,
        unit=unit,
        time=time,
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        method=method,
        alpha=alpha,
        **kwargs,
    )

    mi = _canonicalise_mi(result, treated_unit, treatment_time)

    # --- Sensitivity (optional) ---
    sens_result: Optional[Dict[str, Any]] = None
    if sensitivity:
        # Extract params relevant to sensitivity
        penalization = kwargs.get("penalization", 0.0)
        n_donor_samples = kwargs.pop("n_donor_samples", 100)
        seed = kwargs.pop("sensitivity_seed", None)

        try:
            sens_result = synth_sensitivity(
                data=data,
                outcome=outcome,
                unit=unit,
                time=time,
                treated_unit=treated_unit,
                treatment_time=treatment_time,
                penalization=penalization,
                n_donor_samples=n_donor_samples,
                seed=seed,
                alpha=alpha,
            )
        except Exception:  # pragma: no cover
            # Sensitivity is best-effort; do not fail the whole report
            sens_result = None

    # --- Format ---
    if output == "text":
        return _format_text(result, mi, method, sens_result, alpha)
    elif output == "markdown":
        return _format_markdown(result, mi, method, sens_result, alpha)
    else:
        return _format_latex(result, mi, method, sens_result, alpha)


def synth_report_to_file(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any = None,
    treatment_time: Any = None,
    method: str = "classic",
    output: str = "markdown",
    sensitivity: bool = True,
    alpha: float = 0.05,
    filename: str = "report.md",
    **kwargs,
) -> str:
    """
    Generate an SCM report and write it directly to a file.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any, optional
        Identifier of the treated unit.
    treatment_time : any, optional
        First treatment period (inclusive).
    method : str, default 'classic'
        SCM variant passed to ``synth()``.
    output : str, default 'markdown'
        Output format: ``'text'``, ``'markdown'``, or ``'latex'``.
    sensitivity : bool, default True
        Whether to include the sensitivity analysis section.
    alpha : float, default 0.05
        Significance level.
    filename : str, default 'report.md'
        Output file path.
    **kwargs
        Additional keyword arguments forwarded to ``synth()``.

    Returns
    -------
    str
        The generated report string (also written to *filename*).

    Examples
    --------
    >>> import statspai as sp
    >>> sp.synth_report_to_file(
    ...     df, outcome='cigsale', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ...     filename='california_scm.md',
    ... )
    """
    report = synth_report(
        data=data,
        outcome=outcome,
        unit=unit,
        time=time,
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        method=method,
        output=output,
        sensitivity=sensitivity,
        alpha=alpha,
        **kwargs,
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    return report
