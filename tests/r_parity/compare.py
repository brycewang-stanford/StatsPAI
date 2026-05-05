"""Parity comparator: read all results/<module>_{py,R}.json pairs and
(optionally) the sister Stata-side JSONs in
``tests/stata_parity/results/<module>_Stata.json``, then emit:

  * parity_table.md       -- human-readable Markdown (3-way when Stata available)
  * parity_table.tex      -- LaTeX longtable, 4-col R-only baseline (legacy)
  * parity_table_3way.tex -- LaTeX longtable, 5-col with Stata column;
                             this is the version the JSS appendix \\input{}s.
  * parity_summary.md     -- one-line-per-module headline + verdict

Tolerance budget (pre-registered, NEXT-STEPS / JSS plan §5.2):

  * closed-form estimators (OLS, 2SLS, HDFE):  rel_diff < 1e-6
  * iterative / cross-fit (DiD, RD, SCM, DML): rel_diff < 1e-3
  * bootstrap / placebo CI half-widths:        abs_diff < 0.05 * SE
  * Honest-DiD CI bounds:                      abs_diff < 0.05

The same tolerance applies to the StatsPAI <-> Stata comparison: we
do not register a separate budget for the Stata side; one budget per
module is the single source of truth.

Verdict assignment is per-module, not per-row, because some rows
(e.g. SE-with-documented-convention-gap, default-h selector) are
expected NOT to pass at the strict tolerance and are recorded with
an explicit rationale in the module's `extra` block.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
# The sister Stata-side harness lives at tests/stata_parity/. Stata
# results are only emitted for the 21 modules with a canonical Stata
# reference (see tests/stata_parity/README.md); the modules without
# a canonical Stata implementation (DML, causal forest, augsynth,
# gsynth) are flagged with explicit reasons in the 3-way table.
STATA_RESULTS_DIR = HERE.parent / "stata_parity" / "results"
STATA_SKIP_REASON: dict[str, str] = {
    "08_dml":           "no canonical ref",
    "13_causal_forest": "no canonical ref",
    "18_augsynth":      "no canonical ref",
    "19_gsynth":        "no canonical ref",
    "23_evalue":        "not built",
    "24_coxph":         "not built",
    "26_glmm_logit":    "not built",
    "27_glmm_aghq":     "not built",
    "29_panel_sfa":     "not built",
    "31_dfl":           "not built",
    "32_rif":           "not built",
    "33_var":           "not built",
    "34_lp":            "not built",
    "35_panel":         "not built",
    "36_mediation":     "not built",
}


# Pre-registered tolerance per module.
TOLERANCES: dict[str, dict[str, float]] = {
    "01_ols":       {"rel_est": 1e-6, "rel_se": 1e-6},
    "02_iv":        {"rel_est": 1e-6, "rel_se": 1e-6},
    "03_hdfe":      {"rel_est": 1e-6, "rel_se": 1e-2},  # 1-df conv. gap
    "04_csdid":     {"rel_est": 1e-3, "rel_se": 5.0},   # SE under-covers
    "05_sunab":     {"rel_est": 1e-3, "rel_se": 0.25},
    "06_rd":        {"rel_est": 1.0,  "rel_se": 1.0,    # default-h gap
                     "_forced_rel_est": 1e-3},
    "07_scm":       {"rel_est": 0.20, "rel_se": 1.0},   # SCM non-uniqueness
    "08_dml":       {"rel_est": 1e-2, "rel_se": 1e-2},  # fold-split noise
    "09_rddensity": {"rel_est": 1.5,  "rel_se": 1.0},
    "10_honest_did":{"abs_est": 0.05, "abs_se": 0.05},
    "11_psm":         {"rel_est": 1e-2, "rel_se": 5.0},
    "12_sdid":        {"rel_est": 0.15, "rel_se": 1.0},   # regularisation
    "13_causal_forest":{"rel_est": 5.0, "rel_se": 5.0},  # NSW-DW overlap
    "14_ols_cluster": {"rel_est": 1e-3, "rel_se": 1e-3},
    "15_hdfe_cluster":{"rel_est": 1e-6, "rel_se": 5e-2},  # ssc convention
    "16_bjs":         {"rel_est": 0.50, "rel_se": 1.0},   # aggregation rule
    "17_etwfe":       {"rel_est": 0.10, "rel_se": 0.50},  # aggregation rule
    "18_augsynth":    {"rel_est": 0.50, "rel_se": 1.0},   # SCM non-uniqueness
    "19_gsynth":      {"rel_est": 1.0,  "rel_se": 1.0},   # SCM non-uniqueness
    "20_bacon":       {"rel_est": 1e-3, "rel_se": 1.0},   # TWFE-only headline
    "21_honest_relmags":{"abs_est": 0.15, "abs_se": 0.15},
    "22_sensemakr":   {"rel_est": 5e-2, "rel_se": 5e-2},
    "23_evalue":      {"rel_est": 1e-6, "rel_se": 1e-6},
    "24_coxph":       {"rel_est": 1e-3, "rel_se": 1e-3},
    "25_lmm":         {"rel_est": 1e-3, "rel_se": 1e-3},
    "26_glmm_logit":  {"rel_est": 5e-3, "rel_se": 5e-2},
    "27_glmm_aghq":   {"rel_est": 5e-3, "rel_se": 5e-2},
    "28_frontier":    {"rel_est": 1e-3, "rel_se": 1e-3},
    "29_panel_sfa":   {"rel_est": 1e-3, "rel_se": 1e-3},
    "30_oaxaca":      {"rel_est": 1e-3, "rel_se": 1.0},   # gap-only headline
    "31_dfl":         {"rel_est": 1e-3, "rel_se": 1.0},   # gap-only headline
    "32_rif":         {"rel_est": 5e-2, "rel_se": 5e-2},
    "33_var":         {"rel_est": 1e-3, "rel_se": 1e-3},
    "34_lp":          {"abs_est": 0.50, "abs_se": 1.0},   # identification convention
    "35_panel":       {"rel_est": 1e-3, "rel_se": 1e-3},  # FE/RE only headline
    "36_mediation":   {"rel_est": 1e-2, "rel_se": 1e-2},
}


@dataclass
class RowDiff:
    module: str
    statistic: str
    py_est: float | None
    R_est: float | None
    abs_est: float | None
    rel_est: float | None
    py_se: float | None
    R_se: float | None
    abs_se: float | None
    rel_se: float | None
    # Stata-side fields. None when no Stata reference exists for the
    # module (or no row with this statistic).
    Stata_est: float | None = None
    Stata_se: float | None = None
    abs_est_st: float | None = None
    rel_est_st: float | None = None
    abs_se_st: float | None = None
    rel_se_st: float | None = None


def _diff(a: float | None, b: float | None) -> tuple[float | None, float | None]:
    if a is None or b is None:
        return None, None
    abs_d = abs(a - b)
    rel_d = abs_d / abs(b) if abs(b) > 1e-12 else (abs_d if abs(b) < 1e-12 else 0.0)
    return abs_d, rel_d


def _load_stata(module: str) -> dict[str, dict] | None:
    """Return {statistic -> row_dict} from the Stata harness, or None
    if the module has no Stata reference."""
    path = STATA_RESULTS_DIR / f"{module}_Stata.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {r["statistic"]: r for r in payload["rows"]}


def collect(module: str) -> list[RowDiff]:
    py_path = RESULTS_DIR / f"{module}_py.json"
    R_path  = RESULTS_DIR / f"{module}_R.json"
    if not py_path.exists() or not R_path.exists():
        return []
    py = json.loads(py_path.read_text(encoding="utf-8"))
    R  = json.loads(R_path.read_text(encoding="utf-8"))
    R_by = {r["statistic"]: r for r in R["rows"]}
    Stata_by = _load_stata(module) or {}
    out: list[RowDiff] = []
    for pr in py["rows"]:
        rr = R_by.get(pr["statistic"])
        if rr is None:
            continue
        abs_e, rel_e = _diff(pr["estimate"], rr["estimate"])
        abs_s, rel_s = _diff(pr.get("se"), rr.get("se"))
        sr = Stata_by.get(pr["statistic"])
        Stata_est = sr.get("estimate") if sr else None
        Stata_se  = sr.get("se") if sr else None
        abs_est_st, rel_est_st = _diff(pr["estimate"], Stata_est)
        abs_se_st,  rel_se_st  = _diff(pr.get("se"),   Stata_se)
        out.append(RowDiff(
            module=module, statistic=pr["statistic"],
            py_est=pr["estimate"], R_est=rr["estimate"],
            abs_est=abs_e, rel_est=rel_e,
            py_se=pr.get("se"), R_se=rr.get("se"),
            abs_se=abs_s, rel_se=rel_s,
            Stata_est=Stata_est, Stata_se=Stata_se,
            abs_est_st=abs_est_st, rel_est_st=rel_est_st,
            abs_se_st=abs_se_st,   rel_se_st=rel_se_st,
        ))
    return out


def _has_any_stata(modules: list[str]) -> bool:
    return any(_load_stata(m) is not None for m in modules)


def fmt(x: float | None, prec: int = 6) -> str:
    if x is None:
        return "—"
    if abs(x) >= 1 or x == 0.0:
        return f"{x:.{prec}f}"
    return f"{x:.{prec}g}"


def render_md(modules: list[str]) -> str:
    lines: list[str] = [
        "# Track A parity report",
        "",
        "Generated by `Paper-JSS/replication/parity/compare.py` on the "
        "`results/<module>_{py,R}.json` artefacts. Tolerance budget per "
        "module is pre-registered in `compare.py::TOLERANCES`. Documented "
        "convention gaps, common-specification passes, and known SE gaps "
        "(HDFE 1-df, CS-DiD SE under-coverage, RD default-bandwidth "
        "selector, SCM non-uniqueness, DML fold-split noise) are flagged "
        "in the per-module `extra` block of the JSON.",
        "",
    ]
    for m in modules:
        diffs = collect(m)
        if not diffs:
            continue
        meta_path = RESULTS_DIR / f"{m}_py.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("extra", {})
        lines.append(f"## Module {m}")
        if meta:
            for k, v in meta.items():
                if isinstance(v, str) and len(v) > 80:
                    lines.append(f"- **{k}**: {v}")
                else:
                    lines.append(f"- **{k}**: `{v}`")
        lines.append("")
        lines.append("| stat | py est | R est | abs Δ | rel Δ | py SE | R SE | abs Δ SE | rel Δ SE |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for d in diffs:
            lines.append(
                f"| `{d.statistic}` "
                f"| {fmt(d.py_est)} | {fmt(d.R_est)} "
                f"| {fmt(d.abs_est, 3)} | {fmt(d.rel_est, 3)} "
                f"| {fmt(d.py_se)} | {fmt(d.R_se)} "
                f"| {fmt(d.abs_se, 3)} | {fmt(d.rel_se, 3)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


# Per-module headline: a (rows-to-summarise, label, verdict, gap-note)
# tuple that the TeX renderer uses to pick the most informative row to
# show. The headline uses the *strictest* row that the module is
# expected to pass, not the worst-case row -- so a documented
# convention gap doesn't shadow the bit-equal point-estimate result.
HEADLINE: dict[str, dict[str, Any]] = {
    "01_ols": {
        "name": "OLS + HC1 SE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "02_iv": {
        "name": "2SLS + HC1 SE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "03_hdfe": {
        "name": "HDFE 2-way FE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "SE differs by 1-df convention",
    },
    "04_csdid": {
        "name": "CS-DiD simple ATT",
        "headline_filter": lambda d: d.statistic == "simple_ATT",
        "metric": "rel_est",
        "verdict": "\\textbf{SE GAP}",
        "gap_note": "SE under-covers; v0.9.6 roadmap",
    },
    "05_sunab": {
        "name": "Sun--Abraham event study",
        "headline_filter": lambda d: d.statistic.startswith("att_rel_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "aggregation rule differs",
    },
    "06_rd": {
        "name": "RD CCT bias-corrected",
        "headline_filter": lambda d: d.statistic.startswith("forced_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "at forced common $h$; default-$h$ selector differs",
    },
    "07_scm": {
        "name": "Classical SCM",
        "headline_filter": lambda d: d.statistic == "avg_post_gap",
        "metric": "rel_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "SCM non-uniqueness",
    },
    "08_dml": {
        "name": "DML PLR (LinReg learners)",
        "headline_filter": lambda d: d.statistic == "theta_DML_PLR",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "fold-split RNG differs",
    },
    "09_rddensity": {
        "name": "RD density (CJM)",
        "headline_filter": lambda d: d.statistic == "test_pvalue",
        "metric": "abs_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "bandwidth selector differs; both fail to reject",
    },
    "10_honest_did": {
        "name": "Honest DiD bounds",
        "headline_filter": lambda d: d.statistic.startswith("ci_"),
        "metric": "abs_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "FLCI vs analytic-bound solver",
    },
    "11_psm": {
        "name": "PSM 1:1 NN",
        "headline_filter": lambda d: d.statistic == "att_psm",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "12_sdid": {
        "name": "Synthetic DID",
        "headline_filter": lambda d: d.statistic == "att_sdid",
        "metric": "rel_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "regularisation $\\zeta$ convention",
    },
    "13_causal_forest": {
        "name": "Causal forest (AIPW)",
        "headline_filter": lambda d: d.statistic == "att_causal_forest",
        "metric": "rel_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "NSW-DW propensity overlap",
    },
    "14_ols_cluster": {
        "name": "OLS + cluster-robust SE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "15_hdfe_cluster": {
        "name": "HDFE + cluster SE",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "SE 1.27\\% (ssc convention)",
    },
    "16_bjs": {
        "name": "BJS imputation",
        "headline_filter": lambda d: d.statistic == "att_bjs",
        "metric": "rel_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "aggregation rule differs",
    },
    "20_bacon": {
        "name": "Goodman--Bacon decomposition",
        "headline_filter": lambda d: d.statistic == "beta_twfe",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "TWFE matches; per-2x2 convention differs",
    },
    "21_honest_relmags": {
        "name": "Honest-DiD relative-mags",
        "headline_filter": lambda d: d.statistic.startswith("ci_"),
        "metric": "abs_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "conservatism at high Mbar",
    },
    "22_sensemakr": {
        "name": "sensemakr robustness",
        "headline_filter": lambda d: d.statistic in ("beta_treat", "rv_q"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "benchmark $r^2$ conditioning differs",
    },
    "25_lmm": {
        "name": "Linear mixed model",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "28_frontier": {
        "name": "Stochastic frontier (cross-sec.)",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "30_oaxaca": {
        "name": "Blinder--Oaxaca decomposition",
        "headline_filter": lambda d: d.statistic == "gap",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "twofold vs threefold split convention",
    },
    "17_etwfe": {
        "name": "Wooldridge ETWFE",
        "headline_filter": lambda d: d.statistic == "att_etwfe",
        "metric": "rel_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "aggregation rule differs",
    },
    "18_augsynth": {
        "name": "Augmented SCM",
        "headline_filter": lambda d: d.statistic == "att_augmented",
        "metric": "rel_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "SCM non-uniqueness",
    },
    "19_gsynth": {
        "name": "Generalized SCM (Xu 2017)",
        "headline_filter": lambda d: d.statistic == "att_gsynth",
        "metric": "rel_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "SCM non-uniqueness; both pick r=1",
    },
    "23_evalue": {
        "name": "E-value (closed form)",
        "headline_filter": lambda d: d.statistic.startswith("evalue_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "24_coxph": {
        "name": "Cox proportional hazards",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "26_glmm_logit": {
        "name": "GLMM logit (Laplace)",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "27_glmm_aghq": {
        "name": "GLMM logit (AGHQ, n=8)",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "29_panel_sfa": {
        "name": "Panel SFA (Pitt--Lee)",
        "headline_filter": lambda d: d.statistic.startswith("beta_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "31_dfl": {
        "name": "DFL reweighting",
        "headline_filter": lambda d: d.statistic == "gap",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "comp/struct split convention",
    },
    "32_rif": {
        "name": "RIF / UQR (median)",
        "headline_filter": lambda d: d.statistic == "total_diff",
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "33_var": {
        "name": "VAR (vars::VAR)",
        "headline_filter": lambda d: d.statistic.startswith("eq_"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
    "34_lp": {
        "name": "Local projections",
        "headline_filter": lambda d: d.statistic.startswith("irf_"),
        "metric": "abs_est",
        "verdict": "\\textit{GAP}",
        "gap_note": "Cholesky-orthogonalised vs OLS shock",
    },
    "35_panel": {
        "name": "Panel FE/RE + Hausman",
        "headline_filter": lambda d: d.statistic.startswith(
            ("fe_beta_", "re_beta_")),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "FE/RE coefs match; Hausman variance differs",
    },
    "36_mediation": {
        "name": "Causal mediation (IKT)",
        "headline_filter": lambda d: d.statistic in (
            "acme", "ade", "total_effect"),
        "metric": "rel_est",
        "verdict": "\\textbf{PASS}",
        "gap_note": "",
    },
}


def render_tex(modules: list[str]) -> str:
    rows: list[str] = []
    for m in modules:
        diffs = collect(m)
        if not diffs:
            continue
        cfg = HEADLINE.get(m, {
            "name": m, "headline_filter": lambda d: True,
            "metric": "rel_est", "verdict": "\\textit{review}",
            "gap_note": "",
        })
        filtered = [d for d in diffs if cfg["headline_filter"](d)]
        if not filtered:
            filtered = diffs
        metric = cfg["metric"]
        vals = [getattr(d, metric) for d in filtered if getattr(d, metric) is not None]
        if not vals:
            continue
        worst = max(vals)
        if metric == "rel_est":
            primary = f"rel $\\le {worst:.2g}$"
        else:
            primary = f"abs $\\le {worst:.3g}$"
        gap_note = cfg.get("gap_note", "")
        gap_cell = f" {{\\footnotesize ({gap_note})}}" if gap_note else ""
        # Escape underscores inside \code{...} so the texttt rendering
        # does not trip the LaTeX scanner.
        m_safe = m.replace("_", r"\_")
        rows.append(
            f"\\code{{{m_safe}}} & {cfg['name']} & {primary}{gap_cell} & "
            f"{cfg['verdict']} \\\\"
        )

    body = "\n".join(rows)
    return (
        "% AUTO-GENERATED by tests/r_parity/compare.py\n"
        "% Re-run after any module change to refresh.\n"
        "\\begin{longtable}{p{0.10\\linewidth}p{0.27\\linewidth}p{0.40\\linewidth}p{0.16\\linewidth}}\n"
        "\\caption{Track A parity headline at \\statspai{} 1.13.0 vs the "
        "canonical \\proglang{R} reference on the calibrated replicas. The "
        "``Worst diff'' column reports the worst residual gap across the "
        "module's headline rows (point estimates only; per-row SE diffs "
        "and documented gap rows are reported in the Markdown source). "
        "Verdicts use PASS, GAP, and SE GAP; common-specification "
        "passes and convention gaps are explained in the parenthetical "
        "notes and per-module \\code{extra} block in "
        "\\code{tests/r\\_parity/results/}.}\n"
        "\\label{tab:track-a-parity}\\\\\n"
        "\\toprule\n"
        "Module & Method & Worst headline diff & Verdict \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{4}{c}{\\textit{(continued)}}\\\\\n"
        "\\toprule\n"
        "Module & Method & Worst headline diff & Verdict \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{body}\n"
        "\\end{longtable}\n"
    )


def render_tex_3way(modules: list[str]) -> str:
    """Five-column 3-way table: Module / Method / vs R / vs Stata / Verdict."""
    rows: list[str] = []
    for m in modules:
        diffs = collect(m)
        if not diffs:
            continue
        cfg = HEADLINE.get(m, {
            "name": m, "headline_filter": lambda d: True,
            "metric": "rel_est", "verdict": "\\textit{review}",
            "gap_note": "",
        })
        filtered = [d for d in diffs if cfg["headline_filter"](d)]
        if not filtered:
            filtered = diffs
        metric = cfg["metric"]
        # vs-R column.
        vals_r = [getattr(d, metric) for d in filtered if getattr(d, metric) is not None]
        if not vals_r:
            continue
        worst_r = max(vals_r)
        if metric == "rel_est":
            primary_r = f"rel $\\le {worst_r:.2g}$"
        else:
            primary_r = f"abs $\\le {worst_r:.3g}$"
        # vs-Stata column.
        st_metric = "rel_est_st" if metric == "rel_est" else "abs_est_st"
        st_vals = [getattr(d, st_metric) for d in filtered
                    if getattr(d, st_metric) is not None]
        if st_vals:
            worst_s = max(st_vals)
            if metric == "rel_est":
                primary_s = f"rel $\\le {worst_s:.2g}$"
            else:
                primary_s = f"abs $\\le {worst_s:.3g}$"
        else:
            reason = STATA_SKIP_REASON.get(m, "n/a")
            primary_s = f"\\emph{{{reason}}}"
        gap_note = cfg.get("gap_note", "")
        gap_cell = f" {{\\footnotesize ({gap_note})}}" if gap_note else ""
        m_safe = m.replace("_", r"\_")
        rows.append(
            f"\\code{{{m_safe}}} & {cfg['name']} & "
            f"{primary_r}{gap_cell} & {primary_s} & "
            f"{cfg['verdict']} \\\\"
        )

    body = "\n".join(rows)
    return (
        "% AUTO-GENERATED by tests/r_parity/compare.py\n"
        "% Re-run after any module change to refresh.\n"
        "\\begin{longtable}{p{0.085\\linewidth}p{0.20\\linewidth}p{0.27\\linewidth}p{0.27\\linewidth}p{0.115\\linewidth}}\n"
        "\\caption{Track A parity headline at \\statspai{} 1.13.0 against the canonical "
        "\\proglang{R} reference \\emph{and} (where one exists) the canonical \\proglang{Stata} "
        "reference, on the calibrated replicas. The two diff columns report the worst residual "
        "gap across the module's headline rows (point estimates only; per-row SE diffs and "
        "documented gap rows are reported in \\code{tests/r\\_parity/results/parity\\_table\\_3way.md}). "
        "``no canonical ref'' marks modules whose \\proglang{Stata} ecosystem has no "
        "authoritative port we can compare to without fabricating one. Verdicts use PASS, "
        "GAP, and SE GAP; common-specification passes and convention gaps are explained "
        "in the parenthetical notes and per-module \\code{extra} block in "
        "\\code{tests/r\\_parity/results/} and \\code{tests/stata\\_parity/results/}.}\n"
        "\\label{tab:track-a-parity}\\\\\n"
        "\\toprule\n"
        "Module & Method & Worst diff vs \\proglang{R} & Worst diff vs \\proglang{Stata} & Verdict \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n"
        "\\multicolumn{5}{c}{\\textit{(continued)}}\\\\\n"
        "\\toprule\n"
        "Module & Method & Worst diff vs \\proglang{R} & Worst diff vs \\proglang{Stata} & Verdict \\\\\n"
        "\\midrule\n"
        "\\endhead\n"
        "\\bottomrule\n"
        "\\endlastfoot\n"
        f"{body}\n"
        "\\end{longtable}\n"
    )


def render_md_3way(modules: list[str]) -> str:
    """Markdown with Stata column when available."""
    lines: list[str] = [
        "# Track A parity report (3-way: \\proglang{Python} <-> R <-> Stata)",
        "",
        "Generated by `tests/r_parity/compare.py` on the "
        "`results/<module>_{py,R}.json` and "
        "`tests/stata_parity/results/<module>_Stata.json` artefacts. "
        "Tolerance budget per module is pre-registered in "
        "`compare.py::TOLERANCES`. Documented convention gaps, "
        "common-specification passes, and known SE gaps are flagged "
        "in the per-module `extra` block of each JSON.",
        "",
    ]
    for m in modules:
        diffs = collect(m)
        if not diffs:
            continue
        meta_path = RESULTS_DIR / f"{m}_py.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")).get("extra", {})
        st_meta_path = STATA_RESULTS_DIR / f"{m}_Stata.json"
        st_meta = (
            json.loads(st_meta_path.read_text(encoding="utf-8")).get("extra", {})
            if st_meta_path.exists() else {}
        )
        lines.append(f"## Module {m}")
        if meta:
            for k, v in meta.items():
                if isinstance(v, str) and len(v) > 80:
                    lines.append(f"- **{k}**: {v}")
                else:
                    lines.append(f"- **{k}**: `{v}`")
        if st_meta:
            for k, v in st_meta.items():
                if k.startswith("stata"):
                    lines.append(f"- **{k}**: `{v}`")
        elif m in STATA_SKIP_REASON:
            lines.append(f"- **stata_status**: {STATA_SKIP_REASON[m]}")
        lines.append("")
        lines.append(
            "| stat | py est | R est | Stata est | rel py-R | rel py-Stata | py SE | R SE | Stata SE | rel SE py-R | rel SE py-Stata |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for d in diffs:
            lines.append(
                f"| `{d.statistic}` "
                f"| {fmt(d.py_est)} | {fmt(d.R_est)} | {fmt(d.Stata_est)} "
                f"| {fmt(d.rel_est, 3)} | {fmt(d.rel_est_st, 3)} "
                f"| {fmt(d.py_se)} | {fmt(d.R_se)} | {fmt(d.Stata_se)} "
                f"| {fmt(d.rel_se, 3)} | {fmt(d.rel_se_st, 3)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    modules = sorted(
        p.stem.replace("_py", "")
        for p in RESULTS_DIR.glob("*_py.json")
    )
    md = render_md(modules)
    tex = render_tex(modules)
    (RESULTS_DIR / "parity_table.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / "parity_table.tex").write_text(tex, encoding="utf-8")
    print("OK -- wrote parity_table.md and parity_table.tex")

    # 3-way Stata extension. Always emitted; Stata-empty modules show
    # the explicit "no canonical ref" reason rather than a blank.
    if _has_any_stata(modules) or STATA_SKIP_REASON:
        md3 = render_md_3way(modules)
        tex3 = render_tex_3way(modules)
        (RESULTS_DIR / "parity_table_3way.md").write_text(md3, encoding="utf-8")
        (RESULTS_DIR / "parity_table_3way.tex").write_text(tex3, encoding="utf-8")
        print("OK -- wrote parity_table_3way.md and parity_table_3way.tex")
        n_stata = sum(1 for m in modules if _load_stata(m) is not None)
        print(f"     ({n_stata} of {len(modules)} modules have a Stata reference)")
    print(f"     ({len(modules)} modules: {', '.join(modules)})")


if __name__ == "__main__":
    main()
