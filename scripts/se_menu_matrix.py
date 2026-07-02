"""Standard-error / vcov menu coverage matrix for StatsPAI estimators.

Stata's ``vce()`` and ``boottest`` are *near-universal*: cluster / robust /
bootstrap SEs attach to almost any command with the same grammar.  StatsPAI's
SE menu is, by contrast, uneven — the 2026-06-30 audit found that wild cluster
bootstrap is natively + correctly wired only on the panel ``hdfe_ols`` / panel
``feols`` path, while the flagship pyfixest ``sp.feols``, IV, Poisson/GLM and
DML cannot reach it.  The standalone ``sp.wild_cluster_boot`` / ``sp.cr2_se``
helpers *exist* but several work only by re-parsing the formula and refitting a
plain OLS, which silently ignores fixed effects and the IV two-stage structure.

This module encodes that finding as a **machine-checked matrix** so the gap is
visible, gated, and ratchetable as the wiring work lands.  Cell status:

* ``native``           — exposed as a parameter on the estimator itself and
  numerically correct for that estimator's structure (FE / IV / GLM).
* ``standalone``       — reachable via a standalone ``sp.*`` SE function that
  reads the model's stored design matrix / residuals (correct).
* ``standalone_unsafe``— reachable only via a standalone function that
  re-parses the formula and refits plain OLS, so it is **wrong** for this
  estimator's FE / IV structure.  These are the cells the wiring work targets.
* ``na``               — not available without manual array gymnastics, or
  intentionally orthogonal (e.g. influence-function inference).

The matrix is curated, not auto-probed (live-fitting every estimator × SE type
is slow and flaky), but it is *validated against the live API*: every estimator
and every referenced standalone SE function must still resolve under ``sp.*``,
so the spec cannot silently rot.  ``--check`` additionally ratchets that the
number of ``native`` cells never decreases and ``standalone_unsafe`` never
increases.

Usage
-----
::

    python scripts/se_menu_matrix.py            # human matrix + summary
    python scripts/se_menu_matrix.py --markdown # GitHub-flavoured table
    python scripts/se_menu_matrix.py --json
    python scripts/se_menu_matrix.py --check    # CI ratchet + drift guard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# SE-type columns, in menu order.
SE_TYPES: tuple[str, ...] = (
    "classical",
    "hc_robust",
    "cluster",
    "twoway",
    "cr2_cr3",
    "wild_cluster_boot",
    "conley",
    "jackknife",
)

# Status vocabulary (worst -> best for summary colouring).
STATUSES = ("na", "standalone_unsafe", "standalone", "native")

#: Standalone SE functions the matrix references — must resolve under sp.*.
STANDALONE_SE_FUNCS: tuple[str, ...] = (
    "wild_cluster_bootstrap",
    "wild_cluster_boot",
    "subcluster_wild_bootstrap",
    "wild_cluster_ci_inv",
    "cr2_se",
    "cr3_jackknife_vcov",
    "multiway_cluster_vcov",
    "cluster_robust_se",
    "twoway_cluster",
    "conley",
    "jackknife_se",
)

#: estimator -> {se_type: status}.  Grounded in the 2026-06-30 SE-menu audit.
#: Missing keys default to ``na``.
MATRIX: Dict[str, Dict[str, str]] = {
    "feols": {  # pyfixest path — flagship FE estimator
        "classical": "native",
        "hc_robust": "native",
        "cluster": "native",
        "twoway": "native",
        # Native `feols(..., vce="wild", cluster=...)` runs the WCR wild cluster
        # bootstrap on the FE-absorbed within design. Externally validated
        # against Stata `boottest` (Roodman): point estimate and CRV1 cluster SE
        # match reghdfe to ~1e-9, wild p-value matches boottest's exact 2^15 to
        # Monte-Carlo error. See tests/reference_parity/test_feols_wild_*.
        "wild_cluster_boot": "native",
        # Native `feols(vce="CR2"/"CR3"/"jackknife")` — Pustejovsky-Tipton on the
        # FE-absorbed within design. The within-transform's leverage adjustment
        # matches R clubSandwich (plm) for CR2/CR3 to machine precision.
        "cr2_cr3": "native",
        "jackknife": "native",
        # Native `feols(vce="conley", conley_lat=, conley_lon=, conley_cutoff=)`
        # — spatial HAC on the within design (acreg planar distance); verified
        # equal to sp.regress on the FE-demeaned data (acreg-validated).
        "conley": "native",
    },
    "hdfe_ols": {  # panel within-path — the ONLY correct native wild boot
        "classical": "native",
        "cluster": "native",
        "twoway": "native",
        "wild_cluster_boot": "native",
    },
    "fepois": {
        "classical": "native",
        "hc_robust": "native",
        "cluster": "native",
        "twoway": "native",
        # Native `fepois(vce="CR2"/"CR3"/"jackknife", cluster=...)` — the
        # clubSandwich glm bias-reduced adjustment on the FE-as-dummies design
        # (IRLS-weighted). Matches R `clubSandwich::vcovCR(glm(poisson))`.
        "cr2_cr3": "native",
        "jackknife": "native",
        # Native `fepois(vce="wild", cluster=...)` — restricted score wild
        # cluster bootstrap (Kline-Santos 2012), the method Stata `boottest`
        # runs after `poisson`. Consistent with boottest to ~2 decimals (not
        # bit-exact: boottest's studentization differs). Enumerated for small G.
        "wild_cluster_boot": "native",
    },
    "feglm": {
        "classical": "native",
        "hc_robust": "native",
        "cluster": "native",
        "twoway": "native",
        # Native `feglm(vce="CR2"/"CR3"/"jackknife", cluster=...)` — same
        # clubSandwich glm adjustment, generalised to the logit/probit/gaussian
        # family via d=dμ/dη, V=Var(μ). Matches R `clubSandwich::vcovCR(glm)`.
        "cr2_cr3": "native",
        "jackknife": "native",
        # Native `feglm(vce="wild", cluster=...)` — score wild cluster bootstrap
        # (Kline-Santos 2012), consistent with Stata `boottest` after logit to
        # ~2 decimals (not bit-exact).
        "wild_cluster_boot": "native",
    },
    "regress": {
        "classical": "native",
        "hc_robust": "native",
        "cluster": "native",
        # Native `regress(cluster=["a","b"])` — CGM-2011 inclusion-exclusion two-way
        # cluster sandwich. Matches Stata `reghdfe y x, vce(cluster a b)`.
        "twoway": "native",
        # Native `regress(vce="CR2"/"CR3"/"jackknife")` — Pustejovsky-Tipton 2018
        # bias-reduced cluster-robust; matches R `sandwich::vcovCL(HC2/3)`.
        "cr2_cr3": "native",
        "jackknife": "native",
        # Native `regress(vce="wild", cluster=...)` — WCR cluster bootstrap
        # (Cameron-Gelbach-Miller 2008) on the stored OLS design.
        "wild_cluster_boot": "native",
        # Native `regress(vce="conley", conley_lat/lon/cutoff)` — Stata
        # `acreg` planar-distance spatial HAC.
        "conley": "native",
    },
    "ivreg": {  # IV
        "classical": "native",
        "hc_robust": "native",
        "cluster": "native",
        # Native `ivreg(vce="wild", cluster=...)` runs the WRE wild cluster
        # bootstrap (Davidson-MacKinnon 2010) — the IV-correct bootstrap that
        # resamples both equations. Externally validated against Stata
        # `boottest` after `ivreg2`: matches the WRE p-value across strong-IV
        # (0.2016 vs 0.20155) and weak-IV (0.3415 vs 0.3412) regimes, and the
        # weak-IV case rules out the naive (non-efficient) variant.
        "wild_cluster_boot": "native",
        # Native `ivreg(cluster=["a", "b"])` — two-way IV cluster sandwich
        # (CGM 2011 inclusion-exclusion on the projected regressors). Matches
        # Stata `ivreg2, cluster(a b) small`.
        "twoway": "native",
        # Native `ivreg(vce="CR2")` / `vce="CR3"` (== `vce="jackknife"`) — the
        # Pustejovsky-Tipton (2018) bias-reduced adjustment on the projected 2SLS
        # regressors. Matches R `clubSandwich::vcovCR(ivreg, type=...)` exactly.
        "cr2_cr3": "native",
        "jackknife": "native",
        # Native `ivreg(vce="conley", conley_lat=, conley_lon=, conley_cutoff=)`
        # — spatial HAC on the projected 2SLS scores with acreg's planar distance
        # (111 km/deg, cos(lat) longitude). Matches Stata `acreg ... spatial`.
        "conley": "native",
    },
    "ppmlhdfe": {
        "classical": "native",
        "hc_robust": "native",
        "cluster": "native",
        # Native `ppmlhdfe(cluster=["a","b"])` — CGM-2011 two-way cluster on the
        # FE-residualised PPML design with the single G_min/(G_min-1) factor.
        # Byte-identical to Stata `ppmlhdfe ..., cluster(a b)`.
        "twoway": "native",
    },
    "panel": {  # FE/RE — has Driscoll-Kraay (not modelled as a column here)
        "classical": "native",
        "hc_robust": "native",
        "cluster": "native",
        # Native `panel(method="fe", cluster=["a","b"])` — CGM-2011 two-way
        # cluster on the entity-within design. Equals `sp.regress(cluster=[a,b])`
        # on the FE-demeaned data (reghdfe / ivreg2 small-sample convention).
        "twoway": "native",
        # Native `panel(method="fe", vce="CR2"/"CR3"/"jackknife")` —
        # Pustejovsky-Tipton 2018 bias-reduced cluster-robust on the entity
        # within design. Matches R `clubSandwich::vcovCR(plm, model="within")`
        # exactly (same frozen anchor as sp.feols).
        "cr2_cr3": "native",
        "jackknife": "native",
        # Native `panel(method="fe", vce="conley", conley_lat/lon/cutoff)` —
        # spatial HAC on the entity within design (acreg planar distance).
        # Equals `sp.regress(vce="conley")` on the FE-demeaned data.
        "conley": "native",
    },
    "callaway_santanna": {  # influence-function + multiplier bootstrap
        "cluster": "native",
    },
    "did": {
        "cluster": "native",
    },
    "dml": {  # asymptotic psi-score SE; clustering via dml_panel only
        "classical": "native",
    },
    "rdrobust": {  # RD bias-aware variance + cluster
        "hc_robust": "native",
        "cluster": "native",
    },
    "synth": {  # SC permutation / conformal / SDID jackknife — orthogonal menu
        "jackknife": "native",
    },
}

BASELINE_PATH = Path(__file__).resolve().parent / "se_menu_matrix_baseline.json"


def _cell(estimator: str, se_type: str) -> str:
    return MATRIX.get(estimator, {}).get(se_type, "na")


def _validate_live_api() -> List[str]:
    """Return a list of drift problems (missing estimators / SE functions)."""
    import statspai as sp

    available = set(sp.list_functions())
    problems: List[str] = []
    for est in MATRIX:
        if est not in available:
            problems.append(f"estimator '{est}' no longer in sp.list_functions()")
    for fn in STANDALONE_SE_FUNCS:
        if fn not in available:
            problems.append(f"standalone SE function '{fn}' missing from sp.*")
    return problems


def collect() -> Dict[str, Any]:
    counts: Dict[str, int] = {s: 0 for s in STATUSES}
    per_estimator: Dict[str, Dict[str, int]] = {}
    for est in MATRIX:
        est_counts = {s: 0 for s in STATUSES}
        for se in SE_TYPES:
            status = _cell(est, se)
            counts[status] += 1
            est_counts[status] += 1
        per_estimator[est] = est_counts

    # "Stranded": estimators that cannot reach wild boot or CR2 natively.
    stranded = []
    for est in MATRIX:
        wild = _cell(est, "wild_cluster_boot")
        cr = _cell(est, "cr2_cr3")
        if wild != "native" and cr != "native":
            stranded.append(
                {"estimator": est, "wild_cluster_boot": wild, "cr2_cr3": cr}
            )

    drift = _validate_live_api()
    return {
        "se_types": list(SE_TYPES),
        "status_counts": counts,
        "per_estimator": per_estimator,
        "stranded": stranded,
        "drift": drift,
        "thresholds": {
            "native_min": _baseline().get("native_min", counts["native"]),
            "unsafe_max": _baseline().get("unsafe_max", counts["standalone_unsafe"]),
        },
    }


def _baseline() -> Dict[str, int]:
    if not BASELINE_PATH.exists():
        return {}
    data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in data.items() if isinstance(v, int)}


def update_baseline(report: Dict[str, Any]) -> int:
    counts = report["status_counts"]
    payload = {
        "_comment": (
            "Ratchet floor/ceiling for the SE-menu coverage matrix. "
            "native_min may only rise; unsafe_max may only fall, as wiring "
            "work flips standalone_unsafe / na cells to native. Regenerate "
            "with `python scripts/se_menu_matrix.py --update-baseline`."
        ),
        "native_min": counts["native"],
        "unsafe_max": counts["standalone_unsafe"],
    }
    BASELINE_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[se_menu_matrix] baseline written to {BASELINE_PATH.name}")
    return 0


def check(report: Dict[str, Any]) -> int:
    failures: List[str] = []
    drift = report["drift"]
    if drift:
        failures.extend(f"drift: {d}" for d in drift)
    base = _baseline()
    counts = report["status_counts"]
    if base:
        if counts["native"] < base.get("native_min", 0):
            failures.append(
                f"native cells regressed: {counts['native']} "
                f"< floor {base['native_min']}"
            )
        if counts["standalone_unsafe"] > base.get("unsafe_max", 10**9):
            failures.append(
                "standalone_unsafe grew: "
                f"{counts['standalone_unsafe']} > ceiling {base['unsafe_max']}"
            )
    else:
        failures.append("no baseline frozen; run --update-baseline first")
    if failures:
        print("[se_menu_matrix] REGRESSION", file=sys.stderr)
        for item in failures:
            print(f"  {item}", file=sys.stderr)
        return 1
    print(
        "[se_menu_matrix] OK - "
        f"{counts['native']} native cells, "
        f"{counts['standalone_unsafe']} unsafe, no drift."
    )
    return 0


_GLYPH = {
    "native": "✓",
    "standalone": "○",
    "standalone_unsafe": "⚠",
    "na": "·",
}


def render(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("StatsPAI SE / vcov menu coverage matrix")
    lines.append("=" * 72)
    lines.append(
        "legend: ✓ native   ○ standalone(correct)   ⚠ standalone-unsafe   · n/a"
    )
    lines.append("")
    header = f"{'estimator':18s}" + "".join(f"{s[:9]:>11s}" for s in SE_TYPES)
    lines.append(header)
    lines.append("-" * len(header))
    for est in MATRIX:
        row = f"{est:18s}"
        for se in SE_TYPES:
            row += f"{_GLYPH[_cell(est, se)]:>11s}"
        lines.append(row)
    lines.append("")
    counts = report["status_counts"]
    lines.append("totals: " + ", ".join(f"{s}={counts[s]}" for s in STATUSES))
    stranded = report["stranded"]
    lines.append("")
    lines.append(f"stranded (no native wild-boot AND no native CR2): {len(stranded)}")
    for s in stranded:
        est = s["estimator"]
        wild = s["wild_cluster_boot"]
        lines.append(f"  {est:18s} wild={wild:18s} cr2={s['cr2_cr3']}")
    drift = report["drift"]
    if drift:
        lines.append("")
        lines.append("DRIFT:")
        for d in drift:
            lines.append(f"  {d}")
    return "\n".join(lines)


def render_markdown(report: Dict[str, Any]) -> str:
    head = "| estimator | " + " | ".join(SE_TYPES) + " |"
    sep = "|" + "---|" * (len(SE_TYPES) + 1)
    rows = [head, sep]
    for est in MATRIX:
        cells = " | ".join(_GLYPH[_cell(est, se)] for se in SE_TYPES)
        rows.append(f"| `{est}` | {cells} |")
    rows.append("")
    rows.append(
        "Legend: ✓ native · ○ standalone (correct) · ⚠ standalone-unsafe · · n/a"
    )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args(argv)

    report = collect()
    if args.update_baseline:
        return update_baseline(report)
    if args.check:
        return check(report)
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    if args.markdown:
        print(render_markdown(report))
        return 0
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
