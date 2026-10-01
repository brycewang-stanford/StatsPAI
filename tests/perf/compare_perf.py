"""Track C performance comparator.

Joins per-estimator timings on the StatsPAI and declared reference sides, emits a
Markdown rollup, a LaTeX table for §6 of the manuscript, and a
log-log scaling figure.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # _figstyle.py sits here

import matplotlib  # noqa: E402  (after the sys.path insert above)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from _figstyle import TEXT_WIDTH_IN  # noqa: E402
from _figstyle import apply as _apply_figstyle  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    FuncFormatter,
    LogLocator,
    NullFormatter,
)

_apply_figstyle(plt)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
PAPER_TABLES_DIR = ROOT / "Paper-JSS" / "manuscript" / "tables"
PAPER_FIGURES_DIR = ROOT / "Paper-JSS" / "manuscript" / "figures"
PDF_METADATA = {"CreationDate": None}
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


ESTIMATORS = {
    "01_hdfe": {
        "name": "HDFE 2-way FE",
        "task": "OLS, 2 absorbed FE, iid SE",
        "short": ("HDFE", "OLS, two absorbed FE, iid SE"),
        "ref": "fixest::feols",
        "ref_side": "R",
        "x_label": "N (observations)",
        "agreement": "exact",
    },
    "02_csdid": {
        "name": "CS-DiD",
        "task": "ATT(g,t), pre-test, simple + dynamic agg.",
        "short": ("CS-DiD", "ATT(g,t), pre-test, simple and dynamic aggregation"),
        "ref": "did::att_gt",
        "ref_side": "R",
        "x_label": "N (observations)",
        "agreement": "exact",
    },
    "03_scm": {
        "name": "Classical SCM (ADH)",
        "task": "special predictors, nested V, no placebos",
        "short": ("SCM", "ADH special predictors, nested V, no placebos"),
        "ref": "Synth::synth",
        "ref_side": "R",
        "x_label": "n_donors",
        "agreement": "solver",
    },
    "04_dml": {
        "name": "DML PLR (lin. learners)",
        "task": "PLR, linear learners, 5 folds",
        "short": ("DML", "PLR, linear learners, 5 folds"),
        "ref": "doubleml-for-py",
        "ref_side": "doubleml_py",
        "x_label": "N (observations)",
        "agreement": "folds",
    },
}


def load(estimator: str, side: str) -> list[dict]:
    path = RESULTS_DIR / f"{estimator}_{side}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["rows"]


def load_hardware(estimator: str, side: str) -> dict:
    path = RESULTS_DIR / f"{estimator}_{side}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("hardware") or {}


class PairingError(RuntimeError):
    """The two sides of a row did not time the same input in the same run."""


def check_pairing(allow_mixed: bool = False) -> list[str]:
    """Refuse rows whose sides read different bytes or came from other runs.

    A speed ratio is a statement about two implementations of one task on
    one input. Pairing a fresh Python timing with a stale R timing, or two
    sides that read different data, silently changes what the ratio means.
    """
    problems: list[str] = []
    for est, cfg in ESTIMATORS.items():
        py, ref = load(est, "py"), load(est, cfg["ref_side"])
        if not py or not ref:
            problems.append(f"{est}: a side is missing")
            continue
        run_py = load_hardware(est, "py").get("run_id")
        run_ref = load_hardware(est, cfg["ref_side"]).get("run_id")
        if run_py != run_ref or run_py in (None, "unset"):
            problems.append(f"{est}: run ids differ ({run_py} vs {run_ref})")
        ref_by_n = {row["n"]: row for row in ref}
        for row in py:
            other = ref_by_n.get(row["n"])
            if other is None:
                problems.append(f"{est} n={row['n']}: no reference row")
                continue
            h1 = row.get("extra", {}).get("data_sha256")
            h2 = other.get("extra", {}).get("data_sha256")
            if not h1 or h1 != h2:
                problems.append(f"{est} n={row['n']}: input hashes differ")
    if problems and not allow_mixed:
        raise PairingError("; ".join(problems))
    return problems


def agreement(est: str, prow: dict, rrow: dict) -> str:
    """How far the two sides' target quantity is apart, in the row's terms."""
    a = prow.get("extra", {}).get("estimate")
    b = rrow.get("extra", {}).get("estimate")
    if a is None or b is None:
        return "n/a"
    kind = ESTIMATORS[est]["agreement"]
    diff = abs(a - b)
    if kind == "exact":
        return f"{diff / max(abs(b), 1e-300):.0e}"
    if kind == "folds":
        se = rrow.get("extra", {}).get("se") or float("nan")
        return f"{diff / se:.2f} SE"
    return f"{diff:.3f}"


def measured_versions() -> list[str]:
    """StatsPAI versions recorded by the Python-side result files."""
    versions = set()
    for est in ESTIMATORS:
        versions.add(load_hardware(est, "py").get("statspai_version") or "unrecorded")
    return sorted(versions)


def render_md() -> str:
    lines: list[str] = [
        "# Track C performance report",
        "",
        "Generated by `tests/perf/compare_perf.py`. Both sides of every row read",
        "the same input file (`tests/perf/_data.py`; SHA-256 checked), run the",
        "task named below on one thread, and were timed in one run.",
        "`ratio` is `reference_time / sp_time` (>1 means sp is faster);",
        "`agreement` compares the target quantity of the two sides outside",
        "the timed call.",
        "",
    ]
    for est, cfg in ESTIMATORS.items():
        py = load(est, "py")
        r = load(est, cfg["ref_side"])
        if not py or not r:
            continue
        lines.append(f"## {est}: {cfg['name']} -- {cfg['task']}")
        lines.append("")
        lines.append(
            f"| n | sp median (IQR) s | {cfg['ref']} median (IQR) s | ratio | agreement |"
        )
        lines.append("|---:|---:|---:|---:|---:|")
        r_by_n = {row["n"]: row for row in r}
        for prow in py:
            rrow = r_by_n.get(prow["n"])
            if rrow is None:
                continue
            ratio = rrow["median_time_s"] / prow["median_time_s"]
            lines.append(
                f"| {prow['n']} | {prow['median_time_s']:.4f} ({prow['iqr_time_s']:.4f}) | "
                f"{rrow['median_time_s']:.4f} ({rrow['iqr_time_s']:.4f}) | {ratio:.2f} | "
                f"{agreement(est, prow, rrow)} |"
            )
        if est == "03_scm":
            lines.append("")
            lines.append(
                "Package-default `sp.synth(method='classic')` (V = I, placebos):"
            )
            lines.append("")
            lines.append(
                "| n_donors | default (s) | without placebos (s) | ADH starts |"
            )
            lines.append("|---:|---:|---:|---:|")
            for prow in py:
                x = prow.get("extra", {})
                lines.append(
                    f"| {prow['n']} | {x.get('default_workflow_s', float('nan')):.3f} | "
                    f"{x.get('default_no_placebo_s', float('nan')):.3f} | {x.get('starts')} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def _tex(text: str) -> str:
    """Escape a display label for LaTeX.

    ``ESTIMATORS`` holds plain text because the same strings are drawn by
    Matplotlib, which prints a backslash literally: pre-escaping them put
    ``did::att\\_gt`` and ``n\\_donors`` into the published Figure 3 axes.
    Escaping happens here, at the one boundary that needs it.
    """
    return text.replace("\\", "\\textbackslash{}").replace("_", "\\_")


def render_tex() -> str:
    rows: list[str] = []
    ref_tex = {
        "fixest::feols": "\\pkg{fixest}",
        "did::att_gt": "\\pkg{did}",
        "Synth::synth": "\\pkg{Synth}",
        "doubleml-for-py": "\\pkg{DoubleML}",
    }
    for est, cfg in ESTIMATORS.items():
        py = load(est, "py")
        r = load(est, cfg["ref_side"])
        if not py or not r:
            continue
        r_by_n = {row["n"]: row for row in r}
        max_n = max(prow["n"] for prow in py)
        last = next(p for p in py if p["n"] == max_n)
        rlast = r_by_n[max_n]
        ratio = rlast["median_time_s"] / last["median_time_s"]
        if ratio > 1.5:
            faster = f"sp $\\mathbf{{{ratio:.1f}\\times}}$"
        elif ratio < 1 / 1.5:
            faster = (
                f"ref. $\\mathbf{{{1 / ratio:.0f}\\times}}$"
                if ratio < 0.1
                else (f"ref. $\\mathbf{{{1 / ratio:.1f}\\times}}$")
            )
        else:
            faster = f"tie ${ratio:.2f}\\times$"
        max_n_tex = f"{max_n:,}".replace(",", "{,}")

        def fmt(row: dict) -> str:
            m, q = row["median_time_s"], row["iqr_time_s"]
            digits = 3 if m >= 0.01 else 4
            return f"{m:.{digits}f} ({q:.{digits}f})"

        rows.append(
            f"{_tex(cfg['short'][0])} & {_tex(cfg['short'][1])} & "
            f"{ref_tex.get(cfg['ref'], _tex(cfg['ref']))} & {max_n_tex} & "
            f"{fmt(last)} & {fmt(rlast)} & {faster} & "
            f"{_tex(agreement(est, last, rlast))} \\\\"
        )
    body = "\n".join(rows)
    return (
        "% AUTO-GENERATED by tests/perf/compare_perf.py\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\begingroup\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{2pt}\n"
        "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{0.21\\linewidth}"
        "l r r r l r@{}}\n"
        "\\toprule\n"
        "Estimator & Common task & Ref. & Max.~$N$ & \\statspai{} (s) & Ref. (s) "
        "& Faster & Agree \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\endgroup\n"
        "\\caption{Track C at the largest measured size (Apple Silicon arm64, "
        "one thread on both sides, "
        f"\\statspai{{}} {' / '.join(measured_versions())}). "
        "Both sides read the same input file (SHA-256 checked) and run the "
        "task shown; times are median seconds with the interquartile range in "
        "parentheses. ``Agree'' compares the target quantity outside the timed "
        "call: relative difference of the coefficient or simple ATT (HDFE, "
        "CS), absolute difference of the average post-treatment gap between "
        "the two non-convex SCM solvers, and the DML difference in units of "
        "its SE (the folds are drawn independently on each side).}\n"
        "\\label{tab:track-c-perf}\n"
        "\\end{table}\n"
    )


def render_figure() -> Path:
    # Drawn at the printed width so the point sizes in _figstyle are the
    # point sizes on the page; \includegraphics uses width=\textwidth.
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 0.78 * TEXT_WIDTH_IN))
    for ax, (est, cfg) in zip(axes.flat, ESTIMATORS.items()):
        py = load(est, "py")
        r = load(est, cfg["ref_side"])
        if not py or not r:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        py_n = [row["n"] for row in py]
        py_t = [row["median_time_s"] for row in py]
        r_n = [row["n"] for row in r]
        r_t = [row["median_time_s"] for row in r]
        ax.loglog(py_n, py_t, "o-", label="StatsPAI", linewidth=2, markersize=8)
        ax.loglog(r_n, r_t, "s--", label=cfg["ref"], linewidth=2, markersize=7)
        # Pad every y-range out to full decades so each panel shows the
        # same 10^k tick style (a sub-decade range otherwise falls back
        # to linear-looking minor labels under a log-log title).
        lo = min(py_t + r_t)
        hi = max(py_t + r_t)
        ax.set_ylim(10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi)))
        # Decade ticks on the time axis. Left to itself, Matplotlib
        # labels minor ticks whenever a log axis spans few decades, so
        # the DML panel printed 2x10^-3 ... 6x10^-3 while its three
        # neighbours printed 10^-3 ... 10^0: four panels under one
        # "log-log" heading with two different tick vocabularies.
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_formatter(NullFormatter())
        # The size axis is ticked at the measured design points rather
        # than at decades. Two of the four benchmarks are sampled off
        # the decade grid (N = 5,000/25,000/125,000 and n = 20/50/100),
        # and decade-only ticks left the SCM panel with a single labelled
        # x value -- an axis a reader cannot read a size off.
        sizes = sorted({row["n"] for row in py} | {row["n"] for row in r})
        ax.xaxis.set_major_locator(FixedLocator(sizes))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v):,}"))
        ax.set_xlabel(cfg["x_label"])
        ax.set_ylabel("median wall-clock (s)")
        ax.set_title(f"{est}: {cfg['name']}")
        ax.grid(True, which="both", alpha=0.3, linewidth=0.4)
        ax.legend()
    # No suptitle: the hardware line and the "log-log scaling" framing
    # both live in the LaTeX caption, and repeating them inside the
    # graphic wastes two lines of the panel area at print size.
    fig.tight_layout()
    out = FIGURES_DIR / "track_c_loglog.pdf"
    fig.savefig(out, bbox_inches="tight", metadata=PDF_METADATA)
    fig.savefig(FIGURES_DIR / "track_c_loglog.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    shutil.copyfile(out, PAPER_FIGURES_DIR / "track_c_loglog.pdf")
    shutil.copyfile(
        FIGURES_DIR / "track_c_loglog.png", PAPER_FIGURES_DIR / "track_c_loglog.png"
    )
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--allow-mixed",
        action="store_true",
        help="render even if the sides of a row came from different runs or inputs",
    )
    problems = check_pairing(allow_mixed=ap.parse_args().allow_mixed)
    for p in problems:
        print(f"WARNING: {p}")
    md = render_md()
    tex = render_tex()
    (RESULTS_DIR / "perf_table.md").write_text(md, encoding="utf-8")
    (RESULTS_DIR / "perf_table.tex").write_text(tex, encoding="utf-8")
    (PAPER_TABLES_DIR / "track_c_perf.tex").write_text(tex, encoding="utf-8")
    fig = render_figure()
    print(
        "OK -- wrote perf_table.{md,tex}, "
        "Paper-JSS/manuscript/tables/track_c_perf.tex, and "
        f"{fig.name}"
    )


if __name__ == "__main__":
    main()
