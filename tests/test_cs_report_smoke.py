"""End-to-end smoke test for the v0.7.0 Callaway-Sant'Anna workflow.

Exercises the full pipeline introduced in v0.7.0:

    callaway_santanna() -> aggte() -> honest_did() -> cs_report()
                        -> CSReport.plot() / .to_markdown() / .to_latex()

Doubles as a runnable demo: import and call :func:`run_demo` to produce
artifacts (PNG / Markdown / LaTeX) in a specified directory::

    from tests.test_cs_report_smoke import run_demo
    run_demo("/tmp/my_outputs")
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.did import cs_report


# ---------------------------------------------------------------------------
# Data-generating process
# ---------------------------------------------------------------------------

def simulate_staggered_panel(
    n_per_cohort: int = 60,
    cohorts=(4, 7, 10, 0),
    n_periods: int = 12,
    effect_slope: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """Balanced staggered panel with heterogeneous linear dynamic effects.

    Cohort ``g`` is first treated at period ``g`` and the effect at event
    time ``e = t - g`` is ``max(0, e + 1) * effect_slope``.  Cohort 0 is
    never treated.
    """
    rng = np.random.default_rng(seed)
    rows = []
    unit = 0
    for g_val in cohorts:
        for _ in range(n_per_cohort):
            alpha_i = rng.normal(scale=0.30)
            for t in range(1, n_periods + 1):
                te = max(0, t - g_val + 1) * effect_slope if g_val > 0 else 0
                y = alpha_i + 0.15 * t + te + rng.normal()
                rows.append({"unit": unit, "t": t, "g": g_val, "y": y})
            unit += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Runnable demo
# ---------------------------------------------------------------------------

def run_demo(outdir: str = "/tmp", n_boot: int = 500, seed: int = 0):
    """Execute the v0.7.0 workflow end-to-end and save artifacts to ``outdir``.

    Returns the :class:`statspai.did.CSReport` instance.
    """
    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    df = simulate_staggered_panel()
    rpt = cs_report(df, y="y", g="g", t="t", i="unit",
                    n_boot=n_boot, random_state=seed, verbose=False)

    (outdir_p / "cs_report_demo.md").write_text(
        rpt.to_markdown(float_format="%.3f"))
    (outdir_p / "cs_report_demo.tex").write_text(
        rpt.to_latex(float_format="%.3f",
                     caption="CS report on the simulated DGP.",
                     label="tab:cs_demo"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        fig, _ = rpt.plot(suptitle="Callaway-Sant'Anna report — simulated panel")
        fig.savefig(outdir_p / "cs_report_demo_panel.png",
                    dpi=110, bbox_inches="tight")
    except ImportError:
        pass

    return rpt


# ---------------------------------------------------------------------------
# Assertions (pytest)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_report(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("cs_demo")
    return run_demo(str(outdir), n_boot=300, seed=0)


def test_overall_att_is_positive_and_significant(demo_report):
    o = demo_report.overall
    assert o["estimate"] > 0, \
        f"expected positive overall ATT, got {o['estimate']}"
    assert o["pvalue"] < 0.01, \
        f"overall ATT should be highly significant, p = {o['pvalue']}"


def test_all_post_event_coefficients_positive(demo_report):
    post = demo_report.dynamic[demo_report.dynamic["relative_time"] >= 0]
    assert (post["att"] > 0).all(), \
        "dynamic ATT should be positive on this DGP"


def test_dynamic_coefficients_monotonically_increasing(demo_report):
    """Linear ramp DGP → post-event ATT increases with event time."""
    post = demo_report.dynamic[demo_report.dynamic["relative_time"] >= 0]
    post = post.sort_values("relative_time")
    # Allow occasional bootstrap-driven non-monotonicity at one boundary,
    # but the overall Spearman sign should be strongly positive.
    x = post["relative_time"].values
    y = post["att"].values
    rho = np.corrcoef(x, y)[0, 1]
    assert rho > 0.9, f"expected strong monotonic trend, ρ = {rho:.3f}"


def test_earlier_cohorts_have_larger_group_atts(demo_report):
    """Earlier cohorts experience more of the ramp, so θ(g) should
    decrease in g."""
    g_df = demo_report.group.sort_values("group")
    assert g_df["att"].iloc[0] > g_df["att"].iloc[-1], (
        f"expected earliest cohort > latest cohort; got "
        f"{g_df['att'].iloc[0]:.3f} vs {g_df['att'].iloc[-1]:.3f}"
    )


def test_breakdown_M_all_strictly_positive(demo_report):
    assert (demo_report.breakdown["breakdown_M_star"] > 0).all()
    assert demo_report.breakdown["robust_at_1_SE"].all()


def test_exports_generate_expected_content(demo_report):
    md = demo_report.to_markdown()
    tex = demo_report.to_latex()
    assert "Event study" in md and "θ(g)" in md and "θ(t)" in md
    assert "\\begin{table}" in tex and "\\bottomrule" in tex