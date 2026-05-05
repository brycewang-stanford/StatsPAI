r"""Track B robustness DGPs at B=1000.

Companion to ``run_b1000.py`` — that script reports B=1000 coverage on
the canonical (well-specified) DGPs.  This script re-runs the three
robustness DGPs from ``test_coverage_robustness.py`` at B=1000 and
writes ``results_b1000/coverage_robustness_b1000.json`` for §5.3 of the
JSS draft.

Key difference from the canonical sweep: these rows are *expected* to
miss nominal 0.95 calibration (e.g. weak IV under-covers at ~0.88
because HC1 SEs ignore the weak-instrument bias of 2SLS).  The runner
reports the actual rate; the manuscript discusses the band rather than
asserting calibration.

The DGPs are duplicated between this runner and the pytest module
``test_coverage_robustness.py``.  That duplication is deliberate: the
pytest module asserts a *test-friendly* band measured at B=200/300
(wider Wilson interval), while this runner emits the *manuscript-grade*
explicit-rate JSON at B=1000 with a tighter band.  If you change a
DGP, change both files.

Re-execute end-to-end with::

    python tests/coverage_monte_carlo/run_robustness_b1000.py

Total wall-clock: ~10-15 minutes on Apple M3 Pro / 36 GB.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import statspai as sp


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results_b1000"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

B = 1000


def _ci_covers(ci, truth) -> bool:
    if ci is None or len(ci) != 2:
        return False
    lo, hi = ci
    return lo <= truth <= hi


# ---------------------------------------------------------------------------
# Robustness 1: weak IV
# ---------------------------------------------------------------------------


def coverage_weak_iv() -> dict:
    """2SLS HC1 on weak instrument (pi=0.10, F_med ~ 3)."""
    truth = 1.0
    pi = 0.10
    eu = 1.5
    n = 600
    covered = 0
    Fs = []
    bias = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for seed in range(B):
            rng = np.random.default_rng(seed)
            z = rng.normal(size=n)
            u = rng.normal(size=n)
            d = pi * z + eu * u + rng.normal(scale=0.4, size=n)
            y = truth * d + u + rng.normal(scale=0.5, size=n)
            df = pd.DataFrame({"y": y, "d": d, "z": z})
            # Compute first-stage F directly for record-keeping
            zc = z - z.mean()
            dc = d - d.mean()
            beta_fs = (zc * dc).sum() / (zc * zc).sum()
            d_hat = d.mean() + beta_fs * zc
            ssr = ((d - d_hat) ** 2).sum()
            tss = ((d - d.mean()) ** 2).sum()
            r2 = 1 - ssr / tss
            F = r2 / (1 - r2) * (n - 2)
            Fs.append(F)
            r = sp.ivreg("y ~ (d ~ z)", data=df, robust="hc1")
            beta = float(r.params["d"])
            ci = r.conf_int()
            lo, hi = ci.loc["d"].values
            bias.append(beta - truth)
            if lo <= truth <= hi:
                covered += 1
    return {
        "name": "sp.ivreg (HC1) weak instrument (pi=0.10)",
        "B": B,
        "covered": covered,
        "rate": covered / B,
        "F_median": float(np.median(Fs)),
        "F_mean": float(np.mean(Fs)),
        "abs_bias_median": float(np.median(np.abs(bias))),
        "documented_band": [0.85, 0.95],
    }


# ---------------------------------------------------------------------------
# Robustness 2: heterogeneous-timing CS-DiD
# ---------------------------------------------------------------------------


_CS_HET_TAU_G = {3: 1.0, 5: 2.0, 7: 3.0}


def _cs_het_population_truth() -> float:
    cells = []
    for g in (3, 5, 7):
        for t in range(g, 9):
            cells.append(_CS_HET_TAU_G[g] + 0.5 * (t - g))
    return float(np.mean(cells))


def coverage_cs_heterogeneous() -> dict:
    """Callaway-Sant'Anna on cohort-and-dynamic heterogeneity DGP."""
    truth = _cs_het_population_truth()
    n_units = 200
    T = 8
    cohorts = [3, 5, 7, 0]
    covered = 0
    bias = []
    for seed in range(B):
        rng = np.random.default_rng(seed)
        rows = []
        for i in range(n_units):
            g = cohorts[i % len(cohorts)]
            ui = rng.normal(scale=0.5)
            for t in range(1, T + 1):
                if g > 0 and t >= g:
                    att_eff = _CS_HET_TAU_G[g] + 0.5 * (t - g)
                else:
                    att_eff = 0.0
                y = 0.2 * t + att_eff + ui + rng.normal(scale=0.8)
                rows.append({"i": i, "t": t, "g": g, "y": y})
        df = pd.DataFrame(rows)
        r = sp.callaway_santanna(df, y="y", g="g", t="t", i="i",
                                 estimator="reg")
        bias.append(float(r.estimate) - truth)
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    return {
        "name": "sp.callaway_santanna heterogeneous timing+magnitude",
        "B": B,
        "truth": truth,
        "covered": covered,
        "rate": covered / B,
        "abs_bias_median": float(np.median(np.abs(bias))),
        "documented_band": [0.92, 0.96],
    }


# ---------------------------------------------------------------------------
# Robustness 3: causal forest under propensity overlap loss
# ---------------------------------------------------------------------------


def coverage_causal_forest_overlap() -> dict:
    """Causal forest AIPW on overlap-loss DGP (p in roughly [0.04, 0.96]).

    Capped at B=300 instead of the full B=1000 to keep wall-clock under
    11 minutes; the AIPW IF is the dominant cost.  The 99% Wilson band
    around 0.95 at B=300 is approximately [0.92, 0.97], which is wide
    enough to register the ~0.98 over-coverage we expect under poor
    overlap.
    """
    truth = 1.0
    n = 500
    B_cap = min(B, 300)
    covered = 0
    bias = []
    p_min_avg = []
    p_max_avg = []
    for seed in range(B_cap):
        rng = np.random.default_rng(seed)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        lin = -1.5 + 2.0 * x1
        p = 1.0 / (1.0 + np.exp(-lin))
        p_min_avg.append(float(p.min()))
        p_max_avg.append(float(p.max()))
        d = rng.binomial(1, p)
        y = 0.5 + truth * d + 0.7 * x1 + 0.3 * x2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
        q = sp.causal_question(
            treatment="d", outcome="y", design="causal_forest",
            covariates=["x1", "x2"], data=df,
        )
        r = q.estimate(n_estimators=30, random_state=seed)
        bias.append(float(r.estimate) - truth)
        if r.ci[0] <= truth <= r.ci[1]:
            covered += 1
    return {
        "name": "sp.causal_forest AIPW under overlap loss",
        "B": B_cap,
        "covered": covered,
        "rate": covered / B_cap,
        "p_min_avg": float(np.mean(p_min_avg)),
        "p_max_avg": float(np.mean(p_max_avg)),
        "abs_bias_median": float(np.median(np.abs(bias))),
        "documented_band": [0.85, 0.99],
        "note": "Capped at B=300 (wall-clock cap; AIPW IF dominates).",
    }


def main() -> None:
    out: list[dict] = []
    for fn in [coverage_weak_iv,
               coverage_cs_heterogeneous,
               coverage_causal_forest_overlap]:
        t0 = time.time()
        rec = fn()
        rec["wall_s"] = round(time.time() - t0, 1)
        out.append(rec)
        print(f"  {rec['name']:<60} cov={rec['rate']:.3f}  "
              f"band={rec['documented_band']}  ({rec['wall_s']}s)")
    out_path = RESULTS_DIR / "coverage_robustness_b1000.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"OK -- wrote {out_path}")


if __name__ == "__main__":
    main()
