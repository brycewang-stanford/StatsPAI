"""StatsPAI E-value parity (Python side) -- Module 23.

Exercises sp.evalue / sp.evalue_rd across EVERY effect measure supported
by the R ``EValue`` package and writes, for each case, the E-value for
the point estimate and the E-value for the confidence limit closest to
the null.  The R companion (23_evalue.R) runs ``EValue::evalues.*`` on
the identical inputs; ``compare.py`` joins them.

Measures covered: RR (incl. protective, CI-crosses-null, and a non-null
``true``), OR and HR under both the rare and common-outcome conversions,
the standardised mean difference (MD/SMD), an OLS coefficient
standardised by the outcome SD, and the exact 2x2-table risk difference.

Tolerance: rel < 1e-6 (closed-form / deterministic grid).
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, write_results


MODULE = "23_evalue"


# Each case yields (evalue_est_<label>, evalue_ci_<label>). ``fn`` returns
# the StatsPAI result dict.
CASES = [
    # The first three keep their original labels so the committed Stata
    # anchor (23_evalue.do) still joins in compare.py.
    ("moderate",      lambda: sp.evalue(estimate=2.5, ci=(1.8, 3.2), measure="RR")),
    ("strong",        lambda: sp.evalue(estimate=4.0, ci=(2.5, 6.0), measure="RR")),
    ("borderline",    lambda: sp.evalue(estimate=1.3, ci=(1.0, 1.6), measure="RR")),
    ("rr_protective", lambda: sp.evalue(estimate=0.6, ci=(0.4, 0.9), measure="RR")),
    ("rr_crossnull",  lambda: sp.evalue(estimate=1.1, ci=(0.9, 1.3), measure="RR")),
    ("rr_nonnull",    lambda: sp.evalue(estimate=2.5, ci=(1.8, 3.2), measure="RR", true=1.5)),
    ("or_common",     lambda: sp.evalue(estimate=2.0, ci=(1.5, 2.7), measure="OR", rare=False)),
    ("or_rare",       lambda: sp.evalue(estimate=2.0, ci=(1.5, 2.7), measure="OR", rare=True)),
    ("hr_common",     lambda: sp.evalue(estimate=1.5, ci=(1.1, 2.0), measure="HR", rare=False)),
    ("hr_rare",       lambda: sp.evalue(estimate=1.5, ci=(1.1, 2.0), measure="HR", rare=True)),
    ("md",            lambda: sp.evalue(estimate=0.3, se=0.1, measure="MD")),
    ("ols",           lambda: sp.evalue(estimate=0.5, se=0.1, sd=2.0, delta=1.0, measure="OLS")),
    ("rd",            lambda: sp.evalue_rd(200, 150, 100, 250)),
]


def main() -> None:
    rows: list[ParityRecord] = []
    for label, fn in CASES:
        out = fn()
        rows.append(ParityRecord(
            module=MODULE, side="py",
            statistic=f"evalue_est_{label}",
            estimate=float(out["evalue_estimate"]), n=1,
        ))
        ci = out.get("evalue_ci")
        rows.append(ParityRecord(
            module=MODULE, side="py",
            statistic=f"evalue_ci_{label}",
            estimate=None if ci is None else float(ci), n=1,
        ))

    write_results(MODULE, "py", rows,
                  extra={"reference": "R EValue package",
                         "n_cases": len(CASES)})
    print(f"[{MODULE}] wrote {len(rows)} rows across {len(CASES)} measures")


if __name__ == "__main__":
    main()
