"""StatsPAI E-value parity (Python side) -- Module 23.

E-value (VanderWeele & Ding 2017) is a closed-form sensitivity-to-
unmeasured-confounding bound. We test three canonical inputs (a
moderate RR, a strong RR, an OR converted to RR) against
EValue::evalues.RR.

Tolerance: rel < 1e-6 (closed-form formula).
"""
from __future__ import annotations

import statspai as sp

from _common import ParityRecord, write_results


MODULE = "23_evalue"


CASES = [
    # (rr, rr_lower, rr_upper, label)
    (2.5, 1.8, 3.2, "moderate"),
    (4.0, 2.5, 6.0, "strong"),
    (1.3, 1.0, 1.6, "borderline"),
]


def main() -> None:
    rows: list[ParityRecord] = []
    for rr, lo, hi, label in CASES:
        out = sp.evalue(estimate=rr, ci=(lo, hi), measure="RR")
        rows.append(ParityRecord(
            module=MODULE, side="py",
            statistic=f"evalue_est_{label}",
            estimate=float(out["evalue_estimate"]),
            n=1,
        ))
        rows.append(ParityRecord(
            module=MODULE, side="py",
            statistic=f"evalue_ci_{label}",
            estimate=float(out["evalue_ci"]),
            n=1,
        ))

    write_results(MODULE, "py", rows,
                  extra={"measure": "RR", "n_cases": len(CASES)})


if __name__ == "__main__":
    main()
