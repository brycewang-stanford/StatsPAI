"""Track B size & power audit at B=1000 for the core closed-form estimators.

Re-runs the DGPs in ``test_size_power.py`` at B=1000 and writes
``results_b1000/size_power_b1000.json`` so the manuscript size/power table
can be regenerated automatically. RD is capped lower for wall-clock.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statspai as sp

from test_size_power import (  # reuse the exact DGPs the pytest rows use
    _fit_ols,
    _fit_did,
    _fit_iv,
    _fit_rd,
    _fit_panel,
    _fit_cs,
    _fit_ebalance,
)

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results_b1000"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

B = 1000
RD_CAP = 500  # rdrobust is the slowest fit; cap its reps
CS_CAP = 300  # Callaway-Sant'Anna is influence-function based but per-fit slow

# (fit_fn, label, B, power deltas) — deltas[0]=0 is the null/size point.
PLAN = [
    (_fit_ols, "sp.regress (HC1) RCT", B, [0.0, 0.10, 0.20, 0.30]),
    (_fit_did, "sp.did 2x2", B, [0.0, 0.20, 0.40, 0.60]),
    (_fit_iv, "sp.ivreg strong-Z", B, [0.0, 0.20, 0.40, 0.60]),
    (_fit_rd, "sp.rdrobust sharp", RD_CAP, [0.0, 0.20, 0.40, 0.60]),
    (_fit_panel, "sp.panel two-way FE", B, [0.0, 0.15, 0.30, 0.45]),
    (_fit_cs, "sp.callaway_santanna staggered", CS_CAP, [0.0, 0.30, 0.60, 0.90]),
    (_fit_ebalance, "sp.ebalance (conservative)", B, [0.0, 0.40, 0.70, 1.00]),
]


def main() -> None:
    out: list[dict] = []
    for fit_fn, label, b, deltas in PLAN:
        t0 = time.time()
        powers = []
        for delta in deltas:
            rej = sum(fit_fn(seed, delta) for seed in range(b))
            powers.append(round(rej / b, 4))
        rec = {
            "name": label,
            "B": b,
            "size": powers[0],  # delta=0 rejection rate
            "deltas": deltas,
            "power": powers,  # power[i] at deltas[i]
            "wall_s": round(time.time() - t0, 1),
        }
        out.append(rec)
        print(
            f"  {label:<28} size={rec['size']:.3f}  "
            f"power={powers}  ({rec['wall_s']}s)"
        )
    out_path = RESULTS_DIR / "size_power_b1000.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"OK -- wrote {out_path}")


if __name__ == "__main__":
    main()
