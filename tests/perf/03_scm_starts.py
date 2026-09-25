"""Track C supplement -- how much of the SCM cost is the multi-start search?

The Track C SCM row times StatsPAI's nested-V solver at its default of six
outer starts (equal, regression-based, four random Dirichlet), against
Synth's single BFGS search from its regression-based start. A reader will
ask how much of the gap the extra starts explain. This script times the
same ADH task on the same shared inputs with ``n_random_starts=0`` (the two
deterministic starts only) and records the post-treatment gap, so the
per-start cost and the effect of dropping the random starts on the solution
are both on file. It is not a Track C row: the reference comparison stays
at the package defaults.

Writes ``results/03_scm_starts_py.json``.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

from _common import time_repeat
from _data import SCM_T0, SIZES, load

import statspai as sp

HERE = Path(__file__).resolve().parent
BASE = dict(
    outcome="y",
    unit="unit_id",
    time="year",
    treated_unit=1,
    treatment_time=SCM_T0,
    method="classic",
    special_predictors=[("y", yr, "mean") for yr in range(1970, SCM_T0)],
    v_method="nested",
    placebo=False,
)
N_REPS = 3


def main() -> None:
    rows = []
    for n_donors in SIZES["03_scm"]:
        df, digest = load("03_scm", n_donors)
        fit6 = sp.synth(df, **BASE)
        fit2 = sp.synth(df, **BASE, n_random_starts=0)
        med, iqr, *_ = time_repeat(
            lambda: sp.synth(df, **BASE, n_random_starts=0), n_reps=N_REPS, warmup=0
        )
        rows.append(
            {
                "n_donors": n_donors,
                "data_sha256": digest,
                "starts": int(fit2.model_info["n_starts"]),
                "median_time_s": med,
                "iqr_time_s": iqr,
                "gap_two_starts": float(fit2.estimate),
                "gap_six_starts": float(fit6.estimate),
                "loss_two_starts": float(fit2.model_info["solver_best_loss"]),
                "loss_six_starts": float(fit6.model_info["solver_best_loss"]),
            }
        )
        print(
            f"  n_donors={n_donors:>4}  two starts={med:.2f}s  "
            f"gap {fit2.estimate:.6f} vs {fit6.estimate:.6f}"
        )
    payload = {
        "statspai_version": sp.__version__,
        "platform": platform.platform(),
        "load_1min": os.getloadavg()[0],
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "rows": rows,
    }
    out = HERE / "results" / "03_scm_starts_py.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
