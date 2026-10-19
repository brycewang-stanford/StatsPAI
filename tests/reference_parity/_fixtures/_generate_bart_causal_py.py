"""Python side of the sp.bart_causal seed screen.

Run::

    python tests/reference_parity/_fixtures/_generate_bart_causal_py.py

Writes ``bart_causal_py.json``: for each method ('bcf', 'bart') and the
eight seeds ``1000 + 100000 k`` of ``_generate_bart_causal_R.R``, the
posterior mean and sd of the ATE and the posterior-mean CATE, with the
same data, true propensity and chain length (one chain, 1000 burn-in,
1000 kept) as the R side.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

import statspai as sp

HERE = pathlib.Path(__file__).parent


def main() -> None:
    d = pd.read_csv(HERE / "bart_causal_data.csv")
    out = {}
    for method in ("bcf", "bart"):
        rows = []
        for k in range(1, 9):
            s = 1000 + 100000 * k
            r = sp.bart_causal(
                d,
                "y",
                "z",
                ["x1", "x2", "x3"],
                method=method,
                propensity="pi",
                draws=1000,
                burnin=1000,
                chains=1,
                seed=s,
            )
            rows.append(
                {
                    "seed": s,
                    "ate_mean": r.estimate,
                    "ate_sd": r.se,
                    "cate_mean": r.cate["mean"].tolist(),
                }
            )
        out[method] = rows
    out["versions"] = {"statspai": sp.__version__}
    (HERE / "bart_causal_py.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    print("wrote bart_causal_py.json")


if __name__ == "__main__":
    main()
