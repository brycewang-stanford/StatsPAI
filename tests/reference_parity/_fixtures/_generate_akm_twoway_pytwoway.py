"""pytwoway reference for ``sp.akm`` (AKM + Andrews et al. + KSS).

pytwoway 0.3.21 / bipartitepandas 1.1.9 (Lamadon and coauthors, MIT
licence), installed in a throwaway virtual environment -- never a StatsPAI
dependency. pytwoway 0.3.21 needs the older scientific stack::

    uv venv -p 3.11 /tmp/twvenv
    VIRTUAL_ENV=/tmp/twvenv uv pip install pytwoway==0.3.21 bipartitepandas==1.1.9 \
        "numpy==1.26.4" "pandas==2.1.4" "scipy==1.11.4" "pyamg==5.0.1" "setuptools<70"
    /tmp/twvenv/bin/python tests/reference_parity/_fixtures/_generate_akm_twoway_pytwoway.py

Input: ``akm_twoway.csv`` (``_generate_akm_twoway_data.py``). The outcome
is ``lnw`` itself (pytwoway's FE estimator has no covariates). Settings:

* cleaning: ``connectedness='leave_out_observation'`` on long (not
  collapsed) data, so the estimation sample is the largest set (by firms)
  in which every observation can be left out without disconnecting it;
* estimator: unweighted, exact leverages (``exact_lev_he``); stayers'
  variances use their own leave-one-out residuals
  (``Sii_stayers='upper_bound'``), which is the observation-level KSS
  formula for every row.

Two runs are stored. ``exact`` uses pytwoway's exact traces
(``exact_trace_he`` / ``exact_trace_ho``). Its var(alpha) traces add the
diagonal block ``(W'W)^{-1}`` as a vector (``Dwinv`` is 1-D, so ``Dwinv +
C @ Minv @ B`` broadcasts it along rows instead of adding a diagonal
matrix); under homoskedasticity that drops exactly ``(n_workers - 1) / n``
from the trace, and the KSS var(alpha) and cov traces are affected too.
``hutchinson`` uses the stochastic traces (4000 Rademacher draws each),
which go through the linear solver and not those blocks; their Monte Carlo
standard errors are stored (``*_sd`` over draws).

Writes ``akm_twoway_pytwoway.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import bipartitepandas as bpd
import numpy as np
import pandas as pd
import pytwoway as tw

HERE = Path(__file__).parent
KEYS = (
    "var(y)",
    "var(eps)_fe",
    "var(eps)_ho",
    "var(eps)_he",
    "var(psi)_fe",
    "var(alpha)_fe",
    "cov(psi, alpha)_fe",
    "var(psi)_ho",
    "var(alpha)_ho",
    "cov(psi, alpha)_ho",
    "var(psi)_he",
    "var(alpha)_he",
    "cov(psi, alpha)_he",
    "min_lev",
    "max_lev",
)
DRAWS = 4000


def _fit(bdf, exact: bool):
    params = {
        "he": True,
        "ho": True,
        "weighted": False,
        "exact_lev_he": True,
        "exact_trace_he": exact,
        "exact_trace_ho": exact,
        "Sii_stayers": "upper_bound",
        "Q_var": [tw.Q.VarPsi(), tw.Q.VarAlpha()],
        "Q_cov": [tw.Q.CovPsiAlpha()],
        "attach_fe_estimates": True,
        "preconditioner": "jacobi",
        "progress_bars": False,
        "verbose": False,
    }
    if not exact:
        params["ndraw_trace_he"] = DRAWS
        params["ndraw_trace_ho"] = DRAWS
    est = tw.FEEstimator(bdf, tw.fe_params(params))
    est.fit(rng=np.random.default_rng(1234))
    out = {k: float(est.res[k]) for k in KEYS}
    if not exact:
        for k in list(est.res):
            if k.startswith("tr_") and k.endswith("_sd"):
                out[k] = float(est.res[k])
        out["ndraw_trace"] = DRAWS
    return est, out


def main() -> None:
    d = pd.read_csv(HERE / "akm_twoway.csv")
    bdf = bpd.BipartiteDataFrame(
        i=d.worker, j=d.firm, y=d.lnw, t=d.year, track_id_changes=True
    )
    clean = bpd.clean_params(
        {
            "connectedness": "leave_out_observation",
            "collapse_at_connectedness_measure": False,
            "drop_single_stayers": True,
            "verbose": False,
        }
    )
    bdf = bdf.clean(clean)
    ids = bdf.original_ids()
    est, exact = _fit(bdf.copy(), exact=True)
    _, hutch = _fit(bdf.copy(), exact=False)
    a = est.adata
    out = {
        "package": "pytwoway 0.3.21 / bipartitepandas 1.1.9",
        "rows": {
            "worker": ids["original_i"].astype(int).tolist(),
            "firm": ids["original_j"].astype(int).tolist(),
            "year": a["t"].astype(int).tolist(),
            "alpha": a["alpha_hat"].astype(float).tolist(),
            "psi": a["psi_hat"].astype(float).tolist(),
        },
        "n_obs": int(len(a)),
        "n_workers": int(est.res["n_workers"]),
        "n_firms": int(est.res["n_firms"]),
        "n_movers": int(est.res["n_movers"]),
        "exact": exact,
        "hutchinson": hutch,
    }
    with open(HERE / "akm_twoway_pytwoway.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
