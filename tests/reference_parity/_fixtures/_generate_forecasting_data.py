"""Simulated series behind the forecasting parity fixtures.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_forecasting_data.py
    Rscript tests/reference_parity/_fixtures/_generate_forecasting_R.R

The first command writes ``forecasting_series.csv`` (the series),
``forecasting_hts_*.csv`` (a small hierarchy) and
``forecasting_sp_ets.json`` (the parameters ``sp.ets`` estimates, which the
R script hands to ``forecast``'s own likelihood code). The second writes
``forecasting_R.json``. Everything is simulated from fixed seeds, so the
fixtures carry no third-party data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def series() -> dict:
    rng = np.random.default_rng(20261006)
    out = {}
    # local level, positive
    e = rng.normal(0, 2.0, 60)
    lvl = 50 + np.cumsum(0.4 * e)
    out["level"] = (lvl + e, 1)
    # damped trend, positive
    n = 70
    b, lev, y = 1.2, 20.0, np.empty(n)
    for t in range(n):
        eps = rng.normal(0, 0.8)
        y[t] = lev + 0.9 * b + eps
        lev = lev + 0.9 * b + 0.6 * eps
        b = 0.9 * b + 0.15 * eps
    out["trend"] = (y, 1)
    # quarterly, multiplicative season and error, trend
    n = 80
    t = np.arange(n)
    seas = np.tile([0.85, 1.05, 1.25, 0.85], n // 4)
    out["quarterly"] = ((120 + 1.1 * t) * seas * (1 + rng.normal(0, 0.03, n)), 4)
    # monthly, additive season, stochastic trend
    n = 144
    t = np.arange(n)
    walk = np.cumsum(rng.normal(0.25, 0.6, n))
    seas = 6.0 * np.sin(2 * np.pi * t / 12) + 3.0 * np.cos(4 * np.pi * t / 12)
    out["monthly"] = (80 + walk + seas + rng.normal(0, 1.0, n), 12)
    # stationary ARMA(2, 1) around a mean
    n = 120
    e = rng.normal(0, 1.0, n + 50)
    x = np.zeros(n + 50)
    for i in range(2, n + 50):
        x[i] = 1.1 * x[i - 1] - 0.4 * x[i - 2] + e[i] + 0.5 * e[i - 1]
    out["arma"] = (10 + x[50:], 1)
    # random walk with drift
    out["walk"] = (200 + np.cumsum(rng.normal(0.3, 1.5, 150)), 1)
    # regression with AR(1) errors
    n = 120
    xreg = rng.normal(0, 1.0, n)
    u = np.zeros(n)
    e = rng.normal(0, 0.7, n)
    for i in range(1, n):
        u[i] = 0.6 * u[i - 1] + e[i]
    out["dyn_y"] = (1.5 + 0.8 * xreg + u, 1)
    out["dyn_x"] = (xreg, 1)
    # two seasonal patterns (periods 8 and 40)
    n = 400
    t = np.arange(n)
    out["multi"] = (
        30
        + 0.02 * t
        + 4.0 * np.sin(2 * np.pi * t / 8)
        + 7.0 * np.cos(2 * np.pi * t / 40)
        + rng.normal(0, 1.0, n),
        8,
    )
    return out


def hierarchy() -> dict:
    rng = np.random.default_rng(42)
    groups = [2, 3, 2]
    nb = sum(groups)
    n = 1 + len(groups) + nb
    S = np.zeros((n, nb))
    S[0] = 1
    c = 0
    for i, g in enumerate(groups):
        S[1 + i, c : c + g] = 1
        c += g
    S[1 + len(groups) :] = np.eye(nb)
    A = rng.normal(size=(n, n))
    res = rng.normal(size=(40, n)) @ A * 0.5 + 0.2
    base = np.abs(rng.normal(50, 10, size=(4, nb))) @ S.T
    base = base * (1 + rng.normal(0, 0.05, size=(4, n)))
    return {"S": S, "res": res, "base": base}


def main() -> None:
    ser = series()
    rows = []
    for name, (y, m) in ser.items():
        for i, v in enumerate(y):
            rows.append({"series": name, "t": i + 1, "period": m, "y": repr(float(v))})
    pd.DataFrame(rows).to_csv(HERE / "forecasting_series.csv", index=False)
    for key, arr in hierarchy().items():
        np.savetxt(HERE / f"forecasting_hts_{key}.csv", arr, delimiter=",", fmt="%.17g")

    import statspai as sp

    fits = {}
    specs = [
        ("level", "ANN", None),
        ("level", "MNN", None),
        ("trend", "AAN", False),
        ("trend", "AAN", True),
        ("trend", "MAN", False),
        ("trend", "MMN", False),
        ("quarterly", "AAA", False),
        ("quarterly", "MAM", False),
        ("quarterly", "MAM", True),
        ("quarterly", "MNM", None),
        ("quarterly", "ANA", None),
        ("quarterly", "MNA", None),
        ("quarterly", "MAA", False),
        ("monthly", "AAA", False),
        ("monthly", "ANA", None),
    ]
    for name, model, damped in specs:
        y, m = ser[name]
        fit = sp.ets(y, model, period=m, damped=damped)
        tag = f"{name}_{model}{'d' if damped else ''}"
        fits[tag] = {
            "series": name,
            "model": model,
            "damped": bool(damped),
            "period": int(m),
            "params": {k: float(v) for k, v in fit.params.items()},
            "initial_state": {k: float(v) for k, v in fit.initial_state.items()},
            "log_likelihood": float(fit.log_likelihood),
        }
    (HERE / "forecasting_sp_ets.json").write_text(
        json.dumps(fits, indent=1), encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows, {len(fits)} ETS fits")


if __name__ == "__main__":
    main()
