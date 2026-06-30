"""Fixed-seed flagship-estimator cases for the version golden-master.

Each case fits one flagship estimator on a *deterministic* (seeded) dataset and
returns a flat dict of headline numbers (point estimate + SE). These values are
pinned in ``golden_values.json`` and re-checked on every run, so a future
StatsPAI version that changes a headline output **without a declared
correctness fix** turns the build red. This is the user-facing reproducibility
guarantee made mechanical — distinct from ``tests/reference_parity`` (which
checks agreement with R/Stata): here we check agreement with *our own past
selves* across versions.

A case function must be pure and deterministic: same seed in, same numbers out.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict

import numpy as np
import pandas as pd

import statspai as sp


def _ols_data(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.5).astype(float)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + 2.0 * d + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "d": d})


def case_regress() -> Dict[str, float]:
    df = _ols_data()
    r = sp.regress("y ~ x1 + x2 + d", data=df, robust="hc1")
    return {
        "coef_d": float(r.params["d"]),
        "se_d": float(r.std_errors["d"]),
        "coef_x1": float(r.params["x1"]),
    }


def case_feols_twoway() -> Dict[str, float]:
    rng = np.random.default_rng(11)
    n_unit, n_time = 60, 8
    rows = []
    for i in range(n_unit):
        a = rng.normal()
        for t in range(n_time):
            g = (i + t) % 3
            x = 0.4 * a + rng.normal()
            y = 1.5 * x + a + 0.2 * t + rng.normal()
            rows.append({"y": y, "x": x, "unit": i, "time": t, "g": g})
    df = pd.DataFrame(rows)
    r = sp.feols("y ~ x | unit + time", data=df)
    return {"coef_x": float(r.params["x"]), "se_x": float(r.std_errors["x"])}


def case_ivreg_2sls() -> Dict[str, float]:
    rng = np.random.default_rng(13)
    n = 800
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = 0.8 * z + u + rng.normal(size=n)
    y = 1.0 * d + u + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "z": z})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ivreg("y ~ (d ~ z)", data=df)
    return {"coef_d": float(r.params["d"]), "se_d": float(r.std_errors["d"])}


def case_did_2x2() -> Dict[str, float]:
    rng = np.random.default_rng(17)
    n = 600
    treat = (rng.uniform(size=n) < 0.5).astype(int)
    post = (rng.uniform(size=n) < 0.5).astype(int)
    y = 1.0 + 0.3 * post + 0.5 * treat + 2.0 * (treat * post) + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "treat": treat, "post": post})
    r = sp.regress("y ~ treat + post + treat:post", data=df)
    return {
        "att": float(r.params["treat:post"]),
        "se_att": float(r.std_errors["treat:post"]),
    }


def case_logit() -> Dict[str, float]:
    rng = np.random.default_rng(19)
    n = 700
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.5 + 1.0 * x)))
    y = (rng.uniform(size=n) < p).astype(int)
    df = pd.DataFrame({"y": y, "x": x})
    r = sp.logit("y ~ x", data=df)
    return {"coef_x": float(r.params["x"]), "se_x": float(r.std_errors["x"])}


#: Registry of golden-master cases. Keep names stable — they are the JSON keys.
CASES: Dict[str, Callable[[], Dict[str, float]]] = {
    "regress": case_regress,
    "feols_twoway": case_feols_twoway,
    "ivreg_2sls": case_ivreg_2sls,
    "did_2x2": case_did_2x2,
    "logit": case_logit,
}


def compute_all() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, fn in CASES.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[name] = fn()
    return out
