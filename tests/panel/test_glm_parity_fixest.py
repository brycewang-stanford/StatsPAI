"""Panel GLM estimators must stay pinned to fixest.

Regression lock for the module-67 parity rows: ``sp.feglm(family="logit")``
reproduces ``fixest::feglm`` and ``sp.fepois`` reproduces ``fixest::fepois``
on both coefficients and IID standard errors, with a single entity fixed
effect (``id``) absorbed by both sides.

Reference values are the committed R goldens from
``tests/r_parity/67_panel_glm.R`` (fixest, see ``renv.lock``). This test
regenerates the two balanced panel DGPs and refits the Python estimators so
a silent regression fails here, not only when the parity harness is rebuilt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai import feglm, fepois

_R_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "r_parity"
    / "results"
    / "67_panel_glm_R.json"
)


def _make_logit(seed: int = 42, N: int = 30, T: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(N), T)
    years = np.tile(np.arange(T), N)
    x1 = rng.normal(size=N * T)
    x2 = rng.normal(size=N * T)
    fe = rng.normal(0, 0.5, N)[ids]
    z = 0.2 + 0.4 * x1 - 0.3 * x2 + fe + rng.normal(size=N * T)
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=N * T) < p).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "id": ids, "year": years})


def _make_poisson(seed: int = 43, N: int = 25, T: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(N), T)
    years = np.tile(np.arange(T), N)
    x1 = rng.normal(size=N * T)
    x2 = rng.normal(size=N * T)
    fe = rng.normal(0, 0.5, N)[ids]
    mu = np.exp(0.5 + 0.3 * x1 - 0.2 * x2 + fe + rng.normal(size=N * T) * 0.5)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "id": ids, "year": years})


def _r_reference():
    rows = json.loads(_R_GOLDEN.read_text(encoding="utf-8"))["rows"]
    return {r["statistic"]: r for r in rows}


def test_feglm_logit_matches_fixest():
    df = _make_logit()
    fit = feglm("y ~ x1 + x2 | id", data=df, family="logit")
    ref = _r_reference()
    for key in ["x1", "x2"]:
        r = ref[f"feglm_logit_{key}"]
        assert float(fit.params[key]) == pytest.approx(r["estimate"], rel=1e-5), key
        assert float(fit.std_errors[key]) == pytest.approx(r["se"], rel=5e-5), key


def test_fepois_matches_fixest():
    df = _make_poisson()
    fit = fepois("y ~ x1 + x2 | id", data=df)
    ref = _r_reference()
    for key in ["x1", "x2"]:
        r = ref[f"fepois_{key}"]
        assert float(fit.params[key]) == pytest.approx(r["estimate"], rel=1e-5), key
        assert float(fit.std_errors[key]) == pytest.approx(r["se"], rel=5e-5), key
