"""Diagnostic coverage — no estimator may silently skip a diagnostic.

A ``result.violations()`` check only fires if the estimator populated the
``model_info`` key it reads. When one IV entry point stores ``first_stage_f``
and another does not, the weak-instrument warning silently vanishes for the
second — exactly the kind of gap that erodes trust (it was real for ``sp.liml``
and ``sp.jive`` until fixed). This suite pins the whole IV family: every
estimator must record the first-stage strength and flag a weak instrument, and
none may cry wolf on a strong one. A new IV estimator that forgets the
diagnostic fails here instead of shipping a blind spot.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _iv_df(first_stage_coef: float, n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = first_stage_coef * z + u + rng.normal(size=n)
    y = 1.0 * d + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z})


# (id, fitter) — every single-endogenous IV estimator StatsPAI exposes.
_IV_ESTIMATORS = [
    ("ivreg_2sls", lambda df: sp.ivreg("y ~ (d ~ z)", data=df)),
    ("iv_2sls", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="2sls")),
    ("iv_liml", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="liml")),
    ("iv_fuller", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="fuller")),
    ("iv_gmm", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="gmm")),
    ("liml", lambda df: sp.liml("y ~ (d ~ z)", data=df)),
    ("jive", lambda df: sp.jive(df, y="y", x_endog=["d"], z=["z"])),
]


@pytest.mark.parametrize("name,fit", _IV_ESTIMATORS, ids=[e[0] for e in _IV_ESTIMATORS])
def test_iv_estimator_records_first_stage_and_flags_weak(name, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weak = fit(_iv_df(0.03))
        strong = fit(_iv_df(2.0))

    assert weak.model_info.get("first_stage_f") is not None, (
        f"{name}: model_info['first_stage_f'] is missing — weak IV would be "
        "silently skipped by result.violations()"
    )
    assert "weak_instrument" in {
        v["test"] for v in weak.violations()
    }, f"{name}: a weak first stage did not surface in violations()"
    assert "weak_instrument" not in {
        v["test"] for v in strong.violations()
    }, f"{name}: false-positive weak-instrument flag on a strong first stage"
