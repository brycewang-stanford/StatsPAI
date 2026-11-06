"""Default ``sp.synth`` on the Basque replica against R ``Synth``.

Track A module 07 runs the nested V search on these bytes, where R
``Synth`` and Stata ``synth`` stop at different local optima, so that row
is a T4 disclosure. The default call has no predictors and equal V: the
donor weights are simplex least squares on the fifteen pre-period
outcomes, a convex problem. Here every optimum it solves (Basque Country
and the sixteen in-space placebos) is certified the unique one, so the
default configuration has a strict comparison on the real data, not only
on module 52's constructed design.

R ``Synth`` runs the same problem with ``custom.v`` set to the predictor
variances, which undoes its standardisation
(``_fixtures/_generate_synth_basque_equal_v_R.R``). Its QP solver
``ipop`` does not solve every placebo at its tight settings; the setting
that solved is recorded per placebo, and the test holds each placebo to
the precision that setting attains. Where the two differ, our weights
attain the lower objective, so the residual is ipop's.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.synth._core import simplex_weights_unique

_HERE = pathlib.Path(__file__).parent
_DATA = _HERE.parent / "r_parity" / "data" / "07_scm.csv"
_REF = _HERE / "_fixtures" / "synth_basque_equal_v_R.json"
T0 = 1970
#: The CLAUDE.md section 5.1 budget.
RTOL = 1e-6


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_DATA)


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads(_REF.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fit(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.synth(
            data,
            outcome="gdppc",
            unit="region",
            time="year",
            treated_unit="Basque Country",
            treatment_time=T0,
        )


def _problem(data, unit):
    wide = data.pivot(index="year", columns="region", values="gdppc")
    pre = wide.index < T0
    donors = [c for c in wide.columns if c != unit]
    return wide.loc[pre, unit].to_numpy(), wide.loc[pre, donors].to_numpy(), donors


def _loss(y, X, w):
    r = y - X @ w
    return float(r @ r)


def test_default_is_the_equal_v_configuration(fit):
    assert fit.model_info["v_method"] == "equal"
    assert fit.model_info["weight_uniqueness_method"] == "kkt"
    assert fit.model_info["weight_solution_nonunique"] is False


def test_every_optimum_is_certified_unique(data, fit):
    y, X, donors = _problem(data, "Basque Country")
    table = fit.model_info["weights"].set_index("unit")["weight"]
    w = np.array([float(table.get(d, 0.0)) for d in donors])
    # model_info['weights'] drops weights below 1e-6; renormalise them away
    assert simplex_weights_unique(y, X, w / w.sum()) is True
    placebo = fit.model_info["placebo_weights"]
    assert len(placebo) == 16
    for unit in placebo.index:
        y, X, donors = _problem(data, unit)
        w = placebo.loc[unit, donors].to_numpy(dtype=float)
        assert simplex_weights_unique(y, X, w) is True, unit


def test_estimate_and_weights_match_r_synth(fit, ref):
    assert ref["treated"]["sigf_ipop"] >= 12
    assert fit.estimate == pytest.approx(ref["treated"]["att"], rel=1e-9)
    ours = fit.model_info["weights"].set_index("unit")["weight"]
    for donor, w in ref["treated"]["weights"].items():
        assert float(ours.get(donor, 0.0)) == pytest.approx(w, abs=1e-6), donor
    assert fit.model_info["pre_treatment_rmse"] == pytest.approx(
        ref["treated"]["pre_rmspe"], abs=1e-6  # model_info rounds to 6 places
    )


def test_placebo_effects_match_r_synth(data, fit, ref):
    atts = dict(
        zip(fit.model_info["placebo_weights"].index, fit.model_info["placebo_atts"])
    )
    table = fit.model_info["placebo_weights"]
    for unit, p in ref["placebos"].items():
        # ipop's attainable precision at the setting that solved
        atol = 1e-6 if p["sigf_ipop"] >= 12 else 1e-4
        assert atts[unit] == pytest.approx(p["att"], abs=atol), unit
        y, X, donors = _problem(data, unit)
        w_r = np.array([p["weights"][d] for d in donors])
        w_r = np.clip(w_r, 0.0, None) / np.clip(w_r, 0.0, None).sum()
        w_py = table.loc[unit, donors].to_numpy(dtype=float)
        # where they differ, ours is the better optimum
        assert _loss(y, X, w_py) <= _loss(y, X, w_r) * (1 + 1e-12), unit


def test_se_is_the_placebo_sd_and_matches_r(fit, ref):
    atts = np.asarray(fit.model_info["placebo_atts"], dtype=float)
    assert fit.se == pytest.approx(float(np.std(atts)), rel=1e-14)
    assert fit.se == pytest.approx(ref["sd_population"], rel=RTOL)


def test_validation_scope_credits_this_artifact(fit):
    scope = sp.validation_scope(fit)
    assert scope["status"] == "covered"
    for out in ("estimate", "se"):
        artifacts = {e["artifact"] for e in scope["outputs"][out]["evidence"]}
        assert (
            "tests/reference_parity/test_synth_basque_equal_v_parity.py" in artifacts
        ), out
