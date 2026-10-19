"""``sp.mediate_paths`` against R ``paths`` (Zhou and Yamamoto).

Reference: R ``paths`` 0.1.2 (GPL >= 2, used as a black box) on
``_fixtures/mediate_paths.csv`` from ``_fixtures/_generate_mediate_paths_R.R``:
two ordered mediators with and without treatment x mediator interactions,
a two-column first mediator block, a single mediator, and a binary outcome
with logit outcome models; pure imputation and imputation-based weighting
estimators, Type I and Type II decompositions. Point estimates are
deterministic given the fits and are held to 1e-9 with linear outcome
models (observed: 1e-13) and to 1e-7 where a logit is fitted (R ``glm``'s
convergence; observed 7e-9). The bootstrap is
stochastic and is not compared.

A known-truth test checks the estimand: in a linear structural model the
Type I path effects have closed forms (products of path coefficients).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

_SETTINGS = {
    "inter": dict(y="y", mediators=["m1", "m2"]),
    "additive": dict(y="y", mediators=["m1", "m2"], interaction=False),
    "blocks": dict(y="y", mediators=[["m1", "m1b"], "m2"]),
    "single": dict(y="y", mediators=["m2"]),
    "binary": dict(y="yb", mediators=["m1", "m2"], outcome_model="logistic"),
}


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "mediate_paths_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "mediate_paths.csv", float_precision="round_trip")


@pytest.mark.parametrize("ps", [False, True], ids=["pure", "hybrid"])
@pytest.mark.parametrize("setting", sorted(_SETTINGS))
def test_point_estimates_match_paths(ref, data, setting, ps):
    r = sp.mediate_paths(
        data,
        treat="a",
        covariates=["x1", "x2"],
        ps=ps,
        n_boot=0,
        **_SETTINGS[setting],
    )
    est = "hybrid" if ps else "pure"
    # R glm stops at a relative deviance change of 1e-8, which leaves the
    # logit fits (outcome, propensity) about 1e-8 from the maximum.
    rel = 1e-7 if setting == "binary" or ps else 1e-9
    for dec in ("type1", "type2"):
        R = ref[setting][f"{est}_{dec}"]
        ours = r.model_info[dec]
        for k, v in ours.items():
            assert v == pytest.approx(R[k], rel=rel, abs=1e-12), (dec, k)
        assert r.estimate == pytest.approx(R["total"], rel=1e-9)
        assert sum(ours.values()) == pytest.approx(r.estimate, rel=1e-12)


def test_linear_sem_known_truth():
    """Y = 0.4A + 0.5M1 + 0.8M2 + e, M2 = 0.5A + 0.7M1 + e, M1 = 0.6A + e.
    Type I: direct 0.4, via M2 = 0.5 * 0.8 = 0.4, via M1 =
    0.6 * (0.5 + 0.7 * 0.8) = 0.636; total 1.436."""
    rng = np.random.default_rng(3)
    n = 50_000
    a = rng.integers(0, 2, n).astype(float)
    m1 = 0.6 * a + rng.normal(size=n)
    m2 = 0.5 * a + 0.7 * m1 + rng.normal(size=n)
    y = 0.4 * a + 0.5 * m1 + 0.8 * m2 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "a": a, "m1": m1, "m2": m2})
    r = sp.mediate_paths(df, y="y", treat="a", mediators=["m1", "m2"], n_boot=40)
    truth = {"direct": 0.4, "via_M2": 0.4, "via_M1": 0.636}
    det = r.detail[r.detail.decomposition == "type1"].set_index("effect")
    for k, v in truth.items():
        assert abs(r.model_info["type1"][k] - v) < 4 * det.loc[k, "se"], k
    assert abs(r.estimate - 1.436) < 4 * r.se


def test_bootstrap_is_reproducible_and_brackets_estimate(data):
    kw = dict(y="y", treat="a", mediators=["m1", "m2"], covariates=["x1"], n_boot=30)
    r1 = sp.mediate_paths(data, seed=5, **kw)
    r2 = sp.mediate_paths(data, seed=5, **kw)
    pd.testing.assert_frame_equal(r1.detail, r2.detail)
    assert (r1.detail.ci_lower <= r1.detail.ci_upper).all()


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(mediators=[]), "at least one mediator"),
        (dict(mediators=["m1", "m1"]), "two mediator blocks"),
        (dict(mediators=["m1"], outcome_model="logistic"), "0/1"),
        (dict(mediators=["m1"], decomposition="type3"), "decomposition"),
    ],
)
def test_bad_specifications_are_refused(data, kw, match):
    with pytest.raises(sp.MethodIncompatibility, match=match):
        sp.mediate_paths(data, y="y", treat="a", n_boot=0, **kw)


def test_non_binary_treatment_is_refused(data):
    d = data.assign(a=data["a"] + 1)
    with pytest.raises(sp.MethodIncompatibility, match="0/1"):
        sp.mediate_paths(d, y="y", treat="a", mediators=["m1"], n_boot=0)
