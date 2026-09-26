"""Recommendation cards declare whether they can run (review F05, 2026-09-26).

Before: four frontier benchmark cases recommended the right estimator and then
crashed inside ``.run()`` with a bare ``TypeError`` (Bartik without
shares/shocks, DDD without subgroup) or a 2-period check (repeated
cross-sections with four periods). A card now carries ``ready`` /
``missing_arguments`` / ``blocked`` and ``.run()`` refuses with a message
naming the missing input.
"""

import filecmp
import warnings
from pathlib import Path

import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_every_card_has_readiness_fields():
    df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=1)
    rec = sp.recommend(df, y="y", treatment="treated", id="unit", time="time")
    for card in rec.recommendations:
        assert isinstance(card["ready"], bool)
        assert isinstance(card["missing_arguments"], list)


def test_ddd_without_subgroup_is_not_ready_and_run_names_it():
    df = sp.dgp_did(n_units=200, n_periods=2, seed=3, ddd=True)
    rec = sp.recommend(df, y="y", treatment="group", time="time", design="ddd")
    top = rec.recommendations[0]
    assert top["function"] == "ddd"
    assert top["ready"] is False
    assert top["missing_arguments"] == ["subgroup"]
    with pytest.raises(MethodIncompatibility, match="subgroup"):
        rec.run()
    # supplying it at run time works
    res = rec.run(subgroup="subgroup")
    assert abs(res.estimate - 0.5) < 4 * res.se


def test_ddd_with_subgroup_recovers_effect():
    df = sp.dgp_did(n_units=400, n_periods=2, seed=37, ddd=True)
    rec = sp.recommend(
        df, y="y", treatment="group", time="time", subgroup="subgroup", design="ddd"
    )
    assert rec.recommendations[0]["ready"]
    res = rec.run()
    assert abs(res.estimate - 0.5) < 3 * res.se


def test_bartik_card_runs_with_array_shares_and_shocks():
    out = sp.dgp_bartik(
        n_regions=400, n_industries=20, effect=0.5, seed=39, endogenous=True
    )
    bare = sp.recommend(out["data"], y="y", treatment="x", design="bartik")
    assert set(bare.recommendations[0]["missing_arguments"]) == {"shares", "shocks"}
    rec = sp.recommend(
        out["data"],
        y="y",
        treatment="x",
        design="bartik",
        shares=out["shares"],
        shocks=out["shocks"],
    )
    res = rec.run()
    assert abs(res.params["x"] - 0.5) < 3 * res.std_errors["x"]


def test_rcs_multi_period_group_indicator_builds_post():
    df = sp.dgp_did(n_units=200, n_periods=4, seed=41).drop(columns="unit")
    rec = sp.recommend(
        df, y="y", treatment="group", time="time", treat_time=2, design="did"
    )
    top = rec.recommendations[0]
    assert top["ready"]
    assert ">= 2" in top["code"]
    res = rec.run()
    # pooled 2x2 on the explicit post column == hand-built regression
    manual = df.assign(post=(df["time"] >= 2).astype(int))
    ref = sp.did(manual, y="y", treat="group", time="post")
    assert res.estimate == pytest.approx(ref.estimate, abs=1e-12)
    assert res.se == pytest.approx(ref.se, abs=1e-12)


def test_rcs_switch_indicator_is_reported_not_identified():
    df = sp.dgp_did(n_units=200, n_periods=4, seed=41).drop(columns="unit")
    rec = sp.recommend(df, y="y", treatment="treated", time="time", design="did")
    top = rec.recommendations[0]
    assert top["ready"] is False
    assert "not identified" in top["blocked"]
    assert "treat_time=2" in top["blocked_hint"]
    with pytest.raises(MethodIncompatibility, match="cannot run"):
        rec.run()
    assert "NOT RUNNABLE" in rec.summary()


def test_dgp_did_default_draws_unchanged_by_ddd_option():
    a = sp.dgp_did(n_units=30, n_periods=5, seed=9)
    b = sp.dgp_did(n_units=30, n_periods=5, seed=9, ddd=False)
    assert a.equals(b)
    c = sp.dgp_did(n_units=30, n_periods=5, seed=9, ddd=True)
    assert (c[["unit", "time", "group"]] == a[["unit", "time", "group"]]).all().all()


def test_packaged_corpus_matches_benchmark_corpus():
    """The wheel ships a byte copy of the corpus so recommend_benchmark works
    after ``pip install``; the source-of-truth lives under benchmarks/."""
    assert filecmp.cmp(
        ROOT / "benchmarks" / "recommend_hit_rate" / "corpus.yaml",
        ROOT / "src" / "statspai" / "smart" / "data" / "recommend_corpus.yaml",
        shallow=False,
    ), (
        "copy benchmarks/recommend_hit_rate/corpus.yaml to "
        "src/statspai/smart/data/recommend_corpus.yaml"
    )
