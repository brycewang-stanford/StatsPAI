"""Design guardrails on the regression entry points (core/_design_guardrails.py).

Two designs make ``sp.feols`` / ``fepois`` / ``regress`` / ``hdfe_ols`` /
``panel`` answer a different question than the caller asked, silently:

* a staggered 0/1 treatment under unit and time fixed effects (part of the
  Goodman-Bacon weight sits on already-treated controls);
* a treatment assigned at a level (state, state x period) coarser than the
  standard errors (HC1, or clusters nested in it).

Each test asserts a true proposition: the reported weight share equals the
one ``sp.bacon_decomposition`` computes from the 2x2 regressions, the
coefficient equals the TWFE estimate the decomposition reports (numbers are
untouched), and the checks stay silent on the clean designs.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core._design_guardrails import _absorbing, _codes, bacon_forbidden_share
from statspai.core._spec_counter import MULTIPLICITY_INFO, reset_specification_counter
from statspai.exceptions import AssumptionWarning

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _fresh_counter():
    reset_specification_counter()
    yield
    reset_specification_counter()


def _tests(result):
    return [v["test"] for v in result.violations()]


def _one(result, test):
    hits = [v for v in result.violations() if v["test"] == test]
    assert len(hits) == 1, result.violations()
    return hits[0]


def _staggered(n_units=120, T=10, cohorts=(4, 6, 8, 0), seed=0, het=True):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = cohorts[i % len(cohorts)]
        a = rng.normal()
        for t in range(1, T + 1):
            d = int(g > 0 and t >= g)
            eff = (1.0 + 0.2 * (t - g)) if (d and het) else float(d)
            rows.append((i, t, g, d, a + 0.1 * t + eff + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["unit", "time", "g", "treated", "y"])


def _state_panel(seed=1, staggered=False, within=False):
    """Individuals in 40 states; treatment by state (or by person)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(40):
        # treated states alternate, so no region (5 states) is all treated
        adopt = (4 + s % 3 if staggered else 6) if s % 2 == 0 else 0
        for i in range(12):
            uid = s * 100 + i
            for t in range(1, 9):
                if within:
                    d = int(i % 2 == 0 and t >= 5)
                else:
                    d = int(adopt > 0 and t >= adopt)
                rows.append((uid, s, s // 5, t, d))
    df = pd.DataFrame(rows, columns=["id", "state", "region", "year", "treated"])
    shock = rng.normal(0, 1, (40, 9))
    df["y"] = 0.5 * df.treated + shock[df.state, df.year] + rng.normal(0, 1, len(df))
    return df


# ---------------------------------------------------------------------- #
#  The weight share is the Goodman-Bacon decomposition's
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(12))
def test_forbidden_share_equals_bacon_decomposition(seed):
    rng = np.random.default_rng(seed)
    T = int(rng.integers(4, 12))
    dates = list(
        rng.choice(np.arange(2, T + 1), size=int(rng.integers(1, 4)), replace=False)
    )
    dates += [0] * int(seed % 2 == 0) + [1] * int(seed % 3 == 0)  # never / always
    rows = []
    for i in range(int(rng.integers(12, 40))):
        g = dates[i % len(dates)]
        for t in range(1, T + 1):
            rows.append((i, t, int(g > 0 and t >= g), rng.normal()))
    df = pd.DataFrame(rows, columns=["id", "t", "d", "y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sp.bacon_decomposition(df, y="y", treat="d", time="t", id="id")
    tcode, n_t = _codes(df["t"], sort=True)
    ucode, n_u = _codes(df["id"])
    g = _absorbing(df["d"].to_numpy(float), ucode, n_u, tcode, n_t)
    mine = 0.0 if g is None else bacon_forbidden_share(g, tcode, n_t)[0]
    # identity: the same weights, from counts instead of 2x2 regressions
    assert mine == pytest.approx(ref["already_treated_control_weight_share"], abs=1e-12)


# ---------------------------------------------------------------------- #
#  Staggered adoption under TWFE: fires on every entry point
# ---------------------------------------------------------------------- #


def _fits(df):
    return {
        "feols": lambda: sp.feols("y ~ treated | unit + time", df, cluster="unit"),
        "hdfe_ols": lambda: sp.hdfe_ols(
            "y ~ treated | unit + time", df, cluster="unit"
        ),
        "regress": lambda: sp.regress(
            "y ~ treated + C(unit) + C(time)", df, cluster="unit"
        ),
        "panel": lambda: sp.panel(
            df,
            "y ~ treated",
            entity="unit",
            time="time",
            method="twoway",
            cluster="entity",
        ),
        "fepois": lambda: sp.fepois(
            "count ~ treated | unit + time",
            df.assign(count=np.round(np.exp(df["y"] / 3))),
            cluster="unit",
        ),
    }


@pytest.mark.parametrize("entry", ["feols", "hdfe_ols", "regress", "panel", "fepois"])
def test_staggered_twfe_is_flagged_with_the_bacon_share(entry):
    df = _staggered()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sp.bacon_decomposition(df, y="y", treat="treated", time="time", id="unit")
    with pytest.warns(AssumptionWarning, match="already-treated"):
        res = _fits(df)[entry]()
    v = _one(res, "staggered_twfe")
    assert v["value"] == pytest.approx(
        ref["already_treated_control_weight_share"], abs=1e-6
    )
    assert v["details"] == {"unit": "unit", "time": "time", "n_cohorts": 3}
    assert v["next_calls"][0]["tool"] == "did"
    assert v["next_calls"][0]["arguments"] == {
        "y": "count" if entry == "fepois" else "y",
        "treat": "treated",
        "time": "time",
        "id": "unit",
    }
    if entry != "fepois":
        # the guardrail never touches the estimate: it is the TWFE coefficient
        # the decomposition reports
        assert float(res.params["treated"]) == pytest.approx(ref["beta_twfe"], rel=1e-9)


def test_staggered_with_always_treated_units_is_flagged():
    df = _staggered(cohorts=(1, 6, 0))  # cohort 1 is treated from the start
    with pytest.warns(AssumptionWarning):
        res = sp.feols("y ~ treated | unit + time", df, cluster="unit")
    assert _one(res, "staggered_twfe")["value"] > 0


def test_datetime_time_column_is_oriented_as_time():
    df = _staggered()
    df["time"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(
        (df["time"] - 1) * 365, unit="D"
    )
    with pytest.warns(AssumptionWarning):
        res = sp.feols("y ~ treated | time + unit", df, cluster="unit")
    assert _one(res, "staggered_twfe")["details"]["time"] == "time"


@pytest.mark.parametrize(
    "fit",
    [
        # one adoption date and never-treated units: no forbidden comparison
        lambda: sp.feols(
            "y ~ treated | unit + time", _staggered(cohorts=(6, 0)), cluster="unit"
        ),
        lambda: sp.hdfe_ols(
            "y ~ treated | unit + time", _staggered(cohorts=(6, 0)), cluster="unit"
        ),
        lambda: sp.panel(
            _staggered(cohorts=(6, 0)),
            "y ~ treated",
            entity="unit",
            time="time",
            method="twoway",
            cluster="entity",
        ),
    ],
)
def test_single_adoption_date_is_not_flagged(fit):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = fit()
    assert "staggered_twfe" not in _tests(res)


def _two_by_two(seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(200):
        tr = int(i < 100)
        for t in (0, 1):
            rows.append((i, t, tr, tr * t, i % 20))
    df = pd.DataFrame(rows, columns=["unit", "post", "treat", "did", "grp"])
    df["y"] = rng.normal(size=len(df)) + df["did"]
    return df


@pytest.mark.parametrize(
    "fml, kw",
    [
        ("y ~ did | unit + post", {"cluster": "unit"}),
        ("y ~ treat + post + did", {"cluster": "unit"}),
        ("y ~ treat * post", {"cluster": "unit"}),
        ("y ~ treat + post + did", {"robust": "hc1"}),
    ],
)
def test_two_by_two_did_is_not_flagged(fml, kw):
    df = _two_by_two()
    fn = sp.feols if "|" in fml else sp.regress
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = fn(fml, df, **kw)
    assert not {"staggered_twfe", "treatment_cluster_level"} & set(_tests(res))


def test_event_study_dummies_are_not_flagged():
    df = _staggered(cohorts=(6, 0))
    rel = np.where(df["g"] > 0, df["time"] - df["g"], -99)
    for k in (-3, -2, 0, 1, 2):
        df[f"rel{k + 3}"] = (rel == k).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = sp.feols(
            "y ~ rel0 + rel1 + rel3 + rel4 + rel5 | unit + time", df, cluster="unit"
        )
    assert "staggered_twfe" not in _tests(res)


# ---------------------------------------------------------------------- #
#  Treatment assigned above the clustering level
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "call",
    [
        lambda df: sp.feols("y ~ treated | state + year", df, vcov="hetero"),
        lambda df: sp.feols("y ~ treated | id + year", df, cluster="id"),
        lambda df: sp.hdfe_ols("y ~ treated | id + year", df, cluster="id"),
        lambda df: sp.regress("y ~ treated", df[df.year == 8], robust="hc1"),
        lambda df: sp.panel(
            df,
            "y ~ treated",
            entity="id",
            time="year",
            method="twoway",
            cluster="entity",
        ),
    ],
)
def test_state_level_treatment_with_finer_ses_is_flagged(call):
    df = _state_panel()
    with pytest.warns(AssumptionWarning, match="assigned at the 'state' level"):
        res = call(df)
    v = _one(res, "treatment_cluster_level")
    assert v["value"] == {"assignment_level": "state", "n_levels": 40}
    assert v["next_calls"][0]["arguments"] == {"cluster": "state"}


def test_flagged_se_is_the_one_clustering_at_the_level_would_change():
    df = _state_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loose = sp.feols("y ~ treated | id + year", df, cluster="id")
        right = sp.feols("y ~ treated | id + year", df, cluster="state")
    # the numbers are untouched; clustering at the assignment level is what
    # the guardrail recommends, and it is silent there
    assert float(loose.params["treated"]) == pytest.approx(
        float(right.params["treated"]), rel=1e-12
    )
    assert "treatment_cluster_level" in _tests(loose)
    assert "treatment_cluster_level" not in _tests(right)


@pytest.mark.parametrize(
    "call",
    [
        # clustered at the assignment level, or a coarser one containing it
        lambda df: sp.feols("y ~ treated | id + year", df, cluster="state"),
        lambda df: sp.feols("y ~ treated | id + year", df, cluster="region"),
        lambda df: sp.regress("y ~ treated", df[df.year == 8], cluster="state"),
    ],
)
def test_clustering_at_or_above_the_level_is_not_flagged(call):
    df = _state_panel()
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = call(df)
    assert "treatment_cluster_level" not in _tests(res)


def test_treatment_varying_within_states_is_not_flagged():
    df = _state_panel(within=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        a = sp.feols("y ~ treated | id + year", df, cluster="id")
        b = sp.feols("y ~ treated | state + year", df, vcov="hetero")
    assert "treatment_cluster_level" not in _tests(a) + _tests(b)


def test_cohort_column_is_not_mistaken_for_an_assignment_level():
    # g (the adoption date) is constant within g x period by construction;
    # it describes the treatment path, not where treatment was assigned
    df = _staggered(cohorts=(6, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = sp.feols("y ~ treated | unit + time", df, cluster="unit")
    assert "treatment_cluster_level" not in _tests(res)


# ---------------------------------------------------------------------- #
#  Only user-level calls; warnings point at the user's line
# ---------------------------------------------------------------------- #


def test_warning_points_at_the_callers_line():
    df = _staggered()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.feols("y ~ treated | unit + time", df, cluster="unit")
    ours = [w for w in caught if issubclass(w.category, AssumptionWarning)]
    assert ours and all(w.filename == __file__ for w in ours)


def test_a_regression_run_inside_another_statspai_function_is_skipped():
    import statspai
    from statspai._frame_input import wrap

    # a function whose code lives in the package, called through the public
    # shim, that itself calls sp.feols: an internal regression
    src = (
        "def outer(df):\n"
        "    return sp.feols('y ~ treated | unit + time', df, cluster='unit')\n"
    )
    ns = {"sp": sp}
    exec(  # noqa: S102 - fixed test source
        compile(src, os.path.join(os.path.dirname(statspai.__file__), "x.py"), "exec"),
        ns,
    )
    outer = wrap(ns["outer"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", AssumptionWarning)
        res = outer(_staggered())
    assert _tests(res) == []
    assert "multiplicity" not in res.model_info


# ---------------------------------------------------------------------- #
#  Agent surface: next_steps, audit, MCP next_calls
# ---------------------------------------------------------------------- #


def test_next_steps_lead_with_the_robust_estimator():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.feols("y ~ treated | unit + time", _staggered(), cluster="unit")
    first = res.next_steps()[0]
    assert first["action"] == (
        "sp.did(df, y='y', treat='treated', time='time', id='unit')"
    )
    assert first["priority"] == "essential"


def test_audit_lists_the_guardrail_and_the_count():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.hdfe_ols("y ~ treated | unit + time", _staggered(), cluster="unit")
        report = sp.audit(res)
    assert "staggered_twfe" in {c["name"] for c in report["checks"]}
    assert report["multiplicity"]["specifications_tried"] == 1


def _mcp(name, arguments, _counter=[0]):
    from statspai.agent.mcp_server import handle_request

    _counter[0] += 1
    raw = handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": _counter[0],
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
    )
    return json.loads(raw)["result"]


def _load(tmp_path, frame, name):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    out = _mcp("load_data", {"data_path": str(path), "detail": "minimal"})
    return out["structuredContent"]["data_id"]


def test_mcp_staggered_fit_leads_with_a_ready_did_call(tmp_path):
    df = _staggered()
    data_id = _load(tmp_path, df, "stag")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _mcp("feols", {"fml": "y ~ treated | unit + time", "data_id": data_id})
    body = out["structuredContent"]
    first = body["next_calls"][0]
    assert first["tool"] == "did" and first["ready"] is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        follow = _mcp(first["tool"], first["arguments"])["structuredContent"]
        direct = sp.did(df, y="y", treat="treated", time="time", id="unit")
    assert follow["estimate"] == pytest.approx(direct.estimate, rel=1e-12)


def test_mcp_cluster_follow_up_refits_at_the_assignment_level(tmp_path):
    df = _state_panel()
    data_id = _load(tmp_path, df, "states")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _mcp(
            "feols",
            {"fml": "y ~ treated | id + year", "vcov": "hetero", "data_id": data_id},
        )
    first = out["structuredContent"]["next_calls"][0]
    assert first["tool"] == "feols" and first["ready"] is True
    assert first["arguments"]["cluster"] == "state" and "vcov" not in first["arguments"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refit = _mcp("feols", {**first["arguments"], "detail": "minimal"})
        direct = sp.feols("y ~ treated | id + year", df, cluster="state")
    assert refit["structuredContent"]["se"] == pytest.approx(
        float(direct.std_errors["treated"]), rel=1e-10
    )


# ---------------------------------------------------------------------- #
#  Session multiplicity counter
# ---------------------------------------------------------------------- #


def test_distinct_specifications_are_counted_and_repeats_are_not():
    df = _state_panel(within=True)
    fmls = ["y ~ treated", "y ~ treated + year", "y ~ treated + state"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in fmls:
            r = sp.regress(f, df)
        again = sp.regress(fmls[0], df)
        assert again.model_info["multiplicity"]["specifications_tried"] == 3
        r = sp.regress(fmls[0], df, robust="hc1")  # another SE: a new spec
        assert r.model_info["multiplicity"]["specifications_tried"] == 4
        assert "specification_search" not in _tests(r)
        r = sp.regress(fmls[0], df[df.year > 1])  # a subsample: a new spec
    v = _one(r, "specification_search")
    assert v["value"] == MULTIPLICITY_INFO and v["severity"] == "info"


def test_spec_curve_specifications_count_toward_later_fits():
    df = _state_panel(within=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        curve = sp.spec_curve(
            df,
            y="y",
            x="treated",
            controls=[[], ["year"], ["state"]],
            se_types=["nonrobust", "hc1"],
        )
        assert curve.model_info["multiplicity"]["specifications_tried"] == 6
        assert curve.violations() == []
        r = sp.regress("y ~ treated", df)
    assert r.model_info["multiplicity"]["specifications_tried"] == 7
    assert "specification_search" in _tests(r)
