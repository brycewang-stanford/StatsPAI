"""Open items of the 2026-11-13 agent probe, closed the same week.

``docs/dev/2026-11-13-agent-probe.md`` left seventeen defects open. The
tests here pin the ones on the MCP surface and in ``sp.audit`` (#32, #34,
#37, #39, #40, #46, #47, #48 and the single balance threshold). Each
states an identity with the direct Python call, a number the data fixes,
or that an inapplicable request is refused with the valid choices named.
The MCP calls go through ``handle_request`` in this process (the same
code path as the stdio server).
"""

from __future__ import annotations

import base64
import json
import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

import statspai as sp

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _rpc(tool: str, /, **arguments: Any) -> Dict[str, Any]:
    from statspai.agent.mcp_server import handle_request

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reply = handle_request(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": arguments},
                }
            )
        )
    out: Dict[str, Any] = json.loads(reply)["result"]
    return out


def _call(tool: str, /, **arguments: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = _rpc(tool, **arguments)["structuredContent"]
    return out


def _png(reply: Dict[str, Any]) -> bytes:
    blocks = [c for c in reply["content"] if c["type"] == "image"]
    assert len(blocks) == 1
    return base64.b64decode(blocks[0]["data"])


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("agent_followup")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = {
            "rd": sp.datasets.lee_2008_senate(),
            "synth": sp.datasets.california_prop99(),
            "nsw": sp.datasets.nsw_dw(),
            "mp": sp.datasets.mpdta(),
            "card": sp.datasets.card_1995(),
        }
    mp = frames["mp"]
    frames["mp"] = mp.assign(
        treated=((mp["first_treat"] > 0) & (mp["year"] >= mp["first_treat"])).astype(
            int
        )
    )
    two = sp.dgp_did(n_units=100, n_periods=2, seed=0)
    two["treat"] = two["first_treat"].notna().astype(int)
    two["post"] = (two["time"] >= 1).astype(int)
    frames["two"] = two[["unit", "y", "treat", "post"]]
    paths = {}
    for name, frame in frames.items():
        paths[name] = str(root / f"{name}.csv")
        frame.to_csv(paths[name], index=False)
    return paths, {k: pd.read_csv(v) for k, v in paths.items()}


_SYNTH = dict(
    outcome="cigsale",
    unit="state",
    time="year",
    treated_unit="California",
    treatment_time=1989,
)


# --------------------------------------------------------------------------- #
#  #37 plot_from_result honours kind=
# --------------------------------------------------------------------------- #
def test_rd_plot_is_the_binned_scatter_of_the_fit(files):
    from statspai.agent.workflow_tools import _figure_png

    paths, frames = files
    rid = _call("rdrobust", data_path=paths["rd"], y="y", x="x", c=0, as_handle=True)[
        "result_id"
    ]
    default = _rpc("plot_from_result", result_id=rid)
    asked = _rpc("plot_from_result", result_id=rid, kind="rdplot")
    assert default["structuredContent"]["kind"] == "rdplot"
    assert default["structuredContent"]["valid_kinds"] == ["rdplot", "coef_plot"]
    fig, _ax = sp.rdplot(frames["rd"], y="y", x="x", c=0.0, figsize=(8, 5))
    direct = _figure_png(fig)
    # the picture is sp.rdplot on the fit's own columns, byte for byte
    assert _png(asked) == direct
    assert _png(default) == direct
    coef = _rpc("plot_from_result", result_id=rid, kind="coef_plot")
    assert coef["structuredContent"]["kind"] == "coef_plot"


def test_synth_gap_and_path_are_the_result_own_plots(files):
    from statspai.agent.workflow_tools import _figure_png

    paths, frames = files
    rid = _call("synth", data_path=paths["synth"], as_handle=True, **_SYNTH)[
        "result_id"
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.synth(frames["synth"], **_SYNTH)
    for kind, plot_type in (("synth_gap", "gap"), ("synth_path", "trajectory")):
        reply = _rpc("plot_from_result", result_id=rid, kind=kind)
        assert reply["structuredContent"]["kind"] == kind
        fig = fit.plot(type=plot_type, figsize=(8, 5))[0]
        assert _png(reply) == _figure_png(fig), kind
    # CausalResult.plot's own spelling is accepted and reported canonically
    alias = _rpc("plot_from_result", result_id=rid, kind="gap")
    assert alias["structuredContent"]["kind"] == "synth_gap"


@pytest.mark.parametrize(
    "tool, key, args, bad_kind, valid",
    [
        (
            "rdrobust",
            "rd",
            dict(y="y", x="x", c=0),
            "no_such_plot",
            ["rdplot", "coef_plot"],
        ),
        (
            "synth",
            "synth",
            _SYNTH,
            "rdplot",
            ["synth", "synth_gap", "synth_path", "synth_weights", "coef_plot"],
        ),
        (
            "regress",
            "nsw",
            dict(formula="re78 ~ treat + age"),
            "event_study",
            ["coef_plot", "fitted"],
        ),
        (
            "did",
            "two",
            dict(y="y", treat="treat", time="post"),
            "event_study",
            ["coef_plot"],
        ),
    ],
)
def test_unknown_or_inapplicable_kind_is_refused(
    files, tool, key, args, bad_kind, valid
):
    paths, _ = files
    rid = _call(tool, data_path=paths[key], as_handle=True, **args)["result_id"]
    reply = _rpc("plot_from_result", result_id=rid, kind=bad_kind)
    body = reply["structuredContent"]
    assert reply["isError"] is True
    assert body["error_kind"] == "method_incompatibility"
    assert body["valid_kinds"] == valid
    assert body["requested_kind"] == bad_kind
    assert not [c for c in reply["content"] if c["type"] == "image"]


def test_rdplot_without_the_dataset_says_what_to_pass(files):
    _, frames = files
    records = frames["rd"].dropna().head(500).to_dict(orient="records")
    rid = _call("rdrobust", data_records=records, y="y", x="x", c=0, as_handle=True)[
        "result_id"
    ]
    refused = _rpc("plot_from_result", result_id=rid, kind="rdplot")
    assert refused["isError"] is True
    assert refused["structuredContent"]["error_kind"] == "missing_arguments"
    assert refused["structuredContent"]["missing_arguments"] == ["data_id"]
    # the default falls back to the estimate and says how to get the scatter
    default = _call("plot_from_result", result_id=rid)
    assert default["kind"] == "coef_plot" and "rdplot" in default["note"]
    with_data = _call(
        "plot_from_result", result_id=rid, kind="rdplot", data_records=records
    )
    assert with_data["kind"] == "rdplot"


# --------------------------------------------------------------------------- #
#  #39 seed_injected only where a generator was seeded
# --------------------------------------------------------------------------- #
def test_watched_seed_notices_seeding_and_nothing_else():
    from sklearn.model_selection import KFold

    from statspai.agent._default_seed import WatchedSeed, seed_uses

    seed = WatchedSeed(42)
    before = seed_uses()
    # validation, comparison, display and JSON are not a stochastic step
    assert seed == 42 and seed >= 0 and isinstance(seed, int) and seed is not None
    assert json.dumps({"seed": seed}) == '{"seed": 42}' and repr(seed) == "42"
    assert seed_uses() == before
    draws = {
        "default_rng": lambda s: float(np.random.default_rng(s).normal()),
        "RandomState": lambda s: float(np.random.RandomState(s).normal()),
        "KFold": lambda s: [
            test.tolist()
            for _, test in KFold(3, shuffle=True, random_state=s).split(
                np.zeros((9, 1))
            )
        ],
        "derived": lambda s: float(np.random.default_rng(s + 1).normal()),
    }
    for name, draw in draws.items():
        before = seed_uses()
        # the same stream as the plain integer, and the use is noticed
        assert draw(WatchedSeed(42)) == draw(42), name
        assert seed_uses() > before, name


@pytest.mark.parametrize(
    "tool, key, args",
    [
        ("did", "two", dict(y="y", treat="treat", time="post")),
        (
            "callaway_santanna",
            "mp",
            dict(y="lemp", g="first_treat", t="year", i="countyreal"),
        ),
        ("rdrobust", "rd", dict(y="y", x="x", c=0)),
    ],
)
def test_no_seed_note_on_a_call_without_a_stochastic_step(files, tool, key, args):
    paths, _ = files
    out = _call(tool, data_path=paths[key], **args)
    assert "error" not in out, out.get("error")
    assert "seed_injected" not in out
    replay = str(out["replay"])
    assert "seed" not in replay and "random_state" not in replay
    # and the card does not claim a seed either
    assert out["result_card"]["provenance"].get("seed") is None


def test_seed_note_card_and_replay_agree_on_a_stochastic_call(files):
    """SDID's placebo variance draws: the fill is reported, replayable and
    on the card, and the numbers are those of the direct call with it."""
    paths, frames = files
    out = _call("sdid", data_path=paths["synth"], **_SYNTH)
    assert out["seed_injected"]["param"] == "seed"
    assert out["seed_injected"]["value"] == 42
    assert "seed=42" in str(out["replay"]).replace(" ", "")
    prov = out["result_card"]["provenance"]
    assert (prov["seed"], prov["reproducible"]) == (42, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = sp.sdid(frames["synth"], seed=42, **_SYNTH)
    assert out["estimate"] == pytest.approx(float(direct.estimate), rel=1e-12)
    assert out["se"] == pytest.approx(float(direct.se), rel=1e-12)


# --------------------------------------------------------------------------- #
#  #40 data_provenance once per response
# --------------------------------------------------------------------------- #
def _provenance_blocks(node, path=""):
    """Paths of every dict in ``node`` that carries a file hash."""
    found = []
    if isinstance(node, dict):
        if "sha256" in node:
            found.append(path)
        for key, value in node.items():
            found += _provenance_blocks(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found += _provenance_blocks(value, f"{path}[{i}]")
    return found


def test_source_record_appears_once_per_reply(files):
    paths, _ = files
    loaded = _call("load_data", data_path=paths["nsw"], name="nsw")
    assert _provenance_blocks(loaded) == ["/data_provenance"]
    assert loaded["data_provenance"]["label"] == "nsw"
    assert "provenance" not in loaded
    fit = _call("regress", data_path=paths["nsw"], formula="re78 ~ treat")
    assert _provenance_blocks(fit) == ["/data_provenance"]
    by_handle = _call("regress", data_id=loaded["data_id"], formula="re78 ~ treat")
    assert _provenance_blocks(by_handle) == ["/data_provenance/root"]
    assert by_handle["data_provenance"]["root"]["sha256"] == (
        loaded["data_provenance"]["sha256"]
    )


# --------------------------------------------------------------------------- #
#  #46 the audit reads the variance the fit recorded
# --------------------------------------------------------------------------- #
def _robust_se(fit):
    (check,) = [c for c in sp.audit(fit)["checks"] if c["name"] == "robust_se"]
    return check["status"], check["value"]


def test_clustered_or_robust_fit_satisfies_robust_se(files):
    _, frames = files
    mp = frames["mp"]
    fml = "lemp ~ treated | countyreal + year"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = {
            "feols_cluster": sp.feols(fml, data=mp, vcov={"CRV1": "countyreal"}),
            "feols_hetero": sp.feols(fml, data=mp, vcov="hetero"),
            "feols_iid": sp.feols(fml, data=mp),
            "regress_cluster": sp.regress(
                "lemp ~ treated", data=mp, cluster="countyreal"
            ),
            "regress_hc1": sp.regress("lemp ~ treated", data=mp, robust="hc1"),
            "regress_iid": sp.regress("lemp ~ treated", data=mp),
            "panel_cluster": sp.panel(
                mp,
                "lemp ~ treated",
                entity="countyreal",
                time="year",
                cluster="countyreal",
            ),
            "hdfe_cluster": sp.hdfe_ols(fml, data=mp, cluster="countyreal"),
            "hdfe_iid": sp.hdfe_ols(fml, data=mp),
        }
    got = {name: _robust_se(fit) for name, fit in fits.items()}
    assert got == {
        "feols_cluster": ("passed", "cluster (countyreal)"),
        "feols_hetero": ("passed", "hetero"),
        "feols_iid": ("failed", "nonrobust"),
        # was "failed": sp.regress keeps robust='nonrobust' beside cluster=
        "regress_cluster": ("passed", "cluster (countyreal)"),
        "regress_hc1": ("passed", "hc1"),
        "regress_iid": ("failed", "nonrobust"),
        "panel_cluster": ("passed", "cluster (countyreal)"),
        "hdfe_cluster": ("passed", "cluster (countyreal)"),
        "hdfe_iid": ("failed", "nonrobust"),
    }


# --------------------------------------------------------------------------- #
#  #48 SDID stores the evidence the audit reads
# --------------------------------------------------------------------------- #
def test_sdid_prefit_and_placebo_evidence_are_recognised(files):
    _, frames = files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.sdid(frames["synth"], seed=1, **_SYNTH)
    mi = fit.model_info
    pre = mi["pre_times"]
    gap = (mi["Y_obs"][pre] - mi["Y_synth"][pre]).to_numpy()
    rmse = float(np.sqrt(np.mean(gap**2)))
    ratio = rmse / float(np.std(mi["Y_obs"][pre].to_numpy(), ddof=1))
    assert mi["pre_treatment_rmse_adjusted"] == pytest.approx(rmse, rel=1e-12)
    # the level-fit key of the other synth estimators stays unset
    assert "pre_treatment_rmse" not in mi
    assert mi["pretreatment_rmse_ratio"] == pytest.approx(ratio, rel=1e-12)
    assert mi["placebo_inference"] == {
        "se_method": "placebo",
        "n_reps": 200,
        "se": float(fit.se),
    }
    checks = {c["name"]: c for c in sp.audit(fit)["checks"]}
    assert checks["pretreatment_fit"]["status"] == "passed"
    assert checks["pretreatment_fit"]["value"] == pytest.approx(ratio, rel=1e-12)
    assert ratio < checks["pretreatment_fit"]["threshold"]
    assert checks["placebo_inference"]["status"] == "passed"


def test_sdid_without_a_standard_error_has_no_inference_evidence(files):
    """One treated unit: the jackknife variance is undefined (as in
    synthdid), so there is no inferential statement to recognise."""
    _, frames = files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.sdid(frames["synth"], se_method="jackknife", **_SYNTH)
    assert np.isnan(fit.se)
    assert "placebo_inference" not in fit.model_info
    assert "resampling_inference" not in fit.model_info
    checks = {c["name"]: c["status"] for c in sp.audit(fit)["checks"]}
    assert checks["placebo_inference"] == "missing"


# --------------------------------------------------------------------------- #
#  #47 synth_sensitivity stays small; the cut tables are readable in full
# --------------------------------------------------------------------------- #
def test_synth_sensitivity_reply_is_capped_and_readable_in_full(files):
    from statspai.agent.mcp_server import handle_request
    from statspai.agent.tools._helpers import AGENT_TABLE_ROWS

    paths, frames = files
    out = _call("synth_sensitivity", data_path=paths["synth"], as_handle=True, **_SYNTH)
    assert "error" not in out, out.get("error")
    # 47.8 KB before the cap (36 KB of it the 100 donor-pool draws)
    assert len(json.dumps(out)) < 20_000
    capped = {c["path"]: c for c in out["arrays_capped"]}
    assert capped["/donor_sensitivity"]["total"] == 100
    assert capped["/donor_sensitivity"]["shown"] == AGENT_TABLE_ROWS
    assert len(out["donor_sensitivity"]) == AGENT_TABLE_ROWS
    # small tables are left whole
    assert len(out["time_placebo"]) == 17 and "/time_placebo" not in capped

    def read(uri):
        reply = json.loads(
            handle_request(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "resources/read",
                        "params": {"uri": uri},
                    }
                )
            )
        )
        return reply

    body = json.loads(
        read(capped["/donor_sensitivity"]["full"])["result"]["contents"][0]["text"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = sp.synth_sensitivity(
            data=frames["synth"], seed=out["seed_injected"]["value"], **_SYNTH
        )
    # every draw of the direct call, in order
    assert body["n"] == 100
    np.testing.assert_allclose(
        [row["att"] for row in body["values"]],
        direct["donor_sensitivity"]["att"].to_numpy(),
        rtol=1e-12,
    )
    assert body["values"][:AGENT_TABLE_ROWS] == out["donor_sensitivity"]
    assert out["original_att"] == pytest.approx(direct["original_att"], rel=1e-12)
    # a path that was not cut is still refused
    refused = read(
        capped["/donor_sensitivity"]["full"].replace(
            "donor_sensitivity", "original_att"
        )
    )
    assert "error" in refused
    # detail='standard' keeps every row
    whole = _call(
        "synth_sensitivity", data_path=paths["synth"], detail="standard", **_SYNTH
    )
    assert len(whole["donor_sensitivity"]) == 100


# --------------------------------------------------------------------------- #
#  One balance threshold (owner decision 6 of the probe)
# --------------------------------------------------------------------------- #
def _confounded(strength: float, n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(strength * x1 + 0.6 * strength * x2)))
    d = (rng.uniform(size=n) < ps).astype(int)
    y = 1 + 2 * d + x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def test_violations_and_audit_share_one_balance_threshold():
    import importlib

    from statspai.core import _agent_summary

    # ``statspai.smart.audit`` the attribute is the function; get the module
    audit_module = importlib.import_module("statspai.smart.audit")

    assert audit_module._SMD_MAX is _agent_summary._SMD_MAX
    assert (_agent_summary._SMD_MAX, _agent_summary._SMD_SEVERE) == (0.10, 0.25)
    levels = {}
    for strength in (1.5, 8.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.match(
                _confounded(strength), y="y", treat="d", covariates=["x1", "x2"]
            )
        worst = fit.model_info["max_abs_smd_after"]
        (flag,) = [v for v in fit.violations() if v["test"] == "balance"]
        (check,) = [c for c in sp.audit(fit)["checks"] if c["name"] == "balance_after"]
        # the same number against the same line, in both reports
        assert flag["value"] == pytest.approx(worst) == pytest.approx(check["value"])
        assert flag["threshold"] == check["threshold"] == 0.10
        assert check["status"] == "failed"
        levels[strength] = (flag["level"], worst > 0.25)
    # between 0.10 and 0.25 violations() used to say nothing
    assert levels == {1.5: ("moderate", False), 8.0: ("severe", True)}


# --------------------------------------------------------------------------- #
#  #34 no invented focal term: a null headline that explains itself
# --------------------------------------------------------------------------- #
def _multi_regressor_fits(frames):
    mp, card = frames["mp"], frames["card"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mp = mp.assign(z=np.random.default_rng(3).normal(size=len(mp)))
        return {
            "feols": sp.feols("lemp ~ treated + z | countyreal + year", data=mp),
            "panel": sp.panel(
                mp, "lemp ~ treated + z", entity="countyreal", time="year"
            ),
            # two endogenous regressors: no single one to name
            "liml": sp.liml(
                data=card,
                y="lwage",
                x_endog=["educ", "exper"],
                z=["nearc4", "nearc2", "expersq"],
                x_exog=["black"],
            ),
            "hdfe_ols": sp.hdfe_ols("lemp ~ treated + z | countyreal + year", data=mp),
        }


@pytest.mark.parametrize("name", ["feols", "panel", "liml", "hdfe_ols"])
def test_null_headline_carries_the_table_and_says_why(files, name):
    _, frames = files
    fit = _multi_regressor_fits(frames)[name]
    slopes = [str(t) for t in fit.params.index if str(t) not in ("Intercept", "_cons")]
    assert len(slopes) >= 2
    for detail in ("minimal", "agent"):
        payload = fit.to_dict(detail=detail)
        assert payload["estimate"] is None and payload["se"] is None
        assert payload["focal_term"] is None
        note = payload["headline_note"]
        assert "\n" not in note
        if detail == "minimal":
            # one clause: the minimal payload has a character budget
            assert note == f"no focal term ({len(slopes)} regressors): see coef_head"
        else:
            assert "No focal coefficient" in note
            assert all(t in note for t in slopes[:4])
    table = fit.to_dict(detail="agent")["coefficients"]
    for term in slopes:
        assert table[term]["estimate"] == pytest.approx(
            float(fit.params[term]), rel=1e-12
        )
        assert table[term]["std_error"] == pytest.approx(
            float(fit.std_errors[term]), rel=1e-12
        )
    head = fit.to_dict(detail="minimal")["coef_head"]
    assert list(head) == slopes[:4]
    line = sp.brief(fit)
    assert "best" not in line and "(no focal term)" in line
    assert [line.index(f"{t}=") for t in slopes[:3]] == sorted(
        line.index(f"{t}=") for t in slopes[:3]
    )


def test_liml_names_its_single_endogenous_regressor(files):
    """The rule ``sp.ivreg`` follows (the one endogenous regressor is the
    point of an IV fit), read from the record ``sp.liml`` keeps."""
    _, frames = files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.liml(
            data=frames["card"],
            y="lwage",
            x_endog=["educ"],
            z=["nearc4", "nearc2"],
            x_exog=["exper", "black"],
        )
    payload = fit.to_dict(detail="minimal")
    assert payload["focal_term"] == "educ"
    assert payload["focal_term_rule"] == "endogenous regressor"
    assert payload["estimate"] == pytest.approx(float(fit.params["educ"]), rel=1e-12)
    assert payload["se"] == pytest.approx(float(fit.std_errors["educ"]), rel=1e-12)


def test_single_regressor_is_still_the_headline(files):
    _, frames = files
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = [
            sp.feols("lemp ~ treated | countyreal + year", data=frames["mp"]),
            sp.hdfe_ols("lemp ~ treated | countyreal + year", data=frames["mp"]),
        ]
    for fit in fits:
        payload = fit.to_dict(detail="minimal")
        assert payload["focal_term"] == "treated"
        assert payload["estimate"] == pytest.approx(
            float(fit.params["treated"]), rel=1e-12
        )
        assert payload["se"] == pytest.approx(
            float(fit.std_errors["treated"]), rel=1e-12
        )
        assert "headline_note" not in payload
        assert "(no focal term)" not in sp.brief(fit)


def test_mcp_summaries_read_the_coefficient_table(files):
    paths, frames = files
    out = _call(
        "feols",
        data_path=paths["nsw"],
        fml="re78 ~ treat + age + education | black",
        as_handle=True,
    )
    assert "error" not in out, out.get("error")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.feols("re78 ~ treat + age + education | black", data=frames["nsw"])
    assert out["estimate"] is None and out["focal_term"] is None
    assert "No focal coefficient" in out["headline_note"]
    expected = ", ".join(
        f"{t} = {float(fit.params[t]):.4g} (SE {float(fit.std_errors[t]):.3g})"
        for t in ("treat", "age", "education")
    )
    assert f"Coefficients (no focal term): {expected}" in out["narrative"]
    brief = _call("brief_result", result_id=out["result_id"])["brief"]
    interp = _call("interpret_result", result_id=out["result_id"])
    for text in (brief, interp["summary"]["brief"]):
        assert "(no focal term)" in text and "best" not in text
        for term in ("treat", "age", "education"):
            assert f"{term}=" in text
    assert interp["summary"]["fields"]["coefficients"]["age"]["estimate"] == (
        pytest.approx(float(fit.params["age"]), rel=1e-9)
    )


# --------------------------------------------------------------------------- #
#  #32 sp.stata with several estimation commands says what it did not return
# --------------------------------------------------------------------------- #
_SNIPPET = (
    "regress re78 treat\n"
    "regress re78 treat age, robust\n"
    "logit treat age education"
)


def test_stata_snippet_reports_the_estimations_it_did_not_return(files):
    _, frames = files
    nsw = frames["nsw"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        last = sp.stata(_SNIPPET, data=nsw)
        direct = sp.logit("treat ~ age + education", data=nsw)
        single = sp.stata("regress re78 treat", data=nsw)
    # the return is still the last result, unchanged
    np.testing.assert_allclose(
        last.params.to_numpy(), direct.params.to_numpy(), rtol=1e-12
    )
    run = last.stata_run
    assert run["n_estimation_commands"] == 3
    assert run["returned_command"] == "logit treat age education"
    assert run["earlier_commands_not_returned"] == [
        "regress re78 treat",
        "regress re78 treat age, robust",
    ]
    assert "own sp.stata call" in run["hint"]
    assert single.stata_run == {
        "n_estimation_commands": 1,
        "returned_command": "regress re78 treat",
        "earlier_commands_not_returned": [],
    }


def test_each_earlier_command_runs_on_its_own_as_the_hint_says(files):
    _, frames = files
    nsw = frames["nsw"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = sp.stata(_SNIPPET, data=nsw).stata_run
        alone = [
            sp.stata(cmd, data=nsw) for cmd in run["earlier_commands_not_returned"]
        ]
        expected = [
            sp.regress("re78 ~ treat", data=nsw),
            sp.regress("re78 ~ treat + age", data=nsw, robust="hc1"),
        ]
    for got, want in zip(alone, expected):
        np.testing.assert_allclose(
            got.params.to_numpy(), want.params.to_numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            got.std_errors.to_numpy(), want.std_errors.to_numpy(), rtol=1e-12
        )


def test_a_loop_counts_every_estimation_it_ran(files):
    _, frames = files
    loop = "forvalues i = 1/3 {\n regress re78 treat age if education > `i'\n}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = sp.stata(loop, data=frames["nsw"]).stata_run
    assert run["n_estimation_commands"] == 3
    assert run["earlier_commands_not_returned"] == [
        "regress re78 treat age if education > 1",
        "regress re78 treat age if education > 2",
    ]
    assert run["returned_command"] == "regress re78 treat age if education > 3"


def test_mcp_stata_reply_names_the_results_it_dropped(files):
    paths, _ = files
    several = _call(
        "call_function",
        data_path=paths["nsw"],
        function="stata",
        arguments={"commands": _SNIPPET},
    )
    assert "error" not in several, several.get("error")
    assert several["n_estimation_commands"] == 3
    assert several["earlier_commands_not_returned"] == [
        "regress re78 treat",
        "regress re78 treat age, robust",
    ]
    assert several["returned_command"] == "logit treat age education"
    assert "own sp.stata call" in several["hint"]
    one = _call(
        "call_function",
        data_path=paths["nsw"],
        function="stata",
        arguments={"commands": "regress re78 treat"},
    )
    assert one["n_estimation_commands"] == 1
    assert "earlier_commands_not_returned" not in one


# --------------------------------------------------------------------------- #
#  House style: sp.aft takes vce= like every other estimator with robust=
# --------------------------------------------------------------------------- #
def test_aft_accepts_the_house_spelling_of_its_variance_option():
    """``sp.aft`` gained ``robust=`` on 2026-11-13 without the ``vce=``
    alias, which the CI-only signature ratchet counts as a regression."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["t"] = rng.weibull(1.5, size=n) * np.exp(0.4 * df["x"])
    df["d"] = (rng.uniform(size=n) < 0.8).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_name = sp.aft("t + d ~ x", df, robust="robust")
        by_alias = sp.aft("t + d ~ x", df, vce="robust")
        classical = sp.aft("t + d ~ x", df)
    assert by_alias.to_dict() == by_name.to_dict()
    assert by_alias.vcov_type == "robust" and classical.vcov_type == "nonrobust"
    # a variance option: the coefficients are those of the classical fit
    np.testing.assert_allclose(by_alias.beta, classical.beta, rtol=1e-12)
