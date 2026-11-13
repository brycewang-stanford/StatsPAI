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
