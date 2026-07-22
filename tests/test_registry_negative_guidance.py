"""Contract: high-risk functions carry *negative* guidance and scaling cost.

Background — an agent-usability review of an AER replication attempt found
that every registry description said when to **use** a function and none
said when **not** to. Two calls that "succeeded" cost the reviewing agent
real time:

* ``sp.callaway_santanna`` on a plain 2x2 design — valid, much slower, and
  the same estimand as ``sp.did(method='2x2')``.
* ``sp.feols(vce='conley')`` at n ~ 140,000 — that path materialises dense
  ``n x n`` matrices (~157 GB) and the process was OOM-killed.

``FunctionSpec.not_recommended_when`` / ``FunctionSpec.cost_profile`` exist
to make both facts machine-readable *before* the call. This module locks
them so the guidance cannot silently rot back out:

1. every curated high-risk name still carries guidance,
2. the guidance is actually reachable through all three agent-facing
   surfaces (``agent_card`` / ``function_schema`` / ``to_agent_schema``),
3. the cost claims stay **per-path truthful** — the dense Conley paths are
   flagged, and ``sp.conley`` (sparse cKDTree) is explicitly *not*.

No network / R / Stata.
"""

from __future__ import annotations

import inspect

import pytest

import statspai as sp
from statspai import registry as R

# --------------------------------------------------------------------------- #
#  The curated set
# --------------------------------------------------------------------------- #

#: Names that must warn about a *wrong-tool* situation. These are the calls
#: that return successfully while being the wrong move.
NEGATIVE_GUIDANCE_REQUIRED = [
    # DiD family — "valid but pointless / wrong estimator" cluster
    "did",
    "did_2x2",
    "callaway_santanna",
    "sun_abraham",
    "did_imputation",
    "borusyak_jaravel_spiess",
    "bjs",
    "etwfe",
    "wooldridge_did",
    "stacked_did",
    "event_study",
    "bacon_decomposition",
    "honest_did",
    # dense-memory paths
    "feols",
    "hdfe_ols",
    "ppmlhdfe",
    "optimal_match",
    # synthetic control family
    "synth",
    "augsynth",
    "gsynth",
    "sdid",
    "mc_panel",
    # resampling
    "wild_cluster_bootstrap",
]

#: Names that must publish a runtime/memory scaling profile — either because
#: a path is quadratic, or because a bootstrap/CV knob multiplies runtime.
COST_PROFILE_REQUIRED = [
    "feols",
    "hdfe_ols",
    "ppmlhdfe",
    "conley",
    "callaway_santanna",
    "sun_abraham",
    "did_imputation",
    "etwfe",
    "wooldridge_did",
    "stacked_did",
    "did_multiplegt_dyn",
    "bacon_decomposition",
    "synth",
    "gsynth",
    "scpi",
    "sdid",
    "mc_panel",
    "synth_compare",
    "optimal_match",
    "genmatch",
    "match",
    "wild_cluster_bootstrap",
]

#: Paths that build an explicit dense ``n x n`` matrix and therefore MUST
#: say so. Verified against the source: ``feols``/``hdfe_ols`` route through
#: ``statspai.inference.jackknife.conley_vcov_matrix`` and ``ppmlhdfe``
#: through ``glm_conley_vcov``, both of which allocate full n x n arrays.
DENSE_CONLEY_PATHS = ["feols", "hdfe_ols", "ppmlhdfe"]


@pytest.fixture(scope="module", autouse=True)
def _full_registry():
    R._ensure_full_registry()


def _card(name: str) -> dict:
    return sp.agent_card(name)


# --------------------------------------------------------------------------- #
#  1. The curated entries carry guidance at all
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NEGATIVE_GUIDANCE_REQUIRED)
def test_curated_entry_states_when_not_to_use(name):
    card = _card(name)
    entries = card["not_recommended_when"]
    assert entries, (
        f"{name}: no negative guidance. An agent reading this card cannot "
        "tell when the call is the wrong tool — add "
        "not_recommended_when to _NEGATIVE_GUIDANCE_SEEDS."
    )
    for item in entries:
        assert isinstance(item, str) and item.strip(), (name, item)


@pytest.mark.parametrize("name", COST_PROFILE_REQUIRED)
def test_curated_entry_states_scaling_cost(name):
    card = _card(name)
    cost = card["cost_profile"]
    assert cost.strip(), (
        f"{name}: no cost_profile. This is a path where memory or wall-clock "
        "can blow up at realistic n — say so."
    )


@pytest.mark.parametrize("name", NEGATIVE_GUIDANCE_REQUIRED)
def test_negative_guidance_points_somewhere(name):
    """Guidance must be actionable, not merely discouraging.

    Every curated entry either names a concrete alternative (in the text or
    in ``alternatives``) or explains the cost that motivates the warning —
    "don't do this" with no exit is as unhelpful as no guidance at all.
    """
    card = _card(name)
    blob = " ".join(card["not_recommended_when"]).lower()
    has_pointer = (
        "sp." in blob
        or "use " in blob
        or "instead" in blob
        or bool(card["alternatives"])
        or bool(card["cost_profile"].strip())
    )
    assert has_pointer, f"{name}: negative guidance offers no way forward"


# --------------------------------------------------------------------------- #
#  2. The two observed failures specifically
# --------------------------------------------------------------------------- #


def test_callaway_santanna_warns_against_plain_2x2():
    """The literal first observed failure: CS on a single-cohort design."""
    blob = " ".join(_card("callaway_santanna")["not_recommended_when"]).lower()
    assert "2x2" in blob
    assert "same period" in blob or "same time" in blob


@pytest.mark.parametrize("name", DENSE_CONLEY_PATHS)
def test_dense_conley_paths_publish_quadratic_memory(name):
    """The second observed failure: a 158 GB dense allocation, unannounced."""
    card = _card(name)
    cost = card["cost_profile"].lower()
    assert "conley" in cost, name
    assert "n x n" in cost or "o(n^2)" in cost, name
    # A complexity class alone does not let an agent decide; it needs a
    # figure it can compare against available RAM.
    assert "gb" in cost, f"{name}: quadratic cost stated without a worked figure"


def test_sparse_conley_is_not_mislabelled_as_dangerous():
    """``sp.conley`` is sparse (cKDTree) — do not tar it with the dense paths.

    Accuracy matters in both directions: an agent that avoids the *correct*
    large-n tool because the docs over-warned is just as badly served as one
    that OOMs.
    """
    cost = _card("conley")["cost_profile"].lower()
    assert "sparse" in cost
    assert "kdtree" in cost
    assert "o(n^2)" not in cost.replace("rather than o(n^2)", "")


@pytest.mark.parametrize(
    "name,escape",
    [
        # OLS paths can hand the fitted result to the sparse estimator.
        ("feols", "sp.conley"),
        ("hdfe_ols", "sp.conley"),
        # ppmlhdfe is a GLM; sp.conley does not accept it, so the honest
        # escape hatch is cluster-robust inference, not sp.conley.
        ("ppmlhdfe", "cluster="),
    ],
)
def test_dense_conley_guidance_names_the_right_escape_hatch(name, escape):
    card = _card(name)
    blob = (" ".join(card["not_recommended_when"]) + card["cost_profile"]).lower()
    assert escape.lower() in blob, name


# --------------------------------------------------------------------------- #
#  3. Guidance reaches every agent-facing surface
# --------------------------------------------------------------------------- #


def test_guidance_reaches_the_plain_tool_description():
    """Plain OpenAI/Anthropic tool-callers read ``description`` and nothing else."""
    desc = sp.function_schema("feols")["description"]
    assert "Do NOT use when:" in desc
    assert "Cost:" in desc
    assert "157 GB" in desc


def test_guidance_reaches_the_x_statspai_extension():
    schema = R._REGISTRY["callaway_santanna"].to_agent_schema()
    ext = schema["x_statspai"]
    assert ext["not_recommended_when"]
    assert ext["cost_profile"]


def test_guidance_survives_the_json_bundle_round_trip():
    """``schemas/agent_cards.json`` is what offline agents consume."""
    cards = {c["name"]: c for c in sp.agent_cards()}
    for name in DENSE_CONLEY_PATHS:
        assert cards[name]["cost_profile"], name
    assert cards["callaway_santanna"]["not_recommended_when"]


def test_uncurated_functions_default_to_empty_not_missing():
    """The fields are always present, so agents can read them unconditionally."""
    for card in sp.agent_cards():
        assert isinstance(card["not_recommended_when"], list), card["name"]
        assert isinstance(card["cost_profile"], str), card["name"]


# --------------------------------------------------------------------------- #
#  4. Inheritance semantics
# --------------------------------------------------------------------------- #


def test_variants_inherit_parent_negative_guidance():
    """A variant that declares ``inherits_from`` absorbs the parent's warnings."""
    parent = R.FunctionSpec(
        name="_neg_parent",
        category="causal",
        description="parent",
        not_recommended_when=["parent situation — use sp.other"],
        cost_profile="parent cost",
    )
    child = R.FunctionSpec(
        name="_neg_child",
        category="causal",
        description="child",
        not_recommended_when=["child situation — use sp.other"],
        inherits_from="_neg_parent",
    )
    R.register(parent)
    R.register(child)
    try:
        card = child.agent_card()
        assert card["not_recommended_when"] == [
            "child situation — use sp.other",
            "parent situation — use sp.other",
        ]
        # cost_profile falls through when the child leaves it empty
        assert card["cost_profile"] == "parent cost"
    finally:
        R._REGISTRY.pop("_neg_parent", None)
        R._REGISTRY.pop("_neg_child", None)


def test_seed_names_all_exist():
    """A renamed function must not leave orphaned guidance behind."""
    missing = [n for n in R._NEGATIVE_GUIDANCE_SEEDS if n not in R._REGISTRY]
    assert not missing, f"negative-guidance seeds for unknown functions: {missing}"


# --------------------------------------------------------------------------- #
#  5. The honest_did legacy signature really is gone (dead-branch regression)
# --------------------------------------------------------------------------- #


def test_honest_did_has_no_legacy_betas_signature():
    """Pin the fact that made the workflow_tools fallback branch unreachable.

    The removed branch called ``sp.honest_did(betas=..., sigma=...,
    num_pre_periods=..., num_post_periods=..., method=...)``. None of those
    names exist on the current signature, so the branch could only raise.
    """
    params = set(inspect.signature(sp.honest_did).parameters)
    assert "result" in params
    assert not params & {"betas", "sigma", "num_pre_periods", "num_post_periods"}

    with pytest.raises(TypeError) as exc:
        sp.honest_did(
            betas=[0.0, 1.0],
            sigma=[[1.0, 0.0], [0.0, 1.0]],
            num_pre_periods=1,
            num_post_periods=1,
            method="SD",
        )
    assert "betas" in str(exc.value)


def test_workflow_tools_no_longer_calls_the_legacy_signature():
    """No *call* anywhere in the tool passes the removed keyword names.

    Parsed with ``ast`` rather than grepped, so the comment and hint text
    that deliberately spell out the dead signature (to stop it being
    reintroduced) do not trip the check — only real call syntax does.
    """
    import ast
    import textwrap

    from statspai.agent import workflow_tools

    src = textwrap.dedent(
        inspect.getsource(workflow_tools._tool_honest_did_from_result)
    )
    dead = {"betas", "sigma", "num_pre_periods", "num_post_periods", "m_bar"}
    used = {
        kw.arg
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg
    }
    assert not (used & dead), (
        f"{sorted(used & dead)} passed as keyword arguments in "
        "_tool_honest_did_from_result — sp.honest_did takes none of those "
        "(its signature is honest_did(result, e, m_grid, method, ...)), so "
        "such a call can only raise TypeError."
    )


# --------------------------------------------------------------------------- #
#  6. Argument errors are recoverable
# --------------------------------------------------------------------------- #


def test_missing_argument_error_states_expected_got_and_a_fix():
    from statspai.agent.workflow_tools import _missing_argument_error

    payload = _missing_argument_error(
        tool="preflight",
        argument="method",
        arguments={"data_path": "panel.csv", "unused": None},
        corrected="preflight(data_path='panel.csv', method='did')",
    )
    assert "expected argument `method`" in payload["error"]
    assert "data_path" in payload["error"]  # what actually arrived
    assert payload["expected_argument"] == "method"
    assert payload["got_arguments"] == ["data_path"]
    assert payload["try"].startswith("preflight(")


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("preflight", {"data_path": "x.csv"}),  # missing method
        ("cross_validate", {"data_path": "x.csv"}),  # missing estimand
    ],
)
def test_workflow_argument_errors_are_actionable(tool, arguments, tmp_path):
    """An agent must not have to guess the argument name from a bare error."""
    import pandas as pd

    from statspai.agent import workflow_tools as wt

    data = pd.DataFrame({"y": [1.0, 2.0], "x": [0.0, 1.0]})
    fn = {
        "preflight": wt._tool_preflight,
        "cross_validate": wt._tool_cross_validate,
    }[tool]
    kwargs = {"detail": "agent"}
    if tool == "preflight":
        kwargs["as_handle"] = False
    out = fn(arguments, data, **kwargs)
    assert "expected argument" in out["error"]
    assert out["try"]


def test_honest_did_from_result_failure_names_the_right_argument():
    """The replaced dead branch used to hide this behind a bogus TypeError."""
    from statspai.agent._result_cache import RESULT_CACHE
    from statspai.agent.workflow_tools import _tool_honest_did_from_result

    rid = RESULT_CACHE.put(object(), tool="test", arguments={})
    out = _tool_honest_did_from_result(
        rid, {"method": "SD"}, detail="agent", as_handle=False
    )
    assert "error" in out
    hint = out["hint"]
    assert "honest_did(result," in hint
    assert "does NOT take betas=" in hint.replace("It ", "")
    assert out["failed_call"].startswith("sp.honest_did(")
    assert "diagnosis" in out
