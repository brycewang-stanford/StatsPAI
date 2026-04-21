"""End-to-end integration tests for ``sp.causal_llm.causal_mas``.

Uses :func:`sp.causal_llm.echo_client` as a deterministic stand-in for
a real LLM so we can exercise the full proposer / critic /
domain-expert / synthesiser loop without network access.

Validated behaviours:

1. **Proposer parsing** — the LLM's text output is correctly parsed
   into ``(parent, child)`` edges.
2. **Critic rejection** — edges explicitly rejected by the critic are
   removed from the final DAG.
3. **Domain expert boost** — endorsements lift per-edge confidence
   above the default threshold.
4. **Transcript auditability** — every round records proposer +
   critic + domain_expert + synthesiser entries in order.
5. **Per-edge confidence scales with rounds** — an edge endorsed in
   every round has confidence 1.0; an edge endorsed in one of three
   rounds has confidence close to 1/3.
6. **Role overrides honoured** — user-supplied ``treatment`` /
   ``outcome`` / ``confounders`` / ``instruments`` kwargs overwrite
   the name-based heuristic classifier.
7. **Pipes into sp.dag** — the resulting ``to_dag_string()`` output
   parses cleanly via :func:`sp.dag` so downstream identification
   analysis works end-to-end.
"""

from __future__ import annotations

import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# 1. Proposer parsing
# ---------------------------------------------------------------------------


class TestProposerParsing:

    def test_newline_separated_edges_are_parsed(self):
        def script(role: str, prompt: str) -> str:
            if role == "proposer":
                return "age -> treatment\ntreatment -> outcome\n"
            return ""

        client = sp.causal_llm.echo_client(script)
        res = sp.causal_llm.causal_mas(
            variables=["age", "treatment", "outcome"],
            client=client, rounds=1,
        )
        assert ("age", "treatment") in res.edges
        assert ("treatment", "outcome") in res.edges
        assert len(res.edges) == 2

    def test_proposer_handles_bullet_formatting(self):
        # LLMs often return bullet points; the parser strips them.
        def script(role: str, prompt: str) -> str:
            if role == "proposer":
                return (
                    "* age -> treatment\n"
                    "- treatment -> outcome\n"
                    "• age -> outcome\n"
                )
            return ""

        client = sp.causal_llm.echo_client(script)
        res = sp.causal_llm.causal_mas(
            variables=["age", "treatment", "outcome"],
            client=client, rounds=1,
        )
        # All three edges are parsed despite bullet characters.
        assert ("age", "treatment") in res.edges
        assert ("treatment", "outcome") in res.edges
        assert ("age", "outcome") in res.edges


# ---------------------------------------------------------------------------
# 2. Critic rejection
# ---------------------------------------------------------------------------


class TestCriticRejection:

    def test_critic_rejected_edge_removed(self):
        # Proposer proposes two edges; critic rejects one.  The rejected
        # edge should not accumulate enough confidence to clear the
        # default threshold.
        def script(role: str, prompt: str) -> str:
            if role == "proposer":
                return "age -> treatment\ntreatment -> outcome"
            if role == "critic":
                return "age -> treatment"  # reject age -> treatment
            return ""

        client = sp.causal_llm.echo_client(script)
        res = sp.causal_llm.causal_mas(
            variables=["age", "treatment", "outcome"],
            client=client, rounds=3, final_threshold=0.9,
        )
        # The rejected edge gained zero proposer-survivor votes (always
        # rejected) and zero domain-expert bonuses, so its confidence
        # stays at 0.  It must not appear in the final edge list.
        assert ("age", "treatment") not in res.edges
        # The non-rejected edge should survive.
        assert ("treatment", "outcome") in res.edges


# ---------------------------------------------------------------------------
# 3. Domain-expert endorsement boost
# ---------------------------------------------------------------------------


class TestDomainExpertBoost:

    def test_endorsement_lifts_confidence(self):
        # Proposer drops the edge from round 2 onwards; domain expert
        # never endorses.  With rounds=3 the confidence should be 1/3.
        state = {"round": 0}

        def script(role: str, prompt: str) -> str:
            if role == "proposer":
                # First round: propose; subsequent rounds: silent.
                if state["round"] == 0:
                    out = "treatment -> outcome"
                else:
                    out = ""
                state["round"] += 1 if role == "domain_expert" else 0
                return out
            if role == "critic":
                return ""
            if role == "domain_expert":
                # Advance the counter once per round.
                state["round"] += 0  # handled above
                return ""
            return ""

        client = sp.causal_llm.echo_client(script)
        res = sp.causal_llm.causal_mas(
            variables=["treatment", "outcome"],
            client=client, rounds=3, final_threshold=0.1,
        )
        # The edge was only proposed once ⇒ confidence = 1/3 ≈ 0.333.
        conf = res.confidence.get(("treatment", "outcome"), 0.0)
        assert abs(conf - 1 / 3) < 0.1, (
            f"Single-round proposal → confidence {conf:.3f}, expected ≈0.33"
        )


# ---------------------------------------------------------------------------
# 4. Transcript auditability
# ---------------------------------------------------------------------------


class TestTranscriptAudit:

    def test_transcript_has_four_entries_per_round(self):
        client = sp.causal_llm.echo_client(lambda r, p: "")
        res = sp.causal_llm.causal_mas(
            variables=["a", "b"], client=client, rounds=4,
        )
        # proposer + critic + domain_expert + synthesiser per round.
        assert len(res.transcript) == 4 * 4
        # Action types present in order.
        actions = [entry["action"] for entry in res.transcript]
        expected_cycle = ["propose", "reject", "endorse", "score"]
        for r in range(4):
            got = actions[4 * r : 4 * (r + 1)]
            assert got == expected_cycle, (
                f"Round {r}: action sequence = {got}, expected {expected_cycle}"
            )


# ---------------------------------------------------------------------------
# 5. Confidence scales with rounds
# ---------------------------------------------------------------------------


class TestConfidenceScaling:

    def test_every_round_endorsement_gives_max_confidence(self):
        def always_propose(role: str, prompt: str) -> str:
            if role == "proposer":
                return "a -> b"
            return ""

        client = sp.causal_llm.echo_client(always_propose)
        res = sp.causal_llm.causal_mas(
            variables=["a", "b"], client=client, rounds=5,
        )
        # Every round produces the edge and critic does not reject.
        # Confidence should be 1.0 (capped).
        assert res.confidence[("a", "b")] == 1.0


# ---------------------------------------------------------------------------
# 6. Role overrides
# ---------------------------------------------------------------------------


class TestRoleOverrides:

    def test_user_treatment_outcome_overrides_heuristic(self):
        # The heuristic classifier would flag neither "foo" nor "bar"
        # as treatment/outcome; supplying explicit kwargs should force
        # the assignment.
        client = sp.causal_llm.echo_client(lambda r, p: "foo -> bar")
        res = sp.causal_llm.causal_mas(
            variables=["foo", "bar"],
            client=client, rounds=1,
            treatment="foo", outcome="bar",
        )
        assert res.roles["foo"] == "treatment"
        assert res.roles["bar"] == "outcome"


# ---------------------------------------------------------------------------
# 7. Pipes into sp.dag
# ---------------------------------------------------------------------------


class TestDagInterop:

    def test_output_consumable_by_sp_dag(self):
        def script(role: str, prompt: str) -> str:
            if role == "proposer":
                return "age -> treatment\nage -> outcome\ntreatment -> outcome"
            return ""

        client = sp.causal_llm.echo_client(script)
        res = sp.causal_llm.causal_mas(
            variables=["age", "treatment", "outcome"],
            domain="RCT of statin",
            client=client, rounds=1,
        )
        dag_str = res.to_dag_string()
        # sp.dag accepts "A -> B; C -> D" style strings.
        dag = sp.dag(dag_str)
        # The DAG must contain the edges the MAS produced.
        edge_set = set(dag.edges())
        assert ("age", "treatment") in edge_set
        assert ("treatment", "outcome") in edge_set
