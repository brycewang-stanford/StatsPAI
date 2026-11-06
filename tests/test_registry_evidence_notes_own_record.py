"""Each function's parity notes come from its own parity-index record.

The eight network specs were built from one shared ``validation_notes``
list. The evidence pass appends each function's index notes to its spec,
so through 1.42.1 all eight carried all eight records' notes:
``sp.network_graph`` (known-truth evidence only) read "Cross-language
parity (bit-exact) vs R dyadRobust", which is ``sp.dyadic_regression``'s
record.
"""

from __future__ import annotations

import re

import statspai as sp
from statspai import registry as R

#: Prefixes of the notes ``_index_evidence_note`` renders from a record.
_GENERATED = re.compile(
    r"^(Cross-language parity \(|Cross-language comparison \(|"
    r"Known-truth recovery \(|Published-reference replication \()"
)


def _specs():
    R._ensure_full_registry()
    return dict(R._REGISTRY)


def test_no_two_specs_share_a_list():
    owners: dict = {}
    for name, spec in _specs().items():
        for field, value in vars(spec).items():
            if isinstance(value, list):
                other = owners.setdefault(id(value), (name, field))
                assert other == (name, field), (name, field, other)


def test_generated_notes_are_the_functions_own_record():
    records = R._parity_index_records()
    for name, spec in _specs().items():
        generated = [n for n in spec.validation_notes if _GENERATED.match(n)]
        if not generated:
            continue
        own = set(R._index_evidence_notes(records[name])) if name in records else set()
        stray = [n for n in generated if n not in own]
        assert not stray, (name, stray[:2])


def test_network_graph_claims_only_its_known_truth_evidence():
    notes = sp.describe_function("network_graph")["validation_notes"]
    assert sp.parity_status("network_graph")["status"] == "analytical-only"
    assert not any("Cross-language" in n for n in notes)
    assert not any("dyadRobust" in n for n in notes)


def test_a_t3_promotion_is_not_printed_as_parity():
    rec = sp.describe_function("community_detection")["evidence"]
    assert rec["grade"] == "T3"
    notes = sp.describe_function("community_detection")["validation_notes"]
    assert not any(n.startswith("Cross-language parity (") for n in notes)
    assert any("seed-replicated stochastic equivalence, T3" in n for n in notes)
