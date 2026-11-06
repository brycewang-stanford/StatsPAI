"""``statspai._reference_seeds`` maps functions to existing ``paper.bib`` keys.

Each seed is a method reference shown by ``sp.describe_function``. The
zero-hallucination rule (CLAUDE.md section 10) means a seed may only name a
key that the master bibliography already holds, so these tests resolve
every key against ``paper.bib`` and check the seed reaches the registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import statspai as sp
from statspai import registry as reg
from statspai._reference_seeds import REFERENCE_SEEDS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bib_keys() -> set:
    text = (REPO_ROOT / "paper.bib").read_text(encoding="utf-8")
    return set(re.findall(r"^@\w+\s*\{\s*([^,\s]+)\s*,", text, re.M))


def _split(value: str) -> list:
    return [k.strip() for k in value.split(",")]


@pytest.fixture(scope="module")
def registry():
    sp.list_functions()  # builds the full registry, seeds included
    return reg._REGISTRY


def test_every_seed_key_is_in_master_bib():
    keys = _bib_keys()
    unknown = {
        name: [k for k in _split(value) if k not in keys]
        for name, value in REFERENCE_SEEDS.items()
    }
    unknown = {n: ks for n, ks in unknown.items() if ks}
    assert not unknown, f"seed keys missing from paper.bib: {unknown}"


def test_seed_values_are_bare_key_lists():
    for name, value in REFERENCE_SEEDS.items():
        parts = _split(value)
        assert parts and all(re.fullmatch(r"[A-Za-z0-9_:\-]+", p) for p in parts), name
        assert len(parts) == len(set(parts)), name


def test_every_seeded_name_is_registered(registry):
    missing = sorted(n for n in REFERENCE_SEEDS if n not in registry)
    assert not missing, f"seeded names not in the registry: {missing}"


def test_seeds_fill_empty_references(registry):
    # A seed only fills a field nobody else set. A seed whose function
    # later gets a hand-written reference is dead: remove it then.
    filled = [n for n, v in REFERENCE_SEEDS.items() if registry[n].reference == v]
    assert len(filled) == len(REFERENCE_SEEDS)
    assert sp.describe_function("kaplan_meier")["reference"] == (
        "kaplan1958nonparametric"
    )


def test_seeded_reference_is_tagged_curated(registry):
    card = sp.agent_card("var")
    assert card["reference"] == "lutkepohl2005new"
    assert card["provenance"]["reference"] == "curated"
