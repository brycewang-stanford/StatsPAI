"""The agent-native result protocol across the domain result dataclasses.

Design principle 3 wants every result object to expose ``to_dict`` /
``to_latex`` / ``cite`` through one entry point. ``ResultProtocolMixin``
(in ``statspai._result_serialize``) supplies all three to any result
dataclass that inherits it. This suite pins, for every class that opted in:

1. it subclasses the mixin and the three methods are callable;
2. ``_citation_keys`` did not leak into the dataclass fields;
3. **CLAUDE.md §10** — every citation key resolves to an entry in
   ``paper.bib`` (the single source of truth). A fabricated key fails here.
4. a representative, cheap-to-construct result round-trips through
   ``to_dict`` (strict JSON), ``to_latex`` (a table) and ``cite``.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._result_serialize import ResultProtocolMixin

REPO_ROOT = Path(__file__).resolve().parent.parent
_BIB_KEYS = set(
    re.findall(
        r"@\w+\{([A-Za-z0-9_]+),",
        (REPO_ROOT / "paper.bib").read_text(encoding="utf-8"),
    )
)

# The domain result classes that adopted the protocol (2026-06 sweep).
PROTOCOL_CLASSES = sorted(
    n
    for n in sp.__all__
    if n.endswith(("Result", "Results"))
    and isinstance(getattr(sp, n, None), type)
    and issubclass(getattr(sp, n), ResultProtocolMixin)
)


def test_protocol_class_count_is_substantial():
    # Guard against an accidental mass-revert of the mixin wiring.
    assert len(PROTOCOL_CLASSES) >= 60, PROTOCOL_CLASSES


class TestProtocolShape:
    @pytest.mark.parametrize("name", PROTOCOL_CLASSES)
    def test_methods_callable(self, name):
        c = getattr(sp, name)
        assert callable(c.to_dict)
        assert callable(c.to_latex)
        assert callable(c.cite)

    @pytest.mark.parametrize("name", PROTOCOL_CLASSES)
    def test_citation_keys_not_a_dataclass_field(self, name):
        c = getattr(sp, name)
        if is_dataclass(c):
            assert "_citation_keys" not in {f.name for f in fields(c)}

    @pytest.mark.parametrize("name", PROTOCOL_CLASSES)
    def test_citation_keys_resolve_in_paper_bib(self, name):
        """§10 zero-hallucination: no fabricated citation keys."""
        c = getattr(sp, name)
        missing = [k for k in getattr(c, "_citation_keys", ()) if k not in _BIB_KEYS]
        assert not missing, f"{name} cites keys not in paper.bib: {missing}"


class TestFunctionalRoundTrip:
    """A cheap-to-construct result exercises the three methods end to end."""

    def _check(self, r):
        payload = json.dumps(r.to_dict())  # strict JSON
        assert "NaN" not in payload and "Infinity" not in payload
        tex = r.to_latex()
        assert "\\begin{table}" in tex and "\\bottomrule" in tex
        cited = r.cite()
        assert isinstance(cited, str) and cited

    def test_rosenbaum(self):
        rng = np.random.default_rng(0)
        r = sp.rosenbaum_bounds(rng.normal(1, 1, 80), rng.normal(0, 1, 80))
        self._check(r)
        assert r.cite() == "rosenbaum2002observational"

    def test_its(self):
        t = np.arange(80)
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"y": 1 + 0.1 * t + (t >= 40) * 3 + rng.normal(size=80), "p": t}
        )
        r = sp.its(df, y="y", time="p", intervention=40)
        self._check(r)

    def test_cite_json_form(self):
        rng = np.random.default_rng(0)
        r = sp.rosenbaum_bounds(rng.normal(1, 1, 60), rng.normal(0, 1, 60))
        j = r.cite(format="json")
        assert j["citation_keys"] == ["rosenbaum2002observational"]
        assert j["source"] == "paper.bib"

    def test_placeholder_when_no_keys(self):
        # BCFFactorExposureResult has no paper.bib reference → honest placeholder.
        assert sp.BCFFactorExposureResult._citation_keys == ()
