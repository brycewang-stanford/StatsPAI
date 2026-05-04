"""Regression tests for the generated top-level type stub."""
from __future__ import annotations

from pathlib import Path


def test_type_stub_declares_stability_tiers_once():
    text = Path("src/statspai/__init__.pyi").read_text(encoding="utf-8")
    assert text.count("STABILITY_TIERS:") == 1
    assert "STABILITY_TIERS: frozenset[str]" in text


def test_type_stub_tracks_function_first_exports():
    text = Path("src/statspai/__init__.pyi").read_text(encoding="utf-8")
    for needle in (
        "from .bartik.shift_share import bartik as bartik",
        "from .deepiv.deep_iv import deepiv as deepiv",
        "from .bridge.core import bridge as bridge",
        "from .tmle.tmle import tmle as tmle",
    ):
        assert needle in text
