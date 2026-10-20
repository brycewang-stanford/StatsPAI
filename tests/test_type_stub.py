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


def test_type_stub_declares_the_runtime_namespace():
    """The stub drifted by ~100 declarations (2026-10-20): every name in
    ``sp.__all__`` must be declared from where it lives, and nothing else.
    Fix with ``python3 scripts/generate_stub.py`` (``--check`` reports)."""
    import importlib.util

    import statspai as sp

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_generate_stub", root / "scripts" / "generate_stub.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    diff = mod.stale(sp, root / "src" / "statspai" / "__init__.pyi")
    assert diff == {"missing": [], "extra": []}, (
        f"{len(diff['missing'])} missing / {len(diff['extra'])} extra, e.g. "
        f"{(diff['missing'] + diff['extra'])[:5]}; run "
        "python3 scripts/generate_stub.py"
    )
