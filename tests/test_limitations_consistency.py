"""Runtime consistency tests for ``FunctionSpec.limitations``.

The ``limitations`` field on every :class:`statspai.registry.FunctionSpec`
declares parameter values / variant gaps that are documented as
not-yet-implemented inside an otherwise stable function — see
``docs/guides/stability.md``. Without this test those advertisements
silently rot the moment the underlying code learns a new variant or the
function is renamed: the registry would still claim the gap exists but
the runtime no longer enforces it (or vice versa).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest


ALLOWED_PHRASES: Tuple[str, ...] = (
    "not yet implemented",
    "not implemented",
    "not yet supported",
    "is not supported",
    "not currently supported",
    "raises notimplementederror",
    "raises importerror",
    "currently requires",
    "currently falls back",
    "is an mvp",
    "mvp",
    "only",
    "fallback",
    "silently",
    "deterministic fallback",
)


LIMITATIONS_DESCRIPTIVE_ONLY: Dict[str, List[str]] = {
    "did_multiplegt_dyn": [
        "switch-on only",
        "SE is cluster bootstrap",
        "heteroskedastic-weights variant",
    ],
    "continuous_did": [
        "method='cgs' is an MVP",
    ],
    "network_exposure": [
        "design='complete' currently falls back",
    ],
    "text_treatment_effect": [
        "embedder='sbert' requires the optional sentence-transformers",
        "Veitch et al. (2020) full BERT/topic-model recipe",
    ],
    "principal_strat": [
        # The function now implements the basic AIR / Wald LATE under
        # the encouragement-design path (Step G); the remaining gap is
        # always-survivor SACE under Mealli & Pacini (2013) partial
        # identification, which has no hard exception to test.
        "Always-survivor SACE under encouragement design",
    ],
}


def _runtime_map() -> Dict[Tuple[str, str], Tuple[Callable[[], Any], type | tuple[type, ...]]]:
    """Build the (function_name, limitation_substring) -> (call, exc) map."""
    import statspai as sp

    rng = np.random.default_rng(0)
    n = 200

    df_panel = pd.DataFrame({
        "y": rng.normal(size=n),
        "i": np.repeat(np.arange(n // 4), 4),
        "t": np.tile(np.arange(4), n // 4),
        "g": np.repeat(rng.choice([0, 2, 3], size=n // 4), 4),
        "d": rng.binomial(1, 0.4, size=n),
        "dose": rng.uniform(0, 1, size=n),
    })

    df_cs = pd.DataFrame({
        "y": rng.normal(size=n),
        "d": rng.binomial(1, 0.5, size=n).astype(float),
        "s": rng.binomial(1, 0.5, size=n).astype(float),
        "x": rng.normal(size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "z": rng.binomial(1, 0.5, size=n).astype(float),
        "w": rng.normal(size=n),
        "score": rng.normal(size=n),
    })

    return {
        ("hal_tmle", "variant='projection'"): (
            lambda: sp.hal_tmle(
                df_cs, y="y", treat="d", covariates=["x1", "x2"],
                variant="projection",
            ),
            NotImplementedError,
        ),
        ("callaway_santanna", "panel=False"): (
            lambda: sp.callaway_santanna(
                df_panel, y="y", g="g", t="t", i="i",
                panel=False, estimator="dr",
            ),
            NotImplementedError,
        ),
        ("etwfe", "cgroup='nevertreated' combined with panel=False"): (
            lambda: sp.etwfe(
                df_panel, y="y", group="i", time="t", first_treat="g",
                cgroup="nevertreated", panel=False,
            ),
            NotImplementedError,
        ),
        ("rdrobust", "observation-level weights"): (
            lambda: sp.rdrobust(
                df_cs, y="y", x="x", c=0.0, weights="w",
            ),
            NotImplementedError,
        ),
        ("llm_annotator_correct", "method='hausman' is the only supported"): (
            lambda: sp.llm_annotator_correct(
                annotations_llm=df_cs["d"],
                outcome=df_cs["y"],
                annotations_human=df_cs["d"][:60],
                method="logistic",
            ),
            (NotImplementedError, ValueError),
        ),
    }


def _matches_allowed_vocabulary(limitation: str) -> bool:
    low = limitation.lower()
    return any(phrase in low for phrase in ALLOWED_PHRASES)


def _all_limitations() -> List[Tuple[str, str]]:
    """List every (function_name, limitation_string) pair in the registry."""
    import statspai as sp

    sp.list_functions()
    from statspai.registry import _REGISTRY

    out: List[Tuple[str, str]] = []
    for name, spec in sorted(_REGISTRY.items()):
        for lim in spec.limitations:
            out.append((name, lim))
    return out


@pytest.mark.parametrize("name,limitation", _all_limitations())
def test_limitation_uses_allowed_vocabulary(name: str, limitation: str) -> None:
    """Every registered limitation must use vetted phrasing."""
    assert _matches_allowed_vocabulary(limitation), (
        f"sp.{name} limitation does not use vetted vocabulary: {limitation!r}"
    )


def test_every_limitation_is_classified() -> None:
    """Every limitation must be either runtime-tested or explicitly descriptive."""
    runtime_keys = {key for key in _runtime_map().keys()}
    descriptive_keys = {
        (fn, sub) for fn, subs in LIMITATIONS_DESCRIPTIVE_ONLY.items()
        for sub in subs
    }

    unclassified: List[Tuple[str, str]] = []
    for name, limitation in _all_limitations():
        runtime_match = any(
            fn == name and sub in limitation for fn, sub in runtime_keys
        )
        descriptive_match = any(
            fn == name and sub in limitation for fn, sub in descriptive_keys
        )
        if not (runtime_match or descriptive_match):
            unclassified.append((name, limitation))

    assert not unclassified, (
        "The following limitations are not classified as either "
        "runtime-testable or descriptive:\n"
        + "\n".join(f"  sp.{n}: {lim!r}" for n, lim in unclassified)
    )


@pytest.mark.parametrize(
    "key", list(_runtime_map().keys()),
    ids=lambda k: f"{k[0]}::{k[1][:40]}",
)
def test_limitation_actually_raises(
    key: Tuple[str, str],
) -> None:
    """The documented limitation must trigger the documented exception."""
    call, exc = _runtime_map()[key]
    with pytest.raises(exc):
        call()
