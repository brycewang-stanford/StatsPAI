"""Five-layer score of the Stata translator on a frozen holdout corpus.

``tests/stata_translation_holdout/`` holds 39 commands written against
Stata's documented grammar, none taken from a replication package and
none used to develop the translator, with the numbers Stata 18 MP
produced for the 33 that are meant to run (double precision). For each
command the translator is scored on

1. **recognised** -- ``sp.from_stata`` names a StatsPAI call;
2. **translated** -- with no option left untranslated;
3. **executed** -- ``sp.stata`` runs it;
4. **same sample** -- the number of observations equals Stata's;
5. **same numbers** -- every named coefficient and SE equals Stata's
   (relative 1e-6; observed 1e-9 or better).

The headline is not the success rate. It is that there is **no silent
error**: every command either reproduces Stata or is refused out loud.
A command the translator cannot honour (a prefix that changes the
estimator, an undefined macro, an unknown option) is scored correct when
it is refused. The current gaps are refusals, listed in ``KNOWN_GAPS``;
closing one means moving it out of that set, and the layer counts below
move with it.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

HERE = Path(__file__).resolve().parent / "stata_translation_holdout"
CORPUS = json.loads((HERE / "corpus.json").read_text(encoding="utf-8"))
GOLD = json.loads((HERE / "holdout_Stata.json").read_text(encoding="utf-8"))

RTOL = 1e-6

#: Runnable in Stata, refused by the translator. Translation gaps, not
#: errors: each is a loud refusal that names what is missing.
KNOWN_GAPS = {
    "ols_fweight": "frequency weights",
    "ols_noconstant": "noconstant",
    "qreg": "qreg",
    "xtreg_re": "xtreg, re",
}

#: The constant is a different quantity by construction; the translation
#: note must say so (areg's _cons vs the first group's level).
CONSTANT_DIFFERS = {"areg"}


@pytest.fixture(scope="module")
def frames():
    return {
        "cross": pd.read_csv(HERE / "holdout_cross.csv"),
        "panel": pd.read_csv(HERE / "holdout_panel.csv"),
    }


def _translation(command: str) -> dict:
    try:
        out = sp.from_stata(command)
    except Exception as exc:  # noqa: BLE001 - a refusal is an outcome here
        return {"refused": f"{type(exc).__name__}: {exc}"}
    return out if isinstance(out, dict) else out.to_dict()


def _run(entry: dict, frames: dict):
    command = entry["command"]
    if entry["dataset"] == "panel":
        command = "xtset id t\n" + command
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(command, data=frames[entry["dataset"]])


def _n_obs(result) -> int:
    for attr in ("nobs", "n_obs"):
        value = getattr(result, attr, None)
        if value is not None:
            return int(value)
    return int(result.data_info["nobs"])


def _plain(name: str) -> bool:
    """A regressor both sides spell the same way (no factor / interaction)."""
    return name.replace("_", "").isalnum() and name not in ("_cons", "Intercept")


def _score(entry: dict, frames: dict) -> dict:
    tr = _translation(entry["command"])
    layers = {
        "recognised": "refused" not in tr and bool(tr.get("python_code")),
        "translated": False,
        "executed": False,
        "same_sample": False,
        "same_numbers": False,
        "note": " ".join(tr.get("notes") or []),
    }
    layers["translated"] = layers["recognised"] and not tr.get("untranslated_options")
    try:
        result = _run(entry, frames)
    except Exception as exc:  # noqa: BLE001 - refusal or failure, recorded
        layers["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return layers
    layers["executed"] = True
    gold = GOLD.get(entry["id"])
    if gold is None:
        return layers
    layers["same_sample"] = _n_obs(result) == int(gold["N"])
    params = {str(k): float(v) for k, v in dict(result.params).items()}
    ses = {str(k): float(v) for k, v in dict(result.std_errors).items()}
    stata = {
        name.split(":")[-1]: (b, se)
        for name, b, se in zip(gold["names"], gold["b"], gold["se"])
    }
    named = [n for n in stata if _plain(n) and n in params]
    ok = bool(named)
    for name in named:
        b, se = stata[name]
        ok = ok and np.isclose(params[name], b, rtol=RTOL, atol=1e-12)
        ok = ok and np.isclose(ses[name], se, rtol=RTOL, atol=1e-12)
    # Factor and interaction terms are spelled differently on the two
    # sides: every Stata value must appear among StatsPAI's.
    others = [stata[n][0] for n in stata if not _plain(n) and n != "_cons"]
    pool = list(params.values())
    for value in others:
        ok = ok and any(np.isclose(value, p, rtol=RTOL, atol=1e-12) for p in pool)
    if "_cons" in stata and "Intercept" in params:
        layers["constant_matches"] = bool(
            np.isclose(params["Intercept"], stata["_cons"][0], rtol=RTOL, atol=1e-12)
        )
    layers["same_numbers"] = bool(ok)
    return layers


@pytest.fixture(scope="module")
def scores(frames):
    return {entry["id"]: (entry, _score(entry, frames)) for entry in CORPUS}


def test_corpus_and_gold_are_the_frozen_set():
    assert len(CORPUS) == 39
    runnable = [c["id"] for c in CORPUS if c["expect"] == "run"]
    assert sorted(runnable) == sorted(k for k in GOLD if k != "_meta")
    assert all(
        GOLD[k]["rc"] == 0 for k in runnable
    ), "a holdout command failed in Stata"
    assert GOLD["_meta"]["precision"] == "double"


def test_no_silent_error(scores):
    """Every command reproduces Stata or is refused. Nothing in between."""
    silent = []
    for cid, (entry, layers) in scores.items():
        if entry["expect"] == "refuse":
            if layers["executed"]:
                silent.append(f"{cid}: ran a command that must be refused")
            continue
        if not layers["executed"]:
            continue  # a loud refusal; counted as a gap below
        if not layers["translated"]:
            silent.append(f"{cid}: executed with an untranslated option")
        if not (layers["same_sample"] and layers["same_numbers"]):
            silent.append(f"{cid}: executed, but sample or numbers differ from Stata")
    assert not silent, silent


def test_gaps_are_exactly_the_known_refusals(scores):
    refused = {
        cid: layers.get("error", "")
        for cid, (entry, layers) in scores.items()
        if entry["expect"] == "run" and not layers["executed"]
    }
    assert set(refused) == set(KNOWN_GAPS), refused
    for cid, message in refused.items():
        assert message.startswith("MethodIncompatibility"), (cid, message)
        assert "sp.stata" in message and len(message) > 60


def test_commands_that_must_be_refused_say_why(scores):
    for cid, (entry, layers) in scores.items():
        if entry["expect"] != "refuse":
            continue
        assert not layers["executed"]
        assert layers["error"].startswith("MethodIncompatibility"), (cid, layers)


def test_layer_counts(scores):
    """The five-layer baseline on the 33 runnable commands."""
    runnable = [layers for entry, layers in scores.values() if entry["expect"] == "run"]
    counts = {
        key: sum(1 for layers in runnable if layers[key])
        for key in (
            "recognised",
            "translated",
            "executed",
            "same_sample",
            "same_numbers",
        )
    }
    assert len(runnable) == 33
    assert counts == {
        "recognised": 30,  # fweight, qreg and xtreg, re are refused outright
        "translated": 29,  # noconstant is recognised but not translated
        "executed": 29,
        "same_sample": 29,
        "same_numbers": 29,
    }
    refusals = [e for e, _ in scores.values() if e["expect"] == "refuse"]
    assert len(refusals) == 6


def test_constant_is_the_only_thing_areg_changes_and_the_note_says_so(scores):
    differs = {
        cid
        for cid, (entry, layers) in scores.items()
        if layers.get("constant_matches") is False
    }
    assert differs == CONSTANT_DIFFERS
    _, layers = scores["areg"]
    assert layers["same_numbers"] is True  # slopes and SEs equal areg's
    assert "_cons" in layers["note"] and "Intercept" in layers["note"]


def test_if_in_and_missing_values_select_stata_s_sample(scores, frames):
    cross = frames["cross"]
    expected = {
        "ols_if": int((cross["d"] == 1).sum()),
        "ols_in": 200,
        "ols_if_and": int(((cross["d"] == 1) & (cross["x1"] > 0)).sum()),
        "ols_missing": int(cross["xm"].notna().sum()),
    }
    for cid, n in expected.items():
        assert int(GOLD[cid]["N"]) == n < len(cross)
        assert scores[cid][1]["same_sample"] is True
