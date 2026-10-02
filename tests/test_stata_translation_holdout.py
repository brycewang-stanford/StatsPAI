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
it is refused. The first run found four gaps, all loud refusals; all four
are closed, and every one of the 33 runnable commands now reproduces
Stata. A new gap goes in ``KNOWN_GAPS`` only if it is refused out loud.
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

#: Runnable in Stata, refused by the translator. Empty since 2026-10-03:
#: the first run found four (`[fweight=]`, `noconstant`, `qreg`,
#: `xtreg, re`), all loud refusals. Add a row here only for a command that
#: is refused out loud; a command that runs and differs from Stata is a
#: failure of ``test_no_silent_error``, never a known gap.
KNOWN_GAPS: dict = {}

#: Translated only by ``sp.stata``, which can expand rows; ``sp.from_stata``
#: returns no single call for it and says how to write one.
RUNNER_ONLY = {"ols_fweight"}

#: Rows that were gaps on the first run. `noconstant` and `qreg` were then
#: fixed with this corpus in view, so for them it is a regression set.
#: `[fweight=]` and `xtreg, re` were closed the same night by a separate
#: line of work that had not seen the corpus: for those it was a holdout,
#: and they reproduce Stata.
CLOSED_AFTER_HOLDOUT = {"ols_noconstant", "qreg", "ols_fweight", "xtreg_re"}

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
        "untranslated": list(tr.get("untranslated_options") or []),
        "note": " ".join((tr.get("notes") or []) + [str(tr.get("error") or "")]),
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
        if layers["untranslated"]:
            silent.append(
                f"{cid}: executed although {layers['untranslated']} was not translated"
            )
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
    unrecognised = {
        cid
        for cid, (entry, layers) in scores.items()
        if entry["expect"] == "run" and not layers["recognised"]
    }
    assert unrecognised == RUNNER_ONLY
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
        "recognised": 32,  # [fweight=] is translated by the runner only
        "translated": 32,
        "executed": 33,
        "same_sample": 33,
        "same_numbers": 33,
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


def test_closed_gaps_now_reproduce_stata(scores):
    for cid in CLOSED_AFTER_HOLDOUT:
        _, layers = scores[cid]
        needed = ["executed", "same_sample", "same_numbers"]
        if cid not in RUNNER_ONLY:
            needed += ["recognised", "translated"]
        assert all(layers[k] for k in needed), (cid, layers)


@pytest.mark.parametrize(
    "command,x1,se",
    [
        # Stata 18 MP, double precision, on holdout_cross.csv; none of these
        # is in the corpus.
        ("regress y x1 x2, noconstant vce(robust)", 0.78149693351336, 0.09485394195736),
        ("regress y x1 x2, nocons vce(cluster g)", 0.78149693351336, 0.08416764702955),
        ("regress y x1 x2, noconstant vce(hc3)", 0.78149693351336, 0.09556885417732),
        ("qreg y x1 x2, quantile(0.25)", 0.90146786674513, 0.11233964520534),
        ("qreg y x1 x2, q(75)", 0.81014468527445, 0.10110583641684),
    ],
)
def test_closed_gaps_hold_outside_the_corpus(frames, command, x1, se):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata(command, data=frames["cross"])
    assert float(res.params["x1"]) == pytest.approx(x1, rel=1e-12)
    assert float(res.std_errors["x1"]) == pytest.approx(se, rel=1e-12)


@pytest.mark.parametrize(
    "command,se",
    [
        # Stata 18 MP on holdout_panel.csv; not in the corpus.
        ("xtreg y x", 0.05908036763599),  # the default is random effects
        ("xtreg y x, re vce(cluster id)", 0.06772959480012),
        ("xtreg y x, re vce(robust)", 0.06772959480012),
    ],
)
def test_random_effects_variants_hold_outside_the_corpus(frames, command, se):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata("xtset id t\n" + command, data=frames["panel"])
    assert float(res.params["x"]) == pytest.approx(0.93506756238168, rel=1e-12)
    assert float(res.std_errors["x"]) == pytest.approx(se, rel=1e-12)


def test_random_effects_interval_uses_the_normal_distribution(frames):
    """Stata reports z statistics for xtreg, re; a t interval would be wider."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata("xtset id t\nxtreg y x, re", data=frames["panel"])
    b, se = float(res.params["x"]), float(res.std_errors["x"])
    lo, hi = res.conf_int().loc["x"].tolist()
    assert lo == pytest.approx(b - 1.959963984540054 * se, rel=1e-10)
    assert hi == pytest.approx(b + 1.959963984540054 * se, rel=1e-10)


def test_frequency_weights_expand_rows_and_the_translator_says_how(frames):
    cross = frames["cross"]
    tr = sp.from_stata("regress y x1 x2 [fweight=fw]")
    assert tr["ok"] is False and "index.repeat" in tr["error"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata("regress y x1 x2 [fweight=fw]", data=cross)
    assert _n_obs(res) == int(cross["fw"].sum()) == int(GOLD["ols_fweight"]["N"])


def test_qreg_variance_option_is_refused_not_dropped(frames):
    """Only qreg's default variance is translated; vce() must not be ignored."""
    from statspai.exceptions import MethodIncompatibility

    assert sp.from_stata("qreg y x1 x2, vce(robust)")["untranslated_options"] == ["vce"]
    with pytest.raises(MethodIncompatibility, match="vce"):
        sp.stata("qreg y x1 x2, vce(robust)", data=frames["cross"])
