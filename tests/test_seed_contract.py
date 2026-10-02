"""Every seed argument is one the result card can read (review item A3).

Default seeds are not uniform across StatsPAI and are not made uniform
here: changing one changes seeded numbers. What is enforced is narrower
and cheaper. ``sp.result_card`` reports ``seed`` / ``reproducible`` /
``seed_source`` by looking for a known argument name; a function that
spells its seed differently gets a card that is silent about the
reproducibility of its bootstrap or sample split. Six such spellings were
in the registry (``boot_seed``, ``bootstrap_seed``, ``rng_seed``,
``wild_seed``, ``halton_seed``, ``rng``).
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import pytest

import statspai as sp
from statspai._result_contract import _SEED_PARAMS, seed_record

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "seed_inventory.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="source checkout only (scripts/ not installed)"
)


@pytest.fixture(scope="module")
def rows():
    spec = importlib.util.spec_from_file_location("seed_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inventory()


def test_every_seed_like_argument_is_read_by_the_result_card(rows):
    assert len(rows) > 200, "inventory unexpectedly small"
    unread = sorted(
        f"sp.{r['function']}({r['parameter']}=)"
        for r in rows
        if not r["recognised_by_result_card"]
    )
    assert not unread, (
        "seed arguments the result card does not read (add the name to "
        f"statspai._result_contract._SEED_PARAMS): {unread}"
    )


def test_inventory_covers_every_recognised_spelling(rows):
    used = {r["parameter"] for r in rows}
    assert used <= set(_SEED_PARAMS)
    assert {"seed", "random_state", "boot_seed"} <= used


@pytest.mark.parametrize(
    "seed,expected",
    [(7, (7, True)), (None, (None, False))],
    ids=["integer", "unset"],
)
def test_bootstrap_seed_reaches_the_card(seed, expected):
    data = sp.datasets.mpdta()
    keys = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.did_imputation(
            data, **keys, vce="bootstrap", n_boot=15, boot_seed=seed
        )
    prov = sp.result_card(fit)["provenance"]
    assert (prov["seed"], prov["reproducible"]) == expected
    assert prov["seed_source"] == "model_info"


def test_bootstrap_se_with_a_seed_is_actually_reproducible():
    """The card's ``reproducible: True`` is a claim; check it."""
    data = sp.datasets.mpdta()
    keys = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.gardner_did(data, **keys, vce="bootstrap", n_boot=25, boot_seed=3)
        b = sp.gardner_did(data, **keys, vce="bootstrap", n_boot=25, boot_seed=3)
        c = sp.gardner_did(data, **keys, vce="bootstrap", n_boot=25, boot_seed=4)
    assert sp.result_card(a)["provenance"]["reproducible"] is True
    assert a.se == b.se
    assert a.se != c.se
    assert a.estimate == b.estimate == c.estimate


def test_unrecorded_seed_is_reported_as_unknown_not_as_absent():
    record = seed_record("did_imputation", {}, {})
    assert record["seed_source"] == "not_recorded"
    assert record["reproducible"] is None
    assert "boot_seed" in record["seed_note"]
