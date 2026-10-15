"""Built-in datasets say whether they are real or simulated.

Several replicas carry the names of famous real datasets (``mpdta``,
``nsw_dw``, ``basque_terrorism``). Before 2026-10-16 ``mpdta`` and
``angrist_krueger_1991`` carried no ``simulated`` flag at all, and nothing
at load time told an agent that an estimate on them describes a
simulation. An agent that read only the name reported the replica's
simple ATT (-0.033) as the Callaway-Sant'Anna minimum-wage result.
"""

from __future__ import annotations

import warnings

import pytest

import statspai as sp
from statspai import datasets as D


def test_every_dataset_carries_a_consistent_flag():
    table = sp.list_datasets()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, simulated in zip(table["name"], table["simulated"]):
            df = sp.load_dataset(name)
            assert df.attrs["simulated"] is bool(simulated), name
            assert df.attrs["data_source"] in ("real", "simulated"), name
            assert (df.attrs["data_source"] == "simulated") is bool(simulated), name
    assert set(table.loc[table["simulated"], "name"]) >= {
        "mpdta",
        "nsw_dw",
        "angrist_krueger_1991",
        "basque_terrorism",
    }


def test_catalogue_lists_columns():
    table = sp.list_datasets()
    row = table.set_index("name").loc["mpdta"]
    assert row["columns"] == ["countyreal", "year", "lemp", "first_treat", "treat"]


def test_simulated_replica_warns_once_per_process(monkeypatch):
    monkeypatch.setattr(D, "_WARNED", set())
    sp.list_datasets()  # listing the catalogue does not use up the warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.datasets.mpdta()
        sp.datasets.mpdta()
        sp.datasets.card_1995()
    messages = [str(w.message) for w in caught if "SIMULATED" in str(w.message)]
    assert len(messages) == 1 and "mpdta" in messages[0]


def test_explicit_real_variant_is_flagged_real():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sp.datasets.nsw_dw(simulated=False).attrs["simulated"] is False
        assert sp.datasets.nsw_dw().attrs["simulated"] is True


def test_unknown_name_suggests_the_closest():
    with pytest.raises(sp.exceptions.MethodIncompatibility) as ei:
        sp.load_dataset("mpdat")
    assert ei.value.diagnostics["did_you_mean"][0] == "mpdta"


def test_search_finds_the_catalogue():
    for query in ("load example dataset", "示例数据"):
        names = [r["name"] for r in sp.search_functions(query)][:3]
        assert "load_dataset" in names, (query, names)
