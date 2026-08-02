"""The dataset catalogue must describe the data you actually get.

``sp.datasets.list_datasets()`` is how a user picks an example dataset.
A row that misstates ``n_obs`` sends someone to a dataset of a different
size than they planned for, and nothing catches it — the catalogue was
hand-maintained beside the loaders and had drifted for three entries.

These tests also pin the property the catalogue exists to advertise:
every bundled dataset loads with no network at all.
"""

from __future__ import annotations

import socket
import warnings

import pytest

import statspai as sp


@pytest.fixture
def no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise OSError("network access is blocked in this test")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


def _catalog():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.datasets.list_datasets()


def test_every_row_names_a_real_loader():
    for name in _catalog()["name"]:
        assert hasattr(sp.datasets, name), f"catalogue lists missing {name}"


@pytest.mark.parametrize("name", list(_catalog()["name"]))
def test_row_count_matches_the_loader(name):
    """The number in the table is the number you get."""
    row = _catalog().query("name == @name").iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = getattr(sp.datasets, name)()
    assert (
        len(df) == row["n_obs"]
    ), f"{name}: catalogue says {row['n_obs']} rows, loader returns {len(df)}"


def test_source_column_matches_what_the_loader_returns():
    """'bundled CSV' must mean a real extract, not a replica."""
    table = _catalog()
    for _, row in table.iterrows():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = getattr(sp.datasets, row["name"])()
        is_real = df.attrs.get("data_source") == "real"
        if row["source"] == "bundled CSV":
            assert is_real, f"{row['name']} advertised as bundled but is a replica"
        else:
            assert not is_real, f"{row['name']} advertised simulated but is real"


def test_whole_catalogue_loads_offline(no_network):
    """The point of shipping data: it works with the network down."""
    for name in _catalog()["name"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = getattr(sp.datasets, name)()
        assert len(df) > 0, f"{name} came back empty offline"


# ---------------------------------------------------------------------- #
# The rule: ship the real data, hand back the real data
# ---------------------------------------------------------------------- #


def test_a_bundled_csv_means_a_real_default():
    """One rule, no per-dataset exceptions.

    Four loaders (card_1995, lee_2008_senate, california_prop99,
    nsw_lalonde) carry both a real extract and a calibrated replica. Only
    nsw_lalonde used to default to the real one, so what a bare call
    returned depended on which dataset you happened to reach for — and
    the replicas sit noticeably further from the published numbers
    (card_1995: OLS 0.110 on the replica against 0.075 in Table 2, versus
    0.074 on the real extract).

    If StatsPAI ships the real data, a bare call returns it.
    """
    from importlib import resources

    data_dir = resources.files("statspai.datasets") / "data"
    stems = {p.name.replace(".csv", "") for p in data_dir.iterdir()}

    # Map the CSV stems onto the loaders that read them.
    bundled_loaders = {
        "card_1995": "card_1995",
        "lee_2008_senate": "lee_2008_senate",
        "california_prop99": "california_prop99",
        "lalonde_matchit": "nsw_lalonde",
        "nhefs": "nhefs",
    }
    assert set(bundled_loaders) <= stems, sorted(stems)

    for stem, loader_name in bundled_loaders.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = getattr(sp.datasets, loader_name)()
        assert df.attrs.get("data_source") == "real", (
            f"{loader_name}() ships {stem}.csv but a bare call returns a "
            "replica — the default must be the real extract"
        )


def test_the_replica_is_still_reachable():
    """Flipping the default must not remove the teaching DGP."""
    for name in ("card_1995", "lee_2008_senate", "california_prop99", "nsw_lalonde"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            replica = getattr(sp.datasets, name)(simulated=True)
        assert len(replica) > 0
        assert replica.attrs.get("data_source") != "real"
