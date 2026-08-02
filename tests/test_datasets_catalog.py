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
