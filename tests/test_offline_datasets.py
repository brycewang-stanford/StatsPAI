"""Bundled datasets must load with no network access.

A user behind a blocked or unreliable connection should still be able to
run every example in the docs. The regression this guards against is a
loader (or a doc example) that quietly reaches for a remote CSV: it works
on the maintainer's laptop and fails with a connection reset for the
reader, usually deep inside a pipeline where the real cause is buried
under a cascade of downstream NameErrors.

Sockets are blocked for the duration of each test and restored after, so
this does not affect the rest of the suite.
"""

from __future__ import annotations

import socket
import warnings

import pytest

import statspai as sp

BUNDLED_LOADERS = [
    "nsw_lalonde",
    "nsw_dw",
    "mpdta",
    "card_1995",
    "california_prop99",
    "basque_terrorism",
    "german_reunification",
    "lee_2008_senate",
    "nhefs",
    "angrist_krueger_1991",
    "teen_employment",
]


@pytest.fixture
def no_network(monkeypatch):
    """Make any outbound connection raise, then restore."""

    def _blocked(*args, **kwargs):
        raise OSError("network access is blocked in this test")

    monkeypatch.setattr(socket, "socket", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    return None


@pytest.mark.parametrize("name", BUNDLED_LOADERS)
def test_loader_works_offline(no_network, name):
    loader = getattr(sp.datasets, name, None)
    assert loader is not None, f"sp.datasets.{name} is missing"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = loader()
    assert len(df) > 0, f"sp.datasets.{name}() returned an empty frame"


def test_real_lalonde_extract_is_bundled_not_fetched(no_network):
    """The real n=614 extract must come off disk, not the Rdatasets mirror."""
    df = sp.datasets.nsw_lalonde(simulated=False)
    assert df.shape == (614, 11)
    naive = (
        df.loc[df["treat"].eq(1), "re78"].mean()
        - df.loc[df["treat"].eq(0), "re78"].mean()
    )
    assert naive == pytest.approx(-635.026212, abs=1e-4)


def test_simulated_replica_works_offline(no_network):
    df = sp.datasets.nsw_lalonde(simulated=True)
    assert df.shape == (445, 10)
    assert df.attrs["expected_experimental_att"] == 1794


def test_dataset_module_declares_no_remote_urls():
    """A loader that grows a URL should fail here, not in a user's pipeline."""
    import inspect

    from statspai import datasets as datasets_module

    source = inspect.getsource(datasets_module._canonical)
    for needle in ("urlopen", "urlretrieve", "requests.get", 'read_csv("http'):
        assert needle not in source, (
            f"{needle!r} appeared in datasets/_canonical.py — bundled "
            "loaders must not reach the network"
        )
