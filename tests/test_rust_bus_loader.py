"""sp.datasets.rust_bus (offline) and sp.ddc with a supplied increment distribution.

The published-number checks on the real data are in
``tests/external_parity/test_rust1987_bus.py``. Nothing here touches the
network: the loader is pointed at a small archive built in memory.
"""

import io
import zipfile

import numpy as np
import pytest

import statspai as sp
from statspai.datasets import _fetch, _rust_bus
from statspai.exceptions import DataInsufficient, MethodIncompatibility

# one bus: number 7, first replacement at odometer 9,000, none after
HEADER = [7, 8, 75, 4, 79, 9000, 0, 0, 0, 9, 75]
READINGS = [2000, 6000, 8999, 10000, 13000, 21000]


def _one_bus(n_states=90, max_mileage=450_000.0):
    dt = np.array([HEADER + READINGS], dtype=float)
    return _rust_bus._discretise(dt, 4, n_states, max_mileage)


def test_discretisation_follows_stordat():
    df = _one_bus()
    # replacement counted from the first reading >= 9,000; mileage restarts
    np.testing.assert_array_equal(df["mileage"], [2000, 6000, 8999, 1000, 4000, 12000])
    # 5,000-mile bins, Rust's 1-based ceil shifted to start at 0
    assert df["state"].tolist() == [0, 1, 1, 0, 0, 2]
    # replaced between the 8,999 and the 10,000 readings
    assert df["replace"].tolist() == [0, 0, 1, 0, 0, 0]
    # the step after the replacement is the next reading's 1-based bin (1),
    # not the step from state 0 to state 0
    inc = df["increment"].to_numpy()
    assert np.isnan(inc[0])
    assert inc[1:].tolist() == [1, 0, 1, 0, 2]
    assert df["rust_sample"].tolist() == [False] + [True] * 5
    assert df["bus"].unique().tolist() == [7] and (df["group"] == 4).all()
    assert df["period"].tolist() == list(range(6))


def test_bin_width_follows_the_grid():
    # 175 bins on 450,000 miles: 2,571.4 miles each
    df = _one_bus(n_states=175)
    assert df["state"].tolist() == [0, 2, 3, 0, 1, 4]


def test_grid_that_does_not_cover_the_mileage_is_refused():
    with pytest.raises(MethodIncompatibility, match="max_mileage"):
        _one_bus(n_states=90, max_mileage=10_000.0)


def _archive(tmp_path):
    """A zip laid out like zurcher-data, with synthetic group 1 and 2 files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for group, (stem, n_bus) in _rust_bus._GROUPS.items():
            if group > 2:
                continue
            blocks = []
            for b in range(n_bus):
                blocks += [1000 + b, 1, 80, 0, 0, 0, 0, 0, 0, 1, 80]
                blocks += [1000.0 * (b + 1) * k for k in range(1, 5)]
            text = "\n".join(f"{v:g}" for v in blocks)
            zf.writestr(f"zd-43d37bd/data/original_data/{stem}.asc", text)
    path = tmp_path / "zd.zip"
    path.write_bytes(buf.getvalue())
    return path


def test_loader_reads_the_archive(tmp_path, monkeypatch):
    path = _archive(tmp_path)
    monkeypatch.setattr(_rust_bus, "fetch", lambda spec, download=True: path)
    df = sp.datasets.rust_bus(groups=[2, 1])
    assert list(df.columns) == list(_rust_bus.COLUMNS)
    assert df["group"].drop_duplicates().tolist() == [2, 1]
    assert len(df) == (4 + 15) * 4
    assert df.attrs["data_source"] == "real" and df.attrs["simulated"] is False
    assert df.attrs["sha256"] == _rust_bus.RUST_BUS.sha256
    assert df.attrs["groups"] == [2, 1]
    # a group absent from the archive is a data problem, not a crash
    with pytest.raises(DataInsufficient, match="a530875"):
        sp.datasets.rust_bus(groups=[4])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groups": (9,)},
        {"groups": (1, 1)},
        {"groups": ()},
        {"groups": "some"},
        {"n_states": 1},
        {"n_states": 90.5},
        {"max_mileage": 0.0},
    ],
)
def test_bad_arguments_are_refused_before_any_download(kwargs):
    with pytest.raises(MethodIncompatibility):
        sp.datasets.rust_bus(download=False, **kwargs)


def test_offline_without_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(_fetch.ENV_DATA_DIR, str(tmp_path))
    with pytest.raises(DataInsufficient):
        sp.datasets.rust_bus(download=False)
    assert not any(tmp_path.iterdir())


def test_remote_file_is_pinned_and_listed():
    spec = _rust_bus.RUST_BUS
    assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0
    assert spec.url.startswith("https://zenodo.org/") and spec.size > 0
    assert "MIT" in spec.licence and "3385050" in spec.origin
    row = sp.datasets.list_datasets().set_index("name").loc["rust_bus"]
    assert row["delivery"] == "fetched" and not row["simulated"]
    assert row["columns"] == list(_rust_bus.COLUMNS)


# ----------------------------------------------------------------------
# sp.ddc(transition=<1-D increment distribution>)
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel():
    return sp.dgp_ddc(n_units=200, n_periods=100, seed=3)


def _fit(df, **kw):
    return sp.ddc(
        df, "replace", "mileage", id="bus", time="t", beta=0.99, n_states=90, **kw
    )


def test_supplied_increments_equal_the_estimated_ones(panel):
    est = _fit(panel)
    p = np.array(est.transition_params["increment_probs"])
    sup = _fit(panel, transition=p)
    np.testing.assert_array_equal(sup.transition, est.transition)
    np.testing.assert_array_equal(sup.params.to_numpy(), est.params.to_numpy())
    assert sup.loglik == est.loglik
    assert sup.transition_params["mode"] == "increments_supplied"
    assert sup.transition_params["increments"] == list(range(p.size))


def test_supplied_increments_build_rust_transitions(panel):
    p = np.array([0.25, 0.5, 0.25])
    F = _fit(panel, transition=p).transition
    # keep: x -> x + d; replace: 0 -> d; mass beyond the top stays there
    np.testing.assert_allclose(F[0, 10, 10:13], p)
    np.testing.assert_allclose(F[1, 50, 0:3], p)
    np.testing.assert_allclose(F[0, 88, 88:90], [0.25, 0.75])
    np.testing.assert_allclose(F.sum(axis=2), 1.0)


@pytest.mark.parametrize(
    "p, match",
    [
        ([0.5, 0.6], "sum to"),
        ([1.2, -0.2], "non-negative"),
        ([np.nan, 1.0], "finite"),
        ([], "non-empty"),
    ],
)
def test_bad_increment_distributions_rejected(panel, p, match):
    with pytest.raises(MethodIncompatibility, match=match):
        _fit(panel, transition=np.array(p, dtype=float))


def test_reset_actions_checked_with_supplied_increments(panel):
    with pytest.raises(MethodIncompatibility, match="reset_actions"):
        _fit(panel, transition=[0.5, 0.5], reset_actions=["overhaul"])
