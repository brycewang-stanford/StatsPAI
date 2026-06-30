"""D3/D4b: standalone SE menu on the within-design path.

These lock in the wiring that lets ``cr2_se`` / ``wild_cluster_boot`` /
``conley`` / ``twoway_cluster`` operate on a stored (FE-absorbed) design instead
of re-parsing the formula and refitting plain OLS:

* The reroute is byte-for-byte identical to the old re-parse path on plain OLS
  (``sp.regress``) — proven by forcing both paths on the same model.
* ``sp.feols`` stores its within-transformed design, so the helpers now run on
  it (previously ``KeyError: 'X'`` / silently wrong).
* No-FE anchor: ``feols('y ~ x')`` has no fixed effects, so every standalone SE
  must equal the ``sp.regress`` value exactly.
* FE anchor: ``feols('y ~ x | g')`` wild bootstrap must equal ``sp.regress`` on
  *manually g-demeaned* data — the textbook within wild bootstrap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import statspai as sp  # noqa: E402
from statspai.inference.jackknife import _design_from_result  # noqa: E402


def _panel(seed: int = 7, n: int = 400, g: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, g, n)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "firm": grp,
            "yr": rng.integers(0, 6, n),
            "lat": rng.uniform(0, 10, n),
            "lon": rng.uniform(0, 10, n),
        }
    )
    df["y"] = (
        1.0
        + 0.4 * df["x"]
        - 0.2 * df["z"]
        + rng.normal(size=n)
        + rng.normal(size=g)[grp]
    )
    return df


def _se(res, var: str) -> float:
    return float(res.std_errors[var])


# --- helper path selection ------------------------------------------------


def test_design_source_stored_for_regress() -> None:
    df = _panel()
    r = sp.regress("y ~ x + z", df, cluster="firm")
    X, y, names, cl, src = _design_from_result(r, df, "firm")
    assert src == "stored"
    assert names == ["Intercept", "x", "z"]
    assert X.shape == (len(df), 3)


def test_design_source_falls_back_on_row_mismatch() -> None:
    df = _panel()
    r = sp.regress("y ~ x + z", df, cluster="firm")
    bigger = pd.concat([df, df.iloc[:5]], ignore_index=True)  # rows != stored X
    _, _, _, _, src = _design_from_result(r, bigger, "firm")
    assert src == "reparsed"


def test_reroute_byte_identical_to_reparse_on_regress() -> None:
    """Stored-array path and the legacy re-parse path must agree exactly."""
    df = _panel()
    r = sp.regress("y ~ x + z", df, cluster="firm")
    cr2_stored = sp.cr2_se(r, df, "firm")

    # Force the fallback by hiding the stored design.
    r_noX = sp.regress("y ~ x + z", df, cluster="firm")
    for key in ("X", "y", "var_names"):
        r_noX.data_info.pop(key, None)
    _, _, _, _, src = _design_from_result(r_noX, df, "firm")
    assert src == "reparsed"
    cr2_reparse = sp.cr2_se(r_noX, df, "firm")

    assert np.isclose(_se(cr2_stored, "x"), _se(cr2_reparse, "x"), atol=1e-12, rtol=0)
    assert np.isclose(_se(cr2_stored, "z"), _se(cr2_reparse, "z"), atol=1e-12, rtol=0)


# --- feols stores within design ------------------------------------------


def test_feols_stores_within_design() -> None:
    df = _panel()
    f = sp.feols("y ~ x + z | firm", data=df)
    di = f.data_info
    assert "X" in di and "y" in di and "var_names" in di
    assert di["var_names"] == ["x", "z"]  # FE absorbs the intercept
    assert np.asarray(di["X"]).shape == (len(df), 2)


# --- no-FE anchors: feols('y~x') ≡ regress -------------------------------


def test_feols_no_fe_matches_regress_cr2_and_conley_and_twoway() -> None:
    df = _panel()
    r = sp.regress("y ~ x + z", df)
    f0 = sp.feols("y ~ x + z", data=df)

    cr2_r = sp.cr2_se(r, df, "firm")
    cr2_f0 = sp.cr2_se(f0, df, "firm")
    assert np.isclose(_se(cr2_r, "x"), _se(cr2_f0, "x"), atol=1e-9)

    c_r = sp.conley(r, df, "lat", "lon", dist_cutoff=3.0)
    c_f0 = sp.conley(f0, df, "lat", "lon", dist_cutoff=3.0)
    assert np.isclose(_se(c_r, "x"), _se(c_f0, "x"), atol=1e-9)

    t_r = sp.twoway_cluster(r, df, "firm", "yr")
    t_f0 = sp.twoway_cluster(f0, df, "firm", "yr")
    assert np.isclose(_se(t_r, "x"), _se(t_f0, "x"), atol=1e-9)


def test_feols_no_fe_wild_identical_to_regress() -> None:
    df = _panel()
    r = sp.regress("y ~ x + z", df)
    f0 = sp.feols("y ~ x + z", data=df)
    wb_r = sp.wild_cluster_boot(r, df, "firm", "x", n_boot=499, seed=123)
    wb_f0 = sp.wild_cluster_boot(f0, df, "firm", "x", n_boot=499, seed=123)
    assert np.allclose(wb_r["t_distribution"], wb_f0["t_distribution"])
    assert wb_r["p_boot"] == wb_f0["p_boot"]


# --- FE anchor: feols wild ≡ regress on demeaned data --------------------


def test_feols_fe_wild_matches_within_demeaned_regress() -> None:
    df = _panel()
    f1 = sp.feols("y ~ x + z | firm", data=df)
    wb_fe = sp.wild_cluster_boot(f1, df, "firm", "x", n_boot=999, seed=42)

    dm = df.copy()
    for c in ("y", "x", "z"):
        dm[c] = df[c] - df.groupby("firm")[c].transform("mean")
    r_dm = sp.regress("y ~ x + z - 1", dm, cluster="firm")
    wb_dm = sp.wild_cluster_boot(r_dm, dm, "firm", "x", n_boot=999, seed=42)

    assert np.allclose(wb_fe["t_distribution"], wb_dm["t_distribution"])
    assert np.isclose(wb_fe["se_cluster"], wb_dm["se_cluster"], atol=1e-10)
    # the within-OLS coefficient must equal the feols coefficient
    assert np.isclose(wb_fe["beta_hat"], float(f1.params["x"]), atol=1e-10)


def test_feols_fe_standalone_helpers_run() -> None:
    """Previously KeyError: 'X' on feols — now they execute."""
    df = _panel()
    f1 = sp.feols("y ~ x + z | firm", data=df)
    assert np.isfinite(_se(sp.conley(f1, df, "lat", "lon", dist_cutoff=3.0), "x"))
    assert np.isfinite(_se(sp.twoway_cluster(f1, df, "firm", "yr"), "x"))
    wb = sp.wild_cluster_boot(f1, df, "firm", "x", n_boot=199, seed=1)
    assert 0.0 <= wb["p_boot"] <= 1.0
