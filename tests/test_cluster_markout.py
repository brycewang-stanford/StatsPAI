"""Rows with a missing cluster variable leave the estimation sample.

Stata's ``vce(cluster v)`` marks observations with missing ``v`` out of
``e(sample)``; R ``fixest`` drops them with a note.  Before
``core._vcov_spec.markout_clusters`` StatsPAI's estimators disagreed: some
raised, some dropped the rows, and ``ivreg`` / ``liml`` silently kept them in
the fit (different coefficients and standard errors from Stata).  Every
estimator that takes the Stata ``vce()`` grammar must now return exactly what
it returns on the data with those rows removed, and say so.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import StatsPAIWarning


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "z": rng.normal(size=n),
            "g": rng.integers(0, 25, n).astype(float),
            "id": np.repeat(np.arange(100), 4),
            "t": np.tile(np.arange(4), 100),
        }
    )
    # A cluster variable constant within panel, for clogit / panel logit.
    df["gp"] = (df["id"] % 25).astype(float)
    df["d"] = (df.z + rng.normal(size=n) > 0).astype(float)
    df["y"] = 1 + 0.5 * df.x + 0.4 * df.d + rng.normal(size=n)
    df["yb"] = (df.y > 1.2).astype(int)
    df["yb2"] = (0.3 * df.x2 + rng.normal(size=n) > 0).astype(int)
    df["yc"] = rng.poisson(np.exp(0.3 + 0.3 * df.x))
    df["yo"] = np.digitize(df.y, [0.5, 1.5])
    df["yf"] = 1 / (1 + np.exp(-(df.y - 1)))
    df["alt"] = np.tile(np.arange(4), 100)
    df["chosen"] = 0
    pick = rng.integers(0, 4, 100)
    df.loc[np.arange(100) * 4 + pick, "chosen"] = 1
    with_na = df.copy()
    with_na.loc[[3, 17, 50, 51], ["g", "gp"]] = np.nan
    return with_na, with_na.dropna(subset=["g"])


CALLS = {
    "regress": lambda d: sp.regress("y ~ x", data=d, vce="cluster g"),
    "regress_cluster_kw": lambda d: sp.regress("y ~ x", data=d, cluster="g"),
    "ivreg": lambda d: sp.ivreg("y ~ (d ~ z) + x", data=d, vce="cluster g"),
    "liml": lambda d: sp.liml("y ~ (d ~ z) + x", data=d, vce="vce(cluster g)"),
    "glm": lambda d: sp.glm("yc ~ x", data=d, family="poisson", vce="cluster g"),
    "logit": lambda d: sp.logit("yb ~ x", data=d, vce="cluster g"),
    "probit": lambda d: sp.probit("yb ~ x", data=d, vce="cl g"),
    "cloglog": lambda d: sp.cloglog("yb ~ x", data=d, vce="cluster g"),
    "poisson": lambda d: sp.poisson("yc ~ x", data=d, vce="cluster g"),
    "nbreg": lambda d: sp.nbreg("yc ~ x", data=d, vce="cluster g"),
    "ppmlhdfe": lambda d: sp.ppmlhdfe("yc ~ x | id", data=d, vce="cluster g"),
    "ologit": lambda d: sp.ologit("yo ~ x", data=d, vce="cluster g"),
    "oprobit": lambda d: sp.oprobit("yo ~ x", data=d, vce="cluster g"),
    "mlogit": lambda d: sp.mlogit("yo ~ x", data=d, vce="cluster g"),
    "clogit": lambda d: sp.clogit("chosen ~ x", data=d, group="id", vce="cluster gp"),
    "zip_model": lambda d: sp.zip_model("yc ~ x", data=d, vce="cluster g"),
    "zinb": lambda d: sp.zinb("yc ~ x", data=d, vce="cluster g"),
    "hurdle": lambda d: sp.hurdle("yc ~ x", data=d, vce="cluster g"),
    "fracreg": lambda d: sp.fracreg(d, y="yf", x=["x"], vce="cluster g"),
    "betareg": lambda d: sp.betareg(d, y="yf", x=["x"], vce="cluster g"),
    "truncreg": lambda d: sp.truncreg(d, y="y", x=["x"], ll=0, vce="cluster g"),
    "biprobit": lambda d: sp.biprobit(
        d, y1="yb", y2="yb2", x1=["x"], x2=["x2"], vce="cluster g"
    ),
    "etregress": lambda d: sp.etregress(
        d, y="y", x=["x"], treatment="d", z=["z"], vce="cluster g"
    ),
    "panel_logit": lambda d: sp.panel_logit(
        d, y="yb", x=["x"], id="id", time="t", method="re", vce="cluster gp"
    ),
    "panel_probit": lambda d: sp.panel_probit(
        d, y="yb", x=["x"], id="id", time="t", method="re", vce="cluster gp"
    ),
}


def _fit(call, data):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call(data)
    return result, [w for w in caught if issubclass(w.category, StatsPAIWarning)]


def _vector(values):
    return np.asarray(pd.Series(values).to_numpy(), dtype=float)


@pytest.mark.parametrize("name", list(CALLS))
def test_missing_cluster_rows_are_marked_out(name, frames):
    with_na, complete = frames
    got, notes = _fit(CALLS[name], with_na)
    ref, _ = _fit(CALLS[name], complete)
    np.testing.assert_allclose(_vector(got.params), _vector(ref.params), rtol=1e-10)
    np.testing.assert_allclose(
        _vector(got.std_errors), _vector(ref.std_errors), rtol=1e-10
    )
    assert got.model_info["n_missing_cluster_dropped"] == 4
    assert any(
        "4 observation(s) with a missing cluster" in str(w.message) for w in notes
    )


def test_complete_clusters_are_untouched(frames):
    _, complete = frames
    fit, notes = _fit(CALLS["logit"], complete)
    assert "n_missing_cluster_dropped" not in fit.model_info
    assert not [w for w in notes if "missing cluster" in str(w.message)]


def test_all_missing_clusters_still_raise(frames):
    with_na, _ = frames
    empty = with_na.assign(g=np.nan)
    with pytest.raises(Exception, match="cluster"):
        _fit(CALLS["regress"], empty)


def test_warning_carries_agent_payload(frames):
    with_na, _ = frames
    _, notes = _fit(CALLS["poisson"], with_na)
    (note,) = [w.message for w in notes if "missing cluster" in str(w.message)]
    assert note.diagnostics == {"cluster": ["g"], "n_dropped": 4}


def test_margins_drops_incomplete_rows_like_e_sample_and_warns(frames):
    # Stata averages over e(sample): rows with a missing model variable are
    # not in it. (Until 1.32 sp.margins refused such data outright.) When the
    # rows left differ from the fit's own nobs, data is probably not the
    # estimation data, so that is flagged.
    with_na, complete = frames
    holes = complete.copy()
    holes.loc[holes.index[:3], "x"] = np.nan
    fit, _ = _fit(CALLS["logit"], complete)
    with pytest.warns(StatsPAIWarning, match="3 dropped"):
        out = sp.margins(fit, data=holes)
    assert out.attrs["n"] == len(complete) - 3
    assert np.isfinite(out["dy/dx"]).all()
    # at= fixing the incomplete column makes the rows usable again.
    out = sp.margins(fit, data=holes, at={"x": 0.0})
    assert np.isfinite(out["dy/dx"]).all()


def test_margins_on_marked_out_data_equals_complete_data(frames):
    # The rows the missing-cluster markout dropped are excluded again.
    with_na, complete = frames
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatsPAIWarning)
        fit = sp.logit("yb ~ x", data=with_na, vce="cluster g")
    pd.testing.assert_frame_equal(
        sp.margins(fit, data=with_na), sp.margins(fit, data=complete)
    )


def test_stata_margins_averages_over_the_estimation_sample(frames):
    # Stata's margins averages over e(sample): the rows the cluster markout
    # dropped must not come back through data=.
    with_na, complete = frames
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", StatsPAIWarning)
        got = sp.stata("logit yb x, vce(cluster g)\nmargins, dydx(x)", data=with_na)
        fit = sp.logit("yb ~ x", data=complete, vce="cluster g")
    pd.testing.assert_frame_equal(got, sp.margins(fit, data=complete))
