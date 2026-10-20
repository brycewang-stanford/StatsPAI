"""sp.xtregar (fixed- and random-effects regression with an AR(1)
disturbance) against Stata 18 MP ``xtregar``.

Reference ``tests/fixtures/xtregar/stata_xtregar.do`` on 80 unbalanced
panels with gaps in the time index (and the 30 gap-free panels alone):
every ``rhotype``, iterated and ``twostep``, a fixed ``rhof``, ``lbi``, and
a time-invariant regressor in the random-effects model. Everything agrees
to about 1e-15.

One documented departure: the fixed-effects ``_cons``. Stata reports the
within-regression intercept divided by ``1 - rho`` but leaves its standard
error undivided; ours is divided too (Stata's equals ours times
``1 - rho``), and a simulation below shows ours is the sampling standard
deviation. The random-effects Wald statistic is Stata's; its degrees of
freedom are the number of slopes (Stata's label counts the constant).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

_FIX = Path(__file__).parent / "fixtures" / "xtregar"
_spec = importlib.util.spec_from_file_location("_xtregar_data", _FIX / "_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
STATA = json.loads((_FIX / "stata_xtregar.json").read_text(encoding="utf-8"))

_SCALARS = {
    "rho_ar": "e(rho_ar)",
    "sigma_u": "e(sigma_u)",
    "sigma_e": "e(sigma_e)",
    "rho_fov": "e(rho_fov)",
    "r2_within": "e(r2_w)",
    "r2_between": "e(r2_b)",
    "r2_overall": "e(r2_o)",
    "r2_adj": "e(r2_a)",
    "corr_u_xb": "e(corr)",
    "F": "e(F)",
    "F_u0": "e(F_f)",
    "rss": "e(rss)",
    "mss": "e(mss)",
    "rmse": "e(rmse)",
    "log_likelihood": "e(ll)",
    "log_likelihood_0": "e(ll_0)",
    "chi2": "e(chi2)",
    "Tbar": "e(Tbar)",
    "g_min": "e(g_min)",
    "g_max": "e(g_max)",
    "bfn_dw": "e(d1)",
    "lbi": "e(LBI)",
    "n_lbi": "e(N_LBI)",
    "n_groups": "e(N_g)",
}


@pytest.fixture(scope="module")
def df():
    return _mod.make_panel()


def _call(tag, df):
    parts = tag.split("_")
    method, rest, kw = parts[0], parts[1:], {}
    xs, data = ["x1", "x2"], df
    if rest and rest[-1] == "2s":
        kw["twostep"], rest = True, rest[:-1]
    if rest == ["rhof"]:
        kw["rhof"] = 0.5
    elif rest == ["lbi"]:
        kw["lbi"] = True
    elif rest == ["z"]:
        xs = ["x1", "x2", "z"]
    elif rest == ["bal"]:
        data = df[df["bal"] == 1]
        kw["lbi"] = method == "re"
    elif rest:
        kw["rhotype"] = rest[0]
    fit = sp.xtregar(data, y="y", x=xs, id="id", time="t", method=method, **kw)
    return fit, xs


@pytest.mark.parametrize("tag", sorted(STATA))
def test_xtregar_matches_stata(df, tag):
    fit, xs = _call(tag, df)
    ref = STATA[tag]
    rho = fit.model_info["rho_ar"]
    for v in xs + ["_cons"]:
        np.testing.assert_allclose(fit.params[v], ref[v][0], rtol=1e-10)
        se = fit.std_errors[v]
        if tag.startswith("fe") and v == "_cons":
            se = se * (1 - rho)  # Stata's undivided standard error
        np.testing.assert_allclose(se, ref[v][1], rtol=1e-10)
    for ours, theirs in _SCALARS.items():
        if ours in fit.model_info and ref.get(theirs, [None])[0] is not None:
            np.testing.assert_allclose(
                fit.model_info[ours], ref[theirs][0], rtol=1e-10, err_msg=ours
            )
    assert fit.data_info["nobs"] == ref["e(N)"][0]
    if tag.startswith("fe"):
        assert fit.data_info["df_resid"] == ref["e(df_r)"][0]
        assert fit.model_info["F_u0_df"] == (ref["e(df_a)"][0], ref["e(df_r)"][0])
    else:
        th = fit.model_info["theta"]
        for key, s in [("min", "min"), ("p5", "5"), ("p50", "50"), ("p95", "95")]:
            np.testing.assert_allclose(th[key], ref[f"e(thta_{s})"][0], rtol=1e-10)
        np.testing.assert_allclose(th["max"], ref["e(thta_max)"][0], rtol=1e-10)
        # Stata's e(df_m) counts the constant; the Wald test has p df
        assert fit.model_info["chi2_df"] == ref["e(df_m)"][0] - 1


def test_fe_constant_standard_error_is_the_sampling_sd():
    # rho fixed at its true value, panel effects held fixed: the reported
    # SE of _cons matches its Monte Carlo SD (Stata's is 1 - rho = 0.4
    # times smaller)
    rng = np.random.default_rng(1)
    G, T, rho = 40, 8, 0.6
    nu = rng.normal(size=G)
    X = rng.normal(size=(G, T))
    est, se = [], []
    for _ in range(300):
        e = np.zeros((G, T))
        e[:, 0] = rng.normal(size=G) / np.sqrt(1 - rho**2)
        for t in range(1, T):
            e[:, t] = rho * e[:, t - 1] + rng.normal(size=G)
        frame = pd.DataFrame(
            {
                "id": np.repeat(np.arange(G), T),
                "t": np.tile(np.arange(T), G),
                "y": (1 + 2 * X + nu[:, None] + e).ravel(),
                "x": X.ravel(),
            }
        )
        f = sp.xtregar(frame, y="y", x="x", id="id", time="t", method="fe", rhof=rho)
        est.append(f.params["_cons"])
        se.append(f.std_errors["_cons"])
    # MC SD of an SD from 300 draws is about 4%
    assert np.mean(se) / np.std(est, ddof=1) == pytest.approx(1.0, abs=0.15)


def test_fe_omits_time_invariant_regressor(df):
    with pytest.warns(UserWarning, match="z omitted"):
        with_z = sp.xtregar(
            df, y="y", x=["x1", "x2", "z"], id="id", time="t", method="fe"
        )
    without = sp.xtregar(df, y="y", x=["x1", "x2"], id="id", time="t", method="fe")
    assert list(with_z.params.index) == ["x1", "x2", "_cons"]
    np.testing.assert_allclose(with_z.params, without.params, rtol=1e-12)
    assert with_z.model_info["omitted"][0]["variable"] == "z"


def test_time_shift_and_row_order_do_not_matter(df):
    a = sp.xtregar(df, y="y", x=["x1", "x2"], id="id", time="t")
    shuffled = df.sample(frac=1.0, random_state=3).assign(t=df["t"] + 1990)
    b = sp.xtregar(shuffled, y="y", x=["x1", "x2"], id="id", time="t")
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-12)


def test_invalid_inputs_are_refused(df):
    kw = dict(y="y", x=["x1", "x2"], id="id")
    with pytest.raises(MethodIncompatibility, match="repeated"):
        sp.xtregar(pd.concat([df, df.iloc[:1]]), time="t", **kw)
    with pytest.raises(MethodIncompatibility, match="integer"):
        sp.xtregar(df.assign(t=df["t"] + 0.5), time="t", **kw)
    with pytest.raises(MethodIncompatibility, match="rhof"):
        sp.xtregar(df, time="t", rhof=1.0, **kw)
    with pytest.raises(MethodIncompatibility, match="twostep"):
        sp.xtregar(df, time="t", rhotype="onestep", twostep=True, **kw)
    with pytest.raises(MethodIncompatibility, match="rhotype"):
        sp.xtregar(df, time="t", rhotype="prais", **kw)
    with pytest.raises(MethodIncompatibility, match="collinear"):
        sp.xtregar(
            df.assign(x3=df["x1"] + df["x2"]),
            y="y",
            x=["x1", "x2", "x3"],
            id="id",
            time="t",
        )
    with pytest.raises(DataInsufficient):
        sp.xtregar(df[df["id"] == 1], time="t", **kw)


def test_rho_outside_unit_interval_is_refused():
    # with two periods per panel the demeaned residuals are (e, -e), so the
    # regress estimate of rho is exactly -1; the transformation needs
    # |rho| < 1
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "id": np.repeat(np.arange(30), 2),
            "t": np.tile([1, 2], 30),
            "y": rng.normal(size=60),
            "x": rng.normal(size=60),
        }
    )
    with pytest.raises(MethodIncompatibility, match="rho"):
        sp.xtregar(frame, y="y", x="x", id="id", time="t", rhotype="regress")
