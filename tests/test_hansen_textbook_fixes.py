"""Unit tests for what the pass over Hansen's *Econometrics* changed in the
Stata translator, the .dta reader and the conditional logit.

The numerical evidence against Stata is in
``tests/reference_parity/test_hansen_methods_stata_parity.py``; the replay of
the book's own programs is the opt-in
``tests/external_parity/test_hansen_econometrics_logs.py``.
"""

import struct
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._translation._stata_run import StataSession


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(11)
    n = 240
    out = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "z1": rng.normal(size=n),
            "z2": rng.normal(size=n),
            "g": np.repeat(np.arange(24), 10),
            "a": rng.integers(1, 4, size=n),
            "c": rng.integers(1, 4, size=n),
        }
    )
    out["d"] = 0.6 * out.z1 + 0.4 * out.z2 + rng.normal(size=n)
    out["y"] = 1 + 0.5 * out.x1 + 0.5 * out.x2 + 0.8 * out.d + rng.normal(size=n)
    out["xp"] = np.exp(out.x1 / 2)
    return out


# ------------------------------------------------------------- expressions
def test_square_brackets_group_like_parentheses():
    session = StataSession(pd.DataFrame({"g": [0, 0, 1, 1, 1]}))
    for line in ("gen a = [_n+1]*2", "bys g: gen n = [_N]", "gen t = 2*[_N]"):
        session.run(line)
    # Stata 18 on the same five rows
    assert session.data["a"].tolist() == [4, 6, 8, 10, 12]
    assert session.data["n"].tolist() == [2, 2, 3, 3, 3]
    assert session.data["t"].tolist() == [10] * 5


def test_lincom_divides_by_a_number(df):
    fit = sp.regress("y ~ x1 + x2", data=df)
    a, b = sp.lincom(fit, "x1 + x2/5"), sp.lincom(fit, "x1 + 0.2*x2")
    assert a["estimate"] == b["estimate"] and a["se"] == b["se"]
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="divides"):
        sp.lincom(fit, "x1/x2")


def test_nlcom_reads_an_expression_that_opens_with_a_parenthesis():
    out = sp.from_stata("nlcom (_b[a]+_b[b])/(1-_b[c])")
    assert out["arguments"]["expression"] == "(_b[a]+_b[b])/(1-_b[c])"
    assert sp.from_stata("nlcom (_b[a]/_b[b])")["arguments"]["expression"] == (
        "_b[a]/_b[b]"
    )
    assert not sp.from_stata("nlcom (r1: _b[a]) (r2: _b[b])")["ok"]


# ----------------------------------------------------- new command handlers
def test_cnsreg_reads_the_constraints_defined_above_it(df):
    via = sp.stata(
        """
        constraint define 1 x1 = x2
        constraint 2 x1 + x2 = 1
        cnsreg y x1 x2 d, c(1 2) r
        """,
        data=df,
    )
    direct = sp.cnsreg("y ~ x1 + x2 + d", df, ["x1 = x2", "x1 + x2 = 1"], vce="robust")
    pd.testing.assert_series_equal(via.params, direct.params)
    pd.testing.assert_series_equal(via.std_errors, direct.std_errors)
    assert not sp.from_stata("cnsreg y x1 x2, constraints(1)")["ok"]
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not defined"):
        sp.stata("cnsreg y x1 x2, constraints(7)", data=df)


def test_nl_is_translated_with_its_starting_values(df):
    out = sp.from_stata(
        "nl (y = {a} + {b}*xp^{c}), initial(a 1 b 1 c 1) vce(cluster g)"
    )
    assert out["tool"] == "nls"
    assert out["arguments"] == {
        "formula": "y ~ {a} + {b}*xp^{c}",
        "start": {"a": 1.0, "b": 1.0, "c": 1.0},
        "cluster": "g",
    }
    via = sp.stata("nl (y = {a} + {b}*xp^{c}), initial(a 1 b 1 c 1) r", data=df)
    direct = sp.nls(
        "y ~ {a} + {b}*xp^{c}", df, start={"a": 1, "b": 1, "c": 1}, vce="robust"
    )
    pd.testing.assert_series_equal(via.params, direct.params)
    assert not sp.from_stata("nl (y = {xb: x1 x2}), r")["ok"]


def test_pca_and_factor_are_translated(df):
    out = sp.from_stata("pca x1 x2 z1 z2, comp(2)")
    assert out["python_code"] == "sp.pca(df, ['x1', 'x2', 'z1', 'z2'], n_components=2)"
    out = sp.from_stata("factor x1 x2 z1 z2, pcf fa(2)")
    assert out["arguments"] == {
        "variables": ["x1", "x2", "z1", "z2"], "method": "pcf", "n_factors": 2
    }  # fmt: skip
    assert not sp.from_stata("factor x1 x2 z1 z2, ml")["ok"]
    res = sp.stata("pca x1 x2 z1 z2", data=df)
    assert res.loadings.shape == (4, 4)


def test_estimates_stats_lists_the_stored_models(df):
    table = sp.stata(
        """
        quietly reg y x1
        estimates store m1
        quietly reg y x1 x2 d
        estimates store m2
        estimates stats m1 m2
        """,
        data=df,
    )
    assert list(table.index) == ["m1", "m2"]
    fit = sp.regress("y ~ x1 + x2 + d", data=df)
    n, rss = len(df), fit.data_info["rss"]
    ll = -n / 2 * (1 + np.log(2 * np.pi) + np.log(rss / n))
    assert table.loc["m2", "ll"] == pytest.approx(ll, rel=1e-12)
    assert table.loc["m2", "AIC"] == pytest.approx(-2 * ll + 2 * 4, rel=1e-12)
    assert table.loc["m2", "BIC"] == pytest.approx(-2 * ll + 4 * np.log(n), rel=1e-12)


def test_stored_rank_and_overid_results(df):
    session = StataSession(df)
    session.run("ivregress 2sls y x1 x2 (d = z1 z2 xp), r perfect")
    assert session.stored["e"]["rank"] == 4
    session.run("estat overid, forcenonrobust")
    r = session.stored["r"]
    assert {"sargan", "p_sargan", "basmann", "score", "p_score"} <= set(r)
    session.run("scalar s1 = r(sargan)")
    session.run("scalar p1 = chi2tail(2, s1)")
    assert session.stored["scalars"]["p1"] == pytest.approx(r["p_sargan"])


def test_interacted_factors_may_be_instruments(df):
    out = sp.from_stata("ivregress 2sls y x1 i.a (d = i.c#i.a)")
    assert out["ok"] and "(d ~ C(c):C(a))" in out["python_code"]
    # as regressors the same term is still refused: the coefficients would
    # not be the cell indicators Stata reports
    assert not sp.from_stata("reg y x1 i.a i.c#i.a")["ok"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.stata("ivregress 2sls y x1 i.a (d = i.c#i.a)", data=df)
    cells = pd.get_dummies(
        df["a"].astype(str) + "_" + df["c"].astype(str), drop_first=True
    ).astype(float)
    frame = pd.concat([df, cells], axis=1)
    by_cells = sp.iv(
        "y ~ x1 + C(a) + (d ~ " + " + ".join(f"Q('{c}')" for c in cells) + ")",
        data=frame, small=False,
    )  # fmt: skip
    assert fit.params["d"] == pytest.approx(by_cells.params["d"], rel=1e-9)


def test_lag_list_of_a_difference(df):
    frame = df.assign(t=np.arange(len(df)))
    session = StataSession(frame)
    session.run("tsset t")
    session.run("reg y L(1/2).D.x1")
    assert list(session.last.params.index)[1:] == ["x1_L1D1", "x1_L2D1"]
    expected = frame["x1"].diff().shift(2)
    np.testing.assert_allclose(
        session.data["x1_L2D1"].to_numpy()[3:], expected.to_numpy()[3:]
    )


def test_bootstrap_prefix_runs_and_says_its_draws_are_numpy(df):
    with pytest.warns(UserWarning, match="numpy"):
        out = sp.stata("bootstrap (_b[x1]/_b[x2]), reps(40): reg y x1 x2", data=df)
    fit = sp.regress("y ~ x1 + x2", data=df)
    assert out.estimate == pytest.approx(fit.params["x1"] / fit.params["x2"])
    assert out.n_boot == 40 and out.se > 0
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="strata"):
        sp.stata("bootstrap, reps(20) strata(g): reg y x1", data=df)


# ---------------------------------------------------------------- jackknife
def test_jackknife_of_the_mean_is_the_usual_standard_error(df):
    res = sp.jackknife(df, lambda d: d["y"].mean())
    assert res.se == pytest.approx(df["y"].std(ddof=1) / np.sqrt(len(df)), rel=1e-12)
    assert res.bias == pytest.approx(0.0, abs=1e-10)
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.jackknife(df.iloc[:2], lambda d: d["y"].mean())
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.jackknife(df, lambda d: d["y"].mean(), cluster="nope")


# ------------------------------------------------------------ format 110 .dta
def _format_110(path, names, kinds, rows):
    """A minimal Stata 7 file: ``kinds`` are 'l' (long) or 'd' (double)."""
    n_vars = len(names)
    text = lambda s, width: s.encode("ascii").ljust(width, b"\0")  # noqa: E731
    body = bytearray([110, 2, 1, 0])
    body += struct.pack("<HI", n_vars, len(rows))
    body += text("made for a test", 81) + text("5 Oct 2026 12:00", 18)
    body += bytes(ord(k) for k in kinds)
    body += b"".join(text(nm, 33) for nm in names)
    body += struct.pack("<" + "H" * (n_vars + 1), *([0] * (n_vars + 1)))
    body += b"".join(text("%9.0g", 12) for _ in names)
    body += b"".join(text("", 33) for _ in names)
    body += b"".join(text(f"label of {nm}", 81) for nm in names)
    body += b"\0" * 5
    for row in rows:
        body += b"".join(
            struct.pack("<i" if k == "l" else "<d", v) for k, v in zip(kinds, row)
        )
    path.write_bytes(bytes(body))


def test_read_data_reads_a_stata_7_file(tmp_path):
    path = tmp_path / "old.dta"
    rows = [(1, 0.5), (2, -1.25), (3, 7.0)]
    _format_110(path, ["id", "cost"], "ld", rows)
    with pytest.raises(ValueError):
        (
            pd.read_stata(path)
            if pd.__version__ < "3"
            else (_ for _ in ()).throw(ValueError)
        )
    data = sp.read_data(str(path))
    assert data["id"].tolist() == [1, 2, 3]
    assert data["cost"].tolist() == [0.5, -1.25, 7.0]
    assert data.attrs["_labels"]["cost"] == "label of cost"


# -------------------------------------------------------- conditional logit
def test_clogit_matches_the_groupwise_definition():
    rng = np.random.default_rng(5)
    cases, alts = 150, 4
    frame = pd.DataFrame(
        {
            "case": np.repeat(np.arange(cases), alts),
            "x1": rng.normal(size=cases * alts),
            "x2": rng.normal(size=cases * alts),
        }
    )
    utility = 0.8 * frame.x1 - 0.5 * frame.x2 + rng.gumbel(size=len(frame))
    frame["y"] = (utility == utility.groupby(frame.case).transform("max")).astype(int)
    frame.loc[frame.case == 0, "y"] = 0  # a choice set with nothing chosen
    shuffled = frame.sample(frac=1, random_state=2)
    fit = sp.clogit(data=shuffled, y="y", x=["x1", "x2"], group="case")
    beta = fit.params.to_numpy()

    ll, grad = 0.0, np.zeros(2)
    for _, block in frame[frame.case != 0].groupby("case"):
        X = block[["x1", "x2"]].to_numpy()
        p = np.exp(X @ beta)
        p /= p.sum()
        ll += float(np.log(p[block.y.to_numpy() == 1][0]))
        grad += X[block.y.to_numpy() == 1][0] - p @ X
    assert fit.diagnostics["Log-Likelihood"] == pytest.approx(ll, rel=1e-12)
    np.testing.assert_allclose(grad, 0.0, atol=1e-9)
    assert fit.diagnostics["n_groups"] == cases - 1
    assert (
        shuffled.loc[shuffled.case == 0]
        .index.isin(fit.predicted_probs.index[fit.predicted_probs == 0])
        .all()
    )


# ------------------------------------------------- choice models in sp.stata
@pytest.fixture(scope="module")
def trips():
    rng = np.random.default_rng(8)
    cases, modes = 300, [1, 2, 3]
    frame = pd.DataFrame(
        {
            "trip": np.repeat(np.arange(cases), 3),
            "mode": np.tile(modes, cases),
            "cost": rng.uniform(1, 10, cases * 3),
            "income": np.repeat(rng.uniform(1, 8, cases), 3),
        }
    )
    u = (
        -0.4 * frame.cost
        + 0.3 * frame.income * (frame["mode"] == 2)
        + rng.gumbel(size=len(frame))
    )
    frame["chosen"] = (u == u.groupby(frame.trip).transform("max")).astype(int)
    return frame


def test_cmclogit_is_clogit_on_the_interacted_design(trips):
    session = StataSession(trips)
    session.run("cmset trip mode")
    session.run("cmclogit chosen cost, basealternative(1) casevars(income)")
    fit = session.output
    assert list(fit.params.index) == [
        "cost", "2:income", "2:_cons", "3:income", "3:_cons"
    ]  # fmt: skip
    design = trips.copy()
    for m in (2, 3):
        design[f"i{m}"] = design.income * (design["mode"] == m)
        design[f"c{m}"] = (design["mode"] == m).astype(float)
    direct = sp.clogit(
        data=design, y="chosen", x=["cost", "i2", "c2", "i3", "c3"], group="trip"
    )
    np.testing.assert_allclose(fit.params, direct.params, rtol=1e-10)
    np.testing.assert_allclose(fit.std_errors, direct.std_errors, rtol=1e-10)
    assert fit.model_info["base_alternative"] == "1"
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="cmset"):
        sp.stata("cmclogit chosen cost, casevars(income)", data=trips)


def test_margins_after_cmclogit_is_the_average_derivative(trips):
    session = StataSession(trips)
    session.run("cmset trip mode")
    session.run("cmclogit chosen cost, basealternative(1) casevars(income)")
    fit = session.output
    session.run("margins, dydx(cost) outcome(1) alternative(1)")
    own = session.output
    session.run("margins, dydx(cost) outcome(1) alternative(2)")
    cross = session.output

    design = session.stored["cm_fit"]["frame"]
    X = design[list(fit.params.index)].to_numpy(float)
    beta = fit.params.to_numpy()

    def share_of_one(shift_mode):
        Xs = X.copy()
        Xs[(design["mode"] == shift_mode).to_numpy(), 0] += 1e-5
        e = np.exp(Xs @ beta)
        p = e / pd.Series(e).groupby(design.trip.to_numpy()).transform("sum")
        return p[(design["mode"] == 1).to_numpy()].mean()

    base = share_of_one(shift_mode=99)
    assert own["dydx"] == pytest.approx((share_of_one(1) - base) / 1e-5, rel=1e-4)
    assert cross["dydx"] == pytest.approx((share_of_one(2) - base) / 1e-5, rel=1e-4)
    assert own["dydx"] < 0 < cross["dydx"] and own["se"] > 0
    # another choice model in between: margins no longer answers from the old fit
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        session.run("cmmprobit chosen cost, casevars(income)")
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        session.run("margins, dydx(cost) outcome(1) alternative(1)")


def test_discrete_choice_results_keep_their_covariance():
    """Joint tests after mlogit / ologit / oprobit / clogit were refused:
    the fits kept standard errors only."""
    import statsmodels.api as sm

    rng = np.random.default_rng(0)
    n = 600
    d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    d["y3"] = pd.cut(d.x1 + rng.logistic(size=n), [-9, -0.5, 0.5, 9], labels=False)
    fit = sp.mlogit("y3 ~ x1 + x2", data=d)
    ref = sm.MNLogit(d.y3, sm.add_constant(d[["x1", "x2"]])).fit(
        disp=0, method="newton", tol=1e-12
    )
    np.testing.assert_allclose(
        fit.data_info["var_cov"], np.asarray(ref.cov_params()), rtol=1e-8
    )
    wald = sp.test(fit, ["[1]x2 = 0", "[2]x2 = 0"])
    theirs = ref.wald_test(np.eye(6)[[2, 5]], scalar=True)
    assert wald["statistic"] == pytest.approx(float(theirs.statistic), rel=1e-8)
    for model in (sp.ologit, sp.oprobit):
        ordered = model("y3 ~ x1 + x2", data=d, robust="robust")
        V = np.asarray(ordered.data_info["var_cov"])
        np.testing.assert_allclose(np.sqrt(np.diag(V)), ordered.std_errors)
        joint = sp.test(ordered, ["x1 = 0", "x2 = 0"])
        b = ordered.params[["x1", "x2"]].to_numpy()
        assert joint["statistic"] == pytest.approx(b @ np.linalg.solve(V[:2, :2], b))
    odds = sp.mlogit("y3 ~ x1 + x2", data=d, rrr=True)
    np.testing.assert_allclose(
        np.sqrt(np.diag(odds.data_info["var_cov"])), odds.std_errors
    )
