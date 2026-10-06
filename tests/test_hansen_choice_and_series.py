"""Nested logit with a fixed nest, multinomial probit, series regression and
the small ``sp.stata`` additions of the Hansen pass.

Stata parity on committed data is in
``tests/reference_parity/test_hansen_methods_stata_parity.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _choices(n=500, J=4, seed=0):
    rng = np.random.default_rng(seed)
    names = ["air", "bus", "car", "train"][:J]
    long = pd.DataFrame(
        {
            "case": np.repeat(np.arange(n), J),
            "alt": np.tile(np.arange(1, J + 1), n),
            "name": np.tile(names, n),
            "cost": rng.normal(size=J * n),
            "inc": np.repeat(rng.normal(size=n), J),
        }
    )
    u = -1.0 * long.cost + 0.5 * long.inc * (long.alt == 3) + rng.normal(size=J * n)
    long["choice"] = (u == u.groupby(long.case).transform("max")).astype(int)
    return long


# -------------------------------------------------------------- nested logit
def test_a_nest_fixed_at_one_is_singleton_nests():
    long = _choices()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed = sp.nlogit(
            long, y="choice", x="cost", chid="case", alt="alt",
            nests={"A": [1, 3], "B": [2, 4]}, fixed_lambda={"B": 1.0},
        )  # fmt: skip
        single = sp.nlogit(
            long, y="choice", x="cost", chid="case", alt="alt",
            nests={"A": [1, 3], "b2": [2], "b4": [4]},
        )  # fmt: skip
    assert "lambda:B" not in fixed.params.index
    assert np.isclose(fixed.model_info["ll"], single.model_info["ll"], atol=1e-8)
    np.testing.assert_allclose(fixed.params, single.params, atol=1e-5)
    assert fixed.model_info["lr_iia_df"] == 1
    assert fixed.model_info["fixed_lambda"] == {"B": 1.0}


def test_nlogit_base_and_refusals():
    long = _choices()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = sp.nlogit(
            long, y="choice", x="cost", chid="case", alt="alt",
            nests={"A": [1, 3], "B": [2, 4]},
        )  # fmt: skip
        other = sp.nlogit(
            long, y="choice", x="cost", chid="case", alt="alt",
            nests={"A": [1, 3], "B": [2, 4]}, base=4,
        )  # fmt: skip
    assert "_cons:4" not in other.params.index and "_cons:1" in other.params.index
    # the base is a relabelling: same likelihood, same slope
    assert np.isclose(first.model_info["ll"], other.model_info["ll"], atol=1e-7)
    assert np.isclose(first.params["cost"], other.params["cost"], atol=1e-5)
    assert np.isclose(other.params["_cons:1"], -first.params["_cons:4"], atol=1e-4)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="fixed_lambda"):
        sp.nlogit(long, y="choice", x="cost", chid="case", alt="alt",
                  nests={"A": [1, 3], "B": [2, 4]}, fixed_lambda={"C": 1})  # fmt: skip
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="base"):
        sp.nlogit(long, y="choice", x="cost", chid="case", alt="alt",
                  nests={"A": [1, 3], "B": [2, 4]}, base=9)  # fmt: skip


def test_nlogit_through_sp_stata():
    long = _choices()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = sp.nlogit(
            long, y="choice", x="cost", chid="case", alt="alt",
            nests={"Fast": [1, 3], "Slow": [2, 4]}, fixed_lambda={"Slow": 1.0},
            base=4,
        )  # fmt: skip
        run = sp.stata(
            """
            nlogitgen type = alt(Fast: 1 | 3, Slow: 2 | 4)
            constraint 1 [/type]Slow_tau = 1
            nlogit choice cost || type: || alt:, case(case) base(4) constraints(1)
            """,
            data=long,
        )
    np.testing.assert_allclose(run.params, direct.params, atol=1e-8)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="nlogitgen"):
        sp.stata("nlogit choice cost || type: || alt:, case(case)", data=long)


# --------------------------------------------------------- multinomial probit
def test_orthant_probabilities_against_scipy():
    from scipy.stats import multivariate_normal

    from statspai.regression.mprobit import _nodes, _orthant

    rng = np.random.default_rng(3)
    for d in (2, 3):
        a = rng.normal(size=(d, d))
        sigma = a @ a.T + 0.3 * np.eye(d)
        limits = rng.normal(size=(6, d))
        ours = _orthant(limits, np.linalg.cholesky(sigma), *_nodes(d - 1, None))
        ref = [
            multivariate_normal(mean=np.zeros(d), cov=sigma).cdf(row) for row in limits
        ]
        # scipy's own integration error is about 1e-5 in three dimensions
        np.testing.assert_allclose(ours, ref, atol=2e-6 if d == 2 else 5e-5)


def test_probabilities_sum_to_one_and_structures_nest():
    long = _choices(n=300, J=3)
    kw = dict(y="choice", x="cost", case_vars="inc", chid="case", alt="alt")
    hom = sp.mprobit(long, correlation="independent", stddev="homoskedastic", **kw)
    het = sp.mprobit(long, correlation="independent", stddev="heteroskedastic", **kw)
    uns = sp.mprobit(long, **kw)
    for fit in (hom, het, uns):
        p = fit.model_info["probabilities"].to_numpy()
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
        assert fit.model_info["converged"]
        assert fit.model_info["covariance"].iloc[0, 0] == pytest.approx(2.0)
    # each model contains the one before it
    assert hom.model_info["ll"] <= het.model_info["ll"] + 1e-6
    assert het.model_info["ll"] <= uns.model_info["ll"] + 1e-6
    assert uns.model_info["ll_independent"] == pytest.approx(hom.model_info["ll"])
    assert list(het.params.index)[-1] == "lnsigma:3"
    assert [n for n in uns.params.index if n.startswith("l")] == ["lnl2_2", "l2_1"]
    # the cost coefficient is recovered (true value -1, error variance one)
    assert abs(hom.params["cost"] + 1.0) < 0.2


def test_mprobit_base_is_a_relabelling():
    long = _choices(n=300, J=3)
    kw = dict(y="choice", x="cost", chid="case", alt="alt",
              correlation="independent", stddev="homoskedastic")  # fmt: skip
    first = sp.mprobit(long, **kw)
    last = sp.mprobit(long, base=3, **kw)
    assert np.isclose(first.model_info["ll"], last.model_info["ll"], atol=1e-7)
    assert np.isclose(first.params["cost"], last.params["cost"], atol=1e-5)
    assert np.isclose(last.params["1:_cons"], -first.params["3:_cons"], atol=1e-4)


def test_mprobit_refusals():
    long = _choices(n=120, J=3)
    kw = dict(y="choice", x="cost", chid="case", alt="alt")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not available"):
        sp.mprobit(long, correlation="exchangeable", **kw)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="scale"):
        sp.mprobit(long, base=1, scale=1, **kw)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="exactly once"):
        sp.mprobit(long.iloc[1:], **kw)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="varies within"):
        sp.mprobit(long, y="choice", case_vars="cost", chid="case", alt="alt")
    two = long[long.alt < 3]
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="sp.probit"):
        sp.mprobit(two, **kw)


def test_cmmprobit_through_sp_stata():
    long = _choices(n=300, J=3)
    direct = sp.mprobit(
        long, y="choice", x="cost", case_vars="inc", chid="case", alt="alt",
        base=3, correlation="independent", stddev="homoskedastic",
    )  # fmt: skip
    script = """
        cmset case alt
        cmmprobit choice cost, casevars(inc) basealternative(3) ///
            correlation(independent) stddev(homoskedastic) intpoints(500)
    """
    run = sp.stata(script, data=long)
    np.testing.assert_allclose(run.params, direct.params)
    table = sp.stata(script + "\nestat covariance", data=long)
    np.testing.assert_allclose(table.to_numpy(), [[2, 1], [1, 2]])
    effect = sp.stata(
        script + "\nmargins, dydx(cost) outcome(1) alternative(1)", data=long
    )
    # own effect of a cost: negative, and a number of the right size
    assert -0.5 < effect["dydx"] < 0 and 0 < effect["se"] < 0.05
    cross = sp.stata(
        script + "\nmargins, dydx(cost) outcome(1) alternative(2)", data=long
    )
    assert cross["dydx"] > 0
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="cmset"):
        sp.stata("cmmprobit choice cost, casevars(inc)", data=long)


# ---------------------------------------------------------- series regression
def _curve(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.uniform(0, 3, n), "z": rng.normal(size=n)})
    df["g"] = rng.integers(0, 40, n)
    df["y"] = np.sin(2 * df.x) + 0.5 * df.z + rng.normal(scale=0.3, size=n)
    return df


def test_cross_validation_is_leave_one_out():
    df = _curve(n=120)
    fit = sp.series("y ~ z", df, "x", orders=[1, 2, 3])
    table = fit.model_info["cv"].set_index("order")
    for order in (1, 2, 3):
        errors = []
        for i in range(len(df)):
            rest = df.drop(df.index[i])
            cols = [rest.x**j for j in range(1, order + 1)]
            W = np.column_stack([np.ones(len(rest)), rest.z] + cols)
            b = np.linalg.lstsq(W, rest.y.to_numpy(), rcond=None)[0]
            row = df.iloc[i]
            w = np.array([1.0, row.z] + [row.x**j for j in range(1, order + 1)])
            errors.append(row.y - w @ b)
        assert np.isclose(table.loc[order, "cv"], np.sum(np.square(errors)), rtol=1e-8)
    assert fit.model_info["order"] == int(table["cv"].idxmin())


def test_series_fit_is_ols_on_the_basis_and_recovers_the_curve():
    df = _curve()
    fit = sp.series("y ~ z", df, "x", order=5)
    for j in range(1, 6):
        df[f"x{j}"] = df.x**j
    ols = sp.regress("y ~ z + x1 + x2 + x3 + x4 + x5", data=df, vce="robust")
    assert np.isclose(fit.params["z"], ols.params["z"], rtol=1e-8)
    assert np.isclose(fit.std_errors["z"], ols.std_errors["z"], rtol=1e-6)
    assert np.isclose(
        fit.diagnostics["Residual SS"],
        float((np.asarray(ols.data_info["residuals"]) ** 2).sum()),
        rtol=1e-9,
    )
    chosen = sp.series("y ~ z", df, "x")
    curve = chosen.model_info["function"]
    inside = curve[(curve.x > 0.2) & (curve.x < 2.8)]
    # the function is evaluated with z at its mean
    truth = np.sin(2 * inside.x) + 0.5 * df.z.mean()
    assert np.abs(inside.fit - truth).max() < 0.2
    assert np.abs(inside.derivative - 2 * np.cos(2 * inside.x)).max() < 1.0
    covered = (inside.ci_lower <= truth) & (truth <= inside.ci_upper)
    assert covered.mean() > 0.7
    assert chosen.model_info["order_selected_by"] == "cross-validation"


def test_spline_basis_and_knots():
    df = _curve()
    fit = sp.series("y ~ 1", df, "x", basis="spline", order=3, degree=2)
    knots = fit.model_info["knots"]
    np.testing.assert_allclose(knots, np.quantile(df.x, [0.25, 0.5, 0.75]), rtol=1e-9)
    assert list(fit.params.index) == [
        "Intercept", "s(x):p1", "s(x):p2", "s(x):k1", "s(x):k2", "s(x):k3",
    ]  # fmt: skip
    # the same fit as least squares on the truncated powers
    W = np.column_stack(
        [np.ones(len(df)), df.x, df.x**2]
        + [np.where(df.x > k, (df.x - k) ** 2, 0.0) for k in knots]
    )
    rss = float(((df.y - W @ np.linalg.lstsq(W, df.y, rcond=None)[0]) ** 2).sum())
    assert np.isclose(fit.diagnostics["Residual SS"], rss, rtol=1e-9)
    uniform = sp.series("y ~ 1", df, "x", basis="spline", order=1, knots="uniform")
    assert np.isclose(uniform.model_info["knots"][0], (df.x.min() + df.x.max()) / 2)
    clustered = sp.series("y ~ 1", df, "x", order=3, cluster="g")
    assert clustered.model_info["n_clusters"] == 40


def test_series_refusals():
    df = _curve(n=100)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="must not be in"):
        sp.series("y ~ x", df, "x")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="basis"):
        sp.series("y ~ 1", df, "x", basis="fourier")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="at least 1"):
        sp.series("y ~ 1", df, "x", order=0)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not a column"):
        sp.series("y ~ 1", df, "nope")
    df["flat"] = 1.0
    with pytest.raises(sp.exceptions.DataInsufficient, match="does not vary"):
        sp.series("y ~ 1", df, "flat")


# --------------------------------------------------------- small translations
def test_estat_bootstrap_table():
    df = _curve(n=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = sp.stata(
            "set seed 3\nbootstrap, reps(120) bca: regress y z\nestat bootstrap, all",
            data=df,
        )
        default = sp.stata(
            "set seed 3\nbootstrap, reps(120): regress y z\nestat bootstrap", data=df
        )
    assert list(table["type"]) == ["N", "P", "BC", "BCa"] * 2
    assert list(default["type"]) == ["BC", "BC"]
    ols = sp.regress("y ~ z", data=df)
    row = table[(table.statistic == "z") & (table.type == "N")].iloc[0]
    assert np.isclose(row.observed, ols.params["z"])
    assert np.isclose(row.ci_upper - row.ci_lower, 2 * 1.959963984540054 * row.se)
    # every interval holds the estimate here, and they differ from each other
    z = table[table.statistic == "z"]
    assert ((z.ci_lower < z.observed) & (z.observed < z.ci_upper)).all()
    assert z.ci_lower.nunique() == 4
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="follow a bootstrap"):
        sp.stata("regress y z\nestat bootstrap", data=df)


def test_display_with_commas_and_e_sigma():
    df = _curve(n=60)
    df["t"] = np.arange(len(df))
    assert sp.stata("scalar a = 2\nscalar b = 3\ndisplay a, b", data=df) == "2 3"
    assert sp.stata("display 1,, 2", data=df) == "12"
    sigma = sp.stata("tsset t\nvar y z, lags(1/2)\nmatrix list e(Sigma)", data=df)
    assert list(sigma.columns) == ["y", "z"] and sigma.shape == (2, 2)
    np.testing.assert_allclose(sigma.to_numpy(), sigma.to_numpy().T)


def test_factor_crossing_with_one_main_effect():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "a": rng.integers(1, 5, n),
            "b": rng.integers(30, 34, n),
            "c": rng.integers(1, 4, n),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = df.x + 0.4 * (df.a == 2) * (df.b == 31) + rng.normal(size=n)
    fit = sp.stata("regress y x i.b i.a#i.b", data=df)
    # the indicators Stata keeps: levels 2 to 4 of a within each level of b
    df["cell"] = np.where(df.a > 1, df.a * 100 + df.b, 0)
    cells = sp.regress("y ~ x + C(b) + C(cell)", data=df)
    assert len(fit.params) == len(cells.params) == 17
    assert np.isclose(fit.params["x"], cells.params["x"], rtol=1e-9)
    assert np.isclose(
        fit.params["C(a)[T.2]:C(b)[31]"], cells.params["C(cell)[T.231]"], rtol=1e-8
    )
    joint = sp.stata("regress y x i.b i.a#i.b\ntestparm i.a#i.b", data=df)
    assert joint["df"] == (12, n - 17)
    # two products sharing the factor without a main effect overlap, and
    # the cells each program drops differ: still refused
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="crosses factors"):
        sp.stata("regress y x i.b i.c i.a#i.b i.a#i.c", data=df)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="crosses factors"):
        sp.stata("regress y x i.a#i.b", data=df)
