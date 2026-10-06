"""``sp.path_analysis(missing='fiml')`` against lavaan ``missing = "ml"``.

The data are ``sem_latent.csv`` with values removed: completely at random
in three columns, at random given another variable in three more
(``_fixtures/_generate_sem_fiml_lavaan.R`` writes ``sem_missing.csv`` and
the reference, lavaan 0.6-21). Four models: a saturated path model, a
two-factor CFA, a structural model with an exogenous covariate, a growth
curve; two to sixteen patterns of missing values.

Tolerance 1e-5 (relative, floor of one). As in the complete-data file the
bound is lavaan's optimiser, which stops at a relative change near 1e-6;
the saturated path model, where the solution is unique and the engine here
has a gradient below 1e-9, shows the largest gap (5.5e-6). Standard errors
on both sides come from the observed information.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "sem_fiml_lavaan_R.json").read_text(encoding="utf-8"))["cases"]
TOL = 1e-5
FIT_NAMES = {
    "chisq": "chisq", "df": "df", "pvalue": "pvalue",
    "baseline.chisq": "baseline_chisq", "baseline.df": "baseline_df",
    "cfi": "cfi", "tli": "tli", "rmsea": "rmsea", "srmr": "srmr",
    "logl": "logl", "aic": "aic", "bic": "bic", "npar": "npar",
}  # fmt: skip
CASES = sorted(REF)


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "sem_missing.csv")


def _close(ours, theirs):
    return abs(ours - theirs) <= TOL * max(1.0, abs(theirs))


def _fit(data, case):
    ref = REF[case]
    return sp.path_analysis(ref["model"], data, missing="fiml", growth=ref["growth"])


@pytest.mark.parametrize("case", CASES)
def test_parameter_table_matches_lavaan(data, case):
    ref = REF[case]
    fit = _fit(data, case)
    assert fit.n_obs == ref["nobs"] and fit.n_patterns == ref["npatterns"]
    table = fit.params.set_index(["lhs", "op", "rhs"])
    assert len(table) == len(ref["table"])
    for r in ref["table"]:
        rhs = r["rhs"].replace(" ", "") if r["op"] == ":=" else r["rhs"]
        row = table.loc[(r["lhs"], r["op"], rhs)]
        assert _close(row["est"], r["est"]), (case, r, row["est"])
        assert _close(row["se"], r["se"]), (case, r, row["se"])
        if r.get("std.all") is not None:
            assert _close(row["std_all"], r["std.all"]), (case, r)


@pytest.mark.parametrize("case", CASES)
def test_fit_measures_match_lavaan(data, case):
    fit = _fit(data, case)
    for theirs, ours in FIT_NAMES.items():
        value = REF[case]["fit"][theirs]
        if value is None:
            continue
        assert _close(fit.fit[ours], value), (case, theirs, fit.fit[ours], value)


@pytest.mark.parametrize("case", ["cfa", "growth"])
def test_factor_scores_match_lavpredict(data, case):
    fit = _fit(data, case)
    for name, values in REF[case]["factor_scores"].items():
        np.testing.assert_allclose(
            fit.factor_scores[name].to_numpy()[: len(values)], values, atol=TOL
        )
    assert len(fit.factor_scores) == fit.n_obs  # incomplete rows are scored too


def test_complete_data_gives_the_listwise_fit():
    """No missing value: the two likelihoods are the same function."""
    full = pd.read_csv(FIX / "sem_latent.csv")
    model = REF["cfa"]["model"]
    a = sp.path_analysis(model, full, meanstructure=True)
    b = sp.path_analysis(model, full, missing="fiml")
    assert b.n_patterns == 1
    pd.testing.assert_frame_equal(a.params, b.params)
    assert a.fit == b.fit


def test_fiml_uses_rows_listwise_drops(data):
    model = REF["cfa"]["model"]
    listwise = sp.path_analysis(model, data)
    fiml = sp.path_analysis(model, data, missing="fiml")
    assert listwise.n_obs < fiml.n_obs == len(data)
    assert listwise.n_dropped == len(data) - listwise.n_obs and fiml.n_dropped == 0
    # more rows, tighter loadings
    se_l = listwise.params.query("op == '=~' and se > 0")["se"].to_numpy()
    se_f = fiml.params.query("op == '=~' and se > 0")["se"].to_numpy()
    assert (se_f < se_l).all()
    assert "missing-data patterns" in fiml.summary()


def test_missing_at_random_biases_listwise_not_fiml():
    """Known truth. Three indicators with mean zero; the third is missing
    whenever the first is above 0.3. Among the complete rows the third has
    a negative mean. Full information recovers zero."""
    rng = np.random.default_rng(5)
    n = 20000
    f = rng.normal(size=n)
    df = pd.DataFrame({f"y{j}": f + 0.7 * rng.normal(size=n) for j in (1, 2, 3)})
    df.loc[df["y1"] > 0.3, "y3"] = np.nan
    model = "f =~ y1 + y2 + y3"
    listwise = sp.path_analysis(model, df, meanstructure=True)
    fiml = sp.path_analysis(model, df, missing="fiml")
    pick = "lhs == 'y3' and op == '~1'"
    biased = listwise.params.query(pick).iloc[0]
    recovered = fiml.params.query(pick).iloc[0]
    assert biased["est"] < -0.3 and abs(biased["est"]) > 10 * biased["se"]
    assert abs(recovered["est"]) < 3 * recovered["se"]
    load = fiml.params.query("lhs == 'f' and rhs == 'y3'").iloc[0]
    assert abs(load["est"] - 1.0) < 3 * load["se"]


def test_rows_missing_an_exogenous_value_are_dropped(data):
    holes = data.copy()
    holes.loc[:19, "x"] = np.nan
    fit = sp.path_analysis(REF["sem"]["model"], holes, missing="fiml")
    assert fit.n_obs == len(data) - 20 and fit.n_dropped == 20


def test_options(data):
    model = REF["cfa"]["model"]
    with pytest.raises(MethodIncompatibility, match="robust"):
        sp.path_analysis(model, data, missing="fiml", se="robust")
    with pytest.raises(MethodIncompatibility, match="missing must be"):
        sp.path_analysis(model, data, missing="pairwise")
    # lavaan's and Stata's spellings
    a = sp.path_analysis(model, data, missing="fiml")
    for alias in ("ML", "mlmv"):
        b = sp.path_analysis(model, data, missing=alias)
        assert b.fit["chisq"] == a.fit["chisq"]
