"""GitHub issues #48, #49, #51, #52 (Ian Gow, 2026-11-19).

* #48 -- the regtable sample-size warning names the switch that silences
  it (``consistency_check=False``) and says ``notes=`` alone does not.
* #49 -- ``result.predict()`` takes polars / Arrow frames and returns what
  the equivalent pandas frame gives, in the same row order.
* #51 -- ``sp.regtable(se_type="none")`` (and ``statistic=""``) prints a
  coefficient-only table: no statistic rows, no "Standard errors in
  parentheses" footer, with stats and notes kept.
* #52 -- ``-inf`` as an event-time level read only by ``i()`` / ``C()``
  keeps the never-treated rows; the fit equals the one with ``-inf``
  recoded to a finite sentinel that is also a reference level.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

pl = pytest.importorskip("polars")


# ------------------------------------------------------------------ #48
def _two_samples():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.normal(size=50)})
    d["y"] = d["x"] + rng.normal(size=50)
    return sp.regress("y ~ x", data=d), sp.regress("y ~ x", data=d.iloc[:30])


def test_sample_size_warning_names_the_switch_and_notes_do_not_silence():
    full, sub = _two_samples()
    with pytest.warns(UserWarning, match="Sample sizes differ") as rec:
        out = sp.regtable([full, sub], notes=["Column 2 uses the subsample."])
    msg = " ".join(str(w.message) for w in rec)
    assert "consistency_check=False" in msg
    assert "notes= alone does not" in msg
    assert "Column 2 uses the subsample." in str(out)


def test_consistency_check_false_silences_only_the_sample_size_warning():
    full, sub = _two_samples()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sp.regtable([full, sub], notes=["x"], consistency_check=False)


# ------------------------------------------------------------------ #49
def test_predict_accepts_polars_like_pandas_issue_example():
    train = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [1.1, 3.2, 4.9, 7.1]})
    new = pl.DataFrame({"x": [3.5, -1.0]})
    fit = sp.regress("y ~ x", data=train)
    expected = fit.predict(new.to_pandas())
    # OLS line through the four points: b = 1.97, a = 1.12
    np.testing.assert_allclose(expected, [1.12 + 1.97 * 3.5, 1.12 - 1.97], rtol=1e-12)
    for got in (fit.predict(new), fit.predict(data=new), fit.predict(newdata=new)):
        assert isinstance(got, np.ndarray)
        np.testing.assert_array_equal(got, expected)
    lazy = fit.predict(new.lazy())
    np.testing.assert_array_equal(lazy, expected)
    band = fit.predict(new, what="prediction")
    pd.testing.assert_frame_equal(band, fit.predict(new.to_pandas(), what="prediction"))


@pytest.fixture
def binary_count_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"x": rng.normal(size=200), "z": rng.normal(size=200)})
    d["y"] = (d["x"] + rng.normal(size=200) > 0).astype(int)
    d["c"] = rng.poisson(np.exp(0.3 + 0.2 * d["x"]))
    return d


@pytest.mark.parametrize(
    "fit_fn",
    [
        lambda d: sp.logit("y ~ x", data=d),
        lambda d: sp.probit("y ~ x", data=d),
        lambda d: sp.glm("y ~ x", data=d, family="binomial"),
        lambda d: sp.poisson("c ~ x", data=d),
        lambda d: sp.nbreg("c ~ x", data=d),
        lambda d: sp.ivreg("c ~ (x ~ z)", data=d),
        lambda d: sp.feols("c ~ x", data=d),
    ],
    ids=["logit", "probit", "glm", "poisson", "nbreg", "ivreg", "feols"],
)
def test_predict_polars_equals_pandas_across_estimators(fit_fn, binary_count_frame):
    d = binary_count_frame
    fit = fit_fn(d)
    new = d.iloc[[5, 0, 17]]  # out of order: row order must be kept
    np.testing.assert_array_equal(
        np.asarray(fit.predict(pl.from_pandas(new))), np.asarray(fit.predict(new))
    )


def test_predict_pandas_argument_passes_by_identity(monkeypatch):
    from statspai.core.results import EconometricResults

    seen = {}
    original = EconometricResults.predict.__statspai_frame_input__

    def spy(self, data=None, **kw):
        seen["data"] = data
        return original(self, data, **kw)

    fit = sp.regress("y ~ x", data=pd.DataFrame({"x": [0.0, 1, 2], "y": [1.0, 2, 4]}))
    new = pd.DataFrame({"x": [5.0]})
    monkeypatch.setattr(
        EconometricResults, "predict", sp._frame_input.accepts_frames(spy)
    )
    fit.predict(new)
    assert seen["data"] is new


def test_logit_result_with_polars_predict_still_pickles(binary_count_frame):
    fit = sp.logit("y ~ x", data=binary_count_frame)
    back = pickle.loads(pickle.dumps(fit))
    new = binary_count_frame.head(4)
    np.testing.assert_array_equal(back.predict(pl.from_pandas(new)), fit.predict(new))


# ------------------------------------------------------------------ #51
@pytest.fixture
def small_fit():
    data = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 2.0, 1.0, 5.0]})
    return sp.regress("y ~ x", data=data)


def _body_lines(text: str):
    return [ln for ln in text.splitlines() if ln.strip()]


@pytest.mark.parametrize("kw", [{"se_type": "none"}, {"statistic": ""}])
def test_coefficient_only_text_table(small_fit, kw):
    out = str(
        sp.regtable(small_fit, stats=["R2"], stars=False, notes=["My note."], **kw)
    )
    lines = out.splitlines()
    assert "parenthes" not in out
    # coefficient rows sit directly on top of each other (no blank line,
    # no statistic line); intercept -0.1, slope 1.4 from the 4-point fit
    i = next(k for k, ln in enumerate(lines) if ln.startswith("Intercept"))
    assert lines[i].split()[-1] == "-0.100"
    assert lines[i + 1].startswith("x") and lines[i + 1].split()[-1] == "1.400"
    assert any(ln.startswith("R²") and ln.split()[-1] == "0.700" for ln in lines)
    assert "My note." in out


@pytest.mark.parametrize("output", ["html", "latex", "markdown"])
def test_coefficient_only_other_markups(small_fit, output):
    full = str(sp.regtable(small_fit, stats=["R2"], stars=False, output=output))
    bare = str(
        sp.regtable(small_fit, stats=["R2"], stars=False, se_type="none", output=output)
    )
    assert "parenthes" in full and "parenthes" not in bare
    se_x = f"({float(small_fit.std_errors['x']):.3f})"
    assert se_x in full and se_x not in bare
    assert "1.400" in bare and "0.700" in bare


def test_coefficient_only_dataframe_and_excel(small_fit, tmp_path):
    df = sp.regtable(small_fit, se_type="none").to_dataframe()
    assert list(df.index[:2]) == ["Intercept", "x"]
    assert "" not in list(df.index[:3])
    path = tmp_path / "t.xlsx"
    sp.regtable(small_fit, se_type="none", output="excel", filename=str(path))
    openpyxl = pytest.importorskip("openpyxl")
    rows = [
        [c.value for c in r] for r in openpyxl.load_workbook(path).active.iter_rows()
    ]
    assert rows[1][0] == "Intercept" and rows[2][0] == "x"
    assert not any("parenthes" in str(c) for r in rows for c in r if c)


def test_coefficient_only_transposed(small_fit):
    out = str(sp.regtable(small_fit, small_fit, se_type="none", transpose=True))
    assert "parenthes" not in out
    model_rows = [ln for ln in out.splitlines() if ln.startswith("(")]
    assert len(model_rows) == 2


def test_se_type_is_validated_and_case_insensitive(small_fit):
    with pytest.raises(MethodIncompatibility, match="se_type"):
        sp.regtable(small_fit, se_type="stderr")
    assert str(sp.regtable(small_fit, se_type="SE")) == str(sp.regtable(small_fit))


def test_modelsummary_se_type_none_is_coefficient_only(small_fit):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = str(sp.modelsummary(small_fit, se_type="none"))
    assert "parenthes" not in out
    assert f"({float(small_fit.std_errors['x']):.3f})" not in out


# ------------------------------------------------------------------ #52
def test_feols_inf_event_time_level_issue_example():
    rng = np.random.default_rng(42)
    unit = np.repeat(np.arange(8), 5)
    year = np.tile(np.arange(5), 8)
    adoption = np.repeat([2, 2, 3, 3, np.inf, np.inf, np.inf, np.inf], 5)
    t = year - adoption
    data = pl.DataFrame(
        {
            "unit": unit,
            "year": year,
            "t": t,
            "y": unit + year + 2 * (t >= 0) + rng.normal(size=len(t)),
        }
    )
    finite = data.with_columns(pl.col("t").replace(-np.inf, -1000).cast(pl.Int64))
    ref = sp.feols("y ~ i(t, ref=[-1, -1000]) | unit + year", data=finite, vcov="iid")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.feols("y ~ i(t, ref=-1) | unit + year", data=data, vcov="iid")
    # The -inf indicator is collinear with the unit effects and drops; the
    # never-treated rows stay in the sample as comparisons.
    assert fit.nobs == ref.nobs == 40
    np.testing.assert_allclose(fit.params.values, ref.params.values, atol=1e-12)
    np.testing.assert_allclose(fit.std_errors.values, ref.std_errors.values, atol=1e-12)
