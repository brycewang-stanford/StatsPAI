"""Contract: every export method works with its *defaults*, on ~50 estimators.

The 2026-10-19 audit found ``CausalResult.to_word(path)`` crashing with its
own default ``digits=None`` (``DataFrame.round(None)``) — for every causal
estimator, qreg and tobit included — because no test ever called it without
arguments. This suite fits a spread of registered estimators on simulated
data and calls each export surface the way a user first would: no options.

Each check asserts a true proposition, not just "did not raise":

* ``to_dict()`` is JSON-safe and its headline equals the result's own
  ``estimate`` (identity);
* ``to_excel(path)`` of a ``CausalResult`` stores the headline estimate as a
  number equal to ``round(estimate, 6)`` (the documented data-interchange
  precision);
* ``to_word(path)`` writes a readable ``.docx`` that holds a table;
* ``sp.regtable(result)`` carries the very coefficients the result holds, or
  refuses with the typed :class:`statspai.MethodIncompatibility` when the
  result has no coefficient vector (it used to raise a bare
  ``AttributeError``).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult, EconometricResults

pytest.importorskip("sklearn")


def _cross(n: int = 400, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    x1 = r.normal(size=n)
    x2 = r.normal(size=n)
    d = (r.random(n) < 1 / (1 + np.exp(-0.5 * x1))).astype(int)
    z = r.normal(size=n)
    dz = 0.8 * z + 0.5 * x1 + r.normal(size=n)
    y = 1 + 2 * d + x1 + 0.5 * x2 + r.normal(size=n)
    df = pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2, z=z, dz=dz))
    df["yiv"] = 1 + 1.5 * dz + x1 + r.normal(size=n)
    df["yb"] = (df.y > df.y.median()).astype(int)
    df["yc"] = np.clip(df.y - 1, 0, None)
    df["cnt"] = r.poisson(np.exp(0.3 + 0.3 * x1 + 0.2 * d))
    df["g"] = r.integers(0, 25, n)
    df["m"] = 0.5 * d + 0.3 * x1 + r.normal(size=n)
    df["ym"] = 1 + d + df.m + x1 + r.normal(size=n)
    df["run"] = r.uniform(-1, 1, n)
    df["yrd"] = 1 + 0.5 * df.run + 1.0 * (df.run >= 0) + r.normal(scale=0.3, size=n)
    df["dur"] = r.exponential(np.exp(-0.3 * x1))
    df["ev"] = (r.random(n) < 0.8).astype(int)
    df["sel"] = (0.5 + x2 + z + r.normal(size=n) > 0).astype(int)
    df["ys"] = np.where(df.sel == 1, df.y, np.nan)
    return df


def _panel(n_units: int = 60, T: int = 8, seed: int = 1) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    cohorts = r.choice([0, 4, 6], size=n_units, p=[0.4, 0.3, 0.3])
    a = r.normal(size=n_units)
    rows = []
    for i in range(n_units):
        for t in range(1, T + 1):
            g = cohorts[i]
            treat = int(g > 0 and t >= g)
            x = r.normal()
            y = a[i] + 0.2 * t + 2.0 * treat + 0.5 * x + r.normal(scale=0.5)
            rows.append(
                dict(
                    id=i,
                    t=t,
                    g=g,
                    treat=treat,
                    x=x,
                    y=y,
                    post=int(t >= 4),
                    tr=int(g == 4),
                )
            )
    return pd.DataFrame(rows)


def _synth_panel(seed: int = 2) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    f = r.normal(size=20).cumsum()
    rows = []
    for i in range(12):
        lam = r.normal(1, 0.3)
        for t in range(20):
            y = lam * f[t] + r.normal(scale=0.2) + (3.0 if (i == 0 and t >= 14) else 0)
            rows.append(dict(unit=i, time=t, y=y))
    return pd.DataFrame(rows)


C = _cross()
P = _panel()
S = _synth_panel()
X = ["x1", "x2"]

CASES = {
    "regress": lambda: sp.regress("y ~ d + x1", data=C),
    "regress_cluster": lambda: sp.regress("y ~ d + x1", data=C, cluster="g"),
    "logit": lambda: sp.logit("yb ~ d + x1", data=C),
    "probit": lambda: sp.probit("yb ~ d + x1", data=C),
    "poisson": lambda: sp.poisson("cnt ~ d + x1", data=C),
    "nbreg": lambda: sp.nbreg("cnt ~ d + x1", data=C),
    "glm": lambda: sp.glm("cnt ~ d + x1", data=C, family="poisson"),
    "tobit": lambda: sp.tobit(C, y="yc", x=["d", "x1"]),
    "qreg": lambda: sp.qreg(C, "y ~ d + x1"),
    "ivreg": lambda: sp.ivreg("yiv ~ x1 + (dz ~ z)", data=C),
    "heckman": lambda: sp.heckman(C, y="ys", x=["x1"], select="sel", z=["x1", "z"]),
    "feols": lambda: sp.feols("y ~ d + x1 | g", data=C),
    "feols_cluster": lambda: sp.feols("y ~ d + x1 | g", data=C, vcov={"CRV1": "g"}),
    "fepois": lambda: sp.fepois("cnt ~ d + x1 | g", data=C),
    "hdfe_ols": lambda: sp.hdfe_ols("y ~ d + x1 | g", data=C),
    "panel_fe": lambda: sp.panel(
        P, "y ~ treat + x", entity="id", time="t", method="fe"
    ),
    "did_2x2": lambda: sp.did(P[P.g != 6], y="y", treat="tr", time="post", id="id"),
    "callaway_santanna": lambda: sp.callaway_santanna(P, y="y", g="g", t="t", i="id"),
    "sun_abraham": lambda: sp.sun_abraham(P, y="y", g="g", t="t", i="id"),
    "did_imputation": lambda: sp.did_imputation(
        P, y="y", group="id", time="t", first_treat="g"
    ),
    "gardner_did": lambda: sp.gardner_did(
        P, y="y", group="id", time="t", first_treat="g"
    ),
    "etwfe": lambda: sp.etwfe(P, y="y", group="id", time="t", first_treat="g"),
    "wooldridge_did": lambda: sp.wooldridge_did(
        P, y="y", group="id", time="t", first_treat="g"
    ),
    "stacked_did": lambda: sp.stacked_did(
        P, y="y", group="id", time="t", first_treat="g", window=(-2, 2)
    ),
    "event_study": lambda: sp.event_study(
        P.assign(tt=P.g.replace(0, np.nan)),
        y="y",
        treat_time="tt",
        time="t",
        unit="id",
        window=(-2, 2),
    ),
    "xtabond": lambda: sp.xtabond(P, y="y", x=["x"], id="id", time="t"),
    "rdrobust": lambda: sp.rdrobust(C, y="yrd", x="run"),
    "rd_honest": lambda: sp.rd_honest(C, y="yrd", x="run"),
    "rddensity": lambda: sp.rddensity(C, x="run"),
    "rkd": lambda: sp.rkd(C, y="yrd", x="run"),
    "synth": lambda: sp.synth(
        S, outcome="y", unit="unit", time="time", treated_unit=0, treatment_time=14
    ),
    "sdid": lambda: sp.sdid(
        S, outcome="y", unit="unit", time="time", treated_unit=0, treatment_time=14
    ),
    "gsynth": lambda: sp.gsynth(
        S, outcome="y", unit="unit", time="time", treated_unit=0, treatment_time=14
    ),
    "dml": lambda: sp.dml(C, y="y", d="d", X=X),
    "ipw": lambda: sp.ipw(C, y="y", treat="d", covariates=X, n_bootstrap=20),
    "aipw": lambda: sp.aipw(C, y="y", treat="d", covariates=X),
    "tmle": lambda: sp.tmle(C, y="y", treat="d", covariates=X),
    "match": lambda: sp.match(C, y="y", treat="d", covariates=X),
    "ebalance": lambda: sp.ebalance(C, y="y", treat="d", covariates=X),
    "cbps": lambda: sp.cbps(C, y="y", treat="d", covariates=X, n_bootstrap=20),
    "metalearner": lambda: sp.metalearner(C, y="y", treat="d", covariates=X),
    "mediate": lambda: sp.mediate(
        C, y="ym", treat="d", mediator="m", covariates=["x1"], n_boot=50
    ),
    "bunching": lambda: sp.bunching(
        pd.DataFrame(
            {
                "z": np.r_[
                    np.random.default_rng(5).uniform(0, 10, 2000), np.full(150, 5.0)
                ]
            }
        ),
        running_var="z",
        threshold=5.0,
    ),
    "oaxaca": lambda: sp.oaxaca(C, y="y", group="d", x=["x1", "x2"]),
    "kaplan_meier": lambda: sp.kaplan_meier(C, duration="dur", event="ev"),
    "cox": lambda: sp.cox(data=C, duration="dur", event="ev", x=["x1", "d"]),
    "lpoly": lambda: sp.lpoly(C, y="y", x="x1"),
    "var": lambda: sp.var(
        pd.DataFrame(
            np.random.default_rng(3).normal(size=(120, 2)), columns=["a", "b"]
        ),
        lags=1,
    ),
    "arima": lambda: sp.arima(
        pd.Series(np.random.default_rng(4).normal(size=150)), order=(1, 0, 0)
    ),
}

_FITTED: dict = {}


def _result(name):
    if name not in _FITTED:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _FITTED[name] = CASES[name]()
    return _FITTED[name]


NAMES = sorted(CASES)


def test_case_count_is_representative():
    assert len(CASES) >= 45


@pytest.mark.parametrize("name", NAMES)
def test_text_surfaces_with_defaults(name, capsys):
    res = _result(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = res.summary()
        latex = res.to_latex()
        markdown = res.to_markdown()
        res.cite()
    if summary is not None:
        assert isinstance(summary, str) and summary.strip()
    assert isinstance(latex, str) and "\\begin{" in latex
    assert isinstance(markdown, str) and "|" in markdown


@pytest.mark.parametrize("name", NAMES)
def test_to_dict_is_json_safe_and_carries_the_estimate(name):
    res = _result(name)
    payload = res.to_dict()
    json.dumps(payload, allow_nan=True, default=str)
    if isinstance(res, CausalResult) and np.isfinite(res.estimate):
        assert payload["estimate"] == pytest.approx(float(res.estimate), rel=0, abs=0)


@pytest.mark.parametrize("name", NAMES)
def test_to_word_with_defaults_writes_a_table(name, tmp_path):
    docx = pytest.importorskip("docx")
    res = _result(name)
    path = tmp_path / f"{name}.docx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.to_word(str(path))
    doc = docx.Document(str(path))
    assert len(doc.tables) >= 1
    assert len(doc.tables[0].rows) >= 2


@pytest.mark.parametrize("name", NAMES)
def test_to_excel_with_defaults(name, tmp_path):
    pytest.importorskip("openpyxl")
    res = _result(name)
    path = tmp_path / f"{name}.xlsx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res.to_excel(str(path))
    sheets = pd.read_excel(path, sheet_name=None)
    assert sheets and any(len(f) for f in sheets.values())
    if isinstance(res, CausalResult) and "Estimates" in sheets:
        first = sheets["Estimates"].iloc[0]
        assert float(first["estimate"]) == pytest.approx(
            round(float(res.estimate), 6), abs=1e-12
        )


@pytest.mark.parametrize("name", NAMES)
def test_regtable_carries_the_coefficients_or_refuses_typed(name):
    res = _result(name)
    has_coefs = isinstance(getattr(res, "params", None), pd.Series) or (
        hasattr(res, "estimate") and hasattr(res, "se")
    )
    if not has_coefs:
        with pytest.raises(sp.MethodIncompatibility) as info:
            sp.regtable(res)
        assert info.value.recovery_hint
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = sp.regtable(res)
    coefs = table.to_dict()["models"][0]["coefficients"]
    if isinstance(res, (EconometricResults,)) or (
        isinstance(getattr(res, "params", None), pd.Series)
        and len(res.params) >= 2
        and not isinstance(res, CausalResult)
    ):
        for term, value in res.params.items():
            if np.isfinite(value):
                assert coefs[str(term)]["estimate"] == pytest.approx(
                    float(value), abs=0
                )
    elif isinstance(res, CausalResult) and not (
        isinstance(getattr(res, "params", None), pd.Series) and len(res.params) >= 2
    ):
        (only,) = coefs.values()
        assert only["estimate"] == pytest.approx(float(res.estimate), abs=0)


def test_causal_to_word_digits_default_matches_markdown_precision(tmp_path):
    """``digits=None`` resolves to the same ``auto`` precision as to_markdown."""
    docx = pytest.importorskip("docx")
    res = _result("ipw")
    path = tmp_path / "ipw.docx"
    res.to_word(str(path))
    cells = [c.text for c in docx.Document(str(path)).tables[0].rows[1].cells]
    md = res.to_markdown()
    estimate_cell = cells[1]
    assert estimate_cell and estimate_cell in md
    # And an explicit digits= still wins.
    res.to_word(str(path), digits=2)
    cells2 = [c.text for c in docx.Document(str(path)).tables[0].rows[1].cells]
    assert cells2[1] == f"{res.estimate:.2f}"
