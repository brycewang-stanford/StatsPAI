"""Saved fixed effects of ``reghdfe`` and the ``felsdvreg`` translation.

Stata reference: ``tests/reference_parity/_fixtures/akm_twoway_Stata.*``
(Stata 18, reghdfe 6.13.1, felsdvreg 1.8.0 / st0143) from
``_generate_akm_twoway_Stata.do`` on ``akm_twoway.csv``.

Tolerances. reghdfe recovers the effects iteratively: even at
``tolerance(1e-14)`` its saved effects reproduce its own fitted values only
to ~3e-8 and differ from the exact solution by ~1e-7, so the effects are
compared at 1e-6 (coefficients and constants, which are exact on both
sides, at 1e-12). felsdvreg compresses its output to float: its variables
are compared at 2.4e-7 (half a float ulp for values below 4).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).parent / "reference_parity" / "_fixtures"


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "akm_twoway.csv")


@pytest.fixture(scope="module")
def ref(data):
    s = pd.read_csv(FIX / "akm_twoway_Stata.csv")
    assert (s.worker.to_numpy() == data.worker.to_numpy()).all()
    return pd.concat([data, s.drop(columns=["worker", "firm", "year"])], axis=1)


@pytest.fixture(scope="module")
def stata():
    return json.loads((FIX / "akm_twoway_Stata.json").read_text(encoding="utf-8"))


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


# --------------------------------------------------------------------------
# sp.hdfe_ols(...).fixef()
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tag, formula, kwargs, cols",
    [
        ("reghdfe2", "lnw ~ x | worker + firm", {}, {"worker": "wfe2", "firm": "ffe2"}),
        # Three-way FE: the NumPy demeaning path (no Rust extension) stops at
        # its default tolerance with the SE 8.6e-9 relative from reghdfe's;
        # at tol=1e-12 both backends are within 4e-10, inside the 1e-9 check.
        (
            "reghdfe3",
            "lnw ~ x | worker + firm + year",
            {"cluster": "firm", "tol": 1e-12},
            {"worker": "wfe3", "firm": "ffe3", "year": "yfe3"},
        ),
        (
            "reghdfe_aw",
            "lnw ~ x | worker + firm",
            {"weights": "wt"},
            {"worker": "wfe4", "firm": "ffe4"},
        ),
    ],
)
def test_hdfe_ols_fixef_is_reghdfes_savefe(ref, stata, tag, formula, kwargs, cols):
    fit = _quiet(sp.hdfe_ols, formula, data=ref, **kwargs)
    assert fit.params["x"] == pytest.approx(stata[tag]["b"], rel=1e-12)
    assert fit.std_errors["x"] == pytest.approx(stata[tag]["se"], rel=1e-9)
    fe = fit.fixef(absorb_constant=False)
    smp = ref[ref[{"reghdfe2": "s2", "reghdfe3": "s3", "reghdfe_aw": "s4"}[tag]] == 1]
    for term, col in cols.items():
        np.testing.assert_allclose(smp[term].map(fe[term]), smp[col], atol=1e-6)
        assert fe[term].attrs["constant"] == pytest.approx(
            stata[tag]["cons"], abs=1e-12
        )


def test_fixef_reproduces_the_fitted_values(ref):
    fit = sp.hdfe_ols("lnw ~ x | worker + firm + year", data=ref)
    fe = fit.fixef()  # the first effect carries the constant
    smp = ref.loc[fit._hdfe_fe_state.sample_index]
    total = sum(smp[k].map(v) for k, v in fe.items()) + fit.params["x"] * smp.x
    np.testing.assert_allclose(total + fit.residuals, smp.lnw, atol=1e-10)


def test_fixef_of_an_interacted_term_and_refusal_on_iv(ref):
    d = ref.assign(half=(ref.year >= 2).astype(int))
    fit = sp.hdfe_ols("lnw ~ x | worker + firm^half", data=d)
    fe = fit.fixef()
    assert isinstance(fe["firm^half"].index, pd.MultiIndex)
    smp = d.loc[fit._hdfe_fe_state.sample_index]
    key = pd.MultiIndex.from_frame(smp[["firm", "half"]])
    total = (
        smp.worker.map(fe["worker"]).to_numpy()
        + fe["firm^half"].reindex(key).to_numpy()
        + fit.params["x"] * smp.x.to_numpy()
    )
    np.testing.assert_allclose(total + fit.residuals, smp.lnw, atol=1e-10)
    ivfit = _quiet(sp.hdfe_ols, "lnw ~ 1 | worker + firm | x ~ year", data=ref)
    with pytest.raises(MethodIncompatibility, match="no fixed-effect state"):
        ivfit.fixef()


# --------------------------------------------------------------------------
# sp.from_stata / sp.stata: reghdfe absorb(name=var), savefe
# --------------------------------------------------------------------------


def test_named_effects_are_carried_not_dropped(data, ref):
    out = sp.from_stata("reghdfe lnw x, absorb(wfe2=worker ffe2=firm)")
    assert out["ok"] and out["complete"]
    assert out["save_fe"] == {"wfe2": "worker", "ffe2": "firm"}
    df = data.copy()
    assert out["python_code"].startswith("sp.hdfe_ols(")
    exec(out["python_script"], {"sp": sp, "pd": pd, "df": df})
    smp = ref.s2 == 1
    np.testing.assert_allclose(df.loc[smp, "wfe2"], ref.loc[smp, "wfe2"], atol=1e-6)
    np.testing.assert_allclose(df.loc[smp, "ffe2"], ref.loc[smp, "ffe2"], atol=1e-6)


def test_stata_session_creates_the_saved_variables(data, ref):
    run = _quiet(
        sp.stata,
        "reghdfe lnw x, absorb(wfe2=worker ffe2=firm)\n"
        "reghdfe lnw x, absorb(worker firm, savefe)",
        data=data,
        return_data=True,
    )
    g, smp = run.data, ref.s2 == 1
    np.testing.assert_allclose(g.loc[smp, "wfe2"], ref.loc[smp, "wfe2"], atol=1e-6)
    np.testing.assert_allclose(g.loc[smp, "__hdfe1__"], ref.loc[smp, "sv1"], atol=1e-6)
    np.testing.assert_allclose(g.loc[smp, "__hdfe2__"], ref.loc[smp, "sv2"], atol=1e-6)
    # reghdfe: missing outside e(sample) (the 15 singletons here)
    assert g.loc[~smp, "wfe2"].isna().all() and (~smp).sum() == 15


def test_stata_session_refuses_an_existing_variable_like_stata(data):
    # Stata: "variable wfe already defined", r(110)
    with pytest.raises(MethodIncompatibility, match="already exists"):
        _quiet(
            sp.stata,
            "reghdfe lnw x, absorb(wfe=worker firm)\n"
            "reghdfe lnw x, absorb(wfe=worker firm)",
            data=data,
        )


@pytest.mark.parametrize(
    "line",
    [
        "reghdfe lnw x, absorb(fe=i.firm#c.x worker)",
        "ivreghdfe lnw (x = year), absorb(wfe=worker firm)",
        "reghdfe lnw x [pw=year], absorb(wfe=worker firm)",
        "ppmlhdfe y x, absorb(fe1=id year)",
        "ppmlhdfe y x, absorb(id year, savefe)",
    ],
)
def test_saved_effects_that_cannot_be_recovered_are_reported(line):
    out = sp.from_stata(line)
    assert out["ok"] and not out["complete"]
    assert "absorb(name=...) / savefe" in out["dropped"]
    assert "save_fe" not in out
    # the column names, not `name=var` or the suboption, reach the call
    assert "=" not in str(out["arguments"].get("absorb", "")).replace(" ", "")
    assert "savefe" not in str(out["arguments"])


# --------------------------------------------------------------------------
# felsdvreg -> sp.akm
# --------------------------------------------------------------------------

_FEL = (
    "felsdvreg lnw x, ivar(worker) jvar(firm) feff(ff) peff(pf) xb(xb) res(r) "
    "mover(m) mnum(mn) pobs(po) group(g)"
)


@pytest.mark.parametrize("normalize, pre", [(False, "fel"), (True, "fln")])
def test_felsdvreg_reproduces_stata(data, ref, stata, normalize, pre):
    line = _FEL + (" normalize" if normalize else "")
    out = sp.from_stata(line)
    assert out["ok"] and out["complete"] and out["tool"] == "akm"
    run = _quiet(sp.stata, line, data=data, return_data=True)
    g = run.data
    tag = "felsdvreg_norm" if normalize else "felsdvreg"
    assert run.output.params["x"] == pytest.approx(stata[tag]["b"], rel=1e-12)
    assert run.output.std_errors["x"] == pytest.approx(stata[tag]["se"], rel=1e-9)
    assert run.output.nobs == stata[tag]["N"]
    np.testing.assert_allclose(g.ff, ref[pre + "_f"], atol=2.4e-7)
    np.testing.assert_allclose(g.pf, ref[pre + "_p"], atol=2.4e-7)
    np.testing.assert_array_equal(g.g, ref[pre + "_g"])
    if not normalize:
        np.testing.assert_allclose(g.xb, ref.fel_xb, atol=2.4e-7)
        np.testing.assert_allclose(g.r, ref.fel_r, atol=2.4e-7)
        for mine, theirs in (("m", "fel_m"), ("mn", "fel_mn"), ("po", "fel_po")):
            np.testing.assert_array_equal(g[mine], ref[theirs])
    # the printed code runs and gives the same variables
    df = data.copy()
    scope = {"sp": sp, "pd": pd, "df": df}
    assert out["python_code"].startswith("sp.akm(")
    _quiet(exec, out["python_script"], scope)
    np.testing.assert_allclose(scope["df"].ff, ref[pre + "_f"], atol=2.4e-7)


def test_felsdvreg_options_it_does_not_translate_are_reported():
    out = sp.from_stata(_FEL + " robust")
    assert out["ok"] and not out["complete"] and "robust" in out["dropped"]
    out = sp.from_stata(_FEL + " cluster(firm)")
    assert not out["complete"] and "cluster" in out["dropped"]
    out = sp.from_stata(_FEL + " nocompress")
    assert out["complete"] and "nocompress" in out["ignored_display_options"]
    bad = sp.from_stata("felsdvreg lnw x, ivar(worker) jvar(firm) feff(ff)")
    assert not bad["ok"] and "peff()" in bad["error"]
