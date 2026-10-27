"""``sp.cox(tvc=, texp=)`` vs Stata ``stcox, tvc() texp()`` and R ``coxph(tt=)``.

References (regenerate in ``_fixtures`` with the data script first):

* ``cox_tvc_Stata.csv`` -- Stata 18 MP, ``_generate_cox_tvc_Stata.do``
  (``set type double``, ``import delimited ..., asdouble``, convergence
  tightened to ``nrtolerance(1e-13)`` so the reference is the maximiser);
* ``cox_tvc_R.json`` -- R survival 3.8-3, ``_generate_cox_tvc_R.R``
  (LGPL, used as a black box: documented interface and outputs only).

Data: ``_generate_cox_tvc_data.py``. ``cox_tvc_cont.csv`` has continuous
times (no ties); ``cox_tvc_ties.csv`` has integer times with 15 distinct
values over 600 rows, two strata, 40 clusters, probability weights and
delayed entry. The ten Stata specifications cover both tie rules, the
texp forms ``_t``, ``ln(_t)``, ``_t^2``, ``min(_t, 4)`` and ``_t > 5``, a
tvc variable absent from the main equation, robust, clustered, pweighted,
stratified and delayed-entry fits.

Tolerance 1e-6 relative (the parity budget). The observed worst case is
3.3e-7 on ``c3`` (a ``tvc`` coefficient of -0.006 under ``texp(_t^2)``);
our log likelihood is the larger there by 5.7e-10, so the gap is Stata's
stopping point, not ours. R's default convergence (``eps = 1e-9`` on the
log likelihood) leaves ~1e-8 relative on the coefficients.

Conventions: Stata ``vce(robust)`` is ``robust='hc0', ssc='stata'``
(N/(N-1), N = input rows, not episodes); ``vce(cluster)`` is
``ssc='stata'`` (G/(G-1)). R's robust variance (``cluster=id``) and
clustered variance carry no factor, i.e. ``ssc='r'``; the R and Stata
reference SEs differ by exactly sqrt(600/599) and sqrt(40/39).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

_FIX = Path(__file__).parent / "_fixtures"
STATA = pd.read_csv(_FIX / "cox_tvc_Stata.csv", skipinitialspace=True)
R = json.loads((_FIX / "cox_tvc_R.json").read_text(encoding="utf-8"))
RTOL = 1e-6


@pytest.fixture(scope="module")
def cont():
    return pd.read_csv(_FIX / "cox_tvc_cont.csv")


@pytest.fixture(scope="module")
def ties():
    return pd.read_csv(_FIX / "cox_tvc_ties.csv")


# spec -> (fixture, sp.cox keyword arguments of the Stata call)
SPECS = {
    "c1_breslow_t": ("cont", dict(x=["x1", "race"], tvc="race", ties="breslow")),
    "c2_efron_ln": ("cont", dict(x=["x1", "race"], tvc="race", texp="ln(_t)")),
    "c3_efron_t2_robust": (
        "cont",
        dict(x=["x1"], tvc=["race", "x1"], texp="_t^2", robust="hc0", ssc="stata"),
    ),
    "c4_breslow_min": (
        "cont",
        dict(x=["x1", "race"], tvc="race", texp="min(_t, 4)", ties="breslow"),
    ),
    "t1_efron_t": ("ties", dict(x=["x1", "race"], tvc="race")),
    "t2_breslow_ln_robust": (
        "ties",
        dict(
            x=["x1", "race"],
            tvc=["race", "x1"],
            texp="ln(_t)",
            ties="breslow",
            robust="hc0",
            ssc="stata",
        ),
    ),
    "t3_efron_strata_cluster": (
        "ties",
        dict(x=["x1", "race"], tvc="race", strata="grp", cluster="cl", ssc="stata"),
    ),
    "t5_efron_step": ("ties", dict(x=["x1", "race"], tvc="race", texp="_t > 5")),
    "t4_breslow_pw": (
        "ties",
        dict(
            x=["x1", "race"],
            tvc="race",
            ties="breslow",
            weights="w",
            weights_as="sampling",
            ssc="stata",
        ),
    ),
    "t6_efron_entry_ln": (
        "ties",
        dict(x=["x1", "race"], tvc="race", texp="ln(_t)", start="entry"),
    ),
}


def _fit(frames, spec, **override):
    which, kw = SPECS[spec]
    kw = {**kw, **override}
    return sp.cox(data=frames[which], duration="time", event="died", **kw)


@pytest.fixture(scope="module")
def frames(cont, ties):
    return {"cont": cont, "ties": ties}


def test_every_stata_spec_is_covered():
    assert set(SPECS) == set(STATA.spec)


@pytest.mark.parametrize("spec", sorted(SPECS))
def test_matches_stata_stcox_tvc(frames, spec):
    res = _fit(frames, spec)
    ref = STATA[STATA.spec == spec]
    # Stata names the equations main: and tvc:; sp.cox keeps the main
    # names bare and labels the tvc equation like Stata
    names = [n.replace("main:", "") for n in ref.name]
    assert list(res.params.index) == names
    np.testing.assert_allclose(res.params[names], ref.b, rtol=RTOL, atol=0)
    np.testing.assert_allclose(res.std_errors[names], ref.se, rtol=RTOL, atol=0)
    assert res.data_info["nobs"] == int(ref.N.iloc[0])
    if "weights" not in SPECS[spec][1]:
        # unweighted log partial likelihood: same number in both
        assert abs(res.diagnostics["Log-likelihood"] - ref.ll.iloc[0]) < 1e-6


R_SPECS = {
    "c1_breslow_t": {},
    "c2_efron_ln": {},
    "t1_efron_t": {},
    # R: coxph(cluster = id) / (cluster = cl), no small-sample factor
    "t2_breslow_ln_robust": dict(ssc="r"),
    "t3_efron_strata_cluster": dict(ssc="r"),
    "t6_efron_entry_ln": {},
}


@pytest.mark.parametrize("spec", sorted(R_SPECS))
def test_matches_r_coxph_tt(frames, spec):
    res = _fit(frames, spec, **R_SPECS[spec])
    ref = R[spec]
    np.testing.assert_allclose(res.params.to_numpy(), ref["b"], rtol=RTOL, atol=0)
    np.testing.assert_allclose(res.std_errors.to_numpy(), ref["se"], rtol=RTOL, atol=0)
    assert abs(res.diagnostics["Log-likelihood"] - ref["loglik"]) < 1e-6


def test_r_and_stata_robust_differ_by_their_documented_factors():
    """The two references are the same sandwich: N/(N-1) and G/(G-1) apart."""
    for spec, factor in (
        ("t2_breslow_ln_robust", 600 / 599),
        ("t3_efron_strata_cluster", 40 / 39),
    ):
        stata_se = STATA[STATA.spec == spec].se.to_numpy()
        np.testing.assert_allclose(
            stata_se, np.asarray(R[spec]["se"]) * np.sqrt(factor), rtol=1e-6
        )


def test_tvc_equals_a_hand_split_counting_process_fit(cont):
    """Splitting at the failure times by hand and fitting z*g(stop) with
    start= gives the same fit (the estimator is the exact likelihood)."""
    fails = np.unique(cont.time[cont.died == 1])
    rows = []
    for r in cont.itertuples():
        cuts = fails[fails < r.time]
        lo = np.concatenate([[0.0], cuts])
        hi = np.concatenate([cuts, [r.time]])
        for k, (a, b) in enumerate(zip(lo, hi)):
            last = k == len(hi) - 1
            rows.append((r.id, a, b, r.died if last else 0, r.x1, r.race))
    long = pd.DataFrame(rows, columns=["id", "t0", "t1", "d", "x1", "race"])
    long["race_t"] = long.race * np.log(long.t1)
    hand = sp.cox(
        data=long,
        duration="t1",
        event="d",
        start="t0",
        x=["x1", "race", "race_t"],
        cluster="id",
        ssc="r",
    )
    tvc = sp.cox(
        data=cont,
        duration="time",
        event="died",
        x=["x1", "race"],
        tvc="race",
        texp="ln(_t)",
        robust="hc0",
    )
    np.testing.assert_allclose(hand.params.to_numpy(), tvc.params.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(
        hand.std_errors.to_numpy(), tvc.std_errors.to_numpy(), rtol=1e-9
    )
    assert tvc.data_info["nobs"] == len(cont)
    assert tvc.data_info["n_episodes"] == len(long)


def test_callable_texp_equals_the_string_form(cont):
    kw = dict(data=cont, duration="time", event="died", x=["x1", "race"], tvc="race")
    a = sp.cox(**kw, texp="ln(_t)")
    b = sp.cox(**kw, texp=np.log)
    np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())
    np.testing.assert_array_equal(a.std_errors.to_numpy(), b.std_errors.to_numpy())
    assert a.model_info["tvc"] == ["race"] and a.model_info["texp"] == "ln(_t)"


def test_constant_texp_reproduces_the_proportional_coefficient(cont):
    """texp = 1 makes z*g(t) = z: the tvc coefficient is the ordinary one."""
    kw = dict(data=cont, duration="time", event="died")
    plain = sp.cox(**kw, x=["x1", "race"])
    tvc = sp.cox(**kw, x=["x1"], tvc="race", texp="1")
    np.testing.assert_allclose(
        tvc.params.to_numpy(), plain.params.to_numpy(), rtol=1e-9
    )
    np.testing.assert_allclose(
        tvc.std_errors.to_numpy(), plain.std_errors.to_numpy(), rtol=1e-8
    )
    assert list(tvc.params.index) == ["x1", "tvc:race"]


def test_texp_parser_follows_stata_precedence():
    from statspai.survival._cox_tvc import resolve_texp

    t = np.array([0.5, 1.0, 2.0, 7.0])
    cases = {
        "_t": t,
        "ln(_t)": np.log(t),
        "-_t^2": -(t**2),  # Stata: unary minus binds below ^
        "_t^-1": 1 / t,
        "2*_t + 1": 2 * t + 1,
        "(_t - 1)/2": (t - 1) / 2,
        "_t > 1": (t > 1).astype(float),
        "_t >= 1": (t >= 1).astype(float),
        "min(_t, 1.5)": np.minimum(t, 1.5),
        "exp(-_t/10)": np.exp(-t / 10),
        "sqrt(_t)*log10(_t)": np.sqrt(t) * np.log10(t),
        "1e-1*_t": 0.1 * t,
    }
    for text, want in cases.items():
        g, label = resolve_texp(text)
        np.testing.assert_array_equal(g(t), want, err_msg=text)
        assert label == text


@pytest.mark.parametrize(
    "bad",
    [
        "_t^2^2",
        "x1*_t",
        "ln(_t",
        "eval(_t)",
        "_t +",
        "",
        "__import__('os')",
        "ln(_t, 2)",
    ],
)
def test_texp_parser_refuses_what_it_cannot_read(bad):
    from statspai.survival._cox_tvc import resolve_texp

    with pytest.raises(MethodIncompatibility):
        resolve_texp(bad)


def test_texp_must_be_finite_at_failure_times(cont):
    df = cont.copy()
    df.loc[0, ["time", "died"]] = [0.0, 1]  # a failure at t = 0
    with pytest.raises(DataInsufficient, match="not finite"):
        sp.cox(
            data=df, duration="time", event="died", x=["x1"], tvc="race", texp="ln(_t)"
        )


def test_option_misuse_is_refused(cont):
    kw = dict(data=cont, duration="time", event="died", x=["x1"])
    with pytest.raises(MethodIncompatibility, match="texp= requires tvc="):
        sp.cox(**kw, texp="ln(_t)")
    with pytest.raises(MethodIncompatibility, match="not in the data"):
        sp.cox(**kw, tvc="nope")
    with pytest.raises(MethodIncompatibility, match="design="):
        sp.cox(**kw, tvc="race", design=object())
    fit = sp.cox(**kw, tvc="race")
    with pytest.raises(MethodIncompatibility, match="time-varying"):
        fit.ph_test()


def test_episode_limit_refuses_instead_of_exhausting_memory(cont, monkeypatch):
    from statspai.survival import _cox_tvc

    monkeypatch.setattr(_cox_tvc, "MAX_EPISODES", 1000)
    with pytest.raises(DataInsufficient, match="episodes"):
        sp.cox(data=cont, duration="time", event="died", x=["x1"], tvc="race")


def test_default_fit_without_tvc_is_unchanged(cont):
    """No tvc=: no extra keys, and the fit is the ordinary one."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = sp.cox(data=cont, duration="time", event="died", x=["x1", "race"])
    assert "tvc" not in fit.model_info and "n_episodes" not in fit.data_info
    assert fit.data_info["nobs"] == len(cont)
