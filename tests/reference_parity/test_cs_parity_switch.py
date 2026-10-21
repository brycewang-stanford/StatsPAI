"""Reference parity: ``sp.callaway_santanna(parity=...)`` and ``sp.aggte``
against R ``did`` 2.5.1 and Stata ``csdid`` 1.81 on the same CSV bytes.

``parity='r'`` / ``'stata'`` is meant to make "StatsPAI = R = Stata" a
one-argument claim. This file checks that claim number by number on two
panels: the real ``mpdta`` (500 counties, 2003-2007) and the same panel with
35 never-treated counties recoded as always treated (20 first treated in
2003, the first period, and 15 in 2001, before it). Each case is a fit with
or without the covariate ``lpop``, with never-treated or not-yet-treated
comparisons, and with or without clustering on ``state``; one R case adds
``anticipation=1``. Nothing else is passed: the conventions come from
``parity=`` alone.

What is compared, per case: every ATT(g, t) and its standard error; the
pre-test Wald statistic and its p-value; the number of units kept (both
programs drop the always-treated units); the simple, group, calendar and
dynamic aggregations from ``sp.aggte`` (overall estimate and SE, and every
reported cell), and the fit's own simple ATT. Against Stata also the
confidence limits, which pins the critical value (normal, also when
clustered), and one ``wboot`` fit, whose ``estat`` aggregations are the
analytic ones.

Fixtures (references are run as black boxes, their source is not read):

* ``_fixtures/_generate_cs_parity_data.py`` writes ``cs_parity_mpdta.csv``
  and ``cs_parity_always.csv``;
* ``_fixtures/_generate_cs_parity_R.R`` writes ``cs_parity_R.json``
  (``att_gt`` / ``aggte`` with ``bstrap = FALSE``, so the SEs are the
  analytic ones and can be compared exactly);
* ``_fixtures/_generate_cs_parity_Stata.do`` writes ``cs_parity_Stata.csv``
  (``csdid`` defaults, ``estat pretrend / simple / group / calendar /
  event``; data read with ``set type double`` and ``asdouble``).

Tolerances. Without covariates both sides evaluate the same closed form:
``rtol = 1e-10`` (observed worst 1e-13 against R and Stata). With ``lpop``
the propensity score is a logit fitted to each program's own stopping
rule: ``rtol = 1e-8`` against R (observed 5.5e-10) and ``1e-7`` against
Stata (observed 1.2e-8 on clustered SEs; R and StatsPAI agree with each
other to 5e-10 there, so the gap is csdid's logit tolerance). R prints the
pre-test p-value rounded to five decimals, so that one is compared with
``atol = 5e-6``.

References
----------
- Callaway, B. and Sant'Anna, P.H.C. (2021). "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230.
  [@callaway2021difference]
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_DIR = pathlib.Path(__file__).parent / "_fixtures"
_R = json.loads((_DIR / "cs_parity_R.json").read_text(encoding="utf-8"))["cases"]
_STATA = pd.read_csv(_DIR / "cs_parity_Stata.csv")
_PANELS = {
    name: pd.read_csv(_DIR / f"{name}.csv")
    for name in ("cs_parity_mpdta", "cs_parity_always")
}
_N_PERIODS = 5


def _rtol_r(covariates: bool) -> float:
    return 1e-8 if covariates else 1e-10


def _rtol_stata(covariates: bool) -> float:
    return 1e-7 if covariates else 1e-10


def _fit(panel, *, covariates, notyet, cluster, anticipation=0, **kw):
    args = dict(
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        control_group="notyettreated" if notyet else "nevertreated",
        anticipation=anticipation,
        **kw,
    )
    if covariates:
        args["x"] = ["lpop"]
    if cluster:
        args["clustervars"] = "state"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(_PANELS[panel], **args)


def _close(actual, expected, rtol, what):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, what
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0, err_msg=what)


# ---------------------------------------------------------------------- R did


def _r_id(c):
    return (
        f"{c['panel'][10:]}-{'lpop' if c['covariates'] else 'nocov'}-"
        f"{c['control_group'][:5]}-{'cl' if c['cluster'] else 'unit'}"
        f"-ant{c['anticipation']}"
    )


@pytest.mark.parametrize("case", _R, ids=_r_id)
def test_parity_r_matches_r_did(case):
    rtol = _rtol_r(case["covariates"])
    fit = _fit(
        case["panel"],
        covariates=case["covariates"],
        notyet=case["control_group"] == "notyettreated",
        cluster=case["cluster"],
        anticipation=case["anticipation"],
        parity="r",
    )
    mi = fit.model_info
    assert mi["n_units"] == case["n"]
    assert mi["conventions"]["parity"] == "r"
    assert mi["conventions"]["base_period"] == "varying"

    ref = pd.DataFrame(
        {
            "group": case["group"],
            "time": case["time"],
            "att": case["att"],
            "se": case["se"],
        }
    )
    got = fit.detail.merge(ref, on=["group", "time"], suffixes=("", "_r"))
    assert len(got) == len(ref) == len(fit.detail)
    _close(got["att"], got["att_r"], rtol, "ATT(g,t)")
    _close(got["se"], got["se_r"], rtol, "SE of ATT(g,t)")

    pt = mi["pretrend_test"]
    _close([pt["statistic"]], np.ravel(case["W"]), rtol, "pre-test Wald")
    assert pt["pvalue_method"] == "chi2"
    assert pt["pvalue"] == pt["pvalue_chi2"]
    # R prints the p-value rounded to five decimals.
    assert abs(pt["pvalue"] - np.ravel(case["Wpval"])[0]) <= 5e-6

    simple = case["agg"]["simple"]
    _close([fit.estimate, fit.se], [simple["att"], simple["se"]], rtol, "fit ATT")
    for type_ in ("simple", "dynamic", "group", "calendar"):
        ref_agg = case["agg"][type_]
        agg = sp.aggte(fit, type=type_)
        assert agg.model_info["conventions"]["parity"] == "r"
        _close([agg.estimate, agg.se], [ref_agg["att"], ref_agg["se"]], rtol, type_)
        if type_ != "simple":
            label = agg.detail.columns[0]
            assert agg.detail[label].astype(float).tolist() == [
                float(v) for v in ref_agg["egt"]
            ]
            _close(agg.detail["att"], ref_agg["att_egt"], rtol, f"{type_} cells")
            _close(agg.detail["se"], ref_agg["se_egt"], rtol, f"{type_} cell SEs")


def test_r_fixture_drops_always_treated_units():
    """R's own record of the always-treated rule, used by the tests above."""
    by = {(_r_id(c)): c for c in _R}
    assert by["always-nocov-never-unit-ant0"]["n"] == 465
    assert by["always-nocov-never-unit-ant1"]["n"] == 445
    # jsonlite unboxes a single warning to a string
    assert "Dropped 35 units" in str(by["always-nocov-never-unit-ant0"]["warnings"])
    assert "Dropped 55 units" in str(by["always-nocov-never-unit-ant1"]["warnings"])


# ---------------------------------------------------------------- Stata csdid

_STATA_CASES = sorted(
    {
        tuple(int(v) if not isinstance(v, str) else v for v in k)
        for k in _STATA[
            ["panel", "covariates", "notyet", "cluster", "wboot"]
        ].itertuples(index=False, name=None)
    }
)
_OVERALL = {
    "simple": "ATT",
    "group": "GAverage",
    "calendar": "CAverage",
    "event": "Post_avg",
}
_AGG = {
    "simple": "simple",
    "group": "group",
    "calendar": "calendar",
    "event": "dynamic",
}


def _s_id(k):
    panel, cov, ny, cl, wb = k
    return (
        f"{panel[10:]}-{'lpop' if cov else 'nocov'}-{'notyet' if ny else 'never'}"
        f"-{'cl' if cl else 'unit'}{'-wboot' if wb else ''}"
    )


@pytest.mark.parametrize("key", _STATA_CASES, ids=_s_id)
def test_parity_stata_matches_csdid(key):
    panel, cov, ny, cl, wb = key
    rtol = _rtol_stata(bool(cov))
    rows = _STATA[
        (_STATA.panel == panel)
        & (_STATA.covariates == cov)
        & (_STATA.notyet == ny)
        & (_STATA.cluster == cl)
        & (_STATA.wboot == wb)
    ]
    extra = dict(bstrap=True, random_state=7) if wb else {}
    fit = _fit(
        panel,
        covariates=bool(cov),
        notyet=bool(ny),
        cluster=bool(cl),
        parity="stata",
        **extra,
    )
    mi = fit.model_info
    assert mi["n_units"] * _N_PERIODS == int(rows["n"].iloc[0])
    assert mi["conventions"]["notyet_cutoff"] == "cohort"

    if not wb:
        # wboot replaces csdid's ATT(g, t) SEs by bootstrap ones; the
        # point estimates and every aggregation below stay analytic.
        cells = rows[rows.block == "attgt"].copy()
        cells["group"] = cells["name"].str.extract(r"^g(\d+):")[0].astype(int)
        cells["time"] = cells["name"].str.extract(r"_(\d+)$")[0].astype(int)
        got = fit.detail.merge(cells, on=["group", "time"], suffixes=("", "_s"))
        assert len(got) == len(cells) == len(fit.detail)
        _close(got["att"], got["b"], rtol, "ATT(g,t)")
        _close(got["se"], got["se_s"], rtol, "SE of ATT(g,t)")
        pre = rows[rows.block == "pretrend"].iloc[0]
        pt = mi["pretrend_test"]
        _close([pt["statistic"]], [pre["b"]], rtol, "pre-test chi2")
        assert pt["df"] == pre["df"]
        _close([pt["pvalue"]], [pre["crit"]], rtol, "pre-test p (chi2)")

    simple = rows[(rows.block == "simple") & (rows.name == "ATT")].iloc[0]
    _close([fit.estimate, fit.se], [simple.b, simple.se], rtol, "fit ATT")
    for block, type_ in _AGG.items():
        ref = rows[rows.block == block]
        agg = sp.aggte(fit, type=type_)
        conv = agg.model_info["conventions"]
        assert conv["parity"] == "stata" and conv["se_method"] == "analytic"
        assert conv["share_variance"] is (type_ != "group")
        o = ref[ref.name == _OVERALL[block]].iloc[0]
        _close([agg.estimate, agg.se], [o.b, o.se], rtol, f"{type_} overall")
        _close([agg.ci[0], agg.ci[1]], [o.ll, o.ul], rtol, f"{type_} overall CI")
        assert o.crit == pytest.approx(1.959963984540054, rel=1e-9)
        if type_ != "simple":
            cells = ref[~ref.name.isin([_OVERALL[block], "Pre_avg"])]
            _close(agg.detail["att"], cells.b, rtol, f"{type_} cells")
            _close(agg.detail["se"], cells.se, rtol, f"{type_} cell SEs")
            _close(agg.detail["ci_lower"], cells.ll, rtol, f"{type_} cell CI")


def test_stata_wboot_estat_is_analytic():
    """csdid's estat after wboot reports the analytic SEs (fixture check)."""
    a = _STATA[
        (_STATA.panel == "cs_parity_mpdta")
        & (_STATA.covariates == 0)
        & (_STATA.notyet == 0)
        & (_STATA.cluster == 0)
    ]
    plain = a[(a.wboot == 0) & ~a.block.isin(["attgt", "pretrend"])]
    boot = a[(a.wboot == 1) & ~a.block.isin(["attgt", "pretrend"])]
    np.testing.assert_array_equal(plain.se.to_numpy(), boot.se.to_numpy())
    # while the ATT(g, t) SEs in e(V) are the bootstrap ones
    cells0 = a[(a.wboot == 0) & (a.block == "attgt")].se.to_numpy()
    cells1 = a[(a.wboot == 1) & (a.block == "attgt")].se.to_numpy()
    assert np.abs(cells1 / cells0 - 1).max() > 1e-3


def test_default_numbers_differ_from_both_only_where_documented():
    """Without parity=, post-period ATT(g, t) and every aggregation match R.

    The default keeps the universal base period, so the pre-period cells and
    the pre-test differ; the post-period cells and the aggregates over them
    are R's numbers.
    """
    case = next(c for c in _R if _r_id(c) == "mpdta-nocov-never-unit-ant0")
    fit = _fit("cs_parity_mpdta", covariates=False, notyet=False, cluster=False)
    ref = pd.DataFrame(
        {
            "group": case["group"],
            "time": case["time"],
            "att": case["att"],
            "se": case["se"],
        }
    )
    post = fit.detail[fit.detail.time >= fit.detail.group]
    got = post.merge(ref, on=["group", "time"], suffixes=("", "_r"))
    assert len(got) == 7
    _close(got["att"], got["att_r"], 1e-10, "post ATT(g,t)")
    _close(got["se"], got["se_r"], 1e-10, "post SE")
    for type_ in ("simple", "group", "calendar"):
        agg = sp.aggte(fit, type=type_)
        _close(
            [agg.estimate, agg.se],
            [case["agg"][type_]["att"], case["agg"][type_]["se"]],
            1e-10,
            type_,
        )
