"""Reference parity: ``parity='r' | 'stata'`` off a balanced panel.

``test_cs_parity_switch.py`` checks the switch on balanced panels. This file
does the same on the two designs where R ``did`` 2.5.1 and Stata ``csdid``
1.81 part ways with each other, on the same CSV bytes:

* a repeated cross-section (``cs_parity_rcs.csv``: ``mpdta`` county-years
  sampled with a year-specific probability, one id per row), run as
  ``att_gt(panel = FALSE)`` and ``csdid`` without ``ivar()``;
* an unbalanced panel (``cs_parity_unbal.csv``: ``mpdta`` with 12 percent
  of county-years deleted), run as ``att_gt`` with R's default (incomplete
  units dropped), as ``att_gt(allow_unbalanced_panel = TRUE)``, and as
  ``csdid, ivar()`` (each cell uses the units seen in both of its periods).

Each case is a fit with or without ``lpop``, never- or not-yet-treated
comparisons, unit or state clusters, unweighted and with the sampling
weight ``w`` (R ``weightsname = "w"``, csdid ``[iw=w]``, which its help file
says is treated as a pweight; csdid refuses ``pw`` and ``aw``). The weighted
cases also run on the balanced ``mpdta`` with each county's ``w``. Compared
per case: every ATT(g, t) and
SE, the pre-test, the fit's own simple ATT, and the simple / group /
calendar / event aggregations (overall estimate and SE, every cell; against
Stata also the confidence limits and the pre-period average).

What the comparison established (and ``parity=`` now sets):

* both programs use the "reference" sign of the doubly robust estimator's
  pre-period regression effect (``variance_convention='reference'``); the
  StatsPAI default ``'delta'`` moves the SEs with covariates by up to 15%;
* csdid weights every ATT(g, t) by its own treated count in every
  aggregate (rows of the cohort in the cell's two periods on a repeated
  cross-section, units of the cohort seen in both periods on an unbalanced
  panel; ``N_trt`` of ``e(gtt)``), and carries the estimation of those
  counts into the SEs; its group average weights cohorts by their mean
  post-treatment count and holds those weights fixed. R weights by cohort
  shares throughout. Before this, ``parity='stata'`` matched csdid's simple
  and group aggregates on repeated cross-sections only and warned
  elsewhere: there the overall aggregates were off by up to 5% and single
  calendar or event-time cells by up to 20%;
* with sampling weights csdid weights each cell by the weight mass of
  those treated observations, not by their head count (``N_trt`` of
  ``e(gtt)`` stays the head count), and the group average by the mean
  weight mass of a cohort's post-treatment cells. Head counts missed csdid's
  weighted aggregates by up to 19% (calendar cells on the unbalanced panel)
  and 2.6% overall; ``sp.aggte`` warned instead. R weights by weighted
  cohort shares, which StatsPAI already did;
* R's ``allow_unbalanced_panel = TRUE`` is an option R exposes, so
  ``parity='r'`` accepts ``allow_unbalanced_panel=True`` and reproduces it.

Fixtures (references run as black boxes, their source not read):
``_fixtures/_generate_cs_parity_rc_unbal_data.py`` (data),
``_generate_cs_parity_rc_unbal_R.R`` (``cs_parity_rc_unbal_R.json``,
``bstrap = FALSE`` so the SEs are analytic), and
``_generate_cs_parity_rc_unbal_Stata.do`` (``cs_parity_rc_unbal_Stata.csv``).

Tolerances. Without covariates every number is a closed form: ``rtol =
1e-10`` (observed worst 1.3e-12, weighted, against R). With ``lpop`` the
propensity score is a logit fitted to each program's stopping rule: ``1e-7`` against R (observed
3.9e-8, clustered SEs on the repeated cross-section) and ``1e-6`` against
Stata (observed 3.0e-7 on the same SEs, where csdid's ATT(g, t) agree with
ours to 2e-14 and R and Stata differ from each other by 3.3e-7). R prints
the pre-test p-value rounded to five decimals: ``atol = 5e-6``.

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
_R = json.loads((_DIR / "cs_parity_rc_unbal_R.json").read_text(encoding="utf-8"))[
    "cases"
]
_STATA = pd.read_csv(_DIR / "cs_parity_rc_unbal_Stata.csv")
_DATA = {
    "rcs": pd.read_csv(_DIR / "cs_parity_rcs.csv"),
    "unbal": pd.read_csv(_DIR / "cs_parity_unbal.csv"),
}
# The balanced panel of the weighted cases: mpdta with each county's w from
# the unbalanced panel, as the R script and the do-file build it.
_DATA["bal"] = pd.read_csv(_DIR / "cs_parity_mpdta.csv").merge(
    _DATA["unbal"][["countyreal", "w"]].drop_duplicates(),
    on="countyreal",
    validate="many_to_one",
)


def _rtol(covariates: bool, program: str) -> float:
    if not covariates:
        return 1e-10
    return 1e-7 if program == "r" else 1e-6


def _fit(design, *, covariates, notyet, cluster, weighted=False, **kw):
    rcs = design == "rcs"
    args = dict(
        y="lemp",
        g="first_treat",
        t="year",
        i="rid" if rcs else "countyreal",
        panel=not rcs,
        control_group="notyettreated" if notyet else "nevertreated",
        **kw,
    )
    if covariates:
        args["x"] = ["lpop"]
    if cluster:
        args["clustervars"] = "state"
    if weighted:
        args["weights"] = "w"
    with warnings.catch_warnings():
        # the unbalanced-panel and always-informative notes
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(_DATA[design], **args)


def _aggte(fit, type_):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # parity aggregation warns no more
        return sp.aggte(fit, type=type_)


def _close(actual, expected, rtol, what, atol=0.0):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, what
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=what)


def _close_ci(actual, expected, se, rtol, what):
    """Confidence limits: the SE tolerance carried through z * se.

    A limit near zero is a difference of two numbers, so a relative error in
    the SE shows up magnified in it (1.5e-6 at an upper limit of -0.006 from
    a 2e-7 SE difference); the absolute slack is rtol times the half-width.
    """
    halfwidth = 1.959963984540054 * np.asarray(se, dtype=float)
    _close(actual, expected, rtol, what, atol=rtol * float(np.max(halfwidth)))


# ---------------------------------------------------------------------- R did


def _r_id(c):
    return (
        f"{c['design']}-{'lpop' if c['covariates'] else 'nocov'}-"
        f"{c['control_group'][:5]}-{'cl' if c['cluster'] else 'unit'}"
        f"{'-w' if c['weighted'] else ''}"
    )


@pytest.mark.parametrize("case", _R, ids=_r_id)
def test_parity_r_matches_r_did(case):
    rtol = _rtol(case["covariates"], "r")
    design = case["design"]
    extra = dict(allow_unbalanced_panel=True) if design == "unbal_allow" else {}
    fit = _fit(
        design if design in ("rcs", "bal") else "unbal",
        covariates=case["covariates"],
        notyet=case["control_group"] == "notyettreated",
        cluster=case["cluster"],
        weighted=case["weighted"],
        parity="r",
        **extra,
    )
    mi = fit.model_info
    conv = mi["conventions"]
    assert mi["n_units"] == case["n"]
    assert conv["parity"] == "r"
    if design == "unbal_full":
        assert conv["balance"] == "full"
    if design == "unbal_allow":
        assert conv["balance"] == "none"
    if design not in ("unbal_full", "bal"):
        assert conv["variance_convention"] == "reference"
    assert mi["weighted"] is case["weighted"]

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
    assert pt["pvalue"] == pt["pvalue_chi2"]
    assert abs(pt["pvalue"] - np.ravel(case["Wpval"])[0]) <= 5e-6

    simple = case["agg"]["simple"]
    _close([fit.estimate, fit.se], [simple["att"], simple["se"]], rtol, "fit ATT")
    for type_ in ("simple", "dynamic", "group", "calendar"):
        ref_agg = case["agg"][type_]
        agg = _aggte(fit, type_)
        assert agg.model_info["conventions"]["agg_weights"] == "did"
        _close([agg.estimate, agg.se], [ref_agg["att"], ref_agg["se"]], rtol, type_)
        if type_ != "simple":
            label = agg.detail.columns[0]
            assert agg.detail[label].astype(float).tolist() == [
                float(v) for v in ref_agg["egt"]
            ]
            _close(agg.detail["att"], ref_agg["att_egt"], rtol, f"{type_} cells")
            _close(agg.detail["se"], ref_agg["se_egt"], rtol, f"{type_} cell SEs")


def test_r_fixture_unbalanced_rules():
    """R's own record: n under each rule, and the drop warning."""
    by = {_r_id(c): c for c in _R}
    full = by["unbal_full-nocov-never-unit"]
    allow = by["unbal_allow-nocov-never-unit"]
    complete = (_DATA["unbal"].groupby("countyreal").size() == 5).sum()
    assert full["n"] == complete == 256
    assert allow["n"] == _DATA["unbal"]["countyreal"].nunique() == 500
    assert by["rcs-nocov-never-unit"]["n"] == len(_DATA["rcs"])


# ---------------------------------------------------------------- Stata csdid

_KEYS = ["panel", "covariates", "notyet", "cluster", "weighted"]
_STATA_CASES = sorted(
    {
        tuple(int(v) if not isinstance(v, str) else v for v in k)
        for k in _STATA[_KEYS].itertuples(index=False, name=None)
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


def _design(panel):
    return {"rcs": "rcs", "unbal": "unbal", "mpdta": "bal"}[panel[10:]]


def _s_id(k):
    panel, cov, ny, cl, wt = k
    return (
        f"{_design(panel)}-{'lpop' if cov else 'nocov'}-"
        f"{'notyet' if ny else 'never'}-{'cl' if cl else 'unit'}"
        f"{'-w' if wt else ''}"
    )


@pytest.mark.parametrize("key", _STATA_CASES, ids=_s_id)
def test_parity_stata_matches_csdid(key):
    panel, cov, ny, cl, wt = key
    design = _design(panel)
    rtol = _rtol(bool(cov), "stata")
    rows = _STATA[
        (_STATA.panel == panel)
        & (_STATA.covariates == cov)
        & (_STATA.notyet == ny)
        & (_STATA.cluster == cl)
        & (_STATA.weighted == wt)
    ]
    fit = _fit(
        design,
        covariates=bool(cov),
        notyet=bool(ny),
        cluster=bool(cl),
        weighted=bool(wt),
        parity="stata",
    )
    mi = fit.model_info
    assert mi["weighted"] is bool(wt)
    assert mi["conventions"]["notyet_cutoff"] == "cohort"
    if design == "rcs":
        assert mi["n_obs"] == int(rows["n"].iloc[0])
        assert mi["conventions"]["variance_convention"] == "reference"
    else:
        assert mi["conventions"]["balance"] == "pair"

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
        agg = _aggte(fit, type_)
        conv = agg.model_info["conventions"]
        # On a balanced panel every cell's (weighted) count is its cohort's,
        # so the cohort-share path is csdid's rule.
        assert conv["parity"] == "stata"
        assert conv["agg_weights"] == ("did" if design == "bal" else "csdid")
        assert conv["share_variance"] is (type_ != "group")
        o = ref[ref.name == _OVERALL[block]].iloc[0]
        _close([agg.estimate, agg.se], [o.b, o.se], rtol, f"{type_} overall")
        _close_ci(
            [agg.ci[0], agg.ci[1]], [o.ll, o.ul], [o.se], rtol, f"{type_} overall CI"
        )
        assert o.crit == pytest.approx(1.959963984540054, rel=1e-9)
        if type_ != "simple":
            cells = ref[~ref.name.isin([_OVERALL[block], "Pre_avg"])]
            _close(agg.detail["att"], cells.b, rtol, f"{type_} cells")
            _close(agg.detail["se"], cells.se, rtol, f"{type_} cell SEs")
            _close_ci(
                agg.detail["ci_lower"], cells.ll, cells.se, rtol, f"{type_} cell CI"
            )
        if type_ == "dynamic":
            pre_avg = ref[ref.name == "Pre_avg"].iloc[0]
            pre_cells = agg.detail.loc[agg.detail["relative_time"] < 0, "att"]
            _close([pre_cells.mean()], [pre_avg.b], rtol, "Pre_avg")


@pytest.mark.parametrize("design", ["rcs", "unbal"])
def test_csdid_cell_counts_are_e_gtt_n_trt(design):
    """The weights are csdid's N_trt (e(gtt)); fixture-free identity check.

    On the repeated cross-section a cell counts the cohort's rows in its two
    periods; on the panel, the cohort's units seen in both (csdid's N_trt
    counts their two rows). Values copied from ``matlist e(gtt)`` of the
    first case of the do-file (no covariates, never-treated, unit clusters).
    """
    n_trt = {
        "rcs": [26, 21, 21, 19, 60, 61, 57, 50, 178, 200, 184, 175],
        "unbal": [34, 28, 26, 30, 62, 68, 64, 68, 200, 184, 192, 198],
    }[design]
    from statspai.did.aggte import _csdid_cell_indicators

    fit = _fit(design, covariates=False, notyet=False, cluster=False, parity="stata")
    counts = _csdid_cell_indicators(fit.detail, fit.model_info).sum(axis=0)
    assert counts.dtype == np.int64
    factor = 1 if design == "rcs" else 2
    assert (counts * factor).astype(int).tolist() == n_trt


def test_balanced_panel_keeps_cohort_share_path():
    """On a balanced panel every count is the cohort size: no csdid branch."""
    d = pd.read_csv(_DIR / "cs_parity_mpdta.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.callaway_santanna(
            d, y="lemp", g="first_treat", t="year", i="countyreal", parity="stata"
        )
    assert "_unit_observed" not in fit.model_info
    assert (
        sp.aggte(fit, type="calendar").model_info["conventions"]["agg_weights"] == "did"
    )


@pytest.mark.parametrize("design", ["rcs", "unbal"])
def test_csdid_weighted_counts_are_weight_mass(design):
    """csdid's weighted cell weight is the weight mass of N_trt's rows.

    Identity checks, no fixture numbers: a constant weight leaves every
    aggregate and SE of the unweighted fit unchanged (the cell weights are
    shares, so the scale cancels), and with ``w`` the overall simple ATT is
    the mass-weighted mean of the post-treatment ATT(g, t).
    """
    from statspai.did.aggte import _csdid_cell_indicators

    plain = _fit(design, covariates=False, notyet=False, cluster=False, parity="stata")
    data = _DATA[design]
    _DATA[design] = data.assign(w=2.5)
    try:
        const = _fit(
            design,
            covariates=False,
            notyet=False,
            cluster=False,
            weighted=True,
            parity="stata",
        )
    finally:
        _DATA[design] = data
    for type_ in ("simple", "group", "calendar", "dynamic"):
        a, b = _aggte(plain, type_), _aggte(const, type_)
        _close([b.estimate, b.se], [a.estimate, a.se], 1e-12, type_)

    fit = _fit(
        design,
        covariates=False,
        notyet=False,
        cluster=False,
        weighted=True,
        parity="stata",
    )
    ind = _csdid_cell_indicators(fit.detail, fit.model_info)
    mass = (ind * np.asarray(fit.model_info["_unit_weights"])[:, None]).sum(axis=0)
    post = fit.detail["relative_time"].to_numpy() >= 0
    expected = (
        np.sum(mass[post] * fit.detail["att"].to_numpy()[post]) / mass[post].sum()
    )
    _close([_aggte(fit, "simple").estimate], [expected], 1e-13, "mass-weighted mean")
