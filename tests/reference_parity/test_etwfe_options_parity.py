"""``sp.etwfe`` option by option: estimates and standard errors against R
``etwfe`` and Stata ``jwdid``.

Panel: ``_fixtures/etwfe_options_panel.csv`` (240 units x 2001-2008, cohorts
2004 / 2006 / 2007 and 80 never-treated units; written by
``_generate_etwfe_options_data.py``).

R reference (``_generate_etwfe_options_R.R``; R 4.5.2, etwfe 0.6.2, fixest
0.14.0, marginaleffects 0.32.0): every ``emfx`` aggregation (simple / group /
calendar / event) of

* the linear fit under both control groups, clustered (``vcov = ~id``) and
  heteroskedasticity-robust (``vcov = "hetero"``, i.e. ``sp.etwfe(panel=False)``
  without ``cluster=``);
* the weighted fit (``weights = ~w``), aggregated by ``emfx()`` (every treated
  row weight one: ``agg_weights='unit'``) and by ``emfx(wts = "w")`` (the
  estimation weights: ``agg_weights='estimation'``);
* the covariate-moderated fit (``xvar = xv``, ``emfx(by_xvar = FALSE)``),
  and a categorical moderator ``xcat`` in the per-level design StatsPAI
  fits, by ``fixest::feols`` directly (``etwfe`` 0.6.2 gives a factor one
  summed period slope, a different model -- T4, pinned below);
* the additive control ``x1`` that ``sp.etwfe(controls=)`` fits (Stata
  ``jwdid``'s ``exovar()``; R ``etwfe`` interacts controls with the cells,
  a different model), fitted by ``fixest::feols`` on ``etwfe``'s own cell
  basis and aggregated by ``marginaleffects::slopes`` exactly as ``emfx``
  does;
* the Poisson and logit fits (``family =``), on both scales, and the
  Poisson fit with ``xvar``.

``marginaleffects`` differentiates numerically; the generator switches it to
Richardson extrapolation (its default forward difference leaves the SEs
1e-7 to 1e-5 relative from the exact linear map) and fits the nonlinear
models with ``glm.tol = 1e-12`` and then once more from their own solution
(fixest's default 1e-8 leaves the bread ~1e-6 from the converged one, and
fixest takes the bread from the penultimate IRLS iterate, 8e-7 off in the
logit SEs here).  With these, every StatsPAI estimate agrees to
<= 8e-12 and every SE to <= 3.1e-9 (observed maxima are quoted at each
assertion), so the tolerances below are 1e-9 / 1e-7: same-byte T2 parity.

Stata reference (``_generate_etwfe_options_Stata.do``; Stata 18, jwdid v2.2,
no ``ivar()`` so that jwdid fits the cohort + period design):

* ``estat ..., vce(unconditional)`` after ``jwdid, method(poisson|logit)`` --
  the only reference for ``response_se='unconditional'`` (R has none).  The
  estimates agree to 6e-11; the SEs differ by exactly one documented
  small-sample factor: Stata's ML commands scale the clustered sandwich by
  ``G / (G - 1)``, fixest (and StatsPAI, which follows R etwfe) by
  ``G / (G - 1) * (n - 1) / (n - K)``, so ``se_Stata = se * sqrt((n - K) /
  (n - 1))``, pinned at 1e-9 (T4: documented convention).  The conditional
  SEs reconstruct the same way.
* ``jwdid y, exovar(x1)``: estimates and SEs equal to 3e-13 / 4e-14 (T2;
  without ``ivar()`` reghdfe counts the cohort effects as fixest does).
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).resolve().parent / "_fixtures"
TYPES = ("simple", "group", "calendar", "event")
KW = dict(y="y", group="id", time="year", first_treat="g")

#: R fixture key -> sp.etwfe keyword arguments (linear model).
LINEAR = {
    "notyet": {},
    "never": {"cgroup": "nevertreated"},
    "notyet_hetero": {"panel": False},
    "xv_notyet": {"xvar": "xv"},
    "xv_never": {"xvar": "xv", "cgroup": "nevertreated"},
    "xcat_notyet": {"xvar": "xcat"},
    "xcat_never": {"xvar": "xcat", "cgroup": "nevertreated"},
    "ctrl_notyet": {"controls": ["x1"]},
    "ctrl_never": {"controls": ["x1"], "cgroup": "nevertreated"},
}
#: Unweighted fits under agg_weights='unit' -- R emfx's own rule (N = 1 per
#: treated row) -- against the same R rows.
UNIT_AGG = {
    "notyet": {"agg_weights": "unit"},
    "never": {"agg_weights": "unit", "cgroup": "nevertreated"},
}
#: (R fixture key, emfx rule) -> sp.etwfe keyword arguments (weighted fits).
WEIGHTED = {
    ("w_notyet", "unit"): {"weights": "w", "agg_weights": "unit"},
    ("w_notyet", "estimation"): {"weights": "w", "agg_weights": "estimation"},
    ("w_never", "unit"): {
        "weights": "w",
        "agg_weights": "unit",
        "cgroup": "nevertreated",
    },
    ("w_never", "estimation"): {
        "weights": "w",
        "agg_weights": "estimation",
        "cgroup": "nevertreated",
    },
}
#: R fixture key -> sp.etwfe keyword arguments (nonlinear families).
GLM = {
    "pois_notyet": {"y": "yc", "family": "poisson"},
    "pois_never": {"y": "yc", "family": "poisson", "cgroup": "nevertreated"},
    "logi_notyet": {"y": "yb", "family": "logit"},
    "logi_never": {"y": "yb", "family": "logit", "cgroup": "nevertreated"},
    "pois_xv_notyet": {"y": "yc", "family": "poisson", "xvar": "xv"},
}
#: Stata fixture key -> sp.etwfe keyword arguments.
STATA_GLM = {
    "pois_notyet": {"y": "yc", "family": "poisson"},
    "pois_notyet_uncond": {
        "y": "yc",
        "family": "poisson",
        "response_se": "unconditional",
    },
    "pois_never_uncond": {
        "y": "yc",
        "family": "poisson",
        "cgroup": "nevertreated",
        "response_se": "unconditional",
    },
    "logit_notyet": {"y": "yb", "family": "logit"},
    "logit_notyet_uncond": {
        "y": "yb",
        "family": "logit",
        "response_se": "unconditional",
    },
    "logit_never_uncond": {
        "y": "yb",
        "family": "logit",
        "cgroup": "nevertreated",
        "response_se": "unconditional",
    },
}
STATA_CTRL = {
    "ctrl_notyet": {"controls": ["x1"]},
    "ctrl_never": {"controls": ["x1"], "cgroup": "nevertreated"},
}


@pytest.fixture(scope="module")
def ref_r():
    return json.loads((_FIX / "etwfe_options_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ref_stata():
    return json.loads((_FIX / "etwfe_options_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(_FIX / "etwfe_options_panel.csv")


_FITS: dict = {}


def _fit(data, **kw):
    key = repr(sorted(kw.items()))
    if key not in _FITS:
        args = {**KW, **kw}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _FITS[key] = sp.etwfe(data, **args)
    return _FITS[key]


def _ours(fit, type, scale=None):
    """{key: (estimate, se)} of one emfx aggregation; key None for simple."""
    kw = {} if scale is None else {"scale": scale}
    if type == "event":
        kw["include_leads"] = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = sp.etwfe_emfx(fit, type=type, **kw)
    if type == "simple":
        return {None: (float(e.estimate), float(e.se))}
    det = e.detail
    key = next(
        c
        for c in ("cohort", "calendar_time", "period", "event_time", "relative_time")
        if c in det.columns
    )
    est = "estimate" if "estimate" in det.columns else "att"
    return {
        int(k): (float(b), float(s)) for k, b, s in zip(det[key], det[est], det["se"])
    }


def _r_rows(block):
    """{key: (estimate, se)}; drops the reference period (SE null in emfx)."""
    est = block["estimate"]
    se = block["std.error"]
    if not isinstance(est, list):
        return {None: (est, se)}
    return {int(k): (b, s) for k, b, s in zip(block["key"], est, se) if s is not None}


def _stata_rows(block, type, ours):
    """Stata labels the rows 1.__event__, 2004.g, ...; align on our keys."""
    b, se = block["b"], block["se"]
    if type == "simple":
        return {None: (b[0], se[0])}
    keys = sorted(k for k in ours)
    if type == "event":
        # estat event lists every relative time, the reference one (b = 0,
        # se = 0) included; ours omits the reference.
        rows = [(bb, ss) for bb, ss in zip(b, se) if ss > 0]
    else:
        rows = list(zip(b, se))
    assert len(rows) == len(keys), (type, len(rows), keys)
    return dict(zip(keys, rows))


def _assert_rows(ours, ref, est_rel, se_rel, se_factor=1.0):
    assert set(ours) == set(ref)
    for k, (b, s) in ref.items():
        assert ours[k][0] == pytest.approx(b, rel=est_rel, abs=1e-12), k
        assert ours[k][1] * se_factor == pytest.approx(s, rel=se_rel), k


# --------------------------------------------------------------------------- #
#  R etwfe: linear model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("spec", sorted(LINEAR))
def test_linear_matches_r_etwfe(ref_r, panel, spec, type):
    fit = _fit(panel, **LINEAR[spec])
    # Observed maxima over all specs: estimates 1.8e-12, SEs 3.1e-9.
    _assert_rows(_ours(fit, type), _r_rows(ref_r[spec][type]), 1e-9, 1e-7)


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("spec", sorted(UNIT_AGG))
def test_unweighted_unit_aggregation_matches_r_etwfe(ref_r, panel, spec, type):
    fit = _fit(panel, **UNIT_AGG[spec])
    assert fit.model_info["agg_weights"] == "unit"
    _assert_rows(_ours(fit, type), _r_rows(ref_r[spec][type]), 1e-9, 1e-7)


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("spec", sorted(WEIGHTED))
def test_weighted_matches_r_etwfe(ref_r, panel, spec, type):
    key, rule = spec
    fit = _fit(panel, **WEIGHTED[spec])
    assert fit.model_info["agg_weights"] == rule
    # Observed maxima: estimates 2.2e-14, SEs 1.6e-9.
    _assert_rows(_ours(fit, type), _r_rows(ref_r[key][rule][type]), 1e-9, 1e-7)


@pytest.mark.parametrize("spec", ["notyet", "never", "w_notyet", "xv_notyet"])
def test_small_sample_k_matches_fixest(ref_r, panel, spec):
    kw = LINEAR[spec] if spec in LINEAR else WEIGHTED[(spec, "unit")]
    fit = _fit(panel, **kw)
    assert fit.model_info["ssc"]["n"] == ref_r[spec]["ssc"]["n"]
    assert fit.model_info["ssc"]["K"] == ref_r[spec]["ssc"]["K"]


def test_weighting_rules_differ_on_this_panel(ref_r):
    """The fixture discriminates the two agg_weights rules (weights vary within
    units), so the weighted rows above test which rule is applied."""
    a = ref_r["w_notyet"]["unit"]["simple"]["estimate"]
    b = ref_r["w_notyet"]["estimation"]["simple"]["estimate"]
    assert abs(a - b) > 1e-3


# --------------------------------------------------------------------------- #
#  xvar: R etwfe's covariate-moderated design (correctness fix)
# --------------------------------------------------------------------------- #


def test_xvar_cell_coefficients_are_the_unmoderated_ones(ref_r):
    """The moderator is demeaned within cohort x period cells, so it is
    orthogonal to the cell dummies: R's xvar estimates equal the plain fit's
    and only the SEs move."""
    for cg in ("notyet", "never"):
        plain, mod = ref_r[cg]["simple"], ref_r[f"xv_{cg}"]["simple"]
        assert mod["estimate"] == pytest.approx(plain["estimate"], rel=1e-12)
        assert abs(mod["std.error"] - plain["std.error"]) > 1e-4


def test_xvar_honours_the_control_group(ref_r, panel):
    """Through 1.39.6 the linear xvar fit ignored cgroup='nevertreated' (both
    control groups returned 1.4957, SE 0.0906, on this panel)."""
    a = _fit(panel, **LINEAR["xv_notyet"])
    b = _fit(panel, **LINEAR["xv_never"])
    assert a.estimate == pytest.approx(
        ref_r["xv_notyet"]["simple"]["estimate"], rel=1e-9
    )
    assert b.estimate == pytest.approx(
        ref_r["xv_never"]["simple"]["estimate"], rel=1e-9
    )
    assert abs(a.estimate - b.estimate) > 1e-2


def test_categorical_xvar_is_per_level_not_etwfe_summed_slope(ref_r, panel):
    """A categorical moderator enters as level dummies with one period slope
    per level (the rows above match that design, fitted by fixest).  etwfe
    0.6.2 writes i(year, (xcatb_dm + xcatc_dm)) -- one summed slope -- a
    different model: same estimates (the terms are orthogonal to the cells),
    different SE.  If a later etwfe fits per-level slopes, this fails and the
    xcat rows can be read off etwfe itself."""
    fit = _fit(panel, **LINEAR["xcat_notyet"])
    assert fit.model_info["xvar"] == ["xcat[b]", "xcat[c]"]
    summed = ref_r["xcat_etwfe_notyet"]["simple"]
    assert fit.estimate == pytest.approx(summed["estimate"], rel=1e-9)
    assert abs(fit.se / summed["std.error"] - 1) > 1e-4


@pytest.mark.parametrize("cg", ["notyet", "never"])
def test_xvar_cohort_slopes_match_r(ref_r, panel, cg):
    """Per-cohort moderator slope: the cell slopes ``.Dtreat:g:year:xv_dm`` of
    the R fit averaged over the cohort's post cells by their counts, SE from
    fixest's vcov (computed in the R generator)."""
    fit = _fit(panel, **LINEAR[f"xv_{cg}"])
    ref = ref_r[f"xv_{cg}"]["slopes"]
    det = fit.detail.set_index("cohort")
    for g, b, s in zip(ref["cohort"], ref["estimate"], ref["std.error"]):
        assert det.loc[int(g), "slope_xv"] == pytest.approx(b, rel=1e-9)
        assert det.loc[int(g), "slope_xv_se"] == pytest.approx(s, rel=1e-9)


# --------------------------------------------------------------------------- #
#  R etwfe: nonlinear families
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("scale", ["response", "link"])
@pytest.mark.parametrize("spec", sorted(GLM))
def test_glm_matches_r_etwfe(ref_r, panel, spec, scale, type):
    fit = _fit(panel, **GLM[spec])
    # Observed maxima: estimates 7.8e-12 (IRLS stopping rules), SEs 9.4e-10.
    _assert_rows(_ours(fit, type, scale), _r_rows(ref_r[spec][scale][type]), 1e-9, 1e-7)


# --------------------------------------------------------------------------- #
#  Stata jwdid: response_se='unconditional' and the conditional SE (T4 factor)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("spec", sorted(STATA_GLM))
def test_glm_se_reconstructs_stata_margins(ref_stata, panel, spec, type):
    fit = _fit(panel, **STATA_GLM[spec])
    ref = ref_stata[spec]
    n, k = ref["N"], ref["rank"]
    factor = math.sqrt((n - k) / (n - 1.0))
    ours = _ours(fit, type, "response")
    # Observed: estimates 5.9e-11; SEs after the factor 7.3e-14.
    _assert_rows(ours, _stata_rows(ref[type], type, ours), 1e-9, 1e-9, factor)


def test_stata_rank_is_fixest_k(ref_r, ref_stata):
    """The K of the reconstruction factor is the same parameter count fixest
    uses for its (n - 1) / (n - K) factor."""
    assert ref_stata["pois_notyet"]["rank"] == ref_r["pois_notyet"]["ssc"]["K"]
    assert ref_stata["logit_never_uncond"]["rank"] == ref_r["logi_never"]["ssc"]["K"]


def test_unconditional_differs_from_conditional(ref_stata):
    """The unconditional rows are not the conditional ones relabelled: on
    this panel (cohort effects, so little composition variance) the two
    differ by 2.4e-4 relative for Poisson and more for logit, far above the
    1e-9 the SEs are matched at."""
    for fam in ("pois", "logit"):
        a = ref_stata[f"{fam}_notyet"]["simple"]["se"][0]
        b = ref_stata[f"{fam}_notyet_uncond"]["simple"]["se"][0]
        assert abs(a - b) / a > 1e-5, fam


@pytest.mark.parametrize("type", TYPES)
@pytest.mark.parametrize("spec", sorted(STATA_CTRL))
def test_controls_match_stata_jwdid_exovar(ref_stata, panel, spec, type):
    fit = _fit(panel, **STATA_CTRL[spec])
    ours = _ours(fit, type)
    # Observed: estimates 3.3e-13, SEs 3.9e-14.
    _assert_rows(ours, _stata_rows(ref_stata[spec][type], type, ours), 1e-9, 1e-9)
