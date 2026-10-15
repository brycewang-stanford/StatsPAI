"""Stata ``did_imputation`` parity for the options fixture 84 does not reach.

Custom fixed effects, a time-varying control, two kinds of control at once,
and estimation weights, on mpdta with four columns added
(``option_parity/data_94_bjs_options.csv``):

=======================================  ===================================
Stata ``did_imputation``                  ``sp.did_imputation``
=======================================  ===================================
``fe(countyreal region#year)``            ``fe=['countyreal', 'region#year']``
``fe(region year)``                       ``fe=['region', 'year']``
``controls(x)``                           ``controls=['x']``
``controls(x) timecontrols(lpop)``        ``controls=['x'], time_covariates=``
``controls(x) unitcontrols(year)``        ``controls=['x'], unit_covariates=``
``[aw=w]``                                ``weights='w'``
``[aw=w], horizons(0/3)``                 ``weights='w', horizon=[0, ..., 3]``
``[aw=w], fe(...) controls(x)``           all three
``fe(countyreal region#year) horizons``   ``fe=..., horizon=[0, ..., 3]``
``fe(year)``, a cohort always treated      ``fe=['year']``
=======================================  ===================================

The last row is a regression test as well: a unit treated in every period
it is observed has no imputable unit effect, which matters only when the
model has one. Before 2026-10-15 ``sp.did_imputation`` refused such units
under any ``fe=``; now the check follows the fixed effects actually named,
and the fit equals Stata's.

Golden numbers: Stata 18 MP, ``did_imputation`` (2023-11-22 build),
``tests/stata_parity/option_parity/94_bjs_options.do``, read from
``option_parity/results/94_bjs_options_Stata.json``.

Tolerances
----------
Estimates: relative 1e-6, the default same-byte budget. Stata's estimates
come from reghdfe's fixed-effect solver, whose 1e-8 tolerance
did_imputation does not expose; they sit 2.5e-9 to 9.3e-7 from StatsPAI,
and StatsPAI equals the exact least-squares imputation to 1e-11
(``test_estimates_are_the_exact_least_squares_imputation``), so the gap is
the reference's solver. The worst rows are the region-by-year effect with
horizons (9.3e-7) and the weighted fit with a region-by-year effect and a
control (6.7e-7).

The row with ``unitcontrols()`` is held to 5e-6, as fixture 84 holds
``unitcontrols(year)`` alone: Stata's iteration does not settle there.
Run twice, the do-file returned -0.0296747470 and then -0.0296746634 for
it (8.9e-8 and 2.9e-6 from StatsPAI; the second run differs from the first
only in a preserve / restore earlier in the file). Its estimate is a
disclosure, and the claim that StatsPAI is right rests on the exact
solution; its SE is the exact variance on both runs.

Standard errors: relative 1e-8; worst observed 5.7e-9. did_imputation
iterates its imputation weights to ``tol(1e-12)`` here, so its variance
is the exact one.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_OPT = pathlib.Path(__file__).resolve().parents[1] / "stata_parity" / "option_parity"
_DATA = _OPT / "data_94_bjs_options.csv"
STATA = json.loads(
    (_OPT / "results" / "94_bjs_options_Stata.json").read_text(encoding="utf-8")
)
KEYS = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")
H = [0, 1, 2, 3]

RTOL_EST = 1e-6
RTOL_EST_UNIT_SLOPES = 5e-6
RTOL_SE = 1e-8

#: fixture key -> (sample, sp.did_imputation options); sample is False (all
#: rows), True (units with >= 2 untreated periods) or "always_treated" (the
#: 2004 cohort without its 2003 rows, so treated in every period it is seen)
CASES = {
    "fe_unit_region_year": (False, dict(fe=["countyreal", "region#year"])),
    "fe_region_year": (False, dict(fe=["region", "year"])),
    "controls_x": (False, dict(controls=["x"])),
    "mixed_controls_timecontrols": (
        False,
        dict(controls=["x"], time_covariates=["lpop"]),
    ),
    "weights": (False, dict(weights="w")),
    "weights_horizons": (False, dict(weights="w", horizon=H)),
    "weights_fe_unit_region_year_controls_x": (
        False,
        dict(weights="w", fe=["countyreal", "region#year"], controls=["x"]),
    ),
    "fe_unit_region_year_horizons": (
        False,
        dict(fe=["countyreal", "region#year"], horizon=H),
    ),
    "mixed_controls_unitcontrols_subset": (
        True,
        dict(controls=["x"], unit_covariates=["year"]),
    ),
    "fe_time_only_always_treated": ("always_treated", dict(fe=["year"])),
}


def _data(subset) -> pd.DataFrame:
    d = pd.read_csv(_DATA)
    if subset == "always_treated":
        return d[~((d["first_treat"] == 2004) & (d["year"] == 2003))].copy()
    if subset:
        # Units with >= 2 untreated periods (see test_bjs_fe_covariates_parity).
        untr = (d["first_treat"] == 0) | (d["year"] < d["first_treat"])
        d = d[untr.groupby(d["countyreal"]).transform("sum") >= 2].copy()
    return d


def _fit(case):
    subset, options = CASES[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_imputation(_data(subset), **KEYS, **options)


def _reported(res, case):
    """``{stata column name: (estimate, se)}`` for the fit."""
    if "horizon" not in CASES[case][1]:
        return {"tau": (float(res.estimate), float(res.se))}
    es = res.model_info["event_study"].set_index("relative_time")
    return {f"tau{h}": (float(es.loc[h, "att"]), float(es.loc[h, "se"])) for h in H}


def test_cases_cover_the_whole_fixture():
    assert set(CASES) == set(STATA) - {"_meta"}


@pytest.mark.parametrize("case", sorted(CASES))
def test_estimates_and_ses_match_stata(case):
    res = _fit(case)
    want = STATA[case]
    got = _reported(res, case)
    rtol_est = RTOL_EST_UNIT_SLOPES if "unitcontrols" in case else RTOL_EST
    for name, (est, se) in got.items():
        assert est == pytest.approx(want[name]["att"], rel=rtol_est), (case, name)
        assert se == pytest.approx(want[name]["se"], rel=RTOL_SE), (case, name)
    assert int(res.n_obs) == want["_N"], case


@pytest.mark.parametrize("case", sorted(CASES))
def test_scope_labels_the_y0_model(case):
    """The fits report the configuration sp.validation_scope reads."""
    mi = _fit(case).model_info
    _, options = CASES[case]
    fe = options.get("fe")
    assert mi["y0_fe"] == (
        "unit_time" if fe is None else "time" if fe == ["year"] else "custom"
    )
    kinds = [
        k for k in ("controls", "unit_covariates", "time_covariates") if k in options
    ]
    assert mi["y0_covariates"] == (
        "none" if not kinds else kinds[0] if len(kinds) == 1 else "mixed"
    )


def _dense_imputation(d, options):
    """Exact imputation: dense least squares on the untreated rows."""
    w = d["w"].to_numpy(float) if options.get("weights") else np.ones(len(d))
    D = ((d["first_treat"] > 0) & (d["year"] >= d["first_treat"])).to_numpy()
    blocks = [np.ones((len(d), 1))]
    for spec in options.get("fe", ["countyreal", "year"]):
        key = d[spec.split("#")].astype(str).agg("|".join, axis=1)
        blocks.append(pd.get_dummies(key).to_numpy(float)[:, 1:])
    for c in options.get("controls", []):
        blocks.append(d[[c]].to_numpy(float))
    for c in options.get("time_covariates", []):
        blocks.append(pd.get_dummies(d["year"]).to_numpy(float) * d[[c]].to_numpy())
    for c in options.get("unit_covariates", []):
        # centred: the same column space next to the unit dummies, and an
        # SVD that converges (raw years near 2005 are nearly collinear)
        v = d[[c]].to_numpy(float)
        blocks.append(pd.get_dummies(d["countyreal"]).to_numpy(float) * (v - v.mean()))
    X = np.column_stack(blocks)
    sw = np.sqrt(w[~D])
    b = np.linalg.lstsq(X[~D] * sw[:, None], d["lemp"].to_numpy()[~D] * sw, rcond=None)[
        0
    ]
    tau = d["lemp"].to_numpy()[D] - X[D] @ b
    rel = (d["year"] - d["first_treat"]).to_numpy()[D]
    out = {"tau": float(np.sum(w[D] * tau) / np.sum(w[D]))}
    for h in H:
        on = rel == h
        out[f"tau{h}"] = float(np.sum(w[D][on] * tau[on]) / np.sum(w[D][on]))
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_estimates_are_the_exact_least_squares_imputation(case):
    subset, options = CASES[case]
    exact = _dense_imputation(_data(subset), options)
    for name, (est, _) in _reported(_fit(case), case).items():
        assert est == pytest.approx(exact[name], rel=1e-10), (case, name)


def test_options_change_the_answer():
    """Each option moves the ATT, so none can be silently ignored."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = sp.did_imputation(_data(False), **KEYS).estimate
    for case in (
        "fe_unit_region_year",
        "fe_region_year",
        "controls_x",
        "mixed_controls_timecontrols",
        "weights",
    ):
        assert abs(STATA[case]["tau"]["att"] - default) > 1e-5, case


def test_unimputable_custom_fixed_effect_is_refused():
    """A cell effect with no untreated row cannot be imputed: refuse, as Stata.

    mpdta's treatment is a state minimum wage, so under ``state#year`` every
    treated county-year sits in a state-year with no untreated county. Before
    the fix the 291 rows took the reference cell's effect and the ATT came
    back -1.118, without a warning; Stata stops ("Could not impute FE for
    291 observations").
    """
    d = _data(False)
    with pytest.raises(ValueError, match="could not impute the fixed effects of 291"):
        sp.did_imputation(d, **KEYS, fe=["countyreal", "st#year"])
    # autosample drops them, leaving nothing treated.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="No treated observations"):
            sp.did_imputation(d, **KEYS, fe=["countyreal", "st#year"], autosample=True)
