"""``sp.event_study`` options against ``fixest::feols``.

35 specifications on one staggered panel
(``_fixtures/event_study_options.csv``), each fitted by ``fixest::feols``
in ``_fixtures/_generate_event_study_options_R.R`` from the definitions
``sp.event_study`` documents. Every event-time coefficient and its
clustered standard error is compared.

What the specifications cover, by ``sp.validation_scope`` dimension:

* ``adoption`` -- four adoption dates and a never-treated group, at the
  default window and reference period;
* ``window`` -- six windows besides (-4, 4): narrower, wider than the
  data (nothing clipped), asymmetric either way, and one with no leads;
* ``ref_period`` -- a single earlier period, the clipped endpoint, a
  span, both interval forms (``('<=', v)`` and ``('>=', v)``, the latter
  omitting post-treatment periods);
* ``cluster`` -- a coarser cluster (state) and the time period, which
  change which absorbed effects count in the small-sample K;
* ``bins`` -- widths 2 and 3, with a bin-aligned span and an interval
  reference, with a covariate, and with an absorbed effect;
* ``intensity`` -- a common adoption date with a continuous dose (the
  exposure x period design), clustered by state, weighted, and on a
  staggered adoption column;
* ``absorb`` -- region-by-time and state-by-time cells, with a covariate,
  weights and bins.

A dynamic TWFE regression on a staggered panel is the estimator the
heterogeneity-robust methods replace; agreeing with ``fixest`` there says
the regression StatsPAI reports is the one it documents, computed right,
not that its coefficients are clean dynamic effects.

With ``absorb='region^time'`` the cell effect spans the time effects. The
R side writes that model as ``unit + region^time``: fixest's nested K does
not detect the redundancy of ``time`` in ``region^time`` and counts all 12
levels again (K = 67 against the rank 55), which inflates its SEs by 0.3%
on the identical regression. reghdfe detects it; ``sp.event_study``'s
absorb path follows reghdfe, and Stata ``reghdfe ..., absorb(unit time
region#time) vce(cluster unit)`` returns the same SE to 1e-14.

The last six specifications are an unbalanced panel and weights that
vary over time. Before 2026-10-15 ``sp.event_study`` demeaned by one pass
of unit means and one of time means, which is exact only on a balanced
panel with weights constant within a unit; on these specifications the
coefficients were off by up to 0.035 (unbalanced) and 0.15 (time-varying
weights).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

#: Worst of the 35 x (up to 11) x 2 comparisons, measured 2026-10-15:
#: 1.8e-10 on an estimate (stag_w-2_2), 2.9e-10 on a standard error
#: (unbal_stag_tv_weights_w-3_5). fixest's demeaning stops at its own
#: convergence tolerance; StatsPAI's projection is exact.
RTOL = 1e-9

#: sp.event_study's spelling of each specification's reference period.
#: The R side lists the omitted bin labels; with bins the Python side names
#: the omitted periods (a bin-aligned span or an interval).
_REF = {
    "stag_ref_span-3_-1": [-3, -2, -1],
    "stag_ref_le-3": ("<=", -3),
    "stag_ref_ge5": (">=", 5),
    "stag_bin2": [-2, -1],
    "stag_bin3": [-3, -2, -1],
    "stag_bin2_ref_le-3": ("<=", -3),
    "single_bin2_cov": [-2, -1],
    "stag_bin2_absorb": [-2, -1],
}


@pytest.fixture(scope="module")
def ref():
    payload = json.loads(
        (_FIX / "event_study_options_R.json").read_text(encoding="utf-8")
    )
    return {k: v for k, v in payload.items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def panel():
    d = pd.read_csv(_FIX / "event_study_options.csv")
    d["treat_time"] = d["g"].where(d["g"] > 0, np.nan)
    return d


def _sample(panel, which):
    return {
        "staggered": panel,
        "single": panel[panel["g"].isin([0, 8])],
        "unbalanced": panel[panel["keep"] == 1],
        "unbalanced_single": panel[(panel["keep"] == 1) & panel["g"].isin([0, 8])],
    }[which]


def _as_list(v):
    if v is None:
        return None
    return v if isinstance(v, list) else [v]


def _fit(panel, name, spec):
    kw = {}
    if spec.get("covariates"):
        kw["covariates"] = _as_list(spec["covariates"])
    if spec.get("weights"):
        kw["weights"] = spec["weights"]
    if spec.get("intensity"):
        kw["intensity"] = spec["intensity"]
    if spec.get("absorb"):
        kw["absorb"] = spec["absorb"]
    if spec.get("bin"):
        kw["bin_width"] = int(spec["bin"])
    ref_period = _REF.get(name, spec["ref"])
    if isinstance(ref_period, list) and len(ref_period) == 1:
        ref_period = ref_period[0]
    tt = spec["treat_time"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.event_study(
            _sample(panel, spec["sample"]),
            y=spec["y"],
            treat_time="treat_time" if tt == "g" else tt,
            time="time",
            unit="unit",
            window=tuple(spec["window"]),
            ref_period=ref_period,
            cluster=None if spec["cluster"] == "unit" else spec["cluster"],
            **kw,
        )


def test_specifications_are_the_declared_ones(ref):
    assert len(ref) == 35
    for name, spec in ref.items():
        assert spec["n_coef"] == len(spec["coefs"]) >= 2, name


@pytest.mark.parametrize(
    "name",
    sorted(
        json.loads(
            (_FIX / "event_study_options_R.json").read_text(encoding="utf-8")
        ).keys()
        - {"_meta"}
    ),
)
def test_every_event_time_coefficient_and_se_matches_fixest(ref, panel, name):
    spec = ref[name]
    res = _fit(panel, name, spec)
    es = res.model_info["event_study"]
    est = es[~es["is_reference"]].set_index("relative_time")
    want = {int(k): v for k, v in spec["coefs"].items()}
    assert set(est.index.astype(int)) == set(want), name
    assert res.n_obs == spec["n"], name
    for k, v in want.items():
        for col in ("estimate", "se"):
            got = float(est.loc[k, col])
            assert got == pytest.approx(v[col], rel=RTOL), (name, k, col, got, v[col])


def test_options_change_the_answer(ref):
    """The comparison would be vacuous if an option were silently ignored."""
    base = ref["stag_default"]["coefs"]
    for other in (
        "stag_cov_wt",
        "stag_ref-2",
        "stag_absorb_region_time",
        "stag_tv_weights",
    ):
        assert abs(ref[other]["coefs"]["0"]["estimate"] - base["0"]["estimate"]) > 1e-4
    for other in ("stag_cl_state", "stag_cl_time"):
        assert ref[other]["coefs"]["0"]["estimate"] == pytest.approx(
            base["0"]["estimate"], rel=1e-12
        )
        assert abs(ref[other]["coefs"]["0"]["se"] - base["0"]["se"]) > 1e-4


def test_unbalanced_panel_is_the_two_way_projection(panel):
    """The coefficients equal the dense least-squares dummy-variable fit."""
    d = panel[panel["keep"] == 1]
    for weights in (None, "wt"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.event_study(
                d,
                y="y",
                treat_time="treat_time",
                time="time",
                unit="unit",
                weights=weights,
            )
        es = res.model_info["event_study"]
        es = es[~es["is_reference"]]
        rel = (d["time"] - d["treat_time"]).clip(-4, 4)
        cols = [
            ((rel == k) & d["treat_time"].notna()).to_numpy(float)
            for k in es["relative_time"]
        ]
        X = np.column_stack(
            cols
            + [pd.get_dummies(d["unit"]).to_numpy(float)]
            + [pd.get_dummies(d["time"]).to_numpy(float)[:, 1:]]
        )
        sw = np.sqrt(d[weights].to_numpy(float)) if weights else np.ones(len(d))
        b = np.linalg.lstsq(X * sw[:, None], d["y"].to_numpy() * sw, rcond=None)[0]
        assert es["estimate"].to_numpy() == pytest.approx(b[: len(cols)], rel=1e-9)
