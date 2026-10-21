"""``sp.bacon_decomposition(weights=)`` vs Stata ``bacondecomp [aw=]``.

Fixture: ``_fixtures/_generate_bacon_weights_Stata.do`` (Stata 18,
``bacondecomp`` with analytic weights) on the mpdta replica of Track A
module 20, with the unit weight ``1 + mod(countyreal, 7)`` and, for the
covariate case, ``x = mod(31 countyreal + 17 year, 101) / 10``.

Stata reports the decomposition by control type, so the native table is
summed the same way: the two timing types (or ``Both Treated`` with
covariates) are ``Timing_groups``, ``Treated vs Untreated`` is
``Never_v_timing``. With covariates Stata's type weights are shares of the
whole coefficient, ours of the between part, so ours are scaled by
``1 - omega``; the within row is ``(beta_within, omega)``.

Tolerances: the TWFE coefficient to 1e-12 (the same regression). The
type rows to 1e-7 relative / 2e-8 absolute: the ado's own rounding, the
same size without weights (observed 1e-9 on estimates, 9e-9 on weights).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_HERE = pathlib.Path(__file__).parent
REF = json.loads(
    (_HERE / "_fixtures" / "bacon_weights_Stata.json").read_text(encoding="utf-8")
)
DATA = _HERE.parent / "r_parity" / "data" / "20_bacon.csv"

_STATA_TYPE = {
    "Earlier vs Later Treated": "Timing_groups",
    "Later vs Earlier Treated": "Timing_groups",
    "Both Treated": "Timing_groups",
    "Later vs Always Treated": "Always_v_timing",
    "Treated vs Untreated": "Never_v_timing",
}


def _data() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d["w"] = 1.0 + d["countyreal"] % 7
    d["x"] = ((31 * d["countyreal"] + 17 * d["year"]) % 101) / 10
    return d


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("unweighted", {}),
        ("aweight", {"weights": "w"}),
        ("aweight_x", {"weights": "w", "covariates": ["x"]}),
    ],
)
def test_bacondecomp_type_rows(key, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.bacon_decomposition(
            _data(), y="lemp", treat="treat", time="year", id="countyreal", **kwargs
        )
    ref = REF[key]
    assert r["beta_twfe"] == pytest.approx(ref["b"], abs=1e-12)
    dec = r["decomposition"].assign(k=lambda f: f["type"].map(_STATA_TYPE))
    scale = 1.0 - r.get("omega", 0.0)
    for kind, part in dec.groupby("k"):
        weight = float(part["weight"].sum())
        est = float((part["weight"] * part["estimate"]).sum() / weight)
        ref_est, ref_w = ref["rows"][kind]
        assert est == pytest.approx(ref_est, rel=1e-7, abs=2e-8), kind
        assert weight * scale == pytest.approx(ref_w, abs=2e-8), kind
    if "omega" in r:
        ref_within, ref_omega = ref["rows"]["Within"]
        assert r["beta_within"] == pytest.approx(ref_within, rel=1e-7)
        assert r["omega"] == pytest.approx(ref_omega, rel=1e-7)
