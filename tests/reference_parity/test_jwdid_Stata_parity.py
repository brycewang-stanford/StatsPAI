"""``sp.jwdid`` -- Stata option names -- against Stata ``jwdid``.

Two references on the bytes of ``_fixtures/etwfe_poisson_jwdid_hettype.csv``
(360 units x 8 years; see ``test_etwfe_poisson_jwdid_parity.py``):

* ``etwfe_poisson_jwdid_Stata.json`` -- thirteen ``jwdid ...,
  method(ppmlhdfe)`` specifications (``hettype()``, ``never``, ``i.xcat``,
  ``xc``).  Here they are called through ``sp.jwdid`` with Stata's option
  names, the categorical covariate as the factor term ``"i.xcat"`` on the
  raw integer column rather than a pre-built pandas category.
* ``jwdid_exovar_factor_Stata.json`` (``_generate_jwdid_exovar_factor_Stata.do``,
  jwdid v2.2 / ppmlhdfe 2.3.3, Stata 18) -- ``exovar(i.year#i.xcat)``,
  the same under ``never``, and ``exovar(c.xc#i.xcat)``: the factor-term
  expansion of ``statspai.did._factor_terms``, including the cells Stata
  keeps for a ``#`` product without its main effects.

Tolerances as in the ETWFE parity module: link-scale estimates 1e-9 and SEs
1e-7 relative (closed-form on both sides, PPML to ~1e-15); the
response-scale point estimate 1e-6 (Stata ``margins`` numerical
derivatives).  ``separated='drop'`` because ``ppmlhdfe`` flags the 384
all-zero rows of this panel.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).resolve().parent / "_fixtures"

# fixture spec name -> sp.jwdid keyword arguments (Stata option names)
SPECS = {
    "default": {},
    "event": {"hettype": "event"},
    "cohort": {"hettype": "cohort"},
    "time": {"hettype": "time"},
    "twfe": {"hettype": "twfe"},
    "never": {"never": True},
    "never_cohort": {"never": True, "hettype": "cohort"},
    "never_event": {"never": True, "hettype": "event"},
    "xcat": {"x": "i.xcat"},
    "xc": {"x": "xc"},
    "xcat_event": {"x": "i.xcat", "hettype": "event"},
    "xcat_never": {"x": "i.xcat", "never": True},
    "xcat_xc": {"x": ["i.xcat", "xc"]},
}
EXOVAR = {
    "year_xcat": {"exovar": "i.year#i.xcat"},
    "year_xcat_never": {"exovar": "i.year#i.xcat", "never": True},
    "xc_xcat": {"exovar": "c.xc#i.xcat"},
}


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(_FIX / "etwfe_poisson_jwdid_hettype.csv")  # xcat: int


def _load(name):
    return json.loads((_FIX / name).read_text(encoding="utf-8"))


def _jwdid(df, predict="xb", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.jwdid(
            df,
            "y",
            ivar="id",
            tvar="year",
            gvar="g",
            method="ppmlhdfe",
            predict=predict,
            separated="drop",
            **kw,
        )


def _check_simple(r, R):
    agg = r.model_info["aggregations"]
    assert r.n_obs == R["N"]
    b, se = R["simple_link"]["b"], R["simple_link"]["se"]
    b = b[0] if isinstance(b, list) else b
    se = se[0] if isinstance(se, list) else se
    br = R["simple_response"]["b"]
    br = br[0] if isinstance(br, list) else br
    assert agg["link"]["simple"]["att"] == pytest.approx(b, rel=1e-9, abs=1e-12)
    assert agg["link"]["simple"]["se"] == pytest.approx(se, rel=1e-7)
    assert agg["response"]["simple"]["att"] == pytest.approx(br, rel=1e-6)


@pytest.mark.parametrize("name", list(SPECS))
def test_jwdid_option_names_match_stata(panel, name):
    R = _load("etwfe_poisson_jwdid_Stata.json")[name]
    r = _jwdid(panel, **SPECS[name])
    _check_simple(r, R)
    # The headline follows predict(): xb -> link scale.
    assert r.estimate == pytest.approx(
        r.model_info["aggregations"]["link"]["simple"]["att"], rel=1e-15
    )
    if "over_link" in R:
        by = sp.etwfe_emfx(r, type="simple", scale="link", by_xvar=True).detail
        by = by.set_index("level")
        for lev, b, se in zip(
            sorted(by.index), R["over_link"]["b"], R["over_link"]["se"]
        ):
            assert by.loc[lev, "att"] == pytest.approx(b, rel=1e-9, abs=1e-12)
            assert by.loc[lev, "se"] == pytest.approx(se, rel=1e-7)


@pytest.mark.parametrize("name", list(EXOVAR))
def test_exovar_factor_terms_match_stata(panel, name):
    R = _load("jwdid_exovar_factor_Stata.json")[name]
    _check_simple(_jwdid(panel, **EXOVAR[name]), R)


@pytest.mark.parametrize("name", ["default", "never", "event", "xc"])
def test_default_predict_is_estat_simple(panel, name):
    """``estat simple`` without ``predict()``: the count scale, with the
    ``margins`` SE that ``estat`` reports (``response_se='margins'``, the
    ``sp.jwdid`` default; 7e-7 as in the ETWFE margins parity)."""
    R = _load("etwfe_poisson_jwdid_Stata.json")[name]
    r = _jwdid(panel, predict=None, **SPECS[name])
    assert r.estimate == pytest.approx(R["simple_response"]["b"][0], rel=1e-6)
    assert r.se == pytest.approx(R["simple_response"]["se"][0], rel=1e-6)
    assert r.model_info["stata_estat"] == "estat simple"
    prof = _jwdid(panel, predict=None, response_se="profile", **SPECS[name])
    assert prof.estimate == pytest.approx(r.estimate, rel=1e-12)
    assert prof.se != pytest.approx(r.se, rel=1e-3)


def test_stata_equivalent_and_errors(panel):
    r = _jwdid(panel, never=True, hettype="event", exovar="i.year#i.xcat")
    assert r.model_info["stata_equivalent"] == (
        "jwdid y, ivar(id) tvar(year) gvar(g) method(ppmlhdfe) never "
        "hettype(event) exovar(i.year#i.xcat)"
    )
    with pytest.raises(MethodIncompatibility, match="method"):
        sp.jwdid(panel, "y", ivar="id", tvar="year", gvar="g", method="probit")
    with pytest.raises(MethodIncompatibility, match="predict"):
        _jwdid(panel, predict="pr")
    with pytest.raises(MethodIncompatibility, match="nonlinear"):
        sp.jwdid(panel, "y", ivar="id", tvar="year", gvar="g", predict="mu")
