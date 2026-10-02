"""Entry-point attachments behind ``sp.validation_scope``'s IV map.

Evidence is attached to the call an artifact makes. Two IV attachments
need their own proof:

* Track A module 02 and the Track B IV row call ``sp.ivreg``; the Card
  reference test calls ``sp.iv``. The map shares their rows only because
  the two entry points return bit-identical estimates and SEs on the
  configurations those rows run, asserted here on both data sets.
* Track A module 59 pins LIML against ``ivmodel`` through ``sp.liml``, a
  separate code path from ``sp.iv(method="liml")``. The map's LIML row
  rests on this test, which runs ``sp.iv(method="liml")`` on module 59's
  bytes against the same R golden (coefficient, SE and kappa).

One ETWFE attachment rests here too. Track A module 17 and the event-study
covariance test run ``sp.etwfe(panel=False)`` for the not-yet-treated
design; the default call is ``panel=True``. On a panel the two go through
the same cohort x period cell design and differ only in the default
cluster, which is the unit either way, so the default call inherits those
rows only because it is asserted bit-identical here.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RP = _ROOT / "tests" / "r_parity"
CARD = pd.read_csv(pathlib.Path(__file__).parent / "_fixtures" / "iv_card.csv")
X = "exper + expersq + black + south + smsa"


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


@pytest.mark.parametrize("instruments", ["nearc4", "nearc4 + nearc2"])
@pytest.mark.parametrize(
    "kw", [{}, {"robust": "hc1"}, {"cluster": "cl"}, {"vce": "cr2", "cluster": "cl"}]
)
def test_iv_and_ivreg_are_bit_identical_on_card(instruments, kw):
    formula = f"lwage ~ {X} + (educ ~ {instruments})"
    a = _quiet(sp.iv, formula, data=CARD, **kw)
    b = _quiet(sp.ivreg, formula, data=CARD, **kw)
    np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())
    np.testing.assert_array_equal(a.std_errors.to_numpy(), b.std_errors.to_numpy())


def test_iv_and_ivreg_are_bit_identical_on_module_02():
    df = pd.read_csv(_RP / "data" / "02_iv.csv")
    src = (_RP / "02_iv.py").read_text(encoding="utf-8")
    formula = src.split('FORMULA = "', 1)[1].split('"', 1)[0]
    a = _quiet(sp.iv, formula, data=df, robust="hc1")
    b = _quiet(sp.ivreg, formula, data=df, robust="hc1")
    np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())
    np.testing.assert_array_equal(a.std_errors.to_numpy(), b.std_errors.to_numpy())


def test_iv_method_liml_matches_ivmodel_on_module_59():
    df = pd.read_csv(_RP / "data" / "59_liml.csv")
    ref = {
        r["statistic"]: r
        for r in json.loads(
            (_RP / "results" / "59_liml_R.json").read_text(encoding="utf-8")
        )["rows"]
    }
    fit = _quiet(sp.iv, "y ~ w + (x ~ z1 + z2)", data=df, method="liml")
    assert float(fit.params["x"]) == pytest.approx(ref["beta_x"]["estimate"], rel=1e-9)
    assert float(fit.std_errors["x"]) == pytest.approx(ref["beta_x"]["se"], rel=1e-9)
    assert float(fit.model_info["kappa"]) == pytest.approx(
        ref["kappa"]["estimate"], rel=1e-9
    )
    scope = sp.validation_scope(fit)
    assert scope["configuration"]["estimator"] == "liml"
    assert scope["status"] == "covered"


@pytest.mark.parametrize("vce", ["cr2", "cr3"])
def test_iv_honours_small_sample_cluster_vce(vce):
    """``sp.iv(vce=...)`` used to drop the option and return CR1 SEs."""
    formula = f"lwage ~ {X} + (educ ~ nearc4)"
    cr1 = _quiet(sp.iv, formula, data=CARD, cluster="cl")
    fit = _quiet(sp.iv, formula, data=CARD, vce=vce, cluster="cl")
    assert float(fit.std_errors["educ"]) != float(cr1.std_errors["educ"])
    assert vce.upper() in str(fit.model_info.get("vcov_type"))


def test_iv_rejects_a_vce_its_method_cannot_compute():
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility):
        _quiet(
            sp.iv,
            f"lwage ~ {X} + (educ ~ nearc4 + nearc2)",
            data=CARD,
            method="liml",
            vce="cr2",
            cluster="cl",
        )


@pytest.mark.parametrize("cluster", ["countyreal", None], ids=["explicit", "default"])
def test_etwfe_default_panel_call_is_bit_identical_to_module_17(cluster):
    """``sp.etwfe()`` (panel=True, not-yet-treated) vs module 17's panel=False."""
    data = pd.read_csv(_RP / "data" / "17_etwfe.csv")
    keys = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")
    module_17 = _quiet(sp.etwfe, data, panel=False, cluster="countyreal", **keys)
    extra = {} if cluster is None else {"cluster": cluster}
    default = _quiet(sp.etwfe, data, **keys, **extra)
    assert default.model_info["panel"] is True
    assert default.model_info["cgroup"] == "notyet"
    assert default.estimate == module_17.estimate
    assert default.se == module_17.se
    assert np.array_equal(
        default.model_info["event_vcov"], module_17.model_info["event_vcov"]
    )
    pd.testing.assert_frame_equal(
        default.model_info["event_study"],
        module_17.model_info["event_study"],
        check_exact=True,
    )
