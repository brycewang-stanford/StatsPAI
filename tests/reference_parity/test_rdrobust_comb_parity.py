"""Reference parity: ``sp.rdrobust`` under the comb bandwidth selectors.

``msecomb1`` / ``msecomb2`` / ``cercomb1`` / ``cercomb2`` combine the
``rd``, ``two`` and ``sum`` selectors elementwise (``comb1`` is the
minimum of ``rd`` and ``sum``, ``comb2`` the median of all three).
``test_rdbwselect_comb_rules.py`` pins those identities and Track A
module 88 pins the comb bandwidths of ``sp.rdbwselect``; neither compares
what ``sp.rdrobust`` reports at a comb bandwidth. This file does, against
both canonical implementations, on the same committed CSVs:

* R ``rdrobust`` 4.0.0 -- ``_fixtures/_generate_rdrobust_comb_R.R``
  writes ``rdrobust_comb_R.json``;
* Stata ``rdrobust`` 10.0.0 -- ``tests/stata_parity/option_parity/
  95_rdrobust_comb.do`` writes ``95_rdrobust_comb_Stata.json``.

Grid: four selectors x p in {1, 2} x three kernels on the sharp senate
design, plus the four selectors on the fuzzy design with two-sided
noncompliance (``rdsenate_fuzzy.csv``). Eight outputs per fit: the
conventional and robust estimates, their standard errors, and h and b on
each side.

Measured 2026-10-15: the worst of the 28 x 8 comparisons is 2.2e-12
against R. Against Stata it is 2.5e-9, and that is the distance between
the two references themselves: Stata ``rdrobust`` 10.0.0 and R
``rdrobust`` 4.0.0 differ from each other by up to 2.5e-9 on these fits
(h and b on the sharp grid; 1.9e-9 on the estimates), while StatsPAI
sits within 2.2e-12 of R. Track A module 06 shows the same order of
gap on the Stata side (``tests/test_parity_harness_contract.py`` holds
it below 1e-8).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_HERE = pathlib.Path(__file__).parent
_FIX = _HERE / "_fixtures"
_STATA = _HERE.parent / "stata_parity" / "option_parity" / "results"

#: The bandwidth is the end of a chain of pilot regressions; 1e-9 is the
#: tolerance of the 36-cell grid in test_rdrobust_parity.py and leaves
#: room for platform BLAS differences, two orders above the measured gap
#: and three below the 1e-6 parity budget.
RTOL = 1e-9
#: Stata's own distance from R is 2.5e-9 (module docstring); the bound is
#: four times that, as for Track A module 06 on the Stata side, and two
#: orders inside the 1e-6 budget.
RTOL_STATA = 1e-8

SELECTORS = ("msecomb1", "msecomb2", "cercomb1", "cercomb2")
OUTPUTS = (
    "coef_conventional",
    "coef_robust",
    "se_conventional",
    "se_robust",
    "h_left",
    "h_right",
    "b_left",
    "b_right",
)


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def r_ref() -> dict:
    return _load(_FIX / "rdrobust_comb_R.json")


@pytest.fixture(scope="module")
def stata_ref() -> dict:
    return _load(_STATA / "95_rdrobust_comb_Stata.json")


@pytest.fixture(scope="module")
def data() -> dict:
    return {
        "sharp": pd.read_csv(_FIX / "rdsenate.csv"),
        "fuzzy": pd.read_csv(_FIX / "rdsenate_fuzzy.csv"),
    }


def _pair(value):
    if isinstance(value, dict):
        return float(value["left"]), float(value["right"])
    if isinstance(value, (tuple, list, np.ndarray)):
        return float(value[0]), float(value[1])
    return float(value), float(value)


def _spec(key: str) -> dict:
    design, bw, p, *kernel = key.split("_")
    return dict(design=design, bwselect=bw, p=int(p[1:]), kernel="_".join(kernel))


@pytest.fixture(scope="module")
def fits(r_ref, data) -> dict:
    out = {}
    for key in (k for k in r_ref if not k.startswith("_")):
        spec = _spec(key)
        kw = dict(p=spec["p"], kernel=spec["kernel"], bwselect=spec["bwselect"])
        if spec["design"] == "fuzzy":
            kw["fuzzy"] = "treat2"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.rdrobust(data[spec["design"]], y="vote", x="margin", c=0, **kw)
        h_l, h_r = _pair(res.model_info["bandwidth_h"])
        b_l, b_r = _pair(res.model_info["bandwidth_b"])
        out[key] = {
            "result": res,
            "coef_conventional": float(res.detail["estimate"][0]),
            "coef_robust": float(res.detail["estimate"][1]),
            "se_conventional": float(res.detail["se"][0]),
            "se_robust": float(res.detail["se"][1]),
            "h_left": h_l,
            "h_right": h_r,
            "b_left": b_l,
            "b_right": b_r,
        }
    return out


def test_fixtures_cover_the_same_grid(r_ref, stata_ref):
    keys = {k for k in r_ref if not k.startswith("_")}
    assert keys == {k for k in stata_ref if not k.startswith("_")}
    assert len(keys) == 28
    assert {_spec(k)["bwselect"] for k in keys} == set(SELECTORS)
    assert r_ref["_meta"]["n_sharp"] == 1297
    assert stata_ref["_meta"]["precision"] == "double"


def test_the_grid_separates_the_selectors(r_ref):
    """comb2 must differ from comb1 and from the plain rd selector.

    The defect ``test_rdbwselect_comb_rules.py`` guards against resolved
    every comb variant to ``rd``. On the sharp grid ``comb1`` is the
    ``rd`` bandwidth under the triangular kernel and at p = 2 (``rd`` is
    the smaller there), and the ``sum`` bandwidth under the uniform and
    Epanechnikov kernels at p = 1; ``comb2`` gives h_left != h_right,
    which neither ``rd`` nor ``sum`` can.
    """
    for prefix in ("mse", "cer"):
        c1 = r_ref[f"sharp_{prefix}comb1_p1_triangular"]
        c2 = r_ref[f"sharp_{prefix}comb2_p1_triangular"]
        assert c1["h_left"] == c1["h_right"], prefix
        assert abs(c2["h_left"] / c2["h_right"] - 1) > 1e-2, prefix
        # comb1 picks sum over rd under the uniform kernel at p = 1
        sm = r_ref[f"sharp_{prefix}comb1_p1_uniform"]["h_left"]
        assert abs(sm / c2["h_left"] - 1) > 1e-2, prefix


@pytest.mark.parametrize(
    "reference, rtol", [("r_ref", RTOL), ("stata_ref", RTOL_STATA)]
)
def test_every_output_matches_the_reference(reference, rtol, fits, request):
    ref_all = request.getfixturevalue(reference)
    failures = []
    for key, got in fits.items():
        ref = ref_all[key]
        for name in OUTPUTS:
            if got[name] != pytest.approx(float(ref[name]), rel=rtol):
                failures.append(
                    f"{key}: {name} {got[name]:.12g} vs {float(ref[name]):.12g}"
                )
    assert not failures, f"{len(failures)} mismatches:\n  " + "\n  ".join(failures)


def test_effective_sample_sizes_match(fits, r_ref, stata_ref):
    for key, got in fits.items():
        mi = got["result"].model_info
        n_l, n_r = mi["n_effective_left"], mi["n_effective_right"]
        for ref in (r_ref[key], stata_ref[key]):
            assert (int(n_l), int(n_r)) == (
                int(ref["N_h_left"]),
                int(ref["N_h_right"]),
            ), key


def test_result_records_the_selector_and_the_native_path(fits):
    for key, got in fits.items():
        mi = got["result"].model_info
        assert mi["bwselect"] == _spec(key)["bwselect"]
        assert not mi.get("cct_delegation")
