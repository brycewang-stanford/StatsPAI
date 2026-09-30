"""``sp.did_imputation(autosample=True)`` = Stata ``did_imputation, autosample``.

Units treated from the first period of a panel have no untreated history,
so their unit effects cannot be imputed. Stata stops (rc 198) unless
``autosample`` drops those observations; ``sp.did_imputation`` had no such
option (AI-tocracy / Busting the Princelings replications). The fixture has
two such units (16 observations). Reference: Stata 18 ``did_imputation``
(``_generate_bjs_autosample_Stata.do``); estimates and SEs agree to 1e-7.

The residual ~1e-8 is Stata's: ``did_imputation`` imputes with an iterative
fixed-effect solver whose answer moves with its tolerance (1.05433340 at the
default, 1.05433343 with ``tol(1e-12)``). ``sp.did_imputation`` equals the
exact dense dummy-variable solution to 1e-12 (``test_exact_solution``).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((_FIX / "bjs_autosample_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "bjs_autosample.csv")


def test_default_refuses_like_stata(data):
    assert STATA["rc_default"] != 0
    with pytest.raises(ValueError, match="autosample=True"):
        sp.did_imputation(data, y="y", group="i", time="t", first_treat="g")


def test_overall_att(data):
    with pytest.warns(UserWarning, match="autosample dropped 16"):
        r = sp.did_imputation(
            data, y="y", group="i", time="t", first_treat="g", autosample=True
        )
    assert float(r.estimate) == pytest.approx(STATA["att"], rel=1e-7)
    assert float(r.se) == pytest.approx(STATA["se"], rel=1e-7)
    assert r.model_info["n_obs_autosample_dropped"] == len(data) - STATA["N"]


def test_event_study_with_pretrends(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.did_imputation(
            data,
            y="y",
            group="i",
            time="t",
            first_treat="g",
            autosample=True,
            horizon=[0, 1, 2, 3],
            pretrends=2,
        )
    es = r.model_info["event_study"].set_index("relative_time")
    names = {0: "tau0", 1: "tau1", 2: "tau2", 3: "tau3", -1: "pre1", -2: "pre2"}
    for k, key in names.items():
        b, se = STATA["es"][key]
        assert es.loc[k, "att"] == pytest.approx(b, rel=1e-7)
        assert es.loc[k, "se"] == pytest.approx(se, rel=1e-7)


def test_exact_solution(data):
    """The imputed ATT is the dense least-squares answer to machine precision."""
    import numpy as np

    kept = data[~data.i.isin([5, 9])]
    treated = (kept.t >= kept.g.fillna(np.inf)).to_numpy()
    X = np.column_stack(
        [
            pd.get_dummies(kept.i, dtype=float).to_numpy(),
            pd.get_dummies(kept.t, dtype=float).to_numpy()[:, 1:],
        ]
    )
    b = np.linalg.lstsq(X[~treated], kept.y.to_numpy()[~treated], rcond=None)[0]
    exact = float(np.mean(kept.y.to_numpy()[treated] - X[treated] @ b))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.did_imputation(
            data, y="y", group="i", time="t", first_treat="g", autosample=True
        )
    assert float(r.estimate) == pytest.approx(exact, rel=1e-12)
