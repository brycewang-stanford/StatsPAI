"""``sp.sdid`` placebo and bootstrap standard errors: a stochastic screen (S).

Placebo and bootstrap SEs are Monte Carlo draws. Neither R's nor Stata's
random stream can be reproduced, so no fit here is compared number for
number, and nothing in this file is parity. What is pinned exactly is the
map from one draw to one replication estimate
(``test_did_synth_R_parity.py`` replays R's own index vectors at 1e-9);
this file checks that the end-to-end SE lands where the reference's does.

References, five seeds of 200 replications each on both sides:

* ``_fixtures/sdid_resampling_R.json`` (``_generate_sdid_resampling_R.R``,
  synthdid 0.0.9): ``synthdid_se`` for the ``treated_unit=`` interface,
  ``method = sdid / sc / did``, on the Prop. 99 replica of Track A module 12
  (one treated unit, placebo only) and on ``did_synth_sdid_panel.csv`` (five
  treated units, placebo and bootstrap);
* ``_fixtures/sdid_staggered/sdid_resampling_stata.json``
  (``_generate_sdid_resampling_stata.do``, Stata ``sdid`` 2.0.2): the
  ``treat=`` interface, ``vce(placebo)`` on the Prop. 99 block design for the
  three methods and ``vce(placebo)`` / ``vce(bootstrap)`` on the staggered
  quota panel.

Criterion: the mean SE over StatsPAI's five seeds and the mean over the
reference's five seeds differ by less than four pooled Monte Carlo standard
errors. One B-replication SE has Monte Carlo SD of about ``se / sqrt(2 B)``
(normal-theory approximation), so the pooled SD of the difference of two
five-seed means is ``se / sqrt(2 B) * sqrt(2 / 5)``, about 3.2% of the SE
at B = 200; the bound is about 13%. This is a screen, not an equivalence
test: no margin is pre-registered and the seeds are not spread out.

The normal-theory SD understates the spread when a few placebo units carry
extreme estimates. On the Prop. 99 replica with ``method='did'`` the target
is known exactly -- with one treated unit each placebo replication is one of
38 deterministic estimates, whose population SD is 4.8670 on both sides --
and R's seeds 1-5 average 4.477 while 40 R seeds average 4.939: R's five
seeds sit in their own lower tail, and the gap to StatsPAI's 4.914 (3.1
pooled SEs) is R's sampling, not a difference in the algorithm.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
_TRACK_A = Path(__file__).resolve().parents[1] / "r_parity" / "data"
R_REF = json.loads((_FIX / "sdid_resampling_R.json").read_text(encoding="utf-8"))
STATA_REF = json.loads(
    (_FIX / "sdid_staggered" / "sdid_resampling_stata.json").read_text(encoding="utf-8")
)
SEEDS = (1, 2, 3, 4, 5)
REPS = 200


def _within_screen(ours: np.ndarray, ref: np.ndarray) -> None:
    assert np.isfinite(ours).all() and np.isfinite(ref).all()
    pooled = ref.mean() / np.sqrt(2 * REPS) * np.sqrt(2 / len(SEEDS))
    gap = abs(ours.mean() - ref.mean())
    assert gap < 4 * pooled, (ours.mean(), ref.mean(), gap / pooled)


# --------------------------------------------------------------------------
#  treated_unit= interface vs R synthdid
# --------------------------------------------------------------------------

_DATA = {
    "prop99": dict(
        path=_TRACK_A / "12_sdid.csv", unit="state", time="year", y="cigsale"
    ),
    "panel": dict(
        path=_FIX / "did_synth_sdid_panel.csv", unit="unit", time="year", y="y"
    ),
}


def _treated_unit_ses(ds: str, method: str, se_method: str) -> np.ndarray:
    meta = _DATA[ds]
    df = pd.read_csv(meta["path"])
    treated = sorted(df.loc[df["treated"] == 1, meta["unit"]].unique())
    first = df.loc[df["treated"] == 1, meta["time"]].min()
    out = []
    for s in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.sdid(
                df,
                outcome=meta["y"],
                unit=meta["unit"],
                time=meta["time"],
                treated_unit=treated if len(treated) > 1 else treated[0],
                treatment_time=first,
                method=method,
                se_method=se_method,
                n_reps=REPS,
                seed=s,
            )
        assert res.model_info["se_method"] == se_method
        out.append(res.se)
    return np.asarray(out, dtype=float)


@pytest.mark.parametrize(
    "ds,se_method",
    [("prop99", "placebo"), ("panel", "placebo"), ("panel", "bootstrap")],
)
@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_treated_unit_resampling_se_screen_against_synthdid(ds, se_method, method):
    ref = np.asarray(R_REF[ds][method][se_method], dtype=float)
    assert ref.size == len(SEEDS)
    _within_screen(_treated_unit_ses(ds, method, se_method), ref)


# --------------------------------------------------------------------------
#  treat= interface vs Stata sdid
# --------------------------------------------------------------------------


def _treat_ses(dataset: str, method: str, se_method: str) -> np.ndarray:
    if dataset == "prop99":
        df = pd.read_stata(_FIX / "sdid_staggered" / "prop99_example.dta")
        df["packspercapita"] = df["packspercapita"].astype(float)
        keys = dict(outcome="packspercapita", unit="state", treat="treated")
    else:
        df = pd.read_stata(_FIX / "sdid_staggered" / "quota_example.dta")
        for c in ("womparl", "lngdp"):
            df[c] = df[c].astype(float)
        keys = dict(outcome="womparl", unit="country", treat="quota")
    out = []
    for s in SEEDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.sdid(
                df,
                time="year",
                method=method,
                se_method=se_method,
                n_reps=REPS,
                seed=s,
                **keys,
            )
        out.append(res.se)
    return np.asarray(out, dtype=float)


@pytest.mark.parametrize(
    "dataset,method,se_method",
    [
        ("prop99", "sdid", "placebo"),
        ("prop99", "did", "placebo"),
        ("prop99", "sc", "placebo"),
        ("quota", "sdid", "placebo"),
        ("quota", "sdid", "bootstrap"),
    ],
)
def test_treat_resampling_se_screen_against_stata(dataset, method, se_method):
    ref = np.asarray(
        [STATA_REF[f"{dataset}_{method}_{se_method}_{s}"] for s in SEEDS], dtype=float
    )
    _within_screen(_treat_ses(dataset, method, se_method), ref)
