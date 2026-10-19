"""``sp.weightit`` / ``sp.bal_tab`` vs R ``WeightIt`` 1.7.0 / ``cobalt`` 4.6.2.

Fixture: ``_fixtures/weightit_cobalt_data.csv`` and
``_fixtures/weightit_cobalt_R.json`` from
``_fixtures/_generate_weightit_cobalt_R.R`` (a three-level treatment, its
binary contrast and a continuous dose on the same covariates). Both R
packages are GPL and were used as black boxes (outputs only).

Tolerances: every weight agrees to 1e-9 relative except the CBPS ATT
(1e-8: WeightIt's CBPS solution is 2.7e-9 from its own entropy-balancing
ATT, which it equals in exact arithmetic) and the
multinomial-logit ``glm`` fits, where WeightIt's optimizer stops with a
score of order 1e-6 (checked directly: ``X'(Y - P)`` at its fitted
probabilities); our Newton fit drives the score to 1e-11, which leaves a
relative weight gap of ~1e-7. The balance statistics are computed from
the R weights, so they test the balance formulas alone, to 1e-9.
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
REF = json.loads((_FIX / "weightit_cobalt_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "weightit_cobalt_data.csv")
COV = ["x1", "x2", "x3"]


def _weights(key):
    c = REF["weights"][key]
    kw = {}
    if c.get("moments"):
        kw["moments"] = c["moments"]
    if c.get("int"):
        kw["interactions"] = True
    if c.get("d_moments"):
        kw["d_moments"] = c["d_moments"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W = sp.weightit(
            DATA,
            c["treat"],
            COV,
            method=c["method"],
            estimand=c["estimand"],
            focal=c["focal"],
            **kw,
        )
    return np.asarray(c["w"], dtype=float), W


@pytest.mark.parametrize("key", sorted(REF["weights"]))
def test_weights_match_weightit(key):
    ref, W = _weights(key)
    rel = np.max(np.abs(W.weights.to_numpy() - ref) / ref)
    tol = 1e-9
    if REF["weights"][key]["method"] == "glm" and key.startswith("multi"):
        tol = 2e-7
    elif key == "multi_cbps_ATT":
        # WeightIt's own CBPS ATT is 2.7e-9 from its ebal ATT, which it
        # equals in exact arithmetic; ours is 7e-13 from WeightIt's ebal
        tol = 1e-8
    assert rel < tol, rel


def test_multinomial_glm_is_the_mle():
    # the reason for the looser multinomial tolerance: our fit is at the MLE
    _, W = _weights("multi_glm_ATE")
    P = W.ps.to_numpy()
    Y = (DATA["treat"].to_numpy()[:, None] == np.array(["a", "b", "c"])).astype(float)
    X = np.column_stack([np.ones(len(DATA)), DATA[COV].to_numpy()])
    assert np.max(np.abs(X.T @ (Y - P))) < 1e-8


def test_cbps_att_equals_entropy_balancing():
    # documented WeightIt property: just-identified CBPS ATT = ebal ATT
    _, a = _weights("multi_cbps_ATT")
    _, b = _weights("multi_ebal_ATT")
    np.testing.assert_allclose(a.weights, b.weights, rtol=1e-9)


# ----------------------------------------------------------------------
# balance
# ----------------------------------------------------------------------

_COLS = {
    "Diff.Un": "diff_un",
    "Diff.Adj": "diff_adj",
    "V.Ratio.Un": "vr_un",
    "V.Ratio.Adj": "vr_adj",
    "KS.Un": "ks_un",
    "KS.Adj": "ks_adj",
    "Corr.Un": "corr_un",
    "Corr.Adj": "corr_adj",
    "S.Corr.Un": "scorr_un",
    "S.Corr.Adj": "scorr_adj",
}
_CASES = {
    "multi_ATE": dict(treat="treat", estimand="ATE"),
    "multi_ATE_binstd": dict(treat="treat", estimand="ATE", binary="std"),
    "multi_ATE_nonpair": dict(treat="treat", estimand="ATE", pairwise=False),
    "multi_ATT": dict(treat="treat", estimand="ATT", focal="b"),
    "multi_ATT_nonpair": dict(treat="treat", estimand="ATT", focal="b", pairwise=False),
    "bin_ATE": dict(treat="bin", estimand="ATE"),
    "bin_ATT": dict(treat="bin", estimand="ATT"),
    "bin_ATC": dict(treat="bin", estimand="ATC"),
    "bin_ATE_binstd": dict(treat="bin", estimand="ATE", binary="std"),
    "cont": dict(treat="dose"),
    "cont_binstd": dict(treat="dose", binary="std"),
}


def _compare(ours: pd.DataFrame, ref_rows, prefix="", sign=1.0):
    """Every reported statistic of every covariate. ``sign = -1`` reads
    the comparison the other way round (differences negate, variance
    ratios invert)."""
    n = 0
    for row in ref_rows:
        for col, mine in _COLS.items():
            key = prefix + col
            if key not in row or row[key] is None:
                continue
            got = float(ours.loc[row["var"], mine])
            want = float(row[key])
            if col.startswith("Diff"):
                got = abs(got) if prefix else sign * got
            elif col.startswith("V.Ratio") and sign < 0:
                got = 1.0 / got
            assert abs(got - want) <= 1e-9 * max(1.0, abs(want)), (
                row["var"],
                key,
                got,
                want,
            )
            n += 1
    return n


def _ours(name):
    kw = dict(_CASES[name])
    treat = kw.pop("treat")
    w = np.asarray(REF["balance"][name]["w"], dtype=float)
    return sp.bal_tab(DATA, treat, COV, weights=w, **kw)


def _pair(bt, label: str):
    """Our comparison for a cobalt label. Its focal comparisons are ours
    verbatim (focal minus the other group). Its other labels do not fix
    an orientation (on this fixture 'b vs. a' is b - a but 'c vs. a' is
    a - c, 'All vs. a' is a - All but 'All vs. c' is All - c), so those
    are matched as a set and the sign is read off the first covariate and
    then required of every other statistic of the comparison."""
    a, b = label.split(" vs. ")
    if a == "All":
        return bt.pairs[f"{b} vs. All"], "free"
    if label in bt.pairs:
        return bt.pairs[label], None
    return bt.pairs[f"{b} vs. {a}"], "free"


@pytest.mark.parametrize("name", sorted(_CASES))
def test_balance_matches_cobalt(name):
    rec = REF["balance"][name]
    bt = _ours(name)
    checked = 0
    if "pairs" in rec:
        for label, rows in rec["pairs"].items():
            ours, mode = _pair(bt, label)
            sign = 1.0
            if mode == "free":
                first = rows[0]
                sign = float(
                    np.sign(first["Diff.Un"] * ours.loc[first["var"], "diff_un"])
                )
            checked += _compare(ours, rows, sign=sign)
        checked += _compare(bt.table, rec["summary"], prefix="Max.")
    else:
        checked += _compare(bt.table, rec["summary"])
    assert checked >= 6


@pytest.mark.parametrize("name", sorted(_CASES))
def test_effective_sample_sizes_match_cobalt(name):
    rec = REF["balance"][name]
    bt = _ours(name)
    obs = {r["_row"]: r for r in rec["obs"]}
    adj = obs["Adjusted"]
    for lab, val in adj.items():
        if lab == "_row":
            continue
        key = {"Control": "0", "Treated": "1", "Total": "total"}.get(lab, lab)
        assert abs(bt.observations.loc["adjusted", key] - val) < 1e-9 * val


def test_weightit_result_feeds_bal_tab():
    W = sp.weightit(DATA, "treat", COV, method="ebal", estimand="ATT", focal="b")
    bt = W.balance()
    assert bt.summary_stats["estimand"] == "ATT"
    assert bt.summary_stats["s_d_denom"] == "treated"
    # entropy balancing: exact mean balance in every pair
    assert bt.table["diff_adj"].abs().max() < 1e-8


def test_continuous_ebal_zeroes_correlations():
    W = sp.weightit(DATA, "dose", COV, method="ebal")
    bt = W.balance()
    assert bt.table["corr_adj"].abs().max() < 1e-9
