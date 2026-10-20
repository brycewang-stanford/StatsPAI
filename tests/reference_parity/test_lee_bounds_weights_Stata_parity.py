"""``sp.lee_bounds(weights=...)`` against Stata ``leebounds [pw=]``.

Reference: Stata 18, ``leebounds y d [pw=w], select(s) cieffect``
(Tauchmann, SSC v1.5) on ``_fixtures/leebounds_weights.csv`` from
``_fixtures/_generate_leebounds_weights_stata.do``: continuous and tied
integer outcomes, either arm more retained. ``trimming='leebounds'``
reproduces ``leebounds``' output, which behaves as if the weights were held
in single precision and the thresholds rounded as macro text (black box,
``docs/dev/cleanroom/leebounds.md``); bounds are held to 1e-12 and SEs to
1e-8.

One SE is a documented difference. Without a tie at the threshold, Stata's
SE of its "upper" trimmed mean (the reported upper bound when the treated
arm is trimmed, the lower bound when the control arm is) equals the value
obtained with the untrimmed arm's variance term left unweighted, while
every other term, including the same term of the other bound, matches the
weighted formula. We weight it; the test rebuilds Stata's number by
swapping in the unweighted variance, so the gap is explained to 1e-8.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bounds._lee_weighted import _iw_var

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads(
        (_FIX / "leebounds_weights_stata.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "leebounds_weights.csv", float_precision="round_trip")


def _stata_unweighted_vc_variance(d, mi):
    """Our variance of the bound leebounds computes with an unweighted
    control-arm variance, rebuilt with that variance."""
    w32 = d["w"].to_numpy(dtype=np.float32).astype(float)
    w = (w32 / w32.mean()).astype(np.float32).astype(float)
    D, S = d["d"].to_numpy(float), d["s"].to_numpy(float)
    treated_trimmed = mi["retention_treated"] >= mi["retention_control"]
    t = D if treated_trimmed else 1 - D
    unt = (S == 1) & (t == 0)
    esnt = np.sum(w * unt) / np.sum(w)
    yc = d["y"].to_numpy(float)[unt]
    delta = (np.var(yc, ddof=1) - _iw_var(yc, w[unt])) / esnt / round(np.sum(w))
    key = "se_upper" if treated_trimmed else "se_lower"
    return key, np.sqrt(mi[key] ** 2 + delta)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_weighted_bounds_match_leebounds(ref, data, k):
    R = ref[f"{k}_cie"]
    d = data[data["sample"] == k]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.lee_bounds(
            d,
            y="y",
            treat="d",
            selection="s",
            weights="w",
            trimming="leebounds",
            se_method="analytic",
        )
    mi = r.model_info
    assert mi["lower_bound"] == pytest.approx(R["lower"], rel=1e-12, abs=1e-12)
    assert mi["upper_bound"] == pytest.approx(R["upper"], rel=1e-12, abs=1e-12)
    assert mi["trimming_fraction"] == pytest.approx(R["trim"], rel=1e-12)
    se = {"se_lower": mi["se_lower"], "se_upper": mi["se_upper"]}
    treated_trimmed = mi["retention_treated"] >= mi["retention_control"]
    quirk = "se_upper" if treated_trimmed else "se_lower"
    if not mi["threshold_tie"][quirk[3:]]:
        # no tie at that threshold: leebounds' unweighted variance term
        key, rebuilt = _stata_unweighted_vc_variance(d, mi)
        assert rebuilt == pytest.approx(R[key], rel=1e-8)
        se.pop(key)
    for key, value in se.items():
        assert value == pytest.approx(R[key], rel=1e-8)


def test_unit_weights_reproduce_the_unweighted_estimator(data):
    d = data[data["sample"] == 3].assign(one=1.0)
    kw = dict(
        y="y", treat="d", selection="s", trimming="leebounds", se_method="analytic"
    )
    a = sp.lee_bounds(d, weights="one", **kw).model_info
    b = sp.lee_bounds(d, **kw).model_info
    for key in ("lower_bound", "upper_bound", "se_lower", "se_upper"):
        assert a[key] == pytest.approx(b[key], rel=1e-12)


def test_integer_weights_equal_replicated_rows_for_exact_trimming(data):
    """Bounds are functionals of the weighted distribution: weight 2 on a
    row equals duplicating it."""
    d = data[data["sample"] == 4].copy()
    d["k"] = np.where(np.arange(len(d)) % 3 == 0, 2.0, 1.0)
    kw = dict(y="y", treat="d", selection="s", trimming="exact", n_bootstrap=20)
    a = sp.lee_bounds(d, weights="k", **kw).model_info
    rep = d.loc[d.index.repeat(d["k"].astype(int))]
    b = sp.lee_bounds(rep, **kw).model_info
    assert a["lower_bound"] == pytest.approx(b["lower_bound"], rel=1e-12)
    assert a["upper_bound"] == pytest.approx(b["upper_bound"], rel=1e-12)


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(trimming="quantile"), "trimming"),
        (dict(trimming="leebounds", covariates=["x"]), "covariate"),
    ],
)
def test_unsupported_combinations_are_refused(data, kw, match):
    d = data[data["sample"] == 1]
    with pytest.raises(sp.MethodIncompatibility, match=match):
        sp.lee_bounds(d, y="y", treat="d", selection="s", weights="w", **kw)


def test_negative_weight_is_refused(data):
    d = data[data["sample"] == 1].copy()
    d.iloc[0, d.columns.get_loc("w")] = -1.0
    with pytest.raises(sp.MethodIncompatibility, match="non-negative"):
        sp.lee_bounds(d, y="y", treat="d", selection="s", weights="w", trimming="exact")


@pytest.mark.parametrize("trimming", ["exact", "leebounds"])
def test_zero_weight_rows_are_outside_the_sample(trimming):
    # 2026-10-20 probe: zero-weight rows stayed in the mean-one rescaling
    # and in N, so se_lower was sqrt(1700/2000) of its value with the rows
    # dropped and n_obs counted them.
    rng = np.random.default_rng(1)
    n = 2000
    d = rng.binomial(1, 0.5, n)
    u = rng.normal(size=n)
    s = (u + 0.5 * d + rng.normal(size=n) > 0).astype(int)
    y = np.where(s == 1, 1 + d + u + rng.normal(size=n), np.nan)
    df = pd.DataFrame(
        {"y": y, "d": d, "s": s, "w": np.where(np.arange(n) < 300, 0.0, 1.0)}
    )
    with pytest.warns(UserWarning, match="zero weight"):
        a = sp.lee_bounds(
            df, "y", "d", "s", trimming=trimming, weights="w", se_method="analytic"
        )
    b = sp.lee_bounds(
        df.iloc[300:],
        "y",
        "d",
        "s",
        trimming=trimming,
        weights="w",
        se_method="analytic",
    )
    assert a.n_obs == b.n_obs
    for k in ("lower_bound", "upper_bound", "se_lower", "se_upper"):
        assert a.model_info[k] == pytest.approx(b.model_info[k], rel=1e-12), k
