"""Reference parity: ``sp.rdrobust``'s parameter surface vs rdrobust 4.0.0.

Companion to ``test_rdrobust_parity.py``, which covers the *default* path
(sharp RD, no covariates, no clusters, ``vce='nn'``) across a
``bwselect x p x kernel`` grid and is fully green at ~4e-12.

This file covers what that grid does not: ``covs``, ``fuzzy``, ``cluster``,
``deriv`` and ``vce``. Those paths still route around the CCT implementation
in ``rd/_cct_bandwidth.py``, so several assertions here are
``xfail(strict=True)`` -- the same device used for WP-1, which forces the
markers to be removed deliberately once each path lands rather than letting a
fix pass unnoticed.

Measured gaps against R (relative deviation, `rdsenate_params.csv`)::

    spec           h        conv     se_conv   robust    se_rob
    covs_p1        7.3e-03  9.5e-03  1.8e-03   9.7e-03   1.4e-03
    covs_p2        5.3e-03  1.5e-02  6.4e-03   1.6e-02   5.8e-03
    fuzzy_p1       1.2e-08  3.8e-14  1.1e-02   9.6e-03   2.5e-02
    fuzzy_p2       1.3e-08  2.9e-14  7.6e-03   1.6e-02   2.6e-02
    cluster_p1     3.8e-02  2.3e-03  4.2e-02   3.2e-03   2.9e-02
    cluster_p2     4.0e-02  1.2e-03  4.3e-02   7.8e-03   2.8e-02
    deriv1         6.4e-09  6.6e-14  1.5e-02   3.8e-02   7.6e-02
    covs_fuzzy     7.3e-03  2.5e-02  3.1e-02   2.2e-02   4.3e-02

Two things that read out of that table and are asserted below:

* **``fuzzy`` and ``deriv`` already get the correct bandwidth** (1e-8) and,
  for ``fuzzy``, the correct conventional point estimate (1e-14). The CCT
  cascade is bandwidth-only for those paths; what is missing is the
  variance, which still comes from the legacy refit.
* **``covs`` and ``cluster`` do not** (7e-3 and 4e-2 on ``h``). The cascade
  has no covariate-projection or cluster machinery, so the bandwidth itself
  is wrong before any estimate is formed.

``vce`` is not a parameter of ``sp.rdrobust`` at all; R exposes
``hc0``/``hc1``/``hc2``/``hc3``/``cr*``. That is an API gap rather than a
numerical one and is asserted separately.
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

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-6

_TODO_VAR = (
    "WP-2 remainder: the CCT variance path is gated to sharp RD with "
    "vce='nn'. fuzzy/covs/cluster/deriv still take their SEs from the "
    "legacy refit. See docs/rfc/rd_three_month_plan.md B.5."
)
_TODO_BW = (
    "WP-2 remainder: the CCT bandwidth cascade has no covariate-projection "
    "or cluster machinery, so h itself is off for covs/cluster. See "
    "docs/rfc/rd_three_month_plan.md B.5."
)


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rdrobust_params_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rdrobust_params_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def senate():
    return pd.read_csv(_FIX / "rdsenate_params.csv")


def _scalar(v):
    return float(np.ravel(v)[0])


# ── what already works: pin it so a future change cannot silently break it ──


@pytest.mark.parametrize("p", [1, 2])
def test_fuzzy_bandwidth_and_point_estimate_match_r(rjson, senate, p):
    """The fuzzy path already gets the CCT bandwidth and conventional LATE.

    h to ~1e-8 and the conventional coefficient to ~1e-14 -- the cascade is
    shared. Only the variance is outstanding.
    """
    ref = rjson[f"fuzzy_p{p}"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=p, fuzzy="treat")
    assert _scalar(res.model_info["bandwidth_h"]) == pytest.approx(
        ref["h_left"], rel=1e-6
    )
    assert float(res.detail["estimate"][0]) == pytest.approx(
        ref["coef_conventional"], rel=1e-8
    )


def test_deriv_bandwidth_and_point_estimate_match_r(rjson, senate):
    ref = rjson["deriv1"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=2, deriv=1)
    assert _scalar(res.model_info["bandwidth_h"]) == pytest.approx(
        ref["h_left"], rel=1e-6
    )
    assert float(res.detail["estimate"][0]) == pytest.approx(
        ref["coef_conventional"], rel=1e-8
    )


# ── bandwidth: covs / cluster are not yet in the cascade ────────────────────


@pytest.mark.xfail(strict=True, reason=_TODO_BW)
@pytest.mark.parametrize("p", [1, 2])
def test_covs_bandwidth_matches_r(rjson, senate, p):
    ref = rjson[f"covs_p{p}"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=p, covs=["cov1", "cov2"])
    assert _scalar(res.model_info["bandwidth_h"]) == pytest.approx(
        ref["h_left"], rel=RTOL
    )


@pytest.mark.xfail(strict=True, reason=_TODO_BW)
@pytest.mark.parametrize("p", [1, 2])
def test_cluster_bandwidth_matches_r(rjson, senate, p):
    ref = rjson[f"cluster_p{p}"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=p, cluster="clust")
    assert _scalar(res.model_info["bandwidth_h"]) == pytest.approx(
        ref["h_left"], rel=RTOL
    )


# ── point estimates ────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason=_TODO_BW)
@pytest.mark.parametrize("p", [1, 2])
def test_covs_conventional_matches_r(rjson, senate, p):
    ref = rjson[f"covs_p{p}"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=p, covs=["cov1", "cov2"])
    assert float(res.detail["estimate"][0]) == pytest.approx(
        ref["coef_conventional"], rel=RTOL
    )


@pytest.mark.xfail(strict=True, reason=_TODO_VAR)
@pytest.mark.parametrize("p", [1, 2])
def test_fuzzy_robust_coefficient_matches_r(rjson, senate, p):
    ref = rjson[f"fuzzy_p{p}"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=p, fuzzy="treat")
    assert float(res.detail["estimate"][1]) == pytest.approx(
        ref["coef_robust"], rel=RTOL
    )


# ── standard errors ────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason=_TODO_VAR)
@pytest.mark.parametrize(
    "spec,kw",
    [
        ("fuzzy_p1", dict(p=1, fuzzy="treat")),
        ("covs_p1", dict(p=1, covs=["cov1", "cov2"])),
        ("cluster_p1", dict(p=1, cluster="clust")),
    ],
)
def test_conventional_se_matches_r(rjson, senate, spec, kw):
    ref = rjson[spec]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, **kw)
    assert float(res.detail["se"][0]) == pytest.approx(ref["se_conventional"], rel=RTOL)


@pytest.mark.xfail(strict=True, reason=_TODO_VAR)
@pytest.mark.parametrize(
    "spec,kw",
    [
        ("fuzzy_p1", dict(p=1, fuzzy="treat")),
        ("covs_p1", dict(p=1, covs=["cov1", "cov2"])),
        ("cluster_p1", dict(p=1, cluster="clust")),
    ],
)
def test_robust_se_matches_r(rjson, senate, spec, kw):
    ref = rjson[spec]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, **kw)
    assert float(res.detail["se"][1]) == pytest.approx(ref["se_robust"], rel=RTOL)


def test_deriv_conventional_se_matches_r(rjson, senate):
    """deriv=1 now gets the CCT variance too.

    It was masked by a `np.math.factorial` call -- removed in numpy 2.0 --
    swallowed by a broad `except Exception` around the CCT path, which
    silently sent deriv back to the legacy refit. Narrowing that handler for
    the repo's broad-exception gate is what exposed it.
    """
    ref = rjson["deriv1"]
    res = sp.rdrobust(senate, y="vote", x="margin", c=0, p=2, deriv=1)
    assert float(res.detail["se"][0]) == pytest.approx(ref["se_conventional"], rel=RTOL)


# ── API gap: vce ───────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="WP-2 remainder: sp.rdrobust has no vce=")
@pytest.mark.parametrize("vce", ["hc0", "hc1", "hc2", "hc3"])
def test_vce_variants_are_accepted(senate, vce):
    """R exposes hc0-hc3 and cr*; sp.rdrobust exposes none of them.

    An R script that sets vce= does not port across, and there is no way to
    request anything other than the nearest-neighbour variance.
    """
    sp.rdrobust(senate, y="vote", x="margin", c=0, vce=vce)


# ── the gap is bounded: guard against it getting worse ──────────────────────


def test_parameter_surface_gap_does_not_regress(rjson, senate):
    """Every path is within 10% of R even where it is not yet exact.

    This is deliberately loose. Its job is to catch a *regression* -- if some
    future change pushes one of these paths from 4e-2 back to the 60%-scale
    errors the sharp path used to have, this fails even though the strict
    assertions above are already xfailed.
    """
    calls = {
        "covs_p1": dict(p=1, covs=["cov1", "cov2"]),
        "covs_p2": dict(p=2, covs=["cov1", "cov2"]),
        "fuzzy_p1": dict(p=1, fuzzy="treat"),
        "fuzzy_p2": dict(p=2, fuzzy="treat"),
        "cluster_p1": dict(p=1, cluster="clust"),
        "cluster_p2": dict(p=2, cluster="clust"),
        "deriv1": dict(p=2, deriv=1),
        "covs_fuzzy": dict(fuzzy="treat", covs=["cov1", "cov2"]),
    }
    bad = []
    for spec, kw in calls.items():
        ref = rjson[spec]
        res = sp.rdrobust(senate, y="vote", x="margin", c=0, **kw)
        for label, got, want in (
            ("h", _scalar(res.model_info["bandwidth_h"]), ref["h_left"]),
            ("conv", float(res.detail["estimate"][0]), ref["coef_conventional"]),
            ("se_conv", float(res.detail["se"][0]), ref["se_conventional"]),
            ("robust", float(res.detail["estimate"][1]), ref["coef_robust"]),
            ("se_rob", float(res.detail["se"][1]), ref["se_robust"]),
        ):
            rel = abs(got - want) / max(abs(want), 1e-12)
            if rel > 0.10:
                bad.append(f"{spec}.{label} rel={rel:.2e}")
    assert not bad, "parameter-surface gap widened:\n  " + "\n  ".join(bad)


# ── defect F: covs is a silent no-op ───────────────────────────────────────


def test_covs_actually_changes_the_estimate(senate):
    """``covs=`` must not be silently ignored.

    Regression guard for a defect introduced by the WP-2 bias-correction
    work and fixed in the same series: the CCT substitution was gated on
    ``fuzzy is None`` but not on ``covs``, so ``cct_bias_corrected`` -- which
    has no covariate machinery -- overwrote the covariate-adjusted estimate
    that ``_rd_estimate`` had already produced. The result was identical to
    the unadjusted call at 1e-12.

    The DGP makes the omission unmissable: ``z`` carries a coefficient of 2.0
    against residual noise of 0.3, so adjusting must move both the estimate
    and the SE. R: 0.2856 unadjusted -> 0.0444 adjusted (6.4x).
    """
    rng = np.random.default_rng(42)
    n = 3000
    x = rng.uniform(-1, 1, n)
    z = rng.normal(0, 1, n)
    y = 0.5 * x + 3.0 * (x >= 0) + 2.0 * z + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})

    plain = sp.rdrobust(df, y="y", x="x", c=0)
    adjusted = sp.rdrobust(df, y="y", x="x", c=0, covs=["z"])

    assert float(adjusted.detail["se"][0]) != pytest.approx(
        float(plain.detail["se"][0]), rel=1e-9
    ), (
        "covs= was ignored: adjusted and unadjusted SEs are identical "
        f"({float(plain.detail['se'][0]):.6f}). R reports 0.0444 adjusted "
        "against 0.2856 unadjusted on this DGP."
    )


def test_covs_reduces_se_when_covariate_is_predictive(senate):
    """With z explaining most of the residual variance, the SE must fall.

    R: 0.2856 -> 0.0444. Any implementation that adjusts at all will show a
    large drop; one that ignores covs shows none.
    """
    rng = np.random.default_rng(42)
    n = 3000
    x = rng.uniform(-1, 1, n)
    z = rng.normal(0, 1, n)
    y = 0.5 * x + 3.0 * (x >= 0) + 2.0 * z + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    plain = sp.rdrobust(df, y="y", x="x", c=0)
    adjusted = sp.rdrobust(df, y="y", x="x", c=0, covs=["z"])
    assert float(adjusted.detail["se"][0]) < 0.5 * float(plain.detail["se"][0])
