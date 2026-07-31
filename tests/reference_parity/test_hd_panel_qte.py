"""Panel QTE correctness for ``sp.qte_hd_panel`` (WP-5).

What this file guards
---------------------
Up to and including 1.20.0 ``qte_hd_panel`` within-demeaned ``Y`` and ``D``
by unit and time and then ran quantile regression on the demeaned data.
Quantile regression is **not** invariant to the within transformation, so
the estimator was consistent only under a pure location shift -- precisely
the case in which a QTE is indistinguishable from an ATE.

Measured on a scale-shift design ``Y_it = u_i + (1 + d_it) e_it``,
``e ~ N(0,1)``, whose true ``QTE(tau) = Phi^-1(tau)`` (a *fan*), the old code
returned, at T = 4, n = 2000:

    tau      0.10    0.25    0.50    0.75    0.90
    truth  -1.282  -0.674   0.000   0.674   1.282
    old    -0.707  -0.391  -0.148   0.299   1.108      <- fan flattened ~45%

Two further defects: controls were LASSO-selected from the ``Y ~ X``
equation only (not double selection), and two fallback paths returned a
**hardcoded ``se = 0.1``** which was then used to build confidence
intervals.

Anchors
-------
A. **Fan recovery.** Canay (2011) recovers ``Phi^-1(tau)`` on the design the
   old estimator flattened, and does materially better than the old numbers
   even at short T.
B. **Consistency in T.** Canay is a large-``T`` estimator; its error must
   *shrink* as T grows. This separates "correct estimator, finite-T bias"
   from "wrong estimator".
C. **No fabricated standard errors** anywhere, and a loud failure when the
   quantile-regression backend is unavailable.
D. **Double selection** keeps a covariate that predicts treatment but not
   the outcome -- the one single-equation LASSO drops.
E. **Constant-effect DGP still recovered** (no regression on the case the
   old code did handle).
F. ``seed`` is honoured (it was accepted and discarded pre-1.21).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

TAUS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
TRUTH_FAN = stats.norm.ppf(TAUS)  # from scipy, not from the estimator

# What the pre-1.21 within-demeaning estimator produced on the T=4 design.
OLD_FLATTENED = np.array([-0.707, -0.391, -0.148, 0.299, 1.108])

COVS = ["x1", "x2", "x3"]


def _scale_shift(seed: int, n_units: int, T: int) -> pd.DataFrame:
    """Y_it = u_i + (1 + d_it) e_it  =>  QTE(tau) = Phi^-1(tau), a fan.

    The individual effect u_i IS a pure location shift, so Canay's
    identifying assumption holds; what the treatment changes is the SCALE,
    which is invisible to any mean-based method.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        ui = rng.normal(0, 2.0)
        treated = u >= n_units // 2
        for t in range(T):
            d = 1.0 if (treated and t >= T // 2) else 0.0
            rows.append(
                (
                    u,
                    t,
                    ui + (1.0 + d) * rng.normal(),
                    d,
                    rng.normal(),
                    rng.normal(),
                    rng.normal(),
                )
            )
    return pd.DataFrame(rows, columns=["unit", "time", "y", "d"] + COVS)


def _fit(df, **kw):
    kw.setdefault("quantiles", TAUS)
    kw.setdefault("se", "none")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return sp.qte_hd_panel(
            df, y="y", treat="d", unit="unit", time="time", covariates=COVS, **kw
        )


# ── A. fan recovery ────────────────────────────────────────────────── #


def test_recovers_the_fan_at_long_T():
    """With T = 50 Canay must track Phi^-1(tau) across the whole curve."""
    res = _fit(_scale_shift(11, 200, 50), method="canay")
    assert (
        np.max(np.abs(res.qte - TRUTH_FAN)) < 0.20
    ), f"estimated {np.round(res.qte, 3)} vs truth {np.round(TRUTH_FAN, 3)}"
    # It must actually be a fan, not a flat line.
    assert np.all(np.diff(res.qte) > 0)
    assert res.qte[0] < -0.9 and res.qte[-1] > 0.9
    assert abs(res.qte[2]) < 0.20  # zero at the median


def test_beats_the_old_flattened_estimator_at_short_T():
    """Even at T = 4, where Canay carries real finite-T bias, the rebuilt
    estimator must be closer to the truth than the old demeaning path."""
    res = _fit(_scale_shift(11, 500, 4), method="canay")
    new_err = float(np.mean(np.abs(res.qte - TRUTH_FAN)))
    old_err = float(np.mean(np.abs(OLD_FLATTENED - TRUTH_FAN)))
    assert new_err < old_err, (
        f"mean |error|: new {new_err:.3f} vs old {old_err:.3f} — "
        "the rebuild did not improve on within-demeaning"
    )
    # Specifically, the deep-tail attenuation that motivated the rebuild.
    assert res.qte[0] < OLD_FLATTENED[0], (res.qte[0], OLD_FLATTENED[0])


# ── B. consistency in T ────────────────────────────────────────────── #


def test_error_shrinks_as_panel_lengthens():
    """Canay is a large-T estimator: its bias must vanish with T.

    A wrong estimator (like within-demeaning) has bias that does NOT shrink
    with T, so this is the test that distinguishes the two.
    """
    errs = []
    for T in (10, 50):
        res = _fit(_scale_shift(11, 200, T), method="canay")
        errs.append(float(np.mean(np.abs(res.qte - TRUTH_FAN))))
    assert errs[1] < errs[0], f"error did not shrink with T: {errs}"


# ── C. no fabricated standard errors ───────────────────────────────── #


def test_se_none_gives_nan_not_a_number():
    """``se='none'`` must yield NaN, never a placeholder like 0.1."""
    res = _fit(_scale_shift(1, 60, 8), method="canay", se="none")
    assert np.all(np.isnan(res.se))
    assert np.all(np.isnan(res.ci_low)) and np.all(np.isnan(res.ci_high))


def test_no_hardcoded_point_one_standard_error():
    """Regression guard on the literal defect: ``se_list.append(0.1)``.

    The old code emitted exactly 0.1 for every quantile whenever the
    quantile-regression backend raised. Real bootstrap SEs are never all
    exactly 0.1.
    """
    res = _fit(
        _scale_shift(2, 60, 8), method="canay", se="bootstrap", n_boot=40, seed=0
    )
    assert not np.allclose(res.se, 0.1), f"fabricated SEs returned: {res.se}"
    assert np.all(np.isfinite(res.se)) and np.all(res.se > 0)


def test_bootstrap_se_is_reproducible_and_seed_is_used():
    """``seed`` was accepted and assigned to ``_`` pre-1.21."""
    df = _scale_shift(3, 60, 8)
    a = _fit(df, method="canay", se="bootstrap", n_boot=30, seed=1)
    b = _fit(df, method="canay", se="bootstrap", n_boot=30, seed=1)
    c = _fit(df, method="canay", se="bootstrap", n_boot=30, seed=2)
    np.testing.assert_allclose(a.se, b.se)
    assert not np.allclose(a.se, c.se), "seed had no effect on the bootstrap"


def test_bad_method_and_bad_se_raise():
    df = _scale_shift(4, 30, 6)
    with pytest.raises(ValueError, match="method must be"):
        _fit(df, method="nope")
    with pytest.raises(ValueError, match="se must be"):
        _fit(df, method="canay", se="analytic")


def test_short_panel_warns():
    """Canay's finite-T bias must be surfaced, not hidden."""
    df = _scale_shift(5, 60, 4)
    with pytest.warns(UserWarning, match="average T"):
        sp.qte_hd_panel(
            df,
            y="y",
            treat="d",
            unit="unit",
            time="time",
            covariates=COVS,
            quantiles=TAUS,
            method="canay",
            se="none",
        )


def test_location_shift_assumption_is_recorded():
    res = _fit(_scale_shift(6, 60, 12), method="canay")
    joined = " ".join(res.diagnostics["warnings"])
    assert "LOCATION" in joined or "location" in joined
    assert res.diagnostics["avg_T"] == pytest.approx(12.0)


# ── D. double selection ────────────────────────────────────────────── #


def test_double_selection_keeps_a_treatment_only_predictor():
    """A covariate that drives D but not Y must survive selection.

    Single-equation LASSO on ``Y ~ X`` drops it -- that is the confounding
    channel Belloni-Chernozhukov-Hansen double selection exists to close.
    """
    rng = np.random.default_rng(9)
    n_units, T = 150, 8
    rows = []
    for u in range(n_units):
        ui = rng.normal(0, 1.0)
        # xd drives treatment assignment; it has NO direct outcome effect.
        xd = rng.normal()
        for t in range(T):
            d = 1.0 if (xd + rng.normal(0, 0.3) > 0 and t >= T // 2) else 0.0
            xy = rng.normal()
            noise = rng.normal()
            y = ui + 1.0 * d + 2.0 * xy + 0.9 * xd + noise
            rows.append((u, t, y, d, xy, xd, rng.normal()))
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "d", "xy", "xd", "junk"])

    res = sp.qte_hd_panel(
        df,
        y="y",
        treat="d",
        unit="unit",
        time="time",
        covariates=["xy", "xd", "junk"],
        quantiles=np.array([0.5]),
        method="canay",
        se="none",
    )
    assert "xd" in res.selected_controls, res.selected_controls


def test_selection_never_returns_empty():
    """If the LASSO zeroes everything, fall back to the full set rather
    than silently estimating with no controls."""
    rng = np.random.default_rng(10)
    n_units, T = 40, 6
    rows = []
    for u in range(n_units):
        for t in range(T):
            rows.append(
                (
                    u,
                    t,
                    rng.normal(),
                    float(t >= 3 and u >= 20),
                    rng.normal(),
                    rng.normal(),
                    rng.normal(),
                )
            )
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "d"] + COVS)
    res = sp.qte_hd_panel(
        df,
        y="y",
        treat="d",
        unit="unit",
        time="time",
        covariates=COVS,
        quantiles=np.array([0.5]),
        method="canay",
        se="none",
        lasso_alpha=1e6,  # absurd penalty: nothing can survive
    )
    assert len(res.selected_controls) == len(COVS)


# ── E. no regression on the constant-effect case ───────────────────── #


def _constant_effect_panel() -> pd.DataFrame:
    """Location-shift DGP with genuine unit effects; true QTE == 1.2 flat."""
    rng = np.random.default_rng(42)
    rows = []
    for u in range(120):
        ui = rng.normal(0, 0.5)
        treated = u >= 60
        for t in range(10):
            d = 1.0 if (treated and t >= 5) else 0.0
            x = rng.normal(0, 1, 3)
            y = 1.0 + 1.2 * d + 0.5 * x[0] + ui + rng.normal(0, 1)
            rows.append((u, t, y, d, *x))
    return pd.DataFrame(rows, columns=["unit", "time", "y", "d"] + COVS)


@pytest.mark.parametrize("method", ["canay", "dummy_fe"])
def test_constant_effect_recovered_by_fe_methods(method):
    """The case the old code DID handle; the rebuild must not break it."""
    res = _fit(_constant_effect_panel(), method=method)
    assert np.all(np.abs(res.qte - 1.2) < 0.35), (method, res.qte)


def test_pooled_is_biased_when_unit_effects_exist():
    """``method='pooled'`` ignores individual effects and must show it.

    This DGP has ``u_i ~ N(0, 0.5)``. Asserting that pooled is *worse* than
    the FE methods is what makes the three options meaningfully distinct --
    if every method agreed everywhere, the FE machinery would be untested.
    """
    df = _constant_effect_panel()
    pooled_err = float(np.mean(np.abs(_fit(df, method="pooled").qte - 1.2)))
    canay_err = float(np.mean(np.abs(_fit(df, method="canay").qte - 1.2)))
    assert canay_err < pooled_err, (canay_err, pooled_err)


def test_result_contract():
    res = _fit(
        _scale_shift(7, 60, 10), method="canay", se="bootstrap", n_boot=25, seed=0
    )
    frame = res.to_frame()
    assert list(frame.columns) == ["quantile", "qte", "se", "ci_low", "ci_high"]
    assert len(frame) == len(TAUS)
    out = res.summary()
    assert "Panel QTE" in out
    assert res.n_units == 60 and res.n_periods == 10
