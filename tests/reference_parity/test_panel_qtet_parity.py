"""Reference parity: ``sp.panel_qtet`` vs R ``qte::panel.qtet`` 1.3.1 (WP-6).

Estimator
---------
Callaway & Li (2019) quantile treatment effect on the treated for panel data.
The counterfactual distribution ``F_{Y_t(0)|D=1}`` is recovered from a
distributional DiD plus a **copula stability** assumption, which is why the
estimator needs three periods where a mean DiD needs two.

Parity is EXACT here
--------------------
Unlike the Firpo and QDiD suites -- where R minimises a piecewise-linear
check function with ``stats::optimize`` and its answer on a plateau is an
optimiser artifact -- ``panel.qtet`` composes ordinary ``ecdf`` evaluations
and type-7 quantiles, both of which have exact numpy equivalents. A
hand-rolled R replication of the five algorithm steps reproduces
``panel.qtet`` at ``max |difference| = 0``, and this port matches R at
**6.8e-12** across all 19 quantiles on ``lalonde.psid.panel``.

So this file asserts machine-precision agreement, not a tolerance band.

What the ATT is, and why
------------------------
R reports a plain mean DiD as ``ate``, NOT ``mean(Y_t) - mean(cf)``. That is
the right choice: the mean does not need copula stability, distributional DiD
alone pins it. We match R.

The two can nonetheless diverge sharply, and that divergence is diagnostic.
On ``lalonde.psid.panel`` 131 of 185 treated units have ``re74 == 0``; they
all receive the same rank in step 1 and are mapped to the same ``t-1`` value,
so the rank map stops being measure-preserving. The counterfactual mean comes
out at 8,786 against a distributional-DiD value of 4,023 -- and the QTT curve
inherits that distortion even though the reported ATT does not.

R has the identical behaviour (hence our 1e-12 agreement) but never surfaces
it. ``sp.panel_qtet`` adds a coherence check that does, and a copula-stability
check on the untreated group, where both copulas are observed. Both are
asserted below, in both directions.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "qte_panel_qtet_nocov_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run the panel_qtet fixture generator first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(_FIX / "qte_lalonde_panel.csv")


def _fit_lalonde(panel, probs, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return sp.panel_qtet(
            panel,
            y="re",
            treat="treat",
            unit="id",
            time="year",
            t=1978,
            tmin1=1975,
            tmin2=1974,
            quantiles=probs,
            **kw,
        )


# ── A. exact parity with R ─────────────────────────────────────────── #


def test_matches_r_to_machine_precision(rjson, panel):
    probs = list(np.asarray(rjson["probs"], dtype=float))
    ref = np.asarray(rjson["pkg_qte"], dtype=float)
    res = _fit_lalonde(panel, probs, se="none")
    assert np.max(np.abs(res.effects - ref)) < 1e-8, (
        f"max |diff| = {np.max(np.abs(res.effects - ref))}\n"
        f"ours {np.round(res.effects, 4)}\nR    {np.round(ref, 4)}"
    )


def test_att_matches_r(rjson, panel):
    """R reports the plain mean DiD, which needs no copula assumption."""
    probs = list(np.asarray(rjson["probs"], dtype=float))
    res = _fit_lalonde(panel, probs, se="none")
    assert res.ate == pytest.approx(float(rjson["pkg_ate"]), abs=1e-6)


def test_group_sizes_match_r(rjson, panel):
    res = _fit_lalonde(panel, [0.5], se="none")
    assert res.model_info["n_treated"] == int(rjson["n_treated"])
    assert res.model_info["n_untreated"] == int(rjson["n_untreated"])


# ── B. known-truth recovery on non-degenerate DGPs ─────────────────── #


def _clean_panel(n_units: int, effect: str, seed: int = 1) -> pd.DataFrame:
    """``Y_it(0) = a_i + e_it`` with everything iid normal.

    Copula stability then holds exactly (the dependence between the change
    and the lagged level is the same in every period), and the outcome is
    continuous so the rank map is measure-preserving.

    ``'shift'`` -> ``Y(1) = Y(0) + 2``   => QTT(tau) = 2, flat.
    ``'scale'`` -> ``Y(1) = 2 * Y(0)``   => QTT(tau) = sqrt(2) * Phi^-1(tau),
    a fan that a mean estimator cannot see (its ATT is 0).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        a = rng.normal()
        treated = i >= n_units // 2
        y = [a + rng.normal() for _ in range(3)]
        if treated:
            y[2] = y[2] + 2.0 if effect == "shift" else 2.0 * y[2]
        for per, val in enumerate(y):
            rows.append((i, per, val, int(treated)))
    return pd.DataFrame(rows, columns=["id", "per", "y", "d"])


TAUS = [0.1, 0.25, 0.5, 0.75, 0.9]


def test_recovers_constant_shift():
    res = sp.panel_qtet(
        _clean_panel(40_000, "shift"),
        y="y",
        treat="d",
        unit="id",
        time="per",
        t=2,
        tmin1=1,
        tmin2=0,
        quantiles=TAUS,
        se="none",
    )
    assert np.max(np.abs(res.effects - 2.0)) < 0.06, res.effects


def test_recovers_the_quantile_fan():
    """Treatment doubles the outcome: QTT varies with tau and the ATT is ~0.

    This is the design a mean-based method is blind to, and the reason the
    estimator exists. A constant-shift test alone could not distinguish a
    working QTT estimator from one that returns the ATT at every quantile.
    """
    truth = stats.norm.ppf(TAUS) * np.sqrt(2.0)
    res = sp.panel_qtet(
        _clean_panel(40_000, "scale"),
        y="y",
        treat="d",
        unit="id",
        time="per",
        t=2,
        tmin1=1,
        tmin2=0,
        quantiles=TAUS,
        se="none",
    )
    assert (
        np.max(np.abs(res.effects - truth)) < 0.08
    ), f"estimated {np.round(res.effects, 3)} vs truth {np.round(truth, 3)}"
    assert np.all(np.diff(res.effects) > 0)  # a genuine fan
    assert res.effects[0] < -1.0 and res.effects[-1] > 1.0
    assert abs(res.ate) < 0.1  # the mean effect is zero: only the QTT sees it


# ── C. diagnostics fire correctly, in BOTH directions ──────────────── #


def test_coherence_check_passes_on_continuous_data():
    res = sp.panel_qtet(
        _clean_panel(8000, "shift"),
        y="y",
        treat="d",
        unit="id",
        time="per",
        t=2,
        tmin1=1,
        tmin2=0,
        quantiles=[0.5],
        se="none",
    )
    c = res.model_info["coherence_check"]
    assert c["means_agree"] is True
    assert c["mean_gap_relative"] < 0.05
    assert c["tie_fraction_max"] < 0.05


def test_coherence_check_catches_mass_point_distortion(panel):
    """The lalonde case: 131/185 treated share re74 == 0.

    The rank map collapses them onto one value, so the counterfactual mean
    (8,786) departs from the distributional-DiD value (4,023). R produces
    the same distorted curve without saying so.
    """
    with pytest.warns(UserWarning, match="copula construction looks distorted"):
        res = sp.panel_qtet(
            panel,
            y="re",
            treat="treat",
            unit="id",
            time="year",
            t=1978,
            tmin1=1975,
            tmin2=1974,
            quantiles=[0.5],
            se="none",
        )
    c = res.model_info["coherence_check"]
    assert c["means_agree"] is False
    assert c["tie_fraction_max"] > 0.5
    assert c["counterfactual_mean"] > c["did_counterfactual_mean"]


def test_no_distortion_warning_on_clean_data():
    """The guard must not cry wolf -- otherwise it carries no information."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sp.panel_qtet(
            _clean_panel(8000, "shift"),
            y="y",
            treat="d",
            unit="id",
            time="per",
            t=2,
            tmin1=1,
            tmin2=0,
            quantiles=[0.5],
            se="none",
        )


def test_copula_check_reports_stability_on_iid_data():
    res = sp.panel_qtet(
        _clean_panel(8000, "shift"),
        y="y",
        treat="d",
        unit="id",
        time="per",
        t=2,
        tmin1=1,
        tmin2=0,
        quantiles=[0.5],
        se="none",
    )
    k = res.model_info["copula_check"]
    # iid errors => the same dependence in every period.
    assert abs(k["difference"]) < 0.1, k


# ── D. inference ───────────────────────────────────────────────────── #


def test_bootstrap_se_positive_and_reproducible():
    df = _clean_panel(1500, "shift", seed=3)
    kw = dict(
        y="y",
        treat="d",
        unit="id",
        time="per",
        t=2,
        tmin1=1,
        tmin2=0,
        quantiles=[0.25, 0.5, 0.75],
        n_boot=40,
    )
    a = sp.panel_qtet(df, seed=1, **kw)
    b = sp.panel_qtet(df, seed=1, **kw)
    c = sp.panel_qtet(df, seed=2, **kw)
    assert np.all(np.isfinite(a.se)) and np.all(a.se > 0)
    np.testing.assert_allclose(a.se, b.se)
    assert not np.allclose(a.se, c.se), "seed had no effect"


def test_se_none_returns_nan_not_a_placeholder():
    res = sp.panel_qtet(
        _clean_panel(600, "shift"),
        y="y",
        treat="d",
        unit="id",
        time="per",
        t=2,
        tmin1=1,
        tmin2=0,
        quantiles=[0.5],
        se="none",
    )
    assert np.all(np.isnan(res.se))
    assert np.all(np.isnan(res.ci_lower)) and np.all(np.isnan(res.ci_upper))


# ── E. loud failure on bad input ───────────────────────────────────── #


def test_unbalanced_panel_warns_and_drops():
    df = _clean_panel(600, "shift", seed=5)
    df = df.drop(df[(df["id"] == 0) & (df["per"] == 0)].index)  # hole
    with pytest.warns(UserWarning, match="not observed in all three periods"):
        sp.panel_qtet(
            df,
            y="y",
            treat="d",
            unit="id",
            time="per",
            t=2,
            tmin1=1,
            tmin2=0,
            quantiles=[0.5],
            se="none",
        )


def test_duplicate_periods_raise():
    df = _clean_panel(300, "shift")
    with pytest.raises(ValueError, match="three distinct periods"):
        sp.panel_qtet(
            df,
            y="y",
            treat="d",
            unit="id",
            time="per",
            t=2,
            tmin1=2,
            tmin2=0,
            quantiles=[0.5],
            se="none",
        )


def test_bad_se_raises():
    df = _clean_panel(300, "shift")
    with pytest.raises(ValueError, match="se must be"):
        sp.panel_qtet(
            df,
            y="y",
            treat="d",
            unit="id",
            time="per",
            t=2,
            tmin1=1,
            tmin2=0,
            quantiles=[0.5],
            se="analytic",
        )


def test_bad_quantiles_raise():
    df = _clean_panel(300, "shift")
    with pytest.raises(ValueError, match="strictly inside"):
        sp.panel_qtet(
            df,
            y="y",
            treat="d",
            unit="id",
            time="per",
            t=2,
            tmin1=1,
            tmin2=0,
            quantiles=[0.0, 0.5],
            se="none",
        )


def test_result_contract():
    res = sp.panel_qtet(
        _clean_panel(1200, "shift"),
        y="y",
        treat="d",
        unit="id",
        time="per",
        t=2,
        tmin1=1,
        tmin2=0,
        quantiles=[0.25, 0.5, 0.75],
        n_boot=20,
    )
    assert isinstance(res, sp.QTEResult)
    assert "Callaway" in res.method
    frame = res.to_frame()
    assert list(frame.columns)[:5] == ["quantile", "qte", "se", "ci_lower", "ci_upper"]
    assert len(res.model_info["counterfactual"]) == res.model_info["n_treated"]
