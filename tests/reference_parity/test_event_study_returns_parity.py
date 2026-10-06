"""``sp.abnormal_returns`` against Stata's ``estudy`` on one committed file.

The data are synthetic: twelve securities over 320 business days, one
event each (``_fixtures/_generate_event_study_returns_data.py``). The
reference is ``event_study_returns_Stata.csv`` and
``event_study_returns_Stata_ar.csv``, written by Stata 18 with ``estudy``
(Pacicco, Vena and Venegoni 2018) for four models of normal returns, five
tests and three event windows
(``_generate_event_study_returns_Stata.do``).

What is compared, and how closely.

* Abnormal returns and CARs of each security, all four models:
  ``FLOAT`` (5e-6 relative). ``estudy`` keeps its working variables in
  single precision.
* The standard deviation of a CAR. For the market-adjusted and
  mean-adjusted models the two agree (``FLOAT``). For the market model
  ``estudy`` divides the residual sum of squares by ``n - 1`` where the
  forecast-error variance has ``n - 2``, and scales the market term by
  ``(n - 1) / n``. For the factor model it uses ``L * RSS / (n - 1)``,
  with no term for the error in the estimated coefficients, which puts
  its standard deviations 1 to 6 percent below the forecast-error ones.
  Both of its numbers are rebuilt from ours to ``FLOAT``.
* The tests of the mean CAR. Given the same standardised CARs the
  aggregation is the same arithmetic (``EXACT``, 1e-9), and where the
  standard deviations agree (two models) the whole pipeline is compared
  end to end.
* ``estudy``'s group row is on a log-return footing: each day it takes the
  log of the mean of ``exp(AR)`` across securities and sums those, where
  the mean CAR here is the mean of the CARs. Rebuilt from our abnormal
  returns it matches to ``FLOAT``; the two differ by the cross-sectional
  variance of the abnormal returns (3 to 7 percent of the CAAR on this
  file).
* A missing return in the estimation window: ``estudy`` keeps the market's
  mean and variance over every estimation day while the residual variance
  and ``n`` use the days the security traded. Rebuilt to ``FLOAT``.
* The generalized rank test. ``estudy`` scales ranks by ``L1 + 1`` and
  uses ``L1`` for the number of observations, where there are ``T = L1 +
  1`` of them (the estimation days and the cumulated event day) and the
  scaling that centres the ranks is ``T + 1``. With its constants the
  statistic here reproduces its number to 1e-6; with the right ones it is
  1.5 percent smaller.
* The signed-rank test is compared with Stata's own ``signrank`` on the
  CARs (``EXACT``). ``estudy``'s Wilcoxon row ranks each CAR within its
  own security's time series, not across securities, and reports
  statistics no signed-rank test can produce; it is rebuilt here to show
  what it computes, and is not a reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.timeseries.abnormal_returns import _car_tests

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-9
FLOAT = 5e-6
WINDOWS = {"m1_p1": (-1, 1), "0_0": (0, 0), "m5_p5": (-5, 5)}
MODELS = {
    "SIM": dict(model="market", market="mkt"),
    "MAM": dict(model="market_adjusted", market="mkt"),
    "HMM": dict(model="mean_adjusted"),
    "MFM": dict(model="factor", factors=["mkt", "smb", "hml"]),
}
TESTS = {"patell": "Patell", "bmp": "BMP", "adj_patell": "ADJPatell", "kp": "KP"}
EST = (-200, -11)


def rel(got, ref) -> float:
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return pd.read_csv(FIX / "event_study_returns.csv")


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return pd.read_csv(FIX / "event_study_events.csv")


@pytest.fixture(scope="module")
def stata() -> pd.Series:
    table = pd.read_csv(FIX / "event_study_returns_Stata.csv")
    return table.set_index(["model", "test", "window", "security", "term"])["value"]


def fit(returns, events, model, window, **kwargs):
    res = sp.abnormal_returns(
        returns,
        events,
        event_window=WINDOWS[window],
        estimation_window=EST,
        **MODELS[model],
        **kwargs,
    )
    return res, res.events.set_index("id").loc[list(events["id"])]


def reference(stata, model, window, ids, term, test="Norm"):
    return np.array([stata[model, test, window, i, term] for i in ids])


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("window", list(WINDOWS))
def test_cumulative_abnormal_returns_match_estudy(
    returns, events, stata, model, window
):
    _, ours = fit(returns, events, model, window)
    ids = list(events["id"])
    assert rel(ours["car"], reference(stata, model, window, ids, "car")) < FLOAT


def test_abnormal_return_series_match_estudy(returns, events):
    ar = pd.read_csv(FIX / "event_study_returns_Stata_ar.csv")
    res, _ = fit(returns, events, "SIM", "m5_p5")
    ours = res.ar.set_axis(res.events["id"].to_numpy(), axis=1)[list(events["id"])]
    # estudy stacks the series in event time; day 0 is row 289 of its matrix
    theirs = ar.loc[284:294, list(events["id"])].to_numpy()
    assert np.max(np.abs(ours.to_numpy() - theirs)) < 1e-8


@pytest.mark.parametrize("model", ["MAM", "HMM"])
@pytest.mark.parametrize("window", list(WINDOWS))
def test_standard_deviations_match_estudy_where_the_conventions_agree(
    returns, events, stata, model, window
):
    _, ours = fit(returns, events, model, window)
    ids = list(events["id"])
    assert rel(ours["se"], reference(stata, model, window, ids, "sd")) < FLOAT


@pytest.mark.parametrize("window", list(WINDOWS))
def test_market_model_standard_deviation_is_estudys_up_to_a_located_rule(
    returns, events, stata, window
):
    """Ours: s2_{n-2} [L + L^2/n + (sum dx)^2 / Sxx], the forecast-error
    variance. estudy: s2_{n-1} [L + L^2/n + (sum dx)^2 (n-1) / (n Sxx)]."""
    res, ours = fit(returns, events, "SIM", window)
    lo, hi = WINDOWS[window]
    length = hi - lo + 1
    rebuilt = []
    for unit, row in ours.iterrows():
        sec = returns[returns["id"] == unit].reset_index(drop=True)
        pos = sec.index[pd.to_datetime(sec["date"]) == row["event_day"]][0]
        est = sec.iloc[pos + EST[0] : pos + EST[1] + 1].dropna(subset=["ret"])
        n = len(est)
        assert n == row["n_est"]
        x = est["mkt"].to_numpy()
        dx = sec["mkt"].to_numpy()[pos + lo : pos + hi + 1] - x.mean()
        sxx = np.sum((x - x.mean()) ** 2)
        # our variance is the textbook one
        textbook = row["resid_var"] * (length + length**2 / n + dx.sum() ** 2 / sxx)
        assert rel(row["se"] ** 2, textbook) < 1e-10
        s2_n1 = row["resid_var"] * (n - 2) / (n - 1)
        rebuilt.append(
            np.sqrt(
                s2_n1 * (length + length**2 / n + dx.sum() ** 2 * (n - 1) / (n * sxx))
            )
        )
    theirs = reference(stata, "SIM", window, list(ours.index), "sd")
    # s03 has one missing return in its estimation window. There estudy
    # keeps the market's mean and variance over all 190 days while the
    # residual variance and n use the 189 the security traded.
    complete = (ours["n_est"] == EST[1] - EST[0] + 1).to_numpy()
    assert complete.sum() == 11
    assert rel(np.asarray(rebuilt)[complete], theirs[complete]) < FLOAT
    unit = ours.index[~complete][0]
    row = ours.loc[unit]
    sec = returns[returns["id"] == unit].reset_index(drop=True)
    pos = sec.index[pd.to_datetime(sec["date"]) == row["event_day"]][0]
    x_all = sec["mkt"].to_numpy()[pos + EST[0] : pos + EST[1] + 1]
    dx = sec["mkt"].to_numpy()[pos + lo : pos + hi + 1] - x_all.mean()
    n = row["n_est"]
    s2_n1 = row["resid_var"] * (n - 2) / (n - 1)
    mixed = np.sqrt(
        s2_n1
        * (
            length
            + length**2 / n
            + dx.sum() ** 2 / (np.var(x_all, ddof=1) * len(x_all))
        )
    )
    assert rel(mixed, theirs[~complete][0]) < FLOAT
    # the two differ by about 1 / (2n)
    gap = ours["se"].to_numpy() / theirs - 1
    assert (gap > 0).all() and gap.max() < 1.5 / (2 * ours["n_est"].min())


@pytest.mark.parametrize("window", list(WINDOWS))
def test_factor_model_standard_deviation_is_estudys_up_to_a_located_rule(
    returns, events, stata, window
):
    """estudy: L * RSS / (n - 1). Ours adds the variance of the estimated
    coefficients and divides the residual sum of squares by n - 4."""
    _, ours = fit(returns, events, "MFM", window)
    lo, hi = WINDOWS[window]
    length = hi - lo + 1
    n = ours["n_est"].to_numpy(float)
    rebuilt = np.sqrt(length * ours["resid_var"].to_numpy() * (n - 4) / (n - 1))
    theirs = reference(stata, "MFM", window, list(ours.index), "sd")
    assert rel(rebuilt, theirs) < FLOAT
    gap = ours["se"].to_numpy() / theirs - 1
    assert (gap > 0.005).all() and gap.max() < 0.07


@pytest.mark.parametrize("window", list(WINDOWS))
def test_p_values_of_single_cars_are_t_with_residual_degrees_of_freedom(
    returns, events, stata, window
):
    from scipy import stats

    _, ours = fit(returns, events, "SIM", window)
    ids = list(ours.index)
    z = reference(stata, "SIM", window, ids, "stat")
    theirs = reference(stata, "SIM", window, ids, "pv")
    assert rel(2 * stats.t.sf(np.abs(z), ours["n_est"].to_numpy() - 2), theirs) < EXACT


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("window", list(WINDOWS))
def test_tests_of_the_mean_car_given_estudys_standardised_cars(
    returns, events, stata, model, window
):
    """Patell, BMP and the two Kolari-Pynnonen adjustments are the same
    arithmetic once the standardised CARs are the same."""
    res, ours = fit(returns, events, model, window, correlation="event")
    ids = list(ours.index)
    car = reference(stata, model, window, ids, "car")
    z = car / reference(stata, model, window, ids, "sd")
    # estudy's degrees of freedom: M minus the regression coefficients,
    # and M - 2 for the two models that estimate fewer than two
    dof = ours["n_est"].to_numpy(float) - (4 if model == "MFM" else 2)
    table = _car_tests(z, dof, car, res.mean_correlation)
    for name, theirs in TESTS.items():
        tol = EXACT if name in ("patell", "bmp") else 1e-7
        assert (
            rel(
                table.loc[name, "statistic"],
                stata[model, theirs, window, "group", "stat"],
            )
            < tol
        )
        assert (
            rel(table.loc[name, "pvalue"], stata[model, theirs, window, "group", "pv"])
            < 1e-6
        )


@pytest.mark.parametrize("model", ["MAM", "HMM"])
@pytest.mark.parametrize("window", list(WINDOWS))
def test_end_to_end_tests_match_estudy_for_two_models(
    returns, events, stata, model, window
):
    """The standard deviations agree for these two models, so the whole
    pipeline is comparable. 5e-5: float storage on estudy's side, and the
    3e-5 of the Patell degrees of freedom."""
    res, _ = fit(returns, events, model, window, correlation="event")
    for name, theirs in TESTS.items():
        assert (
            rel(
                res.tests.loc[name, "statistic"],
                stata[model, theirs, window, "group", "stat"],
            )
            < 5e-5
        )


def test_calendar_and_event_time_correlation_differ_and_only_one_is_estudys(
    returns, events, stata
):
    cal, _ = fit(returns, events, "SIM", "0_0")
    evt, _ = fit(returns, events, "SIM", "0_0", correlation="event")
    assert abs(cal.mean_correlation - evt.mean_correlation) > 1e-3
    n = cal.n_events
    ratio = (
        stata["SIM", "ADJPatell", "0_0", "group", "stat"]
        / stata["SIM", "Patell", "0_0", "group", "stat"]
    )
    assert rel(1 / np.sqrt(1 + (n - 1) * evt.mean_correlation), ratio) < 1e-8
    assert rel(1 / np.sqrt(1 + (n - 1) * cal.mean_correlation), ratio) > 1e-3


def _market_adjusted_pieces(returns, events, window):
    """Standardised estimation abnormal returns and standardised CARs of
    the market-adjusted model, straight from the data."""
    lo, hi = WINDOWS[window]
    cols, scar, car, own = {}, [], [], []
    for unit, when in zip(events["id"], events["event_date"]):
        sec = returns[returns["id"] == unit].reset_index(drop=True)
        pos = sec.index[sec["date"] == when][0]
        est = sec.iloc[pos + EST[0] : pos + EST[1] + 1]
        ok = est["ret"].notna().to_numpy()
        a = (est["ret"] - est["mkt"]).to_numpy()[ok]
        sd = a.std(ddof=1)
        c = (sec["ret"] - sec["mkt"]).to_numpy()[pos + lo : pos + hi + 1].sum()
        cols[unit] = pd.Series(a / sd, index=np.arange(EST[0], EST[1] + 1)[ok])
        scar.append(c / (sd * np.sqrt(hi - lo + 1)))
        car.append(c)
        own.append(a)
    return pd.DataFrame(cols), np.array(scar), np.array(car), own


@pytest.mark.parametrize("window", list(WINDOWS))
def test_generalized_rank_test_and_estudys_off_by_one(returns, events, stata, window):
    from scipy import stats

    res, _ = fit(returns, events, "MAM", window)
    frame, scar, _, _ = _market_adjusted_pieces(returns, events, window)
    full = frame.copy()
    full.loc[0] = scar / scar.std(ddof=1)
    n = full.shape[1]
    share = full.notna().sum(axis=1) / n
    l1 = EST[1] - EST[0] + 1
    t_obs = l1 + 1

    def statistic(rank_scale, n_days, t_len):
        u = full.rank() / rank_scale - 0.5
        daily = u.mean(axis=1)
        z = daily.loc[0] / np.sqrt(np.sum(share * daily**2) / n_days)
        t = z * np.sqrt((t_len - 2) / (t_len - 1 - z * z))
        return t, 2 * stats.t.sf(abs(t), t_len - 2)

    # the test as defined: T = L1 + 1 observations, ranks over T + 1
    ours = statistic(full.notna().sum() + 1, t_obs, t_obs)
    assert rel(res.tests.loc["grank", "statistic"], ours[0]) < 1e-10
    assert rel(res.tests.loc["grank", "pvalue"], ours[1]) < 1e-10
    # estudy: ranks over L1 + 1, and L1 wherever T belongs
    theirs = statistic(l1 + 1, l1, l1)
    assert rel(theirs[0], stata["MAM", "GRANK", window, "group", "stat"]) < 1e-6
    assert rel(theirs[1], stata["MAM", "GRANK", window, "group", "pv"]) < 1e-5
    assert 0.005 < theirs[0] / ours[0] - 1 < 0.03
    # ranks over L1 + 1 are not centred: their mean is 1 / (2 T), not 0
    # (for a security with every estimation day; s03 has one fewer)
    whole = full.loc[:, full.notna().all()]
    off = (whole.rank() / (l1 + 1) - 0.5).mean().mean()
    assert off == pytest.approx(1 / (2 * t_obs), rel=1e-9)
    assert abs((full.rank() / (full.notna().sum() + 1) - 0.5).mean().mean()) < 1e-12


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("window", list(WINDOWS))
def test_signed_rank_test_matches_stata_signrank(returns, events, stata, model, window):
    res, _ = fit(returns, events, model, window)
    assert (
        rel(
            res.tests.loc["sign_rank", "statistic"],
            stata[model, "signrank", window, "group", "stat"],
        )
        < EXACT
    )
    assert (
        rel(
            res.tests.loc["sign_rank", "pvalue"],
            stata[model, "signrank", window, "group", "pv"],
        )
        < EXACT
    )


@pytest.mark.parametrize("window", list(WINDOWS))
def test_estudys_wilcoxon_row_ranks_within_each_security(
    returns, events, stata, window
):
    """What estudy's Wilcoxon option computes: the rank of |CAR| among that
    security's own absolute abnormal returns, summed over the securities
    with a positive CAR, then centred and scaled as if it were a
    signed-rank sum over the twelve securities. No signed-rank statistic
    for twelve observations exceeds 3.06 in absolute value."""
    _, _, car, own = _market_adjusted_pieces(returns, events, window)
    n = len(car)
    total = 0.0
    for c, a in zip(car, own):
        if c >= 0:
            pool = np.abs(np.append(a, c))
            total += pd.Series(pool).rank().iloc[-1]
    scale = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    rebuilt = (total - n * (n + 1) / 4) / scale
    theirs = stata["MAM", "Wilcoxon", window, "group", "stat"]
    assert rel(rebuilt, theirs) < 1e-6
    largest_possible = (n * (n + 1) / 2 - n * (n + 1) / 4) / scale
    assert largest_possible < 3.1 < 90 < theirs
    assert stata["MAM", "Wilcoxon", window, "group", "pv"] == 0


@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("window", list(WINDOWS))
def test_estudys_group_row_is_the_log_of_the_mean_gross_abnormal_return(
    returns, events, stata, model, window
):
    """estudy: sum over the window of log(mean_i exp(AR_it)), with the
    variance of MacKinlay's equation 18, sum of the CAR variances over
    N squared, and t(M - k) from the group series."""
    from scipy import stats

    res, ours = fit(returns, events, model, window)
    ids = list(ours.index)
    group_car = float(np.log(np.exp(res.ar).mean(axis=1)).sum())
    assert rel(group_car, stata[model, "Norm", window, "group", "car"]) < 2e-5
    sd = reference(stata, model, window, ids, "sd")
    group_sd = np.sqrt(np.sum(sd**2)) / len(ids)
    assert rel(group_sd, stata[model, "Norm", window, "group", "sd"]) < EXACT
    z = stata[model, "Norm", window, "group", "stat"]
    assert rel(group_car / group_sd, z) < 2e-5
    k = {"SIM": 2, "MFM": 4}.get(model, 2)
    p = 2 * stats.t.sf(abs(z), EST[1] - EST[0] + 1 - k)
    assert rel(p, stata[model, "Norm", window, "group", "pv"]) < EXACT
    # the gap to the mean CAR is Jensen's: half the cross-sectional
    # variance of the abnormal returns, summed over the window
    jensen = float((0.5 * res.ar.var(axis=1, ddof=0)).sum())
    assert group_car - res.caar == pytest.approx(jensen, rel=0.05)


def test_the_mean_car_is_the_mean_of_the_cars_and_estudys_group_row_is_not(
    returns, events, stata
):
    res, ours = fit(returns, events, "SIM", "0_0")
    assert res.caar == pytest.approx(ours["car"].mean(), rel=1e-12)
    ids = list(ours.index)
    theirs = reference(stata, "SIM", "0_0", ids, "car")
    group = stata["SIM", "Norm", "0_0", "group", "car"]
    assert abs(group / theirs.mean() - 1) > 0.02
    # the planted effect: 2% on six of twelve securities
    assert 0.005 < res.caar < 0.02
