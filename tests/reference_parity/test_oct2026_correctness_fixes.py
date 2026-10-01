"""Known-truth anchors for the six correctness fixes of the 2026-10 pass.

Each estimator here had no numerical evidence attached. Probing it against
a DGP with a known answer showed it did not recover that answer, and the
cause was traced to a defect in the code, not to the DGP. The tests below
pin the truth each one must now reach.

* ``sp.balke_pearl`` -- the sharp ATE bounds are the optimum of a linear
  program over the 16 (compliance type) x (outcome response type) cells.
  The program is solved here directly, so the reference is first-principles
  and shares no code with the closed form under test.
* ``sp.rd_multi_score`` / ``sp.multi_score_rd`` -- "treated when every score
  crosses its cutoff".
* ``sp.design_robust_event_study`` -- exposure longer than ``lags``.
* ``sp.cluster_staggered_rollout`` -- a second cohort switching on between
  the reference period and the event time.
* ``sp.rd_extrapolate`` -- standard error of the average over perfectly
  correlated evaluation points.
* ``sp.kitagawa_test`` -- a bootstrap that has to be drawn under the null.

Tolerances: exact identities at 1e-9 or tighter; recoveries at 4 sigma of
the reported or Monte Carlo standard error (tests/reference_parity/
REFERENCES.md, "Tolerance convention").
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

import statspai as sp

# --------------------------------------------------------------------- #
#  Balke-Pearl bounds against the response-type linear program
# --------------------------------------------------------------------- #

_D_TYPES = (  # D as a function of Z: never-taker, complier, defier, always-taker
    lambda z: 0,
    lambda z: z,
    lambda z: 1 - z,
    lambda z: 1,
)
_Y_TYPES = (  # Y as a function of D: never, helped, hurt, always
    lambda d: 0,
    lambda d: d,
    lambda d: 1 - d,
    lambda d: 1,
)
_TYPES = list(itertools.product(range(4), range(4)))


def _cell_probs(q: np.ndarray) -> np.ndarray:
    """P[z, d, y] = P(D=d, Y=y | Z=z) implied by type shares ``q``."""
    P = np.zeros((2, 2, 2))
    for (t, r), w in zip(_TYPES, q):
        for z in (0, 1):
            d = _D_TYPES[t](z)
            P[z, d, _Y_TYPES[r](d)] += w
    return P


def _lp_bounds(P: np.ndarray, no_defiers: bool = False):
    """Sharp ATE bounds given P(D, Y | Z), by linear programming."""
    A, b = [], []
    for z in (0, 1):
        for d in (0, 1):
            for y in (0, 1):
                A.append(
                    [
                        1.0 if (_D_TYPES[t](z) == d and _Y_TYPES[r](d) == y) else 0.0
                        for t, r in _TYPES
                    ]
                )
                b.append(P[z, d, y])
    c = np.array([_Y_TYPES[r](1) - _Y_TYPES[r](0) for _, r in _TYPES], float)
    bounds = [(0.0, 0.0) if (no_defiers and t == 2) else (0.0, 1.0) for t, _ in _TYPES]
    lo = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    hi = linprog(-c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    assert lo.status == 0 and hi.status == 0
    return float(lo.fun), float(-hi.fun)


def _frame_from_cells(P: np.ndarray, per_arm: int = 100_000) -> pd.DataFrame:
    rows = []
    for z in (0, 1):
        for d in (0, 1):
            for y in (0, 1):
                rows += [(y, d, z)] * int(round(P[z, d, y] * per_arm))
    return pd.DataFrame(rows, columns=["y", "d", "z"])


def _empirical_cells(df: pd.DataFrame) -> np.ndarray:
    P = np.zeros((2, 2, 2))
    for z in (0, 1):
        sub = df[df["z"] == z]
        for d in (0, 1):
            for y in (0, 1):
                P[z, d, y] = ((sub["d"] == d) & (sub["y"] == y)).mean()
    return P


class TestBalkePearlMatchesLinearProgram:
    @pytest.mark.parametrize("seed", range(40))
    def test_bounds_equal_the_lp_optimum_and_contain_the_truth(self, seed):
        rng = np.random.default_rng(seed)
        q = rng.dirichlet(np.ones(16))
        true_ate = sum(
            w * (_Y_TYPES[r](1) - _Y_TYPES[r](0)) for (_, r), w in zip(_TYPES, q)
        )
        df = _frame_from_cells(_cell_probs(q))
        # The LP is fed the cell shares of the very frame the estimator
        # sees, so rounding to integer counts cancels and the comparison is
        # an identity. Observed worst gap over these 40 models: < 1e-12.
        lo, hi = _lp_bounds(_empirical_cells(df))
        res = sp.balke_pearl(df, y="y", treat="d", instrument="z")
        assert res.lower == pytest.approx(lo, abs=1e-9)
        assert res.upper == pytest.approx(hi, abs=1e-9)
        # Sharp bounds on a valid IV model must contain the true ATE. The
        # slack covers the count rounding (cells move by at most 1 / 2e5).
        assert res.lower - 1e-4 <= true_ate <= res.upper + 1e-4

    @pytest.mark.parametrize("seed", range(20))
    def test_monotone_bounds_equal_the_lp_without_defiers(self, seed):
        rng = np.random.default_rng(1000 + seed)
        q = rng.dirichlet(np.ones(16))
        for i, (t, _) in enumerate(_TYPES):
            if t == 2:
                q[i] = 0.0
        q /= q.sum()
        df = _frame_from_cells(_cell_probs(q))
        lo, hi = _lp_bounds(_empirical_cells(df), no_defiers=True)
        res = sp.balke_pearl(df, y="y", treat="d", instrument="z")
        assert res.lower_monotone == pytest.approx(lo, abs=1e-9)
        assert res.upper_monotone == pytest.approx(hi, abs=1e-9)

    def test_one_sided_noncompliance_hand_computed(self):
        # Z=0 never treated; 80% compliers; P(Y=1) = 0.3 + 0.4 D for all.
        # E[Y(0)] = 0.3 is identified; E[Y(1)] lies in [0.8*0.7, 0.8*0.7+0.2].
        P = np.zeros((2, 2, 2))
        P[0, 0, 1], P[0, 0, 0] = 0.3, 0.7
        P[1, 1, 1], P[1, 1, 0] = 0.8 * 0.7, 0.8 * 0.3
        P[1, 0, 1], P[1, 0, 0] = 0.2 * 0.3, 0.2 * 0.7
        res = sp.balke_pearl(_frame_from_cells(P), y="y", treat="d", instrument="z")
        assert res.lower == pytest.approx(0.26, abs=1e-9)
        assert res.upper == pytest.approx(0.46, abs=1e-9)
        assert res.lower <= 0.4 <= res.upper  # the true ATE


# --------------------------------------------------------------------- #
#  Multi-score RD: treated when every score crosses
# --------------------------------------------------------------------- #


def _two_score_frame(seed: int, n: int = 8000, jump: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    treated = (x1 >= 0) & (x2 >= 0)
    y = jump * treated + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


class TestMultiScoreRdRecoversTheJump:
    @pytest.mark.parametrize("fn_name", ["rd_multi_score", "multi_score_rd"])
    def test_jump_under_the_all_scores_rule(self, fn_name):
        df = _two_score_frame(seed=42)
        res = getattr(sp, fn_name)(
            df, y="y", running_vars=["x1", "x2"], cutoffs=[0.0, 0.0]
        )
        # Truth 0.8. Before the fix this returned 0.14 (se 0.03) because
        # units with one score across were coded as treated.
        assert abs(res.boundary_effect - 0.8) <= 4.0 * res.se
        assert res.se < 0.05

    def test_alias_and_canonical_agree_exactly(self):
        df = _two_score_frame(seed=7, n=2000)
        a = sp.rd_multi_score(df, y="y", running_vars=["x1", "x2"], cutoffs=[0.0, 0.0])
        b = sp.multi_score_rd(df, y="y", running_vars=["x1", "x2"], cutoffs=[0.0, 0.0])
        assert a.boundary_effect == b.boundary_effect
        assert a.se == b.se

    def test_no_jump_gives_no_effect(self):
        df = _two_score_frame(seed=3, jump=0.0)
        res = sp.rd_multi_score(
            df, y="y", running_vars=["x1", "x2"], cutoffs=[0.0, 0.0]
        )
        assert abs(res.boundary_effect) <= 4.0 * res.se


# --------------------------------------------------------------------- #
#  Design-robust event study: exposure beyond the reported window
# --------------------------------------------------------------------- #


def _staggered_units(seed: int = 0, n_units: int = 300, T: int = 8, tau: float = 2.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = [0, 4, 6][i % 3]
        a = rng.normal()
        for t in range(1, T + 1):
            d = 1.0 if (g and t >= g) else 0.0
            rows.append((i, t, g, a + 0.3 * t + tau * d + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["id", "t", "g", "y"])


class TestDesignRobustEventStudyWindow:
    @pytest.mark.parametrize("lags", [2, 3, 4])
    def test_constant_effect_recovered_for_any_window(self, lags):
        df = _staggered_units()
        res = sp.design_robust_event_study(
            df, y="y", treat="g", time="t", id="id", leads=2, lags=lags
        )
        # Cohort 4 is exposed for up to four periods, so lags=2 and lags=3
        # leave treated rows outside the window. Truth 2.0; lags=2 used to
        # return 1.21.
        assert abs(res.estimate - 2.0) <= 4.0 * res.se
        es = res.model_info["event_study"].set_index("rel_time")
        # No pre-trend in the DGP: the lead is zero within 4 sigma.
        assert abs(es.loc[-2, "att"]) <= 4.0 * es.loc[-2, "se"]

    def test_binned_rows_are_counted_and_only_when_needed(self):
        df = _staggered_units()
        narrow = sp.design_robust_event_study(
            df, y="y", treat="g", time="t", id="id", leads=2, lags=2
        )
        wide = sp.design_robust_event_study(
            df, y="y", treat="g", time="t", id="id", leads=2, lags=4
        )
        # Cohort 4 (100 units) contributes rel_time 3 and 4 beyond lags=2.
        assert narrow.model_info["diagnostics"]["n_obs_binned_post"] == 200
        assert wide.model_info["diagnostics"]["n_obs_binned_post"] == 0
        assert wide.model_info["diagnostics"]["binned_post_coef"] is None
        # The reported in-window coefficients barely move with the window.
        a = narrow.model_info["event_study"].set_index("rel_time")["att"]
        b = wide.model_info["event_study"].set_index("rel_time")["att"]
        assert np.max(np.abs(a.loc[[0, 1, 2]] - b.loc[[0, 1, 2]])) < 0.05


# --------------------------------------------------------------------- #
#  Staggered cluster rollout: never-treated comparisons only
# --------------------------------------------------------------------- #


def _rollout_frame(seed: int, n_clusters: int = 60, tau: float = 1.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        if c < n_clusters // 3:
            ft = 4
        elif c < 2 * n_clusters // 3:
            ft = 6
        else:
            ft = 0
        fe = rng.normal()
        for t in range(8):
            d = 1 if (ft > 0 and t >= ft) else 0
            y = 1.0 + fe + 0.2 * t + tau * d + rng.normal(scale=0.3)
            rows.append({"cluster": c, "time": t, "first_treat": ft, "y": y})
    return pd.DataFrame(rows)


class TestClusterStaggeredRollout:
    def test_event_times_equal_the_never_treated_did_by_hand(self):
        df = _rollout_frame(seed=0)
        res = sp.cluster_staggered_rollout(
            df, y="y", cluster="cluster", time="time", first_treat="first_treat"
        )
        wide = df.pivot(index="cluster", columns="time", values="y")
        g = df.groupby("cluster")["first_treat"].first()
        ctrl = wide[g == 0].mean()
        by_hand = {}
        for k in range(-2, 5):
            cells = []
            for c in (4, 6):
                t = c + k
                if t in wide.columns:
                    trt = wide[g == c].mean()
                    cells.append((trt[t] - ctrl[t]) - (trt[c - 1] - ctrl[c - 1]))
            if cells:
                by_hand[k] = float(np.mean(cells))
        got = res.event_study.set_index("rel_time")["att"]
        for k, value in by_hand.items():
            assert got.loc[k] == pytest.approx(value, abs=1e-12)

    def test_constant_effect_recovered_at_every_post_event_time(self):
        df = _rollout_frame(seed=0)
        res = sp.cluster_staggered_rollout(
            df, y="y", cluster="cluster", time="time", first_treat="first_treat"
        )
        es = res.event_study.set_index("rel_time")
        # rel_time 2 and 3 for cohort 4 fall on t = 6, 7, when cohort 6 is
        # treated too. They used to come out at 0.80 and 0.93.
        for k in (0, 1, 2, 3):
            assert abs(es.loc[k, "att"] - 1.5) <= 4.0 * es.loc[k, "se"]
        assert abs(res.overall_att - 1.5) <= 4.0 * res.overall_se
        # Each event time now carries its own bootstrap SE.
        assert es.loc[[0, 1, 2, 3], "se"].nunique() == 4

    def test_overall_att_is_unbiased_across_seeds(self):
        draws = np.array(
            [
                sp.cluster_staggered_rollout(
                    _rollout_frame(seed=s),
                    y="y",
                    cluster="cluster",
                    time="time",
                    first_treat="first_treat",
                ).overall_att
                for s in range(30)
            ]
        )
        mc_se = draws.std(ddof=1) / np.sqrt(len(draws))
        assert abs(draws.mean() - 1.5) <= 4.0 * mc_se


# --------------------------------------------------------------------- #
#  RD extrapolation: SE of the averaged effect
# --------------------------------------------------------------------- #


class TestRdExtrapolateStandardError:
    def test_average_se_matches_its_sampling_spread(self):
        draws = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for s in range(30):
                rng = np.random.default_rng(s)
                n = 2000
                Z = rng.normal(0, 1, n)
                X = Z + rng.normal(0, 0.5, n)
                D = (X >= 0).astype(int)
                Y = 1.0 + 2.0 * Z + 3.0 * D + rng.normal(0, 0.5, n)
                res = sp.rd_extrapolate(
                    pd.DataFrame({"y": Y, "x": X, "z": Z}),
                    y="y",
                    x="x",
                    c=0,
                    covs=["z"],
                )
                draws.append((res.estimate, res.se))
        est, se = np.array(draws).T
        mc_sd = est.std(ddof=1)
        # Truth 3.0, recovered on average.
        assert abs(est.mean() - 3.0) <= 4.0 * mc_sd / np.sqrt(len(est))
        # The reported SE tracks the Monte Carlo spread. With 30 draws the
        # SD estimate itself has a relative SE of about 13%, hence the band.
        # Before the fix this ratio was about 0.2.
        assert 0.6 <= se.mean() / mc_sd <= 1.6
        cover = np.mean(np.abs(est - 3.0) <= 1.96 * se)
        assert cover >= 0.8


# --------------------------------------------------------------------- #
#  Kitagawa test: size and power
# --------------------------------------------------------------------- #


def _kitagawa_frame(seed: int, violate: bool, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, n)
    v = rng.normal(size=n)
    d = ((0.2 + 0.5 * z + 0.3 * v) > 0.5).astype(int)
    y = 1.0 + 0.8 * d + rng.normal(size=n)
    if violate:
        y = y + 2.0 * z  # the instrument enters the outcome directly
    return pd.DataFrame({"y": y, "d": d, "z": z})


class TestKitagawaSizeAndPower:
    def test_rejects_an_exclusion_violation_and_not_a_valid_instrument(self):
        reject_valid, reject_invalid = [], []
        for s in range(24):
            ok = sp.kitagawa_test(
                _kitagawa_frame(s, violate=False),
                y="y",
                treatment="d",
                instrument="z",
                n_boot=199,
                seed=s,
            )
            bad = sp.kitagawa_test(
                _kitagawa_frame(s, violate=True),
                y="y",
                treatment="d",
                instrument="z",
                n_boot=199,
                seed=s,
            )
            reject_valid.append(ok.p_value < 0.05)
            reject_invalid.append(bad.p_value < 0.05)
        # The test is conservative under a valid instrument: at most 3 of
        # 24 rejections has probability > 0.97 when the true size is 5%.
        assert sum(reject_valid) <= 3
        # Observed power on this DGP is about 0.7; the old bootstrap,
        # centred on the observed statistic, gave 0.
        assert np.mean(reject_invalid) >= 0.4

    def test_p_value_is_a_valid_monte_carlo_p_value(self):
        res = sp.kitagawa_test(
            _kitagawa_frame(0, violate=True),
            y="y",
            treatment="d",
            instrument="z",
            n_boot=99,
            seed=0,
        )
        # (1 + #{T* >= T}) / (B + 1) lies on the grid {1/100, ..., 1}.
        assert 0.01 - 1e-12 <= res.p_value <= 1.0
        assert abs(res.p_value * 100 - round(res.p_value * 100)) < 1e-9
