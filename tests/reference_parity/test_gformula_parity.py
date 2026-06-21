"""Reference parity: ``sp.g_computation`` (parametric g-formula).

Anchored coverage for the point-exposure g-computation estimator in
``src/statspai/inference/g_computation.py``.  Existing tests cover
loose recovery (tests/test_g_computation.py), metamorphic/affine
invariances (tests/tier_eg/test_weighting_gmethods_invariance.py) and
the NHEFS book value (tests/external_parity/test_whatif_nhefs.py);
this file adds the missing *anchored* coverage:

A. Linear collapse (closed form, 1e-8): the default Q-model is a
   SINGLE additive OLS of y on (1, D, X) — see g_computation.py:162-172,
   where ``design = [D, X]`` is fit once via ``sm.OLS(Y, add_constant(...))``
   and counterfactual predictions flip only the D column.  Hence
   ATE = mean(Q(1,X)-Q(0,X)) = beta_D exactly, and (g_computation.py:189-195)
   ATT = mean over treated of the same constant = beta_D too.  The
   expected value is computed independently via ``np.linalg.lstsq``.

B. Saturated discrete cell (closed form, 1e-10): with one binary
   covariate and a saturated Q-model (sklearn DecisionTreeRegressor,
   which interpolates exact cell means on discrete features via the
   ml_Q branch at g_computation.py:174-183), the g-formula reduces to
   nonparametric standardization
       ATE = sum_x (ybar_{1,x} - ybar_{0,x}) * P_hat(X=x)
       ATT = sum_x (ybar_{1,x} - ybar_{0,x}) * P_hat(X=x | D=1)
   hand-computed with pandas groupby (Robins 1986 g-formula;
   Hernan-Robins 2020 ch. 13 standardization).

C. Frozen base-R fixture: ``_fixtures/gformula_R.json`` produced by
   ``_fixtures/_generate_gformula.R`` (stats::lm only — no CRAN deps)
   on the deterministic ``_fixtures/gformula_data.csv``
   (``_generate_gformula_data.py``, seed=20260612).  R computes
   psi = mean(predict(fit, d=1)) - mean(predict(fit, d=0)) with
   fit = lm(y ~ d + x1 + x2 + x3), exactly the implementation's model
   form.  Point estimate pinned at 1e-8.  Bootstrap SE deliberately
   NOT pinned tightly: bootstrap RNG streams differ across languages,
   so we compare the Python bootstrap SE to R's classical OLS SE of
   beta_d only within a loose recovery-grade band (see test comment).

D. Recovery (4-sigma) on the confounded ``matching_cia_data`` DGP
   (conftest.py, true homogeneous effect = 2.0): naive diff-in-means
   must be >4 sigma biased (proves the adjustment is doing real work;
   observed ~16 sigma) while g-computation recovers truth within
   4 sigma of its bootstrap SE.

E. Invariance: an irrelevant noise covariate moves the estimate by
   far less than 4 sigma; and estimand='ATE' ignores ``treat_values``
   (grid is hard-set to [0,1] at g_computation.py:153, docstring line
   "Ignored otherwise"), so default vs explicit ``treat_values=(0,1)``
   with the same seed must be bitwise identical.

References (bib keys verified in paper.bib)
-------------------------------------------
- Robins, J. (1986). A new approach to causal inference in mortality
  studies with a sustained exposure period — application to control of
  the healthy worker survivor effect. *Mathematical Modelling*,
  7(9-12), 1393-1512. [@robins1986new]
- Snowden, J.M., Rose, S., Mortimer, K.M. (2011). Implementation of
  G-Computation on a Simulated Data Set: Demonstration of a Causal
  Inference Technique. *American Journal of Epidemiology*.
  [@snowden2011implementation]
- Hernan, M.A., Robins, J.M. (2020). *Causal Inference: What If*.
  Chapman & Hall/CRC, ch. 13. [@hernan2020causal]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"
_COVS = ["x1", "x2", "x3"]


@pytest.fixture(scope="module")
def gf_data():
    """Deterministic seed=20260612 DGP — same CSV base R consumes."""
    path = _FIXTURE_DIR / "gformula_data.csv"
    if not path.exists():  # pragma: no cover — fixture is committed
        pytest.skip("gformula_data.csv missing — run _generate_gformula_data.py")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def gf_R():
    """Frozen base-R reference (stats::lm standardization)."""
    path = _FIXTURE_DIR / "gformula_R.json"
    if not path.exists():  # pragma: no cover — fixture is committed
        pytest.skip("gformula_R.json missing — run _generate_gformula.R")
    with open(path) as f:
        return json.load(f)


def _ols_beta_d(df: pd.DataFrame) -> float:
    """Independent OLS treatment coefficient via numpy lstsq.

    Hand-rolled design [1, d, x1, x2, x3] — no statspai internals.
    """
    X = np.column_stack(
        [
            np.ones(len(df)),
            df["d"].to_numpy(dtype=float),
            df[_COVS].to_numpy(dtype=float),
        ]
    )
    beta, *_ = np.linalg.lstsq(X, df["y"].to_numpy(dtype=float), rcond=None)
    return float(beta[1])


# ---------------------------------------------------------------------------
# A. Linear collapse: ATE (and ATT) == OLS beta_d under the default Q-model
# ---------------------------------------------------------------------------


class TestLinearCollapse:
    """Default Q is ONE additive OLS (g_computation.py:162-172), so the
    standardized contrast collapses to the treatment coefficient exactly."""

    def test_ate_equals_independent_ols_coefficient(self, gf_data):
        res = sp.g_computation(
            gf_data, y="y", treat="d", covariates=_COVS, n_boot=10, seed=1
        )
        expected = _ols_beta_d(gf_data)
        # atol 1e-8: exact algebra — Q(1,X)-Q(0,X) = beta_d for every row
        # under the additive fit, so the only discrepancy is the linear
        # solver (statsmodels pinv/SVD vs numpy lstsq/SVD), both backward-
        # stable on this well-conditioned 5-column design (~1e-15 observed).
        assert (
            abs(res.estimate - expected) < 1e-8
        ), f"ATE {res.estimate!r} != OLS beta_d {expected!r}"

    def test_att_collapses_to_same_coefficient(self, gf_data):
        """ATT averages the SAME constant contrast over treated rows
        (g_computation.py:189-195) — single-model implementation fact —
        so ATT == ATE == beta_d under the additive default Q."""
        res = sp.g_computation(
            gf_data,
            y="y",
            treat="d",
            covariates=_COVS,
            estimand="ATT",
            n_boot=10,
            seed=1,
        )
        expected = _ols_beta_d(gf_data)
        # atol 1e-8: same exactness argument as the ATE collapse above.
        assert abs(res.estimate - expected) < 1e-8


# ---------------------------------------------------------------------------
# B. Saturated discrete cell: nonparametric standardization identity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def saturated_cell_data():
    """One binary covariate, binary D, interacted cell means.

    y = 1 + 2 d + 1.5 x + 2 d*x + N(0, 0.5); P(D=1|x) = 0.5 / 0.8 for
    x = 0 / 1.  The interaction plus x-varying propensity makes the
    additive-OLS coefficient (the variance-weighted average of the
    x-specific contrasts; Angrist-Pischke MHE ch. 3 [@angrist2009mostly])
    visibly DIFFERENT from the P(X=x)-weighted g-formula target, so the
    anchor below cannot pass tautologically (observed gap ~0.25).
    """
    rng = np.random.default_rng(40806)
    n = 600
    x = rng.binomial(1, 0.4, n)
    p = np.where(x == 1, 0.8, 0.5)
    d = (rng.uniform(size=n) < p).astype(int)
    y = 1.0 + 2.0 * d + 1.5 * x + 2.0 * d * x + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "d": d, "x": x})


class TestSaturatedCellStandardization:
    """g-formula with a saturated Q == groupby cell-mean standardization
    (Robins 1986; Hernan-Robins 2020 ch. 13)."""

    @staticmethod
    def _cell_contrasts(df):
        cell = df.groupby(["d", "x"])["y"].mean()
        return {xv: cell[(1, xv)] - cell[(0, xv)] for xv in (0, 1)}

    @staticmethod
    def _tree():
        # Saturated Q-model: an unrestricted DecisionTreeRegressor on the
        # 4 discrete (d, x) cells splits to purity and predicts EXACT cell
        # means — an external (sklearn) stand-in for a saturated regression,
        # consumed via the ml_Q clone/fit/predict branch
        # (g_computation.py:174-183).
        from sklearn.tree import DecisionTreeRegressor

        return DecisionTreeRegressor(random_state=0)

    def test_ate_equals_groupby_standardization(self, saturated_cell_data):
        df = saturated_cell_data
        contrast = self._cell_contrasts(df)
        px = df["x"].value_counts(normalize=True)
        expected_ate = sum(contrast[xv] * px[xv] for xv in (0, 1))

        res = sp.g_computation(
            df, y="y", treat="d", covariates=["x"], ml_Q=self._tree(), n_boot=10, seed=2
        )
        # atol 1e-10: tree leaf means and pandas groupby means are the same
        # arithmetic means of the same float64 cells; only summation order
        # differs (~1e-15 observed).  Machine-collapse anchor.
        assert (
            abs(res.estimate - expected_ate) < 1e-10
        ), f"ATE {res.estimate!r} != standardized {expected_ate!r}"

    def test_att_uses_treated_covariate_distribution(self, saturated_cell_data):
        df = saturated_cell_data
        contrast = self._cell_contrasts(df)
        px_treated = df.loc[df["d"] == 1, "x"].value_counts(normalize=True)
        expected_att = sum(contrast[xv] * px_treated[xv] for xv in (0, 1))

        res = sp.g_computation(
            df,
            y="y",
            treat="d",
            covariates=["x"],
            estimand="ATT",
            ml_Q=self._tree(),
            n_boot=10,
            seed=2,
        )
        # atol 1e-10: same machine-collapse argument; ATT averages the cell
        # contrasts over treated rows only (g_computation.py:189-195), i.e.
        # weights P_hat(X=x | D=1).
        assert (
            abs(res.estimate - expected_att) < 1e-10
        ), f"ATT {res.estimate!r} != standardized {expected_att!r}"

    def test_anchor_is_not_tautological(self, saturated_cell_data):
        """Non-degeneracy guard: the additive-OLS collapse value (anchor A's
        target) must differ visibly from the saturated standardization
        target here — otherwise B would silently retest A.  Expected gap
        ~0.25 on this DGP (variance-weighting vs P(X)-weighting of the
        x-specific contrasts; Angrist-Pischke MHE ch. 3)."""
        df = saturated_cell_data
        contrast = self._cell_contrasts(df)
        px = df["x"].value_counts(normalize=True)
        expected_ate = sum(contrast[xv] * px[xv] for xv in (0, 1))
        X = np.column_stack(
            [
                np.ones(len(df)),
                df["d"].to_numpy(dtype=float),
                df["x"].to_numpy(dtype=float),
            ]
        )
        beta, *_ = np.linalg.lstsq(X, df["y"].to_numpy(dtype=float), rcond=None)
        assert abs(beta[1] - expected_ate) > 0.05


# ---------------------------------------------------------------------------
# C. Frozen base-R fixture (stats::lm standardization)
# ---------------------------------------------------------------------------


class TestFrozenRParity:
    """sp.g_computation vs base-R lm + predict standardization."""

    def test_point_estimate_matches_R(self, gf_data, gf_R):
        res = sp.g_computation(
            gf_data, y="y", treat="d", covariates=_COVS, n_boot=10, seed=3
        )
        # atol 1e-8: both languages parse identical doubles (pandas writes
        # shortest-roundtrip floats; read.csv restores them exactly) and
        # solve the same least-squares problem (R QR vs statsmodels pinv,
        # both backward-stable); observed agreement ~7e-16.
        assert (
            abs(res.estimate - gf_R["psi"]) < 1e-8
        ), f"Python {res.estimate!r} vs R psi {gf_R['psi']!r}"

    def test_bootstrap_se_loosely_matches_R_classical_se(self, gf_data, gf_R):
        """Bootstrap SE is pinned only LOOSELY (decision documented in
        _generate_gformula.R): bootstrap resampling RNG streams differ
        across languages, so draw-level pinning is meaningless.  Instead,
        the nonparametric-bootstrap SE of psi is consistent for the
        sandwich SE of beta_d, which equals the classical OLS SE under
        this homoskedastic Gaussian DGP; with B=200 the bootstrap-SE
        Monte-Carlo noise is ~SE/sqrt(2B) ~ 5%.  Band: +/-25% relative
        (recovery-grade; observed ratio 0.895 with the fixed seed)."""
        res = sp.g_computation(
            gf_data, y="y", treat="d", covariates=_COVS, n_boot=200, seed=4
        )
        r_se = gf_R["se_d_classical"]
        assert (
            abs(res.se / r_se - 1.0) < 0.25
        ), f"bootstrap SE {res.se:.6f} vs R classical SE {r_se:.6f}"

    def test_fixture_csv_intact(self, gf_data):
        """Guard that the CSV fixture wasn't accidentally mutated."""
        assert len(gf_data) == 800
        assert list(gf_data.columns) == ["y", "d", "x1", "x2", "x3"]
        assert set(np.unique(gf_data["d"])) == {0, 1}

    def test_fixture_R_meta_present(self, gf_R):
        """Guard that the R fixture has reproduction metadata."""
        assert gf_R["meta"]["model"] == "lm(y ~ d + x1 + x2 + x3)"
        assert gf_R["meta"]["n"] == 800
        # R's own self-check: psi must equal coef_d (additive collapse).
        assert abs(gf_R["psi"] - gf_R["coef_d"]) < 1e-12


# ---------------------------------------------------------------------------
# D. Recovery under confounding (the adjustment is real)
# ---------------------------------------------------------------------------


class TestConfoundedRecovery:
    """On matching_cia_data (conftest.py: homogeneous effect 2.0, linear
    outcome in (X1, X2, X3) => default OLS Q is correctly specified)."""

    def test_naive_diff_in_means_is_visibly_biased(self, matching_cia_data):
        """Confounding check: treated units have higher X1 / lower X2 /
        higher X3, all of which raise y0, so naive diff-in-means must sit
        far above truth (observed ~16 sigma).  Hand-rolled two-sample SE —
        no statspai internals."""
        df = matching_cia_data
        truth = df.attrs["true_effect"]
        y1 = df.loc[df["d"] == 1, "y"]
        y0 = df.loc[df["d"] == 0, "y"]
        naive = y1.mean() - y0.mean()
        se_naive = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        # >4 sigma: same n_sigma scale as the recovery test below, so the
        # pair proves the adjustment moves the estimate across a gap the
        # recovery tolerance could never absorb.
        assert abs(naive - truth) > 4.0 * se_naive, (
            f"DGP regression: naive {naive:.4f} is not biased "
            f"(truth {truth}, SE {se_naive:.4f}) — anchor D is vacuous"
        )

    def test_g_computation_recovers_truth_within_4_sigma(self, matching_cia_data):
        df = matching_cia_data
        truth = df.attrs["true_effect"]
        res = sp.g_computation(
            df, y="y", treat="d", covariates=["X1", "X2", "X3"], n_boot=200, seed=5
        )
        # 4-sigma recovery (REFERENCES.md convention): false-failure prob
        # 6.3e-5; observed ~1.2 sigma on this seed.
        assert abs(res.estimate - truth) <= 4.0 * res.se, (
            f"g-computation {res.estimate:.4f} (SE {res.se:.4f}) "
            f"failed to recover truth {truth}"
        )


# ---------------------------------------------------------------------------
# E. Invariances
# ---------------------------------------------------------------------------


class TestInvariances:

    def test_irrelevant_covariate_barely_moves_estimate(self, matching_cia_data):
        """Adding pure noise as a covariate perturbs the additive-OLS
        collapse value by O_p(n^{-1}) — far inside sampling noise.  Bound:
        4 * SE of the base fit (4-sigma convention; observed |delta| ~2e-4
        vs bound ~0.13).  Catches covariate-column wiring bugs."""
        df = matching_cia_data.copy()
        df["noise"] = np.random.default_rng(9090).normal(size=len(df))
        base = sp.g_computation(
            df, y="y", treat="d", covariates=["X1", "X2", "X3"], n_boot=200, seed=6
        )
        aug = sp.g_computation(
            df,
            y="y",
            treat="d",
            covariates=["X1", "X2", "X3", "noise"],
            n_boot=200,
            seed=6,
        )
        assert abs(aug.estimate - base.estimate) <= 4.0 * base.se, (
            f"noise covariate moved ATE from {base.estimate:.4f} "
            f"to {aug.estimate:.4f} (4*SE = {4 * base.se:.4f})"
        )

    def test_ate_ignores_treat_values_bitwise(self, matching_cia_data):
        """For estimand='ATE' the grid is hard-set to [0, 1]
        (g_computation.py:153) and ``treat_values`` is documented as
        ignored, so with the same seed the two calls execute an identical
        code path => bitwise-equal estimate AND bootstrap SE (exact ==,
        no tolerance)."""
        df = matching_cia_data
        a = sp.g_computation(
            df, y="y", treat="d", covariates=["X1", "X2", "X3"], n_boot=50, seed=7
        )
        b = sp.g_computation(
            df,
            y="y",
            treat="d",
            covariates=["X1", "X2", "X3"],
            treat_values=(0, 1),
            n_boot=50,
            seed=7,
        )
        assert a.estimate == b.estimate
        assert a.se == b.se
