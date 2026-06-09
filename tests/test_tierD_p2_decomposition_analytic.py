"""Tier D P2 known-truth upgrades — Gelbach & Shapley decompositions.

Part of the P1/P2 "Tier D analytic special-cases" campaign (see
``.tierd_campaign/CAMPAIGN.md``). Both were graded ``weak`` by
``scripts/tierd_classify.py``. Each anchors to a decomposition adding-up /
axiom identity that holds exactly:

    sp.gelbach             total explained = base coef - full coef, and the
                           per-covariate deltas sum to that total (exact).
    sp.shapley_inequality  Shorrocks-Shapley: symmetric covariates get equal
                           contributions; an irrelevant covariate gets ~0.

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# sp.gelbach — conditional decomposition of coefficient movement
# ---------------------------------------------------------------------------
class TestGelbachAnalytic:

    @staticmethod
    def _dgp(seed=0, n=4000):
        rng = np.random.default_rng(seed)
        x = rng.normal(0, 1, n)
        m1 = 0.6 * x + rng.normal(0, 1, n)
        m2 = -0.4 * x + rng.normal(0, 1, n)
        # True structural model: y = x + 2 m1 + 1.5 m2 + e.
        y = 1.0 * x + 2.0 * m1 + 1.5 * m2 + rng.normal(0, 1, n)
        return pd.DataFrame({"y": y, "x": x, "m1": m1, "m2": m2})

    def test_total_change_equals_base_minus_full(self):
        # Gelbach's identity: the explained movement in the coefficient of
        # interest is exactly base_coef - full_coef.
        df = self._dgp()
        g = sp.gelbach(
            df, y="y", base_x=["x"], added_x=["m1", "m2"], var_of_interest="x"
        )
        assert g.total_change == pytest.approx(g.base_coef - g.full_coef, abs=1e-10)

    def test_contributions_sum_to_total(self):
        # The per-added-variable deltas add up to the total explained change.
        df = self._dgp()
        g = sp.gelbach(
            df, y="y", base_x=["x"], added_x=["m1", "m2"], var_of_interest="x"
        )
        assert g.decomposition["delta"].sum() == pytest.approx(g.total_change, abs=1e-9)

    def test_recovers_known_mediator_contributions(self):
        # delta_j = gamma_j (x->m_j) * beta_j (m_j->y): m1 ~ 0.6*2.0=1.2,
        # m2 ~ -0.4*1.5=-0.6.
        df = self._dgp()
        g = sp.gelbach(
            df, y="y", base_x=["x"], added_x=["m1", "m2"], var_of_interest="x"
        )
        d = g.decomposition.set_index("variable")["delta"]
        assert d["m1"] == pytest.approx(1.2, abs=0.1)
        assert d["m2"] == pytest.approx(-0.6, abs=0.1)


# ---------------------------------------------------------------------------
# sp.shapley_inequality — Shorrocks-Shapley factor decomposition
# ---------------------------------------------------------------------------
class TestShapleyInequalityAnalytic:

    def test_symmetric_covariates_get_equal_contributions(self):
        # x1, x2 iid with the same coefficient enter symmetrically, so the
        # Shapley axiom gives them (approximately) equal contributions.
        rng = np.random.default_rng(1)
        n = 6000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = np.exp(0.5 * x1 + 0.5 * x2 + rng.normal(0, 0.2, n))
        sh = sp.shapley_inequality(
            pd.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y", x=["x1", "x2"]
        )
        c = sh.shapley.set_index("variable")["contribution"]
        assert c["x1"] == pytest.approx(c["x2"], rel=0.15)

    def test_irrelevant_covariate_has_negligible_contribution(self):
        rng = np.random.default_rng(2)
        n = 6000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = np.exp(0.6 * x1 + 0.0 * x2 + rng.normal(0, 0.2, n))  # x2 has no effect
        sh = sp.shapley_inequality(
            pd.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y", x=["x1", "x2"]
        )
        row = sh.shapley.set_index("variable")
        assert abs(row.loc["x2", "pct_of_total"]) < 2.0  # < 2% of total
        assert row.loc["x1", "contribution"] > row.loc["x2", "contribution"]

    def test_contributions_are_a_valid_share(self):
        rng = np.random.default_rng(3)
        n = 4000
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = np.exp(0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.3, n))
        sh = sp.shapley_inequality(
            pd.DataFrame({"y": y, "x1": x1, "x2": x2}), y="y", x=["x1", "x2"]
        )
        pct = sh.shapley["pct_of_total"].values
        # Explained shares are non-negative and cannot exceed 100% (the
        # residual carries the unexplained remainder).
        assert np.all(pct >= -1e-6)
        assert pct.sum() <= 100.0 + 1e-6
