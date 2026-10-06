"""``se_method='abadie_imbens_2016'``: Stata ``teffects psmatch`` AI-2016 SE.

Pins the estimated-propensity-score variance of Abadie & Imbens (2016)
to Stata 18 ``teffects psmatch (re78) (treat age education black hispanic
married re74 re75, logit), atet nneighbor(1)`` on the NSW-DW replica
(``tests/stata_parity/11_psm.do``, golden ``se_teffects_ai``), and
checks that the default ``'abadie_imbens'`` (psmatch2 ``ai(1)``) number
did not move.

Tolerance: rel 1e-6 (the registered 11_psm budget).  The observed gap is
7e-8 and is Stata's ML stopping rule for the treatment-model logit --
plugging Stata's own ``e(bps)`` / ``e(Vps)`` into the same formula
reproduces ``e(V)`` at rel 1e-9 -- so 1e-6 is not a loosened budget but
the module's pre-registered one.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

X_NSW = ["age", "education", "black", "hispanic", "married", "re74", "re75"]

# tests/stata_parity/results/11_psm_Stata.json :: se_teffects_ai
STATA_TEFFECTS_SE_NN2 = 621.79324512017217
# Same command with vce(robust, nn(3)), Stata 18 MP, `set type double`
# (the _common.do import setting), captured 2026-09-22.
STATA_TEFFECTS_SE_NN3 = 630.128055520342
# tests/r_parity/results/11_psm_py.json :: se_abadie_imbens (psmatch2 ai(1))
PSMATCH2_AI1_SE = 643.3456988917197
NSW_ATT = 2277.6907208768525


@pytest.fixture(scope="module")
def nsw():
    return sp.datasets.nsw_dw()


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return sp.psm(df, y="re78", d="treat", X=X_NSW, method="nn", **kw)


class TestTeffectsPsmatchPin:
    def test_nn2_default_matches_stata_teffects_psmatch(self, nsw):
        fit = _fit(nsw, se_method="abadie_imbens_2016")
        assert fit.model_info["se_method"] == "abadie_imbens_2016"
        assert fit.model_info["ai_matches"] == 1
        assert fit.estimate == pytest.approx(NSW_ATT, rel=1e-12)
        assert fit.se == pytest.approx(STATA_TEFFECTS_SE_NN2, rel=1e-6)

    def test_nn3_matches_stata_teffects_psmatch(self, nsw):
        # ai_matches=J is Stata nn(J+1): the unit itself counts.
        fit = _fit(nsw, se_method="abadie_imbens_2016", ai_matches=2)
        assert fit.se == pytest.approx(STATA_TEFFECTS_SE_NN3, rel=1e-6)

    def test_components_rebuild_the_variance(self, nsw):
        fit = _fit(nsw, se_method="abadie_imbens_2016")
        c = fit.model_info["ai2016_components"]
        assert c["n1"] == 185 and c["h"] == 2
        assert c["att"] == pytest.approx(NSW_ATT, rel=1e-12)
        # var = base - c'Vc + d'Vd, and the score term is negative here.
        assert c["var"] == pytest.approx(c["base_var"] - c["c_V_c"] + c["d_V_d"])
        assert c["c_V_c"] > c["d_V_d"] > 0
        assert np.sqrt(c["var"]) == pytest.approx(fit.se)
        # The uncorrected base is the AI-2006 population-ATT SE with the
        # teffects nn(2) conditional-variance convention (673.42 on NSW-DW).
        assert c["base_se"] == pytest.approx(673.4208691076498, rel=1e-6)


class TestDefaultUnchanged:
    def test_default_is_still_psmatch2_ai1(self, nsw):
        fit = _fit(nsw)
        assert fit.model_info["se_method"] == "abadie_imbens"
        assert fit.se == pytest.approx(PSMATCH2_AI1_SE, rel=1e-9)
        assert fit.estimate == pytest.approx(NSW_ATT, rel=1e-12)

    def test_explicit_abadie_imbens_unchanged(self, nsw):
        fit = _fit(nsw, se_method="abadie_imbens")
        assert fit.se == pytest.approx(PSMATCH2_AI1_SE, rel=1e-9)

    def test_matching_and_score_unchanged(self, nsw):
        # The refactored logit fit must hand the matcher the same score.
        a = _fit(nsw).model_info["matched_data"]
        b = _fit(nsw, se_method="abadie_imbens_2016").model_info["matched_data"]
        np.testing.assert_array_equal(a["_pscore"].to_numpy(), b["_pscore"].to_numpy())
        np.testing.assert_array_equal(a["_n1"].to_numpy(), b["_n1"].to_numpy())


class TestScope:
    @pytest.mark.parametrize(
        "kw",
        [
            {"distance": "mahalanobis"},
            # the 2016 variance of the ATE needs every tied match
            {"estimand": "ATE", "ties": "first"},
            {"bias_correction": True},
        ],
    )
    def test_incompatible_paths_fail_loudly(self, nsw, kw):
        with pytest.raises(MethodIncompatibility, match="abadie_imbens_2016"):
            sp.match(
                nsw,
                y="re78",
                treat="treat",
                covariates=X_NSW,
                method="nearest",
                se_method="abadie_imbens_2016",
                **kw,
            )

    def test_kernel_matching_is_rejected(self, nsw):
        with pytest.raises(MethodIncompatibility, match="abadie_imbens_2016"):
            sp.match(
                nsw,
                y="re78",
                treat="treat",
                covariates=X_NSW,
                method="kernel",
                se_method="abadie_imbens_2016",
            )

    def test_bad_se_method_lists_the_new_option(self, nsw):
        with pytest.raises(MethodIncompatibility, match="abadie_imbens_2016"):
            sp.match(
                nsw,
                y="re78",
                treat="treat",
                covariates=X_NSW,
                method="nearest",
                se_method="bogus",
            )
