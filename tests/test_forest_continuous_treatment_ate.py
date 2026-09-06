"""The AIPW aggregation must refuse a treatment that has no propensity.

``average_treatment_effect`` builds the doubly-robust score

    psi_i = tau_i + (T_i - e_i) / (e_i (1 - e_i)) * (Y_i - m_i - (T_i - e_i) tau_i)

which divides by ``e(1 - e)`` and is therefore defined only when ``e`` is a
propensity, i.e. only when ``T`` is binary. With a continuous treatment the
same nuisance slot holds ``E[T | X]`` on the treatment's own scale, and the
clip into ``[0.01, 0.99]`` turned it into a ~1,200x multiplier: on the Card
returns-to-schooling design the reported "ATE" was -1266.6 against a mean
CATE of 0.086, carrying ``p = 0.0000``. These tests pin the guard.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning


@pytest.fixture(scope="module")
def card():
    return sp.datasets.card_1995()


def _forest(data, treatment, *, discrete):
    return sp.causal_forest(
        f"lwage ~ {treatment} | exper + black + south + smsa",
        data=data,
        n_estimators=200,
        discrete_treatment=discrete,
        random_state=42,
    )


class TestContinuousTreatmentAggregation:
    def test_continuous_treatment_warns_and_falls_back_to_plug_in(self, card):
        forest = _forest(card, "educ", discrete=False)
        with pytest.warns(AssumptionWarning, match="requires a binary treatment"):
            payload = forest.average_treatment_effect()
        assert payload["method"] == "plug_in"
        assert payload["plug_in_reason"] == "non_binary_treatment"

    def test_continuous_estimate_tracks_the_mean_cate(self, card):
        forest = _forest(card, "educ", discrete=False)
        mean_cate = float(np.mean(forest.effect(card)))
        with pytest.warns(AssumptionWarning):
            payload = forest.average_treatment_effect()
        # The regression: the AIPW score returned -1266 here. Any estimate
        # that is not the plug-in mean of the fitted effects is the bug.
        assert payload["estimate"] == pytest.approx(mean_cate, rel=1e-10)
        assert abs(payload["estimate"]) < 1.0

    def test_printed_effect_labels_the_descriptive_standard_error(self, card):
        forest = _forest(card, "educ", discrete=False)
        with pytest.warns(AssumptionWarning):
            effect = forest.ate()
        assert "descriptive SE" in str(effect)

    def test_binary_treatment_still_uses_the_aipw_score(self, card):
        forest = _forest(card, "nearc4", discrete=True)
        payload = forest.average_treatment_effect()
        assert payload["method"] == "aipw"
        assert "plug_in_reason" not in payload
        # AIPW and the plug-in mean should be in the same neighbourhood
        # once the propensity really is a probability.
        mean_cate = float(np.mean(forest.effect(card)))
        assert abs(payload["estimate"] - mean_cate) < 0.5

    def test_binary_treatment_effect_prints_the_aipw_label(self, card):
        forest = _forest(card, "nearc4", discrete=True)
        assert "descriptive SE" not in str(forest.ate())

    @pytest.mark.parametrize("target", ["all", "treated"])
    def test_guard_applies_to_every_reachable_target_sample(self, card, target):
        forest = _forest(card, "educ", discrete=False)
        with pytest.warns(AssumptionWarning):
            payload = forest.average_treatment_effect(target_sample=target)
        assert payload["method"] == "plug_in"
        assert abs(payload["estimate"]) < 1.0

    def test_atc_on_a_continuous_treatment_raises_before_the_guard(self, card):
        """``target_sample='control'`` needs ``T == 0`` rows; nobody has zero
        years of schooling, so this fails loudly upstream of the AIPW guard
        rather than aggregating over an empty group."""
        forest = _forest(card, "educ", discrete=False)
        with pytest.raises(sp.DataInsufficient, match="no control observations"):
            forest.average_treatment_effect(target_sample="control")
