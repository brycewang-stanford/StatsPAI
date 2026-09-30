"""Stata parity for Thornton (2008) and the randomization-inference path.

Reference values from **Stata 18 MP** on the same bundled CSV::

    use thornton_hiv.dta, clear
    summarize got if any==1     // 0.789235640, n=2211
    summarize got if any==0     // 0.338683788, n=623
    reg got any, robust         // b=0.450551852  se=0.020857971  N=2834

The point estimate is deterministic and pinned exactly.  The RI p-value
is a Monte Carlo quantity, so it is tested for the property that matters
(the observed effect lies outside the entire permutation distribution)
rather than pinned to a draw.
"""

from __future__ import annotations

import numpy as np
import pytest

import statspai as sp

STATA_TREATED_MEAN = 0.789235640
STATA_CONTROL_MEAN = 0.338683788
STATA_SDO = 0.450551852
STATA_OLS = (0.450551852, 0.020857971)

ATOL = 1e-6


@pytest.fixture(scope="module")
def hiv():
    return sp.datasets.thornton_hiv(complete_case=True)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


def test_full_and_complete_case_shapes():
    full = sp.datasets.thornton_hiv()
    assert full.shape == (4820, 17)
    assert full.attrs["complete_case"] is False

    cc = sp.datasets.thornton_hiv(complete_case=True)
    assert len(cc) == 2834
    assert not cc[["got", "any"]].isna().any().any()


def test_treatment_arms_match_stata_counts(hiv):
    assert int((hiv["any"] == 1).sum()) == 2211
    assert int((hiv["any"] == 0).sum()) == 623
    assert set(hiv["any"].unique()) == {0.0, 1.0}


# ---------------------------------------------------------------------------
# The deterministic estimate
# ---------------------------------------------------------------------------


def test_group_means_match_stata(hiv):
    treated = float(hiv.loc[hiv["any"] == 1, "got"].mean())
    control = float(hiv.loc[hiv["any"] == 0, "got"].mean())
    assert treated == pytest.approx(STATA_TREATED_MEAN, abs=ATOL)
    assert control == pytest.approx(STATA_CONTROL_MEAN, abs=ATOL)
    assert treated - control == pytest.approx(STATA_SDO, abs=ATOL)


def test_ols_reproduces_the_simple_difference(hiv):
    """For a binary regressor the OLS slope *is* the difference in means."""
    r = sp.regress("got ~ any", data=hiv, robust="hc1")
    assert float(r.params["any"]) == pytest.approx(STATA_OLS[0], abs=ATOL)
    assert float(r.std_errors["any"]) == pytest.approx(STATA_OLS[1], abs=ATOL)

    treated = float(hiv.loc[hiv["any"] == 1, "got"].mean())
    control = float(hiv.loc[hiv["any"] == 0, "got"].mean())
    assert float(r.params["any"]) == pytest.approx(treated - control, abs=1e-9)


# ---------------------------------------------------------------------------
# Randomization inference
# ---------------------------------------------------------------------------


def test_ri_observed_statistic_is_the_sdo(hiv):
    """RI changes the p-value, never the point estimate."""
    ri = sp.ri_test(hiv, y="got", treat="any", n_perms=200, seed=42)
    assert ri["observed"] == pytest.approx(STATA_SDO, abs=ATOL)


def test_ri_point_estimate_is_seed_invariant(hiv):
    """Permutation draws move the p-value, not the observed statistic."""
    a = sp.ri_test(hiv, y="got", treat="any", n_perms=100, seed=1)
    b = sp.ri_test(hiv, y="got", treat="any", n_perms=100, seed=999)
    assert a["observed"] == pytest.approx(b["observed"], abs=1e-12)


def test_observed_effect_lies_outside_the_permutation_distribution(hiv):
    """The property worth asserting, rather than a pinned Monte Carlo draw.

    A 45-point effect on a 34-point base is far outside anything
    reshuffling the labels produces, so the sharp null is rejected at the
    resolution the permutation count allows.
    """
    ri = sp.ri_test(hiv, y="got", treat="any", n_perms=500, seed=42)
    perms = np.asarray(ri["perm_distribution"], dtype=float)

    assert len(perms) > 0
    assert ri["observed"] > perms.max()
    assert ri["p_value"] <= 1.0 / len(perms)
    # Reshuffled labels should centre on zero — a sanity check on the
    # permutation machinery itself, not on the data.
    assert abs(float(perms.mean())) < 0.05


def test_ri_supports_a_distributional_statistic(hiv):
    """`stat='ks'` runs the same randomization with a KS statistic.

    The Mixtape has a separate `ks.py` for this; in StatsPAI it is an
    argument to the same function.
    """
    ks = sp.ri_test(hiv, y="got", treat="any", stat="ks", n_perms=200, seed=42)
    assert ks["observed"] > 0
    assert 0.0 <= ks["p_value"] <= 1.0


def test_clustered_ri_refuses_a_treatment_assigned_within_villages(hiv):
    """Thornton randomized the incentive to individuals, not villages.

    Until 1.32 ``cluster='villnum'`` quietly gave every village the incentive
    status of its first respondent and permuted those invented village
    values, a null distribution of a design nobody ran. Now it refuses and
    says how to state the design.
    """
    usable = hiv.dropna(subset=["villnum", "got", "any"])
    assert usable.groupby("villnum")["any"].nunique().max() > 1
    with pytest.warns(UserWarning, match="dropped 4 of 2834"):
        with pytest.raises(sp.exceptions.MethodIncompatibility, match="varies within"):
            sp.ri_test(
                hiv, y="got", treat="any", cluster="villnum", n_perms=200, seed=42
            )


def test_unclustered_ri_does_not_warn_on_a_clean_sample(hiv):
    """The warning must be specific to a real drop, not fire on every call."""
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        sp.ri_test(hiv, y="got", treat="any", n_perms=50, seed=42)
    assert [c for c in caught if "ri_test: dropped" in str(c.message)] == []


# ---------------------------------------------------------------------------
# Replication-guide wiring
# ---------------------------------------------------------------------------


def test_replication_entry_registered_and_renders():
    table = sp.list_replications()
    row = table.loc[table["key"] == "thornton_2008"]
    assert len(row) == 1
    assert bool(row["has_real_data"].iloc[0])

    _data, guide = sp.replicate("thornton_2008")
    assert "randomization inference" in guide.lower()
    # The p = 0 caveat must reach the reader.
    assert "p < 1/n_perms" in guide or "1/n_perms" in guide
