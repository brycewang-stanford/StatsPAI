"""Tests for ``sp.diversity_index``.

Every index is checked against a hand-computed value on a tiny community
where the arithmetic is transparent, plus the identities that tie the
family together (Hill numbers nest richness / Shannon / inverse Simpson).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.fixture
def records() -> pd.DataFrame:
    """Two counties: A = (2, 1, 1) over 3 species, B = (3, 1) over 2."""
    return pd.DataFrame(
        {
            "county": ["A"] * 4 + ["B"] * 4,
            "ym": [1] * 8,
            "species": ["s1", "s1", "s2", "s3", "s1", "s1", "s1", "s2"],
        }
    )


def _shannon(p):
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def test_shannon_matches_hand_computation(records):
    out = sp.diversity_index(records, species="species", by="county")
    np.testing.assert_allclose(out.loc["A"], _shannon([2, 1, 1]), rtol=1e-12)
    np.testing.assert_allclose(out.loc["B"], _shannon([3, 1]), rtol=1e-12)


def test_richness_and_evenness(records):
    out = sp.diversity_index(
        records, species="species", by="county", index=["richness", "pielou"]
    )
    assert out.loc["A", "richness"] == 3
    assert out.loc["B", "richness"] == 2
    np.testing.assert_allclose(
        out.loc["A", "pielou"], _shannon([2, 1, 1]) / np.log(3), rtol=1e-12
    )
    # Evenness is scale-free: both counties have 4 records, but A spreads
    # them over more species, so it is both richer and more even here.
    assert out.loc["A", "pielou"] > out.loc["B", "pielou"]


def test_simpson_family_identities(records):
    out = sp.diversity_index(
        records,
        species="species",
        by="county",
        index=["simpson", "gini_simpson", "inv_simpson"],
    )
    np.testing.assert_allclose(out["simpson"] + out["gini_simpson"], 1.0, rtol=1e-12)
    np.testing.assert_allclose(out["inv_simpson"], 1.0 / out["simpson"], rtol=1e-12)
    np.testing.assert_allclose(out.loc["A", "simpson"], 0.375, rtol=1e-12)


@pytest.mark.parametrize(
    "q,other", [(0.0, "richness"), (1.0, None), (2.0, "inv_simpson")]
)
def test_hill_numbers_nest_the_other_indices(records, q, other):
    hill = sp.diversity_index(
        records, species="species", by="county", index="hill", q=q
    )
    if other is None:  # q = 1 is exp(Shannon)
        shannon = sp.diversity_index(records, species="species", by="county")
        np.testing.assert_allclose(hill, np.exp(shannon), rtol=1e-12)
    else:
        ref = sp.diversity_index(records, species="species", by="county", index=other)
        np.testing.assert_allclose(hill, ref, rtol=1e-12)


def test_count_column_and_record_rows_agree(records):
    long = (
        records.groupby(["county", "species"], observed=True)
        .size()
        .reset_index(name="n")
    )
    from_records = sp.diversity_index(records, species="species", by="county")
    from_counts = sp.diversity_index(long, species="species", count="n", by="county")
    pd.testing.assert_series_equal(from_records, from_counts, check_names=False)


def test_matrix_input_matches_long_input(records):
    mat = np.array([[2.0, 1.0, 1.0], [3.0, 1.0, 0.0]])
    from_mat = sp.diversity_index(mat, index="shannon")
    from_long = sp.diversity_index(records, species="species", by="county")
    np.testing.assert_allclose(from_mat.to_numpy(), from_long.to_numpy(), rtol=1e-12)


def test_multi_key_grouping_returns_panel_index(records):
    out = sp.diversity_index(
        records, species="species", by=["county", "ym"], index="all"
    )
    assert list(out.index.names) == ["county", "ym"]
    assert ("A", 1) in out.index
    assert "n_records" in out.columns
    assert out.loc[("A", 1), "n_records"] == 4


def test_min_records_blanks_thin_cells(records):
    thin = pd.concat(
        [records, pd.DataFrame({"county": ["C"], "ym": [1], "species": ["s9"]})]
    )
    out = sp.diversity_index(
        thin, species="species", by="county", index=["shannon"], min_records=2
    )
    assert np.isnan(out.loc["C", "shannon"])
    assert np.isfinite(out.loc["A", "shannon"])


def test_single_species_site_has_zero_shannon_and_undefined_evenness():
    one = pd.DataFrame({"site": ["A", "A"], "species": ["s1", "s1"]})
    out = sp.diversity_index(
        one, species="species", by="site", index=["shannon", "pielou", "richness"]
    )
    assert out.loc["A", "shannon"] == 0.0
    assert out.loc["A", "richness"] == 1
    # Evenness of a single species is 0/0 -- report NaN, not a spurious 1.
    assert np.isnan(out.loc["A", "pielou"])


def test_log_base_changes_only_the_units(records):
    nat = sp.diversity_index(records, species="species", by="county")
    bits = sp.diversity_index(records, species="species", by="county", base=2)
    np.testing.assert_allclose(bits, nat / np.log(2), rtol=1e-12)


def test_unknown_index_raises():
    df = pd.DataFrame({"species": ["a", "b"]})
    with pytest.raises(MethodIncompatibility, match="Unknown diversity index"):
        sp.diversity_index(df, species="species", index="entropy")


def test_all_missing_raises():
    df = pd.DataFrame({"county": ["A", "B"], "species": [None, None]})
    with pytest.raises(DataInsufficient):
        sp.diversity_index(df, species="species", by="county")


def test_registered_for_agents():
    assert "diversity_index" in sp.list_functions()
    schema = sp.function_schema("diversity_index")
    assert schema["name"] == "diversity_index"
