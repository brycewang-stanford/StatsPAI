"""Round-4 coverage margin: synth.compare (synth_compare / SynthComparison).

Runs a real multi-method comparison on the California tobacco panel and
exercises the SynthComparison export surface (summary / to_latex /
to_markdown / to_excel / plot) plus the ``_recommend`` tie-break
branches. All file output via ``tmp_path``; figures use the Agg backend.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.synth import compare as _compare  # noqa: E402


@pytest.fixture(scope="module")
def comparison():
    df = sp.california_tobacco()
    return _compare.synth_compare(
        data=df,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        methods=["classic", "augmented"],
        placebo=False,
    )


def test_synth_compare_table_and_recommendation(comparison):
    assert not comparison.comparison_table.empty
    assert comparison.recommended in comparison.results
    assert isinstance(comparison.recommendation_reason, str)
    table = comparison.comparison_table.sort_values("method")
    # Both methods recover the well-known negative Prop-99 ATT. The classic SCM
    # weight vector solves a quadratic program whose optimum is platform-
    # sensitive (different BLAS backends converge to different but near-optimal
    # donor weights — deterministic per platform, ~17% apart macOS vs Linux on
    # this fixture), so the exact ATT is not pinned here. We assert sign and a
    # plausible magnitude; numerical parity for the canonical estimators is
    # guarded by tests/reference_parity/.
    atts = table["att"].to_numpy()
    assert np.isfinite(atts).all()
    assert (atts < 0).all(), "Prop 99 ATT should be negative (sales fell)"
    assert ((atts > -30) & (atts < -5)).all()


def test_synth_comparison_summary_repr(comparison):
    s = comparison.summary()
    assert isinstance(s, str) and len(s) > 0
    assert str(comparison)  # __str__
    assert repr(comparison)  # __repr__


def test_synth_comparison_to_latex_markdown(comparison):
    assert "begin{tabular}" in comparison.to_latex() or "\\\\" in comparison.to_latex()
    md = comparison.to_markdown()
    assert isinstance(md, str) and "|" in md


def test_synth_comparison_to_excel(tmp_path, comparison):
    out = tmp_path / "comparison.xlsx"
    path = comparison.to_excel(str(out))
    assert out.exists()
    assert str(out) in str(path)


def test_synth_comparison_plot(comparison):
    fig = comparison.plot()
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close("all")


def test_synth_recommend_returns_name():
    df = sp.california_tobacco()
    name = sp.synth_recommend(
        data=df,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        methods=["classic"],
    )
    assert isinstance(name, str)
    assert name == "classic"
    np.testing.assert_allclose([len(name), int(name == "classic")], [7, 1])


# --- _recommend branch coverage (direct, with crafted tables) ---


def test_recommend_empty_table():
    rec, reason = _compare._recommend(pd.DataFrame())
    assert rec == "classic"
    assert "No methods" in reason


def test_recommend_all_zero_rmspe_keeps_all():
    # min_rmspe == 0 -> "cannot filter meaningfully; keep all".
    table = pd.DataFrame(
        {
            "method": ["classic", "augmented"],
            "pre_rmspe": [0.0, 0.0],
            "ci_lower": [-1.0, -2.0],
            "ci_upper": [1.0, 2.0],
            "simplicity_rank": [1, 2],
        }
    )
    rec, reason = _compare._recommend(table)
    assert rec == "classic"  # simpler wins the tiebreak
    assert isinstance(reason, str)


def test_recommend_nan_rmspe_keeps_all():
    table = pd.DataFrame(
        {
            "method": ["classic"],
            "pre_rmspe": [np.nan],
            "ci_lower": [-1.0],
            "ci_upper": [1.0],
            "simplicity_rank": [1],
        }
    )
    rec, _ = _compare._recommend(table)
    assert rec == "classic"


# --- _extract_n_effective_donors branch coverage ---


class _R:
    def __init__(self, mi):
        self.model_info = mi


def test_extract_n_effective_donors_dict():
    r = _R({"donor_weights": {"a": 0.5, "b": 0.001, "c": 0.4}})
    assert _compare._extract_n_effective_donors(r) == 2


def test_extract_n_effective_donors_array():
    r = _R({"donor_weights": np.array([0.5, 0.0, 0.3])})
    assert _compare._extract_n_effective_donors(r) == 2


def test_extract_n_effective_donors_fallback():
    # The size of the donor pool is not a count of donors with weight.
    r = _R({"n_donors": 7})
    assert np.isnan(_compare._extract_n_effective_donors(r))


def test_extract_n_effective_donors_weight_frames():
    frame = pd.DataFrame({"unit": list("abc"), "weight": [0.6, 0.4, 0.0]})
    assert _compare._extract_n_effective_donors(_R({"weights": frame})) == 2
    assert _compare._extract_n_effective_donors(_R({"unit_weights": frame})) == 2
    # A frame with no weight column is passed over, not miscounted.
    assert np.isnan(
        _compare._extract_n_effective_donors(_R({"weights": frame[["unit"]]}))
    )
