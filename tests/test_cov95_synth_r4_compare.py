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
    np.testing.assert_allclose(
        table["att"].to_numpy(),
        [-15.748268, -13.085166],
        atol=5e-7,
    )


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
    r = _R({"n_donors": 7})
    assert _compare._extract_n_effective_donors(r) == 7
