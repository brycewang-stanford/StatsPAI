"""Coverage round-3 (final) — long-tail branches of ``statspai.synth.exports``.

Targets the still-uncovered field-extractor and table-builder branches:
significance stars, Series / ndarray+donor_names weight conventions, NaN
CI / SE cells, gap-table reconstruction from ``Y_treated``/``Y_synth``,
multi-method comparison where some (or all) methods do not expose donor
weights, and the Excel weights-pivot fall-throughs.

We drive these via the public ``sp.synth_to_latex / _markdown / _excel``
entry points feeding hand-built :class:`CausalResult` objects with
crafted ``model_info`` — this is an exercise of the *formatting* code,
not of any estimator numerics (no numbers are fabricated as estimator
output; the assertions check structural properties of the rendered
tables only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult


def _result(method, estimate, se, pvalue, ci, model_info):
    return CausalResult(
        method=method,
        estimand="ATT",
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=0.05,
        n_obs=100,
        model_info=model_info,
    )


def _gap_mi(treatment_time=11, n_t=20, weights=None, **extra):
    """A model_info dict with reconstructable gap series + weights."""
    times = list(range(1, n_t + 1))
    yt = np.array(
        [1.0 + 0.2 * t + (4.0 if t >= treatment_time else 0.0) for t in times]
    )
    ys = np.array([1.0 + 0.2 * t for t in times])
    mi = {
        "method": "custom",
        "Y_treated": yt,
        "Y_synth": ys,
        "times": times,
        "treatment_time": treatment_time,
        "n_pre_periods": treatment_time - 1,
        "n_post_periods": n_t - treatment_time + 1,
        "n_donors": 5,
    }
    if weights is not None:
        mi["donor_weights"] = weights
    mi.update(extra)
    return mi


# ===========================================================================
# Field extractors: weight storage conventions + stars
# ===========================================================================
def test_latex_weights_as_pandas_series():
    # Series weights -> _donor_weights Series branch (L157-158)
    w = pd.Series({"d0": 0.6, "d1": 0.3, "d2": 0.1})
    r = _result("classic", 4.0, 0.5, 0.001, (3.0, 5.0), _gap_mi(weights=w))
    out = sp.synth_to_latex(r, show_weights=True, show_ci=True)
    assert "d0" in out
    assert "$^{***}$" in out  # p<0.01 stars


def test_latex_weights_as_ndarray_with_donor_names():
    # ndarray weights paired with donor_names list (L171-175)
    mi = _gap_mi()
    mi["donor_weights"] = np.array([0.5, 0.3, 0.2])
    mi["donor_names"] = ["alpha", "beta", "gamma"]
    r = _result("ndarray_sc", 2.0, 0.4, 0.03, (1.0, 3.0), mi)
    out = sp.synth_to_markdown(r, show_weights=True)
    assert "alpha" in out
    assert "**" in out  # p<0.05 stars (markdown)


def test_latex_weights_dataframe_two_columns():
    mi = _gap_mi()
    mi["donor_weights"] = pd.DataFrame({"unit": ["a", "b", "c"], "w": [0.7, 0.2, 0.1]})
    r = _result("df_sc", 1.5, 0.6, 0.2, (0.0, 3.0), mi)
    out = sp.synth_to_latex(r, show_weights=True)
    # p>0.1 -> no stars on the estimate cell
    assert "1.5000" in out


def test_stars_one_star_and_no_se():
    # p<0.1 single star + NaN SE branch (L259-260)
    r = _result(
        "sc", 3.0, float("nan"), 0.08, (np.nan, np.nan), _gap_mi(weights={"d0": 1.0})
    )
    out = sp.synth_to_latex(r, show_ci=True)
    assert "$^{*}$" in out
    # NaN SE row renders an em-dash; NaN CI also em-dash
    assert "—" in out


# ===========================================================================
# Comparison mode: some / all methods missing weights
# ===========================================================================
def test_latex_comparison_some_methods_lack_weights():
    r_w = _result(
        "classic", 4.0, 0.5, 0.01, (3.0, 5.0), _gap_mi(weights={"d0": 0.6, "d1": 0.4})
    )
    r_now = _result(
        "nodonors", 3.5, 0.6, 0.02, (2.3, 4.7), _gap_mi()
    )  # no weights at all
    out = sp.synth_to_latex([r_w, r_now], show_weights=True, show_ci=True)
    assert "classic" in out and "nodonors" in out
    assert "Donor 1" in out  # per_method weights panel built (L488+)


def test_markdown_comparison_method_without_weights():
    r_w = _result(
        "classic", 4.0, 0.5, 0.01, (3.0, 5.0), _gap_mi(weights={"d0": 0.7, "d1": 0.3})
    )
    r_now = _result("nodonors", 3.5, 0.6, 0.02, (2.3, 4.7), _gap_mi())
    out = sp.synth_to_markdown([r_w, r_now], show_weights=True)
    assert "weights not exposed by this method" in out  # L630


def test_markdown_comparison_nan_ci_and_fit():
    # CI NaN -> em-dash (L587), fit NaN -> em-dash (L615-616)
    mi = {"method": "bare", "n_donors": 4}
    r = _result("bare", 2.0, 0.5, 0.04, (np.nan, np.nan), mi)
    out = sp.synth_to_markdown([r, r], show_ci=True)
    assert "—" in out


# ===========================================================================
# Excel export: weights present, weights absent (all-missing pivot)
# ===========================================================================
def test_excel_with_weights(tmp_path):
    r = _result(
        "classic", 4.0, 0.5, 0.01, (3.0, 5.0), _gap_mi(weights={"d0": 0.6, "d1": 0.4})
    )
    path = tmp_path / "synth_weights.xlsx"
    out = sp.synth_to_excel(r, str(path))
    assert path.exists()
    xl = pd.ExcelFile(out)
    assert "Summary" in xl.sheet_names
    assert "Weights" in xl.sheet_names


def test_excel_no_weights_empty_pivot(tmp_path):
    # No donor weights on any method -> weights_pivot = empty columns (L733)
    r = _result("bare", 2.0, 0.5, 0.04, (1.0, 3.0), _gap_mi())
    path = tmp_path / "synth_bare.xlsx"
    out = sp.synth_to_excel(r, str(path))
    assert path.exists()
    xl = pd.ExcelFile(out)
    assert "Summary" in xl.sheet_names


def test_exports_reject_non_result_in_list():
    with pytest.raises(TypeError):
        sp.synth_to_latex([1, 2, 3])


# ===========================================================================
# Fit-quality bands (good / acceptable / poor) — L129-134
# ===========================================================================
def _fit_band_mi(rmspe_frac):
    """Pre-treatment treated with large spread; gap sized to hit a band.

    Treated pre-period spans 0..10 (sd ~ 3.16). The synthetic deviates
    from treated by a constant ``delta`` each pre-period so pre-RMSPE
    == delta; we set delta = rmspe_frac * sd to land in the target band.
    """
    n_pre = 11
    times = list(range(1, 21))
    yt_pre = np.arange(n_pre, dtype=float)  # 0..10
    sd = float(np.std(yt_pre))
    delta = rmspe_frac * sd
    yt = np.concatenate([yt_pre, np.arange(n_pre, 20) + 5.0])
    ys = yt.copy()
    ys[:n_pre] = yt_pre - delta  # constant pre-period gap
    return {
        "method": "band",
        "Y_treated": yt,
        "Y_synth": ys,
        "times": times,
        "treatment_time": n_pre + 1,
        "n_donors": 4,
    }


@pytest.mark.parametrize("frac", [0.07, 0.15, 0.30])
def test_fit_quality_bands(frac):
    r = _result("band", 3.0, 0.5, 0.04, (2.0, 4.0), _fit_band_mi(frac))
    out = sp.synth_to_latex(r, show_ci=False)
    assert "Fit" in out  # one of good/acceptable/poor labels rendered


# ===========================================================================
# n_effective_donors from model_info float (no explicit weights) — L275-277
# ===========================================================================
def test_n_effective_donors_from_float_model_info():
    mi = _gap_mi()  # no donor_weights
    mi["n_active_donors"] = 3.0  # float -> rounded to int (L276-277)
    r = _result("noweights", 2.5, 0.5, 0.04, (1.5, 3.5), mi)
    out = sp.synth_to_latex(r)
    assert "Effective donors" in out


# ===========================================================================
# Pre-RMSPE NaN -> em-dash float fmt (L431) + gap-None Excel skip (L763)
# ===========================================================================
def test_latex_pre_rmspe_nan():
    # No gap series and no rmspe field -> pre_rmspe NaN -> em-dash row.
    mi = {"method": "barest", "n_donors": 4}
    r = _result("barest", 2.0, 0.5, 0.04, (1.0, 3.0), mi)
    out = sp.synth_to_latex([r, r], show_ci=False)
    assert "Pre-RMSPE" in out


def test_excel_gap_none_skipped(tmp_path):
    mi = {"method": "nogap", "n_donors": 4, "donor_weights": {"d0": 1.0}}
    r = _result("nogap", 2.0, 0.5, 0.04, (1.0, 3.0), mi)
    path = tmp_path / "nogap.xlsx"
    out = sp.synth_to_excel(r, str(path))
    xl = pd.ExcelFile(out)
    # No Gap_ sheet because gap is None
    assert not any(s.startswith("Gap_") for s in xl.sheet_names)


# ===========================================================================
# method_names length mismatch + SynthComparison input — L796-817
# ===========================================================================
def test_method_names_length_mismatch_raises():
    r = _result("a", 1.0, 0.1, 0.04, (0.5, 1.5), _gap_mi())
    with pytest.raises(ValueError):
        sp.synth_to_latex([r, r], method_names=["only_one"])


def test_synth_comparison_input_path():
    from statspai.synth.compare import SynthComparison

    r1 = _result(
        "m1", 4.0, 0.5, 0.01, (3.0, 5.0), _gap_mi(weights={"d0": 0.6, "d1": 0.4})
    )
    r2 = _result(
        "m2", 3.5, 0.6, 0.02, (2.3, 4.7), _gap_mi(weights={"d0": 0.5, "d1": 0.5})
    )
    comp = SynthComparison(
        results={"m1": r1, "m2": r2},
        comparison_table=pd.DataFrame(
            {"method": ["m1", "m2"], "pre_rmspe": [0.1, 0.2]}
        ),
        recommended="m1",
        recommendation_reason="lowest pre-RMSPE",
    )
    out = sp.synth_to_latex(comp, show_ci=True)
    assert "m1" in out and "m2" in out
