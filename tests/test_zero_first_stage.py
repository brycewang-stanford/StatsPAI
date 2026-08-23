"""Tests for ``sp.zero_first_stage`` (ZFS exclusion test).

The design is a simulation where the truth is known by construction: the
instrument has a first stage only outside the "desert" subsample, and the
size of its direct effect on the outcome is a knob. The test recovers the
direct effect where there is one and returns a tight interval around zero
where there is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

TRUE_BETA = -0.5


def _simulate(gamma: float, n: int = 1600, seed: int = 0) -> pd.DataFrame:
    """Instrument inert in the ZFS subsample; direct effect ``gamma``."""
    rng = np.random.default_rng(seed)
    desert = rng.integers(0, 2, size=n).astype(bool)
    z = rng.normal(size=n)
    u = rng.normal(size=n)  # confounder
    d = np.where(desert, 0.0, 0.9 * z) + 0.6 * u + rng.normal(size=n)
    y = TRUE_BETA * d + gamma * z + 0.7 * u + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "z": z,
            "desert": desert,
            "unit": rng.integers(0, 40, size=n),
            "x": rng.normal(size=n),
        }
    )


def test_premise_holds_in_the_zfs_subsample():
    out = sp.zero_first_stage(
        _simulate(0.0), y="y", endog="d", instrument="z", zfs="desert", n_boot=0
    )
    # First stage ~0 where it should be, strong where it should be.
    assert abs(out.first_stage_zfs) < 0.1
    assert out.first_stage_main > 0.7
    assert not out.diagnostics["first_stage_not_zero"]


def test_detects_a_real_exclusion_violation():
    out = sp.zero_first_stage(
        _simulate(0.25), y="y", endog="d", instrument="z", zfs="desert", n_boot=0
    )
    assert out.reduced_form_zfs_pvalue < 0.01
    np.testing.assert_allclose(out.reduced_form_zfs, 0.25, atol=0.08)
    assert "EXCLUSION VIOLATED" in out.verdict()


def test_does_not_cry_wolf_when_exclusion_holds():
    out = sp.zero_first_stage(
        _simulate(0.0), y="y", endog="d", instrument="z", zfs="desert", n_boot=0
    )
    assert out.reduced_form_zfs_pvalue > 0.05
    assert "No detectable direct effect" in out.verdict()


def test_correction_recovers_the_true_effect_under_pleiotropy():
    """The naive IV estimate is biased; the corrected one is not."""
    out = sp.zero_first_stage(
        _simulate(0.25),
        y="y",
        endog="d",
        instrument="z",
        zfs="desert",
        n_boot=200,
        random_state=1,
    )
    naive_error = abs(out.beta_iv - TRUE_BETA)
    corrected_error = abs(out.beta_zfs_corrected - TRUE_BETA)
    assert corrected_error < naive_error / 3
    np.testing.assert_allclose(out.beta_zfs_corrected, TRUE_BETA, atol=0.1)
    lo, hi = out.beta_zfs_corrected_ci
    assert lo < TRUE_BETA < hi


def test_bootstrap_reports_how_many_replications_survived():
    out = sp.zero_first_stage(
        _simulate(0.0),
        y="y",
        endog="d",
        instrument="z",
        zfs="desert",
        n_boot=50,
        random_state=0,
    )
    assert 0 < out.n_boot <= 50
    assert np.isfinite(out.beta_zfs_corrected_se)


def test_no_bootstrap_leaves_the_corrected_se_missing():
    out = sp.zero_first_stage(
        _simulate(0.0), y="y", endog="d", instrument="z", zfs="desert", n_boot=0
    )
    assert out.n_boot == 0
    assert np.isnan(out.beta_zfs_corrected_se)


def test_premise_failure_is_flagged_not_silently_used():
    """A subsample that still has a first stage cannot test exclusion."""
    rng = np.random.default_rng(3)
    n = 1200
    grp = rng.integers(0, 2, size=n).astype(bool)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    # "ZFS" group keeps most of the first stage -- the premise is false.
    d = np.where(grp, 0.8 * z, 0.9 * z) + 0.6 * u + rng.normal(size=n)
    y = TRUE_BETA * d + 0.7 * u + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "z": z, "grp": grp})
    out = sp.zero_first_stage(df, y="y", endog="d", instrument="z", zfs="grp", n_boot=0)
    assert out.diagnostics["first_stage_not_zero"]
    assert "PREMISE FAILS" in out.verdict()


def test_accepts_controls_absorb_and_cluster():
    df = _simulate(0.0)
    out = sp.zero_first_stage(
        df,
        y="y",
        endog="d",
        instrument="z",
        zfs="desert",
        exog=["x"],
        absorb=["unit"],
        cluster="unit",
        n_boot=30,
        random_state=0,
    )
    assert np.isfinite(out.reduced_form_zfs)
    assert np.isfinite(out.first_stage_main)
    assert out.n_main + out.n_zfs == len(df)


def test_boolean_mask_and_column_agree():
    df = _simulate(0.0)
    a = sp.zero_first_stage(
        df, y="y", endog="d", instrument="z", zfs="desert", n_boot=0
    )
    b = sp.zero_first_stage(
        df,
        y="y",
        endog="d",
        instrument="z",
        zfs=df["desert"].to_numpy(),
        n_boot=0,
    )
    np.testing.assert_allclose(a.reduced_form_zfs, b.reduced_form_zfs, rtol=1e-12)


def test_missing_column_raises():
    df = _simulate(0.0)
    with pytest.raises(MethodIncompatibility, match="not in `data`"):
        sp.zero_first_stage(
            df, y="y", endog="d", instrument="z", zfs="not_a_column", n_boot=0
        )


def test_tiny_subsample_raises():
    df = _simulate(0.0, n=60)
    df["only_five"] = False
    df.loc[df.index[:5], "only_five"] = True
    with pytest.raises(DataInsufficient, match="at least"):
        sp.zero_first_stage(
            df, y="y", endog="d", instrument="z", zfs="only_five", n_boot=0
        )


def test_summary_and_registration():
    out = sp.zero_first_stage(
        _simulate(0.0), y="y", endog="d", instrument="z", zfs="desert", n_boot=0
    )
    text = out.summary()
    assert "Zero-first-stage" in text
    assert "direct effect" in text
    assert "zero_first_stage" in sp.list_functions()
