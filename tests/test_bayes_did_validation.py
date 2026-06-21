"""PyMC-free validation tests for Bayesian DID input contracts."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from statspai.bayes.did import _prepare_did_frame, bayes_did
from statspai.exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)


def _did_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1.0, 1.5, 2.0, 3.0],
            "treat": [0, 0, 1, 1],
            "post": [0, 1, 0, 1],
            "unit": ["a", "a", "b", "b"],
            "time": [0, 1, 0, 1],
            "x": [0.1, 0.2, 0.3, 0.4],
            "cohort": ["never", "never", "treated", "treated"],
        }
    )


def test_prepare_did_frame_missing_column_uses_taxonomy() -> None:
    with pytest.raises(MethodIncompatibility, match="not found") as exc_info:
        _prepare_did_frame(
            _did_frame(),
            y="missing",
            treat="treat",
            post="post",
            unit=None,
            time=None,
            covariates=None,
        )

    exc = exc_info.value
    assert isinstance(exc, ValueError)
    assert exc.diagnostics["column"] == "missing"
    assert "y" in exc.diagnostics["available_columns"]


def test_public_bayes_did_validates_bad_input_before_pymc_import() -> None:
    with pytest.raises(MethodIncompatibility, match="not found"):
        bayes_did(
            _did_frame(),
            y="missing",
            treat="treat",
            post="post",
            draws=1,
            tune=1,
            chains=1,
        )


def test_prepare_did_frame_rejects_non_dataframe() -> None:
    with pytest.raises(MethodIncompatibility, match="DataFrame"):
        _prepare_did_frame(
            {"y": [1.0]},
            y="y",
            treat="treat",
            post="post",
            unit=None,
            time=None,
            covariates=None,
        )


def test_prepare_did_frame_rejects_nonbinary_treat() -> None:
    df = _did_frame()
    df["treat"] = [0, 2, 1, 1]

    with pytest.raises(MethodIncompatibility, match="binary") as exc_info:
        _prepare_did_frame(
            df,
            y="y",
            treat="treat",
            post="post",
            unit=None,
            time=None,
            covariates=None,
        )

    assert exc_info.value.diagnostics["unique_values"] == [0.0, 1.0, 2.0]


def test_prepare_did_frame_rejects_thin_complete_case_sample() -> None:
    df = _did_frame()
    df.loc[:2, "y"] = np.nan

    with pytest.raises(DataInsufficient, match="at least 4 observations"):
        _prepare_did_frame(
            df,
            y="y",
            treat="treat",
            post="post",
            unit=None,
            time=None,
            covariates=None,
        )


@pytest.mark.parametrize(
    ("column", "kwargs", "match"),
    [
        ("unit", {"unit": "unit"}, "Unit column"),
        ("time", {"time": "time"}, "Time column"),
        ("cohort", {"cohort": "cohort"}, "cohort column"),
    ],
)
def test_prepare_did_frame_rejects_single_panel_or_cohort_level(
    column: str,
    kwargs: Dict[str, str],
    match: str,
) -> None:
    df = _did_frame()
    df[column] = "only"

    with pytest.raises(DataInsufficient, match=match):
        _prepare_did_frame(
            df,
            y="y",
            treat="treat",
            post="post",
            unit=kwargs.get("unit"),
            time=kwargs.get("time"),
            covariates=None,
            cohort=kwargs.get("cohort"),
        )


@pytest.mark.parametrize(
    ("column", "covariates"),
    [
        ("y", None),
        ("x", ["x"]),
    ],
)
def test_prepare_did_frame_rejects_nonfinite_numeric_values(
    column: str,
    covariates: Optional[List[str]],
) -> None:
    df = _did_frame()
    df.loc[0, column] = np.inf

    with pytest.raises(NumericalInstability, match="NaN or infinite"):
        _prepare_did_frame(
            df,
            y="y",
            treat="treat",
            post="post",
            unit=None,
            time=None,
            covariates=covariates,
        )


def test_prepare_did_frame_accepts_scalar_covariate_name() -> None:
    prepared = _prepare_did_frame(
        _did_frame(),
        y="y",
        treat="treat",
        post="post",
        unit="unit",
        time="time",
        covariates="x",
        cohort="cohort",
    )

    assert prepared["X"].shape == (4, 1)
    assert prepared["covariates"] == ["x"]
    assert prepared["n_units"] == 2
    assert prepared["n_times"] == 2
