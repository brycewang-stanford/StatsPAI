"""Tests for the optional gsynth R backend."""

import subprocess

import numpy as np
import pytest

import statspai as sp
from statspai.synth.gsynth import _find_rscript


def test_gsynth_native_matches_reference_fixture():
    result = sp.gsynth(
        sp.datasets.basque_terrorism(),
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        backend="native",
        seed=42,
        placebo=False,
    )
    assert np.isclose(result.estimate, -0.32417115086183, rtol=1e-6)
    assert np.isclose(
        result.model_info["pre_treatment_rmse"],
        0.043094139385699,
        rtol=1e-6,
    )
    assert result.model_info["n_factors"] == 1
    assert result.model_info["backend"] == "native"


def _skip_unless_gsynth_available():
    try:
        rscript = _find_rscript()
    except RuntimeError:
        pytest.skip("Rscript is not installed")
    probe = subprocess.run(
        [
            rscript,
            "-e",
            (
                "quit(status = as.integer("
                "!requireNamespace('gsynth', quietly=TRUE) || "
                "!requireNamespace('jsonlite', quietly=TRUE)))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("R packages gsynth/jsonlite are not installed")


def test_gsynth_backend_matches_reference_fixture():
    _skip_unless_gsynth_available()
    result = sp.gsynth(
        sp.datasets.basque_terrorism(),
        outcome="gdppc",
        unit="region",
        time="year",
        treated_unit="Basque Country",
        treatment_time=1970,
        backend="gsynth",
        seed=42,
    )
    assert np.isclose(result.estimate, -0.32417115086183)
    assert result.model_info["n_factors"] == 1
    assert np.isclose(result.model_info["pre_treatment_rmse"], 0.043094139385699)
    assert result.model_info["backend"] == "gsynth"


def test_gsynth_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        sp.gsynth(
            sp.datasets.basque_terrorism(),
            outcome="gdppc",
            unit="region",
            time="year",
            treated_unit="Basque Country",
            treatment_time=1970,
            backend="unknown",
        )


def _small_panel():
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = [
        dict(u=i, t=t, y=rng.normal() + 0.1 * t + 2.0 * (i == 1 and t >= 10))
        for i in range(12)
        for t in range(15)
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("treated", [[1, 2], (1, 2), np.array([1, 2])])
def test_gsynth_takes_several_treated_units(treated):
    """A list of treated units is the same call as a treatment column."""
    df = _small_panel()
    listed = sp.gsynth(
        df,
        outcome="y",
        unit="u",
        time="t",
        treated_unit=treated,
        treatment_time=10,
        n_factors=1,
        inference="none",
    )
    df["d"] = (df["u"].isin([1, 2]) & (df["t"] >= 10)).astype(int)
    column = sp.gsynth(
        df, outcome="y", unit="u", time="t", treat="d", n_factors=1, inference="none"
    )
    assert listed.estimate == column.estimate
    assert listed.model_info["n_treated_units"] == 2


def test_gsynth_r_backend_still_takes_one_treated_unit():
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="single treated unit"):
        sp.gsynth(
            _small_panel(),
            outcome="y",
            unit="u",
            time="t",
            treated_unit=[1, 2],
            treatment_time=10,
            backend="r",
        )


def test_gsynth_unknown_treated_unit_is_named():
    from statspai.exceptions import DataInsufficient

    with pytest.raises(DataInsufficient, match="not found in column 'u'"):
        sp.gsynth(
            _small_panel(),
            outcome="y",
            unit="u",
            time="t",
            treated_unit=99,
            treatment_time=10,
            placebo=False,
        )
