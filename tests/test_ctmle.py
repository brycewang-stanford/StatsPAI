"""``sp.ctmle``: behaviour that does not need a reference implementation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility

COV = ["x1", "z", "x3"]


def _data(n: int = 800, seed: int = 0, binary: bool = False) -> pd.DataFrame:
    """x1 confounds, z drives treatment only, x3 predicts the outcome only."""
    rng = np.random.default_rng(seed)
    x1, z, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.6 * x1 + 2.0 * z))))
    lin = 1.0 * a + x1 + 0.5 * x3
    if binary:
        y = rng.binomial(1, 1 / (1 + np.exp(-(lin - 0.5)))).astype(float)
    else:
        y = lin + rng.normal(size=n)
    return pd.DataFrame({"y": y, "a": a, "x1": x1, "z": z, "x3": x3})


def _fit(df: pd.DataFrame, **kw):
    binary = set(df["y"].unique()) <= {0.0, 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ctmle(
            df,
            y="y",
            treat="a",
            covariates=COV,
            outcome_library=[
                (
                    # Unpenalised and converged, so relabelling the arms
                    # gives the same fit to round-off.
                    LogisticRegression(penalty=None, tol=1e-12, max_iter=10000)
                    if binary
                    else LinearRegression()
                )
            ],
            n_folds=2,
            **kw,
        )


def test_instrument_stays_out_when_the_outcome_model_is_right() -> None:
    res = _fit(_data())
    assert "z" not in res.model_info["selected_covariates"]
    assert abs(res.estimate - 1.0) < 4 * res.se
    # Leaving the instrument out keeps the propensities away from 0 and 1.
    assert res.model_info["propensity_min"] > 0.05
    assert res.model_info["propensity_max"] < 0.95


def test_confounder_the_outcome_model_missed_is_selected() -> None:
    df = _data(seed=1)
    q_wrong = LinearRegression().fit(df[["a", "x3"]], df["y"])
    q = np.column_stack(
        [
            q_wrong.predict(df[["a", "x3"]].assign(a=0)),
            q_wrong.predict(df[["a", "x3"]].assign(a=1)),
        ]
    )
    res = sp.ctmle(df, y="y", treat="a", covariates=COV, Q=q)
    assert res.model_info["candidate_order"][0] == "x1"
    assert "x1" in res.model_info["selected_covariates"]
    naive = float(np.mean(q[:, 1] - q[:, 0]))
    assert abs(res.estimate - 1.0) < abs(naive - 1.0)


def test_detail_lists_every_step_and_the_estimate_is_the_selected_row() -> None:
    res = _fit(_data())
    table = res.detail
    assert list(table["step"]) == list(range(len(COV) + 1))
    assert table["added"].iloc[0] == "(intercept)"
    assert sorted(table["added"].iloc[1:]) == sorted(COV)
    assert table["selected"].sum() == 1
    row = table[table["selected"]].iloc[0]
    assert row["step"] == res.model_info["step"]
    assert row["estimate"] == pytest.approx(res.estimate, rel=1e-12)
    assert row["cv_criterion"] == table["cv_criterion"].min()
    np.testing.assert_allclose(
        table["cv_criterion"],
        table["cv_rss"] + table["cv_variance"] + len(_data()) * table["cv_bias"] ** 2,
        rtol=1e-12,
    )


def test_influence_function_gives_the_standard_error() -> None:
    df = _data()
    res = _fit(df)
    ic = res.model_info["influence_function"]
    assert ic.shape == (len(df),)
    assert abs(ic.mean()) < 1e-8
    np.testing.assert_allclose(res.se, ic.std(ddof=1) / np.sqrt(len(df)), rtol=1e-12)


def test_only_the_ate_is_offered() -> None:
    for estimand in ("ATT", "ATC", "RR"):
        with pytest.raises(MethodIncompatibility, match="only 'ATE'"):
            _fit(_data(n=200), estimand=estimand)
    assert _fit(_data(n=200), estimand="ate").estimand == "ATE"


def test_binary_outcome_estimate_stays_in_range() -> None:
    res = _fit(_data(binary=True))
    assert -1.0 <= res.estimate <= 1.0
    assert res.model_info["outcome_type"] == "binary"


def test_penalty_none_is_the_residual_sum_of_squares() -> None:
    table = _fit(_data(), penalty="none").detail
    np.testing.assert_allclose(table["cv_criterion"], table["cv_rss"], rtol=0)


def test_bootstrap_includes_the_selection() -> None:
    df = _data(n=400, seed=3)
    plain = _fit(df)
    boot = _fit(df, se_method="bootstrap", n_boot=40)
    info = boot.model_info
    assert boot.estimate == plain.estimate
    assert info["se_influence"] == plain.se
    assert info["bootstrap_estimates"].shape == (40,)
    assert set(info["selection_frequency"]) <= set(COV)
    assert boot.ci[0] < boot.ci[1]
    np.testing.assert_allclose(
        boot.se, info["bootstrap_estimates"].std(ddof=1), rtol=1e-12
    )


def test_cluster_folds_and_standard_error() -> None:
    df = _data()
    df["g"] = np.arange(len(df)) // 8
    res = _fit(df, cluster="g")
    assert res.model_info["se_method"] == "cluster_efficient_influence_function"
    assert res.se > 0


def test_invalid_input_is_refused() -> None:
    df = _data(n=200)
    with pytest.raises(MethodIncompatibility, match="penalty"):
        _fit(df, penalty="aic")
    with pytest.raises(MethodIncompatibility, match="se_method"):
        _fit(df, se_method="jackknife")
    q = np.column_stack([np.zeros(200), np.ones(200)])
    with pytest.raises(MethodIncompatibility, match="resample"):
        sp.ctmle(df, y="y", treat="a", covariates=COV, Q=q, se_method="bootstrap")
    with pytest.raises(MethodIncompatibility, match="not found"):
        sp.ctmle(df, y="y", treat="a", covariates=["nope"])
