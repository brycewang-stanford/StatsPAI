import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _survival_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(133)
    x = rng.normal(size=n)
    treat = (x + rng.normal(scale=0.2, size=n) > 0).astype(int)
    time = 1.0 + rng.exponential(scale=1.0 + 0.2 * treat, size=n)
    event = rng.binomial(1, 0.8, size=n)
    return pd.DataFrame(
        {
            "time": time,
            "event": event,
            "treat": treat,
            "x": x,
        }
    )


def test_causal_survival_forest_accepts_scalar_covariate() -> None:
    df = _survival_frame()
    res = sp.causal_survival_forest(
        df,
        time="time",
        event="event",
        treat="treat",
        covariates="x",
        n_trees=5,
        min_leaf=2,
        random_state=133,
    )

    assert res.n_obs == len(df)
    assert res.cate.shape == (len(df),)
    assert np.isfinite(res.ate_rmst)


def test_causal_survival_forest_rejects_missing_column_with_taxonomy() -> None:
    df = _survival_frame()
    with pytest.raises(MethodIncompatibility, match="Missing columns"):
        sp.causal_survival_forest(
            df,
            time="time",
            event="event",
            treat="missing",
            covariates="x",
            n_trees=5,
            min_leaf=2,
        )


def test_causal_survival_forest_rejects_bad_options_with_taxonomy() -> None:
    df = _survival_frame()
    with pytest.raises(MethodIncompatibility, match="propensity_bounds"):
        sp.causal_survival_forest(
            df,
            time="time",
            event="event",
            treat="treat",
            covariates="x",
            propensity_bounds=(0.9, 0.2),
        )
    with pytest.raises(MethodIncompatibility, match="horizon"):
        sp.causal_survival_forest(
            df,
            time="time",
            event="event",
            treat="treat",
            covariates="x",
            horizon=0.0,
        )


def test_causal_survival_forest_rejects_single_arm_with_taxonomy() -> None:
    df = _survival_frame()
    df["treat"] = 1
    with pytest.raises(DataInsufficient, match="both treatment arms"):
        sp.causal_survival_forest(
            df,
            time="time",
            event="event",
            treat="treat",
            covariates="x",
            n_trees=5,
            min_leaf=2,
        )
