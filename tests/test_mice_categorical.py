"""``sp.mice`` with categorical and binary variables (review E06, 1.32).

Before 1.32 a non-numeric variable was imputed by ``'sample'`` -- random
draws from its own marginal -- and never used as a predictor, which
attenuates every association with it; ``'logreg'`` imputed from the MLE
itself (an improper imputation). Now two-level variables get proper
``'logreg'`` (parameters drawn from their asymptotic posterior, as R
``mice.impute.logreg``) and unordered categoricals get proper ``'polyreg'``;
categorical columns enter the other equations as dummies.

Evidence is simulation (known DGP), since MI is stochastic: on a MAR design
where the categorical's effect is 1.0, polyreg recovers it (60 reps: mean
0.999, MC se 0.023, 95 % CI coverage 98 %) and the old ``'sample'`` route
returns 0.37 with 5 % coverage. The test below reruns a smaller version.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _dgp(seed, n=400):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    eta = np.column_stack([np.zeros(n), 0.8 * x, -0.8 * x])
    P = np.exp(eta)
    P /= P.sum(1, keepdims=True)
    g = np.array(["a", "b", "c"])[(rng.random(n)[:, None] > np.cumsum(P, 1)).sum(1)]
    y = 1 + 0.5 * x + np.select([g == "b", g == "c"], [1.0, -1.0], 0.0)
    y = y + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "g": g})
    miss = rng.random(n) < 1 / (1 + np.exp(-(-1 + 0.8 * y)))  # MAR on y
    df.loc[miss, "g"] = np.nan
    return df


def _coef(df, method, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mr = sp.mice(df, m=5, max_iter=5, method={"g": method}, seed=seed)
        c = sp.mi_estimate(mr, sp.regress, formula="y ~ x + C(g)")
    i = c["var_names"].index("C(g)[T.b]")
    return c["params"][i], c["ci_lower"][i] <= 1.0 <= c["ci_upper"][i]


@pytest.mark.slow
def test_polyreg_recovers_categorical_effect_sample_does_not():
    reps = 20
    poly = [_coef(_dgp(s), "polyreg", s) for s in range(reps)]
    samp = [_coef(_dgp(s), "sample", s) for s in range(reps)]
    est = np.array([p[0] for p in poly])
    assert abs(est.mean() - 1.0) < 3 * est.std(ddof=1) / np.sqrt(reps) + 0.02
    assert np.mean([p[1] for p in poly]) >= 0.8
    assert np.mean([s[0] for s in samp]) < 0.6  # attenuated


def test_default_methods_follow_variable_type():
    df = _dgp(1)
    df["b"] = np.where(np.random.default_rng(0).random(len(df)) < 0.5, "yes", "no")
    df.loc[df.index[:30], "b"] = np.nan
    df.loc[df.index[30:50], "x"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mr = sp.mice(df, m=2, max_iter=2, seed=0)
    assert mr.methods == {"g": "polyreg", "b": "logreg", "x": "pmm"}
    done = mr.complete(0)
    assert done.notna().all().all()
    assert set(done["g"]) <= {"a", "b", "c"}
    assert set(done["b"]) <= {"yes", "no"}
    assert mr.fit_failures == []


def test_categorical_with_numeric_method_is_refused():
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="categorical"):
        sp.mice(_dgp(2), m=2, method={"g": "pmm"})


def test_logreg_draws_parameters():
    """Proper imputation: with the data fixed, two imputations of a binary
    variable differ by more than Bernoulli noise around one fitted p."""
    from statspai.imputation.mice import _impute_logreg

    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 1))
    y = (rng.random(300) < 1 / (1 + np.exp(-x[:, 0]))).astype(float)
    xm = np.zeros((20000, 1))
    means = [
        _impute_logreg(y, x, xm, np.random.default_rng(s)).mean() for s in range(40)
    ]
    # sd of the imputed share across draws >> binomial sd at n = 20000
    assert np.std(means) > 3 * np.sqrt(0.25 / 20000)


def test_high_cardinality_strings_are_not_predictors():
    df = _dgp(3)
    df["id"] = [f"u{i}" for i in range(len(df))]
    with pytest.warns(UserWarning, match="not used as predictors"):
        mr = sp.mice(df, m=2, max_iter=2, seed=0)
    assert mr.complete(0)["g"].notna().all()
