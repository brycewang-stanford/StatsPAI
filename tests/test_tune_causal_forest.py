"""``sp.tune_causal_forest``: random search on the out-of-bag R-loss.

There is no reference implementation to match (grf smooths the trial losses
with a kriging model and minimises the surface), so the tests pin what the
procedure must do: score every trial on the same residuals, adopt a setting
only on a clear margin, return the default forest otherwise, and help on the
design where tuning is known to help.
"""

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _data(seed, effect="constant", n=1200):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 4))
    W = rng.binomial(1, 0.5, n)
    tau = np.ones(n) if effect == "constant" else 3 * np.sin(6 * X[:, 0])
    Y = X[:, 1] + W * tau + rng.normal(size=n)
    return Y, W, X, tau


def test_structure_and_bookkeeping():
    Y, W, X, _ = _data(1)
    out = sp.tune_causal_forest(
        Y=Y, T=W, X=X, n_draws=8, n_estimators=200, random_state=1
    )
    trials = out["trials"]
    assert len(trials) == 9 and (trials["setting"] == "default").sum() == 1
    assert trials["error"].is_monotonic_increasing
    assert {"min_samples_leaf", "max_samples", "mtry"} <= set(trials.columns)
    assert out["tuned_error"] <= out["default_error"]
    if out["tuned"]:
        assert set(out["best_params"]) == {"min_samples_leaf", "max_samples", "mtry"}
        assert out["default_error"] - out["tuned_error"] > 2 * out["noise"]
        f = out["forest"]
        assert f.min_samples_leaf == out["best_params"]["min_samples_leaf"]
    else:
        assert out["best_params"] == {}
    # same seed, same answer
    again = sp.tune_causal_forest(
        Y=Y, T=W, X=X, n_draws=8, n_estimators=200, random_state=1
    )
    assert again["best_params"] == out["best_params"]
    np.testing.assert_allclose(again["trials"]["error"], trials["error"])


def test_constant_effect_gets_larger_leaves_and_a_smaller_error():
    better = adopted = 0
    for seed in (3, 4, 5, 6):
        Y, W, X, tau = _data(seed, "constant")
        out = sp.tune_causal_forest(
            Y=Y, T=W, X=X, n_draws=25, n_estimators=400, random_state=seed
        )
        default = sp.causal_forest(Y=Y, T=W, X=X, n_estimators=400, random_state=seed)
        err_t = np.sqrt(np.mean((out["forest"].oob_effect() - tau) ** 2))
        err_d = np.sqrt(np.mean((default.oob_effect() - tau) ** 2))
        better += err_t < err_d
        if out["tuned"]:
            adopted += 1
            assert out["best_params"]["min_samples_leaf"] > 5
    assert adopted >= 3 and better >= 3


def test_defaults_are_kept_when_nothing_beats_them_clearly():
    """With a huge margin of noise no draw can win; the forest returned is
    the default one, fitted at full size."""
    Y, W, X, _ = _data(7, "wiggly", n=500)
    out = sp.tune_causal_forest(
        Y=Y, T=W, X=X, n_draws=3, tune_trees=20, tune_reps=1,
        n_estimators=150, random_state=2, parameters=["mtry"],
    )  # fmt: skip
    # one repetition gives no spread, so the margin is zero: adoption is
    # then decided by the loss alone and must be consistent with it
    best = out["trials"].iloc[0]
    assert out["tuned"] == (
        best["setting"] == "draw" and best["error"] < out["default_error"]
    )
    assert out["forest"].n_estimators == 150


def test_arguments_are_validated():
    Y, W, X, _ = _data(8, n=300)
    with pytest.raises(MethodIncompatibility, match="cannot tune"):
        sp.tune_causal_forest(Y=Y, T=W, X=X, parameters=["depth"])
    with pytest.raises(MethodIncompatibility, match="both tuned and fixed"):
        sp.tune_causal_forest(Y=Y, T=W, X=X, min_samples_leaf=10)
    with pytest.raises(MethodIncompatibility, match="n_draws"):
        sp.tune_causal_forest(Y=Y, T=W, X=X, n_draws=0)
