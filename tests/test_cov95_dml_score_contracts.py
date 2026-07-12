"""Coverage tests for sp.dml score / fold-index contracts.

Targets uncovered error branches in ``statspai.dml.plr`` and
``statspai.dml._base``: the IV-type score rejects ``sample_weight``, and
custom ``fold_indices`` are validated for fold count. These must raise
``MethodIncompatibility`` loudly (CLAUDE.md section 7).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _dgp(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3))
    d = X @ np.array([0.5, -0.3, 0.2]) + rng.normal(0, 1, n)
    y = 1.5 * d + X @ np.array([1.0, 0.5, -0.2]) + rng.normal(0, 1, n)
    df = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    df["d"] = d
    df["y"] = y
    return df


KW = dict(y="y", d="d", X=["x0", "x1", "x2"])


def test_iv_type_score_rejects_sample_weight():
    df = _dgp()
    with pytest.raises(MethodIncompatibility):
        sp.dml(df, model="plr", score="IV-type",
               sample_weight=np.ones(len(df)), **KW)


def test_fold_indices_fold_count_mismatch_rejected():
    df = _dgp()
    rng = np.random.default_rng(1)
    df = df.copy()
    df["fold"] = rng.integers(0, 3, len(df))  # only 3 distinct folds
    with pytest.raises(MethodIncompatibility):
        sp.dml(df, model="plr", fold_indices="fold", n_folds=5, **KW)


def test_partialling_out_with_sample_weight_runs():
    # The default score *does* accept weights: exercise the accepted path.
    df = _dgp()
    rng = np.random.default_rng(2)
    w = rng.uniform(0.5, 1.5, len(df))
    res = sp.dml(df, model="plr", score="partialling out", sample_weight=w, **KW)
    assert np.isfinite(float(res.estimate))
