"""Every JIVE entry point uses the leave-one-out fit as an instrument.

Through 1.39.3 three of them regressed the outcome on the leave-one-out
fitted values instead, (Xh'Xh)^{-1} Xh'y, which is attenuated toward zero
when there are many instruments: worse than the 2SLS bias JIVE exists to
remove, and in the other direction.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.iv import jive_mw


def _many_weak(seed: int, n: int = 300, k: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, k))
    u = rng.normal(size=n)
    v = 0.8 * u + 0.6 * rng.normal(size=n)
    x = z @ np.full(k, 0.08) + v
    df = pd.DataFrame(z, columns=[f"z{i}" for i in range(k)])
    df["x"] = x
    df["y"] = 1.0 * x + u
    return df


def _formula(df: pd.DataFrame) -> str:
    zs = [c for c in df.columns if c.startswith("z")]
    return "y ~ 1 + (x ~ " + " + ".join(zs) + ")"


def _aik_jive1(df: pd.DataFrame) -> float:
    """(Xh'X)^{-1} Xh'y written out, Angrist-Imbens-Krueger (1999) eq. 9."""
    zs = [c for c in df.columns if c.startswith("z")]
    n = len(df)
    zc = np.column_stack([np.ones(n), df[zs].to_numpy()])
    hat = zc @ np.linalg.solve(zc.T @ zc, zc.T)
    h = np.diag(hat)
    x = df["x"].to_numpy()
    xh = (hat @ x - h * x) / (1 - h)
    big_x = np.column_stack([np.ones(n), x])
    big_xh = np.column_stack([np.ones(n), xh])
    return float(np.linalg.solve(big_xh.T @ big_x, big_xh.T @ df["y"].to_numpy())[1])


@pytest.mark.parametrize("method", ["jive", "jive1"])
def test_jive1_is_the_aik_estimator(method):
    df = _many_weak(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.iv(_formula(df), df, method=method)
    # Same closed form; the slack is the conditioning of a 31-column
    # projection.
    assert float(fit.params["x"]) == pytest.approx(_aik_jive1(df), abs=1e-9)


def test_many_weak_jive_is_the_aik_estimator():
    df = _many_weak(0)
    zs = [c for c in df.columns if c.startswith("z")]
    fit = jive_mw(df, y="y", endog="x", instruments=zs)
    assert fit.estimate == pytest.approx(_aik_jive1(df), abs=1e-9)


@pytest.mark.parametrize("method", ["jive", "jive1", "ujive", "ijive"])
def test_jive_variants_are_centred_on_the_truth_under_many_instruments(method):
    # 30 instruments of strength 0.08 each, n = 300, true coefficient 1.
    # Over 60 draws 2SLS has median 1.27 and the old "OLS on the
    # leave-one-out fit" had median 0.59; every JIVE variant is within 0.1
    # of 1 (their medians are 0.98 to 1.00 with an interquartile range of
    # about 0.3, so 0.1 is three standard errors of a median of 60 draws).
    est = []
    for seed in range(60):
        df = _many_weak(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.append(float(sp.iv(_formula(df), df, method=method).params["x"]))
    assert float(np.median(est)) == pytest.approx(1.0, abs=0.1)


def test_many_weak_jive_interval_covers_the_truth():
    zs = [f"z{i}" for i in range(30)]
    hits = 0
    for seed in range(120):
        df = _many_weak(seed)
        lo, hi = jive_mw(df, y="y", endog="x", instruments=zs).ci
        hits += lo <= 1.0 <= hi
    # Nominal 95%. With 120 draws the binomial standard error is 0.02, so
    # 0.88 is more than three of them below; the old estimator covered 5%.
    assert hits / 120 > 0.88
