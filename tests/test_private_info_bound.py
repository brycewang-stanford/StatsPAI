"""sp.private_info_bound: Hendren's (2013) lower bound on private information.

Known-truth designs: in a linear probability model with a uniform private
component the bound has a closed form; in a probit with a normal garbling
of the private signal, ``P_Z = E[P | X, Z]`` is itself a probit in ``Z``, so
the population value of the bound is computed from the true functions on a
large draw, and Proposition 2 (bound below the true average magnitude) is
visible. Parity with the author's Stata code is in
tests/reference_parity/test_private_info_bound_stata_parity.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import statspai as sp
from statspai.adverse_selection.elicitation import _tail_m
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _linear_design(seed: int, n: int = 20000, h: float = 0.15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=n)
    v = rng.uniform(-h, h, size=n)
    p = 0.3 + 0.2 * x + v
    return pd.DataFrame(
        {
            "loss": (rng.uniform(size=n) < p).astype(int),
            "z": (v + h) / (2 * h),
            "x": x,
        }
    )


A, B, C, S_E = -0.8, 0.5, 0.7, 0.8


def _probit_design(seed: int, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x, v = rng.normal(size=n), rng.normal(size=n)
    z = v + rng.normal(scale=S_E, size=n)
    p = norm.cdf(A + B * x + C * v)
    return pd.DataFrame(
        {"loss": (rng.uniform(size=n) < p).astype(int), "z": z, "x": x, "v": v}
    )


def _probit_oracles(n: int = 400_000) -> tuple:
    """Population bound (garbled Z) and the true average magnitude (P)."""
    rng = np.random.default_rng(99)
    x, v = rng.normal(size=n), rng.normal(size=n)
    z = v + rng.normal(scale=S_E, size=n)
    px = norm.cdf((A + B * x) / np.sqrt(1 + C**2))
    k = 1.0 / (1.0 + S_E**2)  # E[V | Z] = k Z, Var[V | Z] = 1 - k
    pz = norm.cdf((A + B * x + C * k * z) / np.sqrt(1 + C**2 * (1 - k)))
    p = norm.cdf(A + B * x + C * v)
    return float(_tail_m(pz - px).mean()), float(_tail_m(p - px).mean())


def test_linear_design_recovers_the_closed_form():
    # P - Pr{L | X} = V ~ U(-h, h), so E[m(P)] = E[(h - V) / 2] = h / 2, and
    # with Z a linear function of V the bound is attained.
    res = sp.private_info_bound(
        _linear_design(0), "loss", "z", ["x"], model="linear", focal=()
    )
    # SD over 8 seeds at n = 20000 is 0.0018; 4 SD.
    assert abs(res.estimate - 0.075) < 0.0072
    assert res.table.loc["all", "wald_p"] < 1e-10


def test_probit_garbling_recovers_the_population_bound_below_the_truth():
    bound, truth = _probit_oracles()
    assert bound < truth - 0.03  # Proposition 2 is not tight here
    res = sp.private_info_bound(_probit_design(1, 20000), "loss", ["z"], ["x"])
    # SD over 6 seeds is 0.0020; 4 SD.
    assert abs(res.estimate - bound) < 0.008
    assert res.estimate < truth
    # Knowing the true signal attains the true magnitude.
    full = sp.private_info_bound(_probit_design(1, 20000), "loss", ["v"], ["x"])
    assert abs(full.estimate - truth) < 0.01


def test_tail_mean_includes_ties_and_ignores_row_order():
    r = np.array([0.3, -0.1, 0.3, 0.0, -0.1, 0.5])
    hand = np.array([np.mean(r[r >= ri]) - ri for ri in r])
    np.testing.assert_allclose(_tail_m(r), hand, rtol=0, atol=1e-15)
    rng = np.random.default_rng(3)
    n = 1500
    df = pd.DataFrame(
        {
            "loss": rng.integers(0, 2, n),
            "z": rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=n),
            "x": rng.integers(0, 3, n).astype(float),
        }
    )
    a = sp.private_info_bound(df, "loss", "z", ["x"], model="linear")
    b = sp.private_info_bound(
        df.sample(frac=1.0, random_state=1), "loss", "z", ["x"], model="linear"
    )
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)


def test_saturated_basis_is_invariant_to_relabelling_the_elicitation():
    # With three support points, T1, T2 and the focal indicators span every
    # function of Z, so P_Z (and the bound) cannot change under Z -> 1 - Z.
    rng = np.random.default_rng(4)
    n = 3000
    v = rng.choice([0.0, 0.5, 1.0], size=n)
    x = rng.normal(size=n)
    p = norm.cdf(-0.5 + 0.4 * x + 0.8 * v)
    df = pd.DataFrame({"loss": (rng.uniform(size=n) < p).astype(int), "z": v, "x": x})
    a = sp.private_info_bound(df, "loss", "z", ["x"])
    b = sp.private_info_bound(df.assign(z=1.0 - df["z"]), "loss", "z", ["x"])
    assert a.estimate == pytest.approx(b.estimate, rel=1e-9)
    assert a.table.loc["all", "wald_stat"] == pytest.approx(
        b.table.loc["all", "wald_stat"], rel=1e-7
    )


def test_cells_compute_m_within_each_cell():
    df = _linear_design(5, n=4000)
    df["cell"] = np.repeat([1, 2], 2000)
    res = sp.private_info_bound(
        df, "loss", "z", ["x"], model="linear", focal=(), cells="cell"
    )
    rows = res.rows
    hand = np.concatenate(
        [
            _tail_m(rows["residual"].to_numpy()[df["cell"].to_numpy() == g])
            for g in (1, 2)
        ]
    )
    np.testing.assert_allclose(rows["m_z"].to_numpy(), hand, rtol=0, atol=1e-15)
    assert res.estimate == pytest.approx(hand.mean(), rel=1e-13)


def test_compare_delta_and_conservative_pvalue():
    rng = np.random.default_rng(6)
    n = 3000
    reject = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    v = rng.normal(size=n) * np.where(reject == 1, 0.9, 0.2)
    p = norm.cdf(-1 + 0.4 * x + v)
    df = pd.DataFrame(
        {
            "loss": (rng.uniform(size=n) < p).astype(int),
            "z": norm.cdf(v + rng.normal(scale=0.3, size=n)),
            "x": x,
            "reject": reject,
            "hh": np.arange(n) // 2,
        }
    )
    kw = dict(classes="reject", compare=(1, 0), cluster="hh", n_boot=60, seed=2)
    res = sp.private_info_bound(df, "loss", "z", ["x"], **kw)
    t = res.table
    d = res.delta
    assert d["estimate"] == pytest.approx(
        t.loc[1, "estimate"] - t.loc[0, "estimate"], rel=1e-14
    )
    assert d["estimate"] > 0.05 and d["p_boot"] == 0.0
    assert d["pvalue"] == max(d["p_boot"], d["p_wald"])
    assert d["p_wald"] == t.loc[1, "wald_p"]
    assert np.isfinite(t["se"]).all() and (t["se"] > 0).all()
    again = sp.private_info_bound(df, "loss", "z", ["x"], **kw)
    assert again.table["se"].tolist() == t["se"].tolist()


def test_errors():
    df = _linear_design(7, n=500)
    with pytest.raises(MethodIncompatibility):
        sp.private_info_bound(df, "loss", "z", ["x"], model="logit")
    with pytest.raises(MethodIncompatibility):
        sp.private_info_bound(df.assign(z=df["z"] * 300), "loss", "z", ["x"])
    with pytest.raises(MethodIncompatibility):
        sp.private_info_bound(df, "loss", ["z"], ["x"], interact="x")
    with pytest.raises(MethodIncompatibility):
        sp.private_info_bound(df, "loss", "z", ["x"], compare=(1, 0))
    with pytest.raises(MethodIncompatibility):
        sp.private_info_bound(df.assign(loss=df["loss"] * 2), "loss", "z", ["x"])
    with pytest.raises(DataInsufficient):
        sp.private_info_bound(df.assign(loss=1), "loss", "z", ["x"])
    with pytest.raises(MethodIncompatibility):
        sp.private_info_bound(df, "loss", "z", ["x"], n_boot=-1)


@pytest.mark.slow
def test_bootstrap_se_tracks_the_monte_carlo_sd():
    est, ses = [], []
    for s in range(40):
        res = sp.private_info_bound(
            _probit_design(100 + s, 2000), "loss", ["z"], ["x"], n_boot=99, seed=s
        )
        est.append(res.estimate)
        ses.append(res.table.loc["all", "se"])
    sd = float(np.std(est, ddof=1))
    assert abs(np.mean(ses) / sd - 1) < 0.35
