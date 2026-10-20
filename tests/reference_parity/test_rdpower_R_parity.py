"""Reference parity: ``sp.rdpower`` / ``sp.rdsampsi`` vs R ``rdpower`` 3.0.

Fixture: ``_fixtures/_generate_rdpower_R.R`` -> ``rdpower_R.json`` (R
``rdpower()`` / ``rdsampsi()`` on ``rdsenate.csv``, run as a black box). The
module is a clean-room implementation of Cattaneo, Titiunik and Vazquez-Bare
(2019, ``cattaneo2019power``); the spec is ``docs/dev/cleanroom/rdpower.md``.

* ``rdpower(data=)``: robust bias-corrected standard error and power, rel 1e-9
  (observed ~1e-12), for default and ``p = 2``, uniform kernel with HC1, a
  non-zero cutoff, ``alpha = 0.01``, ``msetwo`` bandwidths and user ``h``.
* ``rdsampsi(data=)``: window sample sizes ``M-``, ``M+``, ``M`` are integers
  and must agree exactly, for six effect sizes, ``beta = 0.9``,
  ``alpha = 0.1``, ``p = 2``, uniform/HC1, ``msetwo`` and a non-zero cutoff.

Reference divergence: R's solver does not return for some inputs (fixture cell
``sampsi_tau05`` records the timeout); the root is bracketed here, so a
number is returned, and it is checked against the defining equation instead.
"""

from __future__ import annotations

import json
import math
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

FX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((FX / "rdpower_R.json").read_text(encoding="utf-8"))
SENATE = pd.read_csv(FX / "rdsenate.csv")

_ARG = {"cutoff": "c", "beta": "target_power"}


def _kwargs(cell):
    out = {}
    for k, v in R[cell]["args"].items():
        out[_ARG.get(k, k)] = tuple(v) if isinstance(v, list) else v
    return out


def _cells(prefix):
    return sorted(k for k, v in R.items() if k.startswith(prefix) and "error" not in v)


def _power(tau, se, alpha):
    z = stats.norm.ppf(1 - alpha / 2)
    return 1 - stats.norm.cdf(tau / se + z) + stats.norm.cdf(tau / se - z)


@pytest.mark.parametrize("cell", _cells("power_"))
def test_rdpower_data_mode_matches_R(cell):
    ref = R[cell]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdpower(data=SENATE, y="vote", x="margin", **_kwargs(cell))
    assert res.se == pytest.approx(ref["se.rbc"], rel=1e-9)
    assert res.power == pytest.approx(ref["power.rbc"], rel=1e-9)
    assert (res.n_left, res.n_right) == (ref["N.l"], ref["N.r"])


@pytest.mark.parametrize("cell", _cells("sampsi_"))
def test_rdsampsi_data_mode_matches_R_exactly(cell):
    ref = R[cell]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdsampsi(data=SENATE, y="vote", x="margin", **_kwargs(cell))
    assert (res.n_left, res.n_right, res.n_total) == (
        ref["sampsi.h.l"],
        ref["sampsi.h.r"],
        ref["sampsi.h.tot"],
    )


def test_fixture_records_the_reference_solver_timeout():
    assert "error" in R["sampsi_tau05"]


def test_rdsampsi_returns_where_reference_does_not():
    """tau = 0.5 on the Senate data, where R does not return. Recomputed
    from the paper's equations with R's own tau-free ingredients of the
    tau = 3 cell (scaled variances, side and window counts, bandwidth):
    n* = ceil(root of power(0.5; sqrt(A / m)) = 0.8), A = V+/h + V-/h, and
    the sides ceil(rho n*/f), ceil((1 - rho) n*/f)."""
    from scipy.optimize import brentq

    ref = R["sampsi_tau3"]
    v_l, v_r = ref["var.l"], ref["var.r"]
    a = v_r / ref["samph.r"] + v_l / ref["samph.l"]
    root = brentq(lambda m: _power(0.5, math.sqrt(a / m), 0.05) - 0.8, 1.0, 1e9)
    n_star = math.ceil(root)
    rho = math.sqrt(v_r) / (math.sqrt(v_l) + math.sqrt(v_r))
    f = rho * ref["N.r"] / ref["Nh.r"] + (1 - rho) * ref["N.l"] / ref["Nh.l"]
    m_r, m_l = math.ceil(rho * n_star / f), math.ceil((1 - rho) * n_star / f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdsampsi(tau=0.5, data=SENATE, y="vote", x="margin")
    assert (res.n_left, res.n_right) == (m_l, m_r)


# ------------------------------------------------------------ design mode


def test_design_mode_power_formula():
    """se^2 = 0.35 (v_l/(n_l h_l) + v_r/(n_r h_r)); power is the two-sided
    z-test power at that se."""
    r = sp.rdpower(
        tau=0.2,
        var_left=2.0,
        var_right=0.5,
        n_left=300,
        n_right=700,
        h_left=0.4,
        h_right=0.8,
        alpha=0.1,
    )
    se = math.sqrt(0.35 * (2.0 / (300 * 0.4) + 0.5 / (700 * 0.8)))
    assert r.se == pytest.approx(se, rel=1e-15)
    assert r.power == pytest.approx(_power(0.2, se, 0.1), rel=1e-14)
    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.8)
    assert r.mde == pytest.approx(z * se, rel=1e-14)


@pytest.mark.parametrize("ratio", [0.33, 0.7, 1.0, 1.5])
def test_design_mode_sample_size_is_root_rounded_up(ratio):
    """n_left = ceil(root of power(n, ratio n) = target), n_right =
    ceil(ratio n_left)."""
    tau, target = 0.15, 0.8
    res = sp.rdsampsi(tau=tau, ratio=ratio, target_power=target)

    def se(nl):
        return math.sqrt(0.35 * (1 / (nl * 0.5) + 1 / (ratio * nl * 0.5)))

    root = res.n_left
    assert _power(tau, se(root), 0.05) >= target
    assert _power(tau, se(root - 1), 0.05) < target
    assert res.n_right == math.ceil(ratio * res.n_left)
    assert res.n_total == res.n_left + res.n_right


def test_design_mode_uses_both_tails():
    """At a target power close to alpha the opposite tail matters: the
    returned n is the smallest reaching the target with the exact power."""
    res = sp.rdsampsi(tau=0.2, alpha=0.2, target_power=0.3)
    se = lambda n: math.sqrt(0.35 * 4 / n)  # noqa: E731
    assert _power(0.2, se(res.n_left), 0.2) >= 0.3
    assert _power(0.2, se(res.n_left - 1), 0.2) < 0.3


@pytest.mark.parametrize(
    "kw",
    [
        dict(tau=0.1, n_left=0),
        dict(tau=0.1, var_left=-1.0),
        dict(tau=0.1, h_right=0.0),
        dict(tau=0.1, alpha=1.5),
        dict(tau=0.1, alpha=0.0),
        dict(tau=float("nan")),
    ],
)
def test_rdpower_rejects_invalid_design_inputs(kw):
    with pytest.raises(ValueError):
        sp.rdpower(**kw)


@pytest.mark.parametrize(
    "kw",
    [
        dict(tau=0.0),
        dict(tau=0.1, target_power=1.0),
        dict(tau=0.1, target_power=0.04),
        dict(tau=0.1, ratio=0.0),
        dict(tau=0.1, ratio=-1.0),
    ],
)
def test_rdsampsi_rejects_invalid_inputs(kw):
    with pytest.raises(ValueError):
        sp.rdsampsi(**kw)


def test_rdsampsi_data_mode_refuses_unsupported_options():
    with pytest.raises(ValueError, match="donut"):
        sp.rdsampsi(tau=3, data=SENATE, y="vote", x="margin", donut=1.0)


def test_power_is_alpha_at_zero_effect_and_symmetric():
    r0 = sp.rdpower(tau=0.0, alpha=0.07)
    assert r0.power == pytest.approx(0.07, rel=1e-12)
    a, b = sp.rdpower(tau=0.3), sp.rdpower(tau=-0.3)
    assert a.power == pytest.approx(b.power, rel=1e-14)
    assert np.isfinite(a.mde)
